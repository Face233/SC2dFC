"""Read existing validation reports and probe checkpoints without training or test data."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def summarize():
    rows = []
    for run in sorted((ROOT / 'outputs').glob('E*/runs/*')):
        path = run / 'evaluation_val.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        assert report['split'] == 'val'
        eid = run.parent.parent.name
        row = dict(experiment=eid, run_id=run.name, n=report['n_samples'],
                   evaluation=report['aggregate'])
        audit_path = run / 'dynamic_audit_val/summary.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            row['dynamic_nonoverlap'] = audit['methods'][eid].get('nonoverlap', {})
        log_path = run / 'train.log'
        if log_path.exists():
            logs = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith('{')]
            logs = [entry for entry in logs if 'primary_value' in entry]
            if logs:
                maximize = logs[0]['primary_metric'] == 'long_residual_pearson'
                best = (max if maximize else min)(logs, key=lambda entry: entry['primary_value'])
                row['training_best'] = {key: value for key, value in best.items()
                                        if key in {'epoch', 'objective_loss', 'primary_metric', 'long_residual_pearson'}
                                        or key.startswith('validation_')}
        rows.append(row)
    (DEST / 'validation_evidence.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return rows


def probe():
    import numpy as np
    import torch
    from scdfc.data import DFCSequenceDataset
    from scdfc.evaluation import _load_model, dynamic_calibration_metrics
    from scdfc.config import load_config
    from scdfc.training import seed_everything

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = ROOT / 'outputs/shared/dataset_lr_v1/window_83/training_stats.npz'
    results = []

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))

    def cross_dynamic_corr(x):
        x = x - x.mean(1, keepdims=True)
        x = x.reshape(len(x), -1).astype(np.float64)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return float(np.median((x @ x.T)[np.triu_indices(len(x), 1)]))

    for eid in ['E0013', 'E0018', 'E0024']:
        run = next((ROOT / 'outputs' / eid / 'runs').glob('*'))
        config = load_config(run / 'config_resolved.yaml')
        config['paths']['root'] = str(ROOT)
        artifact = ROOT / config['artifacts']['fc_autoencoder']['path']
        model, payload = _load_model(config, 83, run / 'checkpoints/best.pt', stats, device, artifact)
        dataset = DFCSequenceDataset(config, 83, 'val', stats)
        selected = np.sort(np.random.default_rng(42).choice(len(dataset), 16, replace=False))
        samples = [dataset[int(i)] for i in selected]
        batch = {key: torch.stack([sample[key] for sample in samples]).to(device)
                 for key in ['sc_matrix', 'sc_edges', 'fc_warmup', 'fc_future']}
        with torch.no_grad():
            sc, edges, fc = (batch[key] for key in ['sc_matrix', 'sc_edges', 'fc_warmup'])
            pred = model(sc, edges, fc).fc_z_edges.cpu().numpy()
            swap_sc = model(sc.roll(1, 0), edges.roll(1, 0), fc).fc_z_edges.cpu().numpy()
            swap_fc = model(sc, edges, fc.roll(1, 0)).fc_z_edges.cpu().numpy()
            true = batch['fc_future'].cpu().numpy()
            auto = model.fc_autoencoder
            target_flat = batch['fc_future'].reshape(-1, true.shape[-1])
            reconstructed = torch.cat([auto.decode(auto.encode(part)) for part in target_flat.split(256)])
            reconstructed = reconstructed.reshape(true.shape).cpu().numpy()
        pred, true = pred[:, 17:], true[:, 17:]
        true_spread = rms(true - true.mean(0, keepdims=True))
        row = dict(experiment=eid, split='val', n=16, sampling_seed=42,
                   checkpoint_epoch_1based=payload['epoch'] + 1,
                   checkpoint_training_metrics=payload.get('validation_metrics'),
                   intersubject_rms_ratio=rms(pred - pred.mean(0, keepdims=True)) / true_spread,
                   cross_subject_temporally_centered_prediction_corr=cross_dynamic_corr(pred),
                   cross_subject_temporally_centered_target_corr=cross_dynamic_corr(true),
                   sc_swap_rms_over_target_spread=rms(swap_sc[:, 17:] - pred) / true_spread,
                   fc_swap_rms_over_target_spread=rms(swap_fc[:, 17:] - pred) / true_spread)
        oracle = [dynamic_calibration_metrics(p, t, 3.6) for p, t in zip(reconstructed[:, 17:], true)]
        row['ae_oracle'] = {key: float(np.nanmedian([entry[key] for entry in oracle]))
                            for key in ['temporal_std_ratio', 'difference_std_ratio', 'difference_temporal_pearson']}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, batch, reconstructed, target_flat
    (DEST / 'checkpoint_probe.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    summarize()
    if '--probe' in sys.argv:
        probe()
