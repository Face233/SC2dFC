import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scdfc.evaluation import prediction_objective
from scdfc.metric_records import selection_record, validate_selection_record
from scdfc.training import sequence_criterion, validate_sequence
from scdfc.visualization import ManagedRun, _overview_lines


@pytest.mark.parametrize('weights,loss_type', [
    ({'edge': 0, 'difference': 1, 'variance': 1}, 'mse'),
    ({'edge': 2, 'difference': 0.3, 'variance': 0.7}, 'huber'),
    ({'edge': 1, 'contrastive': 0.5, 'residual_corr': 0.2}, 'mse'),
    ({'static': 0.5, 'long_horizon_variance': 2, 'fcd': 1}, 'mse'),
    ({'edge': 0, 'psd': 1}, 'mse'),
])
def test_evaluation_objective_matches_training_including_partial_batch(weights, loss_type):
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(5, 8, 6)).astype('float32')
    true = rng.normal(size=pred.shape).astype('float32')
    template = np.zeros((8, 6), dtype='float32')
    config = {'data': {'n_nodes': 4}, 'training': {
        'loss_weights': weights, 'loss_type': loss_type, 'batch_size': 2, 'huber_beta': 0.4}}

    class FixedModel:
        group_template = torch.from_numpy(template)

        def eval(self):
            return self

        def __call__(self, sc, edges, fc):
            return SimpleNamespace(fc_z_edges=fc)

    batches = [{'sc_matrix': torch.zeros(len(p), 4, 4), 'sc_edges': torch.zeros(len(p), 6),
                'fc_warmup': torch.from_numpy(p), 'fc_future': torch.from_numpy(t)}
               for p, t in zip(np.array_split(pred, [2, 4]), np.array_split(true, [2, 4]))]
    expected = validate_sequence(FixedModel(), batches, sequence_criterion(config, 2), 2, torch.device('cpu'))
    actual = prediction_objective(pred, true, template, config, 2)
    assert actual['value'] == pytest.approx(expected['objective_loss'])
    for name, value in actual['components'].items():
        assert value == pytest.approx(expected[f'validation_{name}_loss'])
    if weights.get('edge') == 0 and 'variance' in weights:
        shifted = prediction_objective(pred + 10, true, template, config, 2)
        assert shifted['value'] == pytest.approx(actual['value'], rel=1e-5)


@pytest.mark.parametrize('mutation', [
    {'split': 'test'}, {'kind': 'evaluation'}, {'schema_version': 1},
    {'primary_metric': 'mse'}, {'metrics': {'objective_loss': float('nan')}},
])
def test_unverified_or_mislabeled_selection_is_rejected(mutation):
    record = selection_record({'objective_loss': 0.1}, 'objective_loss', 2, definition={'weights': {'edge': 1}})
    record.update(mutation)
    with pytest.raises(ValueError):
        validate_selection_record(record, 'objective_loss')


def test_overview_uses_selected_validation_value_not_evaluation_value(tmp_path):
    run = ManagedRun('E0024', 'run', tmp_path, {'training': {}, 'model': {}, 'experiment': {}}, {})
    best = selection_record({'objective_loss': 0.014812}, 'objective_loss', 98, definition={'metric': 'test'})
    text = str(_overview_lines(run, best, {'aggregate': {'objective_loss': 2.307979}}))
    assert '0.01481' in text
    assert '2.308' not in text


@pytest.mark.parametrize('task', ['analytic', 'sequence'])
@pytest.mark.parametrize('split', ['val', 'test'])
def test_managed_evaluation_never_overwrites_selection(tmp_path, monkeypatch, task, split):
    import scdfc.managed_cli as cli
    run = tmp_path / 'outputs' / 'E0001' / 'runs' / 'run'
    run.mkdir(parents=True)
    config = {'data': {'window_length': 83}, 'experiment': {'task': task},
              'model': {'name': 'group_mean'}, 'evaluation': {'primary_metric': 'objective_loss'}}
    meta = {'level': 2, 'experiment_id': 'E0001', 'run_id': 'run', 'config_sha256': 'sha'}
    (run / 'metadata.json').write_text(json.dumps(meta))
    selected = run / 'metrics_best.json'
    selected.write_text('original selected validation metrics')
    monkeypatch.setattr(cli, '_root', lambda: tmp_path)
    monkeypatch.setattr(cli, 'find_run', lambda *args: run)
    monkeypatch.setattr(cli, 'load_config', lambda *args: config)
    monkeypatch.setattr(cli, 'config_sha256', lambda *args: 'sha')
    monkeypatch.setattr(cli, 'verify_data_bindings', lambda *args: None)
    monkeypatch.setattr(cli, 'verify_artifact', lambda *args: None)
    monkeypatch.setattr(cli, '_stats_path', lambda *args: tmp_path / 'stats')
    monkeypatch.setattr(cli.torch, 'load', lambda *args, **kwargs: meta)

    def evaluate(*args, **kwargs):
        path = run / f'evaluation_{split}.json'
        path.write_text(json.dumps({'aggregate': {'objective_loss': 99}, 'split': split}))
        return path

    monkeypatch.setattr(cli, 'evaluate_checkpoint', evaluate)
    monkeypatch.setattr(cli, 'evaluate_analytic_baseline', evaluate)
    cli.command_evaluate_run(SimpleNamespace(run_id='run', split=split, final_test=split == 'test', device='cpu'))
    assert selected.read_text() == 'original selected validation metrics'
