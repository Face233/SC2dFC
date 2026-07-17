from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .cache import read_cached
from .config import resolve_path
from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset
from .metrics import dynamic_state_metrics, family_bootstrap_difference, projection_report, retrieval_metrics, sequence_metrics
from .models.baselines import group_mean, persistence
from .training import build_sequence_model, device_from_arg


def _load_model(config, window_length, checkpoint, stats_path, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_sequence_model(config, window_length, payload["decoder_type"], stats_path, device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def fit_state_model(dataset: DFCSequenceDataset, n_states: int, seed: int, max_windows: int = 50000) -> MiniBatchKMeans:
    rng = np.random.default_rng(seed)
    windows = []
    per_run = max(1, max_windows // max(len(dataset), 1))
    for subject, run in dataset.samples:
        fc, _ = read_cached(dataset.config, dataset.window_length, subject, run)
        indices = rng.choice(len(fc), size=min(per_run, len(fc)), replace=False)
        windows.append(fc[indices])
    values = np.concatenate(windows)
    model = MiniBatchKMeans(n_clusters=n_states, random_state=seed, batch_size=2048, n_init=10)
    model.fit(values)
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    predictions, targets, subjects, families, runs = [], [], [], [], []
    for batch in loader:
        result = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device), batch["run"].to(device))
        predictions.append(result.fc_z_edges.cpu().numpy())
        targets.append(batch["fc_future"].numpy())
        subjects.extend(batch["subject_id"])
        families.extend(batch["family_id"])
        runs.extend(batch["run"].numpy().tolist())
    return np.concatenate(predictions), np.concatenate(targets), subjects, families, runs


def evaluate_checkpoint(
    config: dict[str, Any],
    window_length: int,
    checkpoint: str | Path,
    stats_path: str | Path,
    baseline_checkpoint: str | Path | None = None,
    save_predictions: bool = False,
    device_name: str | None = None,
) -> Path:
    device = device_from_arg(device_name)
    model, payload = _load_model(config, window_length, checkpoint, Path(stats_path), device)
    test = DFCSequenceDataset(config, window_length, "test", stats_path, payload.get("ablation", "full"))
    train = DFCSequenceDataset(config, window_length, "train", stats_path)
    loader = DataLoader(test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    predictions, targets, subjects, families, runs = collect_predictions(model, loader, device)
    template = model.group_template.cpu().numpy()
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    rows = [sequence_metrics(p, t, template, nonoverlap) for p, t in zip(predictions, targets)]
    state_model = fit_state_model(train, int(config["evaluation"]["state_clusters"]), int(config["seed"]))
    for row, pred, true in zip(rows, predictions, targets):
        row.update(dynamic_state_metrics(state_model.predict(pred), state_model.predict(true), state_model.n_clusters))
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    aggregate.update(retrieval_metrics(predictions, targets, template, nonoverlap, subjects))
    warmup = np.stack([test[index]["fc_warmup"].numpy() for index in range(len(test))])
    analytic = {
        "group_mean": np.broadcast_to(template[None], targets.shape),
        "fc1_persistence": np.broadcast_to(warmup[:, None], targets.shape),
    }
    analytic_reports = {}
    for name, baseline_prediction in analytic.items():
        baseline_rows = [sequence_metrics(p, t, template, nonoverlap) for p, t in zip(baseline_prediction, targets)]
        analytic_reports[name] = {key: float(np.mean([row[key] for row in baseline_rows])) for key in baseline_rows[0]}
    projection_rows = [projection_report(p, int(config["data"]["n_nodes"]), float(config["evaluation"]["projection_epsilon"]))[1] for p in predictions]
    aggregate.update({f"projection_{key}": float(np.mean([row[key] for row in projection_rows])) for key in projection_rows[0]})
    report: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "window_length": window_length,
        "nonoverlap_horizon": nonoverlap,
        "n_samples": len(subjects),
        "aggregate": aggregate,
        "analytic_baselines": analytic_reports,
        "per_sample": [{"subject_id": s, "family_id": f, "run": r, **m} for s, f, r, m in zip(subjects, families, runs, rows)],
    }
    if baseline_checkpoint:
        baseline_model, baseline_payload = _load_model(config, window_length, baseline_checkpoint, Path(stats_path), device)
        baseline_test = DFCSequenceDataset(config, window_length, "test", stats_path, baseline_payload.get("ablation", "fc1_only"))
        baseline_loader = DataLoader(baseline_test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
        baseline_predictions, baseline_targets, baseline_subjects, baseline_families, _ = collect_predictions(baseline_model, baseline_loader, device)
        if subjects != baseline_subjects:
            raise ValueError("Main and baseline checkpoints do not cover the same ordered samples")
        main_scores = np.asarray([row["long_residual_pearson"] for row in rows])
        baseline_scores = np.asarray([sequence_metrics(p, t, template, nonoverlap)["long_residual_pearson"] for p, t in zip(baseline_predictions, baseline_targets)])
        report["success_gate"] = family_bootstrap_difference(main_scores, baseline_scores, families, int(config["evaluation"]["bootstrap_replicates"]), int(config["seed"]))
    output_dir = Path(checkpoint).resolve().parent
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_predictions:
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        for pred, true, subject, run in zip(predictions, targets, subjects, runs):
            projected, projection = projection_report(pred, int(config["data"]["n_nodes"]), float(config["evaluation"]["projection_epsilon"]))
            from .connectivity import edges_to_matrix
            raw_fc = edges_to_matrix(np.tanh(pred), int(config["data"]["n_nodes"]))
            np.savez_compressed(prediction_dir / f"{subject}_{'LR' if run == 0 else 'RL'}.npz", predicted_z=pred, target_z=true, raw_fc=raw_fc, projected_fc=projected, projection_metrics=json.dumps(projection))
    return report_path
