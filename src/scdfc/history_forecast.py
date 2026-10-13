"""Evaluation of the common-origin, full-remaining history experiments.

New metric names and constant-edge rules deliberately do not rewrite historical
metrics. Per-run values are aggregated by subject before bootstrap or reporting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DFCSequenceDataset
from .evaluation import _load_model, collect_predictions, projection_report
from .training import device_from_arg, seed_everything
from .management import file_sha256


def edge_correlations(prediction, target, axis=0, eps=1e-6):
    left, right = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    left = left - left.mean(axis, keepdims=True)
    right = right - right.mean(axis, keepdims=True)
    left_std, right_std = left.std(axis), right.std(axis)
    valid = (left_std >= eps) & (right_std >= eps)
    values = np.full(valid.shape, np.nan, dtype=float)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(right, axis=axis)
    numerator = np.sum(left * right, axis=axis)
    values[valid] = np.clip(numerator[valid] / denominator[valid], -1, 1)
    return values, valid, right_std >= eps, left_std < eps


def mean_defined(values):
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def interval_metrics(prediction, target, template):
    time, valid, true_valid, flat = edge_correlations(prediction, target)
    spatial = edge_correlations(prediction, target, axis=1)[0]
    adjusted, adjusted_valid, _, _ = edge_correlations(prediction-template, target-template)
    difference = edge_correlations(np.diff(prediction, axis=0), np.diff(target, axis=0))[0]
    pred_std, true_std = prediction.astype(np.float64).std(0), target.astype(np.float64).std(0)
    return {
        "mse": float(np.mean((prediction.astype(np.float64)-target)**2)),
        "space_r": mean_defined(spatial), "time_r": mean_defined(time),
        "adjusted_time_r": mean_defined(adjusted),
        "residual_space_r": mean_defined(edge_correlations(prediction-template, target-template, axis=1)[0]),
        "difference_time_r": mean_defined(difference),
        "std_ratio": mean_defined(pred_std[true_valid] / true_std[true_valid]),
        "time_valid_fraction": float(valid.mean()), "true_time_valid_fraction": float(true_valid.mean()),
        "predicted_flat_fraction": float(flat.mean()),
        "adjusted_time_valid_fraction": float(adjusted_valid.mean()),
    }


def forecast_metrics(prediction, target, template, nonoverlap):
    if prediction.shape != target.shape or prediction.shape != template.shape:
        raise ValueError("Prediction, target and aligned template shapes must match")
    if not 0 < nonoverlap < len(target)-1:
        raise ValueError("Forecast must contain at least two nonoverlap windows")
    result = {}
    for prefix, start in (("forecast", 0), ("forecast_nonoverlap", nonoverlap)):
        result.update({f"{prefix}_{key}": value for key, value in
                       interval_metrics(prediction[start:], target[start:], template[start:]).items()})
    return result


def aggregate_subjects(rows):
    subjects = sorted({row["subject_id"] for row in rows})
    keys = [key for key in rows[0] if key not in {"subject_id", "run", "forecast_origin"}]
    per_subject = [{"subject_id": subject, **{
        key: mean_defined(row[key] for row in rows if row["subject_id"] == subject) for key in keys
    }} for subject in subjects]
    return {key: mean_defined(row[key] for row in per_subject) for key in keys}, per_subject


def paired_interval(main, baseline, subjects, minimize=False, replicates=10000, seed=20261013):
    differences = {}
    for a, b, subject in zip(main, baseline, subjects):
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            differences.setdefault(subject, []).append(float(a-b))
    values = np.asarray([np.mean(items) for items in differences.values()])
    if not len(values):
        return {"mean_difference": None, "ci_low": None, "ci_high": None,
                "passes": False, "n_subjects": 0, "replicates": replicates}
    rng = np.random.default_rng(seed)
    # Bounded memory even for large bootstrap counts.
    estimates = np.concatenate([rng.choice(values, size=(min(256, replicates-i), len(values))).mean(1)
                                for i in range(0, replicates, 256)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    adjusted = np.quantile(estimates, [0.05/6, 1-0.05/6])
    return {"mean_difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "passes": bool(high < 0 if minimize else low > 0),
            "passes_simultaneous": bool(adjusted[1] < 0 if minimize else adjusted[0] > 0),
            "simultaneous_98_333_ci": adjusted.tolist(), "n_subjects": len(values),
            "replicates": replicates, "seed": seed}


def compare_forecasts(predictions, baseline, targets, template, subjects, nonoverlap, replicates=10000):
    main_rows = [forecast_metrics(p, t, template, nonoverlap) for p, t in zip(predictions, targets)]
    base_rows = [forecast_metrics(p, t, template, nonoverlap) for p, t in zip(baseline, targets)]
    result = {key: paired_interval([r[key] for r in main_rows], [r[key] for r in base_rows],
                                  subjects, key.endswith("mse"), replicates)
              for key in ("forecast_mse", "forecast_space_r", "forecast_time_r",
                          "forecast_nonoverlap_mse", "forecast_nonoverlap_space_r",
                          "forecast_nonoverlap_time_r", "forecast_nonoverlap_adjusted_time_r")}
    # Correlations must also be compared over identical effective edge sets.
    for start, prefix in ((0, "forecast"), (nonoverlap, "forecast_nonoverlap")):
        for adjusted in (False, True):
            main, other = [], []
            for p, b, t in zip(predictions, baseline, targets):
                offset = template[start:] if adjusted else 0
                a, va, _, _ = edge_correlations(p[start:]-offset, t[start:]-offset)
                c, vc, _, _ = edge_correlations(b[start:]-offset, t[start:]-offset)
                common = va & vc
                main.append(mean_defined(a[common]))
                other.append(mean_defined(c[common]))
            key = f"{prefix}_{'adjusted_' if adjusted else ''}time_r_common_edges"
            result[key] = paired_interval(main, other, subjects, replicates=replicates)
    return result


def retrieval_summary(predictions, targets, template, nonoverlap, subjects):
    # Distinct subjects form candidates; do not turn runs into extra identities.
    identities = sorted(set(subjects))
    outputs = {}
    for dynamic in (False, True):
        width = (predictions.shape[1]-nonoverlap)*predictions.shape[2] if dynamic else predictions.shape[2]
        features = [np.empty((len(identities), width), dtype=np.float32) for _ in range(2)]
        for values, destination in ((predictions, features[0]), (targets, features[1])):
            for index, subject in enumerate(identities):
                indices = [i for i, value in enumerate(subjects) if value == subject]
                sequence = values[indices, nonoverlap:].mean(0, dtype=np.float64)-template[nonoverlap:]
                vector = sequence-sequence.mean(0) if dynamic else sequence.mean(0)
                vector = vector.reshape(-1)
                vector = vector-vector.mean()
                destination[index] = vector
        left, right = features
        ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
        valid = (ln > 1e-6) & (rn > 1e-6)
        left /= np.maximum(ln[:, None], 1e-6)
        right /= np.maximum(rn[:, None], 1e-6)
        similarity = left @ right.T
        similarity[:, rn <= 1e-6] = -np.inf
        ranks = [int(np.flatnonzero(np.argsort(-similarity[i], kind="stable") == i)[0])+1
                 for i in range(len(identities)) if valid[i]]
        outputs["dynamic" if dynamic else "static"] = {
            "top1": mean_defined([rank == 1 for rank in ranks]),
            "top5": mean_defined([rank <= 5 for rank in ranks]),
            "valid_fraction": float(valid.mean()), "n_subjects": len(identities),
            "self_minus_others": mean_defined([
                similarity[i, i]-np.mean(np.delete(similarity[i], i)[np.isfinite(np.delete(similarity[i], i))])
                for i in range(len(identities)) if valid[i] and np.isfinite(np.delete(similarity[i], i)).any()
            ]),
        }
    return outputs


@torch.no_grad()
def evaluate_history_forecast(config, window_length, checkpoint, stats_path,
                              baseline_checkpoint=None, save_predictions=False, device_name=None,
                              split_name="val", output_dir=None, autoencoder_path=None,
                              baseline_config=None):
    if split_name not in {"train", "val", "test"}:
        raise ValueError("Unknown evaluation split")
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    model, payload = _load_model(config, window_length, checkpoint, Path(stats_path), device, autoencoder_path)
    dataset = DFCSequenceDataset(config, window_length, split_name, stats_path, "fc1_only")
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    predictions, targets, subjects, runs = collect_predictions(model, loader, device)
    template = model.group_template.cpu().numpy()
    start = int(config["evaluation"]["nonoverlap_start"])
    rows = [{"subject_id": subject, "run": run, "forecast_origin": dataset.forecast_origin,
             **forecast_metrics(p, t, template, start)}
            for p, t, subject, run in zip(predictions, targets, subjects, runs)]
    aggregate, per_subject = aggregate_subjects(rows)
    destination = Path(output_dir) if output_dir is not None else Path(checkpoint).parent.parent
    destination.mkdir(parents=True, exist_ok=True)
    identities = [(s, r, dataset.forecast_origin) for s, r in zip(subjects, runs)]
    report = {
        "schema_version": 2, "split": split_name, "n_samples": len(rows), "n_subjects": len(per_subject),
        "checkpoint_sha256": file_sha256(checkpoint),
        "forecast_origin": dataset.forecast_origin, "forecast_windows": len(template), "nonoverlap_start": start,
        "sample_manifest_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        "definition": {"representation": "Fisher-z edges", "constant_std_threshold": 1e-6,
                       "space_r_axis": "edges within each window", "time_r_axis": "future windows per edge",
                       "aggregation": "arithmetic mean within run, then subject, then subjects",
                       "constant_time_r": "null; always report valid fractions"},
        "checkpoint_selection": {"metrics": payload["validation_metrics"], "best_epoch": payload["epoch"]},
        "aggregate": aggregate, "per_sample": rows, "per_subject": per_subject,
        "retrieval": retrieval_summary(predictions, targets, template, start, subjects),
    }
    projection_rows = [projection_report(p, int(config["data"]["n_nodes"]),
                                        float(config["evaluation"].get("projection_epsilon", 1e-6)))[1] for p in predictions]
    report["matrix_quality"] = {key: mean_defined(row[key] for row in projection_rows) for key in projection_rows[0]}
    report["horizon_metrics"] = {
        "mse": np.mean([np.mean((p.astype(np.float64)-t)**2, axis=1)
                        for p, t in zip(predictions, targets)], axis=0).tolist(),
        "space_r": [mean_defined(edge_correlations(p[h:h+1], t[h:h+1], axis=1)[0][0]
                                 for p, t in zip(predictions, targets)) for h in range(len(template))],
    }
    report["segments"] = {}
    for name, first, stop in (("near", 0, start), ("short", 0, 40), ("early", start, 68),
                              ("middle", 68, 136), ("late", 136, 204)):
        stop = min(stop, len(template))
        if stop-first < 2:
            continue
        segment = [{"subject_id": s, "run": r, **interval_metrics(p[first:stop], t[first:stop], template[first:stop])}
                   for p, t, s, r in zip(predictions, targets, subjects, runs)]
        report["segments"][name] = {"start": first, "stop": stop, "aggregate": aggregate_subjects(segment)[0]}
    report["analytic_baselines"] = {}
    for name in ("last_fc", "history_mean", "group_mean"):
        baseline = np.stack([np.broadcast_to(
            template if name == "group_mean" else
            dataset[i]["fc_warmup"].numpy().mean(0) if name == "history_mean" else dataset[i]["last_fc"].numpy(),
            targets[i].shape) for i in range(len(dataset))])
        baseline_rows = [{"subject_id": s, "run": r, **forecast_metrics(p, t, template, start)}
                         for p, t, s, r in zip(baseline, targets, subjects, runs)]
        report["analytic_baselines"][name] = {
            "aggregate": aggregate_subjects(baseline_rows)[0],
            "paired_comparison": compare_forecasts(predictions, baseline, targets, template, subjects, start,
                                                   int(config["evaluation"].get("bootstrap_replicates", 10000))),
        }
    if baseline_checkpoint is not None:
        if baseline_config is None:
            raise ValueError("A paired history checkpoint requires its frozen baseline config")
        other, _ = _load_model(baseline_config, window_length, baseline_checkpoint, Path(stats_path), device, autoencoder_path)
        other_data = DFCSequenceDataset(baseline_config, window_length, split_name, stats_path, "fc1_only")
        other_pred, other_targets, other_subjects, other_runs = collect_predictions(
            other, DataLoader(other_data, batch_size=int(config["training"]["batch_size"])), device)
        if subjects != other_subjects or runs != other_runs or not np.array_equal(targets, other_targets):
            raise ValueError("Paired history models must predict identical subject/run targets")
        report["paired_model_comparison"] = compare_forecasts(predictions, other_pred, targets, template,
                                                            subjects, start, int(config["evaluation"].get("bootstrap_replicates", 10000)))
    if save_predictions:
        np.savez_compressed(destination / f"predictions_{split_name}.npz", prediction=predictions, target=targets,
                            subject_id=subjects, run=runs, forecast_origin=dataset.forecast_origin)
    ridge_report = destination / "baselines/history_ridge_v1/evaluation_val.json"
    if split_name == "val" and ridge_report.exists():
        reference = json.loads(ridge_report.read_text(encoding="utf-8"))
        if (reference["parent_checkpoint_sha256"] != file_sha256(checkpoint)
                or reference["spec"] != config["baselines"]["history_ridge"]):
            raise ValueError("Derived Ridge reference differs from the frozen run")
        report["derived_baselines"] = {"history_ridge": {
            "report": str(ridge_report), "aggregate": reference["aggregate"],
            "neural_minus_ridge": reference["neural_minus_ridge"],
        }}
    path = destination / f"evaluation_{split_name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path
