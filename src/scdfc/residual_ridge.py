"""E0033: frozen FC encoder followed by a stable-residual Ridge readout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset, group_template_for_warmup
from .evaluation import subject_bootstrap_loss_difference
from .management import file_sha256
from .metric_records import selection_record
from .training import device_from_arg, load_autoencoder, seed_everything


def stable_baseline(warmup: np.ndarray, template: np.ndarray, context: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Reproduce E0032's train-fitted prediction for one observed first FC."""
    if warmup.ndim != 1 or template.ndim != 2 or context.shape != warmup.shape:
        raise ValueError("E0033 requires one warmup FC vector and matching training templates")
    if len(alpha) != len(template):
        raise ValueError("E0032 alpha length differs from the future template")
    return template + alpha[:, None] * (warmup - context)[None]


def _load_baseline(stats_path: Path, alpha_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(stats_path) as stats:
        if "context_template" not in stats.files:
            raise ValueError("E0033 requires a training-only context_template")
        template = np.asarray(group_template_for_warmup(dict(stats), 1), dtype=np.float64)
        context = np.asarray(stats["context_template"], dtype=np.float64)
    with np.load(alpha_path) as fit:
        alpha = np.asarray(fit["alpha"], dtype=np.float64)
    if len(alpha) != len(template) or not np.all(np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("Invalid E0032 alpha artifact")
    return template, context, alpha


def _collect(
    config: dict[str, Any], window_length: int, split: str, stats_path: Path,
    autoencoder: torch.nn.Module, device: torch.device, template: np.ndarray,
    context: np.ndarray, alpha: np.ndarray, include_predictions: bool = False,
) -> dict[str, Any]:
    dataset = DFCSequenceDataset(config, window_length, split, stats_path, ablation="fc1_only")
    if not len(dataset):
        raise ValueError(f"No {split} samples for E0033")
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    values: dict[str, list[Any]] = {key: [] for key in ("features", "stable", "long_stable", "subjects", "runs")}
    if include_predictions:
        values.update({key: [] for key in ("baseline_long_mean", "baseline_long_mse", "segment_residual_mean", "segment_baseline_mse")})
    residual_sum = np.zeros(len(template), dtype=np.float64)
    residual_square_sum = 0.0
    dynamic_square_sum = 0.0
    dynamic_lag_dot = 0.0
    dynamic_lag_left = 0.0
    dynamic_lag_right = 0.0
    total_elements = 0
    autoencoder.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            warmup = sample["fc_warmup"].numpy().astype(np.float64)
            target = sample["fc_future"].numpy().astype(np.float64)
            baseline = stable_baseline(warmup, template, context, alpha)
            residual = target - baseline
            stable = residual.mean(axis=0)
            dynamic = residual - stable[None]
            encoded = autoencoder.encode(sample["fc_warmup"].to(device)[None]).cpu().numpy()[0]
            values["features"].append(encoded)
            values["stable"].append(stable)
            values["long_stable"].append(residual[nonoverlap:].mean(axis=0))
            values["subjects"].append(sample["subject_id"])
            values["runs"].append(sample["run_name"])
            if include_predictions:
                boundaries = ((0, nonoverlap), (nonoverlap, 85), (85, 154), (154, len(template)))
                values["baseline_long_mean"].append(baseline[nonoverlap:].mean(axis=0))
                values["baseline_long_mse"].append(float(np.mean(residual[nonoverlap:] ** 2)))
                values["segment_residual_mean"].append(np.stack([residual[start:stop].mean(axis=0)
                                                                   for start, stop in boundaries]))
                values["segment_baseline_mse"].append([float(np.mean(residual[start:stop] ** 2))
                                                        for start, stop in boundaries])
            residual_sum += np.sum(residual * residual, axis=1)
            residual_square_sum += float(np.sum(residual * residual))
            dynamic_square_sum += float(np.sum(dynamic * dynamic))
            dynamic_lag_dot += float(np.sum(dynamic[:-1] * dynamic[1:]))
            dynamic_lag_left += float(np.sum(dynamic[:-1] ** 2))
            dynamic_lag_right += float(np.sum(dynamic[1:] ** 2))
            total_elements += residual.size
    stable_array = np.stack(values["stable"])
    result: dict[str, Any] = {
        "features": np.stack(values["features"]),
        "stable": stable_array,
        "long_stable": np.stack(values["long_stable"]),
        "subjects": values["subjects"], "runs": values["runs"],
        "audit": {
            "n_samples": len(dataset),
            "residual_mse_zero": residual_square_sum / total_elements,
            "stable_mse_zero": float(np.mean(stable_array ** 2)),
            "dynamic_mse_zero": dynamic_square_sum / total_elements,
            "stable_between_subject_variance": float(np.mean(np.var(stable_array, axis=0))),
            "dynamic_lag1_correlation": dynamic_lag_dot / max((dynamic_lag_left * dynamic_lag_right) ** 0.5, 1e-12),
            "residual_mse_by_horizon": (residual_sum / (len(dataset) * template.shape[1])).tolist(),
        },
    }
    if include_predictions:
        for key in ("baseline_long_mean", "baseline_long_mse", "segment_residual_mean", "segment_baseline_mse"):
            result[key] = np.asarray(values[key])
    return result


def _retrieval_from_means(predicted: np.ndarray, true: np.ndarray, template: np.ndarray,
                          subjects: list[str]) -> dict[str, float]:
    predicted = predicted - template[None]
    true = true - template[None]
    predicted = predicted - predicted.mean(axis=1, keepdims=True)
    true = true - true.mean(axis=1, keepdims=True)
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-12)
    true /= np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-12)
    similarity = predicted @ true.T
    identities = np.asarray(subjects)
    ranks = np.empty(len(subjects), dtype=int)
    for index in range(len(subjects)):
        order = np.argsort(-similarity[index])
        ranks[index] = int(np.flatnonzero(identities[order] == identities[index])[0]) + 1
    return {"retrieval_top1": float(np.mean(ranks == 1)), "retrieval_top5": float(np.mean(ranks <= 5)),
            "retrieval_mean_rank": float(ranks.mean())}


def _choose_alpha(features: np.ndarray, stable: np.ndarray, long_stable: np.ndarray,
                  subjects: list[str], candidates: list[float], folds: int) -> tuple[float, dict[str, float]]:
    groups = np.asarray(subjects)
    if len(np.unique(groups)) < folds:
        raise ValueError("Fewer training subjects than Ridge selection folds")
    scores: dict[str, float] = {}
    for penalty in candidates:
        errors = []
        for train_index, held_index in GroupKFold(n_splits=folds).split(features, stable, groups):
            scaler = StandardScaler().fit(features[train_index])
            model = Ridge(alpha=penalty).fit(scaler.transform(features[train_index]), stable[train_index])
            prediction = model.predict(scaler.transform(features[held_index]))
            # A constant stable correction minimizes the full long-horizon MSE
            # exactly when it approaches the long-horizon residual mean.
            errors.extend(np.mean((prediction - long_stable[held_index]) ** 2, axis=1).tolist())
        scores[str(penalty)] = float(np.mean(errors))
    chosen = min(candidates, key=lambda value: (scores[str(value)], value))
    return chosen, scores


def _report(data: dict[str, Any], predicted_stable: np.ndarray, template: np.ndarray,
            nonoverlap: int, config: dict[str, Any], split: str) -> dict[str, Any]:
    boundaries = [("overlap_context", 0, nonoverlap), ("early_long", nonoverlap, 85),
                  ("middle_long", 85, 154), ("late_long", 154, template.shape[0])]
    correction_square = np.mean(predicted_stable ** 2, axis=1)
    horizons = {}
    for index, (name, start, stop) in enumerate(boundaries):
        baseline_scores = data["segment_baseline_mse"][:, index]
        ridge_scores = (baseline_scores - 2 * np.mean(predicted_stable * data["segment_residual_mean"][:, index], axis=1)
                        + correction_square)
        horizons[name] = {
            "start_index": start, "stop_index_exclusive": stop, "n_windows": stop - start,
            "ridge_mse": float(np.mean(ridge_scores)),
            "e0032_mse": float(np.mean(baseline_scores)),
        }
    baseline_long = data["baseline_long_mse"]
    ridge_long = baseline_long - 2 * np.mean(predicted_stable * data["long_stable"], axis=1) + correction_square
    stable_target = data["stable"]
    centered_prediction = predicted_stable - predicted_stable.mean(axis=1, keepdims=True)
    centered_target = stable_target - stable_target.mean(axis=1, keepdims=True)
    correlation = np.sum(centered_prediction * centered_target, axis=1) / np.maximum(
        np.linalg.norm(centered_prediction, axis=1) * np.linalg.norm(centered_target, axis=1), 1e-12
    )
    baseline_long_mean = data["baseline_long_mean"]
    target_long_mean = baseline_long_mean + data["long_stable"]
    template_long_mean = template[nonoverlap:].mean(axis=0)
    retrieval_ridge = _retrieval_from_means(baseline_long_mean + predicted_stable, target_long_mean,
                                            template_long_mean, data["subjects"])
    retrieval_baseline = _retrieval_from_means(baseline_long_mean, target_long_mean,
                                               template_long_mean, data["subjects"])
    bootstrap = subject_bootstrap_loss_difference(
        ridge_long, baseline_long, data["subjects"], int(config["evaluation"]["bootstrap_replicates"]), int(config["seed"])
    )
    return {
        "schema_version": 1, "split": split, "n_samples": len(stable_target),
        "nonoverlap_horizon": nonoverlap,
        "aggregate": {
            "long_edge_mse": float(ridge_long.mean()),
            "e0032_long_edge_mse": float(baseline_long.mean()),
            "stable_residual_mse": float(np.mean((predicted_stable - stable_target) ** 2)),
            "zero_stable_residual_mse": float(np.mean(stable_target ** 2)),
            "stable_residual_edge_pearson": float(np.mean(correlation)),
            "stable_between_subject_std_ratio": float(
                np.mean(np.std(predicted_stable, axis=0)) / max(np.mean(np.std(stable_target, axis=0)), 1e-12)
            ),
            **retrieval_ridge,
        },
        "e0032_retrieval": retrieval_baseline,
        "horizon_metrics": horizons,
        "paired_long_mse": bootstrap,
        "dynamic_note": "The time-constant correction leaves each edge's centered temporal trajectory identical to E0032.",
        "per_sample": [
            {"subject_id": subject, "run": run, "long_edge_mse": float(main),
             "e0032_long_edge_mse": float(base)}
            for subject, run, main, base in zip(data["subjects"], data["runs"], ridge_long, baseline_long)
        ],
    }


def run_residual_ridge(config: dict[str, Any], window_length: int, stats_path: Path,
                       autoencoder_path: Path, alpha_path: Path, output_dir: Path,
                       checkpoint_metadata: dict[str, Any], device_name: str | None) -> Path:
    seed_everything(int(config["seed"]))
    if int(config["data"].get("warmup_windows", 1)) != 1:
        raise ValueError("E0033 requires exactly one warmup FC window")
    device = device_from_arg(device_name)
    encoder = load_autoencoder(config, window_length, device, autoencoder_path)
    encoder.eval()
    template, context, alpha = _load_baseline(stats_path, alpha_path)
    train = _collect(config, window_length, "train", stats_path, encoder, device, template, context, alpha)
    penalties = [float(value) for value in config["model"]["ridge_alpha_grid"]]
    chosen, cv_scores = _choose_alpha(train["features"], train["stable"], train["long_stable"],
                                      train["subjects"], penalties, int(config["model"]["cv_folds"]))
    scaler = StandardScaler().fit(train["features"])
    ridge = Ridge(alpha=chosen).fit(scaler.transform(train["features"]), train["stable"])
    validation = _collect(config, window_length, "val", stats_path, encoder, device, template, context, alpha,
                          include_predictions=True)
    predicted_stable = ridge.predict(scaler.transform(validation["features"]))
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    report = _report(validation, predicted_stable, template, nonoverlap, config, "val")
    report["train_residual_audit"] = train["audit"]
    report["val_residual_audit"] = validation["audit"]
    report["ridge_selection"] = {"alpha": chosen, "cv_folds": int(config["model"]["cv_folds"]),
                                 "long_stable_cv_mse": cv_scores}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = checkpoint_dir / "best.pt"
    torch.save({
        "schema_version": 1, "model_name": "encoded_stable_residual_ridge", "ridge_alpha": chosen,
        "ridge_coef": ridge.coef_, "ridge_intercept": ridge.intercept_,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "stats_sha256": file_sha256(stats_path), "alpha_sha256": file_sha256(alpha_path),
        "autoencoder_sha256": file_sha256(autoencoder_path),
        "validation_metrics": report["aggregate"], **checkpoint_metadata,
    }, checkpoint)
    report_path = output_dir / "evaluation_val.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    selection = selection_record(
        report["aggregate"], "long_edge_mse", None,
        definition={"implementation": "encoded_stable_residual_ridge/v1", "split": "val",
                    "nonoverlap_start": nonoverlap, "ridge_alpha_selected_on": "grouped_train_cv"},
        source="ridge_validation",
    )
    (output_dir / "metrics_best.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output_dir / "train.log").write_text(json.dumps({"fit": "encoded_stable_residual_ridge",
                                                      "ridge_alpha": chosen, "cv_scores": cv_scores}) + "\n", encoding="utf-8")
    return checkpoint


def evaluate_residual_ridge(config: dict[str, Any], window_length: int, stats_path: Path,
                            autoencoder_path: Path, alpha_path: Path, checkpoint: Path,
                            output_dir: Path, split: str, device_name: str | None) -> Path:
    """Replay an E0033 checkpoint on train or validation data."""
    if split not in {"train", "val", "test"}:
        raise ValueError("E0033 evaluation split must be train, val, or test")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "encoded_stable_residual_ridge":
        raise ValueError("Checkpoint is not an E0033 residual Ridge model")
    for label, path in (("stats", stats_path), ("alpha", alpha_path), ("autoencoder", autoencoder_path)):
        if file_sha256(path) != payload[f"{label}_sha256"]:
            raise ValueError(f"E0033 {label} artifact changed since fitting")
    device = device_from_arg(device_name)
    encoder = load_autoencoder(config, window_length, device, autoencoder_path)
    encoder.eval()
    template, context, alpha = _load_baseline(stats_path, alpha_path)
    data = _collect(config, window_length, split, stats_path, encoder, device, template, context, alpha,
                    include_predictions=True)
    scaled = (data["features"] - payload["scaler_mean"]) / payload["scaler_scale"]
    predicted_stable = scaled @ payload["ridge_coef"].T + payload["ridge_intercept"]
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    report = _report(data, predicted_stable, template, nonoverlap, config, split)
    report[f"{split}_residual_audit"] = data["audit"]
    report["ridge_selection"] = {"alpha": float(payload["ridge_alpha"])}
    report_path = output_dir / f"evaluation_{split}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path
