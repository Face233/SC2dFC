"""E0034/E0035: predict time-centered residuals from frozen first-FC features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset
from .evaluation import subject_bootstrap_loss_difference
from .management import file_sha256
from .metric_records import selection_record
from .residual_ridge import _load_baseline, stable_baseline
from .training import device_from_arg, load_autoencoder, seed_everything


def _dynamic(sample: dict, template: np.ndarray, context: np.ndarray,
             alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    warmup = sample["fc_warmup"].numpy().astype(np.float32)
    target = sample["fc_future"].numpy().astype(np.float32)
    baseline = stable_baseline(warmup, template, context, alpha).astype(np.float32)
    residual = target - baseline
    return target, baseline, residual - residual.mean(axis=0, keepdims=True)


def _fit_pca(dataset: DFCSequenceDataset, template: np.ndarray, context: np.ndarray,
             alpha: np.ndarray, rank: int, windows_per_subject: int, seed: int) -> PCA:
    """Use a deterministic, subject-balanced sample to bound PCA memory."""
    if rank > template.shape[1]:
        raise ValueError("PCA rank exceeds the number of FC edges")
    rng = np.random.default_rng(seed)
    rows = np.empty((len(dataset) * windows_per_subject, template.shape[1]), dtype=np.float32)
    for index in range(len(dataset)):
        _, _, dynamic = _dynamic(dataset[index], template, context, alpha)
        times = rng.choice(len(dynamic), size=windows_per_subject,
                           replace=windows_per_subject > len(dynamic))
        rows[index * windows_per_subject:(index + 1) * windows_per_subject] = dynamic[times]
    if rank > len(rows):
        raise ValueError("PCA rank exceeds the number of sampled training windows")
    return PCA(n_components=rank, svd_solver="randomized", random_state=seed).fit(rows)


def _training_targets(dataset: DFCSequenceDataset, encoder: torch.nn.Module,
                      device: torch.device, template: np.ndarray, context: np.ndarray,
                      alpha: np.ndarray, components: np.ndarray, start: int) -> dict[str, Any]:
    features, coefficients, energy, subjects = [], [], [], []
    encoder.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            _, _, dynamic = _dynamic(sample, template, context, alpha)
            features.append(encoder.encode(sample["fc_warmup"].to(device)[None]).cpu().numpy()[0])
            coefficients.append(dynamic @ components.T)
            energy.append(float(np.mean(dynamic[start:] ** 2)))
            subjects.append(sample["subject_id"])
    return {"features": np.asarray(features), "coefficients": np.asarray(coefficients),
            "long_dynamic_energy": np.asarray(energy), "subjects": subjects}


def _center_time(coefficients: np.ndarray) -> np.ndarray:
    return coefficients - coefficients.mean(axis=-2, keepdims=True)


def _dynamic_error(predicted: np.ndarray, actual: np.ndarray, energy: np.ndarray,
                   start: int, n_edges: int) -> np.ndarray:
    """Exact edge MSE by orthonormal projection, including omitted PCA energy."""
    predicted = _center_time(predicted)
    cross = np.sum(predicted[:, start:] ** 2 - 2 * predicted[:, start:] * actual[:, start:], axis=(1, 2))
    return energy + cross / ((predicted.shape[1] - start) * n_edges)


def _choose_model(data: dict[str, Any], ranks: list[int], penalties: list[float],
                  folds: int, start: int, n_edges: int) -> tuple[int, float, dict[str, float]]:
    groups = np.asarray(data["subjects"])
    if len(np.unique(groups)) < folds:
        raise ValueError("Fewer training subjects than dynamic Ridge folds")
    features = data["features"]
    targets = data["coefficients"]
    scores = {}
    for rank in ranks:
        for penalty in penalties:
            errors = []
            for train_idx, held_idx in GroupKFold(n_splits=folds).split(features, groups=groups):
                scaler = StandardScaler().fit(features[train_idx])
                model = Ridge(alpha=penalty).fit(scaler.transform(features[train_idx]),
                                                 targets[train_idx, :, :rank].reshape(len(train_idx), -1))
                prediction = model.predict(scaler.transform(features[held_idx])).reshape(len(held_idx), targets.shape[1], rank)
                errors.extend(_dynamic_error(prediction, targets[held_idx, :, :rank],
                                             data["long_dynamic_energy"][held_idx], start, n_edges))
            scores[f"{rank}/{penalty}"] = float(np.mean(errors))
    rank, penalty = min(((rank, penalty) for rank in ranks for penalty in penalties),
                        key=lambda pair: (scores[f"{pair[0]}/{pair[1]}"], pair[0], pair[1]))
    return rank, penalty, scores


def _stable_prediction(feature: np.ndarray, payload: dict[str, Any] | None) -> np.ndarray | None:
    if payload is None:
        return None
    scaled = (feature - payload["scaler_mean"]) / payload["scaler_scale"]
    return scaled @ payload["ridge_coef"].T + payload["ridge_intercept"]


def _time_correlation(predicted: np.ndarray, actual: np.ndarray) -> tuple[float | None, float]:
    """Average edgewise temporal Pearson; undefined flat edges are excluded."""
    p = predicted - predicted.mean(axis=0, keepdims=True)
    a = actual - actual.mean(axis=0, keepdims=True)
    norm_p = np.linalg.norm(p, axis=0)
    norm_a = np.linalg.norm(a, axis=0)
    valid = (norm_p > 1e-7) & (norm_a > 1e-7)
    if not np.any(valid):
        return None, 0.0
    corr = np.sum(p[:, valid] * a[:, valid], axis=0) / (norm_p[valid] * norm_a[valid])
    return float(np.mean(np.clip(corr, -1, 1))), float(np.mean(valid))


def _paired_interval(challenger: np.ndarray, baseline: np.ndarray, replicates: int, seed: int) -> dict[str, float | int | None]:
    valid = np.isfinite(challenger) & np.isfinite(baseline)
    delta = (challenger - baseline)[valid]
    if not len(delta):
        return {"mean_difference": None, "ci_low": None, "ci_high": None, "n_subjects": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(replicates, len(delta)))
    estimates = delta[draws].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"mean_difference": float(delta.mean()), "ci_low": float(low),
            "ci_high": float(high), "n_subjects": int(len(delta))}


def _evaluate(config: dict[str, Any], split: str, dataset: DFCSequenceDataset,
              encoder: torch.nn.Module, device: torch.device, template: np.ndarray,
              context: np.ndarray, alpha: np.ndarray, payload: dict[str, Any],
              stable_payload: dict[str, Any] | None) -> dict[str, Any]:
    start = nonoverlap_horizon(int(config["data"]["window_length"]), int(config["data"]["stride"]))
    components = payload["pca_components"]
    rank = len(components)
    steps, edges = template.shape
    records = []
    encoder.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            target, baseline, true_dynamic = _dynamic(sample, template, context, alpha)
            feature = encoder.encode(sample["fc_warmup"].to(device)[None]).cpu().numpy()[0]
            scaled = (feature - payload["scaler_mean"]) / payload["scaler_scale"]
            predicted_coeff = (scaled @ payload["ridge_coef"].T + payload["ridge_intercept"]).reshape(steps, rank)
            predicted_dynamic = _center_time(predicted_coeff) @ components
            stable = _stable_prediction(feature, stable_payload)
            if stable is None:
                stable = np.zeros(edges, dtype=np.float32)
            candidate = baseline + predicted_dynamic + stable[None]
            zero = baseline + stable[None]
            oracle_pca = (true_dynamic @ components.T) @ components
            raw_corr, raw_valid = _time_correlation(candidate[start:], target[start:])
            zero_corr, _ = _time_correlation(zero[start:], target[start:])
            residual_corr, residual_valid = _time_correlation(candidate[start:] - template[start:],
                                                              target[start:] - template[start:])
            zero_residual_corr, _ = _time_correlation(zero[start:] - template[start:],
                                                      target[start:] - template[start:])
            dynamic_corr, dynamic_valid = _time_correlation(predicted_dynamic[start:], true_dynamic[start:])
            shifted_dynamic_corr, _ = _time_correlation(
                predicted_dynamic[start:], np.roll(true_dynamic[start:], shift=max(start, 1), axis=0))
            diff_corr, diff_valid = _time_correlation(np.diff(predicted_dynamic[start:], axis=0),
                                                      np.diff(true_dynamic[start:], axis=0))
            records.append({
                "subject_id": sample["subject_id"], "run": sample["run_name"],
                "long_edge_mse": float(np.mean((candidate[start:] - target[start:]) ** 2)),
                "zero_dynamic_long_edge_mse": float(np.mean((zero[start:] - target[start:]) ** 2)),
                "e0032_long_edge_mse": float(np.mean((baseline[start:] - target[start:]) ** 2)),
                "dynamic_mse": float(np.mean((predicted_dynamic[start:] - true_dynamic[start:]) ** 2)),
                "zero_dynamic_mse": float(np.mean(true_dynamic[start:] ** 2)),
                "pca_oracle_dynamic_mse": float(np.mean((oracle_pca[start:] - true_dynamic[start:]) ** 2)),
                "time_pearson": raw_corr, "zero_dynamic_time_pearson": zero_corr,
                "time_pearson_valid_fraction": raw_valid,
                "template_adjusted_time_pearson": residual_corr,
                "zero_dynamic_template_adjusted_time_pearson": zero_residual_corr,
                "template_adjusted_valid_fraction": residual_valid,
                "dynamic_time_pearson": dynamic_corr, "dynamic_valid_fraction": dynamic_valid,
                "shifted_dynamic_time_pearson": shifted_dynamic_corr,
                "dynamic_difference_time_pearson": diff_corr, "difference_valid_fraction": diff_valid,
                "dynamic_std_ratio": float(np.std(predicted_dynamic[start:]) /
                                           max(np.std(true_dynamic[start:]), 1e-12)),
            })
    if not records:
        raise ValueError(f"No {split} samples for dynamic Ridge evaluation")
    fields = ("long_edge_mse", "zero_dynamic_long_edge_mse", "e0032_long_edge_mse", "dynamic_mse",
              "zero_dynamic_mse", "pca_oracle_dynamic_mse", "time_pearson", "zero_dynamic_time_pearson",
              "time_pearson_valid_fraction", "template_adjusted_time_pearson",
              "zero_dynamic_template_adjusted_time_pearson", "template_adjusted_valid_fraction",
              "dynamic_time_pearson", "dynamic_valid_fraction", "shifted_dynamic_time_pearson",
              "dynamic_difference_time_pearson",
              "difference_valid_fraction", "dynamic_std_ratio")
    aggregate = {}
    for key in fields:
        values = [record[key] for record in records if record[key] is not None and np.isfinite(record[key])]
        aggregate[key] = float(np.mean(values)) if values else None
    subjects = [record["subject_id"] for record in records]
    repeats = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap = subject_bootstrap_loss_difference(
        np.asarray([r["long_edge_mse"] for r in records]),
        np.asarray([r["zero_dynamic_long_edge_mse"] for r in records]), subjects, repeats, int(config["seed"]))
    corr_bootstrap = _paired_interval(np.asarray([r["time_pearson"] for r in records], dtype=float),
                                      np.asarray([r["zero_dynamic_time_pearson"] for r in records], dtype=float),
                                      repeats, int(config["seed"]))
    shifted_bootstrap = _paired_interval(np.asarray([r["dynamic_time_pearson"] for r in records], dtype=float),
                                         np.asarray([r["shifted_dynamic_time_pearson"] for r in records], dtype=float),
                                         repeats, int(config["seed"]))
    return {"schema_version": 1, "split": split, "n_samples": len(records),
            "nonoverlap_horizon": start, "include_stable": bool(config["model"]["include_stable"]),
            "aggregate": aggregate, "paired_long_mse_vs_zero_dynamic": bootstrap,
            "paired_time_pearson_vs_zero_dynamic": corr_bootstrap,
            "paired_dynamic_pearson_vs_shifted_target": shifted_bootstrap,
            "time_pearson_definition": "Mean per-subject, per-edge Pearson over future indices [nonoverlap, T); flat edges excluded",
            "pca_oracle_note": "Uses true future values for a representation ceiling, never for prediction",
            "per_sample": records}


def _load_stable(config: dict[str, Any], stats_path: Path, alpha_path: Path,
                 autoencoder_path: Path) -> dict[str, Any] | None:
    if not config["model"]["include_stable"]:
        return None
    reference = config["artifacts"]["stable_ridge"]
    path = Path(reference["path"])
    if not path.is_absolute():
        path = Path(config["paths"]["root"]) / path
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("Frozen E0033 stable Ridge checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "encoded_stable_residual_ridge":
        raise ValueError("Stable artifact is not E0033 Ridge")
    for key, artifact in (("stats_sha256", stats_path), ("alpha_sha256", alpha_path),
                          ("autoencoder_sha256", autoencoder_path)):
        if payload.get(key) != file_sha256(artifact):
            raise ValueError(f"Frozen E0033 {key} input differs from this experiment")
    return payload


def run_residual_dynamic_ridge(config: dict[str, Any], window_length: int, stats_path: Path,
                               autoencoder_path: Path, alpha_path: Path, output_dir: Path,
                               checkpoint_metadata: dict[str, Any], device_name: str | None) -> Path:
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    encoder = load_autoencoder(config, window_length, device, autoencoder_path)
    template, context, alpha = _load_baseline(stats_path, alpha_path)
    stable_payload = _load_stable(config, stats_path, alpha_path, autoencoder_path)
    train = DFCSequenceDataset(config, window_length, "train", stats_path, ablation="fc1_only")
    if not len(train):
        raise ValueError("No training samples for dynamic Ridge")
    ranks = [int(k) for k in config["model"]["pca_components_grid"]]
    pca = _fit_pca(train, template, context, alpha, max(ranks),
                   int(config["model"]["pca_fit_windows_per_subject"]), int(config["seed"]))
    features = _training_targets(train, encoder, device, template, context, alpha,
                                 pca.components_, nonoverlap_horizon(window_length, int(config["data"]["stride"])))
    rank, penalty, cv_scores = _choose_model(features, ranks,
        [float(value) for value in config["model"]["ridge_alpha_grid"]], int(config["model"]["cv_folds"]),
        nonoverlap_horizon(window_length, int(config["data"]["stride"])), template.shape[1])
    scaler = StandardScaler().fit(features["features"])
    ridge = Ridge(alpha=penalty).fit(scaler.transform(features["features"]),
                                     features["coefficients"][:, :, :rank].reshape(len(train), -1))
    payload = {"schema_version": 1, "model_name": "encoded_dynamic_residual_ridge",
               "include_stable": bool(config["model"]["include_stable"]), "pca_rank": rank,
               "pca_components": pca.components_[:rank].astype(np.float32), "ridge_alpha": penalty,
               "ridge_coef": ridge.coef_.astype(np.float32), "ridge_intercept": ridge.intercept_.astype(np.float32),
               "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
               "stats_sha256": file_sha256(stats_path), "alpha_sha256": file_sha256(alpha_path),
               "autoencoder_sha256": file_sha256(autoencoder_path),
               "stable_sha256": config["artifacts"]["stable_ridge"]["sha256"] if stable_payload is not None else None,
               **checkpoint_metadata}
    validation = DFCSequenceDataset(config, window_length, "val", stats_path, ablation="fc1_only")
    report = _evaluate(config, "val", validation, encoder, device, template, context, alpha, payload, stable_payload)
    report["model_selection"] = {"pca_rank": rank, "ridge_alpha": penalty, "cv_folds": int(config["model"]["cv_folds"]),
                                 "train_cv_long_dynamic_mse": cv_scores,
                                 "pca_fitted_on": "deterministically sampled training future windows"}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    payload["validation_metrics"] = report["aggregate"]
    torch.save(payload, checkpoint)
    (output_dir / "evaluation_val.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    selection = selection_record(report["aggregate"], "long_edge_mse", None,
        definition={"implementation": "encoded_dynamic_residual_ridge/v1", "split": "val",
                    "include_stable": payload["include_stable"],
                    "nonoverlap_start": report["nonoverlap_horizon"], "parameters_selected_on": "grouped_train_cv"},
        source="dynamic_ridge_validation")
    (output_dir / "metrics_best.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output_dir / "train.log").write_text(json.dumps(report["model_selection"]) + "\n", encoding="utf-8")
    return checkpoint


def evaluate_residual_dynamic_ridge(config: dict[str, Any], window_length: int, stats_path: Path,
                                    autoencoder_path: Path, alpha_path: Path, checkpoint: Path,
                                    output_dir: Path, split: str, device_name: str | None) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError("Dynamic Ridge split must be train, val, or test")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "encoded_dynamic_residual_ridge":
        raise ValueError("Checkpoint is not a dynamic residual Ridge model")
    for key, artifact in (("stats_sha256", stats_path), ("alpha_sha256", alpha_path),
                          ("autoencoder_sha256", autoencoder_path)):
        if payload.get(key) != file_sha256(artifact):
            raise ValueError(f"Dynamic Ridge {key} artifact changed since fitting")
    if payload["include_stable"] != bool(config["model"]["include_stable"]):
        raise ValueError("Dynamic checkpoint stable-head mode differs from config")
    stable = _load_stable(config, stats_path, alpha_path, autoencoder_path)
    if stable is not None and payload["stable_sha256"] != config["artifacts"]["stable_ridge"]["sha256"]:
        raise ValueError("Dynamic checkpoint stable artifact changed since fitting")
    device = device_from_arg(device_name)
    encoder = load_autoencoder(config, window_length, device, autoencoder_path)
    template, context, alpha = _load_baseline(stats_path, alpha_path)
    dataset = DFCSequenceDataset(config, window_length, split, stats_path, ablation="fc1_only")
    report = _evaluate(config, split, dataset, encoder, device, template, context, alpha, payload, stable)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"evaluation_{split}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path
