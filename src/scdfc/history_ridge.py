"""Training-fold PCA/Ridge reference derived from a frozen history run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data import DFCSequenceDataset
from .evaluation import _load_model
from .history_forecast import aggregate_subjects, compare_forecasts, forecast_metrics
from .management import file_sha256
from .progress import emit
from .training import device_from_arg


def predict_history_ridge(payload, encoded_history, last_fc):
    """Replay a derived baseline with the saved scaler, PCA and Ridge only."""
    features = (encoded_history-payload["scaler_mean"])/payload["scaler_scale"]
    coefficients = features @ payload["ridge_coef"].T + payload["ridge_intercept"]
    horizon = int(payload["forecast_windows"])
    rank = len(payload["pca_components"])
    coefficients = coefficients.reshape(len(features), horizon, rank)
    delta = coefficients @ payload["pca_components"] + payload["pca_mean"]
    return last_fc[:, None, :] + delta


@torch.no_grad()
def run_history_ridge(config, window_length, checkpoint, stats_path, autoencoder_path, output_dir, device_name="cpu"):
    """Fit with grouped TRAIN CV only; evaluate val once after alpha selection.

    The same frozen encoder features are used by the neural model and Ridge.
    No fitted PCA or scaler from another CV fold is reused.
    """
    spec = config["baselines"]["history_ridge"]
    components, folds = int(spec["pca_components"]), int(spec["cv_folds"])
    grid = [float(a) for a in spec["alpha_grid"]]
    if components < 1 or folds < 2 or not grid or any(a <= 0 or not np.isfinite(a) for a in grid):
        raise ValueError("Invalid history Ridge PCA/CV/alpha settings")
    device = device_from_arg(device_name)
    model, _ = _load_model(config, window_length, checkpoint, Path(stats_path), device, autoencoder_path)
    train = DFCSequenceDataset(config, window_length, "train", stats_path, "fc1_only")
    val = DFCSequenceDataset(config, window_length, "val", stats_path, "fc1_only")
    subjects = [subject for subject, _ in train.samples]
    if len(set(subjects)) < folds:
        raise ValueError("Not enough independent training subjects for history Ridge CV")
    if components > model.n_edges:
        raise ValueError("PCA components exceed FC edge count")
    horizon = int(config["data"]["forecast_windows"])
    nonoverlap = int(config["evaluation"]["nonoverlap_start"])
    sample_count = min(horizon, int(spec["pca_fit_windows_per_subject"]))
    if sample_count < 1:
        raise ValueError("PCA fit windows must be positive")
    fit_indices = np.unique(np.linspace(0, horizon-1, sample_count, dtype=int))
    destination = Path(output_dir) / "baselines/history_ridge_v1"
    if destination.exists():
        raise FileExistsError(f"Derived baseline already exists: {destination}")

    def features(data):
        return np.stack([model.encode_history(data[i]["fc_warmup"][None].to(device)).cpu().numpy().reshape(-1)
                         for i in range(len(data))])

    def delta(index):
        item = train[int(index)]
        return item["fc_future"].numpy()-item["last_fc"].numpy()[None]

    x = features(train)

    def fit_projection(indices):
        sampled = np.concatenate([delta(i)[fit_indices] for i in indices])
        if len(sampled) < components:
            raise ValueError("Insufficient fold-local samples for the declared PCA rank")
        return PCA(n_components=components, svd_solver="randomized", random_state=int(config["seed"])).fit(sampled)

    def projected_targets(pca, indices):
        return np.stack([pca.transform(delta(i)).reshape(-1) for i in indices])

    def score(pca, fitted, scaler, indices):
        latent = fitted.predict(scaler.transform(x[indices])).reshape(-1, horizon, components)
        values = {}
        for i, coefficients in zip(indices, latent):
            error = pca.inverse_transform(coefficients)-delta(i)
            value = float(np.mean(error[nonoverlap:].astype(np.float64)**2))
            values.setdefault(subjects[int(i)], []).append(value)
        return [float(np.mean(v)) for v in values.values()]

    cv_scores = {alpha: [] for alpha in grid}
    for fold, (fit, heldout) in enumerate(GroupKFold(folds).split(x, groups=subjects)):
        emit("history_ridge_cv_started", fold=fold+1, folds=folds)
        pca = fit_projection(fit)
        scaler = StandardScaler().fit(x[fit])
        y = projected_targets(pca, fit)
        scaled = scaler.transform(x[fit])
        for alpha in grid:
            fitted = Ridge(alpha=alpha).fit(scaled, y)
            cv_scores[alpha].extend(score(pca, fitted, scaler, heldout))
    selected = min(grid, key=lambda alpha: np.mean(cv_scores[alpha]))
    pca = fit_projection(np.arange(len(train)))
    scaler = StandardScaler().fit(x)
    fitted = Ridge(alpha=selected).fit(scaler.transform(x), projected_targets(pca, np.arange(len(train))))
    template = model.group_template.cpu().numpy()
    projected = fitted.predict(scaler.transform(features(val))).reshape(len(val), horizon, components)
    predictions, targets, neural, rows = [], [], [], []
    val_subjects = []
    for i, coefficients in enumerate(projected):
        item = val[i]
        prediction = item["last_fc"].numpy()[None]+pca.inverse_transform(coefficients)
        target = item["fc_future"].numpy()
        predictions.append(prediction)
        targets.append(target)
        neural.append(model(None, None, item["fc_warmup"][None].to(device)).fc_z_edges[0].cpu().numpy())
        val_subjects.append(item["subject_id"])
        rows.append({"subject_id": item["subject_id"], "run": item["run_name"],
                     **forecast_metrics(prediction, target, template, nonoverlap)})
    aggregate, per_subject = aggregate_subjects(rows)
    parent_hash = file_sha256(checkpoint)
    payload = {
        "algorithm": "encoded_history_pca_ridge/v1", "parent_checkpoint_sha256": parent_hash,
        "spec": spec, "selected_alpha": selected, "warmup_windows": config["data"]["warmup_windows"],
        "forecast_origins": config["data"]["forecast_origins"], "forecast_windows": horizon,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "pca_mean": pca.mean_, "pca_components": pca.components_,
        "ridge_coef": fitted.coef_, "ridge_intercept": fitted.intercept_,
        "fc_autoencoder_sha256": config["artifacts"]["fc_autoencoder"]["sha256"],
    }
    destination.mkdir(parents=True)
    torch.save(payload, destination / "ridge.pt")
    report = {
        "algorithm": payload["algorithm"], "split": "val", "spec": spec,
        "parent_checkpoint_sha256": parent_hash, "selected_alpha": selected,
        "selection_split": "grouped_train_cv", "pca_window_indices": fit_indices.tolist(),
        "cv_scores": {str(alpha): float(np.mean(values)) for alpha, values in cv_scores.items()},
        "aggregate": aggregate, "per_subject": per_subject, "per_sample": rows,
        "neural_minus_ridge": compare_forecasts(np.stack(neural), np.stack(predictions), np.stack(targets),
                                                template, val_subjects, nonoverlap,
                                                int(config["evaluation"].get("bootstrap_replicates", 10000))),
    }
    path = destination / "evaluation_val.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    emit("history_ridge_finished", selected_alpha=selected, report=str(path))
    return path
