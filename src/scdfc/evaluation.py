from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, wasserstein_distance
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .config import DEFAULT_SEED, resolve_path
from .connectivity import edges_to_matrix, inverse_fisher_z, nearest_correlation, nonoverlap_horizon
from .data import DFCSequenceDataset, read_cached
from .training import build_sequence_model, device_from_arg, seed_everything


# ======================== 测试指标与统计检验 ========================
def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """沿 FC 边维度计算每个时间窗的 Pearson 相关。"""
    left = left - left.mean(-1, keepdims=True)
    right = right - right.mean(-1, keepdims=True)
    return np.sum(left * right, axis=-1) / np.maximum(np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-12)


def fcd(sequence: np.ndarray) -> np.ndarray:
    """计算窗口×窗口的功能连接动力学（FCD）相似度矩阵。"""
    centered = sequence - sequence.mean(-1, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def _smooth_l1(values: np.ndarray, beta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute < beta, 0.5 * values**2 / beta, absolute - 0.5 * beta)


def sequence_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    template: np.ndarray,
    nonoverlap: int,
    huber_beta: float = 1.0,
    difference_weight: float = 0.25,
) -> dict[str, float]:
    """汇总单个 subject/run 的边级、动态和 FCD 指标。"""
    if huber_beta <= 0:
        raise ValueError("huber_beta must be positive")
    pred_r, target_r = np.tanh(prediction), np.tanh(target)
    long_pred, long_target = prediction[nonoverlap:] - template[nonoverlap:], target[nonoverlap:] - template[nonoverlap:]
    raw_corr = _row_correlation(prediction, target)
    raw_spearman = _row_correlation(rankdata(prediction, axis=-1), rankdata(target, axis=-1))
    residual_corr = _row_correlation(long_pred, long_target)
    pred_fcd, true_fcd = fcd(pred_r), fcd(target_r)
    tri = np.triu_indices(len(pred_fcd), 1)
    pred_diff, target_diff = np.diff(prediction, axis=0), np.diff(target, axis=0)
    edge_huber = float(np.mean(_smooth_l1(prediction - target, huber_beta)))
    difference_huber = float(np.mean(_smooth_l1(pred_diff - target_diff, huber_beta)))
    n_edges = prediction.shape[-1]
    n_nodes = int((1 + np.sqrt(1 + 8 * n_edges)) / 2)
    if n_nodes * (n_nodes - 1) // 2 != n_edges:
        raise ValueError(f"{n_edges} is not a valid undirected edge count")
    pred_strength = np.abs(edges_to_matrix(pred_r, n_nodes)).sum(-1) / (n_nodes - 1)
    target_strength = np.abs(edges_to_matrix(target_r, n_nodes)).sum(-1) / (n_nodes - 1)
    return {
        "objective_loss": edge_huber + float(difference_weight) * difference_huber,
        "edge_huber": edge_huber,
        "difference_huber": difference_huber,
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "raw_edge_pearson": float(np.nanmean(raw_corr)),
        "raw_edge_spearman": float(np.nanmean(raw_spearman)),
        "long_residual_pearson": float(np.nanmean(residual_corr)),
        "difference_mse": float(np.mean((pred_diff - target_diff) ** 2)),
        "variance_mae": float(np.mean(np.abs(prediction.var(0) - target.var(0)))),
        "dynamic_amplitude_mae": float(abs(pred_diff.std() - target_diff.std())),
        "node_strength_pearson": float(np.nanmean(_row_correlation(pred_strength, target_strength))),
        "node_strength_mae": float(np.mean(np.abs(pred_strength - target_strength))),
        "fcd_pearson": float(pearsonr(pred_fcd[tri], true_fcd[tri]).statistic),
        "fcd_wasserstein": float(wasserstein_distance(pred_fcd[tri], true_fcd[tri])),
    }


_DYNAMIC_AUDIT_BANDS_HZ = {
    "low": (0.003, 0.017),
    "mid": (0.017, 0.033),
}


def _median_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> tuple[float, int]:
    """Return a robust ratio across valid edges and how many edges contributed."""
    valid = denominator > eps
    if not np.any(valid):
        return float("nan"), 0
    return float(np.median(numerator[valid] / denominator[valid])), int(valid.sum())


def _median_temporal_correlation(prediction: np.ndarray, target: np.ndarray, eps: float) -> tuple[float, int]:
    prediction = prediction - prediction.mean(0, keepdims=True)
    target = target - target.mean(0, keepdims=True)
    denominator = np.linalg.norm(prediction, axis=0) * np.linalg.norm(target, axis=0)
    valid = denominator > eps
    if not np.any(valid):
        return float("nan"), 0
    correlations = (prediction[:, valid] * target[:, valid]).sum(0) / denominator[valid]
    return float(np.median(correlations)), int(valid.sum())


def _band_power_ratios(prediction: np.ndarray, target: np.ndarray, sample_interval_seconds: float, eps: float) -> dict[str, tuple[float, int]]:
    """Compare Hann-windowed temporal power while deliberately discarding phase."""
    steps = prediction.shape[0]
    window = np.hanning(steps)[:, None]
    prediction_power = np.abs(np.fft.rfft((prediction - prediction.mean(0, keepdims=True)) * window, axis=0)) ** 2
    target_power = np.abs(np.fft.rfft((target - target.mean(0, keepdims=True)) * window, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(steps, d=sample_interval_seconds)
    result: dict[str, tuple[float, int]] = {}
    for name, (lower, upper) in _DYNAMIC_AUDIT_BANDS_HZ.items():
        bins = (frequencies >= lower) & (frequencies < upper)
        if not np.any(bins):
            result[name] = (float("nan"), 0)
            continue
        predicted_band = prediction_power[bins].mean(0)
        target_band = target_power[bins].mean(0)
        valid = target_band > eps
        if not np.any(valid):
            result[name] = (float("nan"), 0)
            continue
        # Geometric median avoids a few low-power edges dominating a raw ratio.
        ratio = np.exp(np.median(np.log((predicted_band[valid] + eps) / (target_band[valid] + eps))))
        result[name] = (float(ratio), int(valid.sum()))
    return result


def dynamic_calibration_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_interval_seconds: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Measure dFC amplitude and spectral calibration for one [time, edge] sequence."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching [time, edge] shapes")
    if prediction.shape[0] < 4:
        raise ValueError("Dynamic calibration requires at least four time windows")
    temporal_ratio, temporal_count = _median_ratio(prediction.std(0), target.std(0), eps)
    prediction_diff, target_diff = np.diff(prediction, axis=0), np.diff(target, axis=0)
    difference_ratio, difference_count = _median_ratio(prediction_diff.std(0), target_diff.std(0), eps)
    difference_corr, correlation_count = _median_temporal_correlation(prediction_diff, target_diff, eps)
    powers = _band_power_ratios(prediction, target, sample_interval_seconds, eps)
    return {
        "temporal_std_ratio": temporal_ratio,
        "difference_std_ratio": difference_ratio,
        "difference_temporal_pearson": difference_corr,
        "low_band_power_ratio": powers["low"][0],
        "mid_band_power_ratio": powers["mid"][0],
        "valid_temporal_edges": float(temporal_count),
        "valid_difference_edges": float(difference_count),
        "valid_difference_correlation_edges": float(correlation_count),
        "valid_low_band_edges": float(powers["low"][1]),
        "valid_mid_band_edges": float(powers["mid"][1]),
    }


def dynamic_state_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray, n_states: int) -> dict[str, float]:
    """比较预测和真实动态状态的占有率、转移概率与停留时间。"""
    def summarize(labels: np.ndarray):
        occupancy = np.bincount(labels, minlength=n_states) / len(labels)
        transitions = np.zeros((n_states, n_states), dtype=float)
        for left, right in zip(labels[:-1], labels[1:]):
            transitions[left, right] += 1
        transitions /= np.maximum(transitions.sum(1, keepdims=True), 1)
        dwell, counts, start = np.zeros(n_states), np.zeros(n_states), 0
        for index in range(1, len(labels) + 1):
            if index == len(labels) or labels[index] != labels[start]:
                state = labels[start]
                dwell[state] += index - start
                counts[state] += 1
                start = index
        return occupancy, transitions, dwell / np.maximum(counts, 1)

    pred_occ, pred_trans, pred_dwell = summarize(predicted_labels)
    true_occ, true_trans, true_dwell = summarize(true_labels)
    return {
        "state_occupancy_mae": float(np.mean(np.abs(pred_occ - true_occ))),
        "state_transition_mae": float(np.mean(np.abs(pred_trans - true_trans))),
        "state_dwell_mae": float(np.mean(np.abs(pred_dwell - true_dwell))),
    }


def retrieval_metrics(predictions: np.ndarray, targets: np.ndarray, template: np.ndarray, nonoverlap: int, subject_ids: Iterable[str] | None = None) -> dict[str, float]:
    """检验预测未来是否更接近同一被试的真实未来。"""
    pred = predictions[:, nonoverlap:].mean(1) - template[nonoverlap:].mean(0)
    true = targets[:, nonoverlap:].mean(1) - template[nonoverlap:].mean(0)
    pred, true = pred - pred.mean(-1, keepdims=True), true - true.mean(-1, keepdims=True)
    similarity = (pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12)) @ (true / np.maximum(np.linalg.norm(true, axis=-1, keepdims=True), 1e-12)).T
    identities = np.arange(len(pred)) if subject_ids is None else np.asarray(list(subject_ids))
    ranks = np.empty(len(pred), dtype=int)
    for index in range(len(pred)):
        order = np.argsort(-similarity[index])
        ranks[index] = int(np.flatnonzero(identities[order] == identities[index])[0]) + 1
    return {"retrieval_top1": float(np.mean(ranks == 1)), "retrieval_top5": float(np.mean(ranks <= 5)), "retrieval_mean_rank": float(ranks.mean())}


def subject_bootstrap_difference(main_scores: np.ndarray, baseline_scores: np.ndarray, subject_ids: Iterable[str], replicates: int = 2000, seed: int = DEFAULT_SEED) -> dict[str, float]:
    """先聚合同一被试的多个 run，再进行被试级 bootstrap。"""
    subject_ids = np.asarray(list(subject_ids))
    subjects = np.unique(subject_ids)
    difference = np.asarray(main_scores) - np.asarray(baseline_scores)
    values = {subject: difference[subject_ids == subject].mean() for subject in subjects}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        estimates[index] = np.mean([values[subject] for subject in rng.choice(subjects, size=len(subjects), replace=True)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_difference": float(difference.mean()), "ci_low": float(low), "ci_high": float(high), "passes": bool(low > 0)}


def projection_report(z_edges: np.ndarray, n_nodes: int = 90, epsilon: float = 1e-6) -> tuple[np.ndarray, dict[str, float]]:
    """将预测边转回相关矩阵，并量化 PSD 投影前后的差异。"""
    raw = edges_to_matrix(inverse_fisher_z(z_edges), n_nodes)
    projected = nearest_correlation(raw, epsilon)
    eig = np.linalg.eigvalsh(raw)
    return projected, {
        "negative_eigenvalue_fraction": float(np.mean(eig < -epsilon)),
        "minimum_eigenvalue": float(eig.min()),
        "projection_rmse": float(np.sqrt(np.mean((raw - projected) ** 2))),
    }


# ======================== 测试集推理与结果导出 ========================


def _load_model(config, window_length, checkpoint, stats_path, device, autoencoder_path=None):
    """按检查点记录的模型类型恢复模型和参数。"""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_sequence_model(
        config,
        window_length,
        payload["decoder_type"],
        stats_path,
        device,
        payload.get("sc_encoder_type", "hybrid"),
        autoencoder_path,
        payload,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def fit_state_model(dataset: DFCSequenceDataset, n_states: int, seed: int, max_windows: int = 50000) -> MiniBatchKMeans:
    """只用训练集 dFC 窗口拟合状态聚类器，避免测试标签泄漏。"""
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
    """批量推理并保持预测与 subject/run 身份一一对应。"""
    predictions, targets, subjects, runs = [], [], [], []
    for batch in loader:
        target = batch["fc_future"]
        result = model(
            batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device),
            steps=target.shape[1],
        )
        predictions.append(result.fc_z_edges.cpu().numpy())
        targets.append(target.numpy())
        subjects.extend(batch["subject_id"])
        runs.extend(batch["run_name"])
    return np.concatenate(predictions), np.concatenate(targets), subjects, runs


def _dynamic_audit_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = [
        "temporal_std_ratio", "difference_std_ratio", "difference_temporal_pearson",
        "low_band_power_ratio", "mid_band_power_ratio",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            aggregate[name] = {"median": float("nan"), "iqr": float("nan"), "n": 0.0}
            continue
        summary = {
            "median": float(np.median(values)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "n": float(len(values)),
        }
        if name.endswith("_ratio"):
            summary["fraction_below_one"] = float(np.mean(values < 1.0))
        aggregate[name] = summary
    return aggregate


def _dynamic_audit_horizons(steps: int, nonoverlap: int, sample_interval_seconds: float) -> list[dict[str, float | int | str]]:
    """Split the future into the overlap-context segment and three long-horizon segments."""
    if steps < 4:
        raise ValueError("Dynamic audit horizons require at least four time windows")
    boundary = min(max(int(nonoverlap), 4), steps - 3)
    long_boundaries = np.linspace(boundary, steps, num=4, dtype=int)
    segments = [("overlap_context", 0, boundary)]
    segments.extend(
        (name, int(start), int(stop))
        for name, start, stop in zip(("early_long", "middle_long", "late_long"), long_boundaries[:-1], long_boundaries[1:])
    )
    return [
        {
            "name": name,
            "start_index": start,
            "stop_index_exclusive": stop,
            "n_windows": stop - start,
            "start_minutes": float(start * sample_interval_seconds / 60.0),
            "stop_minutes": float(stop * sample_interval_seconds / 60.0),
        }
        for name, start, stop in segments
    ]


def dynamic_audit_checkpoint(
    config: dict[str, Any],
    window_length: int,
    checkpoint: str | Path,
    stats_path: str | Path,
    split_name: str = "val",
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    autoencoder_path: str | Path | None = None,
) -> Path:
    """Create a validation-only dynamic calibration audit for one managed checkpoint."""
    if split_name not in {"train", "val"}:
        raise ValueError("Dynamic audit is restricted to train or val splits")
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    model, _payload = _load_model(config, window_length, checkpoint, Path(stats_path), device, autoencoder_path)
    dataset = DFCSequenceDataset(config, window_length, split_name, stats_path)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    prediction, target, subjects, runs = collect_predictions(model, loader, device)
    template = model.group_template.cpu().numpy()[: target.shape[1]]
    warmup = np.stack([dataset[index]["fc_warmup"].numpy() for index in range(len(dataset))])
    model_name = str(config.get("experiment", {}).get("id", "model"))
    methods = {
        model_name: prediction,
        "group_mean": np.broadcast_to(template[None], target.shape),
        "fc1_persistence": np.broadcast_to(warmup[:, None], target.shape),
    }
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    interval = float(config["data"]["stride"]) * float(config["data"]["tr_seconds"])
    per_sample: list[dict[str, Any]] = []
    horizon_per_sample: list[dict[str, Any]] = []
    method_summaries: dict[str, dict[str, dict[str, float]]] = {}
    horizons = _dynamic_audit_horizons(target.shape[1], nonoverlap, interval)
    horizon_summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name, predicted in methods.items():
        rows = []
        horizon_rows: dict[str, list[dict[str, float]]] = {str(segment["name"]): [] for segment in horizons}
        for index, (pred, true, subject, run) in enumerate(zip(predicted, target, subjects, runs)):
            full = dynamic_calibration_metrics(pred, true, interval)
            long = dynamic_calibration_metrics(pred[nonoverlap:], true[nonoverlap:], interval)
            row = {
                "method": name, "subject_id": subject, "run": run, "sample_index": index,
                **{f"full_{key}": value for key, value in full.items()},
                **{f"nonoverlap_{key}": value for key, value in long.items()},
            }
            rows.append({key.removeprefix("full_"): value for key, value in row.items() if key.startswith("full_")})
            per_sample.append(row)
            for segment in horizons:
                start, stop = int(segment["start_index"]), int(segment["stop_index_exclusive"])
                metrics = dynamic_calibration_metrics(pred[start:stop], true[start:stop], interval)
                horizon_row = {
                    "method": name, "subject_id": subject, "run": run, "sample_index": index,
                    **segment, **metrics,
                }
                horizon_rows[str(segment["name"])].append(metrics)
                horizon_per_sample.append(horizon_row)
        method_summaries[name] = {
            "full": _dynamic_audit_aggregate(rows),
            "nonoverlap": _dynamic_audit_aggregate([
                {key.removeprefix("nonoverlap_"): value for key, value in row.items() if key.startswith("nonoverlap_")}
                for row in per_sample if row["method"] == name
            ]),
        }
        horizon_summaries[name] = {
            segment_name: _dynamic_audit_aggregate(segment_rows)
            for segment_name, segment_rows in horizon_rows.items()
        }
    model_delta = np.asarray([
        row["full_difference_std_ratio"] for row in per_sample if row["method"] == model_name
    ], dtype=float)
    model_delta = model_delta[np.isfinite(model_delta)]
    confirmed = bool(len(model_delta) and np.median(model_delta) < 0.80 and np.mean(model_delta < 1.0) >= 0.75)
    destination = Path(output_dir) if output_dir is not None else Path(checkpoint).resolve().parent
    destination = destination / f"dynamic_audit_{split_name}"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "per_sample.csv"
    fieldnames = list(per_sample[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample)
    horizon_csv_path = destination / "horizon_per_sample.csv"
    with horizon_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_per_sample[0]))
        writer.writeheader()
        writer.writerows(horizon_per_sample)
    summary = {
        "schema_version": 2,
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split_name,
        "n_samples": len(subjects),
        "window_length": window_length,
        "sample_interval_seconds": interval,
        "nonoverlap_horizon": nonoverlap,
        "frequency_bands_hz": _DYNAMIC_AUDIT_BANDS_HZ,
        "methods": method_summaries,
        "horizon_segments": horizons,
        "horizon_methods": horizon_summaries,
        "decision": {
            "criterion": f"{model_name} full difference_std_ratio median < 0.80 and fraction_below_one >= 0.75",
            "passes": confirmed,
            "recommended_next_step": "run variance loss experiment A1" if confirmed else "review phase/frequency diagnostics before variance loss",
        },
        "per_sample_csv": str(csv_path.resolve()),
        "horizon_per_sample_csv": str(horizon_csv_path.resolve()),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def evaluate_checkpoint(
    config: dict[str, Any],
    window_length: int,
    checkpoint: str | Path,
    stats_path: str | Path,
    baseline_checkpoint: str | Path | None = None,
    save_predictions: bool = False,
    device_name: str | None = None,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    autoencoder_path: str | Path | None = None,
) -> Path:
    """在测试集生成完整报告，并可选导出逐样本 FC 矩阵。"""
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be train, val, or test")
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    model, payload = _load_model(config, window_length, checkpoint, Path(stats_path), device, autoencoder_path)
    test = DFCSequenceDataset(config, window_length, split_name, stats_path, payload.get("ablation", "full"))
    train = DFCSequenceDataset(config, window_length, "train", stats_path)
    loader = DataLoader(test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    predictions, targets, subjects, runs = collect_predictions(model, loader, device)
    template = model.group_template.cpu().numpy()
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    metric_kwargs = {
        "huber_beta": float(config["training"].get("huber_beta", 1.0)),
        "difference_weight": float(config["training"].get("loss_weights", {}).get("difference", 0.0)),
    }
    rows = [sequence_metrics(p, t, template, nonoverlap, **metric_kwargs) for p, t in zip(predictions, targets)]
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
        baseline_rows = [sequence_metrics(p, t, template, nonoverlap, **metric_kwargs) for p, t in zip(baseline_prediction, targets)]
        analytic_reports[name] = {key: float(np.mean([row[key] for row in baseline_rows])) for key in baseline_rows[0]}
    projection_rows = [projection_report(p, int(config["data"]["n_nodes"]), float(config["evaluation"]["projection_epsilon"]))[1] for p in predictions]
    aggregate.update({f"projection_{key}": float(np.mean([row[key] for row in projection_rows])) for key in projection_rows[0]})
    report: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split_name,
        "window_length": window_length,
        "nonoverlap_horizon": nonoverlap,
        "n_samples": len(subjects),
        "aggregate": aggregate,
        "analytic_baselines": analytic_reports,
        "per_sample": [{"subject_id": s, "run": r, **m} for s, r, m in zip(subjects, runs, rows)],
    }
    if baseline_checkpoint:
        baseline_model, baseline_payload = _load_model(config, window_length, baseline_checkpoint, Path(stats_path), device, autoencoder_path)
        baseline_test = DFCSequenceDataset(config, window_length, split_name, stats_path, baseline_payload.get("ablation", "fc1_only"))
        baseline_loader = DataLoader(baseline_test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
        baseline_predictions, baseline_targets, baseline_subjects, _ = collect_predictions(baseline_model, baseline_loader, device)
        if subjects != baseline_subjects:
            raise ValueError("Main and baseline checkpoints do not cover the same ordered samples")
        main_scores = np.asarray([row["long_residual_pearson"] for row in rows])
        baseline_scores = np.asarray([sequence_metrics(p, t, template, nonoverlap, **metric_kwargs)["long_residual_pearson"] for p, t in zip(baseline_predictions, baseline_targets)])
        report["success_gate"] = subject_bootstrap_difference(main_scores, baseline_scores, subjects, int(config["evaluation"]["bootstrap_replicates"]), int(config["seed"]))
    output_dir = Path(output_dir) if output_dir is not None else Path(checkpoint).resolve().parent
    report_path = output_dir / f"evaluation_{split_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_predictions:
        prediction_dir = output_dir / "predictions" / split_name
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for pred, true, subject, run in zip(predictions, targets, subjects, runs):
            projected, projection = projection_report(pred, int(config["data"]["n_nodes"]), float(config["evaluation"]["projection_epsilon"]))
            raw_fc = edges_to_matrix(np.tanh(pred), int(config["data"]["n_nodes"]))
            np.savez_compressed(prediction_dir / f"{subject}_{run}.npz", predicted_z=pred, target_z=true, raw_fc=raw_fc, projected_fc=projected, projection_metrics=json.dumps(projection))
    return report_path


def evaluate_analytic_baseline(
    config: dict[str, Any],
    window_length: int,
    stats_path: str | Path,
    baseline: str,
    split_name: str,
    output_dir: str | Path,
) -> Path:
    """Evaluate group mean or first-window persistence with the common metric protocol."""
    if baseline not in {"group_mean", "fc1_persistence"}:
        raise ValueError("baseline must be group_mean or fc1_persistence")
    seed_everything(int(config["seed"]))
    dataset = DFCSequenceDataset(config, window_length, split_name, stats_path)
    stats = dict(np.load(stats_path))
    template = stats["group_template"]
    predictions, targets, subjects, runs = [], [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["fc_future"].numpy()
        prediction = (
            template[: len(target)]
            if baseline == "group_mean"
            else np.broadcast_to(sample["fc_warmup"].numpy()[None], target.shape)
        )
        predictions.append(prediction)
        targets.append(target)
        subjects.append(sample["subject_id"])
        runs.append(sample["run_name"])
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    metric_kwargs = {
        "huber_beta": float(config["training"].get("huber_beta", 1.0)),
        "difference_weight": float(config["training"].get("loss_weights", {}).get("difference", 0.0)),
    }
    rows = [sequence_metrics(p, t, template, nonoverlap, **metric_kwargs) for p, t in zip(predictions, targets)]
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    aggregate.update(retrieval_metrics(np.stack(predictions), np.stack(targets), template, nonoverlap, subjects))
    report = {
        "baseline": baseline, "split": split_name, "window_length": window_length,
        "n_samples": len(rows), "aggregate": aggregate,
        "per_sample": [{"subject_id": s, "run": r, **m} for s, r, m in zip(subjects, runs, rows)],
    }
    output_dir = Path(output_dir)
    report_path = output_dir / f"evaluation_{split_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "metrics_best.json").write_text(
        json.dumps({"metrics": aggregate, "primary_metric": config["evaluation"]["primary_metric"]}, indent=2),
        encoding="utf-8",
    )
    return report_path
