from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, wasserstein_distance
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .config import resolve_path
from .connectivity import edges_to_matrix, inverse_fisher_z, nearest_correlation, nonoverlap_horizon
from .data import DFCSequenceDataset, read_cached
from .training import build_sequence_model, device_from_arg


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


def sequence_metrics(prediction: np.ndarray, target: np.ndarray, template: np.ndarray, nonoverlap: int) -> dict[str, float]:
    """汇总单个 subject/run 的边级、动态和 FCD 指标。"""
    pred_r, target_r = np.tanh(prediction), np.tanh(target)
    long_pred, long_target = prediction[nonoverlap:] - template[nonoverlap:], target[nonoverlap:] - template[nonoverlap:]
    raw_corr = _row_correlation(prediction, target)
    raw_spearman = _row_correlation(rankdata(prediction, axis=-1), rankdata(target, axis=-1))
    residual_corr = _row_correlation(long_pred, long_target)
    pred_fcd, true_fcd = fcd(pred_r), fcd(target_r)
    tri = np.triu_indices(len(pred_fcd), 1)
    pred_diff, target_diff = np.diff(prediction, axis=0), np.diff(target, axis=0)
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "raw_edge_pearson": float(np.nanmean(raw_corr)),
        "raw_edge_spearman": float(np.nanmean(raw_spearman)),
        "long_residual_pearson": float(np.nanmean(residual_corr)),
        "difference_mse": float(np.mean((pred_diff - target_diff) ** 2)),
        "variance_mae": float(np.mean(np.abs(prediction.var(0) - target.var(0)))),
        "dynamic_amplitude_mae": float(abs(pred_diff.std() - target_diff.std())),
        "fcd_pearson": float(pearsonr(pred_fcd[tri], true_fcd[tri]).statistic),
        "fcd_wasserstein": float(wasserstein_distance(pred_fcd[tri], true_fcd[tri])),
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
    """检验预测未来是否更接近同一被试的真实未来；LR/RL 视为同一身份。"""
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


def subject_bootstrap_difference(main_scores: np.ndarray, baseline_scores: np.ndarray, subject_ids: Iterable[str], replicates: int = 2000, seed: int = 20260717) -> dict[str, float]:
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


def _load_model(config, window_length, checkpoint, stats_path, device):
    """按检查点记录的模型类型恢复模型和参数。"""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_sequence_model(
        config,
        window_length,
        payload["decoder_type"],
        stats_path,
        device,
        payload.get("sc_encoder_type", "hybrid"),
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
        result = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device), batch["run"].to(device))
        predictions.append(result.fc_z_edges.cpu().numpy())
        targets.append(batch["fc_future"].numpy())
        subjects.extend(batch["subject_id"])
        runs.extend(batch["run"].numpy().tolist())
    return np.concatenate(predictions), np.concatenate(targets), subjects, runs


def evaluate_checkpoint(
    config: dict[str, Any],
    window_length: int,
    checkpoint: str | Path,
    stats_path: str | Path,
    baseline_checkpoint: str | Path | None = None,
    save_predictions: bool = False,
    device_name: str | None = None,
) -> Path:
    """在测试集生成完整报告，并可选导出逐样本 FC 矩阵。"""
    device = device_from_arg(device_name)
    model, payload = _load_model(config, window_length, checkpoint, Path(stats_path), device)
    test = DFCSequenceDataset(config, window_length, "test", stats_path, payload.get("ablation", "full"))
    train = DFCSequenceDataset(config, window_length, "train", stats_path)
    loader = DataLoader(test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    predictions, targets, subjects, runs = collect_predictions(model, loader, device)
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
        "per_sample": [{"subject_id": s, "run": r, **m} for s, r, m in zip(subjects, runs, rows)],
    }
    if baseline_checkpoint:
        baseline_model, baseline_payload = _load_model(config, window_length, baseline_checkpoint, Path(stats_path), device)
        baseline_test = DFCSequenceDataset(config, window_length, "test", stats_path, baseline_payload.get("ablation", "fc1_only"))
        baseline_loader = DataLoader(baseline_test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
        baseline_predictions, baseline_targets, baseline_subjects, _ = collect_predictions(baseline_model, baseline_loader, device)
        if subjects != baseline_subjects:
            raise ValueError("Main and baseline checkpoints do not cover the same ordered samples")
        main_scores = np.asarray([row["long_residual_pearson"] for row in rows])
        baseline_scores = np.asarray([sequence_metrics(p, t, template, nonoverlap)["long_residual_pearson"] for p, t in zip(baseline_predictions, baseline_targets)])
        report["success_gate"] = subject_bootstrap_difference(main_scores, baseline_scores, subjects, int(config["evaluation"]["bootstrap_replicates"]), int(config["seed"]))
    output_dir = Path(checkpoint).resolve().parent
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_predictions:
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        for pred, true, subject, run in zip(predictions, targets, subjects, runs):
            projected, projection = projection_report(pred, int(config["data"]["n_nodes"]), float(config["evaluation"]["projection_epsilon"]))
            raw_fc = edges_to_matrix(np.tanh(pred), int(config["data"]["n_nodes"]))
            np.savez_compressed(prediction_dir / f"{subject}_{'LR' if run == 0 else 'RL'}.npz", predicted_z=pred, target_z=true, raw_fc=raw_fc, projected_fc=projected, projection_metrics=json.dumps(projection))
    return report_path
