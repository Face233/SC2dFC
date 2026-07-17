from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, rankdata, wasserstein_distance

from .connectivity import edges_to_matrix, inverse_fisher_z, nearest_correlation


def _row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.maximum(denominator, 1e-12)


def fcd(sequence: np.ndarray) -> np.ndarray:
    centered = sequence - sequence.mean(-1, keepdims=True)
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def sequence_metrics(prediction: np.ndarray, target: np.ndarray, template: np.ndarray, nonoverlap: int) -> dict[str, float]:
    pred_r, target_r = np.tanh(prediction), np.tanh(target)
    long_pred = prediction[nonoverlap:] - template[nonoverlap:]
    long_target = target[nonoverlap:] - template[nonoverlap:]
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
    def summarize(labels: np.ndarray):
        occupancy = np.bincount(labels, minlength=n_states) / len(labels)
        transitions = np.zeros((n_states, n_states), dtype=float)
        for left, right in zip(labels[:-1], labels[1:]):
            transitions[left, right] += 1
        transitions /= np.maximum(transitions.sum(1, keepdims=True), 1)
        dwell = np.zeros(n_states, dtype=float)
        counts = np.zeros(n_states, dtype=float)
        start = 0
        for index in range(1, len(labels) + 1):
            if index == len(labels) or labels[index] != labels[start]:
                state = labels[start]
                dwell[state] += index - start
                counts[state] += 1
                start = index
        dwell /= np.maximum(counts, 1)
        return occupancy, transitions, dwell
    pred_occ, pred_trans, pred_dwell = summarize(predicted_labels)
    true_occ, true_trans, true_dwell = summarize(true_labels)
    return {
        "state_occupancy_mae": float(np.mean(np.abs(pred_occ - true_occ))),
        "state_transition_mae": float(np.mean(np.abs(pred_trans - true_trans))),
        "state_dwell_mae": float(np.mean(np.abs(pred_dwell - true_dwell))),
    }


def retrieval_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    template: np.ndarray,
    nonoverlap: int,
    subject_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    pred = predictions[:, nonoverlap:].mean(1) - template[nonoverlap:].mean(0)
    true = targets[:, nonoverlap:].mean(1) - template[nonoverlap:].mean(0)
    pred = pred - pred.mean(-1, keepdims=True)
    true = true - true.mean(-1, keepdims=True)
    similarity = (pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12)) @ (
        true / np.maximum(np.linalg.norm(true, axis=-1, keepdims=True), 1e-12)
    ).T
    identities = np.arange(len(pred)) if subject_ids is None else np.asarray(list(subject_ids))
    ranks = np.empty(len(pred), dtype=int)
    for index in range(len(pred)):
        order = np.argsort(-similarity[index])
        matches = np.flatnonzero(identities[order] == identities[index])
        ranks[index] = int(matches[0]) + 1
    return {"retrieval_top1": float(np.mean(ranks == 1)), "retrieval_top5": float(np.mean(ranks <= 5)), "retrieval_mean_rank": float(ranks.mean())}


def subject_bootstrap_difference(
    main_scores: np.ndarray,
    baseline_scores: np.ndarray,
    subject_ids: Iterable[str],
    replicates: int = 2000,
    seed: int = 20260717,
) -> dict[str, float]:
    subject_ids = np.asarray(list(subject_ids))
    subjects = np.unique(subject_ids)
    difference = np.asarray(main_scores) - np.asarray(baseline_scores)
    subject_values = {subject: difference[subject_ids == subject].mean() for subject in subjects}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        estimates[index] = np.mean([subject_values[subject] for subject in sampled])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_difference": float(difference.mean()), "ci_low": float(low), "ci_high": float(high), "passes": bool(low > 0)}


def projection_report(z_edges: np.ndarray, n_nodes: int = 90, epsilon: float = 1e-6) -> tuple[np.ndarray, dict[str, float]]:
    raw = edges_to_matrix(inverse_fisher_z(z_edges), n_nodes)
    projected = nearest_correlation(raw, epsilon)
    eig = np.linalg.eigvalsh(raw)
    return projected, {
        "negative_eigenvalue_fraction": float(np.mean(eig < -epsilon)),
        "minimum_eigenvalue": float(eig.min()),
        "projection_rmse": float(np.sqrt(np.mean((raw - projected) ** 2))),
    }
