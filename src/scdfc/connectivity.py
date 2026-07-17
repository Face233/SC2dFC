from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np


def upper_triangle_indices(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_nodes, k=1)


def matrix_to_edges(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"Expected square matrices, got {matrix.shape}")
    return matrix[..., upper_triangle_indices(matrix.shape[-1])[0], upper_triangle_indices(matrix.shape[-1])[1]]


def edges_to_matrix(edges: np.ndarray, n_nodes: int = 90, diagonal: float = 1.0) -> np.ndarray:
    edges = np.asarray(edges)
    expected = n_nodes * (n_nodes - 1) // 2
    if edges.shape[-1] != expected:
        raise ValueError(f"Expected {expected} edges, got {edges.shape[-1]}")
    out = np.zeros((*edges.shape[:-1], n_nodes, n_nodes), dtype=edges.dtype)
    i, j = upper_triangle_indices(n_nodes)
    out[..., i, j] = edges
    out[..., j, i] = edges
    diagonal_idx = np.arange(n_nodes)
    out[..., diagonal_idx, diagonal_idx] = diagonal
    return out


def fisher_z(correlation_edges: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    return np.arctanh(np.clip(correlation_edges, -clip, clip))


def inverse_fisher_z(z_edges: np.ndarray) -> np.ndarray:
    return np.tanh(z_edges)


def sliding_window_fc(
    timeseries: np.ndarray,
    window_length: int,
    stride: int,
    fisher_clip: float = 0.999999,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rectangular-window Pearson FC and return Fisher-z upper edges."""
    x = np.asarray(timeseries, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("timeseries must have shape [time, node]")
    if not 2 <= window_length <= x.shape[0]:
        raise ValueError("window_length must be between 2 and the number of timepoints")
    if stride < 1:
        raise ValueError("stride must be positive")
    starts = np.arange(0, x.shape[0] - window_length + 1, stride, dtype=np.int32)
    edges = np.empty((len(starts), x.shape[1] * (x.shape[1] - 1) // 2), dtype=np.float32)
    tri = upper_triangle_indices(x.shape[1])
    for k, start in enumerate(starts):
        corr = np.corrcoef(x[start : start + window_length], rowvar=False)
        if not np.isfinite(corr).all():
            raise ValueError(f"Non-finite FC at window beginning {start}")
        edges[k] = fisher_z(corr[tri], fisher_clip).astype(np.float32)
    return edges, starts


def nearest_correlation(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project symmetric matrices to positive-definite correlation matrices."""
    x = np.asarray(matrix, dtype=np.float64)
    symmetric = (x + np.swapaxes(x, -1, -2)) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, epsilon)
    psd = (vectors * values[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    diag = np.sqrt(np.maximum(np.diagonal(psd, axis1=-2, axis2=-1), epsilon))
    corr = psd / (diag[..., :, None] * diag[..., None, :])
    idx = np.arange(corr.shape[-1])
    corr[..., idx, idx] = 1.0
    return corr.astype(matrix.dtype, copy=False)


def config_hash(items: dict) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def expected_windows(n_timepoints: int, window_length: int, stride: int) -> int:
    return (n_timepoints - window_length) // stride + 1


def nonoverlap_horizon(window_length: int, stride: int) -> int:
    return int(np.ceil(window_length / stride))

