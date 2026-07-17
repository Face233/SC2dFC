from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import iter_cached_samples, read_cached
from .config import resolve_path
from .connectivity import matrix_to_edges
from .split import load_split

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows preprocessing utilities in a lightweight environment.
    torch = None
    Dataset = object


def load_sc(config: dict[str, Any], subject: str) -> np.ndarray:
    path = resolve_path(config, "sc_dir") / f"{subject}.csv"
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def fit_training_statistics(config: dict[str, Any], window_length: int, output: str | Path) -> dict[str, np.ndarray]:
    split = load_split(resolve_path(config, "split_csv"))
    train_subjects = set(split.loc[split.split == "train", "subject_id"].astype(str))
    sc_edges: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    seen_sc: set[str] = set()
    for subject, run in iter_cached_samples(config, window_length):
        if subject not in train_subjects:
            continue
        if subject not in seen_sc:
            sc_edges.append(np.log1p(matrix_to_edges(load_sc(config, subject))))
            seen_sc.add(subject)
        fc, _ = read_cached(config, window_length, subject, run)
        sequences.append(fc[1:])
    if not sequences:
        raise ValueError("No cached training sequences found")
    stacked_sc = np.stack(sc_edges)
    stacked_fc = np.stack(sequences)
    stats = {
        "sc_mean": stacked_sc.mean(0).astype(np.float32),
        "sc_std": np.maximum(stacked_sc.std(0), 1e-6).astype(np.float32),
        "fc_mean": stacked_fc.mean((0, 1)).astype(np.float32),
        "fc_std": np.maximum(stacked_fc.std((0, 1)), 1e-6).astype(np.float32),
        "group_template": stacked_fc.mean(0).astype(np.float32),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **stats)
    return stats


class DFCSequenceDataset(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        window_length: int,
        split_name: str,
        stats_path: str | Path,
        ablation: str = "full",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for model datasets")
        self.config = config
        self.window_length = window_length
        self.stats = dict(np.load(stats_path))
        split = load_split(resolve_path(config, "split_csv"))
        self.family = dict(zip(split.subject_id.astype(str), split.family_id.astype(str)))
        allowed = set(split.loc[split.split == split_name, "subject_id"].astype(str))
        self.samples = [(s, r) for s, r in iter_cached_samples(config, window_length) if s in allowed]
        self.ablation = ablation
        self.mean_sc = self.stats["sc_mean"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject, run = self.samples[index]
        sc_matrix = load_sc(self.config, subject)
        sc_edges = np.log1p(matrix_to_edges(sc_matrix))
        sc_edges = (sc_edges - self.stats["sc_mean"]) / self.stats["sc_std"]
        if self.ablation == "fc1_only":
            sc_edges = np.zeros_like(sc_edges)
            sc_matrix = np.zeros_like(sc_matrix)
        elif self.ablation == "mean_sc":
            sc_edges = np.zeros_like(sc_edges)
            from .connectivity import edges_to_matrix
            sc_matrix = edges_to_matrix(np.expm1(self.stats["sc_mean"]), int(self.config["data"]["n_nodes"]), diagonal=0.0)
        elif self.ablation == "shuffled_sc":
            if len(self.samples) < 2:
                raise ValueError("shuffled_sc requires at least two samples")
            candidate = (index + max(1, len(self.samples) // 2)) % len(self.samples)
            while self.samples[candidate][0] == subject:
                candidate = (candidate + 1) % len(self.samples)
            other_subject, _ = self.samples[candidate]
            sc_matrix = load_sc(self.config, other_subject)
            raw = np.log1p(matrix_to_edges(sc_matrix))
            sc_edges = (raw - self.stats["sc_mean"]) / self.stats["sc_std"]
        fc, starts = read_cached(self.config, self.window_length, subject, run)
        warmup = fc[0]
        if self.ablation == "sc_only":
            warmup = np.zeros_like(warmup)
        return {
            "subject_id": subject,
            "family_id": self.family[subject],
            "run": 0 if run.upper() == "LR" else 1,
            "sc_matrix": torch.from_numpy(sc_matrix.astype(np.float32)),
            "sc_edges": torch.from_numpy(sc_edges.astype(np.float32)),
            "fc_warmup": torch.from_numpy(warmup),
            "fc_future": torch.from_numpy(fc[1:]),
            "window_starts": torch.from_numpy(starts),
        }


class FCWindowDataset(Dataset):
    def __init__(self, sequence_dataset: DFCSequenceDataset, windows_per_run: int = 32, seed: int = 0) -> None:
        self.sequence_dataset = sequence_dataset
        self.windows_per_run = windows_per_run
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sequence_dataset) * self.windows_per_run

    def __getitem__(self, index: int):
        run_index = index // self.windows_per_run
        sample = self.sequence_dataset[run_index]
        sequence = torch.cat([sample["fc_warmup"][None], sample["fc_future"]], dim=0)
        generator = np.random.default_rng(self.seed + index)
        window = int(generator.integers(0, len(sequence)))
        return sequence[window]
