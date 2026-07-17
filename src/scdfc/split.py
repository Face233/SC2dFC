from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_subject_split(
    subjects: list[str],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260717,
) -> pd.DataFrame:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    subject_ids = np.asarray(sorted(set(map(str, subjects))))
    if len(subject_ids) < 3:
        raise ValueError("At least three unique subjects are required for train/val/test splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)
    n_train = int(round(len(subject_ids) * fractions[0]))
    n_val = int(round(len(subject_ids) * fractions[1]))
    n_train = min(max(n_train, 1), len(subject_ids) - 2)
    n_val = min(max(n_val, 1), len(subject_ids) - n_train - 1)
    labels = np.full(len(subject_ids), "test", dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "val"
    return pd.DataFrame({"subject_id": subject_ids, "split": labels}).sort_values("subject_id").reset_index(drop=True)


def validate_split(table: pd.DataFrame) -> None:
    if table.subject_id.duplicated().any():
        raise ValueError("A subject occurs in more than one split")
    if set(table.split) - {"train", "val", "test"}:
        raise ValueError("Unknown split label")


def load_split(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"subject_id": str})
    validate_split(table)
    return table
