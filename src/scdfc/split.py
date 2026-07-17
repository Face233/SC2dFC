from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_family_split(
    subjects: list[str],
    family_table: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260717,
) -> pd.DataFrame:
    required = {"subject_id", "family_id"}
    if not required.issubset(family_table.columns):
        raise ValueError(f"Family table must contain {sorted(required)}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    table = family_table.copy()
    table["subject_id"] = table["subject_id"].astype(str)
    table["family_id"] = table["family_id"].astype(str)
    table = table[table.subject_id.isin(set(map(str, subjects)))].drop_duplicates("subject_id")
    missing = sorted(set(map(str, subjects)) - set(table.subject_id))
    if missing:
        raise ValueError(f"Missing family IDs for {len(missing)} subjects; examples: {missing[:5]}")

    sizes = table.groupby("family_id").size().sort_values(ascending=False)
    families = sizes.index.to_numpy().copy()
    rng = np.random.default_rng(seed)
    # Randomize equal-sized groups while placing large families first.
    jitter = rng.random(len(sizes))
    order = np.lexsort((jitter, -sizes.to_numpy()))
    families = families[order]
    target = np.asarray(fractions) * len(table)
    counts = np.zeros(3, dtype=float)
    assignment: dict[str, str] = {}
    labels = np.asarray(["train", "val", "test"])
    size_map = sizes.to_dict()
    for family in families:
        relative_fill = counts / np.maximum(target, 1)
        split_idx = int(np.argmin(relative_fill))
        assignment[str(family)] = str(labels[split_idx])
        counts[split_idx] += size_map[family]
    table["split"] = table.family_id.map(assignment)
    return table[["subject_id", "family_id", "split"]].sort_values("subject_id").reset_index(drop=True)


def validate_split(table: pd.DataFrame) -> None:
    if table.subject_id.duplicated().any():
        raise ValueError("A subject occurs in more than one split")
    family_counts = table.groupby("family_id").split.nunique()
    if (family_counts > 1).any():
        bad = family_counts[family_counts > 1].index.tolist()[:5]
        raise ValueError(f"Families cross splits: {bad}")
    if set(table.split) - {"train", "val", "test"}:
        raise ValueError("Unknown split label")


def load_split(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"subject_id": str, "family_id": str})
    validate_split(table)
    return table

