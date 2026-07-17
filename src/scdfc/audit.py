from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path


def _subject_from_timeseries(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def discover_data(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["root"])
    sc_dir = resolve_path(config, "sc_dir")
    run_dirs = config["paths"]["timeseries"]
    sc = {p.stem: p for p in sc_dir.glob("*.csv")}
    runs: dict[str, dict[str, Path]] = {}
    for run, relative in run_dirs.items():
        directory = Path(relative)
        if not directory.is_absolute():
            directory = root / directory
        runs[run] = {_subject_from_timeseries(p): p for p in directory.glob("*.csv")} if directory.exists() else {}
    subjects = sorted(set(sc).intersection(set().union(*(set(x) for x in runs.values()))))
    return {"sc": sc, "runs": runs, "subjects": subjects}


def audit_dataset(config: dict[str, Any], sample_limit: int | None = None) -> dict[str, Any]:
    found = discover_data(config)
    n_nodes = int(config["data"]["n_nodes"])
    n_timepoints = int(config["data"]["n_timepoints"])
    report: dict[str, Any] = {
        "sc_count": len(found["sc"]),
        "runs": {run: len(paths) for run, paths in found["runs"].items()},
        "paired_subjects_any_run": len(found["subjects"]),
        "run_pair_counts": {},
        "errors": [],
        "warnings": [],
    }
    for run, paths in found["runs"].items():
        report["run_pair_counts"][run] = len(set(paths).intersection(found["sc"]))

    atlas_path = resolve_path(config, "atlas_labels")
    atlas_names = [line.split("\t")[1] for line in atlas_path.read_text(encoding="utf-8").splitlines()]
    if len(atlas_names) < n_nodes:
        report["errors"].append(f"Atlas contains {len(atlas_names)} labels, expected at least {n_nodes}")
    atlas_names = atlas_names[:n_nodes]

    subjects = found["subjects"][:sample_limit]
    sc_zero_fractions: list[float] = []
    for subject in subjects:
        sc = pd.read_csv(found["sc"][subject], header=None).to_numpy(dtype=float)
        if sc.shape != (n_nodes, n_nodes):
            report["errors"].append(f"SC {subject} shape is {sc.shape}")
            continue
        if not np.isfinite(sc).all() or np.max(np.abs(sc - sc.T)) > 1e-6:
            report["errors"].append(f"SC {subject} is non-finite or asymmetric")
        sc_zero_fractions.append(float(np.mean(sc == 0)))
        for run, paths in found["runs"].items():
            if subject not in paths:
                continue
            frame = pd.read_csv(paths[subject])
            roi_names = frame.columns[1:].tolist()
            if frame.shape != (n_timepoints, n_nodes + 1):
                report["errors"].append(f"BOLD {subject}/{run} shape is {frame.shape}")
            if roi_names != atlas_names:
                report["errors"].append(f"ROI order mismatch for {subject}/{run}")
            if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
                report["errors"].append(f"Non-finite BOLD values for {subject}/{run}")
    if sc_zero_fractions:
        report["sc_zero_fraction"] = {
            "median": float(np.median(sc_zero_fractions)),
            "min": float(np.min(sc_zero_fractions)),
            "max": float(np.max(sc_zero_fractions)),
        }
    if not found["runs"].get("RL"):
        report["warnings"].append("RL timeseries are not available")
    return report


def write_audit(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
