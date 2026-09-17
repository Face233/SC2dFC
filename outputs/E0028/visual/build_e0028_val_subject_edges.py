"""Render E0028's validation-only multi-subject, six-edge comparison.

Experiment: E0028; run: E0028-s42-20260917T065522Z-83077fe; split: val.
Inputs: frozen config/checkpoint and shared training statistics.
Outputs: ``*_val_subjects_6edges.png`` and its JSON manifest in this directory.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs" / "E0027" / "visual" / "build_e0027_val_subject_edges.py"
EXPERIMENT_ID = "E0028"
RUN_ID = "E0028-s42-20260917T065522Z-83077fe"


def main() -> None:
    spec = importlib.util.spec_from_file_location("e0027_val_subject_edges", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared multi-subject renderer: {SOURCE}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.EXPERIMENT_ID = EXPERIMENT_ID
    renderer.RUN_ID = RUN_ID
    renderer.main()


if __name__ == "__main__":
    main()
