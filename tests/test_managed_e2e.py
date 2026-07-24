from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from scdfc.data import precompute_dfc
from scdfc.managed_cli import command_run
from scdfc.management import file_sha256, manifest_digest


def test_level0_analytic_experiment_runs_end_to_end(tmp_path: Path, monkeypatch):
    sc_dir = tmp_path / "data" / "sc"
    ts_dir = tmp_path / "data" / "timeseries"
    manifest_dir = tmp_path / "data" / "manifests"
    sc_dir.mkdir(parents=True)
    ts_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    subjects = ["100001", "100002", "100003"]
    rng = np.random.default_rng(7)
    for subject in subjects:
        sc = rng.uniform(size=(4, 4))
        sc = (sc + sc.T) / 2
        np.fill_diagonal(sc, 0)
        np.savetxt(sc_dir / f"{subject}.csv", sc, delimiter=",")
        frame = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("ABCD"))
        frame.insert(0, "timepoint", np.arange(30))
        frame.to_csv(ts_dir / f"{subject}_AAL90_timeseries.csv", index=False)
    split_path = manifest_dir / "split_v1.csv"
    pd.DataFrame({"subject_id": subjects, "split": ["train", "val", "test"]}).to_csv(split_path, index=False)
    manifest = {"dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1", "files": []}
    manifest["manifest_sha256"] = manifest_digest(manifest)
    manifest_path = manifest_dir / "dataset_v1.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "paths": {
            "root": str(tmp_path), "atlas_labels": "unused", "sc_dir": "data/sc",
            "timeseries": {"LR": "data/timeseries"}, "split_csv": "data/manifests/split_v1.csv",
            "cache_dir": "data/cache", "output_dir": "outputs",
        },
        "seed": 42,
        "data": {
            "n_nodes": 4, "n_timepoints": 30, "window_length": 10, "stride": 5,
            "fisher_clip": 0.999999, "cache_dtype": "float32",
            "dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1",
            "manifest_path": "data/manifests/dataset_v1.json", "manifest_sha256": manifest["manifest_sha256"],
            "split_version": "split_v1", "split_sha256": file_sha256(split_path),
        },
        "experiment": {
            "id": "E0001", "name": "mean_smoke", "level": 0, "task": "analytic",
            "research_question": "Does the managed path run?", "hypothesis": "The mean baseline completes",
            "primary_change": "managed runner", "owner": "test", "ablation": "full",
        },
        "model": {"name": "group_mean"},
        "training": {"seeds": [42], "batch_size": 1},
        "evaluation": {"primary_metric": "mse"},
        "decision_rule": {"description": "smoke test completes"},
    }
    config_path = tmp_path / "configs" / "experiments" / "E0001_mean_smoke.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    precompute_dfc(config, 10)
    monkeypatch.setattr(
        "scdfc.management.git_metadata",
        lambda root: {"commit": "a" * 40, "branch": "main", "remote": "", "dirty": False, "status": []},
    )
    command_run(SimpleNamespace(experiment=str(config_path), seed=42, device="cpu"))
    run_dirs = list((tmp_path / "outputs" / "E0001" / "runs").iterdir())
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETED"
    assert (run_dirs[0] / "config_resolved.yaml").exists()
    assert (run_dirs[0] / "metrics_best.json").exists()
