from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest
import yaml

from scdfc.config import load_config
from scdfc.management import (
    REGISTRY_COLUMNS,
    conclude_experiment,
    config_sha256,
    create_run_context,
    file_sha256,
    manifest_digest,
    permitted_evaluation_split,
    summarize_experiment,
    validate_experiment_config,
    verify_data_bindings,
)


def managed_config(root: Path, level: int = 0) -> dict:
    return {
        "paths": {"root": str(root), "output_dir": "outputs", "split_csv": "data/split_v1.csv"},
        "experiment": {
            "id": "E0001", "name": "test", "level": level, "task": "analytic",
            "research_question": "question", "hypothesis": "hypothesis", "primary_change": "none", "owner": "tester",
        },
        "data": {
            "dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1",
            "manifest_path": "data/dataset_v1.json", "manifest_sha256": "pending",
            "split_version": "split_v1", "split_sha256": "pending",
        },
        "model": {"name": "group_mean"}, "training": {"seeds": [42]},
        "evaluation": {"primary_metric": "mse"},
        "decision_rule": {"description": "compare validation metric"},
    }


def test_config_inheritance_and_scientific_hash_ignore_machine_root(tmp_path: Path):
    configs = tmp_path / "configs"
    experiments = configs / "experiments"
    experiments.mkdir(parents=True)
    (configs / "base.yaml").write_text("paths:\n  root: .\n  output_dir: outputs\nmodel:\n  hidden: 8\n", encoding="utf-8")
    (experiments / "E0001_test.yaml").write_text("base: ../base.yaml\nmodel:\n  hidden: 16\n", encoding="utf-8")
    config = load_config(experiments / "E0001_test.yaml")
    assert config["model"]["hidden"] == 16
    other = dict(config)
    other["paths"] = {**config["paths"], "root": "X:/another-machine", "output_dir": "elsewhere"}
    assert config_sha256(config) == config_sha256(other)


def test_managed_config_validation_and_test_gate(tmp_path: Path):
    config = managed_config(tmp_path)
    validate_experiment_config(config)
    assert permitted_evaluation_split(0, "val") == "val"
    assert permitted_evaluation_split(2, final_test=True) == "test"
    with pytest.raises(PermissionError):
        permitted_evaluation_split(1, final_test=True)
    with pytest.raises(PermissionError):
        permitted_evaluation_split(0, "test")


def test_data_manifest_and_split_checksums_are_enforced(tmp_path: Path):
    config = managed_config(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    split = data / "split_v1.csv"
    split.write_text("subject_id,split\n1,train\n2,val\n3,test\n", encoding="utf-8")
    manifest = {"dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1", "files": []}
    manifest["manifest_sha256"] = manifest_digest(manifest)
    (data / "dataset_v1.json").write_text(json.dumps(manifest), encoding="utf-8")
    config["data"]["manifest_sha256"] = manifest["manifest_sha256"]
    config["data"]["split_sha256"] = file_sha256(split)
    assert verify_data_bindings(config)["dataset_version"] == "dataset_v1"
    split.write_text("subject_id,split\n1,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Split checksum mismatch"):
        verify_data_bindings(config)


def test_run_context_blocks_dirty_formal_runs_and_config_reuse(tmp_path: Path, monkeypatch):
    config = managed_config(tmp_path, level=1)
    monkeypatch.setattr("scdfc.management.git_metadata", lambda root: {"commit": "a" * 40, "branch": "main", "remote": "", "dirty": True, "status": ["M file"]})
    with pytest.raises(RuntimeError, match="clean Git"):
        create_run_context(config, 42)
    monkeypatch.setattr("scdfc.management.git_metadata", lambda root: {"commit": "a" * 40, "branch": "main", "remote": "", "dirty": False, "status": []})
    context = create_run_context(config, 42)
    (context.run_dir / "metadata.json").write_text(json.dumps({"config_sha256": context.config_hash}), encoding="utf-8")
    changed = json.loads(json.dumps(config))
    changed["model"]["name"] = "fc1_persistence"
    with pytest.raises(RuntimeError, match="another config hash"):
        create_run_context(changed, 42)


def test_summary_and_human_conclusion_update_registry(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    registry = reports / "experiment_registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerow({"experiment_id": "E0001", "name": "test", "primary_metric": "mse", "status": "PLANNED"})
    run_dir = tmp_path / "outputs" / "E0001" / "runs" / "E0001-s42-now-aaaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": run_dir.name, "status": "COMPLETED"}), encoding="utf-8")
    (run_dir / "metrics_best.json").write_text(json.dumps({"metrics": {"mse": 0.25}}), encoding="utf-8")
    summary = summarize_experiment(tmp_path, "E0001")
    assert summary["mean"] == 0.25
    conclude_experiment(tmp_path, "E0001", "KEEP", "useful", "run L2")
    row = next(csv.DictReader(registry.open("r", encoding="utf-8-sig")))
    assert row["status"] == "KEEP"
    assert row["conclusion"] == "useful"
