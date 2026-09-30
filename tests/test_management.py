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


def test_managed_gru_experiment_accepts_objective_loss(tmp_path: Path):
    config = managed_config(tmp_path)
    config["experiment"]["task"] = "sequence"
    config["model"] = {"name": "gru", "sc_encoder": "hcp_gcn", "output_head": "e0003_reconstruction_decoder"}
    config["evaluation"]["primary_metric"] = "objective_loss"
    config["artifacts"] = {"fc_autoencoder": {"id": "A0003", "path": "best.pt", "sha256": "abc"}}
    assert validate_experiment_config(config)["model"]["name"] == "gru"


def test_residual_ridge_requires_frozen_encoder_and_decay_artifacts(tmp_path: Path):
    config = managed_config(tmp_path)
    config["experiment"]["task"] = "residual_ridge"
    config["model"] = {"name": "encoded_stable_residual_ridge", "ridge_alpha_grid": [1.0, 10.0], "cv_folds": 3}
    config["evaluation"]["primary_metric"] = "long_edge_mse"
    config["artifacts"] = {
        "fc_autoencoder": {"path": "encoder.pt", "sha256": "a"},
        "alpha_fit": {"path": "alpha.npz", "sha256": "b"},
    }
    assert validate_experiment_config(config)["experiment"]["task"] == "residual_ridge"
    del config["artifacts"]["alpha_fit"]
    with pytest.raises(ValueError, match="artifacts.alpha_fit"):
        validate_experiment_config(config)


def test_managed_gru_experiment_accepts_direct_edge_head(tmp_path: Path):
    config = managed_config(tmp_path)
    config["experiment"]["task"] = "sequence"
    config["model"] = {"name": "gru", "sc_encoder": "hcp_gcn", "output_head": "direct_edge_linear"}
    config["evaluation"]["primary_metric"] = "objective_loss"
    config["artifacts"] = {"fc_autoencoder": {"id": "A0003", "path": "best.pt", "sha256": "abc"}}
    assert validate_experiment_config(config)["model"]["output_head"] == "direct_edge_linear"


def test_e0036_e0037_configs_share_protocol_except_sc_ablation():
    base = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    fc_only = validate_experiment_config(load_config(base / "E0036_fc1_direct_residual_transformer_mse_v1.yaml"))
    with_sc = validate_experiment_config(load_config(base / "E0037_fc1_sc_direct_residual_transformer_mse_v1.yaml"))
    for section in ("data", "model", "artifacts", "training", "evaluation"):
        assert fc_only[section] == with_sc[section]
    assert fc_only["experiment"]["ablation"] == "fc1_only"
    assert with_sc["experiment"]["ablation"] == "full"
    assert fc_only["training"]["loss_weights"]["edge"] == 1.0
    assert all(value == 0 for name, value in fc_only["training"]["loss_weights"].items() if name != "edge")


def test_e0038_changes_only_difference_and_variance_objective():
    base = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    e0036 = validate_experiment_config(load_config(base / "E0036_fc1_direct_residual_transformer_mse_v1.yaml"))
    e0038 = validate_experiment_config(load_config(base / "E0038_fc1_direct_residual_transformer_mse_difference_variance_v1.yaml"))
    for section in ("data", "model", "artifacts", "evaluation"):
        assert e0038[section] == e0036[section]
    expected_training = dict(e0036["training"])
    expected_training["loss_weights"] = {
        **expected_training["loss_weights"], "difference": 0.25, "variance": 1.0,
    }
    assert e0038["training"] == expected_training
    assert e0038["experiment"]["ablation"] == "fc1_only"


def test_e0039_uses_only_equal_difference_and_variance_weights():
    base = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    e0036 = validate_experiment_config(load_config(base / "E0036_fc1_direct_residual_transformer_mse_v1.yaml"))
    e0039 = validate_experiment_config(load_config(base / "E0039_fc1_direct_residual_transformer_difference_variance_only_v1.yaml"))
    for section in ("data", "model", "artifacts", "evaluation"):
        assert e0039[section] == e0036[section]
    expected_training = dict(e0036["training"])
    expected_training["loss_weights"] = {
        **expected_training["loss_weights"], "edge": 0.0, "difference": 1.0, "variance": 1.0,
    }
    assert e0039["training"] == expected_training
    assert e0039["experiment"]["ablation"] == "fc1_only"
    e0039["training"]["loss_weights"]["variance"] = 0.0
    with pytest.raises(ValueError, match="positive difference and variance"):
        validate_experiment_config(e0039)


def test_managed_sequence_config_rejects_fc_decoder_finetuning(tmp_path: Path):
    config = managed_config(tmp_path)
    config["experiment"]["task"] = "sequence"
    config["model"] = {"name": "gru", "sc_encoder": "hcp_gcn", "output_head": "e0003_reconstruction_decoder"}
    config["training"]["finetune_fc_decoder"] = True
    config["evaluation"]["primary_metric"] = "objective_loss"
    config["artifacts"] = {"fc_autoencoder": {"id": "A0003", "path": "best.pt", "sha256": "abc"}}
    with pytest.raises(ValueError, match="remain frozen throughout"):
        validate_experiment_config(config)


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
        writer.writerow({"experiment_id": "E0001", "name": "test", "primary_metric": "mse", "status": "PLANNED", "config_sha256": "same"})
    run_dir = tmp_path / "outputs" / "E0001" / "runs" / "E0001-s42-now-aaaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": run_dir.name, "status": "COMPLETED", "config_sha256": "same"}), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump({
        "paths": {"cache_dir": "data/cache/dfc"},
        "data": {
            "dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1", "split_version": "split_v1",
            "window_length": 83, "stride": 5, "tr_seconds": 0.72, "n_timepoints": 1200,
            "n_nodes": 90, "fisher_clip": 0.999999,
        },
    }), encoding="utf-8")
    from scdfc.metric_records import selection_record
    record = selection_record({"mse": 0.25}, "mse", None, definition={"metric": "mse"})
    (run_dir / "metrics_best.json").write_text(json.dumps(record), encoding="utf-8")
    summary = summarize_experiment(tmp_path, "E0001")
    assert summary["mean"] == 0.25
    assert summary["preprocessing"]["window_length_tr"] == 83
    assert summary["preprocessing"]["n_dfc_windows_per_run"] == 224
    conclude_experiment(tmp_path, "E0001", "KEEP", "useful", "run L2")
    row = next(csv.DictReader(registry.open("r", encoding="utf-8-sig")))
    assert row["status"] == "KEEP"
    assert row["conclusion"] == "useful"


def test_summary_rejects_run_from_different_registered_config(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    registry = reports / "experiment_registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerow({"experiment_id": "E0001", "primary_metric": "mse", "config_sha256": "registered"})
    run = tmp_path / "outputs" / "E0001" / "runs" / "wrong-config"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "wrong-config", "status": "COMPLETED", "config_sha256": "different"}))
    with pytest.raises(RuntimeError, match="config_sha256"):
        summarize_experiment(tmp_path, "E0001")
