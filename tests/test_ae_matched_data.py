"""Matched-data AE audit and synthetic preparation/training regression tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from test_history_forecast import history_config
from scdfc.ae_full_data import full_ae_datasets
from scdfc.config import load_config
from scdfc.data import read_cached
from scdfc.management import file_sha256, validate_experiment_config
from scdfc.managed_cli import command_run
from scdfc.training import AutoencoderLoss

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def matched_config(history_config, tmp_path):
    config = load_config(ROOT/"configs/experiments/E0048_ae_fixed32_mse_matched_v1.yaml")
    config["paths"] = deepcopy(history_config["paths"])
    config["data"] = {key: value for key, value in history_config["data"].items()
                      if key not in {"forecast_origins", "forecast_windows"}}
    config["model"].update(hidden_dim=4, fc_latent_dim=4, dropout=0)
    config["experiment"]["level"] = 0
    config["training"].update(autoencoder_epochs=1, patience=2, autoencoder_batch_size=128,
                               autoencoder_expected_subjects={"train": 3, "val": 2},
                               autoencoder_step_checkpoints=[1])
    pca = tmp_path/"reference_pca.npz"
    np.savez(pca, components=np.zeros((4, 6)))
    reference = {"data_manifest_digest": config["data"]["manifest_sha256"],
                 "split_sha256": config["data"]["split_sha256"],
                 "shape_per_run": [224, 6], "samples": {}, "cache_content": {}}
    import hashlib
    for fold, subjects in (("train", ["100001", "100002", "100003"]), ("val", ["100004", "100005"])):
        reference["samples"][fold] = [[s, "LR"] for s in subjects]
        reference["cache_content"][fold] = []
        for subject in subjects:
            values, starts = read_cached(config, 83, subject, "LR")
            digest = hashlib.sha256(values.tobytes(order="C")+starts.astype("<i8").tobytes()).hexdigest()
            reference["cache_content"][fold].append({"subject_id": subject, "run": "LR", "sha256": digest})
    audit = tmp_path/"reference_audit.json"
    audit.write_text(json.dumps(reference))
    config["artifacts"] = {
        "pca_reference": {"path": str(pca), "sha256": file_sha256(pca)},
        "pca_data_audit": {"path": str(audit), "sha256": file_sha256(audit)},
    }
    return config


def test_archived_scope_reconstruction_and_historical_loss():
    spec = importlib.util.spec_from_file_location("ae_scope_audit", ROOT/"scripts/audit_ae_pca_training_scope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(ROOT)
    assert report["status"] == "ARCHIVED_EVIDENCE_VERIFIED"
    assert report["split_counts"] == {"train": 738, "val": 158, "test": 159}
    assert report["train_val_subject_intersection"] == 0
    assert report["source_checks"]["edge_loss_was_smooth_l1"]
    assert report["conclusions"]["strict_runtime_certification"] is False
    assert report["sampling_lr_only_reconstruction"]["unique_windows"] == 22012


def test_official_pair_only_changes_sampling_and_all_use_mse():
    configs = [validate_experiment_config(load_config(next((ROOT/"configs/experiments").glob(f"E00{i}_ae*.yaml"))))
               for i in (48, 49)]
    assert configs[0]["model"] == configs[1]["model"]
    assert configs[0]["artifacts"] == configs[1]["artifacts"]
    assert configs[0]["data"] == configs[1]["data"]
    for config in configs:
        assert config["training"]["autoencoder_loss_type"] == "mse"
        assert config["training"]["autoencoder_epochs"] == 100
        assert config["training"]["autoencoder_step_checkpoints"] == [9300]
    left, right = [dict(config["training"]) for config in configs]
    assert left.pop("autoencoder_train_sampling") == "fixed32"
    assert right.pop("autoencoder_train_sampling") == "all"
    assert left == right
    bad = deepcopy(configs[0])
    bad["training"]["autoencoder_loss_type"] = "smooth_l1"
    with pytest.raises(ValueError, match="MSE"):
        validate_experiment_config(bad)
    value = torch.tensor([[1., -2.]])
    loss, _ = AutoencoderLoss({"edge": 1})(value, torch.zeros_like(value))
    assert loss.item() == pytest.approx(2.5)


def test_exact_windows_no_sc_test_reads_and_reference_drift(matched_config, tmp_path, monkeypatch):
    import scdfc.ae_full_data as module
    original_open = module.zarr.open_group
    class RestrictedStore:
        def __init__(self, real):
            self.real, self.attrs = real, real.attrs
        def __getitem__(self, key):
            assert "100006" not in key, "test FC was accessed"
            assert not key.endswith("/RL"), "RL FC was accessed"
            return self.real[key]
    monkeypatch.setattr(module.zarr, "open_group", lambda *args, **kwargs: RestrictedStore(original_open(*args, **kwargs)))
    monkeypatch.setattr("scdfc.data.load_sc", lambda *args: pytest.fail("SC read in matched AE"))
    manifests = []
    for sampling in ("fixed32", "all"):
        config = deepcopy(matched_config)
        config["training"]["autoencoder_train_sampling"] = sampling
        with full_ae_datasets(config, 83) as (train, val, audit):
            assert len(train) == (3*32 if sampling == "fixed32" else 3*224)
            assert len(val) == 2*224
            assert audit["test_fc_read"] is False
            assert audit["training_sampling"] == sampling
            source, _ = read_cached(config, 83, "100001", "LR")
            first = int(np.random.default_rng(42).integers(224)) if sampling == "fixed32" else 0
            np.testing.assert_array_equal(train[0], source[first])
            if sampling == "all":
                np.testing.assert_array_equal(train[223], source[223])
            manifests.append(audit["training_window_manifest_sha256"])
        assert not list((tmp_path/"data/cache/private_ae_windows").glob("full_ae_*"))
    assert manifests[0] == manifests[1]
    # Keep an intact reference hash while changing the input FC: fail closed.
    store = original_open(str(tmp_path/"data/cache/window_83.zarr"), mode="a")
    store["subjects/100001/LR/fc_z"][0, 0] += .1
    with pytest.raises(ValueError, match="content differs"):
        with full_ae_datasets(matched_config, 83):
            pytest.fail("Unexpected success")
    assert not list((tmp_path/"data/cache/private_ae_windows").glob("full_ae_*"))


def test_managed_synthetic_pair_checkpoints_and_provenance(matched_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scdfc.management.git_metadata", lambda root: {
        "commit": "a"*40, "branch": "main", "remote": "", "dirty": False, "status": []})
    monkeypatch.setattr("scdfc.managed_cli._ensure_stats", lambda *args: pytest.fail("Matched AE must not fit old SC-dependent stats"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    initial = []
    try:
        for number, sampling in ((48, "fixed32"), (49, "all")):
            config = deepcopy(matched_config)
            config["experiment"].update(id=f"E00{number}", artifact_id=f"A00{number}")
            config["training"]["autoencoder_train_sampling"] = sampling
            path = tmp_path/f"configs/experiments/E00{number}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command_run(SimpleNamespace(experiment=str(path), seed=42, device="cpu"))
            run = next((tmp_path/f"outputs/E00{number}/runs").iterdir())
            assert json.loads((run/"metadata.json").read_text())["status"] == "COMPLETED"
            protocol = json.loads((run/"ae_training_protocol.json").read_text())
            assert protocol["train_windows"] == (96 if sampling == "fixed32" else 672)
            assert protocol["validation_windows"] == 448 and protocol["loss_type"] == "mse"
            assert protocol["saved_step_checkpoints"] == [1]
            initial.append(protocol["initial_parameters_sha256"])
            for name in ("best.pt", "last.pt", "step_1.pt"):
                payload = torch.load(run/"checkpoints"/name, map_location="cpu", weights_only=False)
                assert payload["autoencoder_loss_type"] == "mse"
                assert payload["training_window_manifest_sha256"]
                assert payload["ae_data_audit_sha256"] == file_sha256(run/"ae_data_audit.json")
            assert (tmp_path/f"configs/artifacts/A00{number}.yaml").is_file()
        assert initial[0] == initial[1]
    finally:
        torch.set_num_threads(old_threads)
