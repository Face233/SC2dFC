"""Metric invariants and train/val-only E0044 integration on synthetic data."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from scdfc.ae_diagnostic import (aggregate_rows, bootstrap_mean, command_ae_diagnostic,
    correlations, prepare_diagnostic, reconstruction_metrics, run_diagnostic)
from scdfc.cli import build_parser
from scdfc.config import load_config
from scdfc.connectivity import config_hash
from scdfc.management import file_sha256, manifest_digest, validate_experiment_config
from scdfc.models.autoencoder import FCAutoencoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/experiments/E0044_ae_reconstruction_diagnostic_v1.yaml"


def test_metric_identity_scaling_shift_and_decomposition():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 8)) + np.arange(8)
    identity = reconstruction_metrics(x, x)
    assert identity["mse"] == 0
    assert identity["dynamic_nmse"] == 0
    assert identity["time_r"] == pytest.approx(1)
    assert identity["difference_time_r"] == pytest.approx(1)
    assert identity["std_ratio_mean"] == pytest.approx(1)
    shift = reconstruction_metrics(x+3, x)
    assert shift["mean_bias_mse"] == pytest.approx(9)
    assert shift["dynamic_nmse"] == pytest.approx(0, abs=1e-25)
    assert shift["time_r"] == pytest.approx(1)
    centered = x-x.mean(0)
    y = x.mean(0)+.2*centered
    scaled = reconstruction_metrics(y, x)
    assert scaled["time_r"] == pytest.approx(1)
    assert scaled["std_ratio_mean"] == pytest.approx(.2)
    assert scaled["dynamic_nmse"] == pytest.approx(.64)
    negative = reconstruction_metrics(x.mean(0)-centered, x)
    assert negative["time_r"] == pytest.approx(-1)
    assert negative["std_ratio_mean"] == pytest.approx(1)
    assert negative["dynamic_nmse"] == pytest.approx(4)
    general = reconstruction_metrics(.3*x+2+rng.normal(size=x.shape), x)
    assert general["mse"] == pytest.approx(general["mean_bias_mse"]+general["centered_dynamic_mse"])


def test_static_short_constant_and_time_shuffle_rules():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 6))
    constant = np.broadcast_to(x.mean(0), x.shape)
    r = reconstruction_metrics(constant, x)
    assert r["time_r"] is None
    assert r["time_valid_fraction"] == 0
    assert r["std_ratio_mean"] == 0
    assert r["dynamic_nmse"] == pytest.approx(1)
    assert r["difference_nmse"] == pytest.approx(1)
    assert r["reconstructed_flat_fraction"] == 1
    one = reconstruction_metrics(x[:1], x[:1])
    assert one["mse"] == 0
    assert one["time_r"] is None and one["dynamic_nmse"] is None
    assert one["std_ratio_mean"] is None and one["difference_nmse"] is None
    target_constant = reconstruction_metrics(constant, constant)
    assert target_constant["dynamic_nmse"] is None
    assert target_constant["energy_std_ratio"] is None
    shuffled = reconstruction_metrics(x[rng.permutation(len(x))], x)
    assert shuffled["std_ratio_mean"] == pytest.approx(1)
    assert shuffled["time_r"] < .5
    assert shuffled["dynamic_nmse"] > 1
    assert shuffled["difference_nmse"] > 0


def test_flat_edge_coverage_and_common_edge_correlations():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 4))
    x[:, 0] = 7
    y = x.copy()
    y[:, 1] = 0
    r = reconstruction_metrics(y, x)
    assert r["true_valid_fraction"] == .75
    assert r["time_valid_fraction"] == .5
    assert r["reconstructed_flat_fraction"] == .5
    assert r["std_ratio_mean"] == pytest.approx(2/3)
    assert r["time_r"] == pytest.approx(1)
    _, a, _, _ = correlations(y, x)
    _, b, _, _ = correlations(x, x)
    assert (a & b).sum() == 2
    with pytest.raises(ValueError, match="finite"):
        reconstruction_metrics(np.full_like(x, np.nan), x)


def test_bootstrap_subject_weighting():
    # Repeating runs for one subject must not increase that subject's weight.
    base = reconstruction_metrics(np.ones((3, 2)), np.ones((3, 2)))
    rows = []
    for subject, values in (("a", [0, 0, 0]), ("b", [2])):
        for value in values:
            rows.append({"split": "val", "interval": "full", "method": "ae", "subject_id": subject,
                         **base, "mse": value})
    subjects, report = aggregate_rows(rows, 200, 42)
    assert len(subjects) == 2
    assert report["val"]["full"]["ae"]["mse"]["mean"] == 1
    assert bootstrap_mean([1, 2, 3], 200, 42) == bootstrap_mean([1, 2, 3], 200, 42)
    assert bootstrap_mean([None], 10)["mean"] is None


@pytest.fixture
def synthetic_diagnostic(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    config["paths"]["root"] = str(tmp_path)
    config["data"].update(n_nodes=4, n_timepoints=198)  # 24 windows at 83/5.
    config["model"]["fc_latent_dim"] = 4
    config["diagnostic"]["pca"]["components"] = 4
    config["diagnostic"]["expected_subjects"] = {"train": 3, "val": 4}
    config["diagnostic"]["intervals"].update(full=[0, 24], target=[20, 24])
    config["diagnostic"]["bootstrap_replicates"] = 100
    data = tmp_path/"data/manifests"
    data.mkdir(parents=True)
    table = pd.DataFrame({"subject_id": [f"s{i}" for i in range(8)], "split": ["train"]*3+["val"]*4+["test"]})
    split = data/"split_lr_v1.csv"
    table.to_csv(split, index=False)
    config["data"]["split_sha256"] = file_sha256(split)
    manifest = {"schema_version": 1, "dataset_version": "dataset_lr_v1", "preprocessing_version": "preprocess_lr_v1", "files": []}
    manifest["manifest_sha256"] = manifest_digest(manifest)
    (data/"dataset_lr_v1.json").write_text(json.dumps(manifest))
    config["data"]["manifest_sha256"] = manifest["manifest_sha256"]
    store = zarr.open_group(str(tmp_path/"data/cache/dfc/window_83.zarr"), mode="w")
    settings = {key: config["data"][key] for key in ("window_length", "stride", "n_nodes", "fisher_clip")}
    settings["estimator"] = "rectangular_pearson"
    store.attrs.update({**settings, "config_hash": config_hash(settings)})
    rng = np.random.default_rng(42)
    # Deliberately do not create test cache: the diagnostic must never request it.
    for i in range(7):
        group = store.require_group(f"subjects/s{i}/LR")
        group.create_dataset("fc_z", data=rng.normal(size=(24, 6)).astype(np.float32))
        group.create_dataset("window_starts", data=np.arange(24)*5)
    model = FCAutoencoder(n_edges=6, latent_dim=4, dropout=.1)
    artifact = tmp_path/"ae.pt"
    torch.save({"model": model.state_dict()}, artifact)
    config["artifacts"]["fc_autoencoder"].update(path="ae.pt", sha256=file_sha256(artifact))
    monkeypatch.setattr("scdfc.ae_diagnostic.git_metadata", lambda root: {"commit": "synthetic", "dirty": False})
    monkeypatch.setattr("scdfc.ae_diagnostic.environment_snapshot", lambda: "synthetic test environment\n")
    return config, store


def test_preflight_no_writes_and_integrity_failures(synthetic_diagnostic):
    config, store = synthetic_diagnostic
    _, samples, model, _, audit = prepare_diagnostic(config)
    assert len(samples["train"]) == 3 and len(samples["val"]) == 4
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert audit["test_data_read"] is False
    assert not (Path(config["paths"]["root"])/"outputs").exists()
    bad = deepcopy(config)
    bad["artifacts"]["fc_autoencoder"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="Artifact checksum"):
        prepare_diagnostic(bad)
    store.attrs["stride"] = 6
    with pytest.raises(ValueError, match="cache metadata"):
        prepare_diagnostic(config)


def test_protocol_validation_and_cli():
    config = load_config(CONFIG)
    validate_experiment_config(config)
    for path, value in (("batch_size", 0), ("run", "RL"), ("bootstrap_replicates", 0)):
        bad = deepcopy(config)
        bad["diagnostic"][path] = value
        with pytest.raises(ValueError):
            validate_experiment_config(bad)
    bad = deepcopy(config)
    bad["diagnostic"]["intervals"]["full"] = [0, 223]
    with pytest.raises(ValueError, match="Intervals"):
        validate_experiment_config(bad)
    args = build_parser().parse_args(["ae-diagnostic", "--check-only"])
    assert args.check_only and args.device == "cpu"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ae-diagnostic", "--split", "test"])
    from scdfc.managed_cli import command_run
    with pytest.raises(ValueError, match="ae-diagnostic"):
        command_run(SimpleNamespace(experiment=str(CONFIG)))


def test_check_only_and_failed_run_record(synthetic_diagnostic, monkeypatch, capsys):
    config, _ = synthetic_diagnostic
    monkeypatch.setattr("scdfc.ae_diagnostic.load_config", lambda _: config)
    command_ae_diagnostic(SimpleNamespace(experiment="synthetic", device="cpu", check_only=True))
    assert json.loads(capsys.readouterr().out)["status"] == "READY"
    root = Path(config["paths"]["root"])
    assert not (root/"outputs").exists()
    def fail(*args):
        raise RuntimeError("synthetic PCA failure")
    monkeypatch.setattr("scdfc.ae_diagnostic._fit_references", fail)
    with pytest.raises(RuntimeError, match="synthetic PCA failure"):
        run_diagnostic(config, "cpu", "failure-check")
    metadata = json.loads((root/"outputs/E0044/diagnostics/failure-check/metadata.json").read_text())
    assert metadata["status"] == "FAILED"
    assert "synthetic PCA failure" in metadata["error"]


def test_end_to_end_pca_train_only_artifacts_and_regeneration(synthetic_diagnostic, monkeypatch):
    config, store = synthetic_diagnostic
    old_hash = file_sha256(Path(config["paths"]["root"])/"ae.pt")
    fit_values = []
    from scdfc.ae_diagnostic import PCA
    original = PCA.fit
    def inspect_fit(self, x, y=None):
        fit_values.append(np.array(x, copy=True))
        return original(self, x, y)
    monkeypatch.setattr(PCA, "fit", inspect_fit)
    output = run_diagnostic(config, "cpu", "synthetic-check")
    expected = np.concatenate([np.asarray(store[f"subjects/s{i}/LR/fc_z"]) for i in range(3)])
    np.testing.assert_array_equal(fit_values[0], expected)
    with np.load(output/"pca.npz") as pca:
        np.testing.assert_allclose(pca["mean"], expected.mean(0), rtol=1e-5, atol=1e-6)
    audit = json.loads((output/"data_audit.json").read_text())
    assert len(audit["cache_content"]["val"]) == 4
    assert audit["test_data_read"] is False
    metadata = json.loads((output/"metadata.json").read_text())
    assert metadata["status"] == "COMPLETED" and metadata["training_run_created"] is False
    assert len(metadata["images"]) == 6 and all(Path(p).is_file() for p in metadata["images"])
    assert not list(output.glob("pca_train_*"))
    assert file_sha256(Path(config["paths"]["root"])/"ae.pt") == old_hash
    report = json.loads((output/"summary.json").read_text())
    assert report["summary"]["val"]["full"]["identity"]["dynamic_nmse"]["mean"] == 0
    assert report["summary"]["val"]["full"]["oracle_mean"]["dynamic_nmse"]["mean"] == pytest.approx(1)
    assert report["summary"]["val"]["history1"]["ae"]["time_r"]["mean"] is None
    assert report["paired_ae_minus_pca"]["full"]["common_time_r_difference"]["n_subjects"] == 4
    assert not (output.parent.parent/"runs").exists()
    with pytest.raises(FileExistsError):
        run_diagnostic(config, "cpu", "synthetic-check")
    with pytest.raises(ValueError, match="diagnostic-id"):
        run_diagnostic(config, "cpu", "../escape")
    import runpy
    scripts = list((output.parent.parent/"visual").glob("*_rebuild.py"))
    runpy.run_path(str(scripts[0]))
    assert all(Path(p).stat().st_size > 1000 for p in metadata["images"])
