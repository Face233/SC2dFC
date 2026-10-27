"""Frozen FC PCA must preserve the fitted transform and fail on binding drift."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
import torch
import yaml

from scdfc.fc_pca import export_frozen_fc_pca, load_frozen_fc_pca
from scdfc.management import file_sha256, manifest_digest
from scdfc.models import FrozenFCPCA


@pytest.fixture
def archived_pca(tmp_path):
    source = tmp_path / "outputs/E0044/diagnostics/synthetic"
    source.mkdir(parents=True)
    values = np.random.default_rng(7).normal(size=(12, 6)).astype(np.float32)
    fitted = PCA(n_components=2, svd_solver="randomized", random_state=42,
                 n_oversamples=2, iterated_power=4, power_iteration_normalizer="QR").fit(values)
    np.savez(source/"pca.npz", components=fitted.components_, mean=fitted.mean_,
             explained_variance=fitted.explained_variance_, singular_values=fitted.singular_values_)
    private = tmp_path/"data"
    private.mkdir()
    split = private/"split.csv"
    split.write_text("subject_id,split\n1000,train\n1001,train\n1002,train\n1003,train\n2000,val\n")
    dataset = {"dataset_version": "synthetic", "preprocessing_version": "fixture", "files": []}
    (private/"manifest.json").write_text(json.dumps(dataset))
    config = {"paths": {"root": str(tmp_path), "timeseries": {"LR": "data/LR"},
                        "split_csv": "data/split.csv"}, "model": {"fc_latent_dim": 2},
              "data": {"dataset_version": "synthetic", "preprocessing_version": "fixture",
                       "manifest_path": "data/manifest.json", "manifest_sha256": manifest_digest(dataset),
                       "split_version": "fixture", "split_sha256": file_sha256(split),
                       "n_nodes": 4, "n_timepoints": 11, "window_length": 3, "stride": 4,
                       "fisher_clip": .999999},
              "diagnostic": {"seed": 42, "expected_subjects": {"train": 4, "val": 1},
                             "pca": {"components": 2, "n_oversamples": 2,
                                     "iterated_power": 4, "power_iteration_normalizer": "QR"}}}
    (source/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    runs = [{"subject_id": str(1000+i), "run": "LR", "sha256": hashlib.sha256(
        values[i*3:(i+1)*3].tobytes()+np.arange(0, 12, 4, dtype="<i8").tobytes()).hexdigest()}
            for i in range(4)]
    audit = {"samples": {"train": [[r["subject_id"], "LR"] for r in runs], "val": [["2000", "LR"]]},
             "cache_content": {"train": runs}, "shape_per_run": [3, 6], "test_data_read": False,
             "data_manifest_digest": config["data"]["manifest_sha256"],
             "split_sha256": config["data"]["split_sha256"]}
    (source/"data_audit.json").write_text(json.dumps(audit))
    metadata = {"status": "COMPLETED", "experiment_id": "E0044", "diagnostic_id": "synthetic",
                "git": {"commit": "a"*40, "dirty": True},
                "output_sha256": {name: file_sha256(source/name) for name in
                                  ("pca.npz", "data_audit.json", "config_resolved.yaml")}}
    (source/"metadata.json").write_text(json.dumps(metadata))
    return tmp_path, source, fitted, config


def test_frozen_component_matches_sklearn_and_roundtrips_state(archived_pca):
    root, source, fitted, config = archived_pca
    manifest = export_frozen_fc_pca(source, root)
    model = load_frozen_fc_pca(manifest, config=config)
    x = np.random.default_rng(9).normal(size=(2, 3, 6)).astype(np.float32)
    reconstructed, latent = model(torch.from_numpy(x))
    expected = fitted.transform(x.reshape(-1, 6))
    np.testing.assert_allclose(latent.numpy().reshape(-1, 2), expected, atol=1e-6)
    np.testing.assert_allclose(reconstructed.numpy().reshape(-1, 6),
                               fitted.inverse_transform(expected), atol=1e-6)
    assert not list(model.parameters())
    assert not any(buffer.requires_grad for buffer in model.buffers())
    restored = FrozenFCPCA(torch.zeros(2, 6), torch.zeros(6))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(torch.from_numpy(x))[0], reconstructed)
    assert restored.double().encode(torch.from_numpy(x).double()).dtype == torch.float64
    # Frozen coefficients still allow gradients through the projection to an upstream input.
    input_with_grad = torch.from_numpy(x).requires_grad_()
    model.encode(input_with_grad).square().sum().backward()
    assert input_with_grad.grad is not None and torch.isfinite(input_with_grad.grad).all()


def test_export_is_byte_identical_self_contained_and_immutable(archived_pca):
    root, source, _, _ = archived_pca
    manifest_path = export_frozen_fc_pca(source, root)
    manifest = yaml.safe_load(manifest_path.read_text())
    assert (root/manifest["path"]).read_bytes() == (source/"pca.npz").read_bytes()
    assert manifest["training"]["windows"] == 12
    assert manifest["source"]["refitted_on_export"] is False
    assert "subject_id" not in manifest_path.read_text()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_frozen_fc_pca(source, root)
    (source/"pca.npz").unlink()
    assert load_frozen_fc_pca(manifest_path).latent_dim == 2


def test_archive_corruption_and_data_binding_drift_are_rejected(archived_pca):
    root, source, _, config = archived_pca
    manifest_path = export_frozen_fc_pca(source, root)
    config["data"]["stride"] = 5
    with pytest.raises(ValueError, match="differs from config"):
        load_frozen_fc_pca(manifest_path, config=config)
    config["data"]["stride"] = 4
    (root/"data/split.csv").write_text("changed private split")
    with pytest.raises(ValueError, match="Split checksum"):
        load_frozen_fc_pca(manifest_path, config=config)
    archive = root/yaml.safe_load(manifest_path.read_text())["path"]
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        load_frozen_fc_pca(manifest_path)


def test_export_checks_archived_source_before_creating_component(archived_pca):
    root, source, _, _ = archived_pca
    with (source/"pca.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="source checksum"):
        export_frozen_fc_pca(source, root)
    assert not (root/"outputs/components/A0044").exists()


def test_component_rejects_invalid_shapes_and_nonfinite_parameters():
    with pytest.raises(ValueError, match="finite components"):
        FrozenFCPCA([[float("nan"), 0]], [0, 0])
    model = FrozenFCPCA([[1, 0]], [0, 0])
    with pytest.raises(ValueError, match="input"):
        model.encode(torch.zeros(3))
    with pytest.raises(ValueError, match="latent"):
        model.decode(torch.zeros(2))
