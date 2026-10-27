"""Export and load a standalone frozen component from an archived PCA fit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import yaml

from .management import file_sha256, utc_now, verify_data_bindings
from .models.pca import FrozenFCPCA


DATA_KEYS = ("dataset_version", "preprocessing_version", "manifest_sha256", "split_version",
             "split_sha256", "n_nodes", "n_timepoints", "window_length", "stride", "fisher_clip")


def verified_pca_paths(config):
    """Bind the config to both immutable manifest bytes and parameter bytes."""
    reference = config["artifacts"]["fc_pca"]
    root = Path(config["paths"]["root"])
    manifest_path = root / reference["manifest_path"]
    archive_path = root / reference["path"]
    if file_sha256(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("PCA manifest checksum mismatch")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if (manifest["id"] != reference["id"] or manifest["sha256"] != reference["sha256"]
            or (root / manifest["path"]).resolve() != archive_path.resolve()):
        raise ValueError("PCA manifest identity/path/hash differs from experiment binding")
    return manifest_path, archive_path


def load_frozen_fc_pca(manifest_path, *, config=None, device="cpu") -> FrozenFCPCA:
    """Load an artifact from configs/artifacts; optionally enforce a data binding.

    Without a config, only the manifest and its parameter archive are required.
    With a config, its root resolves all paths and private data metadata is verified.
    """
    path = Path(manifest_path)
    if not path.is_absolute() and config is not None:
        path = Path(config["paths"]["root"]) / path
    path = path.resolve()
    root = Path(config["paths"]["root"]) if config is not None else path.parents[2]
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("type") != "fc_pca"
            or manifest.get("format") != "e0044_pca_npz/v1"
            or manifest.get("whiten") is not False
            or manifest.get("representation") != "fisher_z_upper_triangle_k1"):
        raise ValueError("Expected a frozen, unwhitened Fisher-z FC PCA artifact")
    if config is not None:
        if (any(config["data"].get(k) != manifest["data"].get(k) for k in DATA_KEYS)
                or set(config["paths"]["timeseries"]) != {"LR"}
                or config["model"]["fc_latent_dim"] != manifest["latent_dim"]):
            raise ValueError("PCA data/preprocessing/split or latent dimension differs from config")
        verify_data_bindings(config)
    archive_path = Path(manifest["path"])
    if not archive_path.is_absolute():
        archive_path = root / archive_path
    if file_sha256(archive_path) != manifest["sha256"]:
        raise ValueError("PCA parameter archive checksum mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        model = FrozenFCPCA(archive["components"], archive["mean"])
    if (model.latent_dim != manifest["latent_dim"] or model.n_edges != manifest["input_dim"]
            or model.n_edges != manifest["data"]["n_nodes"] * (manifest["data"]["n_nodes"]-1)//2):
        raise ValueError("PCA parameter dimensions differ from artifact manifest")
    return model.to(device).eval()


def export_frozen_fc_pca(diagnostic_dir, root, artifact_id="A0044") -> Path:
    """Copy the original fitted bytes; never fit PCA or read raw train/val/test FC."""
    if not re.fullmatch(r"A[0-9]{4}", artifact_id):
        raise ValueError("PCA artifact ID must have the form A####")
    root = Path(root).resolve()
    source = Path(diagnostic_dir)
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    source_relative = source.relative_to(root).as_posix()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED":
        raise ValueError("Only a completed PCA diagnostic can be exported")
    for name in ("pca.npz", "data_audit.json", "config_resolved.yaml"):
        if file_sha256(source / name) != metadata["output_sha256"][name]:
            raise ValueError(f"Archived PCA source checksum mismatch: {name}")
    config = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    audit = json.loads((source / "data_audit.json").read_text(encoding="utf-8"))
    data = config["data"]
    windows = (data["n_timepoints"]-data["window_length"])//data["stride"]+1
    edges = data["n_nodes"]*(data["n_nodes"]-1)//2
    train = audit["samples"]["train"]
    if (set(config["paths"]["timeseries"]) != {"LR"}
            or any(run != "LR" for _, run in train)
            or len({s for s, _ in train}) != len(train)
            or len(train) != config["diagnostic"]["expected_subjects"]["train"]
            or audit["shape_per_run"] != [windows, edges]
            or audit["data_manifest_digest"] != data["manifest_sha256"]
            or audit["split_sha256"] != data["split_sha256"]
            or config["diagnostic"]["pca"].get("whiten", False) is not False
            or audit["test_data_read"] is not False):
        raise ValueError("Archived PCA data scope does not match the declared protocol")
    with np.load(source / "pca.npz", allow_pickle=False) as archive:
        model = FrozenFCPCA(archive["components"], archive["mean"])
    if model.n_edges != edges or model.latent_dim != config["diagnostic"]["pca"]["components"]:
        raise ValueError("Archived PCA dimensions differ from the declared protocol")
    destination = root / "outputs" / "components" / artifact_id / "pca.npz"
    manifest_path = root / "configs" / "artifacts" / f"{artifact_id}.yaml"
    if destination.parent.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite PCA artifact {artifact_id}")
    manifest = {
        "schema_version": 1, "id": artifact_id, "type": "fc_pca", "format": "e0044_pca_npz/v1",
        "path": destination.relative_to(root).as_posix(), "sha256": file_sha256(source / "pca.npz"),
        "created_at": utc_now(), "input_dim": edges, "latent_dim": model.latent_dim,
        "representation": "fisher_z_upper_triangle_k1", "whiten": False,
        "data": {key: data[key] for key in DATA_KEYS},
        "training": {"split": "train", "run": "LR", "subjects": len(train),
                     "windows_per_subject": windows, "windows": len(train)*windows,
                     "per_edge_standardization": False,
                     "training_window_manifest_sha256": hashlib.sha256(json.dumps({
                         "runs": audit["cache_content"]["train"], "window_indices": list(range(windows))
                     }, sort_keys=True).encode()).hexdigest()},
        "fit": {"seed": config["diagnostic"]["seed"], "svd_solver": "randomized",
                **config["diagnostic"]["pca"]},
        "source": {"experiment_id": metadata["experiment_id"], "diagnostic_id": metadata["diagnostic_id"],
                   "directory": source_relative, "metadata_sha256": file_sha256(source / "metadata.json"),
                   "data_audit_sha256": file_sha256(source / "data_audit.json"),
                   "config_file_sha256": file_sha256(source / "config_resolved.yaml"),
                   "git_commit": metadata["git"]["commit"], "git_dirty": metadata["git"]["dirty"],
                   "refitted_on_export": False},
    }
    destination.parent.mkdir(parents=True, exist_ok=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_created = manifest_created = False
    try:
        with destination.open("xb") as handle:
            archive_created = True
            with (source / "pca.npz").open("rb") as original:
                shutil.copyfileobj(original, handle)
        if file_sha256(destination) != manifest["sha256"]:
            raise ValueError("Exported PCA archive checksum mismatch")
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            manifest_created = True
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
    except Exception:
        if manifest_created:
            manifest_path.unlink()
        if archive_created:
            destination.unlink()
        destination.parent.rmdir()
        raise
    return manifest_path
