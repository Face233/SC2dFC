"""Strict train/val-only, full-window AE datasets matched to archived PCA data."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from .config import resolve_path
from .connectivity import config_hash
from .data import cache_path, load_split
from .management import file_sha256, verify_data_bindings


class FullFCWindowDataset(Dataset):
    """One FC per slot; optional fixed indices retain sampled duplicates."""
    def __init__(self, matrix, indices=None):
        self.matrix = matrix
        self.indices = indices

    def __len__(self):
        return len(self.matrix) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        # A private copy prevents torch operations from modifying the mmap.
        row = index if self.indices is None else self.indices[index]
        return torch.from_numpy(np.array(self.matrix[row], dtype=np.float32, copy=True))


def _reference(config, name):
    reference = config["artifacts"][name]
    path = Path(reference["path"])
    if not path.is_absolute():
        path = Path(config["paths"]["root"])/path
    if file_sha256(path) != reference["sha256"]:
        raise ValueError(f"AE reference checksum mismatch: {name}")
    return path


@contextmanager
def full_ae_datasets(config, window_length):
    """Materialize private mmaps once; never enumerate/read test FC entries.

    Digest checks make the train FC *values*, not only the subject counts,
    identical to those used for the archived E0044 PCA fit.
    """
    verify_data_bindings(config)
    audit_path = _reference(config, "pca_data_audit")
    pca_path = _reference(config, "pca_reference")
    reference = json.loads(audit_path.read_text(encoding="utf-8"))
    data = config["data"]
    if (reference["data_manifest_digest"] != data["manifest_sha256"]
            or reference["split_sha256"] != data["split_sha256"]):
        raise ValueError("AE data/split differ from the PCA reference audit")
    if set(config["paths"]["timeseries"]) != {"LR"}:
        raise ValueError("Full-window AE requires LR-only input")
    store = zarr.open_group(str(cache_path(config, window_length)), mode="r")
    settings = {"window_length": window_length, **{key: data[key] for key in ("stride", "n_nodes", "fisher_clip")},
                "estimator": "rectangular_pearson"}
    if dict(store.attrs) != {**settings, "config_hash": config_hash(settings)}:
        raise ValueError("AE FC cache preprocessing metadata mismatch")
    windows = (data["n_timepoints"]-window_length)//data["stride"]+1
    edges = data["n_nodes"]*(data["n_nodes"]-1)//2
    if reference["shape_per_run"] != [windows, edges]:
        raise ValueError("AE/PCA window or edge dimensions differ")
    table = load_split(resolve_path(config, "split_csv"))
    participants = {fold: sorted(table.loc[table.split == fold, "subject_id"].astype(str))
                    for fold in ("train", "val")}
    if set(participants["train"]) & set(participants["val"]):
        raise ValueError("Train/validation participants overlap")
    for fold, subjects in participants.items():
        expected = config["training"]["autoencoder_expected_subjects"][fold]
        if len(subjects) != expected or [[s, "LR"] for s in subjects] != reference["samples"][fold]:
            raise ValueError(f"AE {fold} participants differ from PCA fit/evaluation")
    workspace = resolve_path(config, "cache_dir")/"private_ae_windows"
    workspace.mkdir(parents=True, exist_ok=True)
    audit = {"mode": "all_lr_windows", "test_fc_read": False,
             "data_manifest_sha256": data["manifest_sha256"], "split_sha256": data["split_sha256"],
             "pca_reference_sha256": file_sha256(pca_path), "pca_audit_sha256": file_sha256(audit_path),
             "shape_per_run": [windows, edges], "splits": {},
             "source_enumeration": "sorted subject, LR, every window index from 0; gradient indices recorded separately"}
    matrices = []
    try:
        with TemporaryDirectory(prefix="full_ae_", dir=workspace) as temporary:
            if not Path(temporary).resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Private AE temporary windows must stay inside the configured cache directory")
            try:
                train_hashes = set()
                for fold, subjects in participants.items():
                    matrix = np.lib.format.open_memmap(Path(temporary)/f"{fold}.npy", mode="w+",
                                                       dtype=np.float32, shape=(len(subjects)*windows, edges))
                    matrices.append(matrix)
                    expected_hashes = {(r["subject_id"], r["run"]): r["sha256"]
                                       for r in reference["cache_content"][fold]}
                    actual = []
                    seen = set()
                    for index, subject in enumerate(subjects):
                        group = store[f"subjects/{subject}/LR"]
                        values = np.asarray(group["fc_z"], dtype=np.float32)
                        starts = np.asarray(group["window_starts"])
                        if (values.shape != (windows, edges) or not np.isfinite(values).all()
                                or not np.array_equal(starts, np.arange(windows)*data["stride"])):
                            raise ValueError(f"Invalid AE FC cache: {subject}/LR")
                        digest = hashlib.sha256(values.tobytes(order="C")+starts.astype("<i8").tobytes()).hexdigest()
                        if digest != expected_hashes.get((subject, "LR")):
                            raise ValueError(f"AE/PCA FC cache content differs: {subject}/LR")
                        if fold == "val" and digest in train_hashes:
                            raise ValueError("Identical whole FC run appears across train/validation")
                        seen.add(digest)
                        matrix[index*windows:(index+1)*windows] = values
                        actual.append({"subject_id": subject, "run": "LR", "sha256": digest})
                    if fold == "train":
                        train_hashes = seen
                    matrix.flush()
                    audit["splits"][fold] = {"subjects": len(subjects), "windows": len(matrix),
                                            "cache_content": actual}
                audit["training_window_manifest_sha256"] = hashlib.sha256(json.dumps({
                    "runs": audit["splits"]["train"]["cache_content"], "window_indices": list(range(windows))
                }, sort_keys=True).encode()).hexdigest()
                indices = None
                sampling = config["training"]["autoencoder_train_sampling"]
                if sampling == "fixed32":
                    indices = np.asarray([(i//32)*windows + int(np.random.default_rng(int(config["seed"])+i).integers(windows))
                                          for i in range(len(participants["train"])*32)])
                elif sampling != "all":
                    raise ValueError("AE sampling must be fixed32 or all")
                gradient_indices = indices if indices is not None else np.arange(len(matrices[0]))
                audit.update(training_sampling=sampling, gradient_update_slots_per_epoch=len(gradient_indices),
                             unique_gradient_windows=int(len(np.unique(gradient_indices))),
                             gradient_window_indices_sha256=hashlib.sha256(gradient_indices.astype("<i8").tobytes()).hexdigest())
                yield FullFCWindowDataset(matrices[0], indices), FullFCWindowDataset(matrices[1]), audit
            finally:
                for matrix in matrices:
                    matrix._mmap.close()
    finally:
        matrices.clear()
