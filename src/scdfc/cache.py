from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .audit import discover_data
from .config import resolve_path
from .connectivity import config_hash, expected_windows, sliding_window_fc


def _zarr():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError("Zarr caching requires `pip install zarr<3 numcodecs`") from exc
    return zarr, Blosc


def cache_path(config: dict[str, Any], window_length: int) -> Path:
    return resolve_path(config, "cache_dir") / f"window_{window_length}.zarr"


def precompute_dfc(
    config: dict[str, Any],
    window_length: int,
    subjects: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    zarr, Blosc = _zarr()
    found = discover_data(config)
    stride = int(config["data"]["stride"])
    n_nodes = int(config["data"]["n_nodes"])
    fisher_clip = float(config["data"]["fisher_clip"])
    destination = cache_path(config, window_length)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(destination), mode="a")
    settings = {
        "window_length": window_length,
        "stride": stride,
        "n_nodes": n_nodes,
        "fisher_clip": fisher_clip,
        "estimator": "rectangular_pearson",
    }
    settings_hash = config_hash(settings)
    previous_hash = root.attrs.get("config_hash")
    if previous_hash and previous_hash != settings_hash and not overwrite:
        raise ValueError("Existing cache was generated with different settings; use --overwrite")
    root.attrs.update({**settings, "config_hash": settings_hash})
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    written = skipped = 0
    for run, paths in found["runs"].items():
        for subject, path in sorted(paths.items()):
            if subject not in found["sc"] or (subjects is not None and subject not in subjects):
                continue
            key = f"subjects/{subject}/{run}"
            if key in root and not overwrite:
                skipped += 1
                continue
            frame = pd.read_csv(path)
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
            fc_z, starts = sliding_window_fc(values, window_length, stride, fisher_clip)
            if key in root:
                del root[key]
            group = root.require_group(key)
            group.create_dataset("fc_z", data=fc_z, chunks=(min(32, len(fc_z)), fc_z.shape[1]), compressor=compressor)
            group.create_dataset("window_starts", data=starts, chunks=(len(starts),), compressor=compressor)
            group.attrs.update({"subject_id": subject, "run": run, "source": str(path.resolve())})
            written += 1
    return {"written": written, "skipped": skipped}


def iter_cached_samples(config: dict[str, Any], window_length: int) -> Iterator[tuple[str, str]]:
    zarr, _ = _zarr()
    root = zarr.open_group(str(cache_path(config, window_length)), mode="r")
    if "subjects" not in root:
        return
    for subject in sorted(root["subjects"].group_keys()):
        for run in sorted(root[f"subjects/{subject}"].group_keys()):
            yield subject, run


def read_cached(config: dict[str, Any], window_length: int, subject: str, run: str) -> tuple[np.ndarray, np.ndarray]:
    zarr, _ = _zarr()
    root = zarr.open_group(str(cache_path(config, window_length)), mode="r")
    group = root[f"subjects/{subject}/{run}"]
    return np.asarray(group["fc_z"]), np.asarray(group["window_starts"])

