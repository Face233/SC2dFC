"""Training-only, common-origin persistence artifacts for history Transformers.

Preparation is separate from neural training. A draft config may have a null
artifact hash; run/evaluation always require a bound, verified artifact.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import resolve_path
from .data import iter_cached_samples, load_split, read_cached
from .management import file_sha256


MODEL_NAME = "fc_history_gru_transformer_decay_residual"
HISTORY_MODELS = {"fc_history_gru_query_residual", MODEL_NAME}
DECOMPOSITION = "history_decay_residual"


def persistence_protocol(config):
    data = {k: v for k, v in config["data"].items() if k != "warmup_windows"}
    return {"schema_version": 1, "data": data,
            "runs": sorted(config["paths"]["timeseries"]),
            "method": "train_subject_edge_least_squares_clip_0_1/v1",
            "anchor": "last_observed_fc", "monotonic": False}


def persistence_path(config):
    reference = config.get("artifacts", {}).get("history_persistence", {})
    if not reference.get("sha256"):
        raise ValueError("Persistence artifact is not bound. Run fit-history-persistence, then bind-history-persistence before training.")
    path = Path(reference["path"])
    return path if path.is_absolute() else Path(config["paths"]["root"]) / path


def load_persistence_artifact(config):
    path = persistence_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Persistence artifact not found: {path}")
    expected = config["artifacts"]["history_persistence"]["sha256"]
    if file_sha256(path) != expected:
        raise ValueError("Persistence artifact checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if json.loads(str(arrays["protocol_json"])) != persistence_protocol(config):
        raise ValueError("Persistence artifact protocol differs from config (origin/data/split/runs)")
    data = config["data"]
    origin, steps = data["forecast_origins"][0], data["forecast_windows"]
    edges = data["n_nodes"] * (data["n_nodes"] - 1) // 2
    shapes = {"alpha": (steps,), "alpha_raw": (steps,), "context_template": (edges,),
              "group_template": (origin + steps, edges)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid persistence artifact {key} shape/values")
    if ((arrays["alpha"] < 0) | (arrays["alpha"] > 1)).any():
        raise ValueError("Persistence alpha must be in [0,1]")
    if not np.array_equal(arrays["alpha"], np.clip(arrays["alpha_raw"], 0, 1).astype(np.float32)):
        raise ValueError("Persistence alpha differs from clipped training fit")
    return arrays


def fit_history_persistence(config, output):
    """Fit one frozen shared baseline, reading only training LR FC caches."""
    if config["model"]["name"] != MODEL_NAME:
        raise ValueError("Persistence fitting requires a history Transformer config")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite persistence artifact: {output}")
    data = config["data"]
    if set(config["paths"]["timeseries"]) != {"LR"}:
        raise ValueError("This protocol requires LR-only data")
    window = int(data["window_length"])
    origin, steps = data["forecast_origins"][0], data["forecast_windows"]
    windows = (data["n_timepoints"] - window) // data["stride"] + 1
    edges = data["n_nodes"] * (data["n_nodes"] - 1) // 2
    if steps != windows-origin-1:
        raise ValueError("Persistence H must cover all remaining windows")
    split = load_split(resolve_path(config, "split_csv"))
    subjects = set(split.loc[split.split == "train", "subject_id"].astype(str))
    samples = [(s, r) for s, r in iter_cached_samples(config, window) if s in subjects]
    counts = Counter(s for s, _ in samples)
    if not subjects or set(counts) != subjects or any(counts[s] != 1 for s in subjects):
        raise ValueError("Persistence fitting requires exactly one complete LR run per training subject")
    total = np.zeros((windows, edges), dtype=np.float64)
    cache_digest = hashlib.sha256()
    def read(subject, run):
        fc, starts = read_cached(config, window, subject, run)
        if (fc.shape != total.shape or not np.isfinite(fc).all()
                or not np.array_equal(starts, np.arange(windows)*data["stride"])):
            raise ValueError(f"Invalid FC cache shape/values/window starts for {subject}/{run}")
        return fc.astype(np.float64)
    for subject, run in samples:
        fc = read(subject, run)
        total += fc
        cache_digest.update(json.dumps([subject, run]).encode())
        cache_digest.update(fc.astype("<f8").tobytes())
    mean = total / len(samples)
    numerator = np.zeros(steps, dtype=np.float64)
    denominator = 0.0
    second_digest = hashlib.sha256()
    for subject, run in samples:
        fc = read(subject, run)
        second_digest.update(json.dumps([subject, run]).encode())
        second_digest.update(fc.astype("<f8").tobytes())
        offset = fc[origin]-mean[origin]
        numerator += np.sum(offset[None]*(fc[origin+1:]-mean[origin+1:]), axis=1)
        denominator += float(np.sum(offset**2))
    if cache_digest.hexdigest() != second_digest.hexdigest():
        raise RuntimeError("Training FC caches changed during persistence fitting")
    raw = numerator / (denominator+1e-12)
    sample_json = json.dumps([(s, r, origin) for s, r in samples])
    arrays = {
        "alpha_raw": raw, "alpha": np.clip(raw, 0, 1).astype(np.float32),
        # Preserve FC[1:] storage convention for the common-origin evaluator.
        "group_template": mean[1:].astype(np.float32),
        "context_template": mean[origin].astype(np.float32),
        "fc_mean": mean[origin+1:].mean(0).astype(np.float32),
        "fc_std": np.ones(edges, dtype=np.float32),
        # Dataset compatibility placeholders: no SC is read, encoded or fitted.
        "sc_mean": np.zeros(edges, dtype=np.float32), "sc_std": np.ones(edges, dtype=np.float32),
        "protocol_json": np.array(json.dumps(persistence_protocol(config), sort_keys=True)),
        "training_samples_json": np.array(sample_json),
        "training_sample_sha256": np.array(hashlib.sha256(sample_json.encode()).hexdigest()),
        "training_fc_sha256": np.array(cache_digest.hexdigest()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent preparation from replacing a fit.
    with output.open("xb") as handle:
        try:
            np.savez_compressed(handle, **arrays)
        except Exception:
            handle.close()
            output.unlink(missing_ok=True)
            raise
    return {"id": config["artifacts"]["history_persistence"]["id"],
            "path": str(output.resolve()), "sha256": file_sha256(output),
            "n_training_subjects": len(subjects), "forecast_origin": origin,
            "forecast_windows": steps, "alpha_range": [float(raw.clip(0, 1).min()), float(raw.clip(0, 1).max())]}


def bind_history_persistence(artifact_path, experiment_paths):
    """Explicitly bind a prepared fit to unrun planned configs and their registry."""
    from copy import deepcopy
    import yaml
    from .config import load_config
    from .management import (config_sha256, registry_path, _read_registry,
                             _write_registry, validate_experiment_config)
    artifact_path = Path(artifact_path).resolve()
    digest = file_sha256(artifact_path)
    paths = [Path(p).resolve() for p in experiment_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Choose distinct planned experiment config paths")
    configs = [validate_experiment_config(load_config(p)) for p in paths]
    roots = {Path(c["paths"]["root"]).resolve() for c in configs}
    if len(roots) != 1 or any(c["model"]["name"] != MODEL_NAME for c in configs):
        raise ValueError("Binding requires history Transformer configs in one workspace")
    root = roots.pop()
    relative = artifact_path.relative_to(root).as_posix()
    registry = registry_path(root)
    rows = _read_registry(registry)
    updated, references = [], []
    baseline_protocol = None
    for path, config in zip(paths, configs):
        experiment_id = config["experiment"]["id"]
        matches = [row for row in rows if row["experiment_id"] == experiment_id]
        if (any((resolve_path(config, "output_dir") / experiment_id / "runs").glob("*/metadata.json"))
                or any(row.get("status") != "PLANNED" or row["config_sha256"] != config_sha256(config) for row in matches)):
            raise ValueError("Only intact, unrun PLANNED experiments can bind a persistence artifact")
        if len(matches) != 1:
            raise ValueError("Register each draft experiment before binding")
        reference = {"id": config["artifacts"]["history_persistence"]["id"],
                     "path": relative, "sha256": digest}
        proposed = deepcopy(config)
        proposed["artifacts"]["history_persistence"] = reference
        validate_experiment_config(proposed)
        load_persistence_artifact(proposed)
        protocol = {"data": {k: v for k, v in proposed["data"].items() if k != "warmup_windows"},
                    **{k: proposed[k] for k in ("model", "training", "evaluation", "artifacts")}}
        if baseline_protocol is not None and baseline_protocol != protocol:
            raise ValueError("Bound history runs must share a scientific protocol except K")
        baseline_protocol = protocol
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document.setdefault("artifacts", {})["history_persistence"] = reference
        updated.append(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
        references.append((experiment_id, reference))
    original = {path: path.read_bytes() for path in paths}
    registry_bytes = registry.read_bytes()
    try:
        for path, content in zip(paths, updated):
            path.write_text(content, encoding="utf-8")
        for path in paths:
            config = validate_experiment_config(load_config(path))
            load_persistence_artifact(config)
            row = next(row for row in rows if row["experiment_id"] == config["experiment"]["id"])
            row["config_sha256"] = config_sha256(config)
            row["notes"] = "Persistence artifact bound; no neural training performed."
        _write_registry(registry, rows)
    except Exception:
        for path, content in original.items():
            path.write_bytes(content)
        registry.write_bytes(registry_bytes)
        raise
    return {"sha256": digest, "experiments": [identity for identity, _ in references]}
