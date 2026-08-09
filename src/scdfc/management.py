from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import load_config, resolve_path


SCHEMA_VERSION = 1
CONCLUSION_STATUSES = {"KEEP", "REJECT", "INCONCLUSIVE", "FAILED", "ARCHIVED"}
REGISTRY_COLUMNS = [
    "experiment_id", "name", "level", "date_created", "research_question", "hypothesis",
    "primary_change", "baseline", "dataset_version", "split_version", "config_path",
    "config_sha256", "seeds", "primary_metric", "completed_runs", "failed_runs",
    "mean", "std", "status", "conclusion", "next_step", "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {
        "dataset_version": manifest["dataset_version"],
        "preprocessing_version": manifest["preprocessing_version"],
        "files": manifest["files"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(config)
    payload.pop("_config_source", None)
    payload.pop("runtime", None)
    paths = payload.get("paths", {})
    paths.pop("root", None)
    paths.pop("output_dir", None)
    return payload


def config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(_canonical_payload(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_level(value: Any) -> int:
    aliases = {"debug": 0, "screening": 1, "formal": 2, "l0": 0, "l1": 1, "l2": 2}
    if isinstance(value, str):
        value = aliases.get(value.lower(), value)
    try:
        level = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("experiment.level must be 0/debug, 1/screening, or 2/formal") from error
    if level not in {0, 1, 2}:
        raise ValueError("experiment.level must be 0, 1, or 2")
    return level


def validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Managed configs require an experiment section")
    required = ["id", "name", "level", "research_question", "hypothesis", "primary_change", "owner"]
    missing = [key for key in required if experiment.get(key) in {None, ""}]
    if missing:
        raise ValueError(f"Missing experiment fields: {', '.join(missing)}")
    experiment_id = str(experiment["id"])
    if len(experiment_id) != 5 or not experiment_id.startswith("E") or not experiment_id[1:].isdigit():
        raise ValueError("experiment.id must match E####")
    experiment["level"] = normalize_level(experiment["level"])
    seeds = config.get("training", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("training.seeds must be a non-empty list of integers")
    data = config.get("data", {})
    for field in ["dataset_version", "preprocessing_version", "manifest_path", "manifest_sha256", "split_version", "split_sha256"]:
        if not data.get(field):
            raise ValueError(f"Managed configs require data.{field}")
    evaluation = config.get("evaluation", {})
    if not evaluation.get("primary_metric"):
        raise ValueError("Managed configs require evaluation.primary_metric")
    task = experiment.get("task", "sequence")
    if task not in {"sequence", "autoencoder", "analytic"}:
        raise ValueError("experiment.task must be sequence, autoencoder, or analytic")
    model_name = config.get("model", {}).get("name")
    allowed_models = {
        "analytic": {"group_mean", "fc1_persistence"},
        "autoencoder": {"fc_autoencoder"},
        "sequence": {"pca_ridge", "mlp", "lstm", "direct_mlp", "gcn_gru", "tcn", "transformer"},
    }
    if model_name not in allowed_models[task]:
        raise ValueError(f"model.name {model_name!r} is not valid for task {task!r}")
    primary_metric = str(config["evaluation"]["primary_metric"])
    if task == "sequence" and primary_metric != "long_residual_pearson":
        raise ValueError("Current sequence training selects checkpoints by long_residual_pearson")
    if task == "autoencoder" and primary_metric != "validation_loss":
        raise ValueError("Autoencoder experiments must use validation_loss as the primary metric")
    if not config.get("decision_rule", {}).get("description"):
        raise ValueError("Managed configs require decision_rule.description")
    if task == "sequence":
        artifact = config.get("artifacts", {}).get("fc_autoencoder", {})
        for field in ["id", "path", "sha256"]:
            if not artifact.get(field):
                raise ValueError(f"Sequence experiments require artifacts.fc_autoencoder.{field}")
    return config


def _run(command: list[str], cwd: Path, check: bool = True) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {' '.join(command)}")
    return result.stdout.strip()


def git_metadata(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    status = _run(["git", "status", "--porcelain"], root)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "branch": _run(["git", "branch", "--show-current"], root),
        "remote": _run(["git", "config", "--get", "remote.origin.url"], root, check=False),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def verify_data_bindings(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["data"]["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path(config["paths"]["root"]) / manifest_path
    split_path = resolve_path(config, "split_csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Data manifest not found: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = str(config["data"]["manifest_sha256"])
    actual_manifest = manifest_digest(manifest)
    if actual_manifest != expected_manifest:
        raise ValueError(f"Data manifest checksum mismatch: expected {expected_manifest}, got {actual_manifest}")
    actual_split = file_sha256(split_path)
    if actual_split != str(config["data"]["split_sha256"]):
        raise ValueError(f"Split checksum mismatch: expected {config['data']['split_sha256']}, got {actual_split}")
    return {"manifest_path": str(manifest_path.resolve()), "split_path": str(split_path.resolve()), **manifest}


def freeze_dataset(
    config: dict[str, Any], dataset_version: str, preprocessing_version: str, split_version: str,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Create private immutable data, audit, and subject-split manifests."""
    from .data import audit_dataset, discover_data, make_subject_split, validate_split

    root = Path(config["paths"]["root"])
    destination = root / "data" / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"{dataset_version}.json"
    split_path = destination / f"{split_version}.csv"
    audit_path = destination / f"audit_{dataset_version}.json"
    existing = [path for path in [manifest_path, split_path, audit_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Frozen version already has files: {', '.join(map(str, existing))}")
    found = discover_data(config)
    files: list[Path] = []
    atlas = resolve_path(config, "atlas_labels")
    if atlas.exists():
        files.append(atlas)
    files.extend(found["sc"].values())
    for run_files in found["runs"].values():
        files.extend(run_files.values())
    unique_files = sorted(set(path.resolve() for path in files), key=lambda path: str(path).lower())
    records = []
    for path in unique_files:
        records.append({
            "path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_version": dataset_version,
        "preprocessing_version": preprocessing_version, "generated_at": utc_now(),
        "counts": {
            "sc": len(found["sc"]), "subjects": len(found["subjects"]),
            "runs": {run: len(paths) for run, paths in found["runs"].items()}, "files": len(records),
        },
        "files": records,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    audit = audit_dataset(config, sample_limit)
    eligible_subjects = sorted(set(found["subjects"]) - set(audit.get("invalid_subjects", [])))
    split = make_subject_split(
        eligible_subjects,
        (float(config["split"]["train"]), float(config["split"]["val"]), float(config["split"]["test"])),
        int(config["seed"]),
    )
    validate_split(split)
    split_csv = split.to_csv(index=False, lineterminator="\n")
    split_hash = hashlib.sha256(split_csv.encode("utf-8")).hexdigest()
    audit.update({
        "dataset_version": dataset_version, "preprocessing_version": preprocessing_version,
        "manifest_sha256": manifest["manifest_sha256"], "split_version": split_version,
        "split_sha256": split_hash, "split_counts": split.groupby("split").size().to_dict(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    split_path.write_text(split_csv, encoding="utf-8", newline="")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset_version": dataset_version, "preprocessing_version": preprocessing_version,
        "manifest_path": str(manifest_path.relative_to(root)), "manifest_sha256": manifest["manifest_sha256"],
        "split_version": split_version, "split_path": str(split_path.relative_to(root)),
        "split_sha256": split_hash, "audit_path": str(audit_path.relative_to(root)),
    }


def verify_artifact(config: dict[str, Any]) -> Path | None:
    if config["experiment"].get("task", "sequence") != "sequence":
        return None
    reference = config["artifacts"]["fc_autoencoder"]
    path = Path(reference["path"])
    if not path.is_absolute():
        path = Path(config["paths"]["root"]) / path
    if not path.exists():
        raise FileNotFoundError(f"FC autoencoder artifact not found: {path}")
    actual = file_sha256(path)
    if actual != str(reference["sha256"]):
        raise ValueError(f"Artifact checksum mismatch: expected {reference['sha256']}, got {actual}")
    return path


def permitted_evaluation_split(level: int, requested_split: str = "val", final_test: bool = False) -> str:
    if final_test:
        if int(level) != 2:
            raise PermissionError("Only Level 2 runs may access the test split")
        return "test"
    if requested_split not in {"train", "val"}:
        raise PermissionError("Managed Level 0/1 evaluation is restricted to train/val")
    return requested_split


def environment_snapshot() -> str:
    lines = [
        f"timestamp_utc={utc_now()}", f"hostname={socket.gethostname()}",
        f"platform={platform.platform()}", f"python={sys.version.replace(os.linesep, ' ')}",
    ]
    try:
        import torch
        lines.extend([
            f"torch={torch.__version__}", f"cuda_runtime={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        ])
    except ImportError:
        lines.append("torch=not-installed")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines.append("\n[pip-freeze]\n" + freeze.stdout.strip())
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class RunContext:
    experiment_id: str
    run_id: str
    run_dir: Path
    config_hash: str
    git: dict[str, Any]
    level: int
    seed: int


def create_run_context(config: dict[str, Any], seed: int) -> RunContext:
    validate_experiment_config(config)
    if seed not in config["training"]["seeds"]:
        raise ValueError(f"Seed {seed} is not declared in training.seeds")
    root = Path(config["paths"]["root"])
    git = git_metadata(root)
    level = int(config["experiment"]["level"])
    if level >= 1 and git["dirty"]:
        raise RuntimeError("Level 1/2 experiments require a clean Git worktree")
    digest = config_sha256(config)
    experiment_id = str(config["experiment"]["id"])
    experiment_dir = resolve_path(config, "output_dir") / experiment_id
    for metadata_path in experiment_dir.glob("runs/*/metadata.json"):
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != digest:
            raise RuntimeError(f"{experiment_id} already has runs with another config hash; allocate a new experiment ID")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_id}-s{seed}-{stamp}-{git['commit'][:7]}"
    run_dir = experiment_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    if level >= 2:
        (run_dir / "predictions").mkdir()
        (run_dir / "figures").mkdir()
    return RunContext(experiment_id, run_id, run_dir, digest, git, level, seed)


def write_run_provenance(config: dict[str, Any], context: RunContext, data_manifest: dict[str, Any]) -> None:
    resolved = deepcopy(config)
    resolved["seed"] = context.seed
    resolved["runtime"] = {"run_id": context.run_id, "config_sha256": context.config_hash}
    (context.run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (context.run_dir / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (context.run_dir / "environment.txt").write_text(environment_snapshot(), encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION, "experiment_id": context.experiment_id,
        "run_id": context.run_id, "seed": context.seed, "level": context.level,
        "status": "RUNNING", "started_at": utc_now(), "config_sha256": context.config_hash,
        "git": context.git,
    }
    (context.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if context.level == 0 and context.git["dirty"]:
        diff = _run(["git", "diff", "--binary", "HEAD"], Path(config["paths"]["root"]), check=False)
        (context.run_dir / "git_diff.patch").write_text(diff, encoding="utf-8")


def finish_run(context: RunContext, status: str, **fields: Any) -> None:
    path = context.run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update({"status": status, "ended_at": utc_now(), **fields})
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def registry_path(root: str | Path) -> Path:
    return Path(root) / "reports" / "experiment_registry.csv"


def _read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_registry(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REGISTRY_COLUMNS})


def allocate_experiment_id(root: str | Path) -> str:
    root = Path(root)
    numbers = []
    for path in (root / "configs" / "experiments").glob("E*.yaml"):
        if len(path.stem) >= 5 and path.stem[1:5].isdigit():
            numbers.append(int(path.stem[1:5]))
    for row in _read_registry(registry_path(root)):
        value = row.get("experiment_id", "")
        if len(value) == 5 and value[1:].isdigit():
            numbers.append(int(value[1:]))
    return f"E{max(numbers, default=0) + 1:04d}"


def register_experiment(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    config = validate_experiment_config(load_config(config_path))
    root = Path(config["paths"]["root"])
    path = registry_path(root)
    rows = _read_registry(path)
    experiment_id = config["experiment"]["id"]
    if any(row["experiment_id"] == experiment_id for row in rows):
        raise ValueError(f"Experiment {experiment_id} is already registered")
    rows.append({
        "experiment_id": experiment_id, "name": config["experiment"]["name"],
        "level": config["experiment"]["level"], "date_created": utc_now(),
        "research_question": config["experiment"]["research_question"],
        "hypothesis": config["experiment"]["hypothesis"],
        "primary_change": config["experiment"]["primary_change"],
        "baseline": config["experiment"].get("baseline", ""),
        "dataset_version": config["data"]["dataset_version"], "split_version": config["data"]["split_version"],
        "config_path": str(config_path.relative_to(root)), "config_sha256": config_sha256(config),
        "seeds": ";".join(map(str, config["training"]["seeds"])),
        "primary_metric": config["evaluation"]["primary_metric"], "status": "PLANNED",
    })
    _write_registry(path, rows)


def find_run(root: str | Path, run_id: str) -> Path:
    matches = list((Path(root) / "outputs").glob(f"E*/runs/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one run directory for {run_id}, found {len(matches)}")
    return matches[0]


def _preprocessing_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the dFC settings that must accompany a result summary."""
    data = config.get("data", {})
    required = ("window_length", "stride", "tr_seconds", "n_timepoints", "n_nodes")
    if any(key not in data for key in required):
        return None
    window_length = int(data["window_length"])
    stride = int(data["stride"])
    n_timepoints = int(data["n_timepoints"])
    tr_seconds = float(data["tr_seconds"])
    if stride < 1 or window_length > n_timepoints:
        raise ValueError("Invalid window_length/stride in resolved run configuration")
    cache_dir = Path(config.get("paths", {}).get("cache_dir", "data/cache/dfc"))
    return {
        "dataset_version": data.get("dataset_version"),
        "preprocessing_version": data.get("preprocessing_version"),
        "split_version": data.get("split_version"),
        "window_length_tr": window_length,
        "stride_tr": stride,
        "tr_seconds": tr_seconds,
        "window_length_seconds": window_length * tr_seconds,
        "stride_seconds": stride * tr_seconds,
        "n_timepoints": n_timepoints,
        "n_dfc_windows_per_run": (n_timepoints - window_length) // stride + 1,
        "n_nodes": int(data["n_nodes"]),
        "fisher_clip": data.get("fisher_clip"),
        "estimator": "rectangular_pearson",
        "fc_representation": "Fisher-z transformed upper-triangle edges",
        "cache_path": (cache_dir / f"window_{window_length}.zarr").as_posix(),
    }


def summarize_experiment(root: str | Path, experiment_id: str) -> dict[str, Any]:
    root = Path(root)
    path = registry_path(root)
    rows = _read_registry(path)
    registry_row = next((row for row in rows if row["experiment_id"] == experiment_id), None)
    if registry_row is None:
        raise ValueError(f"Experiment {experiment_id} is not registered")
    metric = registry_row["primary_metric"]
    values: list[float] = []
    failed = 0
    run_records = []
    preprocessing: dict[str, Any] | None = None
    for run_dir in sorted((root / "outputs" / experiment_id / "runs").glob("*")):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_records.append(metadata)
        config_path = run_dir / "config_resolved.yaml"
        if config_path.exists():
            current_preprocessing = _preprocessing_summary(
                yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            )
            if current_preprocessing is not None:
                if preprocessing is None:
                    preprocessing = current_preprocessing
                elif preprocessing != current_preprocessing:
                    raise RuntimeError(
                        f"{experiment_id} contains runs with different preprocessing settings; "
                        "summarize them as separate experiments"
                    )
        if metadata.get("status") != "COMPLETED":
            failed += 1
            continue
        metric_path = run_dir / "metrics_best.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            candidate = metrics.get("metrics", metrics).get(metric)
            if candidate is not None:
                values.append(float(candidate))
    import numpy as np
    summary = {
        "experiment_id": experiment_id, "primary_metric": metric, "completed_runs": len(values),
        "failed_runs": failed, "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        "preprocessing": preprocessing, "runs": run_records, "generated_at": utc_now(),
    }
    summary_path = root / "reports" / "experiment_notes" / f"{experiment_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    registry_row.update({key: summary[key] for key in ["completed_runs", "failed_runs", "mean", "std"]})
    if values:
        registry_row["status"] = "AWAITING_CONCLUSION"
    _write_registry(path, rows)
    return summary


def conclude_experiment(root: str | Path, experiment_id: str, status: str, conclusion: str, next_step: str, notes: str = "") -> None:
    status = status.upper()
    if status not in CONCLUSION_STATUSES:
        raise ValueError(f"status must be one of {sorted(CONCLUSION_STATUSES)}")
    path = registry_path(root)
    rows = _read_registry(path)
    row = next((item for item in rows if item["experiment_id"] == experiment_id), None)
    if row is None:
        raise ValueError(f"Experiment {experiment_id} is not registered")
    row.update({"status": status, "conclusion": conclusion, "next_step": next_step, "notes": notes})
    _write_registry(path, rows)


def copy_manifest_file(config: dict[str, Any], destination: Path) -> None:
    source = Path(config["data"]["manifest_path"])
    if not source.is_absolute():
        source = Path(config["paths"]["root"]) / source
    shutil.copy2(source, destination)
