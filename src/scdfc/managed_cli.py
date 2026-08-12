from __future__ import annotations

import json
from pathlib import Path

import yaml
import torch

from .config import load_config, resolve_path
from .data import fit_training_statistics
from .evaluation import evaluate_analytic_baseline, evaluate_checkpoint
from .management import (
    allocate_experiment_id,
    conclude_experiment,
    config_sha256,
    create_run_context,
    file_sha256,
    find_run,
    finish_run,
    freeze_dataset,
    register_experiment,
    permitted_evaluation_split,
    summarize_experiment,
    utc_now,
    validate_experiment_config,
    verify_artifact,
    verify_data_bindings,
    write_run_provenance,
)
from .training import train_autoencoder, train_sequence_model
from .progress import emit


def _root() -> Path:
    return Path.cwd().resolve()


def _stats_path(config: dict, window: int) -> Path:
    return resolve_path(config, "output_dir") / "shared" / config["data"]["dataset_version"] / f"window_{window}" / "training_stats.npz"


def _ensure_stats(config: dict, window: int) -> Path:
    path = _stats_path(config, window)
    if not path.exists():
        emit("training_statistics_started", window_length=window, output_path=str(path))
        fit_training_statistics(config, window, path)
        emit("training_statistics_finished", window_length=window, output_path=str(path))
    return path


def command_freeze_data(args) -> None:
    config = load_config(args.config)
    result = freeze_dataset(config, args.dataset_version, args.preprocessing_version, args.split_version, args.sample_limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _artifact_reference(root: Path, artifact_manifest: str | None) -> dict | None:
    if artifact_manifest is None:
        return None
    path = Path(artifact_manifest)
    if not path.is_absolute():
        path = root / path
    artifact = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {key: artifact[key] for key in ["id", "path", "sha256"]}


def command_experiment_create(args) -> None:
    root = _root()
    experiment_id = allocate_experiment_id(root)
    manifest_path = root / "data" / "manifests" / f"{args.dataset_version}.json"
    split_path = root / "data" / "manifests" / f"{args.split_version}.csv"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError("Freeze the requested dataset and split before creating an experiment")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = args.task
    document = {
        "base": "../default.yaml",
        "experiment": {
            "id": experiment_id, "name": args.name, "level": args.level, "task": task,
            "research_question": args.research_question, "hypothesis": args.hypothesis,
            "primary_change": args.primary_change, "baseline": args.baseline or "",
            "owner": args.owner, "status": "planned", "ablation": args.ablation,
        },
        "paths": {"split_csv": f"data/manifests/{args.split_version}.csv"},
        "data": {
            "dataset_version": args.dataset_version, "preprocessing_version": args.preprocessing_version,
            "manifest_path": f"data/manifests/{args.dataset_version}.json",
            "manifest_sha256": manifest["manifest_sha256"], "split_version": args.split_version,
            "split_sha256": file_sha256(split_path),
        },
        "model": {"name": args.model},
        "training": {"seeds": args.seeds},
        "evaluation": {"primary_metric": args.primary_metric},
        "decision_rule": {"description": args.decision_rule},
    }
    if task == "autoencoder":
        document["experiment"]["artifact_id"] = f"A{experiment_id[1:]}"
        document["evaluation"]["primary_metric"] = "validation_loss"
    elif task == "sequence":
        document["model"]["sc_encoder"] = args.sc_encoder
        document["model"]["output_head"] = "e0003_reconstruction_decoder"
        if args.model == "gru":
            document["model"]["gru_layers"] = args.gru_layers
        document["training"].update({
            "huber_beta": args.huber_beta,
            "finetune_fc_decoder": args.finetune_fc_decoder,
            "loss_weights": {"edge": 1.0, "difference": args.difference_weight},
        })
    artifact = _artifact_reference(root, args.artifact)
    if artifact:
        document["artifacts"] = {"fc_autoencoder": artifact}
    safe_name = "".join(character if character.isalnum() or character == "_" else "_" for character in args.name.lower()).strip("_")
    path = root / "configs" / "experiments" / f"{experiment_id}_{safe_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:
        register_experiment(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    print(path)


def _checkpoint_metadata(config: dict, context, artifact_path: Path | None) -> dict:
    return {
        "experiment_id": context.experiment_id, "run_id": context.run_id,
        "config_sha256": context.config_hash,
        "dataset_version": config["data"]["dataset_version"],
        "split_version": config["data"]["split_version"],
        "artifact_sha256": file_sha256(artifact_path) if artifact_path else None,
    }


def _write_artifact_manifest(config: dict, context, checkpoint: Path) -> Path:
    artifact_id = config["experiment"]["artifact_id"]
    root = Path(config["paths"]["root"])
    relative = checkpoint.resolve().relative_to(root).as_posix()
    document = {
        "schema_version": 1, "id": artifact_id, "type": "fc_autoencoder",
        "path": relative, "sha256": file_sha256(checkpoint),
        "experiment_id": context.experiment_id, "run_id": context.run_id,
        "config_sha256": context.config_hash, "created_at": utc_now(),
        "dataset_version": config["data"]["dataset_version"],
        "split_version": config["data"]["split_version"],
        "window_length": int(config["data"]["window_length"]),
    }
    path = root / "configs" / "artifacts" / f"{artifact_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Artifact manifest already exists: {path}")
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def command_run(args) -> None:
    config = validate_experiment_config(load_config(args.experiment))
    data_manifest = verify_data_bindings(config)
    artifact_path = verify_artifact(config)
    context = create_run_context(config, args.seed)
    try:
        emit("run_started", experiment_id=context.experiment_id, run_id=context.run_id, seed=args.seed, run_dir=str(context.run_dir))
        write_run_provenance(config, context, data_manifest)
        config["seed"] = args.seed
        window = int(config["data"]["window_length"])
        stats = _ensure_stats(config, window)
        task = config["experiment"].get("task", "sequence")
        metadata = _checkpoint_metadata(config, context, artifact_path)
        if task == "autoencoder":
            checkpoint = train_autoencoder(config, window, stats, args.device, context.run_dir, metadata)
            artifact_manifest = _write_artifact_manifest(config, context, checkpoint)
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint), artifact_manifest=str(artifact_manifest))
        elif task == "analytic":
            evaluate_analytic_baseline(config, window, stats, config["model"]["name"], "val", context.run_dir)
            finish_run(context, "COMPLETED")
        else:
            checkpoint = train_sequence_model(
                config, window, config["model"]["name"], stats,
                config["experiment"].get("ablation", "full"), args.device,
                config["model"].get("sc_encoder"), context.run_dir, metadata, artifact_path,
            )
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint))
        print(context.run_id)
        emit("run_finished", experiment_id=context.experiment_id, run_id=context.run_id, status="COMPLETED")
    except Exception as error:
        if (context.run_dir / "metadata.json").exists():
            finish_run(context, "FAILED", error=f"{type(error).__name__}: {error}")
        emit("run_failed", experiment_id=context.experiment_id, run_id=context.run_id, error=f"{type(error).__name__}: {error}")
        raise


def command_evaluate_run(args) -> None:
    root = _root()
    run_dir = find_run(root, args.run_id)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise RuntimeError("Resolved config no longer matches the run metadata")
    level = int(metadata["level"])
    split = permitted_evaluation_split(level, args.split, args.final_test)
    if args.final_test:
        lock = run_dir.parent.parent / "final_test.lock"
        if lock.exists():
            raise RuntimeError("This experiment already completed final test evaluation")
    else:
        lock = None
    verify_data_bindings(config)
    artifact = verify_artifact(config)
    window = int(config["data"]["window_length"])
    task = config["experiment"].get("task")
    if task == "autoencoder":
        raise ValueError("Autoencoder runs are evaluated during training; use summarize directly")
    if task == "analytic":
        report_path = evaluate_analytic_baseline(config, window, _stats_path(config, window), config["model"]["name"], split, run_dir)
    else:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id, "config_sha256": metadata["config_sha256"]}.items():
            if checkpoint_payload.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} does not match run metadata")
        report_path = evaluate_checkpoint(
            config, window, checkpoint, _stats_path(config, window),
            save_predictions=args.final_test, device_name=args.device, split_name=split,
            output_dir=run_dir, autoencoder_path=artifact,
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    (run_dir / "metrics_best.json").write_text(
        json.dumps({"metrics": report["aggregate"], "primary_metric": config["evaluation"]["primary_metric"]}, indent=2),
        encoding="utf-8",
    )
    if lock is not None:
        lock.write_text(json.dumps({"run_id": args.run_id, "completed_at": utc_now(), "config_sha256": metadata["config_sha256"]}, indent=2), encoding="utf-8")
    print(report_path)


def command_summarize(args) -> None:
    print(json.dumps(summarize_experiment(_root(), args.experiment), indent=2, ensure_ascii=False))


def command_conclude(args) -> None:
    conclude_experiment(_root(), args.experiment, args.status, args.conclusion, args.next_step, args.notes or "")
    print(args.experiment)
