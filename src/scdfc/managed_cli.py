from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
import torch

from .config import load_config, resolve_path
from .data import fit_training_statistics
from .evaluation import dynamic_audit_checkpoint, evaluate_analytic_baseline, evaluate_checkpoint
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
    require_execution_ready,
    permitted_evaluation_split,
    summarize_experiment,
    utc_now,
    validate_experiment_config,
    verify_artifact,
    verify_decay_artifact,
    verify_data_bindings,
    write_run_provenance,
)
from .training import train_autoencoder, train_sequence_model
from .metric_records import selection_record
from .progress import emit
from .residual_ridge import evaluate_residual_ridge, run_residual_ridge
from .residual_dynamic_ridge import evaluate_residual_dynamic_ridge, run_residual_dynamic_ridge
from .history_persistence import HISTORY_MODELS, MODEL_NAME as HISTORY_TRANSFORMER


def _root() -> Path:
    return Path.cwd().resolve()


def _stats_path(config: dict, window: int) -> Path:
    if config.get("model", {}).get("name") == HISTORY_TRANSFORMER:
        from .history_persistence import persistence_path
        return persistence_path(config)
    filename = ("history_training_stats.npz" if config.get("model", {}).get("name") == "fc_history_gru_query_residual"
                else "training_stats.npz")
    return resolve_path(config, "output_dir") / "shared" / config["data"]["dataset_version"] / f"window_{window}" / filename


def _ensure_stats(config: dict, window: int) -> Path:
    if config.get("model", {}).get("name") == HISTORY_TRANSFORMER:
        from .history_persistence import load_persistence_artifact
        load_persistence_artifact(config)
        return _stats_path(config, window)
    path = _stats_path(config, window)
    needs_refresh = not path.exists()
    if path.exists():
        with np.load(path) as stats:
            # Older shared statistics predate the causal FC0 template used by
            # E0031.  Refreshing is deterministic and preserves all existing
            # training-only statistics while adding the missing field.
            needs_refresh = "context_template" not in stats.files
    if needs_refresh:
        emit("training_statistics_started", window_length=window, output_path=str(path))
        fit_training_statistics(config, window, path)
        emit("training_statistics_finished", window_length=window, output_path=str(path))
    return path


def command_freeze_data(args) -> None:
    config = load_config(args.config)
    result = freeze_dataset(config, args.dataset_version, args.preprocessing_version, args.split_version, args.sample_limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def command_fit_history_persistence(args) -> None:
    from .history_persistence import fit_history_persistence
    config = validate_experiment_config(load_config(args.experiment))
    verify_data_bindings(config)
    output = Path(args.output) if args.output else Path(config["artifacts"]["history_persistence"]["path"])
    if not output.is_absolute():
        output = Path(config["paths"]["root"]) / output
    print(json.dumps(fit_history_persistence(config, output), indent=2, ensure_ascii=False))


def command_bind_history_persistence(args) -> None:
    from .history_persistence import bind_history_persistence
    print(json.dumps(bind_history_persistence(args.artifact, args.experiments), indent=2, ensure_ascii=False))


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
        document["model"]["output_head"] = args.output_head
        if args.model == "gru":
            document["model"]["gru_layers"] = args.gru_layers
        loss_weights = {"edge": 1.0, "difference": args.difference_weight}
        if float(args.variance_weight) != 0.0:
            loss_weights["variance"] = float(args.variance_weight)
        if float(args.long_horizon_variance_weight) != 0.0:
            loss_weights["long_horizon_variance"] = float(args.long_horizon_variance_weight)
        document["training"].update({
            "loss_type": args.loss_type,
            "huber_beta": args.huber_beta,
            "loss_weights": loss_weights,
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
    metadata = {
        "experiment_id": context.experiment_id, "run_id": context.run_id,
        "config_sha256": context.config_hash,
        "dataset_version": config["data"]["dataset_version"],
        "split_version": config["data"]["split_version"],
        "artifact_sha256": file_sha256(artifact_path) if artifact_path else None,
    }
    if config.get("model", {}).get("fc_representation") == "frozen_pca":
        metadata.update(fc_representation="frozen_pca",
                        fc_pca_manifest_sha256=config["artifacts"]["fc_pca"]["manifest_sha256"])
    return metadata


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
    require_execution_ready(config)
    if config["experiment"].get("task") == "ae_diagnostic":
        raise ValueError("Use scdfc ae-diagnostic --experiment for reconstruction diagnostics; no training run is created")
    if config["model"]["name"] == HISTORY_TRANSFORMER:
        from .history_persistence import load_persistence_artifact
        load_persistence_artifact(config)
    data_manifest = verify_data_bindings(config)
    artifact_path = verify_artifact(config)
    if config.get("artifacts", {}).get("sc_alpha"):
        from .alpha_transfer import verify_transfer_bindings
        verify_transfer_bindings(config)
    decay_path = verify_decay_artifact(config) if config["experiment"].get("task") in {"residual_ridge", "residual_dynamic_ridge", "sc_alpha"} or config.get("model", {}).get("output_decomposition") == "e0032_decay_residual" else None
    context = create_run_context(config, args.seed)
    try:
        emit("run_started", experiment_id=context.experiment_id, run_id=context.run_id, seed=args.seed, run_dir=str(context.run_dir))
        write_run_provenance(config, context, data_manifest)
        config["seed"] = args.seed
        window = int(config["data"]["window_length"])
        task = config["experiment"].get("task", "sequence")
        stats = (None if task == "sc_alpha" or (task == "autoencoder" and config["training"].get("autoencoder_data_mode") == "all_lr_windows")
                 else _ensure_stats(config, window))
        metadata = _checkpoint_metadata(config, context, artifact_path)
        if task == "sc_alpha":
            from .sc_alpha import run_sc_alpha
            checkpoint = run_sc_alpha(config, context.run_dir, metadata, args.device)
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint), selection_split="train_cv")
        elif task == "autoencoder":
            checkpoint = train_autoencoder(config, window, stats, args.device, context.run_dir, metadata)
            artifact_manifest = _write_artifact_manifest(config, context, checkpoint)
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint), artifact_manifest=str(artifact_manifest))
        elif task == "analytic":
            report_path = evaluate_analytic_baseline(config, window, stats, config["model"]["name"], "val", context.run_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            primary = config["evaluation"]["primary_metric"]
            definition = (report.get("objective") or {}).get("definition") if primary == "objective_loss" else {"metric": primary, "implementation": "sequence_metrics/v2"}
            record = selection_record(report["aggregate"], primary, None,
                definition=definition, source="analytic_validation")
            (context.run_dir / "metrics_best.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            finish_run(context, "COMPLETED")
        elif task == "residual_ridge":
            checkpoint = run_residual_ridge(
                config, window, stats, artifact_path, decay_path, context.run_dir, metadata, args.device,
            )
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint))
        elif task == "residual_dynamic_ridge":
            checkpoint = run_residual_dynamic_ridge(
                config, window, stats, artifact_path, decay_path, context.run_dir, metadata, args.device,
            )
            finish_run(context, "COMPLETED", checkpoint=str(checkpoint))
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
    # Root is not part of the config hash; resolve moved runs in this checkout.
    config.setdefault("paths", {})["root"] = str(root)
    level = int(metadata["level"])
    split = permitted_evaluation_split(level, args.split, args.final_test)
    if config["experiment"].get("task") == "sc_alpha":
        if split != "val" or args.final_test or getattr(args, "baseline_run_id", None) or getattr(args, "save_predictions", False):
            raise ValueError("SC alpha uses validation-only automatic B0/SC-swap comparisons; private predictions are not exported")
        from .sc_alpha_evaluation import evaluate_sc_alpha_run
        print(evaluate_sc_alpha_run(config, run_dir, args.device or "cpu"))
        return
    if config.get("artifacts", {}).get("sc_alpha"):
        if split != "val" or args.final_test or getattr(args, "baseline_run_id", None) or getattr(args, "save_predictions", False):
            raise ValueError("Alpha transfer is validation-only with its pinned residual reference; predictions remain private")
        from .alpha_transfer_evaluation import evaluate_transfer
        print(evaluate_transfer(config, run_dir, args.device or "cpu"))
        return
    if args.final_test:
        lock = run_dir.parent.parent / "final_test.lock"
        if lock.exists():
            raise RuntimeError("This experiment already completed final test evaluation")
    else:
        lock = None
    verify_data_bindings(config)
    artifact = verify_artifact(config)
    decay = verify_decay_artifact(config) if config["experiment"].get("task") in {"residual_ridge", "residual_dynamic_ridge"} or config.get("model", {}).get("output_decomposition") == "e0032_decay_residual" else None
    window = int(config["data"]["window_length"])
    task = config["experiment"].get("task")
    if task == "autoencoder":
        raise ValueError("Autoencoder runs are evaluated during training; use summarize directly")
    if config.get("model", {}).get("name") in HISTORY_MODELS:
        from .history_forecast import evaluate_history_forecast
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id,
                              "config_sha256": metadata["config_sha256"]}.items():
            if checkpoint_payload.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} does not match run metadata")
        baseline_checkpoint, baseline_config = None, None
        if getattr(args, "baseline_run_id", None):
            baseline_dir = find_run(root, args.baseline_run_id)
            baseline_metadata = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
            baseline_config = load_config(baseline_dir / "config_resolved.yaml")
            if config_sha256(baseline_config) != baseline_metadata["config_sha256"]:
                raise RuntimeError("Baseline resolved config no longer matches its run")
            baseline_data = {k: v for k, v in baseline_config["data"].items() if k != "warmup_windows"}
            main_data = {k: v for k, v in config["data"].items() if k != "warmup_windows"}
            if (baseline_metadata.get("status") != "COMPLETED"
                    or baseline_metadata["seed"] != metadata["seed"]
                    or baseline_config["data"].get("warmup_windows") != 1
                    or baseline_data != main_data
                    or any(baseline_config[section] != config[section] for section in ("model", "training", "artifacts"))):
                raise ValueError("History baseline must be a completed, same-seed K=1 run with matching protocol")
            baseline_config["paths"]["root"] = str(root)
            baseline_checkpoint = baseline_dir / "checkpoints" / "best.pt"
            baseline_payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
            for key, expected in {"experiment_id": baseline_metadata["experiment_id"],
                                  "run_id": args.baseline_run_id,
                                  "config_sha256": baseline_metadata["config_sha256"]}.items():
                if baseline_payload.get(key) != expected:
                    raise RuntimeError(f"Baseline checkpoint {key} does not match its run")
        report_path = evaluate_history_forecast(
            config, window, checkpoint, _stats_path(config, window), baseline_checkpoint,
            args.final_test or getattr(args, "save_predictions", False), args.device,
            split, run_dir, artifact, baseline_config,
        )
    elif task == "analytic":
        report_path = evaluate_analytic_baseline(config, window, _stats_path(config, window), config["model"]["name"], split, run_dir)
    elif task == "residual_ridge":
        checkpoint = run_dir / "checkpoints" / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id,
                              "config_sha256": metadata["config_sha256"]}.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} does not match run metadata")
        report_path = evaluate_residual_ridge(
            config, window, _stats_path(config, window), artifact, decay, checkpoint, run_dir, split, args.device,
        )
    elif task == "residual_dynamic_ridge":
        checkpoint = run_dir / "checkpoints" / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id,
                              "config_sha256": metadata["config_sha256"]}.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} does not match run metadata")
        report_path = evaluate_residual_dynamic_ridge(
            config, window, _stats_path(config, window), artifact, decay, checkpoint, run_dir, split, args.device,
        )
    else:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id, "config_sha256": metadata["config_sha256"]}.items():
            if checkpoint_payload.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} does not match run metadata")
        baseline_checkpoint, baseline_config, baseline_artifact = None, None, None
        baseline_run_id = getattr(args, "baseline_run_id", None)
        if baseline_run_id:
            pca_comparison = config.get("model", {}).get("fc_representation") == "frozen_pca"
            if config.get("model", {}).get("output_decomposition") != "e0032_decay_residual" or (not pca_comparison and config["experiment"].get("ablation") != "full"):
                raise ValueError("--baseline-run-id is available only for the E0037 full SC residual experiment")
            baseline_dir = find_run(root, baseline_run_id)
            baseline_metadata = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
            baseline_config = load_config(baseline_dir / "config_resolved.yaml")
            if config_sha256(baseline_config) != baseline_metadata["config_sha256"]:
                raise RuntimeError("Baseline resolved config no longer matches its run")
            if pca_comparison:
                from .pca_sequence import validate_pca_pair
                validate_pca_pair(config, baseline_config, metadata, baseline_metadata)
            else:
                if baseline_metadata["experiment_id"] != "E0036" or baseline_config["experiment"].get("ablation") != "fc1_only":
                    raise ValueError("Paired baseline must be an E0036 FC1-only run")
                if baseline_metadata["seed"] != metadata["seed"] or any(
                    baseline_config[section] != config[section] for section in ("data", "model", "artifacts", "training")
                ):
                    raise ValueError("E0036 paired baseline must match seed, data, model, artifacts and training")
            baseline_config["paths"]["root"] = str(root)
            verify_data_bindings(baseline_config)
            baseline_artifact = verify_artifact(baseline_config)
            verify_decay_artifact(baseline_config)
            baseline_checkpoint = baseline_dir / "checkpoints" / "best.pt"
            baseline_payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
            for key, expected in {"experiment_id": baseline_metadata["experiment_id"], "run_id": baseline_run_id,
                                  "config_sha256": baseline_metadata["config_sha256"]}.items():
                if baseline_payload.get(key) != expected:
                    raise RuntimeError(f"Baseline checkpoint {key} does not match run metadata")
        report_path = evaluate_checkpoint(
            config, window, checkpoint, _stats_path(config, window),
            baseline_checkpoint=baseline_checkpoint,
            save_predictions=args.final_test, device_name=args.device, split_name=split,
            output_dir=run_dir, autoencoder_path=artifact,
            baseline_config=baseline_config, baseline_autoencoder_path=baseline_artifact,
            baseline_stats_path=_stats_path(baseline_config, window) if baseline_config is not None else None,
        )
    if lock is not None:
        lock.write_text(json.dumps({"run_id": args.run_id, "completed_at": utc_now(), "config_sha256": metadata["config_sha256"]}, indent=2), encoding="utf-8")
    print(report_path)


def command_dynamic_audit(args) -> None:
    """Run the validation-only dynamic calibration report for a managed sequence run."""
    root = _root()
    run_dir = find_run(root, args.run_id)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise RuntimeError("Resolved config no longer matches the run metadata")
    # Managed runs may have been trained on another machine with an absolute root.
    # Preserve the frozen config for provenance, but resolve local data/artifacts from this checkout.
    config["paths"]["root"] = str(root)
    if config["experiment"].get("task") != "sequence":
        raise ValueError("Dynamic audit is available only for sequence-model runs")
    verify_data_bindings(config)
    artifact = verify_artifact(config)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id, "config_sha256": metadata["config_sha256"]}.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Checkpoint {key} does not match run metadata")
    report_path = dynamic_audit_checkpoint(
        config, int(config["data"]["window_length"]), checkpoint, _stats_path(config, int(config["data"]["window_length"])),
        split_name=args.split, output_dir=run_dir, device_name=args.device, autoencoder_path=artifact,
    )
    print(report_path)


def command_summarize(args) -> None:
    print(json.dumps(summarize_experiment(_root(), args.experiment), indent=2, ensure_ascii=False))


def command_alpha_summary(args) -> None:
    from .sc_alpha_evaluation import summarize_sc_alpha_runs
    print(summarize_sc_alpha_runs(_root(), args.run_ids, args.output_dir))


def command_alpha_visual(args) -> None:
    from .sc_alpha_evaluation import plot_sc_alpha_run
    print(plot_sc_alpha_run(find_run(_root(), args.run_id)))


def command_history_ridge(args) -> None:
    """Derived train-CV reference; never accesses test or changes run selection."""
    from .history_ridge import run_history_ridge
    root = _root()
    run_dir = find_run(root, args.run_id)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise RuntimeError("Resolved config no longer matches the run")
    if metadata.get("status") != "COMPLETED" or config["model"]["name"] != "fc_history_gru_query_residual":
        raise ValueError("History Ridge requires a completed history forecast run")
    config["paths"]["root"] = str(root)
    verify_data_bindings(config)
    artifact = verify_artifact(config)
    checkpoint = run_dir / "checkpoints/best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, expected in {"experiment_id": metadata["experiment_id"], "run_id": args.run_id,
                          "config_sha256": metadata["config_sha256"]}.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Checkpoint {key} does not match run metadata")
    config["seed"] = metadata["seed"]
    print(run_history_ridge(config, int(config["data"]["window_length"]), checkpoint,
                            _stats_path(config, int(config["data"]["window_length"])), artifact,
                            run_dir, args.device))


def command_history_summary(args) -> None:
    from .history_summary import summarize_history_runs
    print(summarize_history_runs(_root(), args.run_ids, args.output_dir))


def command_conclude(args) -> None:
    conclude_experiment(_root(), args.experiment, args.status, args.conclusion, args.next_step, args.notes or "")
    print(args.experiment)
