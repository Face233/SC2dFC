"""Restore immutable validation-selection records overwritten by evaluation.

The migration is intentionally local to managed run directories. It never
changes checkpoints or evaluation reports and creates a one-time backup of
every legacy metrics_best.json before replacing it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scdfc.connectivity import nonoverlap_horizon
from scdfc.metric_records import selection_record


def _objective_definition(config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    recorded_type = checkpoint.get("loss_type") or training.get("loss_type")
    if recorded_type is None:
        recorded_type = "huber_historical"
    return {
        "implementation": "restored_checkpoint_validation_metrics/v1",
        "loss_type": recorded_type,
        "huber_beta": float(training.get("huber_beta", 1.0)),
        "weights": {name: float(value) for name, value in training.get("loss_weights", {}).items()
                    if float(value) != 0},
        "nonoverlap_start": nonoverlap_horizon(
            int(config["data"]["window_length"]), int(config["data"]["stride"])),
        "note": "Value restored from the saved best checkpoint; definition is descriptive and was not recomputed.",
    }


def restore_run(run_dir: Path, apply: bool) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.yaml"
    metric_path = run_dir / "metrics_best.json"
    if not config_path.exists() or not metric_path.exists():
        return {"run": run_dir.name, "status": "skipped", "reason": "missing config or metrics"}
    current = json.loads(metric_path.read_text(encoding="utf-8"))
    if current.get("schema_version") == 2 and current.get("kind") == "selection":
        return {"run": run_dir.name, "status": "already_v2"}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    task = config.get("experiment", {}).get("task", "sequence")
    primary = str(config.get("evaluation", {}).get("primary_metric", ""))
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if task in {"sequence", "autoencoder"}:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot restore selection metrics without checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        checkpoint = {}
    if task == "sequence":
        metrics = checkpoint.get("validation_metrics")
        if not metrics or primary not in metrics:
            raise ValueError(f"Checkpoint lacks {primary} validation metrics: {checkpoint_path}")
        record = selection_record(metrics, primary, int(checkpoint["epoch"]),
            definition=_objective_definition(config, checkpoint), source="restored_best_checkpoint")
    elif task == "autoencoder":
        if primary != "validation_loss" or "loss" not in checkpoint:
            raise ValueError(f"Autoencoder checkpoint cannot restore {primary}: {checkpoint_path}")
        weights = config.get("training", {}).get("autoencoder_loss_weights", {})
        record = selection_record({primary: float(checkpoint["loss"])}, primary, int(checkpoint["epoch"]),
            definition={"implementation": "restored_autoencoder_checkpoint/v1", "weights": weights},
            source="restored_best_checkpoint")
    elif task == "analytic":
        evaluation_path = run_dir / "evaluation_val.json"
        if not evaluation_path.exists():
            raise FileNotFoundError(f"Cannot restore analytic validation metric: {evaluation_path}")
        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if report.get("split") != "val" or primary not in report.get("aggregate", {}):
            raise ValueError(f"Analytic report lacks validation {primary}: {evaluation_path}")
        record = selection_record(report["aggregate"], primary, None,
            definition={"implementation": "restored_analytic_validation/v1", "metric": primary},
            source="restored_analytic_validation")
    else:
        raise ValueError(f"Unsupported managed task {task!r}: {run_dir}")
    backup_path = run_dir / "metrics_best.pre_schema_v2.json"
    if apply:
        if backup_path.exists():
            raise FileExistsError(f"Refusing to replace existing migration backup: {backup_path}")
        metric_path.replace(backup_path)
        metric_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "run": run_dir.name, "experiment": config.get("experiment", {}).get("id"),
        "task": task, "primary_metric": primary, "value": record["metrics"][primary],
        "best_epoch": record["best_epoch"], "status": "repaired" if apply else "would_repair",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = [restore_run(run, args.apply) for run in sorted((args.root / "outputs").glob("E*/runs/*"))]
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
