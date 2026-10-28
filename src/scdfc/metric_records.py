"""Explicit checkpoint-selection provenance for validation or training-only CV."""
from __future__ import annotations

import math
from typing import Any


def selection_record(
    metrics: dict[str, float], primary_metric: str, best_epoch: int | None,
    *, definition: dict[str, Any], source: str = "training_validation", split: str = "val",
) -> dict[str, Any]:
    record = {
        "schema_version": 2, "kind": "selection", "split": split,
        "primary_metric": primary_metric, "metrics": metrics,
        "best_epoch": best_epoch, "epoch_index_base": 0,
        "metric_definition": definition, "source": source,
    }
    validate_selection_record(record, primary_metric)
    return record


def validate_selection_record(record: dict[str, Any], primary_metric: str) -> None:
    training_cv = (record.get("split") == "train_cv" and record.get("source") == "sc_alpha_training_cv"
                   and record.get("metric_definition", {}).get("implementation") == "sc_alpha_training_cv/v1"
                   and record.get("metric_definition", {}).get("split") == "train_cv")
    if record.get("schema_version") != 2 or record.get("kind") != "selection" or (record.get("split") != "val" and not training_cv):
        raise ValueError("Unverified selection metrics; restore metrics_best.json from the checkpoint or validation log using scripts/repair_metric_records.py")
    if record.get("primary_metric") != primary_metric:
        raise ValueError("Selection primary_metric does not match experiment primary_metric")
    value = record.get("metrics", {}).get(primary_metric)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Selection metric {primary_metric} must be finite")
    if not record.get("metric_definition"):
        raise ValueError("Selection metrics require a metric definition")


def sequence_objective_definition(config: dict[str, Any], nonoverlap: int) -> dict[str, Any]:
    training = config["training"]
    return {
        "implementation": "CompositeLoss/v1", "loss_type": training.get("loss_type", "mse"),
        "huber_beta": float(training.get("huber_beta", 1.0)),
        "weights": {k: float(v) for k, v in training["loss_weights"].items() if float(v) != 0},
        "nonoverlap_start": nonoverlap,
        "batch_size": int(training["batch_size"]),
        "aggregation": "sample-weighted batch mean; contrastive negatives within each ordered batch",
    }


def sequence_selection_definition(config: dict[str, Any], nonoverlap: int) -> dict[str, Any]:
    if config["evaluation"]["primary_metric"] == "forecast_mse":
        return {
            "implementation": "history_forecast_full_mse/v1", "split": "val",
            "forecast_origins": config["data"]["forecast_origins"],
            "forecast_windows": config["data"]["forecast_windows"],
            "aggregation": "mean per run, then mean per subject, then mean over subjects",
        }
    if config["evaluation"]["primary_metric"] == "forecast_nonoverlap_mse":
        return {
            "implementation": "history_forecast_nonoverlap_mse/v1", "split": "val",
            "nonoverlap_start": nonoverlap,
            "forecast_origins": config["data"]["forecast_origins"],
            "forecast_windows": config["data"]["forecast_windows"],
            "aggregation": "mean per run, then mean per subject, then mean over subjects",
        }
    if config["evaluation"]["primary_metric"] == "long_edge_mse":
        return {
            "implementation": "sequence_validation_long_edge_mse/v1",
            "split": "val",
            "nonoverlap_start": nonoverlap,
            "aggregation": "mean squared Fisher-z edge error per subject/run, then mean over samples",
        }
    return sequence_objective_definition(config, nonoverlap)
