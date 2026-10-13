"""Build a compact K comparison table and figure from evaluated managed runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .management import config_sha256, file_sha256, find_run


METRICS = (
    "forecast_mse", "forecast_space_r", "forecast_time_r",
    "forecast_nonoverlap_mse", "forecast_nonoverlap_space_r", "forecast_nonoverlap_time_r",
    "forecast_nonoverlap_adjusted_time_r", "forecast_nonoverlap_std_ratio",
    "forecast_nonoverlap_time_valid_fraction",
)


def summarize_history_runs(root, run_ids, output_dir=None):
    groups, identities, protocol, samples = {}, set(), None, None
    provenance = []
    for run_id in run_ids:
        run = find_run(root, run_id)
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        config = load_config(run / "config_resolved.yaml")
        if (config_sha256(config) != metadata["config_sha256"] or metadata.get("status") != "COMPLETED"
                or config["model"]["name"] != "fc_history_gru_query_residual"):
            raise ValueError("History summary requires intact completed history runs")
        report_path = run / "evaluation_val.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["split"] != "val" or report["checkpoint_sha256"] != file_sha256(run / "checkpoints/best.pt"):
            raise ValueError("History summary requires a current validation evaluation")
        current = {"data": {k: v for k, v in config["data"].items() if k != "warmup_windows"},
                   **{key: config[key] for key in ("model", "training", "artifacts", "evaluation", "baselines")}}
        if protocol is not None and (current != protocol or samples != report["sample_manifest_sha256"]):
            raise ValueError("History runs must share targets, validation subjects and scientific protocol")
        protocol, samples = current, report["sample_manifest_sha256"]
        identity = (metadata["experiment_id"], metadata["seed"])
        if identity in identities:
            raise ValueError("Choose at most one run per experiment and seed")
        identities.add(identity)
        groups.setdefault(metadata["experiment_id"], []).append((metadata, config, report))
        provenance.append({"run_id": run_id, "config_sha256": metadata["config_sha256"],
                           "evaluation_sha256": file_sha256(report_path), "git_commit": metadata["git"]["commit"]})
    rows = []
    for experiment, entries in groups.items():
        ks = {item[1]["data"]["warmup_windows"] for item in entries}
        if len(ks) != 1:
            raise ValueError("One experiment cannot contain different history lengths")
        row = {"experiment_id": experiment, "K": ks.pop(), "H": entries[0][1]["data"]["forecast_windows"],
               "seeds": ";".join(str(item[0]["seed"]) for item in entries), "completed_runs": len(entries)}
        for metric in METRICS:
            values = [float(item[2]["aggregate"][metric]) for item in entries
                      if item[2]["aggregate"][metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    if not rows:
        raise ValueError("No history runs to summarize")
    rows.sort(key=lambda row: row["K"])
    destination = Path(output_dir) if output_dir is not None else Path(root) / "reports/history_full_forecast"
    destination.mkdir(parents=True, exist_ok=True)
    table = destination / "history_comparison.csv"
    with table.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = destination / "summary.json"
    document.write_text(json.dumps({"rows": rows, "runs": provenance,
                                   "sample_manifest_sha256": samples,
                                   "error_bars": "between-seed standard deviation, not subject bootstrap CI"},
                                  indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metrics, title in (
        (axes[0], ("forecast_space_r", "forecast_nonoverlap_space_r"), "Spatial Pearson"),
        (axes[1], ("forecast_time_r", "forecast_nonoverlap_time_r"), "Temporal Pearson"),
        (axes[2], ("forecast_mse", "forecast_nonoverlap_mse"), "Fisher-z MSE"),
    ):
        for metric in metrics:
            selected = [row for row in rows if row[f"{metric}_mean"] is not None]
            axis.errorbar([row["K"] for row in selected], [row[f"{metric}_mean"] for row in selected],
                          yerr=[row[f"{metric}_std"] or 0 for row in selected], marker="o", capsize=3,
                          label="Nonoverlap" if "nonoverlap" in metric else "All future windows")
        axis.set(title=title, xlabel="History windows K", xticks=[row["K"] for row in rows])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Common forecast origin; error bars: between-seed SD")
    figure.savefig(destination / "history_comparison.png", dpi=180)
    plt.close(figure)
    return table
