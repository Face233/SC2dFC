"""Reproducible static experiment-report visualizations.

All figures are rendered as PNG files under ``outputs/E####/visual``.  This
module deliberately reads the frozen configuration stored with a managed run,
never a mutable experiment YAML.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

from .data import DFCSequenceDataset
from .training import build_sequence_model
from .metric_records import validate_selection_record


FIXED_EDGE_SEED = 42
FIXED_EDGE_COUNT = 4
REPRESENTATIVE_SUBJECT = "140420"
TRUE_COLOR = "#1f77b4"
PREDICTED_COLOR = "#ff7f0e"
BASELINE_COLOR = "#7f7f7f"
SUMMARY_METRICS = (
    "temporal_std_ratio",
    "difference_std_ratio",
    "difference_temporal_pearson",
    "low_band_power_ratio",
    "mid_band_power_ratio",
)
HORIZON_METRICS = (
    "temporal_std_ratio",
    "difference_std_ratio",
    "low_band_power_ratio",
    "difference_temporal_pearson",
)
# Keep calibration bars readable without letting a single-model bar dominate
# the whole categorical axis.  This width is shared by single- and two-model
# reports so the visual grammar stays stable across experiments.
CALIBRATION_BAR_WIDTH = 0.42
CHINESE_BODY_FONT = FontProperties(family="Microsoft YaHei")
CHINESE_HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


@dataclass(frozen=True)
class ManagedRun:
    experiment_id: str
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    metadata: dict[str, Any]


def fixed_edge_indices(n_edges: int = 4005, count: int = FIXED_EDGE_COUNT, seed: int = FIXED_EDGE_SEED) -> np.ndarray:
    """Return the project-wide reproducible display-edge set."""
    if count < 1 or count > n_edges:
        raise ValueError("count must be between one and n_edges")
    return np.sort(np.random.default_rng(seed).choice(n_edges, size=count, replace=False))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required visualization input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frozen_config(path: Path, root: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Frozen config is not a mapping: {path}")
    config.setdefault("paths", {})["root"] = str(root.resolve())
    return config


def _latest_completed_run(root: Path, experiment_id: str) -> ManagedRun | None:
    candidates: list[ManagedRun] = []
    for run_dir in sorted((root / "outputs" / experiment_id / "runs").glob("*")) if (root / "outputs" / experiment_id / "runs").exists() else []:
        metadata_path = run_dir / "metadata.json"
        config_path = run_dir / "config_resolved.yaml"
        if not metadata_path.exists() or not config_path.exists():
            continue
        metadata = _read_json(metadata_path)
        if metadata.get("status") != "COMPLETED":
            continue
        candidates.append(ManagedRun(experiment_id, run_dir.name, run_dir, _read_frozen_config(config_path, root), metadata))
    return candidates[-1] if candidates else None


def _load_run(root: Path, experiment_id: str, run_id: str | None = None) -> ManagedRun:
    if run_id is None:
        result = _latest_completed_run(root, experiment_id)
        if result is None:
            raise FileNotFoundError(f"No completed run found for {experiment_id}")
        return result
    run_dir = root / "outputs" / experiment_id / "runs" / run_id
    metadata = _read_json(run_dir / "metadata.json")
    if metadata.get("status") != "COMPLETED":
        raise ValueError(f"Run is not completed: {run_id}")
    return ManagedRun(experiment_id, run_id, run_dir, _read_frozen_config(run_dir / "config_resolved.yaml", root), metadata)


def _baseline_run(root: Path, current: ManagedRun) -> ManagedRun | None:
    baseline = str(current.config.get("experiment", {}).get("baseline", "")).strip()
    if not baseline or not baseline.startswith("E"):
        return None
    return _latest_completed_run(root, baseline)


def _roi_labels(root: Path, config: dict[str, Any]) -> list[str]:
    path = Path(config["paths"]["atlas_labels"])
    if not path.is_absolute():
        path = root / path
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            labels.append(fields[1] if len(fields) > 1 else fields[0])
    n_nodes = int(config["data"]["n_nodes"])
    if len(labels) < n_nodes:
        raise ValueError(f"Expected {n_nodes} ROI labels, got {len(labels)}")
    return labels[:n_nodes]


def edge_labels(root: Path, config: dict[str, Any], indices: np.ndarray) -> list[str]:
    labels = _roi_labels(root, config)
    rows, cols = np.triu_indices(len(labels), 1)
    return [f"{labels[rows[index]]} ↔ {labels[cols[index]]}" for index in indices]


def _stats_path(root: Path, config: dict[str, Any]) -> Path:
    return root / "outputs" / "shared" / str(config["data"]["dataset_version"]) / f"window_{int(config['data']['window_length'])}" / "training_stats.npz"


def _artifact_path(root: Path, config: dict[str, Any]) -> Path:
    path = Path(config["artifacts"]["fc_autoencoder"]["path"])
    return path if path.is_absolute() else root / path


def _subject_sample(dataset: DFCSequenceDataset) -> tuple[dict[str, Any], str]:
    matches = [index for index, (subject, run) in enumerate(dataset.samples) if subject == REPRESENTATIVE_SUBJECT and run == "LR"]
    index = matches[0] if matches else next(index for index, (_, run) in enumerate(dataset.samples) if run == "LR")
    sample = dataset[index]
    return sample, str(sample["subject_id"])


def _prediction(root: Path, run: ManagedRun, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    stats_path = _stats_path(root, run.config)
    payload = torch.load(run.run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_sequence_model(
        run.config,
        int(run.config["data"]["window_length"]),
        payload["decoder_type"],
        stats_path,
        device,
        payload["sc_encoder_type"],
        _artifact_path(root, run.config),
        payload,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = DFCSequenceDataset(run.config, int(run.config["data"]["window_length"]), "val", stats_path)
    sample, subject = _subject_sample(dataset)
    with torch.no_grad():
        output = model(
            sample["sc_matrix"][None].to(device),
            sample["sc_edges"][None].to(device),
            sample["fc_warmup"][None].to(device),
            steps=len(sample["fc_future"]),
        )
    warmup = int(run.config.get("data", {}).get("warmup_windows", 1))
    starts = sample["window_starts"].numpy()[warmup:]
    return output.fc_z_edges[0].cpu().numpy(), sample["fc_future"].numpy(), starts, subject


def _align_predictions(entries: list[tuple[ManagedRun, np.ndarray, np.ndarray, np.ndarray, str]]) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], str]:
    common = set(entries[0][3].tolist())
    for _, _, _, starts, _ in entries[1:]:
        common &= set(starts.tolist())
    ordered = np.asarray(sorted(common), dtype=int)
    if len(ordered) < 2:
        raise ValueError("Compared runs have no common absolute future windows")
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    subject = entries[0][4]
    for run, prediction, target, starts, entry_subject in entries:
        if entry_subject != subject:
            raise ValueError("Representative subject differs across compared runs")
        positions = {int(start): index for index, start in enumerate(starts)}
        selected = np.asarray([positions[int(start)] for start in ordered])
        aligned[run.experiment_id] = (prediction[selected], target[selected])
    return ordered, aligned, subject


def _figure_header(fig: plt.Figure, title: str) -> tuple[Any, Any]:
    header, body = fig.subfigures(2, 1, height_ratios=(0.12, 0.88))
    header.text(0.5, 0.70, title, ha="center", va="center", fontsize=12, fontweight="semibold", multialignment="center")
    header.legend(
        [Line2D([], [], color=TRUE_COLOR, linewidth=1.8), Line2D([], [], color=PREDICTED_COLOR, linewidth=1.6)],
        ["True", "Predicted"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    return header, body


def _edge_title(label: str) -> str:
    return "\n".join(textwrap.wrap(label.replace("_", " "), width=29, max_lines=2, placeholder="…"))


def plot_fixed_edge_dynamics(
    root: Path,
    current: ManagedRun,
    compared: list[ManagedRun],
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    """Render a four-edge, model-column comparison without floating headers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = [(run, *_prediction(root, run, device)) for run in compared]
    starts, aligned, subject = _align_predictions(entries)
    edges = fixed_edge_indices()
    names = edge_labels(root, current.config, edges)
    minutes = (starts - starts[0]) * float(current.config["data"]["tr_seconds"]) / 60.0
    figure = plt.figure(figsize=(4.8 * len(compared), 10.4), layout="constrained")
    _, body = _figure_header(
        figure,
        f"{current.experiment_id} · fixed four-edge dynamics\nvalidation · subject {subject}",
    )
    axes = body.subplots(len(edges), len(compared), squeeze=False, sharex="col")
    for row, (edge, name) in enumerate(zip(edges, names)):
        values = np.concatenate([np.column_stack((prediction[:, edge], target[:, edge])) for prediction, target in aligned.values()])
        low, high = np.nanmin(values), np.nanmax(values)
        padding = max((high - low) * 0.06, 0.02)
        for col, run in enumerate(compared):
            axis = axes[row, col]
            prediction, target = aligned[run.experiment_id]
            axis.plot(minutes, target[:, edge], color=TRUE_COLOR, linewidth=1.8)
            axis.plot(minutes, prediction[:, edge], color=PREDICTED_COLOR, linewidth=1.5)
            axis.set_ylim(low - padding, high + padding)
            axis.grid(alpha=0.22, linewidth=0.6)
            if row == 0:
                axis.set_title(run.experiment_id, pad=14, fontweight="semibold")
            if col == 0:
                axis.set_ylabel("Fisher-z\n" + _edge_title(name), labelpad=8)
            if row == len(edges) - 1:
                axis.set_xlabel("Minutes after first common future window")
    path = destination / f"{current.run_id}_val_fixed4_dynamics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path, {"subject_id": subject, "edge_indices": edges.tolist(), "edge_labels": names, "common_window_starts": starts.tolist()}


def _read_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "epoch_complete":
            rows.append(row)
    return rows


def _overview_loss_text(training: dict[str, Any]) -> str:
    """Describe the active optimization terms without exposing raw config keys."""
    weights = training.get("loss_weights", {})
    labels = {"edge": "L_edge", "difference": "L_diff", "variance": "L_var", "long_variance": "L_long-var"}
    active = [
        f"{float(weight):g}·{labels.get(str(name), f'L_{name}')}"
        for name, weight in weights.items()
        if float(weight) != 0
    ]
    loss_name = "MSE" if str(training.get("loss_type", "mse")).lower() == "mse" else "Huber"
    if loss_name == "Huber":
        loss_name += f" (β={float(training.get('huber_beta', 1.0)):g})"
    return f"{loss_name}: {' + '.join(active) if active else 'no active sequence-loss term'}"


def _overview_architecture_text(model: dict[str, Any], training: dict[str, Any]) -> str:
    model_name = str(model.get("name", "sequence model")).lower()
    if model_name == "transformer":
        temporal = (
            f"TRANSFORMER ({int(model.get('transformer_layers', 1))} layers, "
            f"{int(model.get('transformer_heads', 1))} heads)"
        )
    elif model_name == "gru":
        temporal = f"GRU ({int(model.get('gru_layers', 1))} layers)"
    elif model_name == "lstm":
        temporal = f"LSTM ({int(model.get('lstm_layers', 1))} layers)"
    else:
        temporal = model_name.upper()
    parts = [f"SC {str(model.get('sc_encoder', 'encoder')).replace('_', '-').upper()}", temporal]
    output_head = str(model.get("output_head", "output head"))
    if output_head == "e0003_reconstruction_decoder":
        output = "E0003 reconstruction decoder (frozen throughout)"
    elif output_head == "direct_edge_linear":
        output = "direct edge linear head (trainable)"
    else:
        output = output_head.replace("_", " ")
    return " → ".join([*parts, output])


def _overview_design_text(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {})
    training = config.get("training", {})
    data = config.get("data", {})
    items = []
    baseline = str(experiment.get("baseline", "")).strip()
    if baseline:
        items.append(f"Baseline: {baseline}")
    warmup = int(data.get("warmup_windows", 0))
    if warmup:
        items.append(f"warm-up: {warmup} windows")
    items.append(f"{int(training.get('epochs', 0))} epochs")
    return " · ".join(items)


def _overview_purpose_text(config: dict[str, Any]) -> str:
    """Give a compact, consistently readable Chinese purpose for every experiment."""
    weights = config.get("training", {}).get("loss_weights", {})
    edge = float(weights.get("edge", 0))
    difference = float(weights.get("difference", 0))
    variance = float(weights.get("variance", 0))
    if edge == 0 and (difference != 0 or variance != 0):
        return "检验移除边重建约束、仅保留动态监督后，能否恢复 dFC 的时间波动。"
    if difference != 0 or variance != 0:
        return "检验动态监督能否同时改善 dFC 重建与时间波动。"
    return "为 dFC 预测流程建立重建基线。"


def _overview_lines(current: ManagedRun, best: dict[str, Any], evaluation: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the bounded experiment card displayed beside the optimization trace."""
    config = current.config
    experiment = config.get("experiment", {})
    training = config.get("training", {})
    aggregate = evaluation.get("aggregate", {})
    primary_metric = str(best.get("primary_metric", "objective_loss"))
    validate_selection_record(best, primary_metric)
    primary_value = best["metrics"].get(primary_metric)
    result = f"Validation {primary_metric}: {float(primary_value):.4g}" if primary_value is not None else "Validation result unavailable"
    compact_metrics = []
    for key, label in (("mse", "MSE"), ("raw_edge_pearson", "edge r"), ("fcd_pearson", "FCD r")):
        value = aggregate.get(key)
        if value is not None and np.isfinite(float(value)):
            compact_metrics.append(f"{label} {float(value):.3g}")
    if compact_metrics:
        result += " · " + " · ".join(compact_metrics)
    return [
        ("EXPERIMENT", f"{current.experiment_id} · {str(experiment.get('name', '')).replace('_', ' ')}"),
        ("目的", _overview_purpose_text(config)),
        ("DESIGN", _overview_design_text(config)),
        ("ARCHITECTURE", _overview_architecture_text(config.get("model", {}), training)),
        ("LOSS", _overview_loss_text(training)),
        ("BEST VALIDATION", result),
    ]


def plot_run_overview(current: ManagedRun, destination: Path) -> Path:
    rows = _read_log(current.run_dir / "train.log")
    best = _read_json(current.run_dir / "metrics_best.json")
    evaluation = _read_json(current.run_dir / "evaluation_val.json")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    if rows:
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row.get("train_loss", np.nan) for row in rows], color=PREDICTED_COLOR, label="Train")
        axes[0].plot(epochs, [row.get("objective_loss", np.nan) for row in rows], color=TRUE_COLOR, label="Validation")
        primary_metric = str(best.get("primary_metric", rows[-1].get("primary_metric", "objective_loss")))
        logged_values = [
            (float(row["primary_value"]), int(row["epoch"]))
            for row in rows
            if row.get("primary_value") is not None and np.isfinite(float(row["primary_value"]))
        ]
        if logged_values:
            # train.log stores one-based epoch numbers; selection records use zero-based epochs.
            checkpoint_epoch = (min if primary_metric == "objective_loss" else max)(logged_values)[1]
        else:
            checkpoint_epoch = int(best.get("best_epoch", -1)) + 1
        axes[0].axvline(checkpoint_epoch, color="#444444", linestyle="--", linewidth=1, label="Best checkpoint")
        axes[0].set(xlabel="Epoch", ylabel="Objective loss", title="Optimization trace")
        axes[0].grid(alpha=0.22)
        axes[0].legend(frameon=False)
    else:
        axes[0].text(0.5, 0.5, "No epoch log available", ha="center", va="center")
        axes[0].set_axis_off()
    y = 0.96
    for heading, content in _overview_lines(current, best, evaluation):
        chinese = heading == "目的"
        axes[1].text(
            0.04, y, heading, ha="left", va="top", fontsize=8, fontweight="bold", color="#555555",
            fontproperties=CHINESE_HEADING_FONT if chinese else None,
        )
        y -= 0.045
        wrapped = textwrap.fill(content, width=43, break_long_words=False)
        axes[1].text(
            0.04, y, wrapped, ha="left", va="top", fontsize=9, linespacing=1.25,
            fontproperties=CHINESE_BODY_FONT if chinese else None,
        )
        y -= 0.048 * (wrapped.count("\n") + 1) + 0.020
    axes[1].set(title="Experiment summary", xticks=[], yticks=[])
    path = destination / f"{current.run_id}_val_overview.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _audit_summary(run: ManagedRun) -> dict[str, Any]:
    return _read_json(run.run_dir / "dynamic_audit_val" / "summary.json")


def _metric_median(summary: dict[str, Any], experiment_id: str, scope: str, metric: str) -> float:
    return float(summary["methods"][experiment_id][scope][metric]["median"])


def plot_dynamic_calibration(current: ManagedRun, compared: list[ManagedRun], destination: Path) -> Path:
    audits = {run.experiment_id: _audit_summary(run) for run in compared}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axis, metric in zip(axes.ravel(), SUMMARY_METRICS):
        values = [_metric_median(audits[run.experiment_id], run.experiment_id, "nonoverlap", metric) for run in compared]
        colors = [BASELINE_COLOR] * (len(compared) - 1) + [PREDICTED_COLOR]
        positions = np.arange(len(compared), dtype=float)
        axis.bar(
            positions,
            values,
            width=CALIBRATION_BAR_WIDTH,
            color=colors,
            align="center",
        )
        axis.set_xlim(-0.5, len(compared) - 0.5)
        axis.set_xticks(positions, [run.experiment_id for run in compared])
        if metric != "difference_temporal_pearson":
            axis.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
        axis.set(title=metric.replace("_", " "), ylabel="Median (non-overlap)")
        axis.grid(axis="y", alpha=0.22)
    axes.ravel()[-1].set_axis_off()
    figure.suptitle(f"{current.experiment_id} · dynamic calibration · validation", fontweight="semibold")
    path = destination / f"{current.run_id}_val_dynamic_calibration.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_horizon_audit(current: ManagedRun, compared: list[ManagedRun], destination: Path) -> Path:
    audits = {run.experiment_id: _audit_summary(run) for run in compared}
    segments = ["overlap_context", "early_long", "middle_long", "late_long"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True)
    for axis, metric in zip(axes.ravel(), HORIZON_METRICS):
        for index, run in enumerate(compared):
            values = [float(audits[run.experiment_id]["horizon_methods"][run.experiment_id][segment][metric]["median"]) for segment in segments]
            axis.plot(segments, values, marker="o", linewidth=1.6, color=BASELINE_COLOR if index < len(compared) - 1 else PREDICTED_COLOR, label=run.experiment_id)
        if metric != "difference_temporal_pearson":
            axis.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
        axis.set(title=metric.replace("_", " "), ylabel="Median")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"{current.experiment_id} · horizon audit · validation", fontweight="semibold")
    path = destination / f"{current.run_id}_val_horizon_audit.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _wrapper_text(experiment_id: str, run_id: str) -> str:
    return f'''"""Generate the static validation report for {experiment_id}/{run_id}.

Inputs: config_resolved.yaml, metrics_best.json, evaluation_val.json, and
dynamic_audit_val/summary.json from this managed run.
Outputs: four validation PNG figures plus a visual-manifest JSON file.
"""\nfrom pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[3]\nsys.path.insert(0, str(ROOT / "src"))\nfrom scdfc.visualization import generate_report\n\nif __name__ == "__main__":\n    generate_report(ROOT, "{experiment_id}", "{run_id}")\n'''


def generate_report(root: str | Path, experiment_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Generate the four-image validation report for one completed experiment."""
    root = Path(root).resolve()
    current = _load_run(root, experiment_id, run_id)
    baseline = _baseline_run(root, current)
    compared = [run for run in (baseline, current) if run is not None]
    destination = root / "outputs" / experiment_id / "visual"
    destination.mkdir(parents=True, exist_ok=True)
    overview = plot_run_overview(current, destination)
    dynamics, dynamic_details = plot_fixed_edge_dynamics(root, current, compared, destination)
    calibration = plot_dynamic_calibration(current, compared, destination)
    horizons = plot_horizon_audit(current, compared, destination)
    wrapper = destination / "generate_report.py"
    wrapper.write_text(_wrapper_text(current.experiment_id, current.run_id), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": current.experiment_id,
        "run_id": current.run_id,
        "split": "val",
        "compared_runs": [{"experiment_id": run.experiment_id, "run_id": run.run_id} for run in compared],
        "sources": {
            "config_resolved": str(current.run_dir / "config_resolved.yaml"),
            "metrics_best": str(current.run_dir / "metrics_best.json"),
            "evaluation": str(current.run_dir / "evaluation_val.json"),
            "dynamic_audit": str(current.run_dir / "dynamic_audit_val" / "summary.json"),
        },
        "fixed_edge_selection": {"seed": FIXED_EDGE_SEED, "count": FIXED_EDGE_COUNT, **dynamic_details},
        "images": [path.name for path in (overview, dynamics, calibration, horizons)],
    }
    manifest_path = destination / f"{current.run_id}_val_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the static four-figure validation report for one experiment")
    parser.add_argument("--root", default=".")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(json.dumps(generate_report(args.root, args.experiment, args.run_id), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
