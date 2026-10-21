"""Render E0045/E0046/E0047 validation curves in the E0037 layout."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset
from scdfc.evaluation import _load_model
from scdfc.management import config_sha256, file_sha256, verify_artifact


RUNS = {
    "E0045": "E0045-s42-20261021T033047Z-19f52fe",
    "E0046": "E0046-s42-20261021T043331Z-e26fccd",
    "E0047": "E0047-s42-20261021T045601Z-2a314e6",
}
REFERENCE = ROOT / "outputs" / "E0037" / "visual" / "E0032_E0036_E0037_multisubject_comparison.json"
STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "history_training_stats.npz"
OUT_DIR = Path(__file__).resolve().parent
IMAGE = OUT_DIR / "E0045_E0046_E0047_multisubject_comparison.png"
MANIFEST = OUT_DIR / "E0045_E0046_E0047_multisubject_comparison.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORECAST_ORIGIN = 9
FORECAST_WINDOWS = 214
COLORS = {
    "target": "#1f77b4",
    "E0045": "#9467bd",
    "E0046": "#2ca02c",
    "E0047": "#d62728",
    "group": "#7f7f7f",
}
LINESTYLES = {experiment: "-" for experiment in RUNS}
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(
        label.replace(" ↔ ", " <-> ").replace("_", " "),
        width=25,
        max_lines=2,
        placeholder="…",
    )
    return f"edge {index}\n{name}"


def load_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    subjects = [str(value) for value in reference["subjects"]]
    edges = np.asarray(reference["edge_indices"], dtype=int)
    labels = reference["edge_labels"]
    if len(subjects) != 4 or len(edges) != 6 or len(labels) != 6:
        raise ValueError("E0037 reference must contain four subjects and six edges")
    return subjects, edges, labels


def load_run(experiment: str, run_id: str):
    run_dir = ROOT / "outputs" / experiment / "runs" / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED" or metadata.get("run_id") != run_id:
        raise ValueError(f"{experiment} run is not completed")

    config = load_config(run_dir / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise ValueError(f"{experiment} resolved config hash differs from metadata")
    config["paths"]["root"] = str(ROOT)
    data_config = config["data"]
    if data_config.get("forecast_origins") != [FORECAST_ORIGIN]:
        raise ValueError(f"Unexpected forecast origin in {experiment}: {data_config.get('forecast_origins')}")
    if int(data_config.get("forecast_windows", -1)) != FORECAST_WINDOWS:
        raise ValueError(f"Unexpected forecast length in {experiment}")
    if config.get("experiment", {}).get("ablation") != "fc1_only":
        raise ValueError(f"Expected fc1_only ablation in {experiment}")

    report = json.loads((run_dir / "evaluation_val.json").read_text(encoding="utf-8"))
    if report.get("split") != "val":
        raise ValueError(f"{experiment} report is not a validation report")
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint_hash = file_sha256(checkpoint)
    if report.get("checkpoint_sha256") not in (None, checkpoint_hash):
        raise ValueError(f"{experiment} validation report does not match its best checkpoint")

    artifact = verify_artifact(config)
    model, _payload = _load_model(
        config,
        int(data_config["window_length"]),
        checkpoint,
        STATS,
        DEVICE,
        artifact,
    )
    dataset = DFCSequenceDataset(
        config,
        int(data_config["window_length"]),
        "val",
        STATS,
        "fc1_only",
    )
    return run_dir, config, model, dataset, checkpoint_hash


@torch.no_grad()
def collect_predictions(subjects):
    predictions = {subject: {} for subject in subjects}
    targets = {}
    reference_starts = None
    checkpoint_hashes = {}
    tr_seconds = None

    for experiment, run_id in RUNS.items():
        _run_dir, config, model, dataset, checkpoint_hash = load_run(experiment, run_id)
        checkpoint_hashes[experiment] = checkpoint_hash
        current_tr = float(config["data"]["tr_seconds"])
        if tr_seconds is None:
            tr_seconds = current_tr
        elif tr_seconds != current_tr:
            raise ValueError(f"TR differs in {experiment}")

        for subject in subjects:
            matches = [i for i, (sid, _run) in enumerate(dataset.samples) if str(sid) == subject]
            if len(matches) != 1:
                raise ValueError(f"Expected one validation sample for {subject} in {experiment}; found {len(matches)}")
            sample = dataset[matches[0]]
            target = sample["fc_future"].numpy()
            starts = sample["window_starts"].numpy()
            if len(target) != FORECAST_WINDOWS or len(starts) != FORECAST_WINDOWS:
                raise ValueError(f"Unexpected validation horizon for {subject} in {experiment}")
            if reference_starts is None:
                reference_starts = starts
            elif not np.array_equal(reference_starts, starts):
                raise ValueError(f"Forecast timeline differs for {subject} in {experiment}")
            if subject in targets and not np.array_equal(targets[subject], target):
                raise ValueError(f"Validation targets differ for {subject} in {experiment}")
            targets[subject] = target

            result = model(
                sample["sc_matrix"][None].to(DEVICE),
                sample["sc_edges"][None].to(DEVICE),
                sample["fc_warmup"][None].to(DEVICE),
                steps=len(target),
            )
            prediction = result.fc_z_edges[0].cpu().numpy()
            if prediction.shape != target.shape:
                raise ValueError(
                    f"{experiment} prediction/target mismatch for {subject}: "
                    f"{prediction.shape} vs {target.shape}"
                )
            predictions[subject][experiment] = prediction

        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if reference_starts is None or tr_seconds is None:
        raise RuntimeError("No validation series were loaded")
    return predictions, targets, reference_starts, tr_seconds, checkpoint_hashes


def render(subjects, edges, edge_labels, predictions, targets, group, minutes):
    relevant = [
        series[:, edges]
        for subject in subjects
        for series in (targets[subject], *(predictions[subject][exp] for exp in RUNS))
    ]
    relevant.append(group[:, edges])
    values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = values.min(axis=0), values.max(axis=0)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body, footer = figure.subfigures(3, 1, height_ratios=(0.085, 0.88, 0.035))
    header.text(
        0.5,
        0.70,
        "E0045 / E0046 / E0047 · 验证集多被试时序对比 · 相同六条边",
        ha="center",
        va="center",
        fontsize=15,
        fontproperties=HEADING_FONT,
    )
    handles = [Line2D([], [], color=COLORS["target"], linewidth=1.8)]
    legend_labels = ["真实边"]
    for experiment in RUNS:
        handles.append(Line2D([], [], color=COLORS[experiment], linewidth=1.6,
                              linestyle=LINESTYLES[experiment]))
        legend_labels.append(f"{experiment}预测")
    handles.append(Line2D([], [], color=COLORS["group"], linewidth=1.4))
    legend_labels.append("训练集组平均（第五列）")
    header.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False, prop=BODY_FONT)

    axes = body.subplots(len(edges), len(subjects) + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, edge_labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"], linewidth=1.15, zorder=4)
            for experiment in RUNS:
                axis.plot(
                    minutes,
                    predictions[subject][experiment][:, edge],
                    color=COLORS[experiment],
                    linewidth=0.95,
                    alpha=0.9,
                    linestyle=LINESTYLES[experiment],
                )
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"验证集\n被试 {subject}", fontsize=10, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(
                    f"{edge_title(int(edge), label)}\nFisher-z",
                    fontsize=8.5,
                    fontproperties=BODY_FONT,
                )
            if row == len(edges) - 1:
                axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

        group_axis = axes[row, -1]
        group_axis.plot(minutes, group[:, edge], color=COLORS["group"], linewidth=1.2)
        group_axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        group_axis.grid(alpha=0.22, linewidth=0.55)
        group_axis.tick_params(labelsize=8)
        if row == 0:
            group_axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == len(edges) - 1:
            group_axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

    footer.text(
        0.5,
        0.5,
        "val only · 四名被试和六条边沿用 E0037 图的选择 · 三组预测共享真实目标 FC[10:] · 第五列为训练集组平均真实时序",
        ha="center",
        fontsize=8,
        color="#555555",
        fontproperties=BODY_FONT,
    )
    figure.savefig(IMAGE, dpi=180)
    plt.close(figure)


def main() -> None:
    if not STATS.exists():
        raise FileNotFoundError(f"Training statistics are missing: {STATS}")
    subjects, edges, edge_labels = load_reference()
    predictions, targets, starts, tr_seconds, checkpoint_hashes = collect_predictions(subjects)
    with np.load(STATS) as stats_file:
        template = stats_file["group_template"]
    group = template[FORECAST_ORIGIN:]
    if len(group) != len(starts):
        raise ValueError("Training group template does not align with the FC[10:] validation timeline")
    minutes = (starts - starts[0]) * tr_seconds / 60.0
    render(subjects, edges, edge_labels, predictions, targets, group, minutes)

    manifest = {
        "experiments": RUNS,
        "checkpoint_sha256": checkpoint_hashes,
        "split": "val",
        "seed": 42,
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": edge_labels,
        "reference_selection": str(REFERENCE),
        "common_target": "FC[10:]",
        "common_forecast_origin_zero_based": FORECAST_ORIGIN,
        "common_forecast_windows": int(len(starts)),
        "colors": COLORS,
        "linestyles": LINESTYLES,
        "layout": {
            "rows": "six fixed edges with shared per-edge y ranges",
            "columns": "four fixed validation subjects plus training-set group mean",
            "style_reference": "E0037",
        },
        "metric_scope": "Curves show four fixed validation subjects and six selected edges; no test data are used.",
        "image": str(IMAGE),
        "note": (
            "E0045, E0046, and E0047 all use zero-based forecast origin 9 and predict the same FC[10:] target. "
            "Their warmup lengths differ (K=1, K=5, and K=10), as configured by each experiment."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(IMAGE), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
