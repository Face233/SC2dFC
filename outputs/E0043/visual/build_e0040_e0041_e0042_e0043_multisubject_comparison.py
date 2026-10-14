"""E0040/E0041/E0042/E0043 validation comparison, seed 42.

Run IDs: see RUNS below. Inputs: each run's frozen config_resolved.yaml,
metadata.json, evaluation_val.json, checkpoints/best.pt; the shared training
history statistics; the E0037 fixed subject/edge selection manifest.
Output: E0040_E0041_E0042_E0043_multisubject_comparison.png and JSON manifest.
"""
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
from scdfc.history_forecast import paired_interval
from scdfc.management import config_sha256, file_sha256, verify_artifact


RUNS = {
    "E0040": "E0040-s42-20261014T030716Z-214c72a",
    "E0041": "E0041-s42-20261014T073910Z-286c11d",
    "E0042": "E0042-s42-20261014T080138Z-64d8770",
    "E0043": "E0043-s42-20261014T082555Z-32b1b87",
}
COLORS = {
    "target": "#1f77b4",
    "E0040": "#d62728",
    "E0041": "#2ca02c",
    "E0042": "#9467bd",
    "E0043": "#ff7f0e",
    "group": "#7f7f7f",
    "boundary": "#666666",
}
OUT_DIR = Path(__file__).resolve().parent
REFERENCE = ROOT / "outputs" / "E0037" / "visual" / "E0032_E0036_E0037_multisubject_comparison.json"
STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "history_training_stats.npz"
IMAGE = OUT_DIR / "E0040_E0041_E0042_E0043_multisubject_comparison.png"
MANIFEST = OUT_DIR / "E0040_E0041_E0042_E0043_multisubject_comparison.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(
        label.replace(" ↔ ", " <-> ").replace("_", " "),
        width=25, max_lines=2, placeholder="…",
    )
    return f"edge {index}\n{name}"


def load_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    subjects = [str(value) for value in reference["subjects"]]
    edges = np.asarray(reference["edge_indices"], dtype=int)
    labels = reference["edge_labels"]
    if len(subjects) != 4 or len(edges) != 6 or len(labels) != 6:
        raise ValueError("The E0037 layout requires four fixed subjects and six edges")
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
    checkpoint = run_dir / "checkpoints" / "best.pt"
    report = json.loads((run_dir / "evaluation_val.json").read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(checkpoint)
    if report.get("split") != "val" or report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{experiment} validation report does not match its best checkpoint")
    artifact = verify_artifact(config)
    model, _ = _load_model(
        config, int(config["data"]["window_length"]), checkpoint, STATS, DEVICE, artifact
    )
    dataset = DFCSequenceDataset(
        config, int(config["data"]["window_length"]), "val", STATS, "fc1_only"
    )
    return run_dir, config, model, dataset, report, checkpoint_sha


@torch.no_grad()
def collect_predictions(subjects):
    predictions = {subject: {} for subject in subjects}
    targets = {}
    reference_starts = None
    group_template = None
    configs, reports, checkpoint_hashes = {}, {}, {}

    for experiment, run_id in RUNS.items():
        run_dir, config, model, dataset, report, checkpoint_sha = load_run(experiment, run_id)
        configs[experiment] = config
        reports[experiment] = report
        checkpoint_hashes[experiment] = checkpoint_sha
        for subject in subjects:
            matches = [i for i, (sid, _) in enumerate(dataset.samples) if str(sid) == subject]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one validation sample for {subject} in {experiment}")
            sample = dataset[matches[0]]
            target = sample["fc_future"].numpy()
            starts = sample["window_starts"].numpy()
            if reference_starts is None:
                reference_starts = starts
            elif not np.array_equal(reference_starts, starts):
                raise ValueError(f"Forecast timeline differs for {subject} in {experiment}")
            if subject in targets and not np.array_equal(targets[subject], target):
                raise ValueError(f"Forecast targets differ for {subject} in {experiment}")
            targets[subject] = target
            prediction = model(
                sample["sc_matrix"][None].to(DEVICE),
                sample["sc_edges"][None].to(DEVICE),
                sample["fc_warmup"][None].to(DEVICE),
                steps=len(target),
            ).fc_z_edges[0].cpu().numpy()
            if prediction.shape != target.shape or len(target) != int(config["data"]["forecast_windows"]):
                raise ValueError(f"Prediction shape differs from target for {subject} in {experiment}")
            predictions[subject][experiment] = prediction

        current_template = model.group_template.cpu().numpy()
        if group_template is None:
            group_template = current_template
        elif not np.array_equal(group_template, current_template):
            raise ValueError(f"{experiment} uses a different training group template")
        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    reference_config = configs["E0040"]
    reference_data = {key: value for key, value in reference_config["data"].items()
                      if key != "warmup_windows"}
    reference_manifest = reports["E0040"]["sample_manifest_sha256"]
    for experiment, expected_k in (("E0040", 1), ("E0041", 5), ("E0042", 10), ("E0043", 20)):
        config = configs[experiment]
        data = {key: value for key, value in config["data"].items()
                if key != "warmup_windows"}
        if (data != reference_data or config["training"] != reference_config["training"]
                or config["model"] != reference_config["model"]
                or config["artifacts"] != reference_config["artifacts"]):
            raise ValueError(f"{experiment} does not match the common forecast protocol")
        if config["data"]["warmup_windows"] != expected_k:
            raise ValueError(f"Unexpected history length in {experiment}")
        report = reports[experiment]
        if (report["sample_manifest_sha256"] != reference_manifest
                or report["forecast_origin"] != 19 or report["nonoverlap_start"] != 17):
            raise ValueError(f"{experiment} validation report uses a different sample or horizon")
    return predictions, targets, group_template, reference_starts, reports, checkpoint_hashes


def paired_comparisons(reports):
    """Compare K=20 to K=1, K=5, K=10 on the same validation subjects."""
    metrics = ("forecast_nonoverlap_mse", "forecast_nonoverlap_time_r",
               "forecast_nonoverlap_adjusted_time_r")
    main = reports["E0043"]["per_subject"]
    subjects = [row["subject_id"] for row in main]
    comparisons = {}
    for baseline in ("E0040", "E0041", "E0042"):
        other = {row["subject_id"]: row for row in reports[baseline]["per_subject"]}
        if set(other) != set(subjects) or len(other) != len(subjects):
            raise ValueError(f"{baseline} does not contain the same validation subjects")
        comparisons[baseline] = {
            metric: paired_interval(
                [row.get(metric) for row in main],
                [other[subject].get(metric) for subject in subjects],
                subjects, minimize=metric.endswith("mse"), replicates=10000,
            ) for metric in metrics
        }
    return comparisons


def render(subjects, edges, labels, predictions, targets, group, minutes, nonoverlap_minute, reports):
    relevant = [
        array[:, edges]
        for subject in subjects
        for array in (targets[subject], *(predictions[subject][experiment] for experiment in RUNS))
    ]
    relevant.append(group[:, edges])
    all_values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = all_values.min(axis=0), all_values.max(axis=0)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(18, 19), layout="constrained")
    header, body, footer = figure.subfigures(3, 1, height_ratios=(0.105, 0.82, 0.075))
    header.text(
        0.5, 0.70, "K=1 / 5 / 10 / 20 · E0040–E0043 验证集多被试预测对比 · seed 42",
        ha="center", va="center", fontsize=14, fontproperties=HEADING_FONT,
    )
    header.legend(
        [
            Line2D([], [], color=COLORS["target"], linewidth=1.8),
            Line2D([], [], color=COLORS["E0040"], linewidth=1.5),
            Line2D([], [], color=COLORS["E0041"], linewidth=1.5),
            Line2D([], [], color=COLORS["E0042"], linewidth=1.5, linestyle="--"),
            Line2D([], [], color=COLORS["E0043"], linewidth=1.7, linestyle="-."),
            Line2D([], [], color=COLORS["group"], linewidth=1.4),
            Line2D([], [], color=COLORS["boundary"], linewidth=0.9, linestyle=":"),
        ],
        ["真实边", "E0040 (K=1)", "E0041 (K=5)", "E0042 (K=10)", "E0043 (K=20)", "训练集组平均（第五列）", "保守无重叠段起点"],
        loc="lower center", ncol=4, frameon=False, prop=BODY_FONT,
    )

    axes = body.subplots(len(edges), len(subjects) + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.axvspan(minutes[0], nonoverlap_minute, color="#f2f2f2", zorder=0)
            axis.axvline(nonoverlap_minute, color=COLORS["boundary"], linewidth=0.8, linestyle=":", alpha=0.8)
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"], linewidth=1.1, zorder=4)
            for experiment in RUNS:
                axis.plot(minutes, predictions[subject][experiment][:, edge], color=COLORS[experiment],
                          linewidth=1.2 if experiment == "E0043" else 0.95,
                          linestyle={"E0042": "--", "E0043": "-."}.get(experiment, "-"),
                          alpha=0.92, zorder=6 if experiment == "E0043" else 5)
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.set_xlim(minutes[0], minutes[-1])
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"验证集\n被试 {subject}", fontsize=10, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{edge_title(int(edge), label)}\nFisher-z", fontsize=8.5, fontproperties=BODY_FONT)
            if row == len(edges) - 1:
                axis.set_xlabel("首个目标窗后分钟", fontsize=9, fontproperties=BODY_FONT)

        axis = axes[row, -1]
        axis.axvspan(minutes[0], nonoverlap_minute, color="#f2f2f2", zorder=0)
        axis.axvline(nonoverlap_minute, color=COLORS["boundary"], linewidth=0.8, linestyle=":", alpha=0.8)
        axis.plot(minutes, group[:, edge], color=COLORS["group"], linewidth=1.2)
        axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        axis.set_xlim(minutes[0], minutes[-1])
        axis.grid(alpha=0.22, linewidth=0.55)
        axis.tick_params(labelsize=8)
        if row == 0:
            axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == len(edges) - 1:
            axis.set_xlabel("首个目标窗后分钟", fontsize=9, fontproperties=BODY_FONT)

    mse_values = [reports[experiment]["aggregate"]["forecast_nonoverlap_mse"] for experiment in RUNS]
    time_values = [reports[experiment]["aggregate"]["forecast_nonoverlap_time_r"] for experiment in RUNS]
    footer.text(
        0.5, 0.78,
        "val only · 沿用 E0037 的四名被试和六条边 · 四组共同截止于 FC[19]，预测目标均为 FC[20:]",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    footer.text(
        0.5, 0.48,
        "灰色背景为近端区间；竖虚线为无重叠段起点 · "
        "K1/K5/K10/K20 无重叠 MSE=" + "/".join(f"{v:.4f}" for v in mse_values)
        + "，时间 r=" + "/".join(f"{v:.4f}" for v in time_values),
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    footer.text(0.5, 0.15, "E0043 run: " + RUNS["E0043"] + " · 各运行与检查点哈希见 JSON",
                ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT)
    figure.savefig(IMAGE, dpi=180)
    plt.close(figure)


def main() -> None:
    subjects, edges, labels = load_reference()
    predictions, targets, group, starts, reports, checkpoint_hashes = collect_predictions(subjects)
    comparisons = paired_comparisons(reports)
    config = load_config(ROOT / "outputs" / "E0043" / "runs" / RUNS["E0043"] / "config_resolved.yaml")
    tr_seconds = float(config["data"]["tr_seconds"])
    minutes = (starts - starts[0]) * tr_seconds / 60.0
    nonoverlap = int(config["evaluation"]["nonoverlap_start"])
    render(subjects, edges, labels, predictions, targets, group, minutes, minutes[nonoverlap], reports)
    manifest = {
        "experiments": RUNS,
        "checkpoint_sha256": checkpoint_hashes,
        "split": "val",
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "reference_selection": str(REFERENCE),
        "forecast_origin": int(config["data"]["forecast_origins"][0]),
        "forecast_windows": int(config["data"]["forecast_windows"]),
        "nonoverlap_start": nonoverlap,
        "nonoverlap_minute": float(minutes[nonoverlap]),
        "colors": COLORS,
        "aggregate": {experiment: {
            metric: report["aggregate"][metric]
            for metric in ("forecast_nonoverlap_mse", "forecast_nonoverlap_time_r",
                           "forecast_nonoverlap_adjusted_time_r", "forecast_nonoverlap_space_r",
                           "forecast_nonoverlap_std_ratio", "forecast_nonoverlap_difference_time_r")
        } for experiment, report in reports.items()},
        "paired_e0043_minus_baseline": comparisons,
        "image": str(IMAGE),
        "note": "All four runs predict identical FC[20:] targets at a common cutoff; only K differs.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(IMAGE), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
