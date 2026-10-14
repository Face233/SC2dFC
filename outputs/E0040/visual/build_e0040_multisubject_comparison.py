"""Plot E0040 validation forecasts in the E0037 six-edge, four-subject layout."""
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


RUN_ID = "E0040-s42-20261014T030716Z-214c72a"
RUN_DIR = ROOT / "outputs" / "E0040" / "runs" / RUN_ID
OUT_DIR = Path(__file__).resolve().parent
REFERENCE = (
    ROOT / "outputs" / "E0037" / "visual"
    / "E0032_E0036_E0037_multisubject_comparison.json"
)
STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "history_training_stats.npz"
IMAGE = OUT_DIR / "E0040_multisubject_forecast_comparison.png"
MANIFEST = OUT_DIR / "E0040_multisubject_forecast_comparison.json"
COLORS = {"target": "#1f77b4", "prediction": "#d62728", "group": "#7f7f7f"}
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(
        label.replace(" ↔ ", " <-> ").replace("_", " "),
        width=25, max_lines=2, placeholder="…",
    )
    return f"edge {index}\n{name}"


def load_sources():
    metadata = json.loads((RUN_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "COMPLETED" or metadata["run_id"] != RUN_ID:
        raise ValueError("The requested E0040 run has not completed")
    config = load_config(RUN_DIR / "config_resolved.yaml")
    if config_sha256(config) != metadata["config_sha256"]:
        raise ValueError("Frozen run configuration differs from its metadata")
    config["paths"]["root"] = str(ROOT)
    checkpoint = RUN_DIR / "checkpoints" / "best.pt"
    report = json.loads((RUN_DIR / "evaluation_val.json").read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(checkpoint)
    if report["split"] != "val" or report["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Validation report does not match the best checkpoint")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    subjects = [str(value) for value in reference["subjects"]]
    edges = np.asarray(reference["edge_indices"], dtype=int)
    labels = reference["edge_labels"]
    if len(subjects) != 4 or len(edges) != 6 or len(labels) != 6:
        raise ValueError("E0037 figure must specify four subjects and six edges")
    return config, checkpoint, checkpoint_sha, subjects, edges, labels


@torch.no_grad()
def collect_selected(config, checkpoint, subjects):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = verify_artifact(config)
    model, _ = _load_model(
        config, int(config["data"]["window_length"]), checkpoint, STATS, device, artifact
    )
    dataset = DFCSequenceDataset(
        config, int(config["data"]["window_length"]), "val", STATS, "fc1_only"
    )
    predictions, targets = {}, {}
    reference_starts = None
    for subject in subjects:
        matches = [i for i, (sid, _) in enumerate(dataset.samples) if str(sid) == subject]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one validation run for subject {subject}")
        sample = dataset[matches[0]]
        target = sample["fc_future"].numpy()
        starts = sample["window_starts"].numpy()
        if reference_starts is None:
            reference_starts = starts
        elif not np.array_equal(reference_starts, starts):
            raise ValueError("Selected validation targets do not share the same timeline")
        output = model(
            sample["sc_matrix"][None].to(device),
            sample["sc_edges"][None].to(device),
            sample["fc_warmup"][None].to(device),
            steps=len(target),
        ).fc_z_edges[0].cpu().numpy()
        if output.shape != target.shape or output.shape[0] != int(config["data"]["forecast_windows"]):
            raise ValueError(f"Forecast shape differs from target for subject {subject}")
        targets[subject] = target
        predictions[subject] = output
    group = model.group_template.cpu().numpy()
    if group.shape != targets[subjects[0]].shape:
        raise ValueError("Training group template is not aligned with E0040 targets")
    return predictions, targets, group, reference_starts


def render(subjects, edges, labels, predictions, targets, group, minutes, nonoverlap_minute):
    relevant = [
        array[:, edges]
        for subject in subjects
        for array in (targets[subject], predictions[subject])
    ]
    relevant.append(group[:, edges])
    values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = values.min(axis=0), values.max(axis=0)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body, footer = figure.subfigures(3, 1, height_ratios=(0.085, 0.86, 0.055))
    header.text(
        0.5, 0.70, "E0040 (K=1) · 验证集多被试预测与真实曲线 · 相同六条边",
        ha="center", va="center", fontsize=15, fontproperties=HEADING_FONT,
    )
    header.legend(
        [
            Line2D([], [], color=COLORS["target"], linewidth=1.8),
            Line2D([], [], color=COLORS["prediction"], linewidth=1.6),
            Line2D([], [], color=COLORS["group"], linewidth=1.4),
            Line2D([], [], color="#666666", linewidth=1, linestyle=":"),
        ],
        ["真实边", "E0040预测", "训练集组平均（第五列）", "保守无重叠段起点"],
        loc="lower center", ncol=4, frameon=False, prop=BODY_FONT,
    )

    axes = body.subplots(len(edges), len(subjects) + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.axvspan(minutes[0], nonoverlap_minute, color="#f2f2f2", zorder=0)
            axis.axvline(nonoverlap_minute, color="#666666", linewidth=0.8, linestyle=":", alpha=0.8)
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"], linewidth=1.15, zorder=4)
            axis.plot(minutes, predictions[subject][:, edge], color=COLORS["prediction"], linewidth=1.05, zorder=5)
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
        axis.axvline(nonoverlap_minute, color="#666666", linewidth=0.8, linestyle=":", alpha=0.8)
        axis.plot(minutes, group[:, edge], color=COLORS["group"], linewidth=1.2)
        axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        axis.set_xlim(minutes[0], minutes[-1])
        axis.grid(alpha=0.22, linewidth=0.55)
        axis.tick_params(labelsize=8)
        if row == 0:
            axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == len(edges) - 1:
            axis.set_xlabel("首个目标窗后分钟", fontsize=9, fontproperties=BODY_FONT)

    footer.text(
        0.5, 0.67,
        "val only · 四名被试和六条边沿用 E0037 的固定选择 · 0 分钟为 FC[20]，最后观测窗为 FC[19]",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    footer.text(
        0.5, 0.20,
        "浅灰区包含与最后观测窗重叠的近端目标；虚线之后为预先定义的保守无重叠段 · 第五列仅作训练集模板参照",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    figure.savefig(IMAGE, dpi=180)
    plt.close(figure)


def main() -> None:
    config, checkpoint, checkpoint_sha, subjects, edges, labels = load_sources()
    predictions, targets, group, starts = collect_selected(config, checkpoint, subjects)
    tr_seconds = float(config["data"]["tr_seconds"])
    minutes = (starts - starts[0]) * tr_seconds / 60.0
    nonoverlap = int(config["evaluation"]["nonoverlap_start"])
    render(subjects, edges, labels, predictions, targets, group, minutes, minutes[nonoverlap])
    manifest = {
        "run_id": RUN_ID,
        "checkpoint_sha256": checkpoint_sha,
        "split": "val",
        "source_evaluation": str(RUN_DIR / "evaluation_val.json"),
        "reference_selection": str(REFERENCE),
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "forecast_origin": int(config["data"]["forecast_origins"][0]),
        "forecast_windows": int(config["data"]["forecast_windows"]),
        "nonoverlap_start": nonoverlap,
        "nonoverlap_minute": float(minutes[nonoverlap]),
        "colors": COLORS,
        "image": str(IMAGE),
        "note": "Four fixed validation subjects; E0040's FC[20:] target and aligned training-only template.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(IMAGE), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
