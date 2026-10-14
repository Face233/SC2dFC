"""Plot E0040 and E0041 forecasts in the E0037 six-edge layout."""
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
    "E0040": "E0040-s42-20261014T030716Z-214c72a",
    "E0041": "E0041-s42-20261014T073910Z-286c11d",
}
COLORS = {
    "target": "#1f77b4",
    "E0040": "#d62728",
    "E0041": "#2ca02c",
    "group": "#7f7f7f",
    "boundary": "#666666",
}
OUT_DIR = Path(__file__).resolve().parent
REFERENCE = ROOT / "outputs" / "E0037" / "visual" / "E0032_E0036_E0037_multisubject_comparison.json"
STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "history_training_stats.npz"
IMAGE = OUT_DIR / "E0040_E0041_multisubject_comparison.png"
MANIFEST = OUT_DIR / "E0040_E0041_multisubject_comparison.json"
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
                raise ValueError(f"E0040 and E0041 targets differ for {subject}")
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
            raise ValueError("E0040 and E0041 use different training group templates")
        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    e40, e41 = configs["E0040"], configs["E0041"]
    data40 = {key: value for key, value in e40["data"].items() if key != "warmup_windows"}
    data41 = {key: value for key, value in e41["data"].items() if key != "warmup_windows"}
    if data40 != data41 or e40["training"] != e41["training"] or e40["model"] != e41["model"]:
        raise ValueError("These forecasts are not matched except for history length")
    if e40["data"]["warmup_windows"] != 1 or e41["data"]["warmup_windows"] != 5:
        raise ValueError("Unexpected history lengths in the frozen configurations")
    return predictions, targets, group_template, reference_starts, reports, checkpoint_hashes


def render(subjects, edges, labels, predictions, targets, group, minutes, nonoverlap_minute, reports):
    relevant = [
        array[:, edges]
        for subject in subjects
        for array in (targets[subject], predictions[subject]["E0040"], predictions[subject]["E0041"])
    ]
    relevant.append(group[:, edges])
    all_values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = all_values.min(axis=0), all_values.max(axis=0)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body, footer = figure.subfigures(3, 1, height_ratios=(0.085, 0.86, 0.055))
    header.text(
        0.5, 0.70, "E0040 (K=1) / E0041 (K=5) · 验证集多被试预测对比 · 相同六条边",
        ha="center", va="center", fontsize=14, fontproperties=HEADING_FONT,
    )
    header.legend(
        [
            Line2D([], [], color=COLORS["target"], linewidth=1.8),
            Line2D([], [], color=COLORS["E0040"], linewidth=1.5),
            Line2D([], [], color=COLORS["E0041"], linewidth=1.5),
            Line2D([], [], color=COLORS["group"], linewidth=1.4),
            Line2D([], [], color=COLORS["boundary"], linewidth=0.9, linestyle=":"),
        ],
        ["真实边", "E0040 预测", "E0041 预测", "训练集组平均（第五列）", "保守无重叠段起点"],
        loc="lower center", ncol=5, frameon=False, prop=BODY_FONT,
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
                          linewidth=0.95, alpha=0.92, zorder=5)
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

    paired = reports["E0041"].get("paired_model_comparison", {})
    delta_time = paired.get("forecast_nonoverlap_time_r", {})
    delta_mse = paired.get("forecast_nonoverlap_mse", {})
    footer.text(
        0.5, 0.70,
        "val only · 沿用 E0037 的四名被试和六条边 · 两组共同截止于 FC[19]，预测目标均为 FC[20:]",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    footer.text(
        0.5, 0.25,
        f"灰色背景为近端区间；虚线为无重叠段起点 · E0041−E0040 无重叠时间 r={delta_time.get('mean_difference', float('nan')):+.4f} "
        f"(95% CI {delta_time.get('ci_low', float('nan')):+.4f}, {delta_time.get('ci_high', float('nan')):+.4f}) · "
        f"MSE差={delta_mse.get('mean_difference', float('nan')):+.4f}",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    figure.savefig(IMAGE, dpi=180)
    plt.close(figure)


def main() -> None:
    subjects, edges, labels = load_reference()
    predictions, targets, group, starts, reports, checkpoint_hashes = collect_predictions(subjects)
    config = load_config(ROOT / "outputs" / "E0041" / "runs" / RUNS["E0041"] / "config_resolved.yaml")
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
        "paired_nonoverlap_mse": reports["E0041"].get("paired_model_comparison", {}).get("forecast_nonoverlap_mse"),
        "paired_nonoverlap_time_r": reports["E0041"].get("paired_model_comparison", {}).get("forecast_nonoverlap_time_r"),
        "image": str(IMAGE),
        "note": "Both runs predict identical FC[20:] targets at a common cutoff; only K differs.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(IMAGE), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
