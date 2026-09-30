"""Build E0039 validation figures in the established E0032 time-series style.

Inputs: immutable E0032/E0036/E0038/E0039 run configs and checkpoints, the
training-only stats file, and the fixed subject/edge selection from E0029.
Outputs: a six-edge multi-subject curve comparison and metric comparison PNG.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scdfc.data import DFCSequenceDataset, group_template_for_warmup
from scdfc.management import verify_artifact
from scdfc.training import build_sequence_model

OUT_DIR = Path(__file__).resolve().parent
STATS_PATH = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "training_stats.npz"
RUNS = {
    "E0032": "E0032-s42-20260923T052241Z-8d7ca92",
    "E0036": "E0036-s42-20260929T082122Z-b5b871f",
    "E0038": "E0038-s42-20260929T094027Z-ad7ce6a",
    "E0039": "E0039-s42-20260930T075338Z-7762d90",
}
COLORS = {
    "target": "#1f77b4",
    "E0032": "#9467bd",
    "E0036": "#2ca02c",
    "E0038": "#ff7f0e",
    "E0039": "#d62728",
    "group": "#7f7f7f",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(label.replace(" ↔ ", " <-> ").replace("_", " "), width=25, max_lines=2, placeholder="…")
    return f"edge {index}\n{name}"


def load_model(config: dict, run_dir: Path):
    checkpoint = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model = build_sequence_model(
        config,
        int(config["data"]["window_length"]),
        payload["decoder_type"],
        STATS_PATH,
        DEVICE,
        payload.get("sc_encoder_type", "hybrid"),
        verify_artifact(config),
        payload,
    )
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"context_template"}:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
    model.eval()
    return model


def predict(experiment: str, model, config: dict, run_dir: Path, sample: dict, stats: dict) -> np.ndarray:
    target = sample["fc_future"].numpy()
    if experiment == "E0032":
        alpha = np.load(run_dir / "alpha_fit.npz")["alpha"].astype(np.float32)
        context = stats.get("context_template", stats["fc_mean"]).astype(np.float32)
        warmup = sample["fc_warmup"].numpy().astype(np.float32)
        group = group_template_for_warmup(stats, int(config["data"].get("warmup_windows", 1))).astype(np.float32)
        return group[: len(target)] + alpha[: len(target), None] * (warmup - context)
    with torch.no_grad():
        result = model(
            sample["sc_matrix"][None].to(DEVICE),
            sample["sc_edges"][None].to(DEVICE),
            sample["fc_warmup"][None].to(DEVICE),
            steps=target.shape[0],
        ).fc_z_edges[0].cpu().numpy()
    if result.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {experiment} {result.shape} != {target.shape}")
    return result


def make_curve_figure(subjects, edges, labels, predictions, targets, group, minutes) -> Path:
    names = list(RUNS)
    relevant = [array[:, edges] for subject in subjects for array in (targets[subject], *[predictions[subject][m] for m in names])]
    relevant.append(group[: len(minutes), edges])
    all_values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = all_values.min(axis=0), all_values.max(axis=0)

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(0.085, 0.915))
    header.text(0.5, 0.70, "E0032 / E0036 / E0038 / E0039 · 验证集多被试时序对比 · 相同六条边",
                ha="center", va="center", fontsize=15, fontproperties=HEADING_FONT)
    handles = [Line2D([], [], color=COLORS["target"], linewidth=1.8)]
    legend_labels = ["真实边"]
    for name in names:
        handles.append(Line2D([], [], color=COLORS[name], linewidth=1.6))
        legend_labels.append(f"{name}预测")
    handles.append(Line2D([], [], color=COLORS["group"], linewidth=1.4))
    legend_labels.append("训练集组平均（第五列）")
    header.legend(handles, legend_labels, loc="lower center", ncol=6, frameon=False, prop=BODY_FONT)

    axes = body.subplots(len(edges), len(subjects) + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"], linewidth=1.15, zorder=4)
            for name in names:
                axis.plot(minutes, predictions[subject][name][:, edge], color=COLORS[name], linewidth=0.95, alpha=0.9)
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"验证集\n被试 {subject}", fontsize=10, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{edge_title(int(edge), label)}\nFisher-z", fontsize=8.5, fontproperties=BODY_FONT)
            if row == len(edges) - 1:
                axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)
        axis = axes[row, -1]
        axis.plot(minutes, group[: len(minutes), edge], color=COLORS["group"], linewidth=1.2)
        axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        axis.grid(alpha=0.22, linewidth=0.55)
        axis.tick_params(labelsize=8)
        if row == 0:
            axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == len(edges) - 1:
            axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "E0032_E0036_E0038_E0039_multisubject_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def make_metric_figure() -> Path:
    ids = ["E0032", "E0036", "E0038", "E0039"]
    dynamic_ids = ["E0036", "E0038", "E0039"]
    reports = {}
    for experiment, run_id in RUNS.items():
        reports[experiment] = json.loads((ROOT / "outputs" / experiment / "runs" / run_id / "evaluation_val.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    panels = [
        (ids, "long_edge_mse", "长期边 MSE", "被试均值及 95% CI；较低较好"),
        (dynamic_ids, "dynamic_time_pearson", "动态时间 Pearson", "点为均值，线为被试 bootstrap 95% CI"),
        (dynamic_ids, "dynamic_std_ratio", "动态标准差比", "对数轴；1 表示幅度匹配"),
    ]
    for axis, (panel_ids, key, title, note) in zip(axes, panels):
        means, intervals = [], []
        for index, name in enumerate(panel_ids):
            if name == "E0032" and key == "long_edge_mse":
                # E0032's legacy report contains aggregate long MSE but no
                # per-subject long MSE values, so plot its aggregate without a CI.
                samples = np.asarray([reports[name]["aggregate"][key]], dtype=float)
            else:
                samples = np.asarray([row[key] for row in reports[name]["per_sample"]], dtype=float)
            rng = np.random.default_rng(42 + index)
            bootstrap = np.mean(samples[rng.integers(0, len(samples), size=(10000, len(samples)))], axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            means.append(float(samples.mean()))
            intervals.append((float(low), float(high)))
        x = np.arange(len(panel_ids))
        axis.set_title(title, fontproperties=HEADING_FONT, fontsize=12)
        axis.set_ylabel(note, fontproperties=BODY_FONT, fontsize=9)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xticks(x, panel_ids)
        axis.tick_params(axis="x", labelsize=9)
        if key == "dynamic_std_ratio":
            axis.set_yscale("log")
            axis.set_ylim(1e-5, 1.0)
        else:
            lower = min(low for low, _ in intervals)
            upper = max(high for _, high in intervals)
            padding = max((upper - lower) * 0.08, 1e-6)
            axis.set_ylim(lower - padding, upper + padding)
        for xi, name, value, (low, high) in zip(x, panel_ids, means, intervals):
            axis.errorbar(xi, value, yerr=[[value - low], [high - value]], fmt="o", color=COLORS[name],
                          ecolor=COLORS[name], capsize=4, markersize=6, linewidth=1.4)
            axis.annotate(f"{value:.4g}", (xi, value), xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=8)
    figure.suptitle("验证集关键指标：E0039 的动态幅度与时间对齐\nE0036/E0038/E0039 为被试 bootstrap 95% CI；E0032 长期 MSE 为历史聚合值且无动态指标", fontproperties=HEADING_FONT, fontsize=12)
    path = OUT_DIR / "E0039_validation_metric_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    previous_manifest = ROOT / "outputs" / "E0029" / "visual" / "E0029-s42-20260917T074229Z-8223ba3_val_subjects_6edges.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    subjects = previous["subjects"]
    edges = np.asarray(previous["edge_indices"], dtype=int)
    labels = previous["edge_labels"]
    with np.load(STATS_PATH) as file:
        stats = {key: file[key] for key in file.files}
    group = group_template_for_warmup(stats, 1).astype(np.float32)
    predictions = {subject: {} for subject in subjects}
    targets = {}
    starts_ref, tr_seconds = None, None

    for experiment, run_id in RUNS.items():
        run_dir = ROOT / "outputs" / experiment / "runs" / run_id
        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        config["paths"]["root"] = str(ROOT)
        dataset = DFCSequenceDataset(config, int(config["data"]["window_length"]), "val", STATS_PATH, "fc1_only")
        matches = {str(subject): [i for i, (sid, _) in enumerate(dataset.samples) if str(sid) == str(subject)] for subject in subjects}
        if any(not indices for indices in matches.values()):
            raise KeyError(f"One or more fixed validation subjects are absent from {experiment}")
        model = None if experiment == "E0032" else load_model(config, run_dir)
        if tr_seconds is None:
            tr_seconds = float(config["data"]["tr_seconds"])
        for subject in subjects:
            sample = dataset[matches[str(subject)][0]]
            target = sample["fc_future"].numpy()
            starts = sample["window_starts"].numpy()[int(config["data"].get("warmup_windows", 1)):]
            if starts_ref is None:
                starts_ref = starts
            elif not np.array_equal(starts_ref, starts):
                raise ValueError(f"Validation timeline differs for subject {subject} in {experiment}")
            if subject in targets and not np.array_equal(targets[subject], target):
                raise ValueError(f"Validation target differs for subject {subject} in {experiment}")
            targets[subject] = target
            predictions[subject][experiment] = predict(experiment, model, config, run_dir, sample, stats)
        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if starts_ref is None or tr_seconds is None:
        raise RuntimeError("No validation series were loaded")
    minutes = (starts_ref - starts_ref[0]) * tr_seconds / 60.0
    curve_path = make_curve_figure(subjects, edges, labels, predictions, targets, group, minutes)
    metric_path = make_metric_figure()
    manifest = {
        "experiments": RUNS,
        "split": "val",
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "reference_selection": str(previous_manifest),
        "figures": [str(curve_path), str(metric_path)],
        "note": "E0032 layout and fixed subjects/edges; plots generated from immutable configs and checkpoints.",
    }
    (OUT_DIR / "E0039_visual_analysis.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
