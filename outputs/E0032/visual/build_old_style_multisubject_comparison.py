"""Render E0026-E0032 as the established six-edge, multi-subject time-series grid."""
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
    "E0026": "E0026-s42-20260917T041533Z-3e974d5",
    "E0030": "E0030-s42-20260917T122244Z-60c4608",
    "E0031": "E0031-s42-20260918T035339Z-badb7a4",
    "E0032": "E0032-s42-20260923T052241Z-8d7ca92",
}
COLORS = {
    "target": "#1f77b4",
    "E0026": "#2ca02c",
    "E0030": "#ff7f0e",
    "E0031": "#d62728",
    "E0032": "#9467bd",
    "group": "#7f7f7f",
}
LINESTYLES = {experiment: "-" for experiment in RUNS}
LABELS = {
    "E0026": "E0026",
    "E0030": "E0030",
    "E0031": "E0031",
    "E0032": "E0032",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EDGES = 6
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(label.replace(" ↔ ", " <-> ").replace("_", " "), width=25, max_lines=2, placeholder="…")
    return f"edge {index}\n{name}"


def load_model(experiment: str, config: dict, run_dir: Path):
    checkpoint = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    artifact_path = verify_artifact(config)
    model = build_sequence_model(
        config,
        int(config["data"]["window_length"]),
        payload["decoder_type"],
        STATS_PATH,
        DEVICE,
        payload.get("sc_encoder_type", "hybrid"),
        artifact_path,
        payload,
    )
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"context_template"}:
        raise RuntimeError(f"Unexpected checkpoint mismatch in {experiment}: {incompatible}")
    model.eval()
    return model


def get_model_prediction(experiment: str, model, config: dict, run_dir: Path, sample: dict, stats: dict) -> np.ndarray:
    target = sample["fc_future"].numpy()
    if experiment == "E0032":
        alpha = np.load(run_dir / "alpha_fit.npz")["alpha"].astype(np.float32)
        context = stats.get("context_template", stats["fc_mean"]).astype(np.float32)
        warmup = sample["fc_warmup"].numpy().astype(np.float32)
        group = group_template_for_warmup(stats, int(config["data"].get("warmup_windows", 1))).astype(np.float32)
        return group[: len(target)] + alpha[: len(target), None] * (warmup - context)

    with torch.no_grad():
        prediction = model(
            sample["sc_matrix"][None].to(DEVICE),
            sample["sc_edges"][None].to(DEVICE),
            sample["fc_warmup"][None].to(DEVICE),
            steps=target.shape[0],
        ).fc_z_edges[0].cpu().numpy()
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch in {experiment}: {prediction.shape} vs {target.shape}")
    return prediction


def main() -> None:
    previous_manifest = ROOT / "outputs" / "E0029" / "visual" / "E0029-s42-20260917T074229Z-8223ba3_val_subjects_6edges.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    subjects = previous["subjects"]
    edges = np.asarray(previous["edge_indices"], dtype=int)
    edge_labels = previous["edge_labels"]

    with np.load(STATS_PATH) as file:
        stats = {key: file[key] for key in file.files}
    group = group_template_for_warmup(stats, 1).astype(np.float32)

    predictions: dict[str, dict[str, np.ndarray]] = {subject: {} for subject in subjects}
    targets: dict[str, np.ndarray] = {}
    reference_starts = None
    tr_seconds = None

    for experiment, run_id in RUNS.items():
        run_dir = ROOT / "outputs" / experiment / "runs" / run_id
        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        config["paths"]["root"] = str(ROOT)
        dataset = DFCSequenceDataset(config, int(config["data"]["window_length"]), "val", STATS_PATH)
        indices = {}
        for subject in subjects:
            matches = [i for i, (sid, _) in enumerate(dataset.samples) if str(sid) == subject]
            if not matches:
                raise KeyError(f"Validation subject {subject} is absent from {experiment}")
            indices[subject] = matches[0]

        model = None if experiment == "E0032" else load_model(experiment, config, run_dir)

        if tr_seconds is None:
            tr_seconds = float(config["data"]["tr_seconds"])
        for subject in subjects:
            sample = dataset[indices[subject]]
            target = sample["fc_future"].numpy()
            starts = sample["window_starts"].numpy()[int(config["data"].get("warmup_windows", 1)):]
            if reference_starts is None:
                reference_starts = starts
            elif not np.array_equal(reference_starts, starts):
                raise ValueError(f"Validation timeline differs for {subject} in {experiment}")
            if subject in targets and not np.array_equal(targets[subject], target):
                raise ValueError(f"Validation target differs for {subject} in {experiment}")
            targets[subject] = target
            predictions[subject][experiment] = get_model_prediction(experiment, model, config, run_dir, sample, stats)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        del dataset

    if reference_starts is None or tr_seconds is None:
        raise RuntimeError("No validation series were loaded")
    minutes = (reference_starts - reference_starts[0]) * tr_seconds / 60.0

    relevant = [array[:, edges] for subject in subjects for array in (targets[subject], *[predictions[subject][m] for m in RUNS])]
    relevant.append(group[: len(reference_starts), edges])
    all_values = np.concatenate(relevant, axis=0)
    ymins, ymaxs = all_values.min(axis=0), all_values.max(axis=0)

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(0.085, 0.915))
    header.text(0.5, 0.70, "E0026 / E0030 / E0031 / E0032 · 验证集多被试时序对比 · 相同六条边",
                ha="center", va="center", fontsize=15, fontproperties=HEADING_FONT)
    legend_handles = [Line2D([], [], color=COLORS["target"], linewidth=1.8)]
    legend_labels = ["真实边"]
    for experiment in RUNS:
        legend_handles.append(Line2D([], [], color=COLORS[experiment], linewidth=1.6,
                                     linestyle=LINESTYLES[experiment]))
        legend_labels.append(f"{LABELS[experiment]}预测")
    legend_handles.append(Line2D([], [], color=COLORS["group"], linewidth=1.4))
    legend_labels.append("训练集组平均（第五列）")
    header.legend(legend_handles, legend_labels, loc="lower center", ncol=6, frameon=False, prop=BODY_FONT)

    axes = body.subplots(N_EDGES, len(subjects) + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, edge_labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"], linewidth=1.15, zorder=4)
            for experiment in RUNS:
                axis.plot(minutes, predictions[subject][experiment][:, edge], color=COLORS[experiment],
                          linewidth=0.95, alpha=0.9, linestyle=LINESTYLES[experiment])
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"验证集\n被试 {subject}", fontsize=10, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{edge_title(int(edge), label)}\nFisher-z", fontsize=8.5, fontproperties=BODY_FONT)
            if row == N_EDGES - 1:
                axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

        group_axis = axes[row, -1]
        group_axis.plot(minutes, group[: len(minutes), edge], color=COLORS["group"], linewidth=1.2)
        group_axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        group_axis.grid(alpha=0.22, linewidth=0.55)
        group_axis.tick_params(labelsize=8)
        if row == 0:
            group_axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == N_EDGES - 1:
            group_axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

    body.text(
        0.5, 0.003,
        "val only · 四名被试和六条边沿用 E0029 图的选择 · 四个实验使用同一真实目标 · 第五列为训练集组平均真实时序",
        ha="center", fontsize=8, color="#555555",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUT_DIR / "E0032_multisubject_dynamic_comparison.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    manifest = {
        "experiments": RUNS,
        "split": "val",
        "purpose": "按旧版多被试时序图样式，六条固定边为行、四名验证集被试为前四列，叠加四个实验预测；第五列为训练集组平均真实时序。",
        "reference_selection": str(previous_manifest),
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": edge_labels,
        "layout": {"rows": "六条共享边", "columns": "验证集被试四列 + 训练集组平均一列"},
        "image": str(image_path),
    }
    (OUT_DIR / "E0032_multisubject_dynamic_comparison.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
