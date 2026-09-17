"""Render E0027's validation-only multi-subject, six-edge comparison.

Experiment: E0027; run: E0027-s42-20260917T043831Z-3e974d5; split: val.
Inputs: frozen config/checkpoint, shared training statistics, and validation dFC.
Outputs: ``*_val_subjects_6edges.png`` and its JSON manifest in this directory.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from scdfc.data import DFCSequenceDataset, group_template_for_warmup
from scdfc.evaluation import _load_model
from scdfc.visualization import _artifact_path, _load_run, _stats_path, edge_labels, fixed_edge_indices


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "E0027"
RUN_ID = "E0027-s42-20260917T043831Z-3e974d5"
SPLIT = "val"
SUBJECT_SEED = 42
EDGE_SEED = 42
N_SUBJECTS = 4
N_EDGES = 6
TRUE_COLOR = "#1f77b4"
PREDICTED_COLOR = "#ff7f0e"
GROUP_MEAN_COLOR = "#7f7f7f"
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def subject_ids(dataset: DFCSequenceDataset) -> list[str]:
    """Select four validation participants reproducibly without touching test data."""
    subjects = sorted({str(subject) for subject, _ in dataset.samples})
    if len(subjects) < N_SUBJECTS:
        raise ValueError(f"Need {N_SUBJECTS} validation subjects, found {len(subjects)}")
    return sorted(np.random.default_rng(SUBJECT_SEED).choice(subjects, size=N_SUBJECTS, replace=False).tolist())


@torch.no_grad()
def subject_series(model, dataset: DFCSequenceDataset, subject_id: str, device: torch.device):
    matches = [index for index, (subject, _) in enumerate(dataset.samples) if str(subject) == subject_id]
    if not matches:
        raise KeyError(f"Validation subject {subject_id} is absent")
    sample = dataset[matches[0]]
    target = sample["fc_future"].numpy()
    prediction = model(
        sample["sc_matrix"][None].to(device),
        sample["sc_edges"][None].to(device),
        sample["fc_warmup"][None].to(device),
        steps=target.shape[0],
    ).fc_z_edges[0].cpu().numpy()
    warmup = int(model.config.get("data", {}).get("warmup_windows", 1)) if hasattr(model, "config") else 1
    starts = sample["window_starts"].numpy()[warmup:]
    if prediction.shape != target.shape or len(starts) != target.shape[0]:
        raise ValueError(f"Prediction/target shape mismatch for {subject_id}")
    return prediction, target, starts


def edge_title(index: int, label: str) -> str:
    name = textwrap.fill(label.replace(" ↔ ", " <-> ").replace("_", " "), width=25, max_lines=2, placeholder="…")
    return f"edge {index}\n{name}"


def main() -> None:
    root = ROOT.resolve()
    run = _load_run(root, EXPERIMENT_ID, RUN_ID)
    stats_path = _stats_path(root, run.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run.run_dir / "checkpoints" / "best.pt"
    model, _ = _load_model(
        run.config, int(run.config["data"]["window_length"]), checkpoint, stats_path, device, _artifact_path(root, run.config)
    )
    model.eval()

    dataset = DFCSequenceDataset(run.config, int(run.config["data"]["window_length"]), SPLIT, stats_path)
    subjects = subject_ids(dataset)
    series = {subject: subject_series(model, dataset, subject, device) for subject in subjects}
    edges = fixed_edge_indices(
        n_edges=int(run.config["data"]["n_nodes"]) * (int(run.config["data"]["n_nodes"]) - 1) // 2,
        count=N_EDGES,
        seed=EDGE_SEED,
    )
    labels = edge_labels(root, run.config, edges)
    warmup = int(run.config["data"].get("warmup_windows", 1))
    with np.load(stats_path) as stats_file:
        group_mean = group_template_for_warmup({key: stats_file[key] for key in stats_file.files}, warmup)

    all_values = [array[:, edges] for prediction, target, _ in series.values() for array in (prediction, target)]
    all_values.append(group_mean[:, edges])
    stacked = np.concatenate(all_values, axis=0)
    ymins, ymaxs = stacked.min(axis=0), stacked.max(axis=0)
    reference_starts = next(iter(series.values()))[2]
    if len(reference_starts) != len(group_mean):
        raise ValueError("Training group template length does not match validation target length")
    tr_seconds = float(run.config["data"]["tr_seconds"])
    minutes = (reference_starts - reference_starts[0]) * tr_seconds / 60.0

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure = plt.figure(figsize=(17, 19), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(0.08, 0.92))
    header.text(0.5, 0.70, f"{EXPERIMENT_ID} · 验证集多被试时序对比 · 相同随机六条边", ha="center", va="center", fontsize=15, fontproperties=HEADING_FONT)
    header.legend(
        [Line2D([], [], color=TRUE_COLOR, linewidth=1.8), Line2D([], [], color=PREDICTED_COLOR, linewidth=1.8), Line2D([], [], color=GROUP_MEAN_COLOR, linewidth=1.8)],
        ["真实边", "预测边", "训练集组平均（第五列）"], loc="lower center", ncol=3, frameon=False, prop=BODY_FONT,
    )
    axes = body.subplots(N_EDGES, N_SUBJECTS + 1, sharex="col", squeeze=False)
    for row, (edge, label) in enumerate(zip(edges, labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            prediction, target, starts = series[subject]
            if not np.array_equal(starts, reference_starts):
                raise ValueError("Selected validation subjects do not share a common dFC timeline")
            axis.plot(minutes, target[:, edge], color=TRUE_COLOR, linewidth=1.15)
            axis.plot(minutes, prediction[:, edge], color=PREDICTED_COLOR, linewidth=1.0)
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
        group_axis.plot(minutes, group_mean[:, edge], color=GROUP_MEAN_COLOR, linewidth=1.2)
        group_axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        group_axis.grid(alpha=0.22, linewidth=0.55)
        group_axis.tick_params(labelsize=8)
        if row == 0:
            group_axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == N_EDGES - 1:
            group_axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

    body.text(
        0.5, 0.003,
        f"checkpoint: {RUN_ID} · split: val only · edge seed={EDGE_SEED} · subject seed={SUBJECT_SEED} · 第五列为训练集组平均真实时序",
        ha="center", fontsize=8, color="#555555",
    )
    destination = root / "outputs" / EXPERIMENT_ID / "visual"
    image_path = destination / f"{RUN_ID}_val_subjects_6edges.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "split": SPLIT,
        "purpose": "六条固定随机边为行，四名验证集被试为前四列，真实/预测时序对比；第五列为训练集组平均真实时序。",
        "subject_seed": SUBJECT_SEED,
        "edge_seed": EDGE_SEED,
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "layout": {"rows": "六条共享边", "columns": "验证集被试四列 + 训练集组平均一列"},
        "training_group_mean": {"split": "train", "source": "training statistics group_template", "warmup_windows": warmup},
        "image": str(image_path),
    }
    manifest_path = destination / f"{RUN_ID}_val_subjects_6edges.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
