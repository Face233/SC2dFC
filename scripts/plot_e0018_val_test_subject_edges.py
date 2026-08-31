"""Plot E0018 true/predicted dFC edge trajectories for validation and test subjects.

The same reproducibly sampled six edges are shown for two randomly selected
subjects in each split.  The test panel is exploratory visualization only; it
does not create a formal managed test evaluation or alter experiment status.
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
from scdfc.visualization import (
    _artifact_path,
    _load_run,
    _stats_path,
    edge_labels,
    fixed_edge_indices,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "E0018-s42-20260831T033422Z-9e36157"
EXPERIMENT_ID = "E0018"
SUBJECT_SEED = 42
EDGE_SEED = 42
N_SUBJECTS = 2
N_EDGES = 6
SPLITS = ("val", "test")
TRUE_COLOR = "#1f77b4"
PREDICTED_COLOR = "#ff7f0e"
GROUP_MEAN_COLOR = "#7f7f7f"
BODY_FONT = FontProperties(family="Microsoft YaHei")
HEADING_FONT = FontProperties(family="Microsoft YaHei", weight="bold")


def _subject_ids(dataset: DFCSequenceDataset, seed: int) -> list[str]:
    subjects = sorted({str(subject) for subject, _run in dataset.samples})
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(subjects, size=N_SUBJECTS, replace=False).tolist())


@torch.no_grad()
def _subject_series(model, dataset: DFCSequenceDataset, subject_id: str, device: torch.device):
    matches = [index for index, (subject, _run) in enumerate(dataset.samples) if str(subject) == subject_id]
    if not matches:
        raise KeyError(f"Subject {subject_id} is absent from the requested split")
    sample = dataset[matches[0]]
    target = sample["fc_future"].numpy()
    result = model(
        sample["sc_matrix"][None].to(device),
        sample["sc_edges"][None].to(device),
        sample["fc_warmup"][None].to(device),
        steps=target.shape[0],
    )
    prediction = result.fc_z_edges[0].cpu().numpy()
    warmup = int(model.config.get("data", {}).get("warmup_windows", 1)) if hasattr(model, "config") else 1
    starts = sample["window_starts"].numpy()[warmup:]
    if prediction.shape != target.shape or len(starts) != target.shape[0]:
        raise ValueError(f"Prediction/target shape mismatch for {dataset.split_name}/{subject_id}")
    return prediction, target, starts


def _load_subject_data(run, stats_path: Path, model, split: str, subjects: list[str], device: torch.device):
    dataset = DFCSequenceDataset(run.config, int(run.config["data"]["window_length"]), split, stats_path)
    return {
        subject: _subject_series(model, dataset, subject, device)
        for subject in subjects
    }


def _training_group_mean(stats_path: Path, warmup_windows: int) -> np.ndarray:
    """Load the training-only target-space group template aligned to FC[K:]."""
    with np.load(stats_path) as stats_file:
        stats = {key: stats_file[key] for key in stats_file.files}
    return group_template_for_warmup(stats, warmup_windows)


def _edge_title(index: int, label: str) -> str:
    wrapped = textwrap.fill(label.replace(" ↔ ", " <-> ").replace("_", " "), width=25, max_lines=2, placeholder="…")
    return f"edge {index}\n{wrapped}"


def main() -> None:
    root = ROOT.resolve()
    run = _load_run(root, EXPERIMENT_ID, RUN_ID)
    stats_path = _stats_path(root, run.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run.run_dir / "checkpoints" / "best.pt"
    model, _payload = _load_model(run.config, int(run.config["data"]["window_length"]), checkpoint, stats_path, device, _artifact_path(root, run.config))
    model.eval()

    split_subjects: dict[str, list[str]] = {}
    split_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for split in SPLITS:
        dataset = DFCSequenceDataset(run.config, int(run.config["data"]["window_length"]), split, stats_path)
        subjects = _subject_ids(dataset, SUBJECT_SEED)
        split_subjects[split] = subjects
        split_data[split] = _load_subject_data(run, stats_path, model, split, subjects, device)

    edges = fixed_edge_indices(n_edges=int(run.config["data"]["n_nodes"]) * (int(run.config["data"]["n_nodes"]) - 1) // 2, count=N_EDGES, seed=EDGE_SEED)
    labels = edge_labels(root, run.config, edges)
    warmup_windows = int(run.config["data"].get("warmup_windows", 1))
    training_group_mean = _training_group_mean(stats_path, warmup_windows)
    all_values = [
        array[:, edges]
        for data in split_data.values()
        for prediction, target, _starts in data.values()
        for array in (prediction, target)
    ]
    all_values.append(training_group_mean[:, edges])
    stacked_values = np.concatenate(all_values, axis=0)
    ymins = np.min(stacked_values, axis=0)
    ymaxs = np.max(stacked_values, axis=0)

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    column_specs = [(split, subject) for split in SPLITS for subject in split_subjects[split]]
    figure, axes = plt.subplots(N_EDGES, len(column_specs) + 1, figsize=(17, 19), sharex="col", squeeze=False)
    tr_seconds = float(run.config["data"]["tr_seconds"])
    stride = int(run.config["data"]["stride"])
    reference_starts = next(iter(next(iter(split_data.values())).values()))[2]
    group_minutes = (reference_starts - reference_starts[0]) * stride * tr_seconds / 60.0
    if len(group_minutes) != len(training_group_mean):
        raise ValueError("Training group template length does not match held-out target length")

    for row, (edge, label) in enumerate(zip(edges, labels)):
        for column, (split, subject) in enumerate(column_specs):
            axis = axes[row, column]
            prediction, target, starts = split_data[split][subject]
            if not np.array_equal(starts, reference_starts):
                raise ValueError("Selected subjects do not share a common dFC timeline")
            minutes = (starts - starts[0]) * stride * tr_seconds / 60.0
            axis.plot(minutes, target[:, edge], color=TRUE_COLOR, linewidth=1.15)
            axis.plot(minutes, prediction[:, edge], color=PREDICTED_COLOR, linewidth=1.0)
            pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                split_name = "验证集" if split == "val" else "测试集"
                axis.set_title(f"{split_name}\n被试 {subject}", fontsize=10, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{_edge_title(int(edge), label)}\nFisher-z", fontsize=8.5, fontproperties=BODY_FONT)
            if row == N_EDGES - 1:
                axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

        group_axis = axes[row, -1]
        group_axis.plot(group_minutes, training_group_mean[:, edge], color=GROUP_MEAN_COLOR, linewidth=1.2)
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        group_axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        group_axis.grid(alpha=0.22, linewidth=0.55)
        group_axis.tick_params(labelsize=8)
        if row == 0:
            group_axis.set_title("训练集\n组平均真实时序", fontsize=10, fontproperties=BODY_FONT, pad=8)
        if row == N_EDGES - 1:
            group_axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

    figure.suptitle(
        "E0018 · 验证集/测试集被试时序对比与训练集组平均 · 相同随机六条边",
        fontsize=15,
        fontproperties=HEADING_FONT,
        y=0.995,
    )
    figure.legend(
        [Line2D([], [], color=TRUE_COLOR, linewidth=1.8), Line2D([], [], color=PREDICTED_COLOR, linewidth=1.8), Line2D([], [], color=GROUP_MEAN_COLOR, linewidth=1.8)],
        ["真实边", "预测边", "训练集组平均（第五列）"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=3,
        frameon=False,
        prop=BODY_FONT,
    )
    figure.text(
        0.5,
        0.008,
        f"checkpoint: {RUN_ID} · edge seed={EDGE_SEED} · subject seed={SUBJECT_SEED} · 第五列为训练集组平均真实时序 · test panel仅作探索性可视化",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.995, 0.94))

    destination = root / "outputs" / EXPERIMENT_ID / "visual"
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / "E0018_val_test_subjects_6edges.png"
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "purpose": "将相同随机6条边置于行、验证集和测试集各2名被试置于前四列，比较真实/预测时序；第五列单独展示训练集组平均真实时序。",
        "test_panel_scope": "exploratory_visualization_only",
        "subject_seed": SUBJECT_SEED,
        "edge_seed": EDGE_SEED,
        "subjects": split_subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "layout": {"rows": "六条共享边", "columns": "验证/测试被试四列 + 训练集组平均一列"},
        "training_group_mean": {"split": "train", "source": "training statistics group_template", "warmup_windows": warmup_windows},
        "image": str(image_path),
    }
    manifest_path = destination / "E0018_val_test_subjects_6edges.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
