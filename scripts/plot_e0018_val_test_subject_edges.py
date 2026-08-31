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

from scdfc.data import DFCSequenceDataset
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
    all_values = [array[:, edges] for data in split_data.values() for prediction, target, _starts in data.values() for array in (prediction, target)]
    stacked_values = np.concatenate(all_values, axis=0)
    ymins = np.min(stacked_values, axis=0)
    ymaxs = np.max(stacked_values, axis=0)

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure, axes = plt.subplots(4, N_EDGES, figsize=(22, 12), sharex="col", squeeze=False)
    row_specs = [(split, subject) for split in SPLITS for subject in split_subjects[split]]
    tr_seconds = float(run.config["data"]["tr_seconds"])
    stride = int(run.config["data"]["stride"])
    for row, (split, subject) in enumerate(row_specs):
        prediction, target, starts = split_data[split][subject]
        minutes = (starts - starts[0]) * stride * tr_seconds / 60.0
        for column, (edge, label) in enumerate(zip(edges, labels)):
            axis = axes[row, column]
            axis.plot(minutes, target[:, edge], color=TRUE_COLOR, linewidth=1.15)
            axis.plot(minutes, prediction[:, edge], color=PREDICTED_COLOR, linewidth=1.0)
            pad = max((ymaxs[column] - ymins[column]) * 0.06, 0.02)
            axis.set_ylim(ymins[column] - pad, ymaxs[column] + pad)
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(_edge_title(int(edge), label), fontsize=9, fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{('验证集' if split == 'val' else '测试集')}\n{subject}\nFisher-z", fontsize=9, fontproperties=BODY_FONT)
            if row == len(row_specs) - 1:
                axis.set_xlabel("预测起点后分钟", fontsize=9, fontproperties=BODY_FONT)

    figure.suptitle(
        "E0018 · 验证集/测试集被试时序对比 · 相同随机六条边",
        fontsize=15,
        fontproperties=HEADING_FONT,
        y=0.995,
    )
    figure.legend(
        [Line2D([], [], color=TRUE_COLOR, linewidth=1.8), Line2D([], [], color=PREDICTED_COLOR, linewidth=1.8)],
        ["真实边", "预测边"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=False,
        prop=BODY_FONT,
    )
    figure.text(
        0.5,
        0.008,
        f"checkpoint: {RUN_ID} · edge seed={EDGE_SEED} · subject seed={SUBJECT_SEED} · test panel仅作探索性可视化",
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
        "purpose": "在验证集和测试集各随机选择2个被试，使用相同随机6条边比较真实/预测时序。",
        "test_panel_scope": "exploratory_visualization_only",
        "subject_seed": SUBJECT_SEED,
        "edge_seed": EDGE_SEED,
        "subjects": split_subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": labels,
        "image": str(image_path),
    }
    manifest_path = destination / "E0018_val_test_subjects_6edges.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
