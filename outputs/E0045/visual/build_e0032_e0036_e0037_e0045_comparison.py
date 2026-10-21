"""Render E0032/E0036/E0037/E0045 validation curves in the E0043 layout."""
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
from scdfc.data import DFCSequenceDataset, group_template_for_warmup
from scdfc.evaluation import _load_model
from scdfc.management import config_sha256, file_sha256, verify_artifact


RUNS = {
    "E0032": "E0032-s42-20260923T052241Z-8d7ca92",
    "E0036": "E0036-s42-20260929T082122Z-b5b871f",
    "E0037": "E0037-s42-20260929T084843Z-6c70d8a",
    "E0045": "E0045-s42-20261021T033047Z-19f52fe",
}
REFERENCE = ROOT / "outputs" / "E0037" / "visual" / "E0032_E0036_E0037_multisubject_comparison.json"
STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "history_training_stats.npz"
LEGACY_STATS = ROOT / "outputs" / "shared" / "dataset_lr_v1" / "window_83" / "training_stats.npz"
OUT_DIR = Path(__file__).resolve().parent
IMAGE = OUT_DIR / "E0032_E0036_E0037_E0045_multisubject_comparison.png"
MANIFEST = OUT_DIR / "E0032_E0036_E0037_E0045_multisubject_comparison.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NONOVERLAP_START = 17
COMMON_ORIGIN = 9
COLORS = {
    "target": "#1f77b4",
    "E0032": "#9467bd",
    "E0036": "#2ca02c",
    "E0037": "#d62728",
    "E0045": "#ff7f0e",
    "group": "#7f7f7f",
    "boundary": "#666666",
}
LINESTYLES = {"E0032": "-", "E0036": "-", "E0037": "--", "E0045": "-."}
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
        raise ValueError("The E0043 layout requires four fixed subjects and six edges")
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
    report = json.loads((run_dir / "evaluation_val.json").read_text(encoding="utf-8"))
    if report.get("split") != "val":
        raise ValueError(f"{experiment} report is not a validation report")
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint_hash = None
    model = None
    alpha_hash = None
    if experiment == "E0032":
        alpha_path = run_dir / "alpha_fit.npz"
        alpha_hash = file_sha256(alpha_path)
        alpha = np.load(alpha_path)["alpha"].astype(np.float32)
        stats = {key: value for key, value in np.load(LEGACY_STATS).items()}
        n_windows = ((int(config["data"]["n_timepoints"]) - int(config["data"]["window_length"]))
                     // int(config["data"]["stride"]) + 1)
        expected_alpha_shape = (n_windows - int(config["data"].get("warmup_windows", 1)),)
        if alpha.shape != expected_alpha_shape:
            raise ValueError(f"E0032 alpha shape {alpha.shape} differs from {expected_alpha_shape}")
        if not np.array_equal(stats["group_template"], np.load(STATS)["group_template"]):
            raise ValueError("Legacy and history training templates differ")
    else:
        checkpoint_hash = file_sha256(checkpoint)
        report_checkpoint_hash = report.get("checkpoint_sha256")
        if report_checkpoint_hash is not None and report_checkpoint_hash != checkpoint_hash:
            raise ValueError(f"{experiment} validation report does not match its best checkpoint")
        artifact = verify_artifact(config)
        model, _ = _load_model(
            config,
            int(config["data"]["window_length"]),
            checkpoint,
            STATS,
            DEVICE,
            artifact,
        )
        alpha = None
        stats = None
    dataset = DFCSequenceDataset(
        config,
        int(config["data"]["window_length"]),
        "val",
        STATS,
        "full",
    )
    return run_dir, config, model, dataset, report, checkpoint_hash, alpha, alpha_hash, stats


@torch.no_grad()
def collect_predictions(subjects):
    predictions = {subject: {} for subject in subjects}
    targets = {}
    reference_starts = None
    hashes = {"checkpoint_sha256": {}, "alpha_fit_sha256": {}}
    native_origins = {}
    tr_seconds = None

    for experiment, run_id in RUNS.items():
        (run_dir, config, model, dataset, report, checkpoint_hash,
         alpha, alpha_hash, stats) = load_run(experiment, run_id)
        hashes["checkpoint_sha256"][experiment] = checkpoint_hash
        hashes["alpha_fit_sha256"][experiment] = alpha_hash
        native_origin = config["data"].get("forecast_origins", [0])[0]
        native_origins[experiment] = int(native_origin)
        if tr_seconds is None:
            tr_seconds = float(config["data"]["tr_seconds"])
        elif tr_seconds != float(config["data"]["tr_seconds"]):
            raise ValueError(f"TR differs in {experiment}")

        for subject in subjects:
            matches = [i for i, (sid, _) in enumerate(dataset.samples) if str(sid) == subject]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one validation sample for {subject} in {experiment}")
            sample = dataset[matches[0]]
            target = sample["fc_future"].numpy()
            starts = sample["window_starts"].numpy()
            if experiment == "E0032":
                warmup = sample["fc_warmup"].numpy().astype(np.float32)
                context = stats.get("context_template", stats["fc_mean"]).astype(np.float32)
                group = group_template_for_warmup(stats, int(config["data"].get("warmup_windows", 1)))
                native_prediction = group[: len(target)] + alpha[: len(target), None] * (warmup - context)
            else:
                native_prediction = model(
                    sample["sc_matrix"][None].to(DEVICE),
                    sample["sc_edges"][None].to(DEVICE),
                    sample["fc_warmup"][None].to(DEVICE),
                    steps=len(target),
                ).fc_z_edges[0].cpu().numpy()
            if native_prediction.shape != target.shape:
                raise ValueError(
                    f"{experiment} prediction/target mismatch for {subject}: "
                    f"{native_prediction.shape} vs {target.shape}"
                )

            if config["data"].get("forecast_origins"):
                common_target = target
                common_prediction = native_prediction
                common_starts = starts
            else:
                # Older runs predict FC[1:] from FC[0]. Keep only the common FC[10:] segment.
                common_target = target[COMMON_ORIGIN:]
                common_prediction = native_prediction[COMMON_ORIGIN:]
                common_starts = starts[int(config["data"].get("warmup_windows", 1)) + COMMON_ORIGIN:]
            if len(common_target) != 214 or len(common_starts) != len(common_target):
                raise ValueError(f"Unexpected common FC[10:] horizon in {experiment}: {len(common_target)}")
            if reference_starts is None:
                reference_starts = common_starts
            elif not np.array_equal(reference_starts, common_starts):
                raise ValueError(f"Forecast timeline differs for {subject} in {experiment}")
            if subject in targets and not np.array_equal(targets[subject], common_target):
                raise ValueError(f"Common FC[10:] validation targets differ for {subject} in {experiment}")
            targets[subject] = common_target
            predictions[subject][experiment] = common_prediction

        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return predictions, targets, reference_starts, tr_seconds, hashes, native_origins


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")


def selected_metrics(subjects, edges, predictions, targets):
    """Summarize only the four plotted validation subjects, after nonoverlap start."""
    result = {}
    for experiment in RUNS:
        per_subject = []
        for subject in subjects:
            truth = targets[subject][NONOVERLAP_START:, edges]
            pred = predictions[subject][experiment][NONOVERLAP_START:, edges]
            time_r = np.nanmean([pearson(truth[:, i], pred[:, i]) for i in range(truth.shape[1])])
            std_ratio = np.nanmean([
                np.std(pred[:, i]) / np.std(truth[:, i])
                for i in range(truth.shape[1]) if np.std(truth[:, i]) > 1e-8
            ])
            per_subject.append({
                "mse": float(np.mean((pred - truth) ** 2)),
                "time_r": float(time_r),
                "std_ratio": float(std_ratio),
            })
        result[experiment] = {
            key: float(np.mean([row[key] for row in per_subject]))
            for key in ("mse", "time_r", "std_ratio")
        }
    return result


def render(subjects, edges, labels, predictions, targets, group, minutes, reports):
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
        0.5, 0.70, "E0032 / E0036 / E0037 / E0045 · 验证集多被试预测对比 · seed 42",
        ha="center", va="center", fontsize=14, fontproperties=HEADING_FONT,
    )
    handles = [Line2D([], [], color=COLORS["target"], linewidth=1.8)]
    labels_legend = ["真实边"]
    for experiment in RUNS:
        handles.append(Line2D([], [], color=COLORS[experiment], linewidth=1.5,
                              linestyle=LINESTYLES[experiment]))
        labels_legend.append(f"{experiment}预测")
    handles.extend([
        Line2D([], [], color=COLORS["group"], linewidth=1.4),
        Line2D([], [], color=COLORS["boundary"], linewidth=0.9, linestyle=":"),
    ])
    labels_legend.extend(["训练集组平均（第五列）", "保守无重叠段起点"])
    header.legend(handles, labels_legend, loc="lower center", ncol=4, frameon=False, prop=BODY_FONT)

    axes = body.subplots(len(edges), len(subjects) + 1, sharex="col", squeeze=False)
    nonoverlap_minute = minutes[NONOVERLAP_START]
    for row, (edge, label) in enumerate(zip(edges, labels)):
        pad = max((ymaxs[row] - ymins[row]) * 0.06, 0.02)
        for column, subject in enumerate(subjects):
            axis = axes[row, column]
            axis.axvspan(minutes[0], nonoverlap_minute, color="#f2f2f2", zorder=0)
            axis.axvline(nonoverlap_minute, color=COLORS["boundary"], linewidth=0.8,
                         linestyle=":", alpha=0.8)
            axis.plot(minutes, targets[subject][:, edge], color=COLORS["target"],
                      linewidth=1.1, zorder=4)
            for experiment in RUNS:
                axis.plot(
                    minutes,
                    predictions[subject][experiment][:, edge],
                    color=COLORS[experiment],
                    linewidth=1.2 if experiment == "E0045" else 0.95,
                    linestyle=LINESTYLES[experiment],
                    alpha=0.92,
                    zorder=6 if experiment == "E0045" else 5,
                )
            axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
            axis.set_xlim(minutes[0], minutes[-1])
            axis.grid(alpha=0.22, linewidth=0.55)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"验证集\n被试 {subject}", fontsize=10,
                               fontproperties=BODY_FONT, pad=8)
            if column == 0:
                axis.set_ylabel(f"{edge_title(int(edge), label)}\nFisher-z",
                                fontsize=8.5, fontproperties=BODY_FONT)
            if row == len(edges) - 1:
                axis.set_xlabel("首个共同目标窗后分钟", fontsize=9, fontproperties=BODY_FONT)

        axis = axes[row, -1]
        axis.axvspan(minutes[0], nonoverlap_minute, color="#f2f2f2", zorder=0)
        axis.axvline(nonoverlap_minute, color=COLORS["boundary"], linewidth=0.8,
                     linestyle=":", alpha=0.8)
        axis.plot(minutes, group[:, edge], color=COLORS["group"], linewidth=1.2)
        axis.set_ylim(ymins[row] - pad, ymaxs[row] + pad)
        axis.set_xlim(minutes[0], minutes[-1])
        axis.grid(alpha=0.22, linewidth=0.55)
        axis.tick_params(labelsize=8)
        if row == 0:
            axis.set_title("训练集\n组平均真实时序", fontsize=10,
                           fontproperties=BODY_FONT, pad=8)
        if row == len(edges) - 1:
            axis.set_xlabel("首个共同目标窗后分钟", fontsize=9, fontproperties=BODY_FONT)

    footer.text(
        0.5, 0.78,
        "val only · 沿用 E0037 的四名被试和六条边 · 所有曲线对齐到共同目标 FC[10:]（214窗）",
        ha="center", fontsize=8, color="#555555", fontproperties=BODY_FONT,
    )
    metric_text = " · ".join(
        f"{experiment} MSE={reports[experiment]['mse']:.4f}, 时间 r={reports[experiment]['time_r']:.4f}"
        for experiment in RUNS
    )
    footer.text(
        0.5, 0.49,
        "灰色为前17窗；点线为无重叠段起点 · 指标仅汇总图中4名被试 · " + metric_text,
        ha="center", fontsize=7.5, color="#555555", fontproperties=BODY_FONT,
    )
    footer.text(
        0.5, 0.15,
        "E0045 使用 FC[9] 起报；E0032/E0036/E0037 沿用 FC[0] 原生预测并截取共同目标区间 · 检查点哈希见 JSON",
        ha="center", fontsize=7.5, color="#555555", fontproperties=BODY_FONT,
    )
    figure.savefig(IMAGE, dpi=180)
    plt.close(figure)


def main() -> None:
    subjects, edges, edge_labels = load_reference()
    predictions, targets, starts, tr_seconds, hashes, native_origins = collect_predictions(subjects)
    group = np.load(STATS)["group_template"][COMMON_ORIGIN:]
    if len(group) != len(starts):
        raise ValueError("Training group template does not align with the common FC[10:] timeline")
    minutes = (starts - starts[0]) * tr_seconds / 60.0
    metrics = selected_metrics(subjects, edges, predictions, targets)
    render(subjects, edges, edge_labels, predictions, targets, group, minutes, metrics)
    manifest = {
        "experiments": RUNS,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "alpha_fit_sha256": hashes["alpha_fit_sha256"]["E0032"],
        "split": "val",
        "subjects": subjects,
        "edge_indices": edges.tolist(),
        "edge_labels": edge_labels,
        "reference_selection": str(REFERENCE),
        "common_target": "FC[10:]",
        "common_forecast_windows": int(len(starts)),
        "native_forecast_origins_zero_based": native_origins,
        "legacy_output_slice_offset": COMMON_ORIGIN,
        "nonoverlap_start": NONOVERLAP_START,
        "nonoverlap_minute": float(minutes[NONOVERLAP_START]),
        "colors": COLORS,
        "linestyles": LINESTYLES,
        "selected_four_subject_nonoverlap_metrics": metrics,
        "metric_scope": "Only the four fixed validation subjects and six displayed edges; not the full validation split.",
        "image": str(IMAGE),
        "note": (
            "All curves share FC[10:] targets. E0032/E0036/E0037 native predictions start after FC[0] "
            "and are sliced to the common target interval; E0045 predicts from FC[9]."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(IMAGE), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
