"""Generate E0032 validation figures using the established E0024 report style."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "E0032-s42-20260923T052241Z-8d7ca92"
RUN_DIR = ROOT / "outputs" / "E0032" / "runs" / RUN_ID
OUT_DIR = Path(__file__).resolve().parent
REPORT = json.loads((RUN_DIR / "evaluation_val.json").read_text(encoding="utf-8"))

METHODS = ["group_mean", "fc1_persistence", "context_residual_zero", "fc1_decay_template"]
LABELS = {
    "group_mean": "Group template",
    "fc1_persistence": "FC1 persistence",
    "context_residual_zero": "Fixed offset (alpha=1)",
    "fc1_decay_template": "Fitted decay (E0032)",
}
COLORS = {
    "group_mean": "#9a9a9a",
    "fc1_persistence": "#7f7f7f",
    "context_residual_zero": "#555555",
    "fc1_decay_template": "#ff7f0e",
}
SEGMENTS = ["overlap_context", "early_long", "middle_long", "late_long"]
SEGMENT_LABELS = ["Overlap\n0–17", "Early long\n17–85", "Middle long\n85–154", "Late long\n154–223"]

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
})


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")


def bootstrap_ci(values: np.ndarray, seed: int = 42, replicates: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def subject_values(method: str, metric: str) -> dict[str, float]:
    rows = REPORT["analytic_baselines"][method]["per_sample"]
    values: dict[str, list[float]] = {}
    for row in rows:
        values.setdefault(row["subject_id"], []).append(float(row[metric]))
    return {subject: float(np.mean(scores)) for subject, scores in values.items()}


def save_alpha_curve() -> plt.Figure:
    alpha = np.asarray(REPORT["alpha_fit"]["alpha"], dtype=float)
    step_minutes = 5 * 0.72 / 60
    times = np.arange(len(alpha)) * step_minutes
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.plot(times, alpha, color=COLORS["fc1_decay_template"], lw=1.8, marker="o", ms=2.5,
            markevery=16, label="E0032 fitted offset")
    ax.axhline(1, color="#444444", lw=1.0, ls="--", label="E0031 fixed offset")
    ax.set(xlabel="Prediction start time (minutes)", ylabel="Offset retention, alpha[t]",
           title="E0032 · offset retention · validation", ylim=(-0.035, 1.05))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.98, 0.55,
            f"Train subjects: {REPORT['alpha_fit']['training_samples']}\nalpha=1: full FC1 offset\nalpha=0: group template",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")
    save_figure(fig, "E0032_alpha_decay.png")
    return fig


def save_horizon_mse() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(SEGMENTS))
    for method in METHODS:
        rows = REPORT["analytic_baselines"][method]["per_sample"]
        values = [float(np.mean([r["horizon_mse"][segment] for r in rows])) for segment in SEGMENTS]
        ax.plot(x, values, marker="o", ms=5, lw=1.6, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, SEGMENT_LABELS)
    ax.set_ylabel("Mean edge MSE (Fisher-z²)")
    ax.set_title("E0032 · horizon audit · validation")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save_figure(fig, "E0032_horizon_mse.png")
    return fig


def save_subject_gain() -> plt.Figure:
    decay = subject_values("fc1_decay_template", "long_edge_mse")
    comparisons = ["group_mean", "context_residual_zero"]
    labels = ["Vs group template", "Vs fixed offset (alpha=1)"]
    paired = []
    for method in comparisons:
        other = subject_values(method, "long_edge_mse")
        ids = sorted(set(decay) & set(other))
        paired.append(np.asarray([other[s] - decay[s] for s in ids], dtype=float))

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    positions = np.arange(len(paired))
    means = []
    lows = []
    highs = []
    for vals in paired:
        mean, low, high = bootstrap_ci(vals)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    ax.bar(positions, means, width=0.42,
           color=[COLORS["group_mean"], COLORS["fc1_decay_template"]], align="center")
    ax.errorbar(positions, means,
                yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
                fmt="none", ecolor="#444444", capsize=4, linewidth=1.1)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Mean non-overlap long-horizon MSE gain")
    ax.set_xlabel("Baseline MSE − E0032 MSE (positive means E0032 improves)")
    ax.set_title("E0032 · paired long-horizon improvement · validation")
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.99, 0.98, "Bars: mean across 158 subjects; whiskers: 95% subject bootstrap CI",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")
    save_figure(fig, "E0032_subject_mse_gain.png")
    return fig


def save_retrieval() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    labels = ["Group mean", "FC1 persistence", "Fixed offset", "Fitted decay"]
    x = np.arange(len(METHODS))
    for ax, metric, title in zip(axes, ["retrieval_top1", "retrieval_top5"], ["Top-1 retrieval", "Top-5 retrieval"]):
        values = [REPORT["analytic_baselines"][m][metric] for m in METHODS]
        colors = [COLORS[m] for m in METHODS]
        bars = ax.bar(x, values, color=colors, width=0.42)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, max(values) * 1.25 + 0.04)
        ax.set_title(title)
        ax.set_ylabel("Correct subject identification")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Static future-average FC retrieval · validation", fontweight="semibold")
    save_figure(fig, "E0032_subject_retrieval.png")
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = [save_alpha_curve(), save_horizon_mse(), save_subject_gain(), save_retrieval()]
    for fig in figures:
        plt.close(fig)
    print(f"Wrote E0032 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
