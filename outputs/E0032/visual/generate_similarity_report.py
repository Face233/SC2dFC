"""Plot inter-subject prediction similarity and top-k results for E0026-E0032."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA = json.loads((OUT_DIR / "subject_similarity_val.json").read_text(encoding="utf-8"))
METHODS = ["E0026", "E0030", "E0031", "E0032"]
LABELS = {
    "E0026": "E0026",
    "E0030": "E0030",
    "E0031": "E0031",
    "E0032": "E0032",
}
COLORS = {"E0026": "#b0b0b0", "E0030": "#7f7f7f", "E0031": "#555555", "E0032": "#ff7f0e"}
N = DATA["n_subjects"]
CHANCE = DATA["chance"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 180, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False, "grid.alpha": 0.22, "grid.linestyle": "-",
})


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")


def plot_pairwise_similarity() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(METHODS))
    means = []
    for i, method in enumerate(METHODS):
        result = DATA["methods"][method]["prediction_pairwise_dynamic_pearson"]
        means.append(result["mean_off_diagonal"])
        ax.errorbar(i, result["mean_off_diagonal"],
                    yerr=[[result["mean_off_diagonal"] - result["ci_low"]],
                          [result["ci_high"] - result["mean_off_diagonal"]]],
                    fmt="o", ms=7, capsize=4, lw=1.4, color=COLORS[method])
    ax.plot(x, means, color="#7f7f7f", lw=1.4, zorder=0)
    ax.plot(x[-2:], means[-2:], color=COLORS["E0032"], lw=1.6, zorder=1)
    target = DATA["methods"]["E0032"]["target_pairwise_dynamic_pearson"]
    ax.axhline(target["mean_off_diagonal"], color="#444444", lw=1.0, ls="--")
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mean pairwise Pearson similarity")
    ax.set_title("E0032 · cross-subject prediction similarity · validation")
    ax.grid(alpha=0.22)
    ax.text(0.02, 0.10,
            f"Observed mean = {target['mean_off_diagonal']:.3f} "
            f"(95% CI {target['ci_low']:.3f}–{target['ci_high']:.3f})",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#555555")
    ax.text(0.99, 0.02,
            "Time-demeaned non-overlap trajectories; bars: 95% subject bootstrap CI, n=158",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555555")
    save(fig, "E0032_pairwise_prediction_similarity.png")
    return fig


def plot_topk() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True, sharey="row")
    settings = [
        ("static_topk", "top1", "Static · Top-1", CHANCE["top1"]),
        ("static_topk", "top5", "Static · Top-5", CHANCE["top5"]),
        ("dynamic_topk", "top1", "Dynamic · Top-1", CHANCE["top1"]),
        ("dynamic_topk", "top5", "Dynamic · Top-5", CHANCE["top5"]),
    ]
    for ax, (section, metric, title, chance) in zip(axes.flat, settings):
        values = []
        for i, method in enumerate(METHODS):
            info = DATA["methods"][method][section]
            rate = float(info[metric])
            values.append(rate)
        bars = ax.bar(np.arange(len(METHODS)), values, width=0.42,
                      color=[COLORS[m] for m in METHODS])
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
        ax.axhline(chance, color="#444444", ls="--", lw=1,
                   label=f"Chance = {chance:.2%}")
        ax.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS], rotation=0)
        ax.set_ylim(0, 0.98 if section == "static_topk" and metric == "top5" else
                    0.78 if section == "static_topk" else 0.16)
        ax.set_title(title)
        ax.set_ylabel("Correct subject identification")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Subject retrieval · validation · n=158", fontweight="semibold")
    save(fig, "E0032_static_dynamic_topk.png")
    return fig


def main() -> None:
    figures = [plot_pairwise_similarity(), plot_topk()]
    for figure in figures:
        plt.close(figure)
    print(f"Wrote similarity figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
