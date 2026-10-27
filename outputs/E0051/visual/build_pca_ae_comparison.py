"""E0036/E0037/E0050/E0051, seed42, validation only.

Runs are pinned in RUNS below. Inputs: frozen config_resolved.yaml, best.pt,
metrics_best.json, train.log, evaluation_val.json, comparison_*_val.json and
dynamic_audit_val/summary.json. Requires the authorized local validation data
and training statistics to redraw fixed-edge examples; never reads test.
Outputs: per-run overview, fixed_edges, dynamic_calibration, horizon_audit PNGs
under E0050/visual and E0051/visual, plus four-model metrics, dynamics and paired
CI PNGs under E0051/visual. Public JSON contains aggregate results only.
Run from any directory with the fmri Python environment.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from scdfc.visualization import (
    _load_run, _prediction, _align_predictions, _read_log, fixed_edge_indices,
    edge_labels, plot_dynamic_calibration, plot_horizon_audit,
)

RUNS = {
    "E0036": "E0036-s42-20260929T082122Z-b5b871f",
    "E0037": "E0037-s42-20260929T084843Z-6c70d8a",
    "E0050": "E0050-s42-20261027T021543Z-bc0af78",
    "E0051": "E0051-s42-20261027T025607Z-bc0af78",
}
LABELS = {"E0036": "AE / FC1", "E0037": "AE / FC1 + SC",
          "E0050": "PCA / FC1", "E0051": "PCA / FC1 + SC"}
COLORS = ["#8393a8", "#3b5b80", "#e8b366", "#d47728"]
PREFIX = "E0036_E0037_E0050_E0051_val"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def aggregate_only(value):
    if isinstance(value, dict):
        return {k: aggregate_only(v) for k, v in value.items()
                if k not in {"per_sample", "subject_id", "subject_ids", "per_sample_csv", "horizon_per_sample_csv"}}
    if isinstance(value, list):
        return [aggregate_only(v) for v in value]
    return value


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def overview(run, path):
    best = read(run.run_dir / "metrics_best.json")
    rows = _read_log(run.run_dir / "train.log")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for key, label, color in [("train_loss", "Train MSE", "#ff7f0e"),
                              ("long_edge_mse", "Val long MSE", "#1f77b4")]:
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label, color=color)
    epoch = best["best_epoch"] + 1
    axes[0].axvline(epoch, color="black", ls="--", lw=1, label=f"Best: epoch {epoch}")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="Optimization and checkpoint selection")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    axes[1].axis("off")
    lines = [f"{run.experiment_id} | {LABELS[run.experiment_id]}",
             "Frozen FC component: A0044 PCA, 4005 -> 256",
             "Forecast: Transformer + learned edge residual",
             "Input: FC[0]; output: FC[1:224]",
             "Long evaluation: FC[18:224], 158 validation subjects",
             "Seed 42 | 150 epochs | pure MSE",
             f"Best epoch: {epoch} (1-based)",
             f"Selection long MSE: {best['metrics']['long_edge_mse']:.9f}",
             "Single-seed validation exploration; test untouched"]
    for i, line in enumerate(lines):
        axes[1].text(0, .96 - i * .102, line, va="top", fontsize=10)
    fig.suptitle(f"{run.run_id} | validation", fontsize=11)
    return save(fig, path)


def dynamics(runs, entries, path):
    starts, aligned, _ = _align_predictions([entries[r.experiment_id] for r in runs])
    edges = fixed_edge_indices()
    names = edge_labels(ROOT, runs[0].config, edges)
    minutes = (starts - starts[0]) * float(runs[0].config["data"]["tr_seconds"]) / 60
    fig = plt.figure(figsize=(4 * len(runs), 10.5), layout="constrained")
    header, body = fig.subfigures(2, 1, height_ratios=[.14, .86])
    header.text(.5, .8, "Fixed four-edge forecasts | seed42 | validation example", ha="center", fontsize=14)
    header.text(.5, .52, "Same prespecified validation example and edges across models; shared y-scale per edge", ha="center", fontsize=9)
    header.legend([Line2D([], [], color="#1f77b4"), Line2D([], [], color="#ff7f0e")],
                  ["True", "Predicted"], loc="lower center", ncol=2, frameon=False)
    axes = body.subplots(4, len(runs), squeeze=False, sharex=True, sharey="row")
    for row, edge in enumerate(edges):
        for col, run in enumerate(runs):
            pred, true = aligned[run.experiment_id]
            ax = axes[row, col]
            ax.plot(minutes, true[:, edge], color="#1f77b4", lw=1.25)
            ax.plot(minutes, pred[:, edge], color="#ff7f0e", lw=1.15)
            ax.axvspan(minutes[0], minutes[17], color="gray", alpha=.10)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f"{run.experiment_id}\n{LABELS[run.experiment_id]}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Edge {edge}\nFisher-z", fontsize=10)
            if row == 3:
                ax.set_xlabel("Minutes from first future window")
    return save(fig, path)


def summary_charts(runs, public, out):
    ids = list(RUNS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    metrics = [("long_edge_mse", "Long MSE (lower is better)"),
               ("time_pearson", "Raw FC time Pearson"),
               ("template_adjusted_time_pearson", "Template-adjusted time Pearson"),
               ("dynamic_difference_time_pearson", "Residual difference time Pearson"),
               ("raw_edge_pearson", "Raw spatial Pearson (all future windows)"),
               ("dynamic_std_ratio", "Residual dynamic std ratio (target = 1)")]
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [public["runs"][e]["aggregate"][key] for e in ids]
        ax.bar(ids, values, width=.42, color=COLORS)
        ax.set_xlim(-.5, 3.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
        if min(values) < 0:
            ax.axhline(0, color="black", lw=.7)
        for i, value in enumerate(values):
            ax.annotate(f"{value:.6f}", (i, value), xytext=(0, 5 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.24)
    fig.suptitle("AE to PCA substitution | best checkpoints | seed42 | 158 validation subjects\nE0036/E0050: FC1 only; E0037/E0051: FC1 + SC", fontsize=13)
    save(fig, out / f"{PREFIX}_metrics.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    contrasts = public["contrasts"]
    for ax, key, title in zip(axes, ["long_edge_mse", "time_pearson", "template_adjusted_time_pearson"],
                              ["Long MSE difference\nnegative favors new", "Raw time Pearson difference\npositive favors new", "Template-adjusted time r difference\npositive favors new"]):
        for i, (label, comparison) in enumerate(contrasts.items()):
            r = comparison[key]
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=COLORS[i+1], lw=3)
            ax.plot(r["mean_difference"], i, "o", color=COLORS[i+1])
            if key == "long_edge_mse" and label != "E0051 - E0050":
                ci = r["family_adjusted_ci"]
                ax.plot([ci["ci_low"], ci["ci_high"]], [i, i], color=COLORS[i+1], lw=1)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set(yticks=range(3), yticklabels=list(contrasts), title=title, ylim=(-.6, 2.6), xlabel="New minus reference")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Paired subject bootstrap | 10,000 replicates | seed 20261026 | validation\nThick: 95% CI; thin MSE lines: 97.5% intervals for the two primary AE contrasts", fontsize=12)
    save(fig, out / f"{PREFIX}_paired_intervals.png")


def main(only=None):
    runs = {e: _load_run(ROOT, e, rid) for e, rid in RUNS.items()}
    out = ROOT / "outputs/E0051/visual"
    out.mkdir(parents=True, exist_ok=True)
    public = {"split": "val", "seed": 42, "n_subjects": 158, "runs": {}, "contrasts": {},
              "test_data_read": False, "fixed_edges": fixed_edge_indices().tolist(), "sources_sha256": {}}
    public["edge_labels"] = edge_labels(ROOT, runs["E0051"].config, fixed_edge_indices())
    for e, run in runs.items():
        evaluation = read(run.run_dir / "evaluation_val.json")
        public["runs"][e] = {"run_id": run.run_id, "aggregate": evaluation["aggregate"],
                              "selection": read(run.run_dir / "metrics_best.json"),
                              "dynamic_audit": aggregate_only(read(run.run_dir / "dynamic_audit_val/summary.json"))}
        for name in ("config_resolved.yaml", "metrics_best.json", "evaluation_val.json", "checkpoints/best.pt", "dynamic_audit_val/summary.json"):
            path = run.run_dir / name
            public["sources_sha256"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for new, old in [("E0050", "E0036"), ("E0051", "E0037"), ("E0051", "E0050")]:
        public["contrasts"][f"{new} - {old}"] = aggregate_only(read(runs[new].run_dir / f"comparison_{old}_val.json"))
    dump(out / f"{PREFIX}_aggregate.json", public)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = {e: (run, *_prediction(ROOT, run, device)) for e, run in runs.items()}
    for new, old in [("E0050", "E0036"), ("E0051", "E0037")]:
        if only and only != new:
            continue
        run = runs[new]
        dest = ROOT / "outputs" / new / "visual"
        dest.mkdir(parents=True, exist_ok=True)
        overview(run, dest / f"{run.run_id}_val_overview.png")
        dynamics([runs[old], run], entries, dest / f"{old}_{new}_val_fixed_edges.png")
        plot_dynamic_calibration(run, [runs[old], run], dest)
        plot_horizon_audit(run, [runs[old], run], dest)
    if not only:
        dynamics(list(runs.values()), entries, out / f"{PREFIX}_fixed_edges.png")
        summary_charts(runs, public, out)
    for e in ("E0050", "E0051"):
        dest = ROOT / "outputs" / e / "visual"
        if dest.exists():
            dump(dest / f"{e}_val_visual_manifest.json", {
                "runs": RUNS, "split": "val", "seed": 42, "fixed_edges": fixed_edge_indices().tolist(),
                "representative_example": "project-prespecified validation example; identifier withheld",
                "images": [p.name for p in sorted(dest.glob("*.png"))],
                "aggregate_source": "outputs/E0051/visual/" + PREFIX + "_aggregate.json",
                "script": "build_pca_ae_comparison.py" if e == "E0051" else "build_e0050_visuals.py",
                "test_data_read": False})
    print("Saved validation figures and aggregate-only provenance.", flush=True)


if __name__ == "__main__":
    main()
