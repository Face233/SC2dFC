"""E0044 scientific PNGs from frozen diagnostic CSV/JSON and private examples.

This module is standalone so a copy stored with a diagnostic can regenerate its
images without importing a later version of the scdfc implementation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


COLORS = {"identity": "#777777", "train_static": "#bcbd22", "train_template": "#8c564b",
          "oracle_mean": "#17becf", "pca": "#2ca02c", "ae": "#ff7f0e", "target": "#1f77b4"}


def render_diagnostic(diagnostic_dir):
    directory = Path(diagnostic_dir).resolve()
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((directory/"config_resolved.yaml").read_text(encoding="utf-8"))
    selection = json.loads((directory/"selection.json").read_text(encoding="utf-8"))
    identity = report["diagnostic_id"]
    out = directory.parent.parent/"visual"
    out.mkdir(exist_ok=True)
    with (directory/"per_subject_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    images = []
    values = report["summary"]["val"]["full"]
    methods = ("identity", "train_static", "train_template", "oracle_mean", "pca", "ae")
    labels = ("Identity", "Train\nstatic", "Train\ntemplate", "Oracle\nmean", "PCA", "A0003\nAE")
    figure = plt.figure(figsize=(16, 5), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(.16, .84))
    header.text(.5, .5, f"E0044 val / full scan — {identity}", ha="center", va="center", fontsize=11)
    for ax, metric, title in zip(body.subplots(1, 4), ("mse", "dynamic_nmse", "time_r", "std_ratio_mean"),
                                 ("Fisher-z MSE", "Centered dynamic NMSE", "Edge temporal Pearson r", "Predicted / true temporal SD")):
        for i, method in enumerate(methods):
            item = values[method][metric]
            if item["mean"] is None:
                ax.text(i, .03, "null", rotation=90, ha="center", transform=ax.get_xaxis_transform(), fontsize=8)
            else:
                mean = item["mean"]
                errors = [[max(0, mean-item["ci_low"])], [max(0, item["ci_high"]-mean)]]
                ax.bar(i, mean, width=.42, color=COLORS[method], yerr=errors, capsize=3)
        ax.set_xticks(range(len(methods)), labels, fontsize=8)
        ax.set_xlim(-.5, len(methods)-.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.2)
    path = out/f"{identity}_overview.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    images.append(path)

    figure = plt.figure(figsize=(8, 6), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(.16, .84))
    header.text(.5, .5, f"E0044 val / full scan — {identity}", ha="center", va="center", fontsize=9)
    axis = body.subplots()
    for method in ("pca", "ae"):
        group = [r for r in rows if r["split"] == "val" and r["interval"] == "full" and r["method"] == method
                 and r["time_r"] and r["std_ratio_mean"]]
        axis.scatter([float(r["time_r"]) for r in group], [float(r["std_ratio_mean"]) for r in group],
                     label=f"{method.upper()} (n={len(group)})", color=COLORS[method], alpha=.65, s=22)
    axis.axhline(1, color="#888888", linestyle=":", linewidth=1)
    axis.axvline(1, color="#888888", linestyle=":", linewidth=1)
    axis.set_xlabel("Mean edge temporal Pearson r")
    axis.set_ylabel("Mean edge reconstructed / true temporal SD")
    axis.grid(alpha=.2)
    axis.legend()
    path = out/f"{identity}_shape_amplitude.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    images.append(path)

    with np.load(directory/"private_examples.npz", allow_pickle=False) as examples:
        subjects = examples["subjects"].tolist()
        edges = examples["edge_indices"].tolist()
        for interval, stop in (("full", examples["target"].shape[1]), ("history20", 20)):
            for centered in (False, True):
                series = {method: examples[method][:, :stop].astype(np.float64) for method in ("target", "ae", "pca")}
                if centered:
                    series = {key: value-value.mean(axis=1, keepdims=True) for key, value in series.items()}
                minutes = np.arange(stop)*float(config["data"]["stride"])*float(config["data"]["tr_seconds"])/60
                figure = plt.figure(figsize=(max(7, 3.7*len(subjects)), max(7, 2.2*len(edges))), layout="constrained")
                header, body = figure.subfigures(2, 1, height_ratios=(.08, .92))
                mode = "each curve temporally centered" if centered else "raw Fisher-z"
                header.text(.5, .76, f"E0044 val / {interval} / {mode}\n{identity}", ha="center", va="center", fontsize=10)
                header.legend([Line2D([], [], color=COLORS[k]) for k in series], ["True FC", "A0003 AE", "PCA"],
                              ncol=3, loc="lower center", frameon=False)
                axes = body.subplots(len(edges), len(subjects), squeeze=False, sharex=True, sharey="row")
                for row, edge in enumerate(edges):
                    all_values = np.concatenate([v[:, :, row].ravel() for v in series.values()])
                    low, high = all_values.min(), all_values.max()
                    pad = max((high-low)*.07, .02)
                    for col, subject in enumerate(subjects):
                        ax = axes[row, col]
                        for method, data in series.items():
                            ax.plot(minutes, data[col, :, row], color=COLORS[method],
                                    linewidth=1.0, linestyle="--" if method == "pca" else "-")
                        ax.set_ylim(low-pad, high+pad)
                        ax.grid(alpha=.2)
                        ax.tick_params(labelsize=8)
                        if row == 0:
                            ax.set_title(selection["display_labels"][subject], fontsize=10)
                        if col == 0:
                            ax.set_ylabel(f"edge {edge}\nFisher-z", fontsize=9)
                        if row == len(edges)-1:
                            ax.set_xlabel("Minutes from first scan window", fontsize=8)
                path = out/f"{identity}_{interval}_{'centered' if centered else 'raw'}_curves.png"
                figure.savefig(path, dpi=180)
                plt.close(figure)
                images.append(path)
    # A script adjacent to the PNGs runs this diagnostic's frozen renderer copy.
    script = out/f"{identity}_rebuild.py"
    script.write_text(
        '"""E0044 val; frozen JSON/CSV/private_examples inputs; regenerate six PNGs."""\n'
        "from pathlib import Path\nimport runpy\n"
        f"diagnostic = Path(__file__).resolve().parent.parent / 'diagnostics' / {identity!r}\n"
        "renderer = runpy.run_path(str(diagnostic / 'ae_diagnostic_visual.py'))\n"
        "renderer['render_diagnostic'](diagnostic)\n", encoding="utf-8")
    manifest = {"experiment_id": report["experiment_id"], "diagnostic_id": identity, "split": "val",
                "source": str(directory), "images": [p.name for p in images], "script": script.name,
                "selection": selection, "private_inputs": True}
    (out/f"{identity}_visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return images
