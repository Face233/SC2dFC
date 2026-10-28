from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent
data = list(json.loads((root/"E0061_E0066_comparison_val.json").read_text(encoding="utf-8")).values())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels = [r["experiment_id"] for r in data]
for ax, key, title in zip(axes, ("long_edge_mse", "time_pearson", "dynamic_std_ratio"), ("Long MSE (lower is better)", "Edge temporal Pearson", "Dynamic std / true std")):
    ax.plot(labels, [r["aggregate"]["model"][key] for r in data], "o-", label="Retrained residual + frozen alpha")
    ax.plot(labels, [r["aggregate"]["original"][key] for r in data], "x--", label="Original AE residual")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(root/"E0061_E0066_metrics_val.png", dpi=180)
