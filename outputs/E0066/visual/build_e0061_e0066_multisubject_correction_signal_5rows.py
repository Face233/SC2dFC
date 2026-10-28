"""E0061-E0066 validation: five-row FC / alpha-correction comparison.

Inputs: pinned run_sources in private/E0061_E0066_val_multisubject_cache.json,
its checksum-bound NPZ, private participant mapping, frozen E0061 config,
training statistics and the original validation FC cache. No test data.
Create the private inputs locally with build_e0061_e0066_multisubject.py;
they require the private validation data and frozen checkpoint artifacts.
Outputs: E0061_E0066_val_multisubject_correction_signal_5rows.png and JSON.
Rows: raw FC; actual FC correction vs true residual; separately standardized
Linear, MLP and GCN corrections vs true residual. Four anonymous participants.
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
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from scdfc.data import DFCSequenceDataset
from scdfc.management import file_sha256, config_sha256

HERE = Path(__file__).resolve().parent
VIS = ROOT/"outputs/E0066/visual"
private = VIS/"private"
manifest = json.loads((private/"E0061_E0066_val_multisubject_cache.json").read_text(encoding="utf-8"))
cache = private/"E0061_E0066_val_multisubject_plot_cache.npz"
if file_sha256(cache) != manifest["cache_sha256"]:
    raise ValueError("Private curve cache checksum differs")
with np.load(cache) as f:
    data = {k: f[k] for k in f.files}
reference = json.loads((ROOT/"outputs/E0037/visual/E0032_E0036_E0037_multisubject_comparison.json").read_text(encoding="utf-8"))
mapping = json.loads((private/"subject_mapping.json").read_text(encoding="utf-8"))["mapping"]
source = manifest["input_bindings"]["run_sources"]["E0061"]
run = ROOT/"outputs/E0061/runs"/source["run_id"]
config = yaml.safe_load((run/"config_resolved.yaml").read_text(encoding="utf-8"))
config["paths"]["root"] = str(ROOT)
if config_sha256(config) != source["config_sha256"]:
    raise ValueError("Frozen configuration changed")
stats = ROOT/config["artifacts"]["training_stats"]["path"]
if file_sha256(stats) != config["artifacts"]["training_stats"]["sha256"]:
    raise ValueError("Training template changed")
dataset = DFCSequenceDataset(config, 83, "val", stats, "fc1_only")
with np.load(stats) as f:
    context = f["context_template"].copy()
    template = f["group_template"].copy()

# Use the first of the six previously fixed edges; no output-based selection.
edge = int(reference["edge_indices"][0])
edge_label = reference["edge_labels"][0].replace(" ↔ ", " <-> ")
offsets = []
for i in range(4):
    subject = mapping[f"S{i+1}"]
    matches = [j for j, (s, _) in enumerate(dataset.samples) if s == subject]
    if len(matches) != 1:
        raise ValueError("Selected sample not uniquely in validation")
    sample = dataset[matches[0]]
    np.testing.assert_array_equal(sample["fc_future"].numpy()[:, edge], data["true"][i, :, 0])
    offsets.append(float(sample["fc_warmup"].numpy()[edge]-context[edge]))
offsets = np.array(offsets)
for a, b in (("E0061", "E0064"), ("E0062", "E0065"), ("E0063", "E0066")):
    np.testing.assert_array_equal(data[a+"_alpha"], data[b+"_alpha"])

start = 17
x = np.arange(1, 224)[start:]
true = data["true"][:, start:, 0].astype(float)
baseline = data["E0032"][:, start:, 0].astype(float)
residual = true-baseline
methods = {"Linear-223": "E0061", "MLP-223": "E0062", "GCN-223": "E0063"}
colors = {"Linear-223": "#e57a17", "MLP-223": "#8b55b5", "GCN-223": "#c23a3a"}
correction = {name: (data[eid+"_alpha"][:, start:].astype(float)-data["alpha0"][None, start:])*offsets[:, None]
              for name, eid in methods.items()}

def zscore(values):
    scale = values.std(1, keepdims=True)
    if np.any(scale < 1e-12):
        raise ValueError("Flat example; temporal shape is undefined")
    return (values-values.mean(1, keepdims=True))/scale

ztrue = zscore(residual)
zcorr = {name: zscore(c) for name, c in correction.items()}
pearson = {name: np.mean(ztrue*z, axis=1) for name, z in zcorr.items()}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9})
fig = plt.figure(figsize=(17, 15.5), layout="constrained")
header, body, footer = fig.subfigures(3, 1, height_ratios=(.085, .865, .05))
header.text(.5, .82, "不同被试的 α 修正量与真实信号比较｜长期预测窗 18–223", ha="center", fontsize=16, weight="bold")
header.text(.5, .57, f"固定边 {edge}：{edge_label}｜与前图相同的验证被试 S1–S4", ha="center", fontsize=10)
handles = [Line2D([], [], color="#1f77b4", lw=1.5, label="真实 FC / 真实残差"),
           Line2D([], [], color="#666666", ls="--", lw=1.2, label="E0032 共享基线")]
handles += [Line2D([], [], color=colors[name], lw=1.3, label=name+" 的 FC 修正") for name in methods]
header.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
axes = body.subplots(5, 4, sharex=True, sharey="row")
for s in range(4):
    axes[0, s].plot(x, true[s], color="#1f77b4", lw=1.2)
    axes[0, s].plot(x, baseline[s], color="#666666", lw=1.1, ls="--")
    axes[0, s].set_title(f"验证被试 S{s+1}", fontsize=11, pad=8)
    axes[1, s].plot(x, residual[s], color="#1f77b4", lw=1.2)
    for method_index, name in enumerate(methods):
        axes[1, s].plot(x, correction[name][s], color=colors[name], lw=1.2)
        row = method_index+2
        axes[row, s].plot(x, ztrue[s], color="#1f77b4", lw=1.15)
        axes[row, s].plot(x, zcorr[name][s], color=colors[name], lw=1.1, alpha=.9)
        axes[row, s].text(.02, .98, f"{name} | r={pearson[name][s]:+.2f}",
                          transform=axes[row, s].transAxes, va="top", fontsize=8.5,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    for row in range(5):
        ax = axes[row, s]
        ax.grid(alpha=.18, lw=.5)
        ax.set_xlim(18, 223)
        if row > 0:
            ax.axhline(0, color="#aaaaaa", ls=":", lw=.7)
    axes[4, s].set_xlabel("未来窗口编号")

axes[0, 0].set_ylabel("真实 FC 与共享基线\nFisher-z")
axes[1, 0].set_ylabel("真实残差与实际修正量\nFisher-z（相同单位、同一纵轴）")
for row, name in enumerate(methods, 2):
    axes[row, 0].set_ylabel(f"{name} 修正 vs 真实残差\n分别标准化，仅比较形状")
raw = np.concatenate([true.ravel(), baseline.ravel()])
rem = np.concatenate([residual.ravel(), *[v.ravel() for v in correction.values()]])
shape = np.concatenate([ztrue.ravel(), *[v.ravel() for v in zcorr.values()]])
for row, values in enumerate((raw, rem, shape, shape, shape)):
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi-lo)*.09, .03)
    if row >= 2:
        pad = (hi-lo)*.16  # reserve room above traces for descriptive correlations
    for ax in axes[row]:
        ax.set_ylim(lo-pad, hi+pad)
footer.text(.5, .72, r"真实残差 = 真实 FC − E0032；实际 FC 修正 = $\Delta\alpha_s(t)\,[FC_s(0)-\mu_0]$。第二行保留真实幅度；后三行分别标准化，只比较形状。", ha="center", fontsize=9)
footer.text(.5, .3, "α 冻结：E0061/E0064、E0062/E0065、E0063/E0066 各对共享同一修正。示例不代表总体统计；val only。", ha="center", fontsize=9, color="#555555")
destination = HERE/"E0061_E0066_val_multisubject_correction_signal_5rows.png"
fig.savefig(destination, dpi=180)
plt.close(fig)
provenance = {
    "experiments": list(manifest["input_bindings"]["run_sources"]),
    "run_sources": manifest["input_bindings"]["run_sources"],
    "split": "val", "subjects": ["S1", "S2", "S3", "S4"],
    "selection": "First of six previously fixed reference edges; same four validation participants; no new output-based selection",
    "edge_index": edge, "edge_label": edge_label, "window_range_inclusive": [18, 223],
    "rows": ["raw true FC and E0032", "actual alpha FC correction and true residual in Fisher-z",
             "standardized Linear correction and true residual", "standardized MLP correction and true residual",
             "standardized GCN correction and true residual"],
    "correction_definition": "delta_alpha_s(t) * [FC_s(0)-mu_0]",
    "true_residual_definition": "true_FC_s(t)-E0032_baseline_s(t)",
    "shape_normalization": "Separately subtract temporal mean and divide by population temporal standard deviation on windows 18-223",
    "pearson_with_true_residual": {name: dict(zip(("S1", "S2", "S3", "S4"), map(float, values))) for name, values in pearson.items()},
    "alpha_frozen_pairs": [["E0061", "E0064"], ["E0062", "E0065"], ["E0063", "E0066"]],
    "cache_sha256": file_sha256(cache), "reference_sha256": file_sha256(ROOT/"outputs/E0037/visual/E0032_E0036_E0037_multisubject_comparison.json"),
    "script": Path(__file__).name, "script_sha256": file_sha256(Path(__file__)),
    "image": destination.name, "image_sha256": file_sha256(destination),
    "test_data_read": False, "raw_subject_identifiers_published": False,
}
destination.with_suffix(".json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print(destination)
print(json.dumps({"edge_index": edge, "split": "val", "window_range": [18, 223],
                  "pearson_with_true_residual": {name: dict(zip(("S1", "S2", "S3", "S4"), map(float, values))) for name, values in pearson.items()}}, ensure_ascii=False))
