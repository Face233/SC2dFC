"""E0061-E0066 fixed validation examples and frozen subject-alpha curves.

Run IDs below pin the patience20 runs. Inputs are their config_resolved.yaml,
checkpoints/evaluation_val.json, pinned E0036/E0037/E0032 artifacts, and the
E0037 reference plot's four validation participants / six edges. No test data.
Outputs: each experiment's *_val_multisubject_comparison.png; two backbone
comparisons; six-panel alpha / delta-alpha / alpha-window18-60 figures.
Private arrays and participant mappings stay under visual/private/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from scdfc.alpha_transfer import pinned_path, verify_transfer_bindings
from scdfc.data import DFCSequenceDataset
from scdfc.evaluation import _load_model
from scdfc.management import config_sha256, file_sha256, verify_artifact
from scdfc.sc_alpha import tensor_digest

RUNS = {
    "E0061": "E0061-s42-20261028T033946Z-88b6157",
    "E0062": "E0062-s42-20261028T034508Z-88b6157",
    "E0063": "E0063-s42-20261028T034901Z-88b6157",
    "E0064": "E0064-s42-20261028T042222Z-1b6e0b2",
    "E0065": "E0065-s42-20261028T042759Z-1b6e0b2",
    "E0066": "E0066-s42-20261028T043331Z-1b6e0b2",
}
OUT = ROOT/"outputs/E0066/visual"
PRIVATE = OUT/"private"
REFERENCE = ROOT/"outputs/E0037/visual/E0032_E0036_E0037_multisubject_comparison.json"
SUBJECT_COLORS = ["#007C91", "#DD7524", "#6D4BA5", "#B9476E"]
COLORS = {"true": "#1f77b4", "E0032": "#929292", "E0036": "#2b8c50", "E0037": "#2b8c50",
          "E0061": "#e57a17", "E0062": "#8b55b5", "E0063": "#c23a3a",
          "E0064": "#e57a17", "E0065": "#8b55b5", "E0066": "#c23a3a", "group": "#737373"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def frozen_config(eid):
    directory = ROOT/"outputs"/eid/"runs"/RUNS[eid]
    cfg = yaml.safe_load((directory/"config_resolved.yaml").read_text(encoding="utf-8-sig"))
    cfg["paths"]["root"] = str(ROOT)
    meta = read_json(directory/"metadata.json")
    if meta["status"] != "COMPLETED" or meta["run_id"] != RUNS[eid] or meta["config_sha256"] != config_sha256(cfg):
        raise ValueError(f"Frozen run identity/configuration mismatch: {eid}")
    if cfg["training"]["patience"] != 20 or cfg["model"]["fc_representation"] != "autoencoder":
        raise ValueError("Use the completed patience20 AE runs")
    return directory, cfg


def inputs():
    ref = read_json(REFERENCE)
    if ref["split"] != "val" or len(ref["subjects"]) != 4 or len(ref["edge_indices"]) != 6:
        raise ValueError("Expected reference selection: four validation subjects and six edges")
    configs, sources = {}, {}
    for eid in RUNS:
        directory, cfg = frozen_config(eid)
        checkpoint = directory/"checkpoints/best.pt"
        evaluation = read_json(directory/"evaluation_val.json")
        digest = file_sha256(checkpoint)
        if digest != evaluation["checkpoint_sha256"] or evaluation["split"] != "val":
            raise ValueError(f"Checkpoint differs from evaluated model: {eid}")
        configs[eid] = cfg
        sources[eid] = {"run_id": RUNS[eid], "config_sha256": config_sha256(cfg),
                        "checkpoint_sha256": digest, "evaluation_sha256": file_sha256(directory/"evaluation_val.json"),
                        "alpha_source": cfg["artifacts"]["sc_alpha"], "reference": cfg["artifacts"]["residual_reference"],
                        "best_epoch_one_based": evaluation["checkpoint_selection"]["epoch"]+1}
    binding = {"run_sources": sources, "reference_sha256": file_sha256(REFERENCE),
               "reference_edges": ref["edge_indices"], "selection_sha256": hashlib.sha256(json.dumps(ref["subjects"]).encode()).hexdigest(),
               "script_sha256": file_sha256(Path(__file__)),
               "source_modules": {name: file_sha256(ROOT/"src/scdfc"/name) for name in
                                   ("alpha_transfer.py", "sc_alpha.py", "evaluation.py", "training.py", "data.py", "models/sequence.py")}}
    return ref, configs, binding


@torch.no_grad()
def collect(ref, configs, device):
    cfg = configs["E0061"]
    stats = pinned_path(cfg, "training_stats")
    dataset = DFCSequenceDataset(cfg, 83, "val", stats, "fc1_only")
    subjects = [str(s) for s, _ in dataset.samples]
    indices = []
    for subject in ref["subjects"]:
        matches = [i for i, s in enumerate(subjects) if s == subject]
        if len(matches) != 1:
            raise ValueError("Reference participant not uniquely in validation split")
        indices.append(matches[0])
    samples = [dataset[i] for i in indices]
    sc = torch.stack([s["sc_matrix"] for s in samples]).to(device)
    sc_edges = torch.stack([s["sc_edges"] for s in samples]).to(device)
    warmup = torch.stack([s["fc_warmup"] for s in samples]).to(device)
    true = np.stack([s["fc_future"].numpy() for s in samples])
    edges = np.asarray(ref["edge_indices"], dtype=int)
    starts = samples[0]["window_starts"].numpy()[1:]
    if not all(np.array_equal(s["window_starts"].numpy()[1:], starts) for s in samples):
        raise ValueError("Prediction timelines differ")
    with np.load(stats) as f:
        template = f["group_template"].copy()
        context = f["context_template"].copy()
    with np.load(pinned_path(cfg, "alpha_fit")) as f:
        alpha0 = f["alpha_raw"].astype(np.float32)
    baseline = template[None]+alpha0[None, :, None]*(warmup.cpu().numpy()-context)[:, None]
    arrays = {"true": true[:, :, edges], "E0032": baseline[:, :, edges], "group": template[:, edges],
              "alpha0": alpha0, "minutes": (starts-starts[0])*cfg["data"]["tr_seconds"]/60.0}
    # Every subject/edge is fixed by the historical reference, before new outputs.
    loaded_originals = set()
    for eid, current in configs.items():
        _, old_cfg = verify_transfer_bindings(current)
        directory = ROOT/"outputs"/eid/"runs"/RUNS[eid]
        model, payload = _load_model(current, 83, directory/"checkpoints/best.pt", stats, device, verify_artifact(current))
        if tensor_digest(model.sc_alpha_component) != payload["sc_alpha_state_sha256"]:
            raise ValueError("Frozen alpha differs from its final training state")
        result = model(sc, sc_edges, warmup)
        alpha = model.sc_alpha_component(sc).cpu().numpy()
        source = model.sc_alpha_component
        # Test source-component replay through the actual saved transfer model.
        expected = model.group_template[None]+torch.as_tensor(alpha, device=device)[:, :, None]*(warmup-model.context_template[None])[:, None]
        torch.testing.assert_close(result.residual_baseline, expected)
        arrays[eid] = result.fc_z_edges.cpu().numpy()[:, :, edges]
        arrays[eid+"_alpha"] = alpha
        if not np.isfinite(arrays[eid]).all() or not np.isfinite(alpha).all():
            raise FloatingPointError("Nonfinite curves")
        original = current["artifacts"]["residual_reference"]["experiment_id"]
        del model, source, result, expected
        if original not in loaded_originals:
            old_model, _ = _load_model(old_cfg, 83, pinned_path(current, "residual_reference"), stats, device, verify_artifact(old_cfg))
            arrays[original] = old_model(sc, sc_edges, warmup).fc_z_edges.cpu().numpy()[:, :, edges]
            loaded_originals.add(original)
            del old_model
        print(f"Replayed {eid} for four fixed validation subjects", flush=True)
    for a, b in (("E0061", "E0064"), ("E0062", "E0065"), ("E0063", "E0066")):
        np.testing.assert_array_equal(arrays[a+"_alpha"], arrays[b+"_alpha"])
    write_json(PRIVATE/"subject_mapping.json", {"split": "val", "selection": "exact reference E0037 selection; no new selection",
                                               "mapping": {f"S{i+1}": s for i, s in enumerate(ref["subjects"])}})
    return arrays


def footer_base():
    return "val only · 四被试/六条边沿用指定 E0037 参考图 · S1–S4 为匿名代号 · 第五列为训练集组均值 · 竖虚线：长期区间起点"


def fc_figure(arrays, ref, experiments, destination, name, title, metadata):
    fig = plt.figure(figsize=(18, 19.5), layout="constrained")
    head, body, foot = fig.subfigures(3, 1, height_ratios=(.095, .87, .035))
    head.text(.5, .81, title, ha="center", va="center", fontsize=15, weight="bold")
    run_text = " | ".join(RUNS[e] for e in experiments if e in RUNS)
    head.text(.5, .58, run_text, ha="center", fontsize=8.5, color="#444444")
    plotted = ["E0032", *experiments]
    def line_color(eid):
        return "#ff7f0e" if len(experiments) == 2 and eid in RUNS else COLORS[eid]
    handles = [Line2D([], [], color=COLORS["true"], lw=1.4, label="真实 FC")]
    for eid in plotted:
        handles.append(Line2D([], [], color=line_color(eid), lw=1.2,
                              ls=":" if eid == "E0032" else "--" if eid in ("E0036", "E0037") else "-",
                              label=eid+" 预测"))
    handles.append(Line2D([], [], color=COLORS["group"], lw=1.2, label="训练集组均值（第五列）"))
    head.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=9)
    axes = body.subplots(6, 5, sharex=True)
    x = arrays["minutes"]
    # Identical per-edge range across all six single figures and both comparisons.
    all_values = np.concatenate([arrays["true"].reshape(-1, 6), arrays["group"], arrays["E0032"].reshape(-1, 6),
                                 *[arrays[e].reshape(-1, 6) for e in ["E0036", "E0037", *RUNS]]])
    low, high = all_values.min(0), all_values.max(0)
    for row, (edge, label) in enumerate(zip(ref["edge_indices"], ref["edge_labels"])):
        pad = max((high[row]-low[row])*.05, .02)
        for col in range(5):
            ax = axes[row, col]
            if col < 4:
                ax.plot(x, arrays["true"][col, :, row], color=COLORS["true"], lw=1.1, zorder=3)
                for eid in plotted:
                    ax.plot(x, arrays[eid][col, :, row], color=line_color(eid), lw=.95,
                            ls=":" if eid == "E0032" else "--" if eid in ("E0036", "E0037") else "-", alpha=.94)
            else:
                ax.plot(x, arrays["group"][:, row], color=COLORS["group"], lw=1.2)
            ax.axvline(x[17], color="#aaaaaa", ls="--", lw=.65, alpha=.65)
            ax.set_ylim(low[row]-pad, high[row]+pad)
            ax.grid(alpha=.17, lw=.5)
            ax.tick_params(labelsize=7.5)
            if row == 0:
                ax.set_title(f"验证被试 S{col+1}" if col < 4 else "训练集\n组平均真实时序", fontsize=10, pad=8)
            if col == 0:
                label = textwrap.fill(label.replace(" ↔ ", " <-> "), width=26)
                ax.set_ylabel(f"edge {edge}\n{label}\nFisher-z", fontsize=8)
            if row == 5:
                ax.set_xlabel("距首个预测窗（分钟）", fontsize=9)
    foot.text(.5, .5, footer_base(), ha="center", va="center", fontsize=8, color="#555555")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/(name+".png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    write_json(destination/(name+".json"), {**metadata, "plotted_experiments": plotted, "image": path.relative_to(ROOT).as_posix(),
                                           "layout": "six reference edges x four fixed val subjects + training mean", "shared_y_limits": True})
    return path


def alpha_figure(arrays, metadata, kind):
    delta = kind == "delta"
    zoom = kind == "window18_60"
    x = np.arange(1, 224)
    mask = (x >= 18) & (x <= 60) if zoom else np.ones(223, dtype=bool)
    series = {eid: arrays[eid+"_alpha"]-(arrays["alpha0"][None] if delta else 0) for eid in RUNS}
    base = np.zeros(223) if delta else arrays["alpha0"]
    limits = np.concatenate([v[:, mask].ravel() for v in series.values()]+[base[mask]])
    low, high = limits.min(), limits.max()
    pad = max((high-low)*.08, .002 if delta or zoom else .01)
    fig = plt.figure(figsize=(16, 9.8), layout="constrained")
    head, body, foot = fig.subfigures(3, 1, height_ratios=(.16, .77, .07))
    title = "个体化 α：未来第18–60窗" if zoom else "个体化修正 Δα：全部223个未来窗" if delta else "个体化 α：全部223个未来窗"
    head.text(.5, .83, "E0061–E0066 · 验证集相同四名被试 · "+title, ha="center", fontsize=14, weight="bold")
    head.text(.5, .6, "每种颜色始终对应同一被试；各面板共用坐标尺度，未对曲线做平滑或标准化", ha="center", fontsize=10)
    handles = [Line2D([], [], color=c, lw=1.5, label=f"验证被试 S{i+1}") for i, c in enumerate(SUBJECT_COLORS)]
    handles.append(Line2D([], [], color="#444444", ls="--", lw=1.2, label="Δα=0" if delta else r"E0032 共享 $\alpha_0$"))
    head.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    axes = body.subplots(2, 3, sharex=True, sharey=True)
    for i, eid in enumerate(RUNS):
        ax = axes.flat[i]
        for s, color in enumerate(SUBJECT_COLORS):
            ax.plot(x[mask], series[eid][s, mask], color=color, lw=1.35)
        ax.plot(x[mask], base[mask], color="#444444", ls="--", lw=1.1)
        if not zoom:
            ax.axvline(18, color="#aaaaaa", ls=":", lw=.8)
        ax.set_xlim(18 if zoom else 1, 60 if zoom else 223)
        ax.set_ylim(low-pad, high+pad)
        source = metadata["run_sources"][eid]["alpha_source"]["experiment_id"]
        backbone = "E0036 / FC1残差" if i < 3 else "E0037 / FC1+SC残差"
        kind_name = ["Linear-223", "MLP-223", "GCN-223"][i % 3]
        ax.set_title(f"{eid} · {kind_name}（{source}）\n{backbone}", fontsize=10)
        ax.grid(alpha=.2)
        if i % 3 == 0:
            ax.set_ylabel("修正量 Δα" if delta else "最终个体化 α")
        if i >= 3:
            ax.set_xlabel("未来窗口编号（从1开始）")
    foot.text(.5, .68, "α 组件冻结：E0061=E0064、E0062=E0065、E0063=E0066（同一被试逐点一致，已核对）", ha="center", fontsize=10)
    foot.text(.5, .27, "与多被试 FC 图使用同一 S1–S4；这四个示例用于看形状，不代表总体统计结论。val only", ha="center", fontsize=9, color="#555555")
    path = OUT/f"E0061_E0066_val_multisubject_alpha_{kind}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    write_json(path.with_suffix(".json"), {**metadata, "kind": kind, "image": path.relative_to(ROOT).as_posix(),
                                         "alpha_pairs_identical": [["E0061", "E0064"], ["E0062", "E0065"], ["E0063", "E0066"]]})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refresh", action="store_true", help="Replay checkpoints instead of verified private plot cache")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    ref, configs, binding = inputs()
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache = PRIVATE/"E0061_E0066_val_multisubject_plot_cache.npz"
    cache_manifest = PRIVATE/"E0061_E0066_val_multisubject_cache.json"
    if (not args.refresh and cache.exists() and cache_manifest.exists()
            and read_json(cache_manifest).get("signature") == signature
            and read_json(cache_manifest).get("cache_sha256") == file_sha256(cache)):
        with np.load(cache) as f:
            arrays = {k: f[k] for k in f.files}
        print("Using verified private figure cache", flush=True)
    else:
        arrays = collect(ref, configs, torch.device(args.device))
        np.savez_compressed(cache, **arrays)
        write_json(cache_manifest, {"signature": signature, "input_bindings": binding, "cache_sha256": file_sha256(cache)})
    metadata = {**binding, "split": "val", "subjects": ["S1", "S2", "S3", "S4"],
                "reference_selection": REFERENCE.relative_to(ROOT).as_posix(), "edge_indices": ref["edge_indices"],
                "edge_labels": ref["edge_labels"], "raw_identifiers": "local private subject_mapping.json only",
                "alpha_frozen": True, "smoothing": False, "test_data_read": False}
    paths = []
    for eid in RUNS:
        original = "E0036" if int(eid[1:]) < 64 else "E0037"
        dest = ROOT/"outputs"/eid/"visual"
        paths.append(fc_figure(arrays, ref, [original, eid], dest, f"{eid}_val_multisubject_comparison",
                               f"{eid} · {original}＋冻结{configs[eid]['artifacts']['sc_alpha']['experiment_id']} α · 多被试时序对比", metadata))
        # Keep a complete redraw script beside every individual experiment image.
        script = dest/Path(__file__).name
        if script.resolve() != Path(__file__).resolve():
            script.write_bytes(Path(__file__).read_bytes())
    for original, group in (("E0036", ["E0061", "E0062", "E0063"]), ("E0037", ["E0064", "E0065", "E0066"])):
        dest = ROOT/"outputs"/group[-1]/"visual"
        paths.append(fc_figure(arrays, ref, [original, *group], dest,
                               f"{group[0]}_{group[-1]}_val_multisubject_comparison",
                               f"{group[0]}–{group[-1]} · {original}骨干 · 三种223参数 α 的多被试对比", metadata))
    for kind in ("full", "delta", "window18_60"):
        paths.append(alpha_figure(arrays, metadata, kind))
    import csv
    with (OUT/"E0061_E0066_val_selected_subjects_alpha.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "subject_alias", "future_window", "alpha", "alpha_shared", "delta_alpha"])
        for eid in RUNS:
            for s in range(4):
                for t in range(223):
                    a = float(arrays[eid+"_alpha"][s, t]); base = float(arrays["alpha0"][t])
                    writer.writerow([eid, f"S{s+1}", t+1, a, base, a-base])
    write_json(OUT/"E0061_E0066_multisubject_visual_manifest.json", {**metadata, "images": [p.relative_to(ROOT).as_posix() for p in paths]})
    lines = ["# E0061–E0066 多被试图与α曲线", "", "验证集固定四被试 S1–S4、六条边，沿用用户指定 E0037 参考图的选择；第五列为训练集组均值。", "",
             "FC图同一条边跨所有实验共用y轴范围。α曲线不平滑、不截断；六面板共用尺度；另提供Δα和第18–60窗放大图。", "",
             "α为冻结组件，同被试下E0061/E0064、E0062/E0065、E0063/E0066的α逐点相同，已验证。", ""]
    lines += [f"- [{p.stem}]({p.relative_to(OUT).as_posix() if p.parent == OUT else '../../'+p.parent.parent.name+'/visual/'+p.name})" for p in paths]
    (OUT/"E0061_E0066_multisubject_visual_index.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
