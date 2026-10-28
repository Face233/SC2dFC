"""E0052-E0060 frozen final SC-alpha components; two train subjects, SC only.

Inputs: frozen config_resolved.yaml, component manifests/checkpoints, E0032
alpha_fit.npz, frozen train split and two SC matrices. No FC targets are read.
Outputs: E0052_E0060_personal_alpha_{segment,correction}.png, anonymous CSVs.
Run IDs are frozen below; subject mapping is local in private/.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from scdfc.config import resolve_path
from scdfc.data import load_sc
from scdfc.management import file_sha256, verify_data_bindings
from scdfc.sc_alpha import load_sc_alpha_component

RUNS = {
    "E0052": "E0052-s42-20261028T020855Z-0eb37e3",
    "E0053": "E0053-s42-20261028T021445Z-0eb37e3",
    "E0054": "E0054-s42-20261028T021613Z-0eb37e3",
    "E0055": "E0055-s42-20261028T023111Z-0eb37e3",
    "E0056": "E0056-s42-20261028T023304Z-0eb37e3",
    "E0057": "E0057-s42-20261028T023448Z-0eb37e3",
    "E0058": "E0058-s42-20261028T023813Z-0eb37e3",
    "E0059": "E0059-s42-20261028T024001Z-0eb37e3",
    "E0060": "E0060-s42-20261028T024149Z-0eb37e3",
}
OUT = Path(__file__).resolve().parent
COLORS = ["#007C91", "#DD7524"]


def frozen_config(eid, rid):
    run = ROOT / "outputs" / eid / "runs" / rid
    config = yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))
    config["paths"]["root"] = str(ROOT)
    return run, config


def main():
    first_run, config = frozen_config("E0052", RUNS["E0052"])
    manifest = verify_data_bindings(config)
    split = pd.read_csv(resolve_path(config, "split_csv"), dtype={"subject_id": str})
    train = sorted(split.loc[split.split == "train", "subject_id"].tolist())
    # Fixed random choice before inspecting any alpha outputs; no extreme-case selection.
    subjects = [train[i] for i in np.sort(np.random.default_rng(42).choice(len(train), 2, replace=False))]
    files = {item["path"].replace("\\", "/"): item for item in manifest["files"]}
    for subject in subjects:
        path = resolve_path(config, "sc_dir") / f"{subject}.csv"
        relative = path.relative_to(ROOT).as_posix()
        if file_sha256(path) != files[relative]["sha256"]:
            raise ValueError("Selected train SC differs from frozen manifest")
    sc = np.stack([load_sc(config, subject) for subject in subjects])
    private = OUT / "private"
    private.mkdir(exist_ok=True)
    (private / "personal_alpha_subjects.json").write_text(json.dumps({
        "split": "train", "selection": "sorted train IDs; default_rng(42), two without replacement",
        "mapping": dict(zip(("S1", "S2"), subjects))}, indent=2), encoding="utf-8")
    alpha, metadata = {}, {}
    baseline = None
    for eid, rid in RUNS.items():
        run, cfg = frozen_config(eid, rid)
        component = load_sc_alpha_component(run / "checkpoints/best.pt", cfg)
        prediction = component.predict_alpha(sc)
        current_base = component.payload["alpha_raw"].astype(np.float64)
        if baseline is None:
            baseline = current_base
        if not np.array_equal(baseline, current_base):
            raise ValueError("Models use different shared alpha references")
        alpha[eid] = prediction
        metadata[eid] = {"run_id": rid, "checkpoint_sha256": file_sha256(run / "checkpoints/best.pt"),
                         "encoder": cfg["model"]["sc_encoder"], "K": cfg["model"]["alpha_parameters"]}
    full = []
    for eid, values in {"E0032": np.tile(baseline, (2, 1)), **alpha}.items():
        for s in range(2):
            for t in range(len(baseline)):
                full.append({"experiment": eid, "subject": f"S{s+1}", "future_window": t+1,
                             "alpha": float(values[s, t]), "alpha_shared": float(baseline[t]),
                             "delta_alpha": float(values[s, t]-baseline[t])})
    table = pd.DataFrame(full)
    prefix = "E0052_E0060_personal_alpha"
    table.to_csv(OUT / f"{prefix}_all_windows.csv", index=False)
    table.loc[table.future_window.isin([1, 18, 60, 112, 180, 223])].to_csv(
        OUT / f"{prefix}_selected_windows.csv", index=False)
    x = np.arange(1, len(baseline)+1)
    segment_mask = (x >= 18) & (x <= 60)
    segment_values = np.concatenate([baseline[segment_mask], *[v[:, segment_mask].ravel() for v in alpha.values()]])
    segment_min, segment_max = segment_values.min(), segment_values.max()
    segment_margin = max((segment_max-segment_min)*.08, .005)
    for correction in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True, sharex=True, sharey=True)
        for i, (eid, values) in enumerate(alpha.items()):
            ax = axes.flat[i]
            for s in range(2):
                y = values[s]-baseline if correction else values[s]
                ax.plot(x, y, color=COLORS[s], lw=1.7, label=f"训练被试 S{s+1}")
            ax.plot(x, np.zeros_like(baseline) if correction else baseline,
                    color="#444444", lw=1.3, ls="--",
                    label="E0032 共享基线（Δα=0）" if correction else r"E0032 共享基线 $\alpha_0$")
            encoder = ["线性映射", "多层感知机（MLP）", "图卷积网络（HCP-GCN）"][i//3]
            kind = ["K=1：单参数恒定修正", "K=6：三次B样条修正", "K=223：逐时间点修正"][i%3]
            ax.set_title(f"{eid} | {encoder}\n{kind}", fontsize=10)
            ax.grid(alpha=.22)
            ax.set_xlim((1, 223) if correction else (18, 60))
            if not correction:
                ax.set_ylim(segment_min-segment_margin, segment_max+segment_margin)
            if i%3 == 0:
                ax.set_ylabel("修正量 Δα" if correction else "最终个体化 α")
            if i//3 == 2:
                ax.set_xlabel("未来窗口编号 t（从1开始）")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        title = "全部223个未来窗口的个体化修正量 Δα" if correction else "最终个体化 α：未来第18–60个窗口"
        fig.suptitle(f"{title}\nE0052–E0060｜最终保存组件｜训练被试 S1/S2｜随机种子42｜代码快照0eb37e3", fontsize=13)
        suffix = "correction" if correction else "segment"
        fig.savefig(OUT / f"{prefix}_{suffix}.png", dpi=180)
        fig.savefig(OUT / f"{prefix}_{suffix}_zh.png", dpi=180)
        plt.close(fig)
    diagnostics = {}
    for eid, values in alpha.items():
        delta = values-baseline
        diagnostics[eid] = {f"S{s+1}": {"delta_min": float(delta[s].min()),
            "delta_max": float(delta[s].max()), "alpha_min": float(values[s].min()),
            "alpha_max": float(values[s].max()), "alpha_negative_count": int((values[s]<0).sum()),
            "alpha_above_one_count": int((values[s]>1).sum())} for s in range(2)}
    (OUT / f"{prefix}_manifest.json").write_text(json.dumps({"split": "train",
        "subjects": ["S1", "S2"], "selection_seed": 42, "selection": "random, before output inspection",
        "FC_targets_read": False, "run_sources": metadata, "diagnostics": diagnostics}, indent=2), encoding="utf-8")
    print(table.loc[table.experiment.isin(["E0032", "E0052", "E0053", "E0054"]) &
                    table.future_window.isin([18, 112, 223])].pivot(
                    index=["experiment", "subject"], columns="future_window", values="alpha").to_string())
    print(json.dumps(diagnostics, indent=2))
    print(OUT / f"{prefix}_segment.png")


if __name__ == "__main__":
    main()
