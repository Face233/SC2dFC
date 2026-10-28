"""E0032 and E0052-E0060 frozen-component static/dynamic retrieval, seed42 val.

Inputs: completed run configs/checkpoints, E0032 alpha artifact, frozen val FC/SC.
Output: E0032_E0052_E0060_retrieval_val.json/.md and local private rank arrays.
Matches outputs/E0032/visual/analyze_subject_similarity.py definitions.
No fitting, checkpoint selection, or test access.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scdfc.management import file_sha256, verify_data_bindings
from scdfc.sc_alpha import load_cohort, load_sc_alpha_component
from scdfc.sc_alpha_evaluation import require_suite_ready, run_config


def unit_features(x):
    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(1, keepdim=True)
    return torch.nn.functional.normalize(x, dim=1, eps=1e-12)


def summarize(similarity):
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.argmax(order == np.arange(len(order))[:, None], axis=1) + 1
    return ranks, {"top1_count": int((ranks <= 1).sum()), "top5_count": int((ranks <= 5).sum()),
                   "top1": float((ranks <= 1).mean()), "top5": float((ranks <= 5).mean()),
                   "mean_rank": float(ranks.mean())}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    candidates = sorted((ROOT / "outputs/E0052/runs").glob("*/component_manifest.json"))
    config, _ = run_config(candidates[-1].parent)
    config["paths"]["root"] = str(ROOT)
    directories = require_suite_ready(config)
    verify_data_bindings(config)
    selection_hashes = {str(p): file_sha256(p / "metrics_best.json") for p in directories}
    cohort = load_cohort(config, "val")
    first = load_sc_alpha_component(directories[0] / "checkpoints/best.pt", config, str(device))
    payload = first.payload
    start = config["evaluation"]["nonoverlap_start"]
    mu = payload["group_template"].astype(np.float64)[start:]
    context = payload["context_template"].astype(np.float64)
    offset = cohort.fc[:, 0].astype(np.float64) - context
    true = torch.as_tensor(cohort.fc[:, start+1:], device=device)
    target_dynamic_unit = unit_features(true - true.mean(1, keepdim=True))
    # Static scores follow the existing template-adjusted future-mean definition.
    true_static = cohort.fc[:, start+1:].astype(np.float64).mean(1) - mu.mean(0)
    static_target_unit = unit_features(torch.as_tensor(true_static, device=device))
    mu_gpu = torch.as_tensor(mu, device=device)
    offset_gpu = torch.as_tensor(offset, device=device)
    rows, rank_arrays = [], {"subjects": np.asarray(cohort.subjects)}
    methods = [("E0032", None)] + [(p.parent.parent.name, p) for p in directories]
    for eid, directory in methods:
        if directory is None:
            alpha = np.broadcast_to(payload["alpha_raw"].astype(np.float64), (len(cohort.subjects), len(payload["alpha_raw"])))
            source = {"alpha_sha256": config["artifacts"]["alpha_fit"]["sha256"]}
        else:
            cfg, meta = run_config(directory)
            cfg["paths"]["root"] = str(ROOT)
            component = load_sc_alpha_component(directory / "checkpoints/best.pt", cfg, str(device))
            for key in ("context_template", "group_template", "alpha_raw"):
                if not np.array_equal(component.payload[key], payload[key]):
                    raise ValueError("Frozen templates/alpha differ across models")
            alpha = component.predict_alpha(cohort.sc)
            source = {"run_id": meta["run_id"], "config_sha256": meta["config_sha256"],
                      "checkpoint_sha256": file_sha256(directory / "checkpoints/best.pt")}
        # Mean(prediction)-mean(mu) = mean(alpha)*first-window offset exactly.
        static_pred_unit = unit_features(torch.as_tensor(alpha[:, start:].mean(1)[:, None] * offset, device=device))
        static_ranks, static = summarize((static_pred_unit @ static_target_unit.T).cpu().numpy())
        # Cast predictions to float32 before demeaning, as in the historical audit.
        a = torch.as_tensor(np.ascontiguousarray(alpha[:, start:]), device=device)
        predicted = (mu_gpu[None] + a[:, :, None] * offset_gpu[:, None]).float()
        pred_unit = unit_features(predicted - predicted.mean(1, keepdim=True))
        dynamic_ranks, dynamic = summarize((pred_unit @ target_dynamic_unit.T).cpu().numpy())
        rows.append({"experiment": eid, "static": static, "dynamic": dynamic,
                     "mean_alpha_long_min": float(alpha[:, start:].mean(1).min()), **source})
        rank_arrays[eid + "_static_ranks"] = static_ranks
        rank_arrays[eid + "_dynamic_ranks"] = dynamic_ranks
        print(f"{eid}: static {static['top1_count']}/{static['top5_count']}; dynamic {dynamic['top1_count']}/{dynamic['top5_count']} (of {len(cohort.subjects)})", flush=True)
        del predicted, pred_unit, static_pred_unit, a
    if any(file_sha256(Path(p) / "metrics_best.json") != digest for p, digest in selection_hashes.items()):
        raise ValueError("Training selection records changed")
    private = directories[-1] / "private/static_dynamic_retrieval_val.npz"
    np.savez_compressed(private, **rank_arrays)
    report = {"split": "val", "n_subjects": len(cohort.subjects), "seed": 42,
              "nonoverlap_start": start, "validation_cohort_sha256": cohort.digest,
              "static_definition": "Pearson of future mean FC minus mean training template across edges",
              "dynamic_definition": "Pearson of per-edge time-demeaned future FC trajectories flattened over time and edges",
              "rank_direction": "each prediction ranks all true validation subjects; stable descending similarity",
              "chance": {"top1": 1/len(cohort.subjects), "top5": 5/len(cohort.subjects)},
              "device": str(device), "private_ranks_sha256": file_sha256(private), "methods": rows,
              "test_data_read": False, "models_refitted": False,
              "scope": "supplementary frozen-component retrieval; not the full primary MSE/SC-swap evaluation"}
    destination = Path(__file__).with_name("E0032_E0052_E0060_retrieval_val.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E0032 与 E0052–E0060：静态与动态检索", "",
             "158名验证被试，非重叠未来206窗；随机Top-1=0.63%，Top-5=3.16%。", "",
             "静态：扣除训练群体模板后的未来平均FC，用跨边Pearson排序。",
             "动态：每名被试的每条边沿时间去均值，再将时间和边展平，用Pearson排序。", "",
             "| 实验 | 静态Top-1 | 静态Top-5 | 动态Top-1 | 动态Top-5 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = [f"{row[k][metric]:.2%} ({row[k][metric+'_count']}/158)" for k, metric in
                  (("static", "top1"), ("static", "top5"), ("dynamic", "top1"), ("dynamic", "top5"))]
        lines.append("| " + " | ".join([row["experiment"], *values]) + " |")
    lines += ["", "本次只计算冻结模型的验证检索指标，完整长期MSE与SC交换评价另行执行。", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
