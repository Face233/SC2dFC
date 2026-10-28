"""Validation-only streaming comparisons for frozen-alpha transfer.

All dynamic residual diagnostics subtract the same shared E0032 baseline.
Only local private files contain participant identifiers and example arrays.
"""
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .alpha_transfer import pinned_path, verify_transfer_bindings
from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset
from .evaluation import _load_model, _e0032_sequence_diagnostics, _edge_time_pearson
from .management import config_sha256, file_sha256, verify_artifact
from .pca_sequence import paired_metric
from .sc_alpha import tensor_digest, write_json
from .training import device_from_arg


METRICS = ("long_edge_mse", "time_pearson", "template_adjusted_time_pearson",
           "dynamic_time_pearson", "dynamic_difference_time_pearson", "dynamic_std_ratio",
           "stable_residual_mse", "centered_dynamic_mse")


def metric_row(pred, true, b0, template, start):
    pred, true, b0, template = (np.asarray(x, dtype=np.float64) for x in (pred, true, b0, template))
    if not all(np.isfinite(x).all() for x in (pred, true, b0, template)):
        raise FloatingPointError("Nonfinite transfer evaluation input")
    row = _e0032_sequence_diagnostics(pred, true, b0, template, start)
    row["long_edge_mse"] = float(np.mean((pred[start:]-true[start:])**2))
    row["all_future_mse"] = float(np.mean((pred-true)**2))
    for name, a, b in (("overlap", 0, start), ("early_long", start, 85),
                       ("middle_long", 85, 154), ("late_long", 154, len(pred))):
        if a < min(b, len(pred)):
            row[name+"_mse"] = float(np.mean((pred[a:b]-true[a:b])**2))
    centered_pred = pred[start:]-pred[start:].mean(0)
    centered_true = true[start:]-true[start:].mean(0)
    row["fc_temporal_std_ratio"] = float(np.std(centered_pred)/max(np.std(centered_true), 1e-12))
    return row


def aggregate(rows):
    return {key: (float(np.mean([r[key] for r in rows if r.get(key) is not None]))
                  if any(r.get(key) is not None for r in rows) else None) for key in rows[0]}


def paired_rows(main, baseline, subjects, plan, primary=False):
    result = {key: paired_metric(
        [np.nan if r.get(key) is None else r[key] for r in main],
        [np.nan if r.get(key) is None else r[key] for r in baseline], subjects,
        replicates=plan["bootstrap_replicates"], seed=plan["bootstrap_seed"],
        minimize=key.endswith("mse"),
        family_size=plan["primary_family_size"] if primary and key == "long_edge_mse" else 1,
    ) for key in METRICS}
    # More dynamic amplitude is not intrinsically better calibration.
    if result["dynamic_std_ratio"] is not None:
        result["dynamic_std_ratio"]["passes"] = None
        result["dynamic_std_ratio"]["family_adjusted_ci"]["passes"] = None
    return result


def donor_indices(subjects, seed):
    unique = sorted(set(subjects))
    if len(unique) < 2:
        raise ValueError("SC swapping needs at least two validation subjects")
    order = np.random.default_rng(seed).permutation(unique).tolist()
    mapping = dict(zip(order, order[1:]+order[:1]))
    first = {s: subjects.index(s) for s in unique}
    return np.array([first[mapping[s]] for s in subjects])


@torch.no_grad()
def evaluate_transfer(config, run_dir, device_name="cpu"):
    run_dir = Path(run_dir)
    source, old_config = verify_transfer_bindings(config)
    device = device_from_arg(device_name)
    checkpoint = run_dir/"checkpoints/best.pt"
    meta = json.loads((run_dir/"metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "COMPLETED" or meta["config_sha256"] != config_sha256(config):
        raise ValueError("Complete the matching transfer training before validation evaluation")
    stats = pinned_path(config, "training_stats")
    model, payload = _load_model(config, config["data"]["window_length"], checkpoint, stats, device, verify_artifact(config))
    for key, expected in (("run_id", meta["run_id"]), ("experiment_id", meta["experiment_id"]),
                          ("config_sha256", meta["config_sha256"])):
        if payload.get(key) != expected:
            raise ValueError("Transfer checkpoint identity mismatch")
    if tensor_digest(model.sc_alpha_component) != payload["sc_alpha_state_sha256"]:
        raise ValueError("Frozen alpha changed during residual training")
    old_model, _ = _load_model(old_config, config["data"]["window_length"],
                              pinned_path(config, "residual_reference"), stats, device, verify_artifact(old_config))
    dataset = DFCSequenceDataset(config, config["data"]["window_length"], "val", stats,
                                 config["experiment"]["ablation"])
    subjects = [s for s, _ in dataset.samples]
    if len(subjects) != config["comparison_plan"]["expected_val_subjects"] or len(set(subjects)) != len(subjects):
        raise ValueError("Expected one LR sample per declared validation subject")
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    start = nonoverlap_horizon(config["data"]["window_length"], config["data"]["stride"])
    template = model.group_template.cpu().numpy()
    methods = {name: [] for name in ("model", "original", "posthoc", "e0032", "alpha_only")}
    alpha_rows, branch_rows, all_alpha = [], [], []
    rng = np.random.default_rng(42)
    example_indices = set(rng.choice(len(dataset), size=min(3, len(dataset)), replace=False).tolist())
    example_edges = np.sort(rng.choice(template.shape[1], size=4, replace=False))
    examples = {name: [] for name in ("true", "model", "original", "posthoc", "alpha_only", "indices")}
    offset_index = 0
    for batch in loader:
        sc, edges, warmup = (batch[k].to(device) for k in ("sc_matrix", "sc_edges", "fc_warmup"))
        result = model(sc, edges, warmup)
        old = old_model(sc, edges, warmup).fc_z_edges
        context = warmup[:, -1] if warmup.ndim == 3 else warmup
        d = context-model.context_template[None]
        b0 = model.group_template[None]+model.decay_alpha[None, :, None]*d[:, None]
        correction = result.residual_baseline-b0
        alpha = model.sc_alpha_component(sc).cpu().numpy()
        all_alpha.append(alpha)
        values = {"model": result.fc_z_edges.cpu().numpy(), "original": old.cpu().numpy(),
                  "posthoc": (old+correction).cpu().numpy(), "e0032": b0.cpu().numpy(),
                  "alpha_only": result.residual_baseline.cpu().numpy()}
        true = batch["fc_future"].numpy()
        residual = result.dynamic_residual.cpu().numpy()
        delta = correction.cpu().numpy()
        for i in range(len(true)):
            for name, pred in values.items():
                methods[name].append(metric_row(pred[i], true[i], values["e0032"][i], template, start))
            da = alpha[i]-model.decay_alpha.cpu().numpy()
            alpha_rows.append({"alpha_temporal_std": float(alpha[i, start:].std()),
                               "delta_temporal_std": float(da[start:].std()),
                               "delta_rms": float(np.sqrt(np.mean(da**2))),
                               "alpha_below_zero": float(np.mean(alpha[i] < 0)),
                               "alpha_above_one": float(np.mean(alpha[i] > 1)),
                               "alpha_max_abs": float(np.max(np.abs(alpha[i])))})
            a, r = delta[i, start:].astype(float), residual[i, start:].astype(float)
            a, r = a-a.mean(0), r-r.mean(0)
            energy = float(np.sum(a*a)+np.sum(r*r))
            branch_rows.append({"alpha_dynamic_rms": float(np.sqrt(np.mean(a*a))),
                                "residual_dynamic_rms": float(np.sqrt(np.mean(r*r))),
                                "dynamic_cross_cosine": float(np.sum(a*r)/max(np.linalg.norm(a)*np.linalg.norm(r), 1e-12)),
                                "combined_to_separate_dynamic_energy": float(np.sum((a+r)**2)/max(energy, 1e-12))})
            if offset_index+i in example_indices:
                examples["indices"].append(offset_index+i)
                examples["true"].append(true[i][:, example_edges])
                for name in ("model", "original", "posthoc", "alpha_only"):
                    examples[name].append(values[name][i][:, example_edges])
        offset_index += len(true)
    plan = config["comparison_plan"]
    # Small SC tensors are cached locally; future FC targets are never swapped.
    sc_cache = torch.stack([dataset[i]["sc_matrix"] for i in range(len(dataset))]).to(device)
    edge_cache = torch.stack([dataset[i]["sc_edges"] for i in range(len(dataset))]).to(device)
    modes = ["alpha"] if config["experiment"]["ablation"] == "fc1_only" else ["alpha", "residual", "both"]
    swap_report = {}
    for mode in modes:
        trial_losses, trial_times = [], []
        for seed in plan["swap_seeds"]:
            donors = donor_indices(subjects, seed)
            losses, times = [], []
            cursor = 0
            for batch in loader:
                count = len(batch["fc_future"])
                own = slice(cursor, cursor+count)
                donor = torch.as_tensor(donors[own], device=device)
                swapped_sc = sc_cache[donor] if mode in {"residual", "both"} else sc_cache[own]
                swapped_edges = edge_cache[donor] if mode in {"residual", "both"} else edge_cache[own]
                alpha_sc = sc_cache[donor] if mode in {"alpha", "both"} else sc_cache[own]
                prediction = model(swapped_sc, swapped_edges, batch["fc_warmup"].to(device),
                                   sc_alpha_matrix=alpha_sc).fc_z_edges.cpu().numpy()
                for pred, true in zip(prediction, batch["fc_future"].numpy()):
                    p, t = pred[start:].astype(float), true[start:].astype(float)
                    losses.append(float(np.mean((p-t)**2)))
                    corr = _edge_time_pearson(p, t)[0]
                    times.append(np.nan if corr is None else corr)
                cursor += count
            trial_losses.append(losses)
            trial_times.append(times)
        average_loss = np.mean(trial_losses, axis=0)
        average_time = np.nanmean(trial_times, axis=0)
        swap_report[mode] = {
            "swapped_long_edge_mse": float(average_loss.mean()),
            "swapped_minus_matched_long_mse": paired_metric(average_loss,
                [r["long_edge_mse"] for r in methods["model"]], subjects,
                replicates=plan["bootstrap_replicates"], seed=plan["bootstrap_seed"]),
            "matched_minus_swapped_time_pearson": paired_metric(
                [r["time_pearson"] for r in methods["model"]], average_time, subjects,
                replicates=plan["bootstrap_replicates"], seed=plan["bootstrap_seed"]),
            "seeds": plan["swap_seeds"], "n_subjects": len(set(subjects)),
            "aggregation": "average ten swaps within subject before subject bootstrap"}
        print(f"{meta['experiment_id']} SC swap path={mode} complete", flush=True)
    alpha = np.concatenate(all_alpha)
    report = {"schema": "sc_alpha_transfer/v1", "experiment_id": meta["experiment_id"],
              "run_id": meta["run_id"], "split": "val", "n_subjects": len(subjects),
              "fc_representation": "E0003 frozen AE encoder, 256 dimensions",
              "sc_alpha_source": config["artifacts"]["sc_alpha"],
              "residual_reference": config["artifacts"]["residual_reference"],
              "checkpoint_sha256": file_sha256(checkpoint), "nonoverlap_start": start,
              "dynamic_reference": "fixed shared E0032 baseline for every method",
              "checkpoint_selection": {"epoch": payload["epoch"], "index_base": 0,
                                       "metric": payload["primary_metric"], "split": "val"},
              "aggregate": {name: aggregate(rows) for name, rows in methods.items()},
              "paired_model_vs_original": paired_rows(methods["model"], methods["original"], subjects, plan, True),
              "paired_posthoc_vs_original": paired_rows(methods["posthoc"], methods["original"], subjects, plan),
              "paired_model_vs_e0032": paired_rows(methods["model"], methods["e0032"], subjects, plan),
              "sc_swap": swap_report, "alpha_diagnostics": aggregate(alpha_rows),
              "alpha_between_subject_std_mean": float(alpha[:, start:].std(0).mean()),
              "branch_diagnostics": aggregate(branch_rows),
              "interpretation": "Exploratory reused validation cohort; six primary long-MSE contrasts use family-adjusted intervals."}
    private = run_dir/"private"
    write_json(private/"transfer_per_subject_val.json", {
        "subjects": subjects, "runs": [r for _, r in dataset.samples], "methods": methods,
        "alpha": alpha_rows, "branches": branch_rows})
    np.savez_compressed(private/"transfer_examples_val.npz", edges=example_edges, **{k: np.asarray(v) for k, v in examples.items()})
    destination = run_dir/"evaluation_val.json"
    write_json(destination, report)
    return destination


def summarize_transfer(root, run_dirs):
    root = Path(root)
    reports = [json.loads((Path(d)/"evaluation_val.json").read_text(encoding="utf-8")) for d in run_dirs]
    experiment_ids = [r["experiment_id"] for r in reports]
    suite_name = f"{experiment_ids[0]}_{experiment_ids[-1]}"
    destination = root/"outputs"/experiment_ids[-1]/"visual"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/f"{suite_name}_comparison_val.json", {r["experiment_id"]: r for r in reports})
    lines = [f"# {experiment_ids[0]}–{experiment_ids[-1]}：冻结223参数α＋AE残差Transformer", "",
             "158名验证被试；长期target[17:223]；本轮为探索性比较。", "",
             "| 实验 | α来源 | 长期MSE | 原骨干MSE | 相对变化 | 时间Pearson | 动态标准差比 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in reports:
        a, b = r["aggregate"]["model"], r["aggregate"]["original"]
        lines.append(f"| {r['experiment_id']} | {r['sc_alpha_source']['experiment_id']} | {a['long_edge_mse']:.9f} | {b['long_edge_mse']:.9f} | {(a['long_edge_mse']/b['long_edge_mse']-1)*100:+.4f}% | {a['time_pearson']:.6f} | {a['dynamic_std_ratio']:.6f} |")
    (destination/f"{suite_name}_comparison_val.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # Reusable redraw script lives beside the figure and reads only aggregates.
    script = '''from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent
data = list(json.loads((root/"{suite_name}_comparison_val.json").read_text(encoding="utf-8")).values())
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
fig.savefig(root/"{suite_name}_metrics_val.png", dpi=180)
'''
    script = script.replace("{suite_name}", suite_name)
    redraw = destination/"plot_transfer_comparison.py"
    redraw.write_text(script, encoding="utf-8")
    exec(compile(script, str(redraw), "exec"), {"__file__": str(redraw)})
    return destination/f"{suite_name}_comparison_val.md"
