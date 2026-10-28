"""Validation-only comparison, SC swaps and reproducible SC-alpha figures."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import itertools
import json

import numpy as np

from .config import load_config
from .management import config_sha256, file_sha256, verify_data_bindings
from .sc_alpha import (SCHEMA, load_cohort, load_sc_alpha_component, statistics, write_json)
from .pca_sequence import paired_metric


def run_config(run_dir):
    metadata = json.loads((Path(run_dir)/"metadata.json").read_text(encoding="utf-8"))
    config = load_config(Path(run_dir)/"config_resolved.yaml")
    if (metadata.get("status") != "COMPLETED" or config_sha256(config) != metadata["config_sha256"]
            or config["experiment"].get("task") != "sc_alpha"
            or metadata.get("experiment_id") != config["experiment"]["id"]
            or metadata.get("run_id") != Path(run_dir).name):
        raise ValueError("Expected a completed SC alpha run with immutable config")
    return config, metadata


def suite_signature(config):
    signature = {k: deepcopy(config[k]) for k in ("data", "training", "evaluation", "artifacts", "comparison_plan", "paths", "model")}
    signature["paths"].pop("root", None)
    for key in ("sc_encoder", "alpha_parameters"):
        signature["model"].pop(key)
    return signature


def require_suite_ready(config):
    root = Path(config["paths"]["root"])
    if not config["comparison_plan"]["require_suite_ready"]:
        if int(config["experiment"]["level"]) != 0:
            raise ValueError("Suite readiness can only be bypassed in synthetic Level0 checks")
        return []
    selected, combinations, cohort_hashes, folds_hashes = [], set(), set(), set()
    for e in config["comparison_plan"]["suite_experiments"]:
        candidates = []
        for path in (root/"outputs"/e/"runs").glob("*"):
            if (path/"metadata.json").exists():
                metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
                if metadata.get("status") == "COMPLETED" and metadata.get("seed") == 42:
                    candidates.append(path)
        if not candidates:
            raise RuntimeError(f"Complete all nine train-only runs before validation; missing {e}")
        path = sorted(candidates)[-1]
        other, metadata = run_config(path)
        if suite_signature(other) != suite_signature(config):
            raise ValueError("SC alpha suite protocol mismatch")
        if metadata["experiment_id"] != e:
            raise ValueError("SC alpha suite experiment mismatch")
        component = load_sc_alpha_component(path/"checkpoints/best.pt")
        for key in ("experiment_id", "run_id", "config_sha256"):
            if component.payload[key] != metadata[key]:
                raise ValueError("SC alpha suite checkpoint identity mismatch")
        selection = component.payload["selection"]
        cohort_hashes.add(component.payload["training_cohort_sha256"])
        folds_hashes.add(selection["cv_folds_sha256"])
        combinations.add((other["model"]["sc_encoder"], other["model"]["alpha_parameters"]))
        selected.append(path)
    steps = (config["data"]["n_timepoints"]-config["data"]["window_length"])//config["data"]["stride"]
    if (combinations != set(itertools.product(("linear", "mlp", "gcn"), (1, 6, steps)))
            or len(cohort_hashes) != 1 or len(folds_hashes) != 1):
        raise ValueError("SC alpha suite must use all nine conditions with identical train subjects/CV folds")
    return selected


def derangement(n, seed):
    if n < 2:
        raise ValueError("SC swaps require at least two subjects")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        order = rng.permutation(n)
        if np.all(order != np.arange(n)):
            return order
    raise RuntimeError("Could not generate a no-self SC permutation")


def aggregate(rows):
    out = {}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row[key] is not None and np.isfinite(row[key])]
        out[key] = float(np.mean(values)) if values else None
    return out


def sample_metrics(pred, true, base, template, config):
    from .evaluation import _e0032_sequence_diagnostics, _row_correlation, dynamic_calibration_metrics
    start = config["evaluation"]["nonoverlap_start"]
    interval = config["data"]["stride"]*config["data"]["tr_seconds"]
    out = {"mse": float(np.mean((pred-true)**2)),
           "long_edge_mse": float(np.mean((pred[start:]-true[start:])**2)),
           "raw_edge_pearson": float(np.mean(_row_correlation(pred, true)))}
    out.update(_e0032_sequence_diagnostics(pred, true, base, template, start))
    boundaries = np.linspace(start, len(pred), 4, dtype=int)
    for scope, a, b in [("full", 0, len(pred)), ("long", start, len(pred))]:
        for key, value in dynamic_calibration_metrics(pred[a:b], true[a:b], interval).items():
            out[f"{scope}_{key}"] = float(value) if np.isfinite(value) else None
    for name, a, b in [("overlap_context", 0, start), *zip(("early_long", "middle_long", "late_long"), boundaries[:-1], boundaries[1:])]:
        out[f"{name}_mse"] = float(np.mean((pred[a:b]-true[a:b])**2))
    return out


def evaluate_sc_alpha_run(config, run_dir, device="cpu"):
    run_dir = Path(run_dir)
    frozen, metadata = run_config(run_dir)
    if config_sha256(config) != metadata["config_sha256"]:
        raise ValueError("Evaluation configuration differs from the frozen run")
    verify_data_bindings(config)
    ready = require_suite_ready(config)  # Must happen before reading validation data.
    component = load_sc_alpha_component(run_dir/"checkpoints/best.pt", config, device)
    for key in ("run_id", "experiment_id", "config_sha256"):
        if component.payload[key] != metadata[key]:
            raise ValueError("SC alpha checkpoint run identity mismatch")
    cohort = load_cohort(config, "val")
    p = component.payload
    alpha = component.predict_alpha(cohort.sc)
    mu, context = p["group_template"].astype(np.float64), p["context_template"].astype(np.float64)
    base_alpha = p["alpha_raw"].astype(np.float64)
    rows, baseline_rows = [], []
    example = int(np.random.default_rng(42).integers(len(cohort.subjects)))
    private = run_dir/"private"; private.mkdir(exist_ok=True)
    for i, fc in enumerate(cohort.fc):
        offset = fc[0].astype(np.float64)-context
        baseline = mu+base_alpha[:, None]*offset
        predicted = mu+alpha[i, :, None]*offset
        rows.append(sample_metrics(predicted, fc[1:], baseline, mu, config))
        baseline_rows.append(sample_metrics(baseline, fc[1:], baseline, mu, config))
        if i == example:
            np.savez_compressed(private/"visual_example_val.npz", prediction=predicted, target=fc[1:], baseline=baseline)
    q, u, v = statistics(cohort, context, mu)
    start = config["evaluation"]["nonoverlap_start"]
    seeds = config["comparison_plan"]["swap_seeds"]
    swapped = []
    for seed in seeds:
        order = derangement(len(cohort.subjects), seed)
        swap_alpha = component.predict_alpha(cohort.sc[order])
        swapped.append((v-2*swap_alpha*u+swap_alpha**2*q[:, None])[:, start:].mean(1))
    swapped = np.mean(swapped, axis=0)
    plan = config["comparison_plan"]
    bootstrap = {"replicates": plan["bootstrap_replicates"], "seed": plan["bootstrap_seed"],
                 "minimize": True, "family_size": plan["primary_family_size"]}
    main_scores = np.array([r["long_edge_mse"] for r in rows])
    base_scores = np.array([r["long_edge_mse"] for r in baseline_rows])
    comparison = paired_metric(main_scores, base_scores, cohort.subjects, **bootstrap)
    swap_comparison = paired_metric(swapped, main_scores, cohort.subjects,
        replicates=plan["secondary_bootstrap_replicates"], seed=plan["bootstrap_seed"], minimize=False)
    delta = alpha-base_alpha[None]
    diagnostics = {"alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
        "below_zero_fraction": float(np.mean(alpha < 0)), "above_one_fraction": float(np.mean(alpha > 1)),
        "delta_rms": float(np.sqrt(np.mean(delta**2))), "delta_max_abs": float(np.max(np.abs(delta))),
        "delta_second_difference_rms": float(np.sqrt(np.mean(np.diff(delta, n=2, axis=1)**2))),
        "alpha_quantiles": np.quantile(alpha, [0, .01, .25, .5, .75, .99, 1]).tolist(),
        "alpha_cross_subject_std_by_time": alpha.std(0).tolist(),
        "alpha_quantiles_by_time": np.quantile(alpha, [.1, .5, .9], axis=0).tolist()}
    keys = list(rows[0])
    np.savez_compressed(private/"evaluation_val_samples.npz", subjects=np.asarray(cohort.subjects),
        metric_names=np.asarray(keys), model=np.array([[np.nan if r[k] is None else r[k] for k in keys] for r in rows]),
        baseline=np.array([[np.nan if r[k] is None else r[k] for k in keys] for r in baseline_rows]),
        alpha=alpha, swapped_long_mse=swapped)
    report = {"schema": SCHEMA, "experiment_id": metadata["experiment_id"], "run_id": metadata["run_id"],
        "config_sha256": metadata["config_sha256"], "split": "val", "n_subjects": len(rows),
        "validation_cohort_sha256": cohort.digest, "training_cohort_sha256": p["training_cohort_sha256"],
        "checkpoint_sha256": file_sha256(run_dir/"checkpoints/best.pt"),
        "private_samples_sha256": file_sha256(private/"evaluation_val_samples.npz"),
        "suite_training_runs": [x.name for x in ready], "aggregate": aggregate(rows),
        "baseline_aggregate": aggregate(baseline_rows), "paired_vs_B0": comparison,
        "sc_swap": {"seeds": seeds, "no_self_pairs": True, "mean_swapped_long_mse": float(swapped.mean()),
                    "paired_swap_minus_matched": swap_comparison},
        "alpha_diagnostics": diagnostics, "test_data_read": False,
        "selection": p["selection"], "selection_record_sha256": file_sha256(run_dir/"metrics_best.json"),
        "aggregation": "mean of subject/run metrics; calibration ratios are within-subject edge medians",
        "scope": {"training": "all 223 future windows", "primary": "target[17:223] (production)",
                  "fixed_B0_residual": True}}
    path = run_dir/"evaluation_val.json"
    write_json(path, report)
    plot_sc_alpha_run(run_dir)
    return path


def summarize_sc_alpha_runs(root, run_ids, output_dir=None):
    from .management import find_run
    root = Path(root)
    directories = [find_run(root, rid) for rid in run_ids]
    first, _ = run_config(directories[0]); first["paths"]["root"] = str(root)
    if len(directories) != 9 or len(set(run_ids)) != 9:
        raise ValueError("Alpha summary requires exactly nine distinct runs")
    ready = require_suite_ready(first)
    if ready and set(p.name for p in ready) != set(run_ids):
        raise ValueError("Use the suite's matching completed runs in alpha-summary")
    conditions, samples, cohort_hash = {}, {}, set()
    b0_scores, subject_order = None, None
    for path in directories:
        config, metadata = run_config(path)
        if suite_signature(config) != suite_signature(first):
            raise ValueError("Alpha summary protocols differ")
        report = json.loads((path/"evaluation_val.json").read_text(encoding="utf-8"))
        if (report["checkpoint_sha256"] != file_sha256(path/"checkpoints/best.pt")
                or report["config_sha256"] != metadata["config_sha256"]
                or report["run_id"] != path.name
                or report["selection_record_sha256"] != file_sha256(path/"metrics_best.json")
                or report["private_samples_sha256"] != file_sha256(path/"private/evaluation_val_samples.npz")):
            raise ValueError("Alpha evaluation artifacts changed after evaluation")
        with np.load(path/"private/evaluation_val_samples.npz") as a:
            names = a["metric_names"].tolist(); position = names.index("long_edge_mse")
            ids = a["subjects"].tolist(); base = a["baseline"][:, position]
            model = a["model"][:, position]
        if subject_order is not None and (ids != subject_order or not np.allclose(base, b0_scores, atol=1e-12, rtol=0)):
            raise ValueError("Alpha suite subject order/B0 mismatch")
        subject_order, b0_scores = ids, base
        cohort_hash.add(report["validation_cohort_sha256"])
        label = f"{config['model']['sc_encoder']}-{config['model']['alpha_parameters']}"
        if label in conditions:
            raise ValueError("Repeated SC alpha condition")
        conditions[label] = report; samples[label] = model
    steps = (first["data"]["n_timepoints"]-first["data"]["window_length"])//first["data"]["stride"]
    expected = {f"{e}-{k}" for e in ("linear", "mlp", "gcn") for k in (1, 6, steps)}
    if set(conditions) != expected or len(cohort_hash) != 1:
        raise ValueError("Need the full 3x3 matrix on one validation cohort")
    contrasts = {}
    for left, right in itertools.combinations(sorted(conditions), 2):
        e1, k1 = left.split("-"); e2, k2 = right.split("-")
        if e1 != e2 and k1 != k2:
            continue
        contrasts[f"{left}_minus_{right}"] = paired_metric(samples[left], samples[right], subject_order,
            replicates=first["comparison_plan"]["secondary_bootstrap_replicates"],
            seed=first["comparison_plan"]["bootstrap_seed"], minimize=True)
    dest = Path(output_dir) if output_dir else root/"outputs"/first["comparison_plan"]["suite_experiments"][-1]/"visual"
    dest.mkdir(parents=True, exist_ok=True)
    output = dest/"sc_alpha_suite_val.json"
    write_json(output, {"schema": SCHEMA, "split": "val", "seed": 42, "conditions": conditions,
                        "secondary_exploratory_contrasts": contrasts, "baseline_aggregate": next(iter(conditions.values()))["baseline_aggregate"],
                        "test_data_read": False})
    wrapper = dest/"build_sc_alpha_suite_visuals.py"
    wrapper.write_text('"""SC alpha nine-model suite, seed42 validation. Input: sc_alpha_suite_val.json.\n'
        'Outputs: sc_alpha_suite_metrics.png, sc_alpha_suite_paired.png and sc_alpha_suite_diagnostics.png. No test access.\n"""\n'
        'from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parents[3]\n'
        'sys.path.insert(0,str(ROOT/"src"))\nfrom scdfc.sc_alpha_evaluation import plot_suite\n'
        'plot_suite(Path(__file__).with_name("sc_alpha_suite_val.json"))\n', encoding="utf-8")
    plot_suite(output)
    return output


def plot_suite(report_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = sorted(report["conditions"], key=lambda s: (s.split("-")[0], int(s.split("-")[1])))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, key in zip(axes, ("long_edge_mse", "time_pearson", "template_adjusted_time_pearson")):
        values = [np.nan if report["conditions"][s]["aggregate"][key] is None
                  else report["conditions"][s]["aggregate"][key] for s in labels]
        ax.bar(np.arange(9), values, width=.42)
        if report["baseline_aggregate"][key] is not None:
            ax.axhline(report["baseline_aggregate"][key], color="orange", ls="--", label="B0")
        ax.set(xticks=np.arange(9), xticklabels=labels, xlim=(-.5, 8.5), title=key)
        ax.tick_params(axis="x", rotation=55); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.suptitle("SC-conditioned alpha | seed42 | validation | nine models + fixed B0")
    fig.savefig(report_path.with_name("sc_alpha_suite_metrics.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for i, label in enumerate(labels):
        r = report["conditions"][label]["paired_vs_B0"]; ci = r["family_adjusted_ci"]
        ax.plot([ci["ci_low"], ci["ci_high"]], [i, i], color="steelblue", lw=1)
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color="steelblue", lw=3)
        ax.plot(r["mean_difference"], i, "o", color="steelblue")
    ax.axvline(0, color="black", ls="--")
    ax.set(yticks=range(9), yticklabels=labels, xlabel="Long MSE: model minus B0 (negative is better)",
           title="Validation paired bootstrap | thick 95% CI | thin family-adjusted CI")
    fig.savefig(report_path.with_name("sc_alpha_suite_paired.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    values = [report["conditions"][s]["sc_swap"]["paired_swap_minus_matched"]["mean_difference"] for s in labels]
    axes[0].bar(np.arange(9), values, width=.42)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set(title="Frozen SC swaps: swapped minus matched MSE", xticks=np.arange(9), xticklabels=labels)
    for label in labels:
        median = report["conditions"][label]["alpha_diagnostics"]["alpha_quantiles_by_time"][1]
        axes[1].plot(median, label=label, lw=1)
    axes[1].set(title="Median individual alpha by future window", xlabel="Future window index", ylabel="Unbounded alpha")
    axes[1].legend(fontsize=8, ncol=2, frameon=False)
    axes[0].tick_params(axis="x", rotation=55)
    fig.suptitle("SC alpha | seed42 | validation diagnostics")
    fig.savefig(report_path.with_name("sc_alpha_suite_diagnostics.png"), dpi=160); plt.close(fig)


def plot_sc_alpha_run(run_dir):
    """Render five validation figures from aggregate reports/local private examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .visualization import fixed_edge_indices, _read_log
    run_dir = Path(run_dir)
    config, metadata = run_config(run_dir)
    report = json.loads((run_dir/"evaluation_val.json").read_text(encoding="utf-8"))
    if report["private_samples_sha256"] != file_sha256(run_dir/"private/evaluation_val_samples.npz"):
        raise ValueError("Private alpha visualization inputs changed")
    destination = run_dir.parent.parent/"visual"; destination.mkdir(exist_ok=True)
    prefix = f"{run_dir.name}_val"
    images = []
    def save(fig, suffix):
        path = destination/f"{prefix}_{suffix}.png"
        fig.savefig(path, dpi=160); plt.close(fig); images.append(path.name)
    rows = _read_log(run_dir/"train.log")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot([r["epoch"] for r in rows], [r["train_mse"] for r in rows], color="orange")
    axes[0].set(xlabel="Epoch", ylabel="Training full-future MSE", title="Final fixed-length train-only fit")
    axes[1].axis("off")
    selected = report["selection"]
    axes[1].text(0, .95, f"{metadata['experiment_id']} | {config['model']['sc_encoder']} | K={config['model']['alpha_parameters']}\n\n"
        f"Seed42 | train CV lambda={selected['selected_lambda']}\nFinal epochs={selected['final_epochs']} | unbounded alpha\n\n"
        f"Train CV long MSE: {selected['cv_long_edge_mse']:.7f}\nExternal validation long MSE: {report['aggregate']['long_edge_mse']:.7f}\n"
        f"B0 validation long MSE: {report['baseline_aggregate']['long_edge_mse']:.7f}\n\n"
        "Checkpoint selected without external validation.\nTest data not read.", va="top", fontsize=10)
    fig.suptitle(f"{run_dir.name} | validation overview", fontsize=11); save(fig, "overview")
    with np.load(run_dir/"private/evaluation_val_samples.npz") as a:
        alphas = a["alpha"]
    component = load_sc_alpha_component(run_dir/"checkpoints/best.pt")
    baseline = component.payload["alpha_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    lo, mid, hi = np.quantile(alphas, [.1, .5, .9], axis=0)
    axes[0].fill_between(np.arange(len(mid)), lo, hi, alpha=.2)
    axes[0].plot(mid, label="Individual median"); axes[0].plot(baseline, "--", label="Common alpha")
    axes[0].set(xlabel="Future window index", ylabel="Alpha", title="Validation 10/50/90% quantiles")
    axes[0].legend(frameon=False)
    axes[1].hist((alphas-baseline[None]).ravel(), bins=40, color="orange")
    axes[1].set(xlabel="Delta alpha", ylabel="Subject-window count", title="Unbounded corrections (descriptive)")
    fig.suptitle(f"{run_dir.name} | validation alpha"); save(fig, "alpha")
    with np.load(run_dir/"private/visual_example_val.npz") as a:
        pred, true, base = (a[k] for k in ("prediction", "target", "baseline"))
    edges = fixed_edge_indices(true.shape[1], count=min(4, true.shape[1]))
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    header, body = fig.subfigures(2, 1, height_ratios=(.12, .88))
    header.text(.5, .8, f"{run_dir.name}\nValidation fixed-edge example | same project-prespecified sample", ha="center", fontsize=10)
    header.legend([Line2D([], [], color="tab:blue"), Line2D([], [], color="tab:orange")],
                  ["True", "Predicted"], loc="lower center", ncol=2, frameon=False)
    axes = body.subplots(len(edges), 2, squeeze=False, sharex=True, sharey="row")
    minutes = np.arange(len(true))*config["data"]["stride"]*config["data"]["tr_seconds"]/60
    for row, edge in enumerate(edges):
        for col, values in enumerate((base, pred)):
            ax = axes[row, col]
            ax.plot(minutes, true[:, edge], color="tab:blue", lw=1.1)
            ax.plot(minutes, values[:, edge], color="tab:orange", lw=1.1)
            ax.axvspan(0, minutes[config["evaluation"]["nonoverlap_start"]], color="gray", alpha=.1)
            if row == 0: ax.set_title("B0" if col == 0 else metadata["experiment_id"])
            if col == 0: ax.set_ylabel(f"Edge {edge}\nFisher-z")
            if row == len(edges)-1: ax.set_xlabel("Minutes from first future window")
            ax.grid(alpha=.2)
    save(fig, "fixed_edges")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key in zip(axes, ("long_temporal_std_ratio", "long_difference_std_ratio", "long_difference_temporal_pearson")):
        values = [report['baseline_aggregate'][key], report['aggregate'][key]]
        ax.bar([0, 1], [np.nan if value is None else value for value in values], width=.42)
        if not key.endswith("pearson"): ax.axhline(1, color="gray", ls="--")
        ax.set(xticks=[0, 1], xticklabels=["B0", metadata["experiment_id"]], xlim=(-.5, 1.5), title=key.replace("long_", ""))
    fig.suptitle(f"{run_dir.name} | validation long dynamic calibration", fontsize=11); save(fig, "dynamic_calibration")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    segments = ["overlap_context", "early_long", "middle_long", "late_long"]
    for label, values in [("B0", report["baseline_aggregate"]), (metadata["experiment_id"], report["aggregate"])]:
        ax.plot(segments, [values[s+"_mse"] for s in segments], marker="o", label=label)
    ax.set(ylabel="Mean subject MSE", title=f"{run_dir.name} | validation horizon audit")
    ax.legend(frameon=False); save(fig, "horizon_audit")
    wrapper = destination/"build_sc_alpha_visuals.py"
    wrapper.write_text(f'"""{metadata["experiment_id"]}/{run_dir.name} validation figures.\n'
        'Inputs: frozen run config, evaluation_val.json, training log, checkpoint and local private example/alpha arrays.\n'
        'Outputs: overview, alpha, fixed_edges, dynamic_calibration and horizon_audit PNGs.\n"""\n'
        'from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parents[3]\n'
        'sys.path.insert(0,str(ROOT/"src"))\nfrom scdfc.sc_alpha_evaluation import plot_sc_alpha_run\n'
        f'plot_sc_alpha_run(ROOT/"outputs/{metadata["experiment_id"]}/runs/{run_dir.name}")\n', encoding="utf-8")
    manifest = destination/f"{prefix}_visual_manifest.json"
    write_json(manifest, {"run_id": run_dir.name, "split": "val", "images": images,
        "fixed_edge_seed": 42, "edges": edges.tolist(), "test_data_read": False,
        "evaluation_sha256": file_sha256(run_dir/"evaluation_val.json")})
    return manifest
