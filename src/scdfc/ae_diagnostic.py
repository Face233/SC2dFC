"""E0044: frozen, contemporaneous FC encode/decode diagnostics (never forecasting).

Inputs: frozen config, verified A0003, LR cache, private train/val split.
Outputs: a new diagnostics/<id> directory and reproducible PNGs in visual/.
Only PCA and group references are fitted, exclusively on training subjects.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from sklearn.decomposition import PCA

from .config import load_config, resolve_path
from .connectivity import config_hash
from .data import cache_path, load_split
from .management import (config_sha256, environment_snapshot, file_sha256,
                         git_metadata, utc_now, verify_artifact, verify_data_bindings)
from .progress import emit
from .training import device_from_arg, load_autoencoder, seed_everything


METHODS = ("identity", "train_static", "train_template", "oracle_mean", "pca", "ae")
SUMMARY_METRICS = ("mse", "mae", "space_r", "mean_bias_mse", "centered_dynamic_mse",
                   "dynamic_nmse", "time_r", "difference_time_r", "std_ratio_mean",
                   "std_ratio_median", "std_ratio_q05", "std_ratio_q25", "std_ratio_q75",
                   "std_ratio_q95", "energy_std_ratio", "difference_nmse",
                   "true_valid_fraction", "time_valid_fraction", "reconstructed_flat_fraction",
                   "difference_valid_fraction", "raw_r_mse", "raw_r_mae", "raw_r_std_ratio_mean")


def _mean(values):
    values = np.asarray([x for x in values if x is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _center(values, axis=0):
    # Subtract an anchor first: a repeated constant then has exactly zero energy.
    shifted = values - np.take(values, [0], axis=axis)
    return shifted - shifted.mean(axis=axis, keepdims=True)


def correlations(prediction, target, axis=0, threshold=1e-6):
    left, right = _center(prediction, axis), _center(target, axis)
    ls, rs = left.std(axis=axis), right.std(axis=axis)
    valid = (ls >= threshold) & (rs >= threshold)
    values = np.full(ls.shape, np.nan)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(right, axis=axis)
    numerator = (left * right).sum(axis=axis)
    values[valid] = np.clip(numerator[valid] / denominator[valid], -1, 1)
    return values, valid, rs >= threshold, ls < threshold


def reconstruction_metrics(prediction, target, threshold=1e-6):
    """Run-level float64 metrics; temporal quantities are null for T=1."""
    y, x = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if y.ndim != 2 or y.shape != x.shape or min(x.shape) < 1:
        raise ValueError("Expected matching nonempty [time,edge] arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Metrics require finite arrays and a positive threshold")
    result = {name: None for name in SUMMARY_METRICS}
    result.update(mse=float(np.mean((y-x)**2)), mae=float(np.mean(np.abs(y-x))),
                  space_r=_mean(correlations(y, x, axis=1, threshold=threshold)[0]),
                  mean_bias_mse=float(np.mean((y.mean(0)-x.mean(0))**2)),
                  raw_r_mse=float(np.mean((np.tanh(y)-np.tanh(x))**2)),
                  raw_r_mae=float(np.mean(np.abs(np.tanh(y)-np.tanh(x)))))
    if len(x) < 2:
        return result
    yc, xc = _center(y), _center(x)
    error, energy = float(np.sum((yc-xc)**2)), float(np.sum(xc**2))
    result["centered_dynamic_mse"] = error / x.size
    if not np.isclose(result["mse"], result["mean_bias_mse"] + error/x.size, rtol=1e-9, atol=1e-12):
        raise ArithmeticError("MSE decomposition failed")
    result["dynamic_nmse"] = error / energy if energy > 0 else None
    result["energy_std_ratio"] = float(np.sqrt(np.sum(yc**2)/energy)) if energy > 0 else None
    corr, valid, true_valid, flat = correlations(y, x, threshold=threshold)
    ratio = yc.std(0)[true_valid] / xc.std(0)[true_valid]
    result.update(time_r=_mean(corr), true_valid_fraction=float(true_valid.mean()),
                  time_valid_fraction=float(valid.mean()), reconstructed_flat_fraction=float(flat.mean()),
                  std_ratio_mean=_mean(ratio))
    if len(ratio):
        for name, quantile in (("median", .5), ("q05", .05), ("q25", .25), ("q75", .75), ("q95", .95)):
            result[f"std_ratio_{name}"] = float(np.quantile(ratio, quantile))
    dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
    difference_energy = float(np.sum(dx**2))
    result["difference_nmse"] = float(np.sum((dy-dx)**2)/difference_energy) if difference_energy > 0 else None
    if len(dx) >= 2:
        dc, dv, _, _ = correlations(dy, dx, threshold=threshold)
        result.update(difference_time_r=_mean(dc), difference_valid_fraction=float(dv.mean()))
    rx, ry = _center(np.tanh(x)), _center(np.tanh(y))
    rv = rx.std(0) >= threshold
    result["raw_r_std_ratio_mean"] = _mean(ry.std(0)[rv]/rx.std(0)[rv])
    return result


def bootstrap_mean(values, replicates=10000, seed=20261020):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    result = {"mean": _mean(values), "ci_low": None, "ci_high": None, "n_subjects": len(values)}
    if len(values):
        if np.all(values == values[0]):
            result.update(ci_low=float(values[0]), ci_high=float(values[0]))
            return result
        rng = np.random.default_rng(seed)
        means = np.concatenate([rng.choice(values, (min(256, replicates-i), len(values))).mean(1)
                                for i in range(0, replicates, 256)])
        result.update(ci_low=float(np.quantile(means, .025)), ci_high=float(np.quantile(means, .975)))
    return result


def aggregate_rows(rows, replicates, seed):
    """Average runs within each subject before bootstrapping distinct subjects."""
    per_subject, summary = [], {}
    groups = sorted({(r["split"], r["interval"], r["method"]) for r in rows})
    for split, interval, method in groups:
        group = [r for r in rows if (r["split"], r["interval"], r["method"]) == (split, interval, method)]
        subject_runs = {}
        for row in group:
            subject_runs.setdefault(row["subject_id"], []).append(row)
        subjects = sorted(subject_runs)
        averaged = [{"split": split, "interval": interval, "method": method, "subject_id": subject,
                     **{key: _mean(r[key] for r in subject_runs[subject])
                        for key in SUMMARY_METRICS}} for subject in subjects]
        per_subject.extend(averaged)
        summary.setdefault(split, {}).setdefault(interval, {})[method] = {
            key: (bootstrap_mean([r[key] for r in averaged], replicates, seed) if split == "val" else
                  {"mean": _mean(r[key] for r in averaged), "ci_low": None, "ci_high": None,
                   "n_subjects": sum(r[key] is not None for r in averaged)}) for key in SUMMARY_METRICS}
    return per_subject, summary


def validate_diagnostic_config(config):
    d, data = config.get("diagnostic", {}), config["data"]
    if config["experiment"].get("task") != "ae_diagnostic":
        raise ValueError("Expected experiment.task=ae_diagnostic")
    if d.get("run") != "LR" or set(config["paths"].get("timeseries", {})) != {"LR"}:
        raise ValueError("AE diagnostic requires LR-only inputs")
    for key in ("batch_size", "bootstrap_replicates"):
        if type(d.get(key)) is not int or d[key] < 1:
            raise ValueError(f"diagnostic.{key} must be positive")
    if not np.isfinite(float(d.get("constant_std_threshold", 0))) or float(d.get("constant_std_threshold", 0)) <= 0:
        raise ValueError("constant_std_threshold must be positive")
    for key in ("seed", "bootstrap_seed"):
        if type(d.get(key)) is not int:
            raise ValueError(f"diagnostic.{key} must be an integer")
    if config["training"]["seeds"] != [d["seed"]]:
        raise ValueError("Diagnostic seed must match the single declared reproducibility seed")
    if set(d.get("expected_subjects", {})) != {"train", "val"} or any(
            type(v) is not int or v < 1 for v in d["expected_subjects"].values()):
        raise ValueError("Declare positive expected train/val subject counts")
    windows = (int(data["n_timepoints"])-int(data["window_length"]))//int(data["stride"])+1
    intervals = d.get("intervals", {})
    if set(intervals) != {"full", "target", "history20", "history10", "history5", "history1"}:
        raise ValueError("All six declared intervals are required")
    expected = {"full": [0, windows], "target": [20, windows], "history20": [0, 20],
                "history10": [10, 20], "history5": [15, 20], "history1": [19, 20]}
    if windows <= 20 or intervals != expected:
        raise ValueError("Intervals must match the common FC[19] cutoff and full scan length")
    pca = d.get("pca", {})
    if (type(pca.get("components")) is not int or pca["components"] < 1
            or pca["components"] != int(config["model"]["fc_latent_dim"])
            or type(pca.get("n_oversamples")) is not int or pca["n_oversamples"] < 1
            or type(pca.get("iterated_power")) is not int or pca["iterated_power"] < 0
            or pca.get("power_iteration_normalizer") != "QR"):
        raise ValueError("Declare a matching PCA rank and fixed randomized-SVD settings")
    ref = config.get("artifacts", {}).get("fc_autoencoder", {})
    if any(not ref.get(key) for key in ("id", "path", "sha256")):
        raise ValueError("A frozen FC autoencoder artifact is required")


def prepare_diagnostic(config, device_name="cpu"):
    """Read only metadata for train/val cache entries, then load verified AE weights."""
    from .management import validate_experiment_config
    validate_experiment_config(config)
    bindings = verify_data_bindings(config)
    artifact = verify_artifact(config)
    data, options = config["data"], config["diagnostic"]
    store = zarr.open_group(str(cache_path(config, int(data["window_length"]))), mode="r")
    settings = {key: data[key] for key in ("window_length", "stride", "n_nodes", "fisher_clip")}
    settings["estimator"] = "rectangular_pearson"
    if dict(store.attrs) != {**settings, "config_hash": config_hash(settings)}:
        raise ValueError("FC cache metadata do not match the frozen preprocessing settings")
    table = load_split(resolve_path(config, "split_csv"))
    samples = {}
    shape = (options["intervals"]["full"][1], int(data["n_nodes"])*(int(data["n_nodes"])-1)//2)
    for split in ("train", "val"):
        subjects = sorted(table.loc[table.split == split, "subject_id"].astype(str))
        if len(subjects) != options["expected_subjects"][split]:
            raise ValueError(f"Unexpected {split} subject count: {len(subjects)}")
        samples[split] = [(subject, options["run"]) for subject in subjects]
        for subject, run in samples[split]:
            key = f"subjects/{subject}/{run}"
            if key not in store or tuple(store[key]["fc_z"].shape) != shape:
                raise ValueError(f"Missing or malformed {split} FC cache: {key}")
            if tuple(store[key]["window_starts"].shape) != (shape[0],):
                raise ValueError(f"Malformed window starts: {key}")
    if options["pca"]["components"] > min(shape[1], shape[0]*len(samples["train"])):
        raise ValueError("PCA rank exceeds the training matrix dimensions")
    device = device_from_arg(device_name)
    model = load_autoencoder(config, int(data["window_length"]), device, artifact)
    model.requires_grad_(False).eval()
    audit = {"checkpoint_sha256": file_sha256(artifact), "checkpoint_bytes": artifact.stat().st_size,
             "data_manifest_digest": bindings["manifest_sha256"],
             "split_sha256": data["split_sha256"], "cache_metadata": dict(store.attrs),
             "shape_per_run": list(shape), "samples": {s: [list(x) for x in v] for s, v in samples.items()},
             "test_data_read": False}
    return store, samples, model, device, audit


def _read_sequence(store, subject, run, config):
    group = store[f"subjects/{subject}/{run}"]
    values = np.asarray(group["fc_z"], dtype=np.float32)
    starts = np.asarray(group["window_starts"])
    expected = np.arange(len(values))*int(config["data"]["stride"])
    if not np.array_equal(starts, expected) or not np.isfinite(values).all():
        raise ValueError(f"Invalid FC values or timeline: {subject}/{run}")
    digest = hashlib.sha256(values.tobytes(order="C") + starts.astype("<i8").tobytes()).hexdigest()
    return values, digest


@torch.inference_mode()
def reconstruct(model, values, batch_size, device):
    outputs = [model(torch.from_numpy(np.ascontiguousarray(values[i:i+batch_size])).to(device))[0].cpu().numpy()
               for i in range(0, len(values), batch_size)]
    result = np.concatenate(outputs)
    if result.shape != values.shape or not np.isfinite(result).all():
        raise ValueError("AE returned malformed or nonfinite reconstruction")
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_references(store, samples, config, destination):
    """Fit exact protocol PCA using a disposable, train-only float32 matrix."""
    options = config["diagnostic"]
    windows = options["intervals"]["full"][1]
    edges = int(config["data"]["n_nodes"])*(int(config["data"]["n_nodes"])-1)//2
    template = np.zeros((windows, edges), dtype=np.float64)
    variance = np.zeros(edges, dtype=np.float64)
    digests = []
    with tempfile.TemporaryDirectory(prefix="pca_train_", dir=destination) as temporary:
        if not Path(temporary).resolve().is_relative_to(destination.resolve()):
            raise ValueError("Temporary training matrix must stay inside the diagnostic directory")
        matrix = np.lib.format.open_memmap(Path(temporary)/"train.npy", mode="w+", dtype=np.float32,
                                           shape=(len(samples)*windows, edges))
        try:
            for index, (subject, run) in enumerate(samples):
                values, digest = _read_sequence(store, subject, run, config)
                template += values
                variance += _center(values.astype(np.float64)).var(0)
                matrix[index*windows:(index+1)*windows] = values
                digests.append({"subject_id": subject, "run": run, "sha256": digest})
                if (index+1) % 25 == 0 or index+1 == len(samples):
                    emit("ae_diagnostic_training_cache", completed=index+1, total=len(samples))
            template /= len(samples)
            variance /= len(samples)
            settings = options["pca"]
            emit("ae_diagnostic_pca_started", training_windows=len(matrix), components=settings["components"])
            pca = PCA(n_components=settings["components"], svd_solver="randomized", copy=False,
                      random_state=options["seed"], n_oversamples=settings["n_oversamples"],
                      iterated_power=settings["iterated_power"],
                      power_iteration_normalizer=settings["power_iteration_normalizer"])
            pca.fit(matrix)
        finally:
            # Release the mmap before TemporaryDirectory cleanup on Windows.
            matrix._mmap.close()
            del matrix
    np.savez_compressed(destination/"pca.npz", components=pca.components_, mean=pca.mean_,
                        explained_variance=pca.explained_variance_, singular_values=pca.singular_values_)
    np.savez_compressed(destination/"training_references.npz", template=template,
                        static_mean=template.mean(0), temporal_variance=variance)
    rng = np.random.default_rng(options["seed"])
    # Rank thirds, with deterministic tie-breaking by edge index; two per third.
    bins = np.array_split(np.argsort(variance, kind="stable"), 3)
    selected = [int(e) for group in bins for e in np.sort(rng.choice(group, min(2, len(group)), replace=False))]
    return pca, template, selected, digests


def _summarize_comparisons(per_subject, common_rows, options):
    comparisons = {}
    for interval in options["intervals"]:
        group = [r for r in per_subject if r["split"] == "val" and r["interval"] == interval]
        ae = {r["subject_id"]: r for r in group if r["method"] == "ae"}
        pca = {r["subject_id"]: r for r in group if r["method"] == "pca"}
        comparisons[interval] = {key: bootstrap_mean([
            ae[s][key]-pca[s][key] for s in sorted(ae)
            if ae[s][key] is not None and pca[s][key] is not None],
            options["bootstrap_replicates"], options["bootstrap_seed"])
            for key in ("dynamic_nmse", "mse", "time_r", "difference_time_r", "std_ratio_mean")}
        common = [r for r in common_rows if r["interval"] == interval]
        by_subject = {s: [r for r in common if r["subject_id"] == s] for s in sorted(ae)}
        for key in ("common_time_r_difference", "common_valid_fraction"):
            comparisons[interval][key] = bootstrap_mean(
                [_mean(r[key] for r in values) for values in by_subject.values()],
                options["bootstrap_replicates"], options["bootstrap_seed"])
    return comparisons


def run_diagnostic(config, device_name="cpu", diagnostic_id=None):
    store, samples, model, device, audit = prepare_diagnostic(config, device_name)
    options = config["diagnostic"]
    seed_everything(options["seed"])
    identity = diagnostic_id or f"{config['experiment']['id']}-{utc_now().replace('-', '').replace(':', '')}-{config_sha256(config)[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,110}", identity):
        raise ValueError("diagnostic-id must contain only letters, digits, underscores and hyphens")
    base = resolve_path(config, "output_dir")/config["experiment"]["id"]
    destination = base/"diagnostics"/identity
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "experiment_id": config["experiment"]["id"], "diagnostic_id": identity,
                "task": "contemporaneous_ae_reconstruction", "status": "RUNNING", "started_at": utc_now(),
                "config_sha256": config_sha256(config), "device": str(device), "training_run_created": False}
    _write_json(destination/"metadata.json", metadata)
    try:
        metadata["git"] = git_metadata(config["paths"]["root"])
        (destination/"config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (destination/"environment.txt").write_text(environment_snapshot(), encoding="utf-8")
        for module in (Path(__file__), Path(__file__).with_name("ae_diagnostic_visual.py")):
            shutil.copyfile(module, destination/module.name)
        pca, template, edges, train_digests = _fit_references(store, samples["train"], config, destination)
        rng = np.random.default_rng(options["seed"])
        selected_subjects = sorted(rng.choice([s for s, _ in samples["val"]], min(4, len(samples["val"])), replace=False).tolist())
        _write_json(destination/"selection.json", {"subjects": selected_subjects, "edge_indices": edges,
                    "edge_rule": "training-only mean within-run temporal variance; stable rank thirds; seed-42 two per third",
                    "display_labels": {s: f"V{i+1:02d}" for i, s in enumerate(selected_subjects)}})
        rows, common_rows, examples, val_digests = [], [], {}, []
        threshold = options["constant_std_threshold"]
        total = sum(len(v) for v in samples.values())
        completed = 0
        train_hashes = {(r["subject_id"], r["run"]): r["sha256"] for r in train_digests}
        for split, sample_list in samples.items():
            for subject, run in sample_list:
                x, digest = _read_sequence(store, subject, run, config)
                if split == "train":
                    if digest != train_hashes[(subject, run)]:
                        raise ValueError("Training cache changed during the diagnostic")
                else:
                    val_digests.append({"subject_id": subject, "run": run, "sha256": digest})
                ae = reconstruct(model, x, options["batch_size"], device)
                linear = pca.inverse_transform(pca.transform(x))
                if subject in selected_subjects and split == "val":
                    examples[subject] = {"target": x[:, edges], "ae": ae[:, edges], "pca": linear[:, edges]}
                for interval, (start, stop) in options["intervals"].items():
                    target, reconstructed, projected = x[start:stop], ae[start:stop], linear[start:stop]
                    methods = {"identity": target, "ae": reconstructed, "pca": projected,
                               "train_static": np.broadcast_to(template.mean(0), target.shape),
                               "train_template": template[start:stop],
                               "oracle_mean": np.broadcast_to(target.astype(np.float64).mean(0), target.shape)}
                    for method in METHODS:
                        rows.append({"split": split, "subject_id": subject, "run": run, "interval": interval,
                                     "method": method, **reconstruction_metrics(methods[method], target, threshold)})
                    if split == "val":
                        if len(target) >= 2:
                            a, av, _, _ = correlations(reconstructed.astype(np.float64), target.astype(np.float64), threshold=threshold)
                            p, pv, _, _ = correlations(projected.astype(np.float64), target.astype(np.float64), threshold=threshold)
                            common = av & pv
                            difference, fraction = _mean(a[common]-p[common]), float(common.mean())
                        else:
                            difference = fraction = None
                        common_rows.append({"subject_id": subject, "run": run, "interval": interval,
                                            "common_time_r_difference": difference, "common_valid_fraction": fraction})
                completed += 1
                if completed % 25 == 0 or completed == total:
                    emit("ae_diagnostic_progress", completed=completed, total=total)
        per_subject, summary = aggregate_rows(rows, options["bootstrap_replicates"], options["bootstrap_seed"])
        report = {"schema_version": 1, "experiment_id": config["experiment"]["id"], "diagnostic_id": identity,
                  "primary": {"split": "val", "interval": "full", "metric": "dynamic_nmse", "method": "ae"},
                  "definition": {"representation": "Fisher-z edges; raw_r_* after tanh",
                    "dynamic_nmse": "sum((center(Y)-center(X))**2)/sum(center(X)**2) within run",
                    "aggregation": "runs within subject, then subjects equally; edge ratio quantiles within run before averaging",
                    "constant_std_threshold": threshold, "bootstrap_replicates": options["bootstrap_replicates"],
                    "bootstrap_seed": options["bootstrap_seed"], "ci_level": .95,
                    "training_summary": "descriptive subject means without inferential confidence intervals",
                    "inference": "val/full AE-minus-PCA dynamic NMSE is primary; other contrasts exploratory",
                    "pca_training": "all train LR windows, raw Fisher-z, no per-edge standardization",
                    "limitation": "fixed AE encode/decode; cannot separate encoder/decoder errors or infer future predictability"},
                  "intervals": options["intervals"], "summary": summary,
                  "paired_ae_minus_pca": _summarize_comparisons(per_subject, common_rows, options),
                  "checkpoint_sha256": audit["checkpoint_sha256"], "config_sha256": metadata["config_sha256"]}
        audit["cache_content"] = {"train": train_digests, "val": val_digests}
        audit["cache_content_sha256_definition"] = "SHA256 float32 C-order FC bytes + little-endian int64 window-start bytes"
        _write_json(destination/"data_audit.json", audit)
        _write_json(destination/"summary.json", report)
        _write_csv(destination/"per_run_metrics.csv", rows)
        _write_csv(destination/"per_subject_metrics.csv", per_subject)
        _write_csv(destination/"common_edge_comparisons.csv", common_rows)
        np.savez_compressed(destination/"private_examples.npz", subjects=np.array(selected_subjects),
            edge_indices=np.array(edges), **{method: np.stack([examples[s][method] for s in selected_subjects])
                                             for method in ("target", "ae", "pca")})
        from .ae_diagnostic_visual import render_diagnostic
        images = render_diagnostic(destination)
        metadata.update(status="COMPLETED", ended_at=utc_now(), images=[str(p) for p in images],
                        output_sha256={p.name: file_sha256(p) for p in destination.iterdir()
                                       if p.is_file() and p.name != "metadata.json"})
        _write_json(destination/"metadata.json", metadata)
        return destination
    except Exception as error:
        metadata.update(status="FAILED", ended_at=utc_now(), error=f"{type(error).__name__}: {error}")
        _write_json(destination/"metadata.json", metadata)
        raise


def command_ae_diagnostic(args):
    config = load_config(args.experiment)
    if args.check_only:
        _, _, model, device, audit = prepare_diagnostic(config, args.device)
        edges = int(config["data"]["n_nodes"])*(int(config["data"]["n_nodes"])-1)//2
        with torch.inference_mode():
            output, latent = model(torch.zeros(2, edges, device=device))
        if not torch.isfinite(output).all():
            raise ValueError("Nonfinite synthetic AE forward")
        print(json.dumps({"status": "READY", "checkpoint_sha256": audit["checkpoint_sha256"],
                          "counts": {k: len(v) for k, v in audit["samples"].items()},
                          "shape_per_run": audit["shape_per_run"], "output_shape": list(output.shape),
                          "latent_shape": list(latent.shape), "test_data_read": False}, indent=2))
    else:
        print(run_diagnostic(config, args.device, args.diagnostic_id))
