"""Managed train-only cross-validation and frozen SC-conditioned alpha components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .config import resolve_path
from .connectivity import config_hash
from .data import load_split, load_sc, read_cached, cache_path, _zarr
from .management import file_sha256, verify_decay_artifact
from .models.sc_alpha import SCAlphaGenerator, alpha_basis, sufficient_mse


MODEL_NAME = "sc_conditioned_alpha"
SCHEMA = "sc_conditioned_alpha/v1"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def tensor_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validate_sc_alpha_config(config):
    m, t, d, e = (config[s] for s in ("model", "training", "data", "evaluation"))
    steps = (int(d["n_timepoints"]) - int(d["window_length"])) // int(d["stride"])
    if (m.get("sc_encoder") not in {"linear", "mlp", "gcn"}
            or m.get("alpha_parameters") not in {1, 6, steps}
            or m.get("alpha_clip") is not False or int(d.get("warmup_windows", 1)) != 1
            or set(config["paths"]["timeseries"]) != {"LR"}
            or config["experiment"].get("ablation") != "full"
            or e.get("primary_metric") != "long_edge_mse"):
        raise ValueError("SC alpha requires LR-only one-window inputs, unbounded 1/6/all-point alpha and long MSE")
    for key in ("pca_components", "mlp_hidden", "gcn_hidden", "gcn_output"):
        if type(m.get(key)) is not int or m[key] < 1:
            raise ValueError(f"SC alpha model.{key} must be positive")
    if not 0 <= float(m.get("dropout", -1)) < 1:
        raise ValueError("SC alpha dropout must be in [0,1)")
    if (type(t.get("cv_folds")) is not int or t["cv_folds"] < 2
            or any(type(t.get(k)) is not int or t[k] < 1 for k in ("epochs", "batch_size", "patience"))
            or not isinstance(t.get("alpha_lambda_grid"), list) or not t["alpha_lambda_grid"]
            or any(not math.isfinite(float(v)) or float(v) < 0 for v in t["alpha_lambda_grid"])
            or len(set(t["alpha_lambda_grid"])) != len(t["alpha_lambda_grid"])
            or t.get("alpha_training_scope") != "all_future"
            or t.get("loss_type") != "mse"
            or t.get("learning_rate_schedule", {}).get("name") != "constant"):
        raise ValueError("Invalid SC alpha CV/optimization protocol")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(float(t.get(key, 0))) or float(t[key]) <= 0:
            raise ValueError(f"SC alpha training.{key} must be positive")
    if not math.isfinite(float(t.get("weight_decay", -1))) or float(t["weight_decay"]) < 0:
        raise ValueError("SC alpha weight decay must be nonnegative")
    if t.get("seeds") != [42] or int(config.get("seed", 42)) != 42:
        raise ValueError("SC alpha v1 is a seed42 experiment")
    if not 0 < int(e.get("nonoverlap_start", -1)) < steps - 3:
        raise ValueError("Invalid SC alpha nonoverlap interval")
    from .connectivity import nonoverlap_horizon
    if int(e["nonoverlap_start"]) != nonoverlap_horizon(d["window_length"], d["stride"]):
        raise ValueError("SC alpha nonoverlap must match the conservative project interval")
    for name in ("alpha_fit", "training_stats"):
        ref = config.get("artifacts", {}).get(name, {})
        if not ref.get("path") or not ref.get("sha256") or len(ref["sha256"]) != 64:
            raise ValueError(f"SC alpha requires frozen artifacts.{name}")
    expected = t.get("alpha_expected_subjects", {})
    if set(expected) != {"train", "val"} or any(type(x) is not int or x < 2 for x in expected.values()):
        raise ValueError("Declare SC alpha train/val counts")
    plan = config.get("comparison_plan", {})
    if (len(plan.get("suite_experiments", [])) != 9 or len(set(plan["suite_experiments"])) != 9
            or config["experiment"]["id"] not in plan["suite_experiments"]
            or int(plan.get("primary_family_size", 0)) != 9
            or int(plan.get("bootstrap_replicates", 0)) < 1
            or int(plan.get("secondary_bootstrap_replicates", 0)) < 1):
        raise ValueError("SC alpha needs the declared nine-model comparison family")
    if plan.get("require_suite_ready") is not True and int(config["experiment"]["level"]) != 0:
        raise ValueError("Formal alpha evaluation requires all nine training runs to finish")
    seeds = plan.get("swap_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Declare unique integer SC swap seeds")
    if int(config["experiment"]["level"]) >= 1 and (len(seeds) != 10 or plan.get("test_access") is not False):
        raise ValueError("Formal SC alpha uses ten SC swaps and forbids test access")


def reference_arrays(config):
    alpha_path = verify_decay_artifact(config)
    ref = config["artifacts"]["training_stats"]
    stats_path = Path(config["paths"]["root"]) / ref["path"]
    if file_sha256(stats_path) != ref["sha256"]:
        raise ValueError("SC alpha training template checksum mismatch")
    with np.load(alpha_path) as a, np.load(stats_path) as s:
        alpha = np.array(a["alpha_raw"], dtype=np.float32)
        mean = np.array(s["group_template"], dtype=np.float32)
        context = np.array(s["context_template"], dtype=np.float32)
    steps = (config["data"]["n_timepoints"] - config["data"]["window_length"]) // config["data"]["stride"]
    edges = config["data"]["n_nodes"] * (config["data"]["n_nodes"]-1)//2
    if (alpha.shape != (steps,) or mean.shape != (steps, edges) or context.shape != (edges,)
            or not all(np.isfinite(x).all() for x in (alpha, mean, context))):
        raise ValueError("SC alpha frozen reference shape/finite mismatch")
    return context, mean, alpha


@dataclass
class AlphaCohort:
    subjects: list[str]
    sc: np.ndarray
    fc: np.ndarray
    digest: str


def load_cohort(config, split):
    """Read only the declared split's LR caches and SC; never discovers test files."""
    if split not in {"train", "val"}:
        raise PermissionError("SC alpha v1 permits train/val only")
    rows = load_split(resolve_path(config, "split_csv"))
    subjects = sorted(rows.loc[rows.split == split, "subject_id"].astype(str).tolist())
    expected = config["training"]["alpha_expected_subjects"][split]
    if len(subjects) != expected or len(set(subjects)) != len(subjects):
        raise ValueError("SC alpha split subject count/uniqueness mismatch")
    d = config["data"]
    windows = (d["n_timepoints"]-d["window_length"])//d["stride"]+1
    edges = d["n_nodes"]*(d["n_nodes"]-1)//2
    store = _zarr()[0].open_group(str(cache_path(config, d["window_length"])), mode="r")
    settings = {"window_length": d["window_length"], **{k: d[k] for k in ("stride", "n_nodes", "fisher_clip")},
                "estimator": "rectangular_pearson"}
    if dict(store.attrs) != {**settings, "config_hash": config_hash(settings)}:
        raise ValueError("SC alpha FC cache preprocessing mismatch")
    manifest = json.loads((Path(config["paths"]["root"])/d["manifest_path"]).read_text(encoding="utf-8"))
    files = {r["path"].replace("\\", "/"): r for r in manifest["files"]}
    fc = np.empty((len(subjects), windows, edges), dtype=np.float32)
    sc = np.empty((len(subjects), d["n_nodes"], d["n_nodes"]), dtype=np.float32)
    h = hashlib.sha256()
    for i, subject in enumerate(subjects):
        sc_path = resolve_path(config, "sc_dir") / f"{subject}.csv"
        # Keep lexical paths: managed worktrees may use a data junction to the main checkout.
        relative = sc_path.absolute().relative_to(Path(config["paths"]["root"]).absolute()).as_posix()
        if relative not in files or file_sha256(sc_path) != files[relative]["sha256"]:
            raise ValueError("SC content does not match the frozen data manifest")
        values, starts = read_cached(config, d["window_length"], subject, "LR")
        graph = load_sc(config, subject)
        if (values.shape != (windows, edges) or not np.isfinite(values).all()
                or not np.array_equal(starts, np.arange(windows)*d["stride"])
                or graph.shape != (d["n_nodes"], d["n_nodes"]) or not np.isfinite(graph).all()
                or np.any(graph < 0) or not np.allclose(graph, graph.T, rtol=1e-5, atol=1e-5)):
            raise ValueError("SC alpha requires finite symmetric nonnegative SC and complete LR FC timelines")
        fc[i], sc[i] = values, graph
        h.update(subject.encode()); h.update(fc[i].tobytes()); h.update(sc[i].tobytes()); h.update(starts.tobytes())
    return AlphaCohort(subjects, sc, fc, h.hexdigest())


def fit_templates(cohort, indices):
    # Stream float64 accumulation instead of copying the full multi-GB cohort.
    mean = np.zeros(cohort.fc.shape[1:], dtype=np.float64)
    for i in indices:
        mean += cohort.fc[i]
    mean /= len(indices)
    context, template = mean[0], mean[1:]
    numerator = np.zeros(len(template), dtype=np.float64)
    denominator = 0.
    for i in indices:
        offset = cohort.fc[i, 0].astype(np.float64)-context
        numerator += np.sum(offset[None]*(cohort.fc[i, 1:].astype(np.float64)-template), axis=1)
        denominator += np.sum(offset**2)
    return context.astype(np.float32), template.astype(np.float32), (numerator/(denominator+1e-12)).astype(np.float32)


def statistics(cohort, context, template):
    q, u, v = [], [], []
    for fc in cohort.fc:
        offset = fc[0].astype(np.float64)-context.astype(np.float64)
        target = fc[1:].astype(np.float64)-template.astype(np.float64)
        q.append(np.mean(offset**2)); u.append(np.mean(offset[None]*target, axis=1)); v.append(np.mean(target**2, axis=1))
    return np.asarray(q), np.asarray(u), np.asarray(v)


def fit_sc_transform(sc, indices, config):
    if config["model"]["sc_encoder"] == "gcn":
        return {"kind": "gcn", "n_nodes": sc.shape[1]}
    rows, cols = np.triu_indices(sc.shape[1], 1)
    x = np.log1p(sc[indices][:, rows, cols].astype(np.float64))
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    dimensions = config["model"]["pca_components"]
    if dimensions > min(x.shape):
        raise ValueError("SC PCA rank exceeds the fold's fitting samples/features")
    pca = PCA(n_components=dimensions, whiten=False, svd_solver="full").fit((x-mean)/scale)
    return {"kind": "pca", "n_nodes": sc.shape[1], "mean": mean, "scale": scale,
            "pca_mean": pca.mean_, "components": pca.components_}


def transform_sc(sc, transform):
    sc = np.asarray(sc)
    if sc.ndim != 3 or sc.shape[1:] != (transform["n_nodes"], transform["n_nodes"]) or not np.isfinite(sc).all() or np.any(sc < 0):
        raise ValueError("Expected finite nonnegative SC matrices")
    if transform["kind"] == "gcn":
        return sc.astype(np.float32)
    r, c = np.triu_indices(sc.shape[1], 1)
    x = np.log1p(sc[:, r, c].astype(np.float64))
    return (((x-transform["mean"])/transform["scale"]-transform["pca_mean"]) @ transform["components"].T).astype(np.float32)


def generator(config):
    m, d = config["model"], config["data"]
    steps = (d["n_timepoints"]-d["window_length"])//d["stride"]
    return SCAlphaGenerator(m["sc_encoder"], m["alpha_parameters"], steps,
        n_nodes=d["n_nodes"], pca_components=m["pca_components"], mlp_hidden=m["mlp_hidden"],
        dropout=m["dropout"], gcn_hidden=m["gcn_hidden"], gcn_output=m["gcn_output"])


def fit_one(config, features, stat, alpha, fitting, heldout, penalty, device, *, epochs=None, log_path=None):
    from .training import seed_everything
    seed_everything(42)
    model = generator(config).to(device)
    initial = {"encoder_sha256": tensor_digest(model.encoder), "model_sha256": tensor_digest(model),
               "parameters": sum(p.numel() for p in model.parameters())}
    seed_everything(42)
    initial["cpu_rng_sha256"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    initial["device_rng_sha256"] = (hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
                                    if device.type == "cuda" else None)
    x = torch.as_tensor(features, device=device)
    q, u, v = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in stat)
    base = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    t = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    shuffle = np.random.default_rng(42)
    best_score, best_epoch, stale = float("inf"), 0, 0
    last = None
    for epoch in range(1, (epochs or t["epochs"])+1):
        model.train()
        order = shuffle.permutation(fitting)
        total = 0.
        for start in range(0, len(order), t["batch_size"]):
            ids = order[start:start+t["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            delta = model(x[ids]); values = sufficient_mse(base[None]+delta, q[ids], u[ids], v[ids])
            loss = values.mean()+penalty*delta.square().mean()
            if not torch.isfinite(loss) or not torch.isfinite(delta).all():
                raise FloatingPointError("Nonfinite SC alpha loss/correction")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=False)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite SC alpha gradient")
            optimizer.step()
            total += float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            delta = model(x)
            if not torch.isfinite(delta).all():
                raise FloatingPointError("Nonfinite SC alpha prediction")
            values = sufficient_mse(base[None]+delta, q, u, v)
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite SC alpha squared error")
            train_mse = float(values[fitting].mean())
            score = (float(values[heldout, config["evaluation"]["nonoverlap_start"]:].mean())
                     if heldout is not None else None)
        last = {"event": "epoch_complete", "epoch": epoch, "train_loss": total/len(fitting),
                "train_mse": train_mse, "cv_long_edge_mse": score}
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last, allow_nan=False)+"\n")
        if heldout is not None:
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= t["patience"]:
                break
    return model.cpu(), {"best_score": best_score if heldout is not None else None,
                         "best_epoch": best_epoch if heldout is not None else epochs,
                         "last": last, "initial": initial}


def run_sc_alpha(config, run_dir, metadata, device_name=None):
    """Fit using train subjects only. No validation data access or model selection."""
    from .training import device_from_arg
    device = device_from_arg(device_name)
    context, template, alpha = reference_arrays(config)
    cohort = load_cohort(config, "train")
    fitting_all = np.arange(len(cohort.subjects))
    empirical = fit_templates(cohort, fitting_all)
    # Match the saved float32 E0032 baseline, allowing only summation precision.
    for actual, expected in zip(empirical, (context, template, alpha)):
        if not np.allclose(actual, expected, atol=2e-6, rtol=2e-5):
            raise ValueError("Frozen E0032 templates/alpha do not match this training cohort")
    folds = list(KFold(config["training"]["cv_folds"], shuffle=True, random_state=42).split(fitting_all))
    private = Path(run_dir)/"private"
    private.mkdir(exist_ok=True)
    write_json(private/"cv_folds.json", {"subjects": cohort.subjects,
        "folds": [{"fit": a.tolist(), "heldout": b.tolist()} for a, b in folds]})
    records = {float(p): [] for p in config["training"]["alpha_lambda_grid"]}
    for fold, (fitting, heldout) in enumerate(folds):
        c, mu, a = fit_templates(cohort, fitting)
        stat = statistics(cohort, c, mu)
        transform = fit_sc_transform(cohort.sc, fitting, config)
        features = transform_sc(cohort.sc, transform)
        for penalty in records:
            label = f"cv_fold{fold+1}_lambda{penalty:g}.log"
            try:
                _, result = fit_one(config, features, stat, a, fitting, heldout, penalty, device,
                                    log_path=Path(run_dir)/label)
                records[penalty].append({"fold": fold, "n_heldout": len(heldout), "status": "COMPLETED", **result})
            except FloatingPointError as error:
                records[penalty].append({"fold": fold, "status": "FAILED", "error": str(error)})
            write_json(Path(run_dir)/"cv_results.json", {"schema": SCHEMA, "candidates": records})
            print(f"SC alpha fold={fold+1}/{len(folds)} lambda={penalty:g}: {records[penalty][-1]['status']}", flush=True)
    scores = {p: sum(r["best_score"]*r["n_heldout"] for r in rows)/sum(r["n_heldout"] for r in rows)
              for p, rows in records.items() if all(r["status"] == "COMPLETED" for r in rows)}
    if not scores:
        raise RuntimeError("All SC alpha regularization candidates failed")
    selected = min(scores, key=lambda p: (scores[p], -p))
    epochs = int(np.median([r["best_epoch"] for r in records[selected]]))
    transform = fit_sc_transform(cohort.sc, fitting_all, config)
    model, final = fit_one(config, transform_sc(cohort.sc, transform), statistics(cohort, context, template),
        alpha, fitting_all, None, selected, device, epochs=epochs, log_path=Path(run_dir)/"train.log")
    selection = {"selected_lambda": selected, "final_epochs": epochs, "cv_long_edge_mse": scores[selected],
                 "training_cohort_sha256": cohort.digest, "cv_folds_sha256": file_sha256(private/"cv_folds.json"),
                 "n_subjects": len(cohort.subjects), "validation_data_read": False,
                 "initialization": final["initial"], "all_candidate_scores": scores}
    write_json(Path(run_dir)/"alpha_selection.json", selection)
    payload = {"schema": SCHEMA, **metadata, "config": deepcopy(config), "model": model.state_dict(),
               "sc_transform": transform, "context_template": context, "group_template": template, "alpha_raw": alpha,
               "selection": selection, "training_cohort_sha256": cohort.digest}
    path = Path(run_dir)/"checkpoints/best.pt"
    torch.save(payload, path)
    write_json(Path(run_dir)/"component_manifest.json", {"schema": SCHEMA, "checkpoint": "checkpoints/best.pt",
        "sha256": file_sha256(path), "config_sha256": metadata["config_sha256"],
        "alpha_clip": False, "sc_only_generator": True, "validation_data_read": False})
    # A CV selection record must never pretend to be external validation metrics.
    from .metric_records import selection_record
    record = selection_record({"long_edge_mse": scores[selected]}, "long_edge_mse", None,
        definition={"implementation": "sc_alpha_training_cv/v1", "split": "train_cv",
                    "aggregation": "out-of-fold subject-weighted long MSE at fold-selected epochs",
                    "nonoverlap_start": config["evaluation"]["nonoverlap_start"]},
        source="sc_alpha_training_cv", split="train_cv")
    write_json(Path(run_dir)/"metrics_best.json", record)
    return path


class FrozenSCAlpha:
    def __init__(self, payload, device):
        self.payload = payload
        self.device = torch.device(device)
        self.model = generator(payload["config"]).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def predict_alpha(self, sc):
        x = torch.as_tensor(transform_sc(sc, self.payload["sc_transform"]), device=self.device)
        delta = self.model(x).cpu().numpy().astype(np.float64)
        if not np.isfinite(delta).all():
            raise FloatingPointError("Nonfinite frozen SC alpha")
        return self.payload["alpha_raw"].astype(np.float64)[None]+delta

    def predict(self, sc, first_fc):
        alpha = self.predict_alpha(sc)
        offset = np.asarray(first_fc, dtype=np.float64)-self.payload["context_template"].astype(np.float64)
        if offset.shape != (len(sc), self.payload["group_template"].shape[1]) or not np.isfinite(offset).all():
            raise ValueError("First-window FC dimensions/finite mismatch")
        return self.payload["group_template"].astype(np.float64)[None]+alpha[:, :, None]*offset[:, None]


def load_sc_alpha_component(checkpoint, config=None, device="cpu"):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint.parent.parent/"component_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or file_sha256(checkpoint) != manifest["sha256"]:
        raise ValueError("SC alpha component checksum mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unknown SC alpha component schema")
    from .management import config_sha256
    if (config_sha256(payload["config"]) != payload["config_sha256"]
            or manifest["config_sha256"] != payload["config_sha256"]):
        raise ValueError("SC alpha checkpoint protocol mismatch")
    if config is not None:
        if config_sha256(config) != payload["config_sha256"]:
            raise ValueError("SC alpha run configuration differs from component")
        expected = reference_arrays(config)
        for key, value in zip(("context_template", "group_template", "alpha_raw"), expected):
            if not np.array_equal(payload[key], value):
                raise ValueError("SC alpha component frozen reference mismatch")
    m, d = payload["config"]["model"], payload["config"]["data"]
    steps = len(payload["alpha_raw"])
    basis = alpha_basis(steps, m["alpha_parameters"])
    if not np.array_equal(payload["model"]["basis"].numpy(), basis):
        raise ValueError("SC alpha component basis mismatch")
    return FrozenSCAlpha(payload, device)
