"""PCA sequence provenance, replay checks and subject-paired comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .fc_pca import load_frozen_fc_pca, verified_pca_paths


def load_sequence_pca(config, device, artifact_path=None, checkpoint_payload=None):
    manifest, archive = verified_pca_paths(config)
    if artifact_path is not None and Path(artifact_path).resolve() != archive.resolve():
        raise ValueError("PCA parameter path differs from the experiment binding")
    component = load_frozen_fc_pca(manifest, config=config, device=device)
    if checkpoint_payload is not None:
        ref = config["artifacts"]["fc_pca"]
        expected = {"fc_representation": "frozen_pca", "artifact_sha256": ref["sha256"],
                    "fc_pca_manifest_sha256": ref["manifest_sha256"]}
        for key, value in expected.items():
            if checkpoint_payload.get(key) != value:
                raise ValueError(f"Checkpoint PCA provenance mismatch: {key}")
        # The historical state_dict stores the same component under two aliases.
        # Validate both before load_state_dict can overwrite the verified basis.
        for prefix in ("fc_autoencoder", "condition_encoder.fc_autoencoder"):
            for key in ("components", "mean"):
                stored = checkpoint_payload["model"].get(f"{prefix}.{key}")
                if stored is None or not torch.equal(stored.cpu(), getattr(component, key).cpu()):
                    raise ValueError(f"Checkpoint PCA buffer differs from frozen artifact: {prefix}.{key}")
    return component


def component_training_protocol(config, model, samples, device):
    initial = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            initial.update(name.encode())
            initial.update(parameter.detach().cpu().numpy().tobytes())
    ref = config["artifacts"]["fc_pca"]
    return {"fc_representation": "frozen_pca", "fc_component_id": ref["id"],
            "artifact_sha256": ref["sha256"], "fc_pca_manifest_sha256": ref["manifest_sha256"],
            "fc_input_component_frozen": True, "fc_reconstruction_decoder_used": False,
            "fc_reconstruction_decoder_frozen": None,
            "initialization_implementation": "historical_ae_constructor_rng/v1",
            "initial_parameters_sha256": initial.hexdigest(),
            "initial_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "initial_device_rng_sha256": (hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
                                          if device.type == "cuda" else None),
            "sample_manifest_sha256": hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
            "loader_seed": int(config["seed"]),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def validate_pca_pair(config, baseline_config, metadata, baseline_metadata):
    """Allow only predeclared encoder or SC contrasts with the same protocol."""
    plan = config["comparison_plan"]
    baseline_id = baseline_metadata["experiment_id"]
    reference = plan["ae_reference"]
    sc_reference = plan.get("sc_reference")
    if baseline_id not in {reference, sc_reference}:
        raise ValueError("Baseline is not a predeclared PCA experiment contrast")
    if (baseline_metadata.get("status") != "COMPLETED"
            or baseline_metadata["seed"] != metadata["seed"]
            or any(config[s] != baseline_config[s] for s in ("data", "training"))):
        raise ValueError("Paired PCA baseline must be completed with matching seed/data/training")
    main_model, other_model = dict(config["model"]), dict(baseline_config["model"])
    main_representation = main_model.pop("fc_representation", "autoencoder")
    other_representation = other_model.pop("fc_representation", "autoencoder")
    if main_representation != "frozen_pca" or main_model != other_model:
        raise ValueError("Paired PCA prediction architectures differ")
    if config["artifacts"]["alpha_fit"] != baseline_config["artifacts"]["alpha_fit"]:
        raise ValueError("Paired PCA baselines must use the same frozen alpha artifact")
    if config["evaluation"]["primary_metric"] != baseline_config["evaluation"]["primary_metric"]:
        raise ValueError("Paired PCA baselines must use the same selection metric")
    if baseline_id == reference:
        if (other_representation != "autoencoder" or baseline_config["experiment"]["ablation"] != config["experiment"]["ablation"]):
            raise ValueError("AE reference must match the PCA arm's SC condition")
    elif (other_representation != "frozen_pca" or config["experiment"]["ablation"] != "full"
          or baseline_config["experiment"]["ablation"] != "fc1_only"
          or config["artifacts"]["fc_pca"] != baseline_config["artifacts"]["fc_pca"]):
        raise ValueError("SC contrast must use a matching FC1-only PCA baseline")


def paired_metric(main, baseline, subjects, *, replicates, seed, confidence=.95, minimize=False, family_size=1):
    """Bootstrap subject means, with marginal and family-adjusted intervals."""
    main, baseline = np.asarray(main, float), np.asarray(baseline, float)
    subjects = np.asarray(subjects)
    valid = np.isfinite(main) & np.isfinite(baseline)
    if not valid.any():
        return None
    if replicates < 1 or not 0 < confidence < 1 or family_size < 1:
        raise ValueError("Invalid paired bootstrap protocol")
    ids = np.unique(subjects[valid])
    differences = main-baseline
    values = np.array([differences[valid & (subjects == s)].mean() for s in ids])
    baseline_mean = float(np.mean([baseline[valid & (subjects == s)].mean() for s in ids]))
    rng = np.random.default_rng(seed)
    estimates = np.concatenate([values[rng.integers(len(values), size=(min(256, replicates-i), len(values)))].mean(1)
                                for i in range(0, replicates, 256)])
    tail = (1-confidence)/2
    low, high = np.quantile(estimates, [tail, 1-tail])
    adjusted_low, adjusted_high = np.quantile(estimates, [tail/family_size, 1-tail/family_size])
    return {"mean_difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "ci_level": confidence, "n_subjects": len(ids), "n_valid_samples": int(valid.sum()),
            "replicates": replicates, "seed": seed, "bootstrap_unit": "subject",
            "passes": bool(high < 0 if minimize else low > 0),
            "relative_change": float(values.mean()/baseline_mean) if baseline_mean else None,
            "family_adjusted_ci": {"level": 1-(1-confidence)/family_size,
                                   "ci_low": float(adjusted_low), "ci_high": float(adjusted_high),
                                   "family_size": family_size,
                                   "passes": bool(adjusted_high < 0 if minimize else adjusted_low > 0)}}


def paired_pca_report(config, baseline_config, rows, baseline_rows, subjects):
    plan = config["comparison_plan"]
    primary = baseline_config["experiment"]["id"] == plan["ae_reference"]
    metrics = ("long_edge_mse", "time_pearson", "template_adjusted_time_pearson",
               "dynamic_time_pearson", "dynamic_difference_time_pearson", "dynamic_std_ratio",
               "time_pearson_valid_fraction", "template_adjusted_valid_fraction",
               "dynamic_valid_fraction", "difference_valid_fraction", "raw_edge_pearson",
               "long_spatial_pearson", "stable_residual_mse", "centered_dynamic_mse")
    return {metric: paired_metric(
        [np.nan if r.get(metric) is None else r[metric] for r in rows],
        [np.nan if r.get(metric) is None else r[metric] for r in baseline_rows], subjects,
        replicates=int(plan["bootstrap_replicates"]), seed=int(plan["bootstrap_seed"]),
        confidence=float(plan["marginal_ci_level"]), minimize=metric.endswith("_mse"),
        family_size=int(plan["primary_family_size"]) if primary and metric == "long_edge_mse" else 1,
    ) for metric in metrics}
