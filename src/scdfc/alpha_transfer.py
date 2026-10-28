"""Frozen subject-specific alpha in the AE direct-residual Transformer.

The FC-only ablation applies to the residual condition, not the SC-alpha path.
Constructing a frozen component must not consume the residual training RNG.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .config import load_config
from .management import config_sha256, file_sha256, verify_artifact, verify_data_bindings
from .sc_alpha import generator, load_sc_alpha_component, tensor_digest, transform_sc


def validate_transfer_config(config):
    m, t, d, e = (config[k] for k in ("model", "training", "data", "evaluation"))
    if (config["experiment"].get("task") != "sequence" or m.get("name") != "transformer"
            or m.get("fc_representation", "autoencoder") != "autoencoder"
            or m.get("output_decomposition") != "e0032_decay_residual"
            or m.get("sc_alpha_frozen") is not True or m.get("alpha_parameters") != 223
            or m.get("alpha_clip") is not False or d.get("warmup_windows") != 1
            or e.get("primary_metric") != "long_edge_mse"
            or t.get("loss_type") != "mse"
            or {k: float(v) for k, v in t["loss_weights"].items() if float(v)} != {"edge": 1.0}):
        raise ValueError("Alpha transfer requires AE, frozen unbounded 223-alpha, FC1 and pure MSE")
    for name in ("sc_alpha", "residual_reference", "training_stats"):
        ref = config["artifacts"].get(name, {})
        digest = str(ref.get("sha256", ""))
        if not ref.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Alpha transfer requires pinned artifacts.{name}")
    source = config["artifacts"]["sc_alpha"].get("experiment_id")
    if source not in {"E0054", "E0057", "E0060"}:
        raise ValueError("Alpha transfer sources are E0054, E0057 and E0060")
    reference = "E0036" if config["experiment"]["ablation"] == "fc1_only" else "E0037"
    if config["artifacts"]["residual_reference"].get("experiment_id") != reference:
        raise ValueError("Residual reference must match the FC1/full residual ablation")
    plan = config.get("comparison_plan", {})
    if (plan.get("primary_family_size") != 6 or plan.get("test_access") is not False
            or len(plan.get("swap_seeds", [])) != 10
            or plan.get("initialization_policy") != "same_residual_initial_state_and_rng"):
        raise ValueError("Declare six comparisons, ten SC swaps and paired residual initialization")


def pinned_path(config, name):
    ref = config["artifacts"][name]
    path = Path(config["paths"]["root"]) / ref["path"]
    if file_sha256(path) != ref["sha256"]:
        raise ValueError(f"Alpha transfer {name} checksum mismatch")
    return path


def verify_transfer_bindings(config):
    validate_transfer_config(config)
    source = load_sc_alpha_component(pinned_path(config, "sc_alpha"))
    payload = source.payload
    ref = config["artifacts"]["sc_alpha"]
    if (payload["config"]["experiment"]["id"] != ref["experiment_id"]
            or payload.get("run_id") != ref["run_id"]
            or payload["config"]["model"]["alpha_parameters"] != 223
            or payload["selection"].get("validation_data_read") is not False):
        raise ValueError("SC alpha source identity or training provenance mismatch")
    for key in ("dataset_version", "preprocessing_version", "split_version", "manifest_sha256", "split_sha256",
                "n_nodes", "n_timepoints", "window_length", "stride", "warmup_windows"):
        if payload["config"]["data"][key] != config["data"][key]:
            raise ValueError(f"SC alpha data binding differs: {key}")
    if payload["config"]["artifacts"]["alpha_fit"]["sha256"] != config["artifacts"]["alpha_fit"]["sha256"]:
        raise ValueError("SC alpha and Transformer shared alpha differ")
    with np.load(pinned_path(config, "training_stats")) as stats:
        for key in ("context_template", "group_template"):
            if not np.array_equal(payload[key], stats[key]):
                raise ValueError(f"SC alpha and Transformer {key} differ")
    reference = pinned_path(config, "residual_reference")
    old = load_config(reference.parent.parent / "config_resolved.yaml")
    old["paths"]["root"] = config["paths"]["root"]
    meta = json.loads((reference.parent.parent / "metadata.json").read_text(encoding="utf-8"))
    if (meta["status"] != "COMPLETED" or meta["config_sha256"] != config_sha256(old)
            or meta["experiment_id"] != config["artifacts"]["residual_reference"]["experiment_id"]
            or meta["seed"] != config["seed"] or old["data"] != config["data"]
            or {k: v for k, v in old["training"].items() if k != "patience"}
                != {k: v for k, v in config["training"].items() if k != "patience"}
            or old["experiment"]["ablation"] != config["experiment"]["ablation"]
            or old["artifacts"]["fc_autoencoder"] != config["artifacts"]["fc_autoencoder"]):
        raise ValueError("Residual reference data/AE/optimization protocol mismatch")
    extra = {"fc_representation", "sc_alpha_frozen", "alpha_parameters", "alpha_clip"}
    if {k: v for k, v in config["model"].items() if k not in extra} != old["model"]:
        raise ValueError("Residual architecture differs from the pinned AE reference")
    verify_data_bindings(config)
    verify_artifact(config)
    return source, old


class FrozenAlphaBranch(nn.Module):
    def __init__(self, payload):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.generator = generator(payload["config"])
        self.generator.load_state_dict(payload["model"])
        self.transform = deepcopy(payload["sc_transform"])
        self.register_buffer("alpha_raw", torch.as_tensor(payload["alpha_raw"], dtype=torch.float32))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent model.train() must never re-enable alpha MLP dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, sc):
        features = transform_sc(sc.detach().cpu().numpy(), self.transform)
        x = torch.as_tensor(features, device=self.alpha_raw.device)
        alpha = self.alpha_raw[None] + self.generator(x)
        if not torch.isfinite(alpha).all():
            raise FloatingPointError("Nonfinite frozen subject alpha")
        return alpha


def attach_frozen_alpha(model, config, device, checkpoint_payload=None):
    # Loading an independent module must preserve CPU/CUDA training RNG.
    with torch.random.fork_rng(devices=[]):
        frozen = load_sc_alpha_component(pinned_path(config, "sc_alpha"))
        branch = FrozenAlphaBranch(frozen.payload).to(device)
    for key in ("context_template", "group_template"):
        if not np.array_equal(getattr(model, key).detach().cpu().numpy(), frozen.payload[key]):
            raise ValueError(f"Frozen alpha {key} differs from residual model")
    if not torch.equal(branch.alpha_raw.cpu(), model.decay_alpha.cpu()):
        raise ValueError("Frozen subject alpha and residual shared alpha differ")
    if checkpoint_payload is not None:
        if checkpoint_payload.get("sc_alpha_sha256") != config["artifacts"]["sc_alpha"]["sha256"]:
            raise ValueError("Checkpoint SC alpha binding differs")
        for key, value in branch.state_dict().items():
            saved = checkpoint_payload["model"].get("sc_alpha_component." + key)
            if saved is None or not torch.equal(saved.cpu(), value.cpu()):
                raise ValueError("Frozen SC alpha state changed inside residual checkpoint")
    model.sc_alpha_component = branch


def training_protocol(config, model, samples):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return {
        "fc_representation": "autoencoder", "sc_alpha_frozen": True,
        "sc_alpha_sha256": config["artifacts"]["sc_alpha"]["sha256"],
        "sc_alpha_state_sha256": tensor_digest(model.sc_alpha_component),
        "initial_parameters_sha256": digest.hexdigest(),
        "initial_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "initial_cuda_rng_sha256": (hashlib.sha256(torch.cuda.get_rng_state(next(model.parameters()).device).cpu().numpy().tobytes()).hexdigest()
                                    if next(model.parameters()).is_cuda else None),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sample_manifest_sha256": hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
    }
