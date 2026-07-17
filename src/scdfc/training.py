from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import resolve_path
from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset, FCWindowDataset
from .losses import CompositeLoss, correlation_loss, psd_penalty
from .metrics import sequence_metrics
from .models import ConditionalSequenceModel, FCAutoencoder
from .models.baselines import DirectSCMLP, GCNGRUBaseline


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str | None = None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def autoencoder_checkpoint_path(config: dict[str, Any], window_length: int) -> Path:
    return resolve_path(config, "output_dir") / f"window_{window_length}" / "fc_autoencoder.pt"


def train_autoencoder(config: dict[str, Any], window_length: int, stats_path: Path, device_name: str | None = None) -> Path:
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    sequence = DFCSequenceDataset(config, window_length, "train", stats_path)
    dataset = FCWindowDataset(sequence, windows_per_run=32, seed=int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["training"]["autoencoder_batch_size"]), shuffle=True, num_workers=0)
    val_sequence = DFCSequenceDataset(config, window_length, "val", stats_path)
    val_dataset = FCWindowDataset(val_sequence, windows_per_run=8, seed=int(config["seed"]) + 1)
    val_loader = DataLoader(val_dataset, batch_size=int(config["training"]["autoencoder_batch_size"]), shuffle=False, num_workers=0)
    model = FCAutoencoder(4005, int(config["model"]["fc_latent_dim"]), float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    best = float("inf")
    checkpoint = autoencoder_checkpoint_path(config, window_length)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stale = 0
    for epoch in range(int(config["training"]["autoencoder_epochs"])):
        model.train()
        total = 0.0
        for edges in loader:
            edges = edges.to(device)
            reconstructed, _ = model(edges)
            loss = torch.nn.functional.smooth_l1_loss(reconstructed, edges) + 0.1 * correlation_loss(reconstructed, edges) + 0.01 * psd_penalty(reconstructed[:, None])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            total += float(loss) * len(edges)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for edges in val_loader:
                edges = edges.to(device)
                reconstructed, _ = model(edges)
                loss = torch.nn.functional.smooth_l1_loss(reconstructed, edges) + 0.1 * correlation_loss(reconstructed, edges) + 0.01 * psd_penalty(reconstructed[:, None])
                val_total += float(loss) * len(edges)
        epoch_loss = val_total / len(val_dataset)
        if epoch_loss < best:
            best, stale = epoch_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": best, "window_length": window_length}, checkpoint)
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    return checkpoint


def load_autoencoder(config: dict[str, Any], window_length: int, device: torch.device) -> FCAutoencoder:
    model = FCAutoencoder(4005, int(config["model"]["fc_latent_dim"]), float(config["model"]["dropout"])).to(device)
    payload = torch.load(autoencoder_checkpoint_path(config, window_length), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model


def build_sequence_model(config: dict[str, Any], window_length: int, decoder_type: str, stats_path: Path, device: torch.device):
    autoencoder = load_autoencoder(config, window_length, device)
    stats = dict(np.load(stats_path))
    model_cfg = config["model"]
    group_template = torch.from_numpy(stats["group_template"])
    if decoder_type == "direct_mlp":
        return DirectSCMLP(autoencoder, group_template, hidden=512, latent_dim=int(model_cfg["fc_latent_dim"])).to(device)
    if decoder_type == "gcn_gru":
        return GCNGRUBaseline(autoencoder, group_template, n_nodes=int(config["data"]["n_nodes"]), hidden=int(model_cfg["fc_latent_dim"])).to(device)
    return ConditionalSequenceModel(
        autoencoder,
        group_template,
        decoder_type=decoder_type,
        n_nodes=int(config["data"]["n_nodes"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        graph_layers=int(model_cfg["sc_graph_layers"]),
        graph_heads=int(model_cfg["sc_graph_heads"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        transformer_ffn_dim=int(model_cfg["transformer_ffn_dim"]),
        tcn_dilations=tuple(model_cfg["tcn_dilations"]),
        dropout=float(model_cfg["dropout"]),
        sc_mean=torch.from_numpy(stats["sc_mean"]),
        sc_std=torch.from_numpy(stats["sc_std"]),
    ).to(device)


@torch.no_grad()
def validate(model: ConditionalSequenceModel, loader: DataLoader, nonoverlap: int, device: torch.device) -> float:
    model.eval()
    scores = []
    template = model.group_template.detach().cpu().numpy()
    for batch in loader:
        output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device), batch["run"].to(device))
        pred = output.fc_z_edges.cpu().numpy()
        true = batch["fc_future"].numpy()
        scores.extend(sequence_metrics(p, t, template, nonoverlap)["long_residual_pearson"] for p, t in zip(pred, true))
    return float(np.mean(scores))


def train_sequence_model(
    config: dict[str, Any], window_length: int, decoder_type: str, stats_path: Path, ablation: str = "full", device_name: str | None = None
) -> Path:
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    train_data = DFCSequenceDataset(config, window_length, "train", stats_path, ablation)
    val_data = DFCSequenceDataset(config, window_length, "val", stats_path, ablation)
    train_loader = DataLoader(train_data, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"]["num_workers"]))
    val_loader = DataLoader(val_data, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]))
    model = build_sequence_model(config, window_length, decoder_type, stats_path, device)
    for parameter in model.fc_autoencoder.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.fc_autoencoder.decoder.parameters():
        parameter.requires_grad = False
    main_parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("fc_autoencoder.decoder")]
    optimizer = torch.optim.AdamW(main_parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    criterion = CompositeLoss(config["training"]["loss_weights"], nonoverlap, int(config["data"]["n_nodes"]))
    output_dir = resolve_path(config, "output_dir") / f"window_{window_length}" / f"{decoder_type}_{ablation}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    best, stale = -float("inf"), 0
    for epoch in range(int(config["training"]["epochs"])):
        if epoch == int(config["training"]["decoder_frozen_epochs"]):
            for parameter in model.fc_autoencoder.decoder.parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({"params": model.fc_autoencoder.decoder.parameters(), "lr": float(config["training"]["learning_rate"]) * float(config["training"]["decoder_learning_rate_scale"])})
        model.train()
        for batch in train_loader:
            output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device), batch["run"].to(device))
            target = batch["fc_future"].to(device)
            loss, _ = criterion(output.fc_z_edges, target, model.group_template)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
        score = validate(model, val_loader, nonoverlap, device)
        if score > best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "decoder_type": decoder_type, "ablation": ablation, "window_length": window_length}, checkpoint)
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    return checkpoint
