from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import resolve_path
from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset, FCWindowDataset
from .models import ConditionalSequenceModel, FCAutoencoder
from .models.baselines import DirectSCMLP, GCNGRUBaseline
from .models.sequence import torch_edges_to_matrix


# ======================== 训练损失函数 ========================
def correlation_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算每个样本/时间窗内 FC 边模式的 Pearson 相关损失。"""
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (prediction * target).sum(dim=-1)
    denominator = prediction.square().sum(dim=-1).sqrt() * target.square().sum(dim=-1).sqrt()
    return (1 - numerator / denominator.clamp_min(eps)).mean()


def variance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """约束预测序列的逐边时间方差不坍缩。"""
    return F.smooth_l1_loss(prediction.var(dim=1, unbiased=False), target.var(dim=1, unbiased=False))


def fcd_gram_loss(prediction: torch.Tensor, target: torch.Tensor, max_windows: int = 32) -> torch.Tensor:
    """用抽样窗口间 FC 相似度矩阵近似 FCD 损失，控制显存开销。"""
    steps = prediction.shape[1]
    if steps > max_windows:
        indices = torch.linspace(0, steps - 1, max_windows, device=prediction.device).long()
        prediction, target = prediction[:, indices], target[:, indices]
    prediction = F.normalize(prediction - prediction.mean(-1, keepdim=True), dim=-1)
    target = F.normalize(target - target.mean(-1, keepdim=True), dim=-1)
    return F.smooth_l1_loss(prediction @ prediction.transpose(1, 2), target @ target.transpose(1, 2))


def contrastive_loss(prediction: torch.Tensor, target: torch.Tensor, start: int, temperature: float = 0.1) -> torch.Tensor:
    """鼓励预测的长时距个体表征与本人真实未来序列匹配。"""
    pred_embed = F.normalize(prediction[:, start:].mean(1), dim=-1)
    true_embed = F.normalize(target[:, start:].mean(1), dim=-1)
    logits = pred_embed @ true_embed.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def psd_penalty(z_edges: torch.Tensor, n_nodes: int = 90, max_windows: int = 4) -> torch.Tensor:
    """抽样检查预测相关矩阵的负特征值，并对其施加软惩罚。"""
    steps = z_edges.shape[1]
    indices = torch.linspace(0, steps - 1, min(max_windows, steps), device=z_edges.device).long()
    eigenvalues = torch.linalg.eigvalsh(torch_edges_to_matrix(torch.tanh(z_edges[:, indices]), n_nodes))
    return torch.relu(-eigenvalues).square().mean()


class CompositeLoss:
    """将边重建、个体化、动态性和 PSD 约束按配置权重组合。"""

    def __init__(self, weights: dict[str, float], nonoverlap_start: int, n_nodes: int = 90) -> None:
        self.weights = weights
        self.nonoverlap_start = nonoverlap_start
        self.n_nodes = n_nodes

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, group_template: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        template = group_template[: target.shape[1]][None]
        components = {
            "edge": F.smooth_l1_loss(prediction, target),
            "residual_corr": correlation_loss(prediction[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :], target[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :]),
            "difference": F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1]),
            "static": F.smooth_l1_loss(prediction.mean(1), target.mean(1)),
            "variance": variance_loss(prediction, target),
            "fcd": fcd_gram_loss(prediction, target),
            "contrastive": contrastive_loss(prediction, target, self.nonoverlap_start),
            "psd": psd_penalty(prediction, self.n_nodes),
        }
        return sum(self.weights[name] * value for name, value in components.items()), components


# ======================== 训练与早停 ========================


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机源，保证实验可复现。"""
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
    """先训练 FC 自编码器，并按验证重建损失保存最佳检查点。"""
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


def build_sequence_model(
    config: dict[str, Any],
    window_length: int,
    decoder_type: str,
    stats_path: Path,
    device: torch.device,
    sc_encoder_type: str | None = None,
):
    """加载共享 FC 解码器，并按名称构建主模型或学习型基线。"""
    autoencoder = load_autoencoder(config, window_length, device)
    stats = dict(np.load(stats_path))
    model_cfg = config["model"]
    sc_encoder_type = sc_encoder_type or str(model_cfg.get("sc_encoder", "hybrid"))
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
        sc_encoder_type=sc_encoder_type,
        hcp_gcn_hidden_dim=int(model_cfg.get("hcp_gcn_hidden_dim", 128)),
        hcp_gcn_output_dim=int(model_cfg.get("hcp_gcn_output_dim", 64)),
    ).to(device)


@torch.no_grad()
def _long_residual_score(prediction: torch.Tensor, target: torch.Tensor, template: torch.Tensor, nonoverlap: int) -> torch.Tensor:
    """早停专用：在无窗口重叠区间计算去群体模板后的边相关。"""
    pred = prediction[:, nonoverlap:] - template[None, nonoverlap:]
    true = target[:, nonoverlap:] - template[None, nonoverlap:]
    pred = pred - pred.mean(dim=-1, keepdim=True)
    true = true - true.mean(dim=-1, keepdim=True)
    corr = (pred * true).sum(-1) / (pred.square().sum(-1).sqrt() * true.square().sum(-1).sqrt()).clamp_min(1e-6)
    return corr.mean(dim=-1)


@torch.no_grad()
def validate(model: ConditionalSequenceModel, loader: DataLoader, nonoverlap: int, device: torch.device) -> float:
    """返回验证集主指标均值，仅用于选择最佳训练 epoch。"""
    model.eval()
    scores = []
    for batch in loader:
        output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device), batch["run"].to(device))
        scores.extend(_long_residual_score(output.fc_z_edges, batch["fc_future"].to(device), model.group_template, nonoverlap).cpu().tolist())
    return float(np.mean(scores))


def train_sequence_model(
    config: dict[str, Any],
    window_length: int,
    decoder_type: str,
    stats_path: Path,
    ablation: str = "full",
    device_name: str | None = None,
    sc_encoder_type: str | None = None,
) -> Path:
    """训练 TCN、Transformer 或学习型基线，并按主验证指标早停。"""
    seed_everything(int(config["seed"]))
    device = device_from_arg(device_name)
    train_data = DFCSequenceDataset(config, window_length, "train", stats_path, ablation)
    val_data = DFCSequenceDataset(config, window_length, "val", stats_path, ablation)
    train_loader = DataLoader(train_data, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"]["num_workers"]))
    val_loader = DataLoader(val_data, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]))
    requested_sc_encoder_type = sc_encoder_type
    sc_encoder_type = requested_sc_encoder_type or str(config["model"].get("sc_encoder", "hybrid"))
    if decoder_type in {"direct_mlp", "gcn_gru"} and requested_sc_encoder_type not in {None, "hybrid"}:
        raise ValueError("--sc-encoder applies only to the tcn and transformer conditional models")
    if decoder_type in {"direct_mlp", "gcn_gru"}:
        sc_encoder_type = "hybrid"
    model = build_sequence_model(config, window_length, decoder_type, stats_path, device, sc_encoder_type)
    for parameter in model.fc_autoencoder.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.fc_autoencoder.decoder.parameters():
        parameter.requires_grad = False
    main_parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("fc_autoencoder.decoder")]
    optimizer = torch.optim.AdamW(main_parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    criterion = CompositeLoss(config["training"]["loss_weights"], nonoverlap, int(config["data"]["n_nodes"]))
    conditional_name = (
        decoder_type
        if sc_encoder_type == "hybrid" or decoder_type in {"direct_mlp", "gcn_gru"}
        else f"{decoder_type}_{sc_encoder_type}"
    )
    output_dir = resolve_path(config, "output_dir") / f"window_{window_length}" / f"{conditional_name}_{ablation}"
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "score": score,
                    "decoder_type": decoder_type,
                    "sc_encoder_type": sc_encoder_type,
                    "ablation": ablation,
                    "window_length": window_length,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    return checkpoint
