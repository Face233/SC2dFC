from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DEFAULT_SEED, resolve_path
from .connectivity import nonoverlap_horizon
from .data import DFCSequenceDataset, FCWindowDataset
from .models import ConditionalSequenceModel, FCAutoencoder
from .models.baselines import CommonInputLSTM, CommonInputMLP, DirectSCMLP, GCNGRUBaseline, PCARidgeBaseline
from .models.sequence import torch_edges_to_matrix
from .progress import append_jsonl, emit


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
    """按配置组合预测损失；默认只启用边重建和时间差分两项。"""

    _SUPPORTED = {"edge", "residual_corr", "difference", "static", "variance", "fcd", "contrastive", "psd"}

    def __init__(
        self, weights: dict[str, float], nonoverlap_start: int, n_nodes: int = 90, huber_beta: float = 1.0
    ) -> None:
        unknown = set(weights) - self._SUPPORTED
        if unknown:
            raise ValueError(f"Unknown loss components: {sorted(unknown)}")
        self.weights = {name: float(value) for name, value in weights.items() if float(value) != 0.0}
        if not self.weights:
            raise ValueError("At least one loss component must have a non-zero weight")
        if len(self.weights) > 3:
            raise ValueError("At most three loss components may be enabled")
        self.nonoverlap_start = nonoverlap_start
        self.n_nodes = n_nodes
        self.huber_beta = float(huber_beta)
        if self.huber_beta <= 0:
            raise ValueError("huber_beta must be positive")

    def _huber(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(prediction, target, beta=self.huber_beta)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, group_template: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        if "edge" in self.weights:
            components["edge"] = self._huber(prediction, target)
        if "residual_corr" in self.weights:
            template = group_template[: target.shape[1]][None]
            components["residual_corr"] = correlation_loss(
                prediction[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :],
                target[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :],
            )
        if "difference" in self.weights:
            components["difference"] = self._huber(
                prediction[:, 1:] - prediction[:, :-1],
                target[:, 1:] - target[:, :-1],
            )
        if "static" in self.weights:
            components["static"] = self._huber(prediction.mean(1), target.mean(1))
        if "variance" in self.weights:
            components["variance"] = variance_loss(prediction, target)
        if "fcd" in self.weights:
            components["fcd"] = fcd_gram_loss(prediction, target)
        if "contrastive" in self.weights:
            components["contrastive"] = contrastive_loss(prediction, target, self.nonoverlap_start)
        if "psd" in self.weights:
            components["psd"] = psd_penalty(prediction, self.n_nodes)
        total = sum((self.weights[name] * value for name, value in components.items()), prediction.new_zeros(()))
        return total, components


class AutoencoderLoss:
    """Configurable FC autoencoder reconstruction loss."""

    _SUPPORTED = {"edge", "correlation", "psd"}

    def __init__(self, weights: dict[str, float], n_nodes: int = 90) -> None:
        unknown = set(weights) - self._SUPPORTED
        if unknown:
            raise ValueError(f"Unknown autoencoder loss components: {sorted(unknown)}")
        self.weights = {name: float(value) for name, value in weights.items() if float(value) != 0.0}
        if not self.weights:
            raise ValueError("At least one autoencoder loss component must have a non-zero weight")
        self.n_nodes = n_nodes

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        if "edge" in self.weights:
            components["edge"] = F.smooth_l1_loss(prediction, target)
        if "correlation" in self.weights:
            components["correlation"] = correlation_loss(prediction, target)
        if "psd" in self.weights:
            components["psd"] = psd_penalty(prediction[:, None], self.n_nodes)
        total = sum((self.weights[name] * value for name, value in components.items()), prediction.new_zeros(()))
        return total, components


# ======================== 训练与早停 ========================


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """固定 Python、NumPy 与 PyTorch 随机源，保证实验可复现。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def device_from_arg(name: str | None = None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _synchronize(device: torch.device) -> None:
    """Synchronize asynchronous CUDA work before recording wall-clock timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autoencoder_checkpoint_path(config: dict[str, Any], window_length: int, artifact_path: str | Path | None = None) -> Path:
    if artifact_path is not None:
        return Path(artifact_path)
    return resolve_path(config, "output_dir") / f"window_{window_length}" / "fc_autoencoder.pt"


def train_autoencoder(
    config: dict[str, Any],
    window_length: int,
    stats_path: Path,
    device_name: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> Path:
    """先训练 FC 自编码器，并按验证重建损失保存最佳检查点。"""
    seed = int(config["seed"])
    seed_everything(seed)
    device = device_from_arg(device_name)
    sequence = DFCSequenceDataset(config, window_length, "train", stats_path)
    dataset = FCWindowDataset(sequence, windows_per_run=32, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["autoencoder_batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=_loader_generator(seed),
        worker_init_fn=_seed_worker,
    )
    val_sequence = DFCSequenceDataset(config, window_length, "val", stats_path)
    val_dataset = FCWindowDataset(val_sequence, windows_per_run=8, seed=seed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["autoencoder_batch_size"]),
        shuffle=False,
        num_workers=0,
        generator=_loader_generator(seed),
        worker_init_fn=_seed_worker,
    )
    n_nodes = int(config["data"]["n_nodes"])
    n_edges = n_nodes * (n_nodes - 1) // 2
    model = FCAutoencoder(n_edges, int(config["model"]["fc_latent_dim"]), float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    criterion = AutoencoderLoss(config["training"]["autoencoder_loss_weights"], n_nodes)
    best = float("inf")
    managed = output_dir is not None
    output_dir = Path(output_dir) if output_dir is not None else autoencoder_checkpoint_path(config, window_length).parent
    checkpoint_dir = output_dir / "checkpoints" if managed else output_dir
    checkpoint = checkpoint_dir / ("best.pt" if managed else "fc_autoencoder.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    best_epoch = -1
    stale = 0
    epoch_durations: list[float] = []
    max_epochs = int(config["training"]["autoencoder_epochs"])
    emit(
        "train_started", task="autoencoder", device=str(device), window_length=window_length,
        train_samples=len(dataset), validation_samples=len(val_dataset), output_dir=str(output_dir),
    )
    for epoch in range(max_epochs):
        _synchronize(device)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total = 0.0
        for edges in loader:
            edges = edges.to(device)
            reconstructed, _ = model(edges)
            loss, _ = criterion(reconstructed, edges)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            total += float(loss) * len(edges)
        _synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for edges in val_loader:
                edges = edges.to(device)
                reconstructed, _ = model(edges)
                loss, _ = criterion(reconstructed, edges)
                val_total += float(loss) * len(edges)
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_loss = val_total / len(val_dataset)
        improved = epoch_loss < best
        if epoch_loss < best:
            best, stale, best_epoch = epoch_loss, 0, epoch
            payload = {"schema_version": 1, "model": model.state_dict(), "epoch": epoch, "loss": best, "window_length": window_length}
            payload.update(checkpoint_metadata or {})
            torch.save(payload, checkpoint)
        else:
            stale += 1
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_seconds)
        mean_epoch_seconds = float(np.mean(epoch_durations))
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        append_jsonl(
            log_path, "epoch_complete", task="autoencoder", epoch=epoch + 1,
            train_loss=total / max(len(dataset), 1), validation_loss=epoch_loss,
            best_validation_loss=best, improved=improved, stale_epochs=stale,
            train_seconds=train_seconds, validation_seconds=validation_seconds,
            epoch_seconds=epoch_seconds, mean_epoch_seconds=mean_epoch_seconds,
            estimated_seconds_to_max_epochs=mean_epoch_seconds * (max_epochs - epoch - 1),
            estimated_seconds_if_no_more_improvement=mean_epoch_seconds * max(int(config["training"]["patience"]) - stale, 0),
            train_samples_per_second=len(dataset) / max(train_seconds, 1e-9),
            gpu_peak_memory_gb=peak_memory_gb,
        )
        if stale >= int(config["training"]["patience"]):
            emit("early_stopped", task="autoencoder", epoch=epoch + 1, best_epoch=best_epoch + 1, best_validation_loss=best)
            break
    (output_dir / "metrics_best.json").write_text(
        json.dumps({"metrics": {"validation_loss": best}, "best_epoch": best_epoch}, indent=2), encoding="utf-8"
    )
    emit("train_finished", task="autoencoder", best_epoch=best_epoch + 1, best_validation_loss=best, checkpoint=str(checkpoint))
    return checkpoint


def load_autoencoder(
    config: dict[str, Any], window_length: int, device: torch.device, artifact_path: str | Path | None = None
) -> FCAutoencoder:
    n_nodes = int(config["data"]["n_nodes"])
    n_edges = n_nodes * (n_nodes - 1) // 2
    model = FCAutoencoder(n_edges, int(config["model"]["fc_latent_dim"]), float(config["model"]["dropout"])).to(device)
    payload = torch.load(autoencoder_checkpoint_path(config, window_length, artifact_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model


def build_sequence_model(
    config: dict[str, Any],
    window_length: int,
    decoder_type: str,
    stats_path: Path,
    device: torch.device,
    sc_encoder_type: str | None = None,
    autoencoder_path: str | Path | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
    ablation: str | None = None,
):
    """加载共享 FC 解码器，并按名称构建主模型或学习型基线。"""
    autoencoder = load_autoencoder(config, window_length, device, autoencoder_path)
    stats = dict(np.load(stats_path))
    model_cfg = config["model"]
    sc_encoder_type = sc_encoder_type or str(model_cfg.get("sc_encoder", "hybrid"))
    ablation = ablation or (checkpoint_payload or {}).get("ablation", "full")
    output_head = str(model_cfg.get("output_head", "e0003_reconstruction_decoder"))
    if decoder_type in {"gru", "tcn", "transformer"} and output_head != "e0003_reconstruction_decoder":
        raise ValueError(f"Unsupported conditional output head: {output_head}")
    group_template = torch.from_numpy(stats["group_template"])
    if decoder_type == "pca_ridge":
        if checkpoint_payload is None:
            raise ValueError("A fitted checkpoint payload is required to build pca_ridge")
        dimensions = checkpoint_payload["ridge_dimensions"]
        return PCARidgeBaseline(
            autoencoder, group_template, int(dimensions["n_components"]),
            int(dimensions["sc_edges"]), int(dimensions["latent_dim"]),
        ).to(device)
    if decoder_type == "direct_mlp":
        return DirectSCMLP(autoencoder, group_template, hidden=512, latent_dim=int(model_cfg["fc_latent_dim"])).to(device)
    if decoder_type == "gcn_gru":
        return GCNGRUBaseline(autoencoder, group_template, n_nodes=int(config["data"]["n_nodes"]), hidden=int(model_cfg["fc_latent_dim"])).to(device)
    if decoder_type == "mlp":
        return CommonInputMLP(autoencoder, group_template, hidden=int(model_cfg.get("baseline_hidden_dim", 512))).to(device)
    if decoder_type == "lstm":
        return CommonInputLSTM(autoencoder, group_template, hidden=int(model_cfg.get("baseline_hidden_dim", 256))).to(device)
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
        gru_layers=int(model_cfg.get("gru_layers", 2)),
        tcn_dilations=tuple(model_cfg["tcn_dilations"]),
        dropout=float(model_cfg["dropout"]),
        sc_mean=torch.from_numpy(stats["sc_mean"]),
        sc_std=torch.from_numpy(stats["sc_std"]),
        sc_encoder_type=sc_encoder_type,
        hcp_gcn_hidden_dim=int(model_cfg.get("hcp_gcn_hidden_dim", 128)),
        hcp_gcn_output_dim=int(model_cfg.get("hcp_gcn_output_dim", 64)),
        ablation=ablation,
    ).to(device)


def train_pca_ridge_baseline(
    config: dict[str, Any], window_length: int, stats_path: Path, output_dir: Path,
    checkpoint_metadata: dict[str, Any], autoencoder_path: str | Path, device: torch.device,
) -> Path:
    """Fit a training-only PCA + multi-output Ridge model in the shared FC latent space."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    train_data = DFCSequenceDataset(config, window_length, "train", stats_path)
    val_data = DFCSequenceDataset(config, window_length, "val", stats_path)
    autoencoder = load_autoencoder(config, window_length, device, autoencoder_path)
    autoencoder.eval()
    stats = dict(np.load(stats_path))
    template = torch.from_numpy(stats["group_template"]).to(device)
    sc_values, warmups, targets = [], [], []
    with torch.no_grad():
        for index in range(len(train_data)):
            sample = train_data[index]
            sc_values.append(sample["sc_edges"].numpy())
            warmups.append(autoencoder.encode(sample["fc_warmup"].to(device)[None]).cpu().numpy()[0])
            residual = sample["fc_future"].to(device) - template[: len(sample["fc_future"])]
            targets.append(autoencoder.encode(residual).cpu().numpy().reshape(-1))
    sc_array = np.stack(sc_values)
    n_components = min(int(config["model"].get("ridge_pca_components", 128)), len(sc_array), sc_array.shape[1])
    pca = PCA(n_components=n_components, random_state=int(config["seed"]))
    projected = pca.fit_transform(sc_array)
    features = np.concatenate([projected, np.stack(warmups)], axis=1)
    ridge = Ridge(alpha=float(config["model"].get("ridge_alpha", 1.0)))
    ridge.fit(features, np.stack(targets))
    model = PCARidgeBaseline(autoencoder, template.cpu(), n_components, sc_array.shape[1], len(warmups[0])).to(device)
    model.pca_mean.copy_(torch.from_numpy(pca.mean_).to(device, dtype=torch.float32))
    model.pca_components.copy_(torch.from_numpy(pca.components_).to(device, dtype=torch.float32))
    model.ridge_coef.copy_(torch.from_numpy(ridge.coef_).to(device, dtype=torch.float32))
    model.ridge_intercept.copy_(torch.from_numpy(ridge.intercept_).to(device, dtype=torch.float32))
    loader = DataLoader(val_data, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    score = validate(model, loader, nonoverlap_horizon(window_length, int(config["data"]["stride"])), device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best.pt"
    payload = {
        "schema_version": 1, "model": model.state_dict(), "epoch": 0, "score": score,
        "primary_metric": config["evaluation"]["primary_metric"], "decoder_type": "pca_ridge",
        "sc_encoder_type": "hybrid", "ablation": "full", "window_length": window_length,
        "ridge_dimensions": {"n_components": n_components, "sc_edges": sc_array.shape[1], "latent_dim": len(warmups[0])},
        **checkpoint_metadata,
    }
    torch.save(payload, checkpoint)
    (output_dir / "train.log").write_text(json.dumps({"fit": "pca_ridge", "validation_long_residual_pearson": score}) + "\n", encoding="utf-8")
    (output_dir / "metrics_best.json").write_text(
        json.dumps({"metrics": {config["evaluation"]["primary_metric"]: score}, "best_epoch": 0}, indent=2), encoding="utf-8"
    )
    return checkpoint


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
        output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device))
        scores.extend(_long_residual_score(output.fc_z_edges, batch["fc_future"].to(device), model.group_template, nonoverlap).cpu().tolist())
    return float(np.mean(scores))


@torch.no_grad()
def validate_sequence(
    model: ConditionalSequenceModel,
    loader: DataLoader,
    criterion: CompositeLoss,
    nonoverlap: int,
    device: torch.device,
) -> dict[str, float]:
    """一次验证前向同时计算组合 Huber 目标、各分量与长时距诊断指标。"""
    model.eval()
    total = 0.0
    component_totals: dict[str, float] = {name: 0.0 for name in criterion.weights}
    count = 0
    residual_scores = []
    for batch in loader:
        output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device))
        target = batch["fc_future"].to(device)
        loss, components = criterion(output.fc_z_edges, target, model.group_template)
        batch_size = len(target)
        total += float(loss) * batch_size
        for name, value in components.items():
            component_totals[name] += float(value) * batch_size
        count += batch_size
        residual_scores.extend(_long_residual_score(output.fc_z_edges, target, model.group_template, nonoverlap).cpu().tolist())
    metrics = {
        "objective_loss": total / max(count, 1),
        "long_residual_pearson": float(np.mean(residual_scores)),
    }
    metrics.update({f"validation_{name}_loss": value / max(count, 1) for name, value in component_totals.items()})
    return metrics


def train_sequence_model(
    config: dict[str, Any],
    window_length: int,
    decoder_type: str,
    stats_path: Path,
    ablation: str = "full",
    device_name: str | None = None,
    sc_encoder_type: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    autoencoder_path: str | Path | None = None,
) -> Path:
    """训练条件序列模型或学习型基线，并按声明的主验证指标早停。"""
    seed = int(config["seed"])
    seed_everything(seed)
    device = device_from_arg(device_name)
    if decoder_type == "pca_ridge":
        if output_dir is None or checkpoint_metadata is None or autoencoder_path is None:
            raise ValueError("pca_ridge is available only through the managed experiment runner")
        return train_pca_ridge_baseline(
            config, window_length, stats_path, Path(output_dir), checkpoint_metadata, autoencoder_path, device
        )
    train_data = DFCSequenceDataset(config, window_length, "train", stats_path, ablation)
    val_data = DFCSequenceDataset(config, window_length, "val", stats_path, ablation)
    num_workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        generator=_loader_generator(seed),
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        generator=_loader_generator(seed),
        worker_init_fn=_seed_worker,
    )
    requested_sc_encoder_type = sc_encoder_type
    sc_encoder_type = requested_sc_encoder_type or str(config["model"].get("sc_encoder", "hybrid"))
    baseline_types = {"direct_mlp", "gcn_gru", "mlp", "lstm"}
    if decoder_type in baseline_types and requested_sc_encoder_type not in {None, "hybrid"}:
        raise ValueError("--sc-encoder applies only to the gru, tcn, and transformer conditional models")
    if decoder_type in baseline_types:
        sc_encoder_type = "hybrid"
    model = build_sequence_model(
        config, window_length, decoder_type, stats_path, device, sc_encoder_type, autoencoder_path, ablation=ablation
    )
    for parameter in model.fc_autoencoder.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.fc_autoencoder.decoder.parameters():
        parameter.requires_grad = False
    main_parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("fc_autoencoder.decoder")]
    optimizer = torch.optim.AdamW(main_parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    nonoverlap = nonoverlap_horizon(window_length, int(config["data"]["stride"]))
    criterion = CompositeLoss(
        config["training"]["loss_weights"], nonoverlap, int(config["data"]["n_nodes"]),
        float(config["training"].get("huber_beta", 1.0)),
    )
    conditional_name = (
        decoder_type
        if sc_encoder_type == "hybrid" or decoder_type in baseline_types
        else f"{decoder_type}_{sc_encoder_type}"
    )
    managed = output_dir is not None
    output_dir = Path(output_dir) if output_dir is not None else resolve_path(config, "output_dir") / f"window_{window_length}" / f"{conditional_name}_{ablation}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints" if managed else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    log_path = output_dir / "train.log"
    primary_metric = str(config.get("evaluation", {}).get("primary_metric", "long_residual_pearson"))
    if primary_metric not in {"objective_loss", "long_residual_pearson"}:
        raise ValueError("Sequence primary_metric must be objective_loss or long_residual_pearson")
    minimize = primary_metric == "objective_loss"
    finetune_fc_decoder = bool(config["training"].get("finetune_fc_decoder", False))
    best_epoch = -1
    best, stale = (float("inf") if minimize else -float("inf")), 0
    best_validation_metrics: dict[str, float] = {}
    epoch_durations: list[float] = []
    max_epochs = int(config["training"]["epochs"])
    emit(
        "train_started", task="sequence", model=decoder_type, ablation=ablation,
        sc_encoder=sc_encoder_type, device=str(device), window_length=window_length,
        train_samples=len(train_data), validation_samples=len(val_data), output_dir=str(output_dir),
    )
    for epoch in range(max_epochs):
        _synchronize(device)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if finetune_fc_decoder and epoch == int(config["training"]["decoder_frozen_epochs"]):
            for parameter in model.fc_autoencoder.decoder.parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({"params": model.fc_autoencoder.decoder.parameters(), "lr": float(config["training"]["learning_rate"]) * float(config["training"]["decoder_learning_rate_scale"])})
        model.train()
        # 冻结权重还不够；必须同时关闭 E0003 encoder/decoder 内的 Dropout。
        model.fc_autoencoder.encoder.eval()
        decoder_trainable = finetune_fc_decoder and epoch >= int(config["training"]["decoder_frozen_epochs"])
        if not decoder_trainable:
            model.fc_autoencoder.decoder.eval()
        train_total = 0.0
        train_components = {name: 0.0 for name in criterion.weights}
        train_count = 0
        for batch in train_loader:
            output = model(batch["sc_matrix"].to(device), batch["sc_edges"].to(device), batch["fc_warmup"].to(device))
            target = batch["fc_future"].to(device)
            loss, components = criterion(output.fc_z_edges, target, model.group_template)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            train_total += float(loss) * len(target)
            for name, value in components.items():
                train_components[name] += float(value) * len(target)
            train_count += len(target)
        _synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation_metrics = validate_sequence(model, val_loader, criterion, nonoverlap, device)
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        score = validation_metrics[primary_metric]
        improved = score < best if minimize else score > best
        if improved:
            best, stale, best_epoch = score, 0, epoch
            best_validation_metrics = validation_metrics
            payload = {
                "schema_version": 1, "model": model.state_dict(), "epoch": epoch, "score": score,
                "primary_metric": primary_metric, "decoder_type": decoder_type,
                "sc_encoder_type": sc_encoder_type, "ablation": ablation, "window_length": window_length,
                "validation_metrics": validation_metrics, "output_head": "e0003_reconstruction_decoder",
                "fc_reconstruction_decoder_frozen": not finetune_fc_decoder,
            }
            payload.update(checkpoint_metadata or {})
            torch.save(payload, checkpoint)
        else:
            stale += 1
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_seconds)
        mean_epoch_seconds = float(np.mean(epoch_durations))
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        train_metrics = {f"train_{name}_loss": value / max(train_count, 1) for name, value in train_components.items()}
        append_jsonl(
            log_path, "epoch_complete", task="sequence", model=decoder_type, epoch=epoch + 1,
            train_loss=train_total / max(train_count, 1), **train_metrics, **validation_metrics,
            primary_metric=primary_metric, primary_value=score, best_primary_value=best,
            improved=improved, stale_epochs=stale,
            train_seconds=train_seconds, validation_seconds=validation_seconds,
            epoch_seconds=epoch_seconds, mean_epoch_seconds=mean_epoch_seconds,
            estimated_seconds_to_max_epochs=mean_epoch_seconds * (max_epochs - epoch - 1),
            estimated_seconds_if_no_more_improvement=mean_epoch_seconds * max(int(config["training"]["patience"]) - stale, 0),
            train_samples_per_second=train_count / max(train_seconds, 1e-9),
            gpu_peak_memory_gb=peak_memory_gb,
        )
        if stale >= int(config["training"]["patience"]):
            emit("early_stopped", task="sequence", model=decoder_type, epoch=epoch + 1, best_epoch=best_epoch + 1, primary_metric=primary_metric, best_primary_value=best)
            break
        if int(config.get("experiment", {}).get("level", 0)) >= 2:
            last_payload = {
                "schema_version": 1, "model": model.state_dict(), "epoch": epoch, "score": score,
                "primary_metric": primary_metric, "decoder_type": decoder_type,
                "sc_encoder_type": sc_encoder_type, "ablation": ablation, "window_length": window_length,
                "validation_metrics": validation_metrics, "output_head": "e0003_reconstruction_decoder",
                "fc_reconstruction_decoder_frozen": not finetune_fc_decoder,
            }
            last_payload.update(checkpoint_metadata or {})
            torch.save(last_payload, last_checkpoint)
    (output_dir / "metrics_best.json").write_text(
        json.dumps({"metrics": best_validation_metrics, "primary_metric": primary_metric, "best_epoch": best_epoch}, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics_last.json").write_text(
        json.dumps({"metrics": validation_metrics, "primary_metric": primary_metric, "last_epoch": epoch}, indent=2), encoding="utf-8"
    )
    emit("train_finished", task="sequence", model=decoder_type, best_epoch=best_epoch + 1, primary_metric=primary_metric, best_primary_value=best, checkpoint=str(checkpoint))
    return checkpoint
