from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .autoencoder import FCAutoencoder
from .sc_encoders import HCPGCNEncoder


@dataclass(frozen=True)
class Prediction:
    """模型对一个批次未来 dFC 序列的数值输出。"""

    fc_z_edges: torch.Tensor
    fc_matrices: torch.Tensor
    latent: torch.Tensor | None = None


# ======================== SC 与首窗 FC 的条件编码 ========================
class BiasedGraphAttention(nn.Module):
    """将 SC 边权作为注意力偏置的单层图注意力模块。"""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        # 每个注意力头学习结构连接强度应占多大权重。
        self.bias_scale = nn.Parameter(torch.ones(heads))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm1(tokens)
        batch, nodes, dim = x.shape
        qkv = self.qkv(x).view(batch, nodes, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.head_dim)
        # 仅正值 SC 进入偏置；原始图本身仍保留在其他编码路径中。
        edge_bias = torch.log1p(torch.clamp(adjacency, min=0))
        scores = scores + self.bias_scale[None, :, None, None] * edge_bias[:, None]
        attention = self.dropout(scores.softmax(dim=-1))
        mixed = torch.einsum("bhnm,bmhd->bnhd", attention, v).reshape(batch, nodes, dim)
        tokens = residual + self.output(mixed)
        return tokens + self.ffn(self.norm2(tokens))


class SCGraphEncoder(nn.Module):
    """从节点强度、节点度、ROI 身份和 SC 图拓扑生成节点/全局表示。"""

    def __init__(self, n_nodes: int = 90, dim: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.roi_embedding = nn.Embedding(n_nodes, dim)
        self.features = nn.Linear(2, dim)
        self.layers = nn.ModuleList([BiasedGraphAttention(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strength = adjacency.sum(dim=-1)
        degree = (adjacency > 0).float().sum(dim=-1) / adjacency.shape[-1]
        features = torch.stack([torch.log1p(strength), degree], dim=-1)
        roi = torch.arange(adjacency.shape[-1], device=adjacency.device)
        tokens = self.features(features) + self.roi_embedding(roi)[None]
        for layer in self.layers:
            tokens = layer(tokens, adjacency)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1), tokens


class ConditionEncoder(nn.Module):
    """将 SC 与首窗 FC 分别编码为 256 维，再融合成共享全局条件。"""

    def __init__(
        self,
        fc_autoencoder: FCAutoencoder,
        n_nodes: int = 90,
        n_edges: int = 4005,
        hidden_dim: int = 256,
        graph_layers: int = 3,
        graph_heads: int = 4,
        dropout: float = 0.1,
        sc_encoder_type: str = "hybrid",
        hcp_gcn_hidden_dim: int = 128,
        hcp_gcn_output_dim: int = 64,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        if sc_encoder_type not in {"hybrid", "hcp_gcn"}:
            raise ValueError("sc_encoder_type must be 'hybrid' or 'hcp_gcn'")
        if ablation not in {"full", "fc1_only", "sc_only", "mean_sc", "shuffled_sc"}:
            raise ValueError(f"Unknown ablation: {ablation}")
        self.fc_autoencoder = fc_autoencoder
        self.sc_encoder_type = sc_encoder_type
        self.ablation = ablation
        if sc_encoder_type == "hybrid":
            self.graph = SCGraphEncoder(n_nodes, 128, graph_layers, graph_heads, dropout)
            self.edge_mlp = nn.Sequential(nn.Linear(n_edges, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 128))
        else:
            self.hcp_gcn = HCPGCNEncoder(n_nodes, hcp_gcn_hidden_dim, hcp_gcn_output_dim)
            self.hcp_global_projection = nn.Linear(hcp_gcn_output_dim, 256)
        self.sc_norm = nn.LayerNorm(256)
        fc_dim = fc_autoencoder.encoder[-1].normalized_shape[0]
        combined = 256 + fc_dim
        # 门控融合确保模型可按被试调整各类条件信息的贡献。
        self.value = nn.Linear(combined, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(combined, hidden_dim), nn.Sigmoid())
        self.condition_norm = nn.LayerNorm(hidden_dim)

    def encode_modalities(
        self, sc_matrix: torch.Tensor, sc_edges: torch.Tensor, fc_warmup: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sc_encoder_type == "hybrid":
            graph_global, _ = self.graph(sc_matrix)
            edge_global = self.edge_mlp(sc_edges)
            sc_global = torch.cat([graph_global, edge_global], dim=-1)
        else:
            hcp_global, _ = self.hcp_gcn(sc_matrix)
            sc_global = self.hcp_global_projection(hcp_global)
        warmup = self.fc_autoencoder.encode(fc_warmup)
        return self.sc_norm(sc_global), warmup

    def forward(
        self, sc_matrix: torch.Tensor, sc_edges: torch.Tensor, fc_warmup: torch.Tensor
    ) -> torch.Tensor:
        sc_global, warmup = self.encode_modalities(sc_matrix, sc_edges, fc_warmup)
        # 信息消融必须发生在编码后，避免零原始输入通过 bias/ROI embedding 产生伪条件。
        if self.ablation == "fc1_only":
            sc_global = torch.zeros_like(sc_global)
        elif self.ablation == "sc_only":
            warmup = torch.zeros_like(warmup)
        combined = torch.cat([sc_global, warmup], dim=-1)
        condition = self.value(combined) * self.gate(combined)
        return self.condition_norm(condition)


# ======================== 未来潜轨迹解码 ========================
class FiLMTCNBlock(nn.Module):
    """用条件向量调制的膨胀 TCN 残差块。"""

    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, 2 * dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(dim, 2 * dim)
        self.output = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        gamma_beta = self.condition(condition)[:, None]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        content, gate = y.chunk(2, dim=-1)
        y = (content * (1 + gamma) + beta) * torch.sigmoid(gate)
        y = self.output(y.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(y)


class TCNDecoder(nn.Module):
    """一次性预测全部未来窗口的非自回归 TCN 解码器。"""

    def __init__(self, dim: int = 256, max_steps: int = 256, dilations=(1, 2, 4, 8, 16, 32), dropout: float = 0.1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        self.blocks = nn.ModuleList([FiLMTCNBlock(dim, dilation, dropout) for dilation in dilations])
        self.norm = nn.LayerNorm(dim)

    def forward(self, condition: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        x = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        for block in self.blocks:
            x = block(x, condition)
        return self.norm(x)


class TransformerTrajectoryDecoder(nn.Module):
    """在统一全局条件下，用时间 self-attention 预测完整未来潜轨迹。"""

    def __init__(
        self, dim: int = 256, max_steps: int = 256, layers: int = 4, heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim))

    def forward(self, condition: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        query = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        return self.encoder(query)


class GRUTrajectoryDecoder(nn.Module):
    """在与 Transformer 相同的全局条件下预测完整未来潜轨迹。"""

    def __init__(self, dim: int = 256, max_steps: int = 256, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GRU layers must be positive")
        self.layers = layers
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        self.initial = nn.Linear(dim, layers * dim)
        self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, condition: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        sequence = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        initial = self.initial(condition).view(condition.shape[0], self.layers, -1).transpose(0, 1).contiguous()
        output, _ = self.gru(sequence, initial)
        return self.norm(output)


def torch_edges_to_matrix(edges: torch.Tensor, n_nodes: int = 90) -> torch.Tensor:
    """把无对角线的上三角边向量恢复成对称相关矩阵。"""
    expected = n_nodes * (n_nodes - 1) // 2
    if edges.shape[-1] != expected:
        raise ValueError(f"Expected {expected} edges")
    result = edges.new_zeros(*edges.shape[:-1], n_nodes, n_nodes)
    indices = torch.triu_indices(n_nodes, n_nodes, offset=1, device=edges.device)
    result[..., indices[0], indices[1]] = edges
    result[..., indices[1], indices[0]] = edges
    diagonal = torch.arange(n_nodes, device=edges.device)
    result[..., diagonal, diagonal] = 1
    return result


class ConditionalSequenceModel(nn.Module):
    """SC + 首窗 FC 条件下预测未来 dFC 的完整主模型。"""
    def __init__(
        self,
        fc_autoencoder: FCAutoencoder,
        group_template: torch.Tensor,
        decoder_type: str = "tcn",
        n_nodes: int = 90,
        hidden_dim: int = 256,
        graph_layers: int = 3,
        graph_heads: int = 4,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 1024,
        gru_layers: int = 2,
        tcn_dilations=(1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        sc_mean: torch.Tensor | None = None,
        sc_std: torch.Tensor | None = None,
        sc_encoder_type: str = "hybrid",
        hcp_gcn_hidden_dim: int = 128,
        hcp_gcn_output_dim: int = 64,
        ablation: str = "full",
        output_head: str = "e0003_reconstruction_decoder",
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_nodes * (n_nodes - 1) // 2
        fc_latent_dim = int(fc_autoencoder.decoder[0].in_features)
        if hidden_dim != fc_latent_dim:
            raise ValueError(
                "hidden_dim must equal the frozen FC autoencoder latent dimension "
                f"({fc_latent_dim}), got {hidden_dim}"
            )
        self.fc_autoencoder = fc_autoencoder
        if output_head not in {"e0003_reconstruction_decoder", "direct_edge_linear"}:
            raise ValueError(f"Unsupported output_head: {output_head}")
        self.output_head = output_head
        self.condition_encoder = ConditionEncoder(
            fc_autoencoder,
            n_nodes,
            self.n_edges,
            hidden_dim,
            graph_layers,
            graph_heads,
            dropout,
            sc_encoder_type,
            hcp_gcn_hidden_dim,
            hcp_gcn_output_dim,
            ablation,
        )
        if decoder_type == "tcn":
            self.temporal = TCNDecoder(hidden_dim, 256, tcn_dilations, dropout)
        elif decoder_type == "transformer":
            self.temporal = TransformerTrajectoryDecoder(
                hidden_dim, 256, transformer_layers, transformer_heads, transformer_ffn_dim, dropout
            )
        elif decoder_type == "gru":
            self.temporal = GRUTrajectoryDecoder(hidden_dim, 256, gru_layers, dropout)
        else:
            raise ValueError("decoder_type must be 'gru', 'tcn', or 'transformer'")
        self.direct_edge_head = nn.Linear(hidden_dim, self.n_edges) if output_head == "direct_edge_linear" else None
        # 模板与 SC 标准化参数随检查点保存，但不参与梯度更新。
        self.register_buffer("group_template", group_template.float())
        self.register_buffer("sc_mean", torch.zeros(self.n_edges) if sc_mean is None else sc_mean.float())
        self.register_buffer("sc_std", torch.ones(self.n_edges) if sc_std is None else sc_std.float())

    def forward(
        self,
        sc_matrix: torch.Tensor,
        sc_edges: torch.Tensor,
        fc_warmup: torch.Tensor,
        steps: int | None = None,
    ) -> Prediction:
        steps = steps or self.group_template.shape[0]
        condition = self.condition_encoder(sc_matrix, sc_edges, fc_warmup)
        latent = self.temporal(condition, steps)
        decoded = self.fc_autoencoder.decode(latent) if self.direct_edge_head is None else self.direct_edge_head(latent)
        # E0004--E0007 只允许冻结的 E0003 reconstruction decoder 映射回 FC；
        # 不使用额外的 4005 维旁路，后续 direct edge head 才是独立 decoder 对照。
        fc_z = decoded
        matrices = torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes)
        return Prediction(fc_z_edges=fc_z, fc_matrices=matrices, latent=latent)

    @torch.no_grad()
    def predict(self, sc: torch.Tensor, fc_warmup: torch.Tensor, sc_edges: torch.Tensor | None = None) -> torch.Tensor:
        """公开推理接口；未传入边向量时在内部完成 SC 上三角标准化。"""
        if sc_edges is None:
            idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1, device=sc.device)
            sc_edges = torch.log1p(sc[..., idx[0], idx[1]])
            sc_edges = (sc_edges - self.sc_mean) / self.sc_std.clamp_min(1e-6)
        return self(sc, sc_edges, fc_warmup).fc_matrices
