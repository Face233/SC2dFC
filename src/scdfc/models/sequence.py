from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .autoencoder import FCAutoencoder


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
    """融合 SC 图、SC 上三角边、首窗 FC 和 LR/RL run 条件。"""

    def __init__(
        self,
        fc_autoencoder: FCAutoencoder,
        n_nodes: int = 90,
        n_edges: int = 4005,
        hidden_dim: int = 256,
        graph_layers: int = 3,
        graph_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc_autoencoder = fc_autoencoder
        self.graph = SCGraphEncoder(n_nodes, 128, graph_layers, graph_heads, dropout)
        self.edge_mlp = nn.Sequential(nn.Linear(n_edges, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 128))
        self.run_embedding = nn.Embedding(2, 32)
        fc_dim = fc_autoencoder.encoder[-1].normalized_shape[0]
        combined = 128 + 128 + fc_dim + 32
        # 门控融合确保模型可按被试调整各类条件信息的贡献。
        self.value = nn.Linear(combined, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(combined, hidden_dim), nn.Sigmoid())
        self.graph_token_projection = nn.Linear(128, hidden_dim)
        self.edge_token_projection = nn.Linear(128, hidden_dim)
        self.fc_token_projection = nn.Linear(fc_dim, hidden_dim)

    def forward(
        self, sc_matrix: torch.Tensor, sc_edges: torch.Tensor, fc_warmup: torch.Tensor, run: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_global, graph_tokens = self.graph(sc_matrix)
        edge_global = self.edge_mlp(sc_edges)
        warmup = self.fc_autoencoder.encode(fc_warmup)
        combined = torch.cat([graph_global, edge_global, warmup, self.run_embedding(run)], dim=-1)
        condition = self.value(combined) * self.gate(combined)
        # Transformer 使用所有 token；TCN 虽不读取 memory，也保留统一调用接口。
        memory = torch.cat(
            [
                self.graph_token_projection(graph_tokens),
                self.edge_token_projection(edge_global)[:, None],
                self.fc_token_projection(warmup)[:, None],
                condition[:, None],
            ],
            dim=1,
        )
        return condition, memory


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

    def forward(self, condition: torch.Tensor, memory: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        x = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        for block in self.blocks:
            x = block(x, condition)
        return self.norm(x)


class TransformerTrajectoryDecoder(nn.Module):
    """让每个未来时距 query 交叉注意 SC/首窗 FC 条件 token 的解码器。"""

    def __init__(
        self, dim: int = 256, max_steps: int = 256, layers: int = 4, heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers, norm=nn.LayerNorm(dim))

    def forward(self, condition: torch.Tensor, memory: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        query = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        return self.decoder(query, memory)


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
        tcn_dilations=(1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        sc_mean: torch.Tensor | None = None,
        sc_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_nodes * (n_nodes - 1) // 2
        self.fc_autoencoder = fc_autoencoder
        self.condition_encoder = ConditionEncoder(
            fc_autoencoder, n_nodes, self.n_edges, hidden_dim, graph_layers, graph_heads, dropout
        )
        if decoder_type == "tcn":
            self.temporal = TCNDecoder(hidden_dim, 256, tcn_dilations, dropout)
        elif decoder_type == "transformer":
            self.temporal = TransformerTrajectoryDecoder(
                hidden_dim, 256, transformer_layers, transformer_heads, transformer_ffn_dim, dropout
            )
        else:
            raise ValueError("decoder_type must be 'tcn' or 'transformer'")
        self.static_head = nn.Linear(hidden_dim, self.n_edges)
        # 模板与 SC 标准化参数随检查点保存，但不参与梯度更新。
        self.register_buffer("group_template", group_template.float())
        self.register_buffer("sc_mean", torch.zeros(self.n_edges) if sc_mean is None else sc_mean.float())
        self.register_buffer("sc_std", torch.ones(self.n_edges) if sc_std is None else sc_std.float())

    def forward(
        self,
        sc_matrix: torch.Tensor,
        sc_edges: torch.Tensor,
        fc_warmup: torch.Tensor,
        run: torch.Tensor,
        steps: int | None = None,
    ) -> Prediction:
        steps = steps or self.group_template.shape[0]
        condition, memory = self.condition_encoder(sc_matrix, sc_edges, fc_warmup, run)
        latent = self.temporal(condition, memory, steps)
        decoded = self.fc_autoencoder.decode(latent)
        # 解码后的时间均值移除，确保这一项只表达动态残差。
        dynamic = decoded - decoded.mean(dim=1, keepdim=True)
        static = self.static_head(condition)[:, None]
        template = self.group_template[:steps][None]
        fc_z = template + static + dynamic
        matrices = torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes)
        return Prediction(fc_z_edges=fc_z, fc_matrices=matrices, latent=latent)

    @torch.no_grad()
    def predict(self, sc: torch.Tensor, fc_warmup: torch.Tensor, run: torch.Tensor, sc_edges: torch.Tensor | None = None) -> torch.Tensor:
        """公开推理接口；未传入边向量时在内部完成 SC 上三角标准化。"""
        if sc_edges is None:
            idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1, device=sc.device)
            sc_edges = torch.log1p(sc[..., idx[0], idx[1]])
            sc_edges = (sc_edges - self.sc_mean) / self.sc_std.clamp_min(1e-6)
        return self(sc, sc_edges, fc_warmup, run).fc_matrices
