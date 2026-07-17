from __future__ import annotations

import math

import torch
from torch import nn

from .autoencoder import FCAutoencoder


class BiasedGraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
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
        edge_bias = torch.log1p(torch.clamp(adjacency, min=0))
        scores = scores + self.bias_scale[None, :, None, None] * edge_bias[:, None]
        attention = self.dropout(scores.softmax(dim=-1))
        mixed = torch.einsum("bhnm,bmhd->bnhd", attention, v).reshape(batch, nodes, dim)
        tokens = residual + self.output(mixed)
        return tokens + self.ffn(self.norm2(tokens))


class SCGraphEncoder(nn.Module):
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
        combined = 128 + 128 + fc_autoencoder.encoder[-1].normalized_shape[0] + 32
        self.value = nn.Linear(combined, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(combined, hidden_dim), nn.Sigmoid())
        self.graph_token_projection = nn.Linear(128, hidden_dim)
        self.edge_token_projection = nn.Linear(128, hidden_dim)
        self.fc_token_projection = nn.Linear(fc_autoencoder.encoder[-1].normalized_shape[0], hidden_dim)

    def forward(
        self, sc_matrix: torch.Tensor, sc_edges: torch.Tensor, fc_warmup: torch.Tensor, run: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_global, graph_tokens = self.graph(sc_matrix)
        edge_global = self.edge_mlp(sc_edges)
        warmup = self.fc_autoencoder.encode(fc_warmup)
        combined = torch.cat([graph_global, edge_global, warmup, self.run_embedding(run)], dim=-1)
        condition = self.value(combined) * self.gate(combined)
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

