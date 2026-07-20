from __future__ import annotations

import torch
from torch import nn


def symmetric_normalize_with_self_loops(adjacency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return D^(-1/2) (A + I) D^(-1/2) for a batch of non-negative SC matrices."""
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError(f"Expected [batch, nodes, nodes] adjacency, got {tuple(adjacency.shape)}")
    adjacency = adjacency.clamp_min(0)
    identity = torch.eye(adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device)[None]
    adjacency = adjacency + identity
    inverse_sqrt_degree = adjacency.sum(dim=-1).clamp_min(eps).rsqrt()
    return inverse_sqrt_degree[:, :, None] * adjacency * inverse_sqrt_degree[:, None, :]


class GraphConvolution(nn.Module):
    """Bias-free graph convolution matching the original HCP_GCN implementation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.bmm(adjacency, self.linear(features)))


class HCPGCNEncoder(nn.Module):
    """Encode SC with identity ROI features, two normalized GCN layers, and max pooling."""

    def __init__(self, n_nodes: int = 90, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.gcn1 = GraphConvolution(n_nodes, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)
        self.register_buffer("roi_identity", torch.eye(n_nodes), persistent=False)

    def forward(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.shape[-2:] != (self.n_nodes, self.n_nodes):
            raise ValueError(f"Expected {self.n_nodes}x{self.n_nodes} SC matrices, got {tuple(adjacency.shape[-2:])}")
        normalized = symmetric_normalize_with_self_loops(adjacency)
        features = self.roi_identity.to(dtype=adjacency.dtype)[None].expand(adjacency.shape[0], -1, -1)
        tokens = self.gcn1(normalized, features)
        tokens = self.gcn2(normalized, tokens)
        return tokens.max(dim=1).values, tokens
