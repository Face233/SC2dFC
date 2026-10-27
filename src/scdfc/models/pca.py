"""Frozen, unwhitened FC PCA with the same encode/decode contract as an AE."""
from __future__ import annotations

import torch
from torch import nn


class FrozenFCPCA(nn.Module):
    """Fixed Fisher-z edge projection; mean and basis are checkpointed buffers."""

    def __init__(self, components, mean):
        super().__init__()
        basis = torch.as_tensor(components, dtype=torch.float32).detach().clone()
        center = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        if (basis.ndim != 2 or center.ndim != 1 or basis.shape[1] != len(center)
                or not 0 < basis.shape[0] <= basis.shape[1]
                or not torch.isfinite(basis).all() or not torch.isfinite(center).all()):
            raise ValueError("PCA requires finite components [latent,edges] and mean [edges]")
        self.register_buffer("components", basis)
        self.register_buffer("mean", center)

    @property
    def latent_dim(self) -> int:
        return self.components.shape[0]

    @property
    def n_edges(self) -> int:
        return self.components.shape[1]

    def encode(self, edges: torch.Tensor) -> torch.Tensor:
        """Map [...,edges] to [...,latent], without whitening or extra scaling."""
        if edges.ndim < 1 or edges.shape[-1] != self.n_edges:
            raise ValueError(f"Expected PCA input with last dimension {self.n_edges}")
        return (edges - self.mean) @ self.components.T

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct a complete FC; the fitted training mean is added once."""
        if latent.ndim < 1 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected PCA latent with last dimension {self.latent_dim}")
        return latent @ self.components + self.mean

    def forward(self, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(edges)
        return self.decode(latent), latent
