from __future__ import annotations

import torch
from torch import nn


class FCAutoencoder(nn.Module):
    def __init__(self, n_edges: int = 4005, latent_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_edges, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_edges),
        )

    def encode(self, edges: torch.Tensor) -> torch.Tensor:
        return self.encoder(edges)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(edges)
        return self.decode(latent), latent

