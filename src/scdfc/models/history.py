from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .autoencoder import FCAutoencoder
from .sequence import torch_edges_to_matrix


@dataclass
class HistoryPrediction:
    fc_z_edges: torch.Tensor
    dynamic_residual: torch.Tensor
    residual_baseline: torch.Tensor
    latent: torch.Tensor
    n_nodes: int

    @property
    def fc_matrices(self) -> torch.Tensor:
        # Training only needs edges; avoid allocating B*H*N*N on every forward.
        return torch_edges_to_matrix(torch.tanh(self.fc_z_edges), self.n_nodes)


class FCHistoryGRU(nn.Module):
    """Shared frozen window encoder and ordered history GRU; no SC branch."""

    def __init__(self, fc_autoencoder: FCAutoencoder, group_template: torch.Tensor,
                 n_nodes: int = 90, hidden_dim: int = 256, dropout: float = 0.1,
                 warmup_windows: int = 1, forecast_origin: int = 19) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_nodes * (n_nodes - 1) // 2
        self.warmup_windows = warmup_windows
        self.forecast_origin = forecast_origin
        self.output_decomposition = "last_observed_fc_residual"
        self.fc_autoencoder = fc_autoencoder
        self.fc_autoencoder.requires_grad_(False)
        self.fc_autoencoder.eval()
        latent_dim = int(fc_autoencoder.encoder[-1].normalized_shape[0])
        if latent_dim != hidden_dim or group_template.ndim != 2 or group_template.shape[1] != self.n_edges:
            raise ValueError("History latent/hidden dimensions and FC template must match")
        self.register_buffer("group_template", group_template)
        self.history_gru = nn.GRU(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.queries = nn.Parameter(torch.randn(len(group_template), hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, self.n_edges))

    def train(self, mode: bool = True):
        super().train(mode)
        self.fc_autoencoder.eval()
        return self

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 2:
            history = history[:, None, :]
        if history.ndim != 3 or history.shape[1:] != (self.warmup_windows, self.n_edges):
            raise ValueError("History must match the configured [batch,K,edges]")
        batch, windows, edges = history.shape
        with torch.no_grad():
            latent = self.fc_autoencoder.encode(history.reshape(batch * windows, edges))
        return latent.reshape(batch, windows, -1)

    def forward(self, sc_matrix, sc_edges, fc_warmup: torch.Tensor,
                steps: int | None = None) -> HistoryPrediction:
        steps = len(self.queries) if steps is None else steps
        if steps != len(self.queries):
            raise ValueError("Prediction length must match the frozen forecast protocol")
        latent = self.encode_history(fc_warmup)
        _, hidden = self.history_gru(latent)
        future = self.norm(hidden[-1][:, None] + self.queries[None])
        delta = self.head(future)
        last = fc_warmup if fc_warmup.ndim == 2 else fc_warmup[:, -1]
        baseline = last[:, None].expand_as(delta)
        return HistoryPrediction(baseline + delta, delta, baseline, future, self.n_nodes)
