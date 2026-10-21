"""FC-history GRU + future self-attention + frozen persistence baseline."""
from __future__ import annotations

import torch
from torch import nn

from .history import FCHistoryGRU, HistoryPrediction
from .sequence import TransformerTrajectoryDecoder


class FCHistoryPersistenceTransformer(nn.Module):
    # Reuse the frozen per-window encoding contract, without creating unused heads.
    encode_history = FCHistoryGRU.encode_history

    def __init__(self, fc_autoencoder, group_template, context_template, alpha,
                 n_nodes=90, hidden_dim=256, dropout=0.1, warmup_windows=1,
                 forecast_origin=9, transformer_layers=4, transformer_heads=8,
                 transformer_ffn_dim=1024):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_nodes*(n_nodes-1)//2
        self.warmup_windows = warmup_windows
        self.forecast_origin = forecast_origin
        self.output_decomposition = "history_decay_residual"
        self.fc_autoencoder = fc_autoencoder
        self.fc_autoencoder.requires_grad_(False)
        self.fc_autoencoder.eval()
        latent_dim = int(fc_autoencoder.encoder[-1].normalized_shape[0])
        steps = len(group_template)
        if (latent_dim != hidden_dim or group_template.shape != (steps, self.n_edges)
                or context_template.shape != (self.n_edges,) or alpha.shape != (steps,)):
            raise ValueError("History Transformer dimensions do not match persistence artifact")
        if not torch.isfinite(alpha).all() or ((alpha < 0) | (alpha > 1)).any():
            raise ValueError("Persistence alpha must be finite and in [0,1]")
        self.register_buffer("group_template", group_template.float())
        self.register_buffer("context_template", context_template.float())
        self.register_buffer("decay_alpha", alpha.float())
        self.history_gru = nn.GRU(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.temporal = TransformerTrajectoryDecoder(hidden_dim, steps, transformer_layers,
                                                     transformer_heads, transformer_ffn_dim, dropout)
        self.direct_edge_head = nn.Linear(hidden_dim, self.n_edges)

    def train(self, mode=True):
        super().train(mode)
        self.fc_autoencoder.eval()
        return self

    def forward(self, sc_matrix, sc_edges, fc_warmup, steps=None):
        steps = len(self.group_template) if steps is None else steps
        if steps != len(self.group_template):
            raise ValueError("Prediction length must match the frozen forecast protocol")
        encoded = self.encode_history(fc_warmup)
        _, hidden = self.history_gru(encoded)
        future = self.temporal(self.condition_norm(hidden[-1]), steps)
        residual = self.direct_edge_head(future)
        last = fc_warmup if fc_warmup.ndim == 2 else fc_warmup[:, -1]
        baseline = (self.group_template[None]
                    + self.decay_alpha[None, :, None]*(last-self.context_template)[..., None, :])
        return HistoryPrediction(baseline+residual, residual, baseline, future, self.n_nodes)
