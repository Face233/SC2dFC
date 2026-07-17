from __future__ import annotations

import torch
from torch import nn

from .autoencoder import FCAutoencoder
from .sequence import Prediction, torch_edges_to_matrix


def persistence(fc_warmup: torch.Tensor, steps: int) -> torch.Tensor:
    return fc_warmup[:, None].expand(-1, steps, -1)


def group_mean(template: torch.Tensor, batch_size: int) -> torch.Tensor:
    return template[None].expand(batch_size, -1, -1)


class DirectSCMLP(nn.Module):
    """Feasible direct-sequence MLP baseline using the shared FC latent space."""

    def __init__(self, autoencoder: FCAutoencoder, group_template: torch.Tensor, hidden: int = 512, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc_autoencoder = autoencoder
        self.n_edges = group_template.shape[-1]
        self.steps = group_template.shape[0]
        self.latent_dim = latent_dim
        self.network = nn.Sequential(nn.Linear(self.n_edges, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, self.steps * latent_dim))
        self.register_buffer("group_template", group_template.float())

    def forward(self, sc_matrix, sc_edges, fc_warmup, run) -> Prediction:
        latent = self.network(sc_edges).view(-1, self.steps, self.latent_dim)
        residual = self.fc_autoencoder.decode(latent)
        fc_z = self.group_template[None] + residual
        return Prediction(fc_z, torch_edges_to_matrix(torch.tanh(fc_z)), latent)


class GCNGRUBaseline(nn.Module):
    def __init__(self, autoencoder: FCAutoencoder, group_template: torch.Tensor, n_nodes: int = 90, hidden: int = 256) -> None:
        super().__init__()
        self.fc_autoencoder = autoencoder
        self.register_buffer("group_template", group_template.float())
        self.node_projection = nn.Linear(2, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, sc_matrix, sc_edges, fc_warmup, run) -> Prediction:
        degree = (sc_matrix > 0).float().sum(-1)
        strength = sc_matrix.sum(-1)
        nodes = self.node_projection(torch.stack([torch.log1p(strength), degree], -1))
        normalized = sc_matrix / sc_matrix.sum(-1, keepdim=True).clamp_min(1e-6)
        graph = torch.bmm(normalized, nodes).mean(1)
        sequence = graph[:, None].expand(-1, self.group_template.shape[0], -1)
        hidden, _ = self.gru(sequence, graph[None])
        latent = self.output(hidden)
        fc_z = self.group_template[None] + self.fc_autoencoder.decode(latent)
        return Prediction(fc_z, torch_edges_to_matrix(torch.tanh(fc_z)), latent)
