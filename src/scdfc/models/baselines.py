from __future__ import annotations

import torch
from torch import nn

from .autoencoder import FCAutoencoder
from .sequence import Prediction, torch_edges_to_matrix


def _nodes_from_edges(n_edges: int) -> int:
    n_nodes = int((1 + (1 + 8 * n_edges) ** 0.5) / 2)
    if n_nodes * (n_nodes - 1) // 2 != n_edges:
        raise ValueError(f"{n_edges} is not a valid undirected edge count")
    return n_nodes


def persistence(fc_warmup: torch.Tensor, steps: int) -> torch.Tensor:
    return fc_warmup[:, None].expand(-1, steps, -1)


def group_mean(template: torch.Tensor, batch_size: int) -> torch.Tensor:
    return template[None].expand(batch_size, -1, -1)


class DirectSCMLP(nn.Module):
    """Legacy v1 SC-only MLP baseline using the shared FC latent space."""

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
    """Legacy v1 SC-only graph/GRU baseline."""
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


class CommonInputMLP(nn.Module):
    """Non-temporal baseline using SC, warm-up FC, and run inputs."""

    def __init__(self, autoencoder: FCAutoencoder, group_template: torch.Tensor, hidden: int = 512) -> None:
        super().__init__()
        self.fc_autoencoder = autoencoder
        self.register_buffer("group_template", group_template.float())
        n_edges = int(group_template.shape[-1])
        latent_dim = int(autoencoder.encoder[-1].normalized_shape[0])
        self.steps = int(group_template.shape[0])
        self.latent_dim = latent_dim
        self.n_nodes = _nodes_from_edges(n_edges)
        self.network = nn.Sequential(
            nn.Linear(n_edges + latent_dim + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, self.steps * latent_dim),
        )

    def forward(self, sc_matrix, sc_edges, fc_warmup, run) -> Prediction:
        warmup = self.fc_autoencoder.encode(fc_warmup)
        run_one_hot = torch.nn.functional.one_hot(run, 2).to(sc_edges.dtype)
        values = torch.cat([sc_edges, warmup, run_one_hot], dim=-1)
        latent = self.network(values).view(-1, self.steps, self.latent_dim)
        fc_z = self.group_template[None] + self.fc_autoencoder.decode(latent)
        return Prediction(fc_z, torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes), latent)


class CommonInputLSTM(nn.Module):
    """LSTM baseline conditioned on the same inputs as the main model."""

    def __init__(self, autoencoder: FCAutoencoder, group_template: torch.Tensor, hidden: int = 256) -> None:
        super().__init__()
        self.fc_autoencoder = autoencoder
        self.register_buffer("group_template", group_template.float())
        n_edges = int(group_template.shape[-1])
        latent_dim = int(autoencoder.encoder[-1].normalized_shape[0])
        self.steps = int(group_template.shape[0])
        self.n_nodes = _nodes_from_edges(n_edges)
        self.condition = nn.Sequential(nn.Linear(n_edges + latent_dim + 2, hidden), nn.Tanh())
        self.step_embedding = nn.Parameter(torch.randn(self.steps, hidden) * 0.02)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, latent_dim)

    def forward(self, sc_matrix, sc_edges, fc_warmup, run) -> Prediction:
        warmup = self.fc_autoencoder.encode(fc_warmup)
        run_one_hot = torch.nn.functional.one_hot(run, 2).to(sc_edges.dtype)
        condition = self.condition(torch.cat([sc_edges, warmup, run_one_hot], dim=-1))
        sequence = self.step_embedding[None].expand(len(sc_edges), -1, -1) + condition[:, None]
        hidden, _ = self.lstm(sequence, (condition[None], torch.zeros_like(condition)[None]))
        latent = self.output(hidden)
        fc_z = self.group_template[None] + self.fc_autoencoder.decode(latent)
        return Prediction(fc_z, torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes), latent)


class PCARidgeBaseline(nn.Module):
    """PCA + Ridge baseline stored as Torch buffers for portable checkpoint recovery."""

    def __init__(
        self, autoencoder: FCAutoencoder, group_template: torch.Tensor, n_components: int,
        sc_edges: int, latent_dim: int,
    ) -> None:
        super().__init__()
        self.fc_autoencoder = autoencoder
        self.register_buffer("group_template", group_template.float())
        self.register_buffer("pca_mean", torch.zeros(sc_edges))
        self.register_buffer("pca_components", torch.zeros(n_components, sc_edges))
        feature_dim = n_components + latent_dim + 2
        output_dim = int(group_template.shape[0]) * latent_dim
        self.register_buffer("ridge_coef", torch.zeros(output_dim, feature_dim))
        self.register_buffer("ridge_intercept", torch.zeros(output_dim))
        self.steps = int(group_template.shape[0])
        self.latent_dim = latent_dim
        self.n_nodes = _nodes_from_edges(sc_edges)

    def forward(self, sc_matrix, sc_edges, fc_warmup, run) -> Prediction:
        projected_sc = (sc_edges - self.pca_mean) @ self.pca_components.T
        warmup = self.fc_autoencoder.encode(fc_warmup)
        run_one_hot = torch.nn.functional.one_hot(run, 2).to(sc_edges.dtype)
        features = torch.cat([projected_sc, warmup, run_one_hot], dim=-1)
        latent = (features @ self.ridge_coef.T + self.ridge_intercept).view(-1, self.steps, self.latent_dim)
        fc_z = self.group_template[None] + self.fc_autoencoder.decode(latent)
        return Prediction(fc_z, torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes), latent)
