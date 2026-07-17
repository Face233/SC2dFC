from __future__ import annotations

import torch
from torch import nn

from ..types import Prediction
from .autoencoder import FCAutoencoder
from .condition import ConditionEncoder
from .temporal import TCNDecoder, TransformerTrajectoryDecoder


def torch_edges_to_matrix(edges: torch.Tensor, n_nodes: int = 90) -> torch.Tensor:
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
        dynamic = decoded - decoded.mean(dim=1, keepdim=True)
        static = self.static_head(condition)[:, None]
        template = self.group_template[:steps][None]
        fc_z = template + static + dynamic
        matrices = torch_edges_to_matrix(torch.tanh(fc_z), self.n_nodes)
        return Prediction(fc_z_edges=fc_z, fc_matrices=matrices, latent=latent)

    @torch.no_grad()
    def predict(self, sc: torch.Tensor, fc_warmup: torch.Tensor, run: torch.Tensor, sc_edges: torch.Tensor | None = None) -> torch.Tensor:
        if sc_edges is None:
            idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1, device=sc.device)
            sc_edges = torch.log1p(sc[..., idx[0], idx[1]])
            sc_edges = (sc_edges - self.sc_mean) / self.sc_std.clamp_min(1e-6)
        return self(sc, sc_edges, fc_warmup, run).fc_matrices
