"""SC-only generators of unbounded first-window persistence corrections."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import SplineTransformer

from .sc_encoders import HCPGCNEncoder


def alpha_basis(steps: int, parameters: int) -> np.ndarray:
    if steps < 6 or parameters not in {1, 6, steps}:
        raise ValueError("Alpha basis requires K=1, K=6 or K=forecast steps")
    if parameters == 1:
        return np.ones((steps, 1), dtype=np.float32)
    if parameters == steps:
        return np.eye(steps, dtype=np.float32)
    return SplineTransformer(n_knots=4, degree=3, knots="uniform", include_bias=True).fit_transform(
        np.linspace(0, 1, steps)[:, None]).astype(np.float32)


class SCAlphaGenerator(nn.Module):
    """Only SC features enter this network; no FC or target tensors are accepted."""

    def __init__(self, encoder: str, parameters: int, steps: int, *, n_nodes=90,
                 pca_components=64, mlp_hidden=32, dropout=.1, gcn_hidden=128, gcn_output=64):
        super().__init__()
        # The same encoder state and training RNG must be used across K choices.
        if encoder == "linear":
            self.encoder = nn.Identity()
            dim = pca_components
        elif encoder == "mlp":
            self.encoder = nn.Sequential(nn.Linear(pca_components, mlp_hidden), nn.GELU(), nn.Dropout(dropout))
            dim = mlp_hidden
        elif encoder == "gcn":
            self.encoder = HCPGCNEncoder(n_nodes, gcn_hidden, gcn_output)
            dim = gcn_output
        else:
            raise ValueError("SC alpha encoder must be linear, mlp or gcn")
        # Constructing different output widths must not change subsequent RNG.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.Linear(dim, parameters)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.encoder_kind = encoder
        self.register_buffer("basis", torch.from_numpy(alpha_basis(steps, parameters)))

    def forward(self, sc: torch.Tensor) -> torch.Tensor:
        features = self.encoder(sc)
        if self.encoder_kind == "gcn":
            features = features[0]
        return self.head(features) @ self.basis.T


def sufficient_mse(alpha, q, u, v):
    """Exact per-subject/time mean over edges; no division by a tiny offset norm."""
    return v - 2 * alpha * u + alpha.square() * q[:, None]
