from __future__ import annotations

import torch
from torch import nn


class FiLMTCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, 2 * dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(dim, 2 * dim)
        self.output = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        gamma_beta = self.condition(condition)[:, None]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        content, gate = y.chunk(2, dim=-1)
        y = (content * (1 + gamma) + beta) * torch.sigmoid(gate)
        y = self.output(y.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(y)


class TCNDecoder(nn.Module):
    def __init__(self, dim: int = 256, max_steps: int = 256, dilations=(1, 2, 4, 8, 16, 32), dropout: float = 0.1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        self.blocks = nn.ModuleList([FiLMTCNBlock(dim, dilation, dropout) for dilation in dilations])
        self.norm = nn.LayerNorm(dim)

    def forward(self, condition: torch.Tensor, memory: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        x = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        for block in self.blocks:
            x = block(x, condition)
        return self.norm(x)


class TransformerTrajectoryDecoder(nn.Module):
    def __init__(
        self, dim: int = 256, max_steps: int = 256, layers: int = 4, heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_steps, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers, norm=nn.LayerNorm(dim))

    def forward(self, condition: torch.Tensor, memory: torch.Tensor, steps: int) -> torch.Tensor:
        if steps > len(self.queries):
            raise ValueError(f"Requested {steps} steps, maximum is {len(self.queries)}")
        query = self.queries[:steps][None].expand(condition.shape[0], -1, -1) + condition[:, None]
        return self.decoder(query, memory)

