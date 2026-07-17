from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.sequence import torch_edges_to_matrix


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (prediction * target).sum(dim=-1)
    denominator = prediction.square().sum(dim=-1).sqrt() * target.square().sum(dim=-1).sqrt()
    return (1 - numerator / denominator.clamp_min(eps)).mean()


def variance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction.var(dim=1, unbiased=False), target.var(dim=1, unbiased=False))


def fcd_gram_loss(prediction: torch.Tensor, target: torch.Tensor, max_windows: int = 32) -> torch.Tensor:
    steps = prediction.shape[1]
    if steps > max_windows:
        indices = torch.linspace(0, steps - 1, max_windows, device=prediction.device).long()
        prediction, target = prediction[:, indices], target[:, indices]
    prediction = F.normalize(prediction - prediction.mean(-1, keepdim=True), dim=-1)
    target = F.normalize(target - target.mean(-1, keepdim=True), dim=-1)
    return F.smooth_l1_loss(prediction @ prediction.transpose(1, 2), target @ target.transpose(1, 2))


def contrastive_loss(prediction: torch.Tensor, target: torch.Tensor, start: int, temperature: float = 0.1) -> torch.Tensor:
    pred_embed = F.normalize(prediction[:, start:].mean(1), dim=-1)
    true_embed = F.normalize(target[:, start:].mean(1), dim=-1)
    logits = pred_embed @ true_embed.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def psd_penalty(z_edges: torch.Tensor, n_nodes: int = 90, max_windows: int = 4) -> torch.Tensor:
    steps = z_edges.shape[1]
    indices = torch.linspace(0, steps - 1, min(max_windows, steps), device=z_edges.device).long()
    matrices = torch_edges_to_matrix(torch.tanh(z_edges[:, indices]), n_nodes)
    eigenvalues = torch.linalg.eigvalsh(matrices)
    return torch.relu(-eigenvalues).square().mean()


class CompositeLoss:
    def __init__(self, weights: dict[str, float], nonoverlap_start: int, n_nodes: int = 90) -> None:
        self.weights = weights
        self.nonoverlap_start = nonoverlap_start
        self.n_nodes = n_nodes

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, group_template: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        template = group_template[: target.shape[1]][None]
        components = {
            "edge": F.smooth_l1_loss(prediction, target),
            "residual_corr": correlation_loss(prediction[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :], target[:, self.nonoverlap_start :] - template[:, self.nonoverlap_start :]),
            "difference": F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1]),
            "static": F.smooth_l1_loss(prediction.mean(1), target.mean(1)),
            "variance": variance_loss(prediction, target),
            "fcd": fcd_gram_loss(prediction, target),
            "contrastive": contrastive_loss(prediction, target, self.nonoverlap_start),
            "psd": psd_penalty(prediction, self.n_nodes),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components

