import pytest
import torch

from scdfc.training import CompositeLoss
from scdfc.models import ConditionalSequenceModel, FCAutoencoder
from scdfc.models.baselines import GCNGRUBaseline


@pytest.mark.parametrize("decoder", ["tcn", "transformer"])
def test_sequence_models_return_full_valid_shape(decoder):
    torch.manual_seed(0)
    batch, nodes, edges, steps = 2, 90, 4005, 5
    autoencoder = FCAutoencoder(edges, latent_dim=32, dropout=0)
    model = ConditionalSequenceModel(
        autoencoder,
        torch.zeros(steps, edges),
        decoder_type=decoder,
        n_nodes=nodes,
        hidden_dim=32,
        graph_layers=1,
        graph_heads=4,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ffn_dim=64,
        tcn_dilations=(1, 2),
        dropout=0,
    )
    sc = torch.rand(batch, nodes, nodes)
    sc = (sc + sc.transpose(1, 2)) / 2
    sc[:, torch.arange(nodes), torch.arange(nodes)] = 0
    sc_edges = sc[:, torch.triu_indices(nodes, nodes, 1)[0], torch.triu_indices(nodes, nodes, 1)[1]]
    result = model(sc, sc_edges, torch.randn(batch, edges), torch.tensor([0, 1]))
    assert result.fc_z_edges.shape == (batch, steps, edges)
    assert result.fc_matrices.shape == (batch, steps, nodes, nodes)
    torch.testing.assert_close(result.fc_matrices, result.fc_matrices.transpose(-1, -2))
    torch.testing.assert_close(torch.diagonal(result.fc_matrices, dim1=-2, dim2=-1), torch.ones(batch, steps, nodes))
    assert result.fc_matrices.abs().max() <= 1


def test_composite_loss_backpropagates():
    prediction = torch.randn(3, 30, 4005, requires_grad=True)
    target = torch.randn_like(prediction)
    template = torch.zeros(30, 4005)
    weights = {"edge": 1, "residual_corr": 0.5, "difference": 0.25, "static": 0.25, "variance": 0.25, "fcd": 0.1, "contrastive": 0.1, "psd": 0.01}
    loss, components = CompositeLoss(weights, 17)(prediction, target, template)
    assert set(components) == set(weights)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_gcn_gru_baseline_uses_common_prediction_contract():
    autoencoder = FCAutoencoder(4005, latent_dim=16, dropout=0)
    model = GCNGRUBaseline(autoencoder, torch.zeros(4, 4005), hidden=16)
    sc = torch.rand(2, 90, 90)
    output = model(sc, torch.rand(2, 4005), torch.rand(2, 4005), torch.tensor([0, 1]))
    assert output.fc_z_edges.shape == (2, 4, 4005)
    assert output.fc_matrices.shape == (2, 4, 90, 90)
