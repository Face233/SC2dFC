import pytest
import torch
import numpy as np

from scdfc.training import CompositeLoss
from scdfc.evaluation import _load_model
from scdfc.models import CommonInputLSTM, CommonInputMLP, ConditionalSequenceModel, FCAutoencoder, HCPGCNEncoder, PCARidgeBaseline
from scdfc.models.baselines import GCNGRUBaseline
from scdfc.models.sc_encoders import symmetric_normalize_with_self_loops


@pytest.mark.parametrize("decoder", ["tcn", "transformer"])
@pytest.mark.parametrize("sc_encoder", ["hybrid", "hcp_gcn"])
def test_sequence_models_return_full_valid_shape(decoder, sc_encoder):
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
        sc_encoder_type=sc_encoder,
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


def test_hcp_gcn_encoder_normalizes_and_backpropagates():
    torch.manual_seed(0)
    adjacency = torch.rand(2, 6, 6)
    adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
    adjacency[:, torch.arange(6), torch.arange(6)] = 0
    normalized = symmetric_normalize_with_self_loops(adjacency)
    torch.testing.assert_close(normalized, normalized.transpose(1, 2))
    assert torch.isfinite(normalized).all()
    assert (torch.diagonal(normalized, dim1=-2, dim2=-1) > 0).all()

    encoder = HCPGCNEncoder(n_nodes=6, hidden_dim=8, output_dim=4)
    global_embedding, tokens = encoder(adjacency)
    assert global_embedding.shape == (2, 4)
    assert tokens.shape == (2, 6, 4)
    global_embedding.sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


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


@pytest.mark.parametrize("model_type", [CommonInputMLP, CommonInputLSTM])
def test_common_input_baselines_use_prediction_contract(model_type):
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = model_type(autoencoder, torch.zeros(3, 6), hidden=8)
    sc = torch.rand(2, 4, 4)
    result = model(sc, torch.rand(2, 6), torch.rand(2, 6), torch.tensor([0, 1]))
    assert result.fc_z_edges.shape == (2, 3, 6)
    assert result.fc_matrices.shape == (2, 3, 4, 4)


def test_pca_ridge_baseline_is_portable_torch_module():
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = PCARidgeBaseline(autoencoder, torch.zeros(3, 6), n_components=2, sc_edges=6, latent_dim=4)
    model.pca_components.normal_()
    model.ridge_coef.normal_()
    result = model(torch.rand(2, 4, 4), torch.rand(2, 6), torch.rand(2, 6), torch.tensor([0, 1]))
    assert result.fc_z_edges.shape == (2, 3, 6)


def test_checkpoint_recovers_without_current_default_config(tmp_path):
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    artifact = tmp_path / "autoencoder.pt"
    torch.save({"model": autoencoder.state_dict()}, artifact)
    stats = tmp_path / "stats.npz"
    np.savez(stats, group_template=np.zeros((3, 6), dtype=np.float32))
    model = PCARidgeBaseline(autoencoder, torch.zeros(3, 6), n_components=2, sc_edges=6, latent_dim=4)
    checkpoint = tmp_path / "best.pt"
    torch.save({
        "model": model.state_dict(), "decoder_type": "pca_ridge", "sc_encoder_type": "hybrid",
        "ridge_dimensions": {"n_components": 2, "sc_edges": 6, "latent_dim": 4},
    }, checkpoint)
    config = {"data": {"n_nodes": 4}, "model": {"fc_latent_dim": 4, "dropout": 0}}
    recovered, payload = _load_model(config, 10, checkpoint, stats, torch.device("cpu"), artifact)
    assert payload["decoder_type"] == "pca_ridge"
    assert isinstance(recovered, PCARidgeBaseline)
