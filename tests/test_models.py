import inspect

import pytest
import torch
import numpy as np

from scdfc.training import AutoencoderLoss, CompositeLoss, build_sequence_model, learning_rate_scale, long_horizon_variance_loss, loss_inputs
from scdfc.management import file_sha256
from scdfc.evaluation import _load_model
from scdfc.models import CommonInputLSTM, CommonInputMLP, ConditionalSequenceModel, FCAutoencoder, HCPGCNEncoder, PCARidgeBaseline
from scdfc.models.baselines import DirectSCMLP, GCNGRUBaseline
from scdfc.models.sc_encoders import symmetric_normalize_with_self_loops


def test_autoencoder_edge_only_loss_is_mse():
    prediction = torch.tensor([[0.0, 2.0, -2.0]])
    target = torch.zeros_like(prediction)
    loss, components = AutoencoderLoss({"edge": 1.0})(prediction, target)

    assert set(components) == {"edge"}
    assert torch.allclose(loss, torch.nn.functional.mse_loss(prediction, target))


def test_warmup_cosine_learning_rate_schedule_reaches_base_and_eta_min():
    schedule = {"name": "warmup_cosine", "warmup_epochs": 5, "eta_min": 1e-5, "base_learning_rate": 3e-4}
    assert learning_rate_scale(schedule, epoch=0, max_epochs=100) == pytest.approx(0.2)
    assert learning_rate_scale(schedule, epoch=4, max_epochs=100) == pytest.approx(1.0)
    assert learning_rate_scale(schedule, epoch=5, max_epochs=100) == pytest.approx(1.0)
    assert learning_rate_scale(schedule, epoch=99, max_epochs=100) == pytest.approx(1e-5 / 3e-4)


@pytest.mark.parametrize("decoder", ["gru", "tcn", "transformer"])
@pytest.mark.parametrize("sc_encoder", ["hybrid", "hcp_gcn"])
@pytest.mark.parametrize("output_head", ["e0003_reconstruction_decoder", "direct_edge_linear"])
def test_sequence_models_return_full_valid_shape(decoder, sc_encoder, output_head):
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
        gru_layers=2,
        tcn_dilations=(1, 2),
        dropout=0,
        sc_encoder_type=sc_encoder,
        output_head=output_head,
    )
    sc = torch.rand(batch, nodes, nodes)
    sc = (sc + sc.transpose(1, 2)) / 2
    sc[:, torch.arange(nodes), torch.arange(nodes)] = 0
    sc_edges = sc[:, torch.triu_indices(nodes, nodes, 1)[0], torch.triu_indices(nodes, nodes, 1)[1]]
    result = model(sc, sc_edges, torch.randn(batch, edges))
    assert result.fc_z_edges.shape == (batch, steps, edges)
    assert result.fc_matrices.shape == (batch, steps, nodes, nodes)
    if output_head == "e0003_reconstruction_decoder":
        torch.testing.assert_close(result.fc_z_edges, autoencoder.decode(result.latent))
    torch.testing.assert_close(result.fc_matrices, result.fc_matrices.transpose(-1, -2))
    torch.testing.assert_close(torch.diagonal(result.fc_matrices, dim1=-2, dim2=-1), torch.ones(batch, steps, nodes))
    assert result.fc_matrices.abs().max() <= 1


@pytest.mark.parametrize(
    ("ablation", "zero_slice"),
    [("fc1_only", slice(0, 256)), ("sc_only", slice(256, None))],
)
def test_information_ablation_zeros_embeddings_after_encoding(ablation, zero_slice):
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = ConditionalSequenceModel(
        autoencoder,
        torch.zeros(3, 6),
        decoder_type="gru",
        n_nodes=4,
        hidden_dim=4,
        graph_layers=1,
        graph_heads=4,
        gru_layers=1,
        dropout=0,
        ablation=ablation,
    )
    captured = {}

    def capture_combined(_module, arguments):
        captured["combined"] = arguments[0].detach()

    handle = model.condition_encoder.value.register_forward_pre_hook(capture_combined)
    sc = torch.rand(2, 4, 4)
    sc = (sc + sc.transpose(1, 2)) / 2
    edges = sc[:, torch.triu_indices(4, 4, 1)[0], torch.triu_indices(4, 4, 1)[1]]
    model(sc, edges, torch.rand(2, 6))
    handle.remove()
    assert torch.count_nonzero(captured["combined"][:, zero_slice]) == 0


def test_e0032_residual_model_replays_alpha_and_uses_only_matched_sc():
    torch.manual_seed(7)
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    template = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
    context = torch.full((6,), 0.2)
    alpha = torch.tensor([1.0, 0.5, 0.0])
    model = ConditionalSequenceModel(
        autoencoder, template, decoder_type="transformer", n_nodes=4, hidden_dim=4,
        transformer_layers=1, transformer_heads=2, transformer_ffn_dim=16,
        sc_encoder_type="hcp_gcn", hcp_gcn_hidden_dim=8, hcp_gcn_output_dim=4,
        output_head="direct_edge_linear", output_decomposition="e0032_decay_residual",
        context_template=context, decay_alpha=alpha, fusion_type="concat_projection",
        ablation="fc1_only", dropout=0,
    ).eval()
    warmup = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
    sc = torch.rand(2, 4, 4)
    sc = (sc + sc.transpose(1, 2)) / 2
    edges = sc[:, torch.triu_indices(4, 4, 1)[0], torch.triu_indices(4, 4, 1)[1]]
    first = model(sc, edges, warmup)
    expected_baseline = template[None] + alpha[None, :, None] * (warmup - context)[:, None]
    torch.testing.assert_close(first.residual_baseline, expected_baseline)
    torch.testing.assert_close(first.fc_z_edges, expected_baseline + first.dynamic_residual)
    prediction, target = loss_inputs(first, first.fc_z_edges + 0.25)
    torch.testing.assert_close(prediction, first.dynamic_residual)
    torch.testing.assert_close(target, first.dynamic_residual + 0.25)
    torch.testing.assert_close(model(sc.flip(0), edges.flip(0), warmup).fc_z_edges, first.fc_z_edges)
    model.condition_encoder.ablation = "full"
    assert not torch.allclose(model(sc.flip(0), edges.flip(0), warmup).fc_z_edges,
                              model(sc, edges, warmup).fc_z_edges)


def test_e0032_sequence_builder_verifies_frozen_alpha_artifact(tmp_path):
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    autoencoder_path = tmp_path / "autoencoder.pt"
    torch.save({"model": autoencoder.state_dict()}, autoencoder_path)
    stats_path = tmp_path / "training_stats.npz"
    np.savez_compressed(stats_path, group_template=np.zeros((3, 6), np.float32),
                        context_template=np.zeros(6, np.float32), fc_mean=np.zeros(6, np.float32),
                        sc_mean=np.zeros(6, np.float32), sc_std=np.ones(6, np.float32))
    alpha_path = tmp_path / "alpha_fit.npz"
    np.savez_compressed(alpha_path, alpha=np.array([1.0, 0.5, 0.0], np.float32))
    config = {
        "paths": {"root": str(tmp_path)},
        "data": {"n_nodes": 4, "warmup_windows": 1},
        "model": {"fc_latent_dim": 4, "hidden_dim": 4, "dropout": 0.0,
                  "sc_graph_layers": 1, "sc_graph_heads": 2,
                  "transformer_layers": 1, "transformer_heads": 2, "transformer_ffn_dim": 16,
                  "tcn_dilations": [1], "sc_encoder": "hcp_gcn", "hcp_gcn_hidden_dim": 8,
                  "hcp_gcn_output_dim": 4, "output_head": "direct_edge_linear",
                  "output_decomposition": "e0032_decay_residual", "fusion_type": "concat_projection"},
        "artifacts": {"alpha_fit": {"path": str(alpha_path), "sha256": file_sha256(alpha_path)}},
    }
    model = build_sequence_model(config, 3, "transformer", stats_path, torch.device("cpu"),
                                 autoencoder_path=autoencoder_path, ablation="fc1_only")
    torch.testing.assert_close(model.decay_alpha, torch.tensor([1.0, 0.5, 0.0]))
    config["artifacts"]["alpha_fit"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_sequence_model(config, 3, "transformer", stats_path, torch.device("cpu"),
                             autoencoder_path=autoencoder_path, ablation="fc1_only")


def test_warmup_gru_encodes_multiple_fc_windows_and_backpropagates():
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = ConditionalSequenceModel(
        autoencoder,
        torch.zeros(3, 6),
        decoder_type="gru",
        n_nodes=4,
        hidden_dim=4,
        graph_layers=1,
        graph_heads=4,
        gru_layers=1,
        dropout=0,
        warmup_encoder="gru",
        warmup_gru_layers=1,
    )
    sc = torch.rand(2, 4, 4)
    sc = (sc + sc.transpose(1, 2)) / 2
    sc[:, torch.arange(4), torch.arange(4)] = 0
    indices = torch.triu_indices(4, 4, 1)
    edges = sc[:, indices[0], indices[1]]
    result = model(sc, edges, torch.randn(2, 5, 6))
    assert result.fc_z_edges.shape == (2, 3, 6)
    result.fc_z_edges.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.condition_encoder.warmup_gru.parameters())


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
    weights = {"edge": 1.0, "difference": 0.25}
    loss, components = CompositeLoss(weights, 17)(prediction, target, template)
    assert set(components) == set(weights)
    assert len(components) <= 3
    with pytest.raises(ValueError, match="At most three"):
        CompositeLoss({**weights, "residual_corr": 0.5, "static": 0.25}, 17)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_composite_loss_uses_mse_for_edge_and_difference():
    prediction = torch.tensor([[[0.0], [2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    criterion = CompositeLoss({"edge": 1.0, "difference": 0.25}, 1)
    loss, components = criterion(prediction, target, torch.zeros_like(target[0]))
    assert components["edge"].item() == pytest.approx(2.0)
    assert components["difference"].item() == pytest.approx(4.0)
    assert loss.item() == pytest.approx(3.0)


def test_composite_loss_uses_mse_when_configured():
    prediction = torch.tensor([[[0.0], [2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    criterion = CompositeLoss({"edge": 1.0, "difference": 0.25}, 1, loss_type="mse")
    loss, components = criterion(prediction, target, torch.zeros_like(target[0]))
    assert components["edge"].item() == pytest.approx(2.0)
    assert components["difference"].item() == pytest.approx(4.0)
    assert loss.item() == pytest.approx(3.0)


def test_long_horizon_dynamic_losses_are_zero_for_exact_prediction():
    time = torch.arange(30, dtype=torch.float32)
    target = torch.stack((torch.sin(time), torch.cos(time)), dim=-1)[None].repeat(2, 1, 1)
    assert long_horizon_variance_loss(target, target, nonoverlap_start=3).item() == pytest.approx(0.0)


def test_gcn_gru_baseline_uses_common_prediction_contract():
    autoencoder = FCAutoencoder(4005, latent_dim=16, dropout=0)
    model = GCNGRUBaseline(autoencoder, torch.zeros(4, 4005), hidden=16)
    sc = torch.rand(2, 90, 90)
    output = model(sc, torch.rand(2, 4005), torch.rand(2, 4005))
    assert output.fc_z_edges.shape == (2, 4, 4005)
    assert output.fc_matrices.shape == (2, 4, 90, 90)


@pytest.mark.parametrize("model_type", [CommonInputMLP, CommonInputLSTM])
def test_common_input_baselines_use_prediction_contract(model_type):
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = model_type(autoencoder, torch.zeros(3, 6), hidden=8)
    sc = torch.rand(2, 4, 4)
    result = model(sc, torch.rand(2, 6), torch.rand(2, 6))
    assert result.fc_z_edges.shape == (2, 3, 6)
    assert result.fc_matrices.shape == (2, 3, 4, 4)


def test_pca_ridge_baseline_is_portable_torch_module():
    autoencoder = FCAutoencoder(6, latent_dim=4, dropout=0)
    model = PCARidgeBaseline(autoencoder, torch.zeros(3, 6), n_components=2, sc_edges=6, latent_dim=4)
    model.pca_components.normal_()
    model.ridge_coef.normal_()
    result = model(torch.rand(2, 4, 4), torch.rand(2, 6), torch.rand(2, 6))
    assert result.fc_z_edges.shape == (2, 3, 6)


@pytest.mark.parametrize(
    "model_type",
    [ConditionalSequenceModel, DirectSCMLP, GCNGRUBaseline, CommonInputMLP, CommonInputLSTM, PCARidgeBaseline],
)
def test_model_interfaces_do_not_accept_run_direction(model_type):
    assert "run" not in inspect.signature(model_type.forward).parameters


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
