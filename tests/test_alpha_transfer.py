from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scdfc.alpha_transfer import FrozenAlphaBranch, attach_frozen_alpha
from scdfc.alpha_transfer_evaluation import donor_indices, metric_row, paired_rows
from scdfc.config import load_config
from scdfc.management import config_sha256, file_sha256, validate_experiment_config
from scdfc.models.autoencoder import FCAutoencoder
from scdfc.models.sequence import ConditionalSequenceModel
from scdfc.sc_alpha import generator, transform_sc, tensor_digest, SCHEMA
from scdfc.training import build_sequence_model, loss_inputs

ROOT = Path(__file__).resolve().parents[1]


def source_payload(kind):
    config = load_config(next((ROOT/"configs/experiments").glob("E0054_*.yaml")))
    config["data"]["n_nodes"] = 4
    config["model"].update(sc_encoder=kind, pca_components=3, mlp_hidden=5, dropout=.8,
                            gcn_hidden=5, gcn_output=3)
    g = generator(config)
    with torch.no_grad():
        g.head.weight.normal_(0, .2)
        g.head.bias.copy_(torch.linspace(-2, 2, 223))
    transform = ({"kind": "gcn", "n_nodes": 4} if kind == "gcn" else
        {"kind": "pca", "n_nodes": 4, "mean": np.ones(6)*.2, "scale": np.ones(6)*.7,
         "pca_mean": np.ones(6)*.1, "components": np.arange(18).reshape(3, 6)/20})
    return dict(schema=SCHEMA, config=config, model=g.state_dict(), sc_transform=transform,
                alpha_raw=np.ones(223, np.float32)*.5, context_template=np.zeros(6, np.float32),
                group_template=np.zeros((223, 6), np.float32))


def residual_model(branch=None, ablation="fc1_only"):
    model = ConditionalSequenceModel(FCAutoencoder(6, latent_dim=4, dropout=0),
        torch.zeros(223, 6), decoder_type="transformer", n_nodes=4, hidden_dim=4,
        transformer_layers=1, transformer_heads=2, transformer_ffn_dim=8,
        sc_encoder_type="hcp_gcn", hcp_gcn_hidden_dim=5, hcp_gcn_output_dim=3,
        output_head="direct_edge_linear", output_decomposition="e0032_decay_residual",
        context_template=torch.zeros(6), decay_alpha=torch.ones(223)*.5,
        fusion_type="concat_projection", ablation=ablation, dropout=0)
    model.sc_alpha_component = branch
    return model


@pytest.mark.parametrize("kind", ["linear", "mlp", "gcn"])
def test_frozen_branch_replays_source_without_rng_dropout_or_gradients(kind):
    payload = source_payload(kind)
    before = torch.get_rng_state().clone()
    branch = FrozenAlphaBranch(payload)
    assert torch.equal(before, torch.get_rng_state())
    source = generator(payload["config"]).eval()
    source.load_state_dict(payload["model"])
    sc = torch.rand(2, 4, 4)
    expected = source(torch.from_numpy(transform_sc(sc.numpy(), payload["sc_transform"])))+.5
    branch.train()
    assert not branch.training and not branch.generator.training
    torch.testing.assert_close(branch(sc), expected)
    torch.testing.assert_close(branch(sc), branch(sc))
    assert branch(sc).min() < 0 and branch(sc).max() > 1
    model = residual_model(branch).train()
    model.fc_autoencoder.eval()
    state = tensor_digest(branch)
    result = model(sc, torch.zeros(2, 6), torch.randn(2, 6))
    (result.fc_z_edges-torch.ones_like(result.fc_z_edges)).square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in branch.parameters())
    assert model.direct_edge_head.weight.grad.abs().sum() > 0
    assert tensor_digest(branch) == state and not branch.training


def test_alpha_path_survives_fc_only_ablation_and_is_separate_from_residual_path():
    torch.manual_seed(7)
    model = residual_model(FrozenAlphaBranch(source_payload("linear"))).eval()
    sc, edges, fc = torch.rand(2, 4, 4), torch.rand(2, 6), torch.randn(2, 6)
    # Avoid an accidentally dead small random ReLU graph encoder in this path test.
    with torch.no_grad():
        model.condition_encoder.hcp_gcn.gcn1.linear.weight.copy_(torch.arange(20).reshape(5, 4)/20+.1)
        model.condition_encoder.hcp_gcn.gcn2.linear.weight.copy_(torch.arange(15).reshape(3, 5)/15+.1)
    sc[0].zero_()
    own = model(sc, edges, fc)
    swapped = model(sc.flip(0), edges.flip(0), fc)
    torch.testing.assert_close(own.dynamic_residual, swapped.dynamic_residual)
    assert not torch.allclose(own.residual_baseline, swapped.residual_baseline)
    model.condition_encoder.ablation = "full"
    own = model(sc, edges, fc)
    residual_swap = model(sc.flip(0), edges.flip(0), fc, sc_alpha_matrix=sc)
    torch.testing.assert_close(own.residual_baseline, residual_swap.residual_baseline)
    assert not torch.allclose(own.dynamic_residual, residual_swap.dynamic_residual)
    alpha_swap = model(sc, edges, fc, sc_alpha_matrix=sc.flip(0))
    torch.testing.assert_close(own.dynamic_residual, alpha_swap.dynamic_residual)


def test_zero_alpha_delta_reproduces_original_and_loss_uses_new_residual():
    model = residual_model().eval()
    sc, edges, fc = torch.rand(2, 4, 4), torch.rand(2, 6), torch.randn(2, 6)
    original = model(sc, edges, fc)
    payload = source_payload("mlp")
    payload["model"]["head.weight"].zero_()
    payload["model"]["head.bias"].zero_()
    model.sc_alpha_component = FrozenAlphaBranch(payload)
    actual = model(sc, edges, fc)
    torch.testing.assert_close(actual.fc_z_edges, original.fc_z_edges, atol=0, rtol=0)
    target = torch.randn_like(actual.fc_z_edges)
    p, y = loss_inputs(actual, target)
    torch.testing.assert_close((p-y).square().mean(), (actual.fc_z_edges-target).square().mean())


def test_builder_checkpoint_reload_and_tamper_guard(tmp_path):
    payload = source_payload("mlp")
    payload["config_sha256"] = config_sha256(payload["config"])
    component = tmp_path/"source/checkpoints/best.pt"
    component.parent.mkdir(parents=True)
    torch.save(payload, component)
    (component.parent.parent/"component_manifest.json").write_text(json.dumps(
        dict(schema=SCHEMA, sha256=file_sha256(component), config_sha256=payload["config_sha256"])))
    ae = FCAutoencoder(6, latent_dim=4, dropout=0)
    ae_path = tmp_path/"ae.pt"
    torch.save({"model": ae.state_dict()}, ae_path)
    stats = tmp_path/"stats.npz"
    np.savez(stats, group_template=payload["group_template"], context_template=payload["context_template"],
             fc_mean=np.zeros(6, np.float32), sc_mean=np.zeros(6, np.float32), sc_std=np.ones(6, np.float32))
    alpha_path = tmp_path/"alpha.npz"
    np.savez(alpha_path, alpha=payload["alpha_raw"])
    config = dict(paths=dict(root=str(tmp_path)), data=dict(n_nodes=4, warmup_windows=1),
        model=dict(fc_latent_dim=4, hidden_dim=4, dropout=0, sc_graph_layers=1, sc_graph_heads=2,
                   transformer_layers=1, transformer_heads=2, transformer_ffn_dim=8, tcn_dilations=[1],
                   sc_encoder="hcp_gcn", hcp_gcn_hidden_dim=5, hcp_gcn_output_dim=3,
                   output_head="direct_edge_linear", output_decomposition="e0032_decay_residual", fusion_type="concat_projection"),
        artifacts=dict(alpha_fit=dict(path=str(alpha_path), sha256=file_sha256(alpha_path)),
                       sc_alpha=dict(path=str(component), sha256=file_sha256(component))))
    model = build_sequence_model(config, 83, "transformer", stats, torch.device("cpu"), autoencoder_path=ae_path, ablation="fc1_only").eval()
    saved = dict(model=deepcopy(model.state_dict()), sc_alpha_sha256=file_sha256(component))
    rebuilt = build_sequence_model(config, 83, "transformer", stats, torch.device("cpu"), autoencoder_path=ae_path, checkpoint_payload=saved, ablation="fc1_only").eval()
    rebuilt.load_state_dict(saved["model"])
    sc, edges, fc = torch.rand(2, 4, 4), torch.rand(2, 6), torch.randn(2, 6)
    torch.testing.assert_close(model(sc, edges, fc).fc_z_edges, rebuilt(sc, edges, fc).fc_z_edges)
    saved["model"]["sc_alpha_component.generator.head.bias"][0] += 1
    with pytest.raises(ValueError, match="state changed"):
        attach_frozen_alpha(rebuilt, config, torch.device("cpu"), saved)


def test_fc_only_dataset_preserves_raw_sc_only_for_transfer(monkeypatch, tmp_path):
    import scdfc.data as data
    stats = tmp_path/"stats.npz"
    np.savez(stats, sc_mean=np.zeros(6), sc_std=np.ones(6))
    config = dict(paths=dict(root=str(tmp_path), split_csv="split.csv"), data=dict(warmup_windows=1, n_nodes=4))
    monkeypatch.setattr(data, "load_split", lambda p: pd.DataFrame(dict(subject_id=["s"], split=["train"])))
    monkeypatch.setattr(data, "iter_cached_samples", lambda *a: [("s", "LR")])
    monkeypatch.setattr(data, "load_sc", lambda *a: np.ones((4, 4), np.float32))
    monkeypatch.setattr(data, "read_cached", lambda *a: (np.ones((224, 6), np.float32), np.arange(224)*5))
    legacy = data.DFCSequenceDataset(config, 83, "train", stats, "fc1_only")[0]
    assert not legacy["sc_matrix"].any()
    config["artifacts"] = {"sc_alpha": {"path": "pinned.pt"}}
    transfer = data.DFCSequenceDataset(config, 83, "train", stats, "fc1_only")[0]
    assert transfer["sc_matrix"].all()
    torch.testing.assert_close(legacy["fc_warmup"], transfer["fc_warmup"])


@pytest.mark.parametrize("number", range(61, 67))
def test_six_configs_keep_ae_and_223_sources(number):
    config = load_config(next((ROOT/"configs/experiments").glob(f"E{number:04d}_*.yaml")))
    validate_experiment_config(config)
    assert config["artifacts"]["fc_autoencoder"]["id"] == "A0003"
    assert config["model"]["fc_representation"] == "autoencoder"
    assert config["model"]["alpha_parameters"] == 223
    assert config["training"]["epochs"] == 150
    config["model"]["sc_alpha_frozen"] = False
    with pytest.raises(ValueError, match="frozen"):
        validate_experiment_config(config)


def test_metrics_and_swaps_use_fixed_b0_and_subject_family():
    rng = np.random.default_rng(42)
    b0, true = rng.normal(size=(223, 6)), rng.normal(size=(223, 6))
    row = metric_row(b0, true, b0, np.zeros_like(b0), 17)
    assert row["dynamic_time_pearson"] is None and row["dynamic_std_ratio"] == 0
    subjects = ["a", "a", "b", "c"]
    donors = donor_indices(subjects, 42)
    assert all(subjects[i] != subjects[j] for i, j in enumerate(donors))
    improved = [dict(row, long_edge_mse=i) for i in (1, 1, 2, 3)]
    baseline = [dict(row, long_edge_mse=i+1) for i in (1, 1, 2, 3)]
    p = paired_rows(improved, baseline, subjects,
        dict(bootstrap_replicates=100, bootstrap_seed=42, primary_family_size=6), True)
    assert p["long_edge_mse"]["n_subjects"] == 3
    assert p["long_edge_mse"]["family_adjusted_ci"]["family_size"] == 6
    assert p["long_edge_mse"]["mean_difference"] == -1


@pytest.mark.parametrize("ablation", ["fc1_only", "full"])
def test_streaming_evaluation_covers_original_posthoc_and_separate_swap_paths(tmp_path, monkeypatch, ablation):
    import scdfc.alpha_transfer_evaluation as ev
    config = load_config(next((ROOT/"configs/experiments").glob("E0061_*.yaml")))
    config["paths"]["root"] = str(tmp_path)
    config["experiment"]["ablation"] = ablation
    config["comparison_plan"].update(expected_val_subjects=2, swap_seeds=[42, 43], bootstrap_replicates=100)
    model = residual_model(FrozenAlphaBranch(source_payload("linear")), ablation).eval()
    old = residual_model(ablation=ablation).eval()
    run = tmp_path/"run"
    (run/"checkpoints").mkdir(parents=True)
    checkpoint = run/"checkpoints/best.pt"
    checkpoint.write_bytes(b"synthetic-evaluation-only")
    meta = dict(status="COMPLETED", run_id="synthetic", experiment_id="E0061", config_sha256=config_sha256(config))
    (run/"metadata.json").write_text(json.dumps(meta))
    payload = dict(meta, sc_alpha_state_sha256=tensor_digest(model.sc_alpha_component), epoch=0, primary_metric="long_edge_mse")
    class SyntheticDataset(torch.utils.data.Dataset):
        samples = [("a", "LR"), ("b", "LR")]
        def __init__(self, *args):
            g = torch.Generator().manual_seed(4)
            self.items = [dict(sc_matrix=torch.rand(4, 4, generator=g), sc_edges=torch.rand(6, generator=g),
                               fc_warmup=torch.rand(6, generator=g), fc_future=torch.rand(223, 6, generator=g)) for _ in self.samples]
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return self.items[i]
    monkeypatch.setattr(ev, "verify_transfer_bindings", lambda cfg: (None, config))
    monkeypatch.setattr(ev, "verify_artifact", lambda cfg: checkpoint)
    monkeypatch.setattr(ev, "pinned_path", lambda cfg, key: checkpoint)
    monkeypatch.setattr(ev, "DFCSequenceDataset", SyntheticDataset)
    models = iter([(model, payload), (old, {})])
    monkeypatch.setattr(ev, "_load_model", lambda *a: next(models))
    path = ev.evaluate_transfer(config, run, "cpu")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["n_subjects"] == 2 and report["split"] == "val"
    assert set(report["aggregate"]) == {"model", "original", "posthoc", "e0032", "alpha_only"}
    assert set(report["sc_swap"]) == ({"alpha"} if ablation == "fc1_only" else {"alpha", "residual", "both"})
    assert report["paired_model_vs_original"]["long_edge_mse"]["family_adjusted_ci"]["family_size"] == 6
    assert (run/"private/transfer_examples_val.npz").exists()
