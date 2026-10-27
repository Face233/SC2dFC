"""PCA backend regressions; all data, models and runs here live in pytest tmp dirs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.decomposition import PCA
import torch
import yaml

from test_history_forecast import history_config
from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset, fit_training_statistics, read_cached
from scdfc.evaluation import _load_model, collect_sc_swapped_predictions
from scdfc.fc_pca import DATA_KEYS
from scdfc.management import file_sha256, verify_artifact
from scdfc.managed_cli import command_run, command_evaluate_run, _stats_path
from scdfc.pca_sequence import component_training_protocol, paired_metric, validate_pca_pair
from scdfc.training import build_sequence_model, seed_everything

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def pca_configs(history_config, tmp_path):
    config = load_config(next((ROOT/"configs/experiments").glob("E0050_*.yaml")))
    config["paths"] = deepcopy(history_config["paths"])
    config["data"] = {k: v for k, v in history_config["data"].items()
                      if k not in {"forecast_origins", "forecast_windows"}}
    config["data"]["warmup_windows"] = 1
    config["model"].update(fc_latent_dim=4, hidden_dim=4, transformer_layers=1,
                           transformer_heads=2, transformer_ffn_dim=16, dropout=.1,
                           hcp_gcn_hidden_dim=8, hcp_gcn_output_dim=4)
    config["training"].update(epochs=1, patience=2, batch_size=2, num_workers=0,
                              learning_rate_schedule={"name": "constant"})
    config["evaluation"].update(bootstrap_replicates=20, state_clusters=2)
    config["comparison_plan"].update(bootstrap_replicates=30)
    config["experiment"].update(level=0, execution_ready=True)
    alpha_path = tmp_path/"alpha.npz"
    np.savez(alpha_path, alpha=np.linspace(.8, .1, 223).astype(np.float32))
    config["artifacts"]["alpha_fit"] = {"id": "E0032-alpha-fit", "path": str(alpha_path),
                                           "sha256": file_sha256(alpha_path)}
    train = np.concatenate([read_cached(config, 83, str(s), "LR")[0] for s in (100001, 100002, 100003)])
    fitted = PCA(n_components=4, random_state=42).fit(train)
    archive = tmp_path/"outputs/components/A0044/pca.npz"
    archive.parent.mkdir(parents=True)
    np.savez(archive, components=fitted.components_, mean=fitted.mean_)
    manifest = tmp_path/"configs/artifacts/A0044.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "A0044", "type": "fc_pca", "format": "e0044_pca_npz/v1",
        "path": str(archive.relative_to(tmp_path)), "sha256": file_sha256(archive),
        "whiten": False, "representation": "fisher_z_upper_triangle_k1", "latent_dim": 4,
        "input_dim": 6, "data": {k: config["data"][k] for k in DATA_KEYS}}))
    config["artifacts"]["fc_pca"] = {
        "id": "A0044", "path": str(archive), "sha256": file_sha256(archive),
        "manifest_path": str(manifest), "manifest_sha256": file_sha256(manifest)}
    reference = deepcopy(config)
    reference["model"].pop("fc_representation")
    reference["comparison_plan"] = {}
    reference["experiment"].update(id="E0036", baseline="E0032")
    reference["artifacts"].pop("fc_pca")
    reference["artifacts"]["fc_autoencoder"] = deepcopy(history_config["artifacts"]["fc_autoencoder"])
    stats = _stats_path(config, 83)
    fit_training_statistics(reference, 83, stats)
    return config, reference, stats


def build(config, stats):
    seed_everything(42)
    model = build_sequence_model(config, 83, "transformer", stats, torch.device("cpu"),
                                 autoencoder_path=verify_artifact(config),
                                 ablation=config["experiment"]["ablation"])
    model.fc_autoencoder.requires_grad_(False)
    model.fc_autoencoder.eval()
    return model


def test_pca_and_ae_share_exact_trainable_initialization_and_rng(pca_configs, monkeypatch):
    config, reference, stats = pca_configs
    ae = build(reference, stats)
    ae_rng = torch.get_rng_state().clone()
    ae_state = {n: p.detach().clone() for n, p in ae.named_parameters() if p.requires_grad}
    monkeypatch.setattr("scdfc.training.load_autoencoder", lambda *a: pytest.fail("PCA read AE weights"))
    model = build(config, stats)
    assert torch.equal(ae_rng, torch.get_rng_state())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, ae_state[name], rtol=0, atol=0)
    monkeypatch.setattr(model.fc_autoencoder, "decode", lambda *a: pytest.fail("PCA decoder was used"))
    x = torch.randn(2, 6)
    output = model(torch.zeros(2, 4, 4), torch.zeros(2, 6), x)
    expected_base = model.group_template[None]+model.decay_alpha[None, :, None]*(x-model.context_template)[:, None]
    torch.testing.assert_close(output.residual_baseline, expected_base)
    output.fc_z_edges.square().mean().backward()
    assert model.direct_edge_head.weight.grad is not None
    assert not list(model.fc_autoencoder.parameters())


def test_pca_checkpoint_rejects_overridden_basis_provenance_and_baseline(pca_configs, tmp_path):
    config, _, stats = pca_configs
    model = build(config, stats)
    protocol = component_training_protocol(config, model, [("100001", "LR")], torch.device("cpu"))
    payload = {"model": model.state_dict(), "decoder_type": "transformer", "sc_encoder_type": "hcp_gcn",
               "ablation": "fc1_only", "output_decomposition": "e0032_decay_residual",
               "alpha_fit_sha256": config["artifacts"]["alpha_fit"]["sha256"], **protocol}
    checkpoint = tmp_path/"best.pt"
    torch.save(payload, checkpoint)
    recovered, _ = _load_model(config, 83, checkpoint, stats, torch.device("cpu"), verify_artifact(config))
    torch.testing.assert_close(recovered.fc_autoencoder.components, model.fc_autoencoder.components)
    for key in ("fc_autoencoder.mean", "condition_encoder.fc_autoencoder.components", "decay_alpha", "group_template"):
        bad = deepcopy(payload)
        bad["model"][key] = bad["model"][key]+1
        torch.save(bad, checkpoint)
        with pytest.raises(ValueError, match="buffer differs"):
            _load_model(config, 83, checkpoint, stats, torch.device("cpu"))
    bad = deepcopy(payload)
    bad["artifact_sha256"] = "0"*64
    torch.save(bad, checkpoint)
    with pytest.raises(ValueError, match="provenance mismatch"):
        _load_model(config, 83, checkpoint, stats, torch.device("cpu"))
    config["artifacts"]["fc_pca"]["manifest_sha256"] = "0"*64
    with pytest.raises(ValueError, match="manifest checksum"):
        verify_artifact(config)


def test_subject_bootstrap_uses_subject_means_and_adjusted_primary_intervals():
    result = paired_metric([1, 3, 10], [0, 0, 0], ["A", "A", "B"], replicates=200,
                           seed=20261026, family_size=2)
    assert result["mean_difference"] == 6
    assert result["n_subjects"] == 2 and result["seed"] == 20261026
    assert result["family_adjusted_ci"]["level"] == .975
    assert result["family_adjusted_ci"]["ci_low"] <= result["ci_low"]
    assert result["family_adjusted_ci"]["ci_high"] >= result["ci_high"]
    assert paired_metric([np.nan], [0], ["A"], replicates=20, seed=42) is None


def test_seeded_sc_swap_has_no_self_donors_and_keeps_fc_fixed():
    class Samples:
        samples = [("A", "LR"), ("B", "LR"), ("C", "LR")]
        def __len__(self):
            return 3
        def __getitem__(self, i):
            return {"sc_matrix": torch.tensor([[float(i)]]), "sc_edges": torch.tensor([float(i)]),
                    "fc_warmup": torch.tensor([float(i)]), "fc_future": torch.zeros(2, 1)}
    class Echo(torch.nn.Module):
        def forward(self, sc, edges, fc, steps):
            assert torch.all(edges[:, 0] != fc[:, 0])
            # The result exposes both donor SC and unchanged recipient FC.
            return SimpleNamespace(fc_z_edges=torch.stack((edges[:, 0], fc[:, 0]), 1)[:, :, None])
    first = collect_sc_swapped_predictions(Echo(), Samples(), torch.device("cpu"), 2, seed=20261026)
    second = collect_sc_swapped_predictions(Echo(), Samples(), torch.device("cpu"), 2, seed=20261026)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 1, 0], np.arange(3))


def test_pca_dataset_ignores_rl_and_rejects_missing_lr(pca_configs, tmp_path):
    import zarr
    config, _, stats = pca_configs
    store = zarr.open_group(str(tmp_path/"data/cache/window_83.zarr"), mode="a")
    fc, starts = read_cached(config, 83, "100001", "LR")
    rl = store.require_group("subjects/100001/RL")
    rl.create_dataset("fc_z", data=fc+100)
    rl.create_dataset("window_starts", data=starts)
    assert len(DFCSequenceDataset(config, 83, "train", stats)) == 3
    del store["subjects/100001/LR"]
    with pytest.raises(ValueError, match="one complete LR"):
        DFCSequenceDataset(config, 83, "train", stats)


def test_synthetic_managed_pca_train_reload_and_independent_ae_sc_comparisons(pca_configs, tmp_path, monkeypatch):
    config, reference, stats = pca_configs
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scdfc.management.git_metadata", lambda *a: {
        "commit": "a"*40, "branch": "test", "remote": "", "dirty": False, "status": []})
    runs = {}
    for number, template, ablation in ((36, reference, "fc1_only"), (37, reference, "full"),
                                        (50, config, "fc1_only"), (51, config, "full")):
        cfg = deepcopy(template)
        exp = f"E{number:04d}"
        cfg["experiment"].update(id=exp, ablation=ablation)
        if number == 51:
            cfg["comparison_plan"].update(ae_reference="E0037", sc_reference="E0050")
        path = tmp_path/f"configs/experiments/{exp}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        command_run(SimpleNamespace(experiment=str(path), seed=42, device="cpu"))
        run = next((tmp_path/f"outputs/{exp}/runs").iterdir())
        assert json.loads((run/"metadata.json").read_text())["status"] == "COMPLETED"
        runs[number] = run
        if number >= 50:
            assert (run/"checkpoints/last.pt").exists()
            protocol = json.loads((run/"component_protocol.json").read_text())
            assert protocol["fc_representation"] == "frozen_pca"
            assert protocol["initialization_implementation"] == "historical_ae_constructor_rng/v1"
    for number, baseline in ((50, 36), (51, 37), (51, 50)):
        command_evaluate_run(SimpleNamespace(run_id=runs[number].name, baseline_run_id=runs[baseline].name,
                                             split="val", final_test=False, device="cpu"))
        report = json.loads((runs[number]/"evaluation_val.json").read_text())
        paired = report["paired_model_comparison"]
        assert paired["baseline_component"] == ("autoencoder" if baseline < 50 else "frozen_pca")
        assert paired["long_edge_mse"]["seed"] == 20261026
        assert paired["long_edge_mse"]["family_adjusted_ci"]["family_size"] == (2 if baseline < 50 else 1)
        assert "template_adjusted_time_pearson" in paired
        assert "dynamic_difference_time_pearson" in paired
        assert (runs[number]/f"comparison_E{baseline:04d}_val.json").exists()
        if number == 51:
            assert report["sc_swap_diagnostic"]["swap_seed"] == 20261026
            assert report["sc_swap_diagnostic"]["matched_minus_swapped_long_mse"]["seed"] == 20261026
    for run in (runs[50], runs[51]):
        payload = torch.load(run/"checkpoints/best.pt", map_location="cpu", weights_only=False)
        model, _ = _load_model(load_config(run/"config_resolved.yaml"), 83,
                                run/"checkpoints/best.pt", stats, torch.device("cpu"))
        assert model.fc_autoencoder.latent_dim == 4 and payload["fc_component_id"] == "A0044"


def test_declared_comparison_rejects_wrong_seed_or_sc_condition(pca_configs):
    config, reference, _ = pca_configs
    metadata = {"seed": 42}
    baseline = {"experiment_id": "E0036", "status": "COMPLETED", "seed": 42}
    validate_pca_pair(config, reference, metadata, baseline)
    baseline["seed"] = 43
    with pytest.raises(ValueError, match="matching seed"):
        validate_pca_pair(config, reference, metadata, baseline)
    baseline["seed"] = 42
    reference["experiment"]["ablation"] = "full"
    with pytest.raises(ValueError, match="SC condition"):
        validate_pca_pair(config, reference, metadata, baseline)
