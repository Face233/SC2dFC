"""Synthetic preparation/inference checks; no neural training is launched here."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from test_history_forecast import history_config  # shared tiny private-cache fixture
from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset, read_cached
from scdfc.evaluation import _load_model
from scdfc.history_forecast import paired_interval
from scdfc.history_persistence import (fit_history_persistence, load_persistence_artifact,
                                       bind_history_persistence, MODEL_NAME)
from scdfc.history_summary import summarize_history_runs
from scdfc.managed_cli import command_evaluate_run, command_run, _ensure_stats
from scdfc.management import (config_sha256, file_sha256, register_experiment,
                               validate_experiment_config)
from scdfc.metric_records import sequence_selection_definition
from scdfc.models.autoencoder import FCAutoencoder
from scdfc.models.history_transformer import FCHistoryPersistenceTransformer
from scdfc.training import build_sequence_model, loss_inputs, sequence_criterion, validate_sequence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def persistence_config(history_config, tmp_path):
    config = load_config(ROOT / "configs/experiments/E0045_history_k1_transformer_persistence_v1.yaml")
    config["paths"] = deepcopy(history_config["paths"])
    config["data"] = deepcopy(history_config["data"])
    config["data"].update(forecast_origins=[9], forecast_windows=214)
    config["model"].update(hidden_dim=4, fc_latent_dim=4, transformer_layers=1,
                           transformer_heads=2, transformer_ffn_dim=8, dropout=0)
    config["training"].update(epochs=1, patience=2, batch_size=2)
    config["evaluation"]["bootstrap_replicates"] = 20
    config["experiment"]["level"] = 0
    config["artifacts"]["fc_autoencoder"] = deepcopy(history_config["artifacts"]["fc_autoencoder"])
    config["artifacts"]["history_persistence"].update(path="outputs/shared/persistence.npz", sha256=None)
    return config


def prepare(config):
    path = Path(config["paths"]["root"]) / config["artifacts"]["history_persistence"]["path"]
    fit_history_persistence(config, path)
    config["artifacts"]["history_persistence"]["sha256"] = file_sha256(path)
    return path


def test_real_configs_and_pending_run_fail_before_creating_outputs(tmp_path):
    configs = [validate_experiment_config(load_config(next((ROOT / "configs/experiments").glob(f"E00{i}_*.yaml"))))
               for i in (45, 46, 47)]
    assert [c["data"]["warmup_windows"] for c in configs] == [1, 5, 10]
    for c in configs:
        assert c["data"]["forecast_origins"] == [9] and c["data"]["forecast_windows"] == 214
        assert c["model"] == configs[0]["model"] and c["training"] == configs[0]["training"]
        assert c["artifacts"] == configs[0]["artifacts"]
        artifact_hash = c["artifacts"]["history_persistence"]["sha256"]
        assert artifact_hash is None or (len(artifact_hash) == 64 and all(x in "0123456789abcdef" for x in artifact_hash))
        assert c["evaluation"]["primary_metric"] == "forecast_mse"
    for changes in ({"data": {"warmup_windows": 11}}, {"data": {"forecast_windows": 204}},
                    {"model": {"sc_encoder": "hcp_gcn"}},
                    {"evaluation": {"primary_metric": "forecast_nonoverlap_mse"}}):
        bad = deepcopy(configs[0])
        for section, values in changes.items():
            bad[section].update(values)
        with pytest.raises(ValueError):
            validate_experiment_config(bad)
    draft = deepcopy(configs[0])
    draft["artifacts"]["history_persistence"]["sha256"] = None
    draft_path = tmp_path / "unbound.yaml"
    draft_path.write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        command_run(SimpleNamespace(experiment=str(draft_path), seed=42, device="cpu"))


def test_fit_train_only_lr_alignment_shared_alpha_and_no_sc(persistence_config, tmp_path, monkeypatch):
    import scdfc.history_persistence as module
    config = persistence_config
    original = module.read_cached
    def training_only(c, window, subject, run):
        assert subject in {"100001", "100002", "100003"} and run == "LR"
        return original(c, window, subject, run)
    monkeypatch.setattr(module, "read_cached", training_only)
    monkeypatch.setattr("scdfc.data.load_sc", lambda *args: pytest.fail("No SC in persistence fitting"))
    path = prepare(config)
    artifact = load_persistence_artifact(config)
    train = np.stack([original(config, 83, str(100001+i), "LR")[0] for i in range(3)]).astype(np.float64)
    mean = train.mean(0)
    offset = train[:, 9]-mean[9]
    raw = np.sum(offset[:, None]*(train[:, 10:]-mean[10:]), axis=(0, 2)) / (np.sum(offset**2)+1e-12)
    np.testing.assert_allclose(artifact["context_template"], mean[9], atol=1e-6)
    np.testing.assert_allclose(artifact["group_template"][9:], mean[10:], atol=1e-6)
    np.testing.assert_allclose(artifact["alpha_raw"], raw, atol=1e-10)
    np.testing.assert_allclose(artifact["alpha"], raw.clip(0, 1), atol=1e-7)
    assert _ensure_stats(config, 83) == path
    targets, anchors = [], []
    for k in (1, 5, 10):
        candidate = deepcopy(config)
        candidate["data"]["warmup_windows"] = k
        load_persistence_artifact(candidate)
        dataset = DFCSequenceDataset(candidate, 83, "val", path, "fc1_only")
        sample = dataset[0]
        fc, _ = original(config, 83, *dataset.samples[0])
        np.testing.assert_array_equal(sample["fc_warmup"], fc[10-k:10])
        np.testing.assert_array_equal(sample["fc_future"], fc[10:])
        targets.append(sample["fc_future"])
        anchors.append(sample["last_fc"])
    assert all(torch.equal(targets[0], t) for t in targets)
    assert all(torch.equal(anchors[0], a) for a in anchors)
    with pytest.raises(FileExistsError):
        fit_history_persistence(config, path)
    bad = deepcopy(config)
    bad["data"].update(forecast_origins=[19], forecast_windows=204)
    with pytest.raises(ValueError, match="protocol"):
        load_persistence_artifact(bad)
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_persistence_artifact(config)


def test_zero_residual_parameter_matching_gradients_and_future_interaction():
    states = []
    for k in (1, 5, 10):
        torch.manual_seed(42)
        template = torch.randn(214, 6)
        context = torch.randn(6)
        alpha = torch.linspace(1, 0, 214)
        model = FCHistoryPersistenceTransformer(FCAutoencoder(6, 4, 0.5), template, context, alpha,
                    n_nodes=4, hidden_dim=4, dropout=0, warmup_windows=k,
                    transformer_layers=1, transformer_heads=2, transformer_ffn_dim=8)
        states.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        model.train()
        assert not model.fc_autoencoder.training
        history = torch.randn(2, k, 6)
        output = model(None, None, history)
        expected = template[None]+alpha[None, :, None]*(history[:, -1]-context)[:, None]
        torch.testing.assert_close(output.residual_baseline, expected)
        torch.testing.assert_close(output.fc_z_edges, expected+output.dynamic_residual)
        target = torch.randn_like(output.fc_z_edges)
        pred_residual, true_residual = loss_inputs(output, target)
        torch.testing.assert_close((pred_residual-true_residual).square().mean(), (output.fc_z_edges-target).square().mean())
        output.fc_z_edges[:, 0].square().mean().backward()
        assert all(p.grad is None for p in model.fc_autoencoder.parameters())
        assert model.history_gru.weight_ih_l0.grad is not None
        # A first-horizon loss reaches later query tokens via future self-attention.
        assert model.temporal.queries.grad[1:].abs().sum() > 0
        model.eval()
        torch.testing.assert_close(model(None, None, history).fc_z_edges,
                                   model(torch.randn(2, 4, 4), torch.randn(2, 6), history).fc_z_edges)
        with torch.no_grad():
            model.direct_edge_head.weight.zero_()
            model.direct_edge_head.bias.zero_()
        torch.testing.assert_close(model(None, None, history).fc_z_edges, expected)
        with pytest.raises(ValueError, match="length"):
            model(None, None, history, steps=204)
    for state in states[1:]:
        for name, value in state.items():
            torch.testing.assert_close(value, states[0][name], atol=0, rtol=0)


def test_explicit_binding_updates_only_planned_configs(persistence_config, tmp_path):
    config = persistence_config
    artifact = prepare(config)
    config["artifacts"]["history_persistence"]["sha256"] = None
    paths = []
    for number, k in ((45, 1), (46, 5), (47, 10)):
        candidate = deepcopy(config)
        candidate["data"]["warmup_windows"] = k
        candidate["experiment"].update(id=f"E00{number}", name=f"synthetic_k{k}")
        path = tmp_path / "configs/experiments" / f"E00{number}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        register_experiment(path)
        paths.append(path)
    result = bind_history_persistence(artifact, paths)
    assert result["experiments"] == ["E0045", "E0046", "E0047"]
    rows = list(csv.DictReader((tmp_path / "reports/experiment_registry.csv").open(encoding="utf-8-sig")))
    for path, row in zip(paths, rows):
        bound = load_config(path)
        assert row["config_sha256"] == config_sha256(bound)
        assert row["status"] == "PLANNED"
        load_persistence_artifact(bound)
    run = tmp_path / "outputs/E0045/runs/existing/metadata.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    previous = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError, match="unrun"):
        bind_history_persistence(artifact, paths)
    assert [p.read_bytes() for p in paths] == previous


def test_inference_checkpoint_reload_full_selection_evaluation_and_summary(persistence_config, tmp_path, monkeypatch):
    # Build random models and synthetic checkpoints without optimizer steps.
    config = persistence_config
    stats = prepare(config)
    monkeypatch.chdir(tmp_path)
    run_ids = []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for number, k in ((45, 1), (46, 5), (47, 10)):
            candidate = deepcopy(config)
            candidate["data"]["warmup_windows"] = k
            candidate["experiment"].update(id=f"E00{number}", name=f"synthetic_k{k}")
            torch.manual_seed(42)
            model = build_sequence_model(candidate, 83, MODEL_NAME, stats, torch.device("cpu"),
                         "none", tmp_path / "ae.pt", ablation="fc1_only")
            dataset = DFCSequenceDataset(candidate, 83, "val", stats, "fc1_only")
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            criterion = sequence_criterion(candidate, 17)
            metrics = validate_sequence(model, loader, criterion, 17, torch.device("cpu"))
            assert metrics["forecast_mse"] == pytest.approx(metrics["objective_loss"], rel=1e-5)
            assert "full_mse" in sequence_selection_definition(candidate, 17)["implementation"]
            run_id = f"E00{number}-synthetic-no-training"
            run_ids.append(run_id)
            run = tmp_path / f"outputs/E00{number}/runs" / run_id
            (run / "checkpoints").mkdir(parents=True)
            digest = config_sha256(candidate)
            metadata = {"experiment_id": f"E00{number}", "run_id": run_id, "config_sha256": digest,
                        "level": 0, "status": "COMPLETED", "seed": 42, "git": {"commit": "synthetic"}}
            (run / "metadata.json").write_text(json.dumps(metadata))
            (run / "config_resolved.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
            payload = {**metadata, "model": model.state_dict(), "decoder_type": MODEL_NAME,
                       "sc_encoder_type": "none", "ablation": "fc1_only", "epoch": 0,
                       "validation_metrics": metrics, "output_decomposition": "history_decay_residual",
                       "persistence_sha256": candidate["artifacts"]["history_persistence"]["sha256"],
                       **{key: candidate["data"][key] for key in ("warmup_windows", "forecast_windows", "forecast_origins")}}
            checkpoint = run / "checkpoints/best.pt"
            torch.save(payload, checkpoint)
            loaded, _ = _load_model(candidate, 83, checkpoint, stats, torch.device("cpu"), tmp_path / "ae.pt")
            history = next(iter(loader))["fc_warmup"]
            model.eval()
            torch.testing.assert_close(model(None, None, history).fc_z_edges, loaded(None, None, history).fc_z_edges)
            command_evaluate_run(SimpleNamespace(run_id=run_id, split="val", final_test=False, device="cpu",
                                 baseline_run_id=run_ids[0] if k > 1 else None, save_predictions=False))
            report = json.loads((run / "evaluation_val.json").read_text())
            assert report["forecast_windows"] == 214
            assert report["segments"]["late"]["stop"] == 214
            assert "persistence" in report["analytic_baselines"]
            assert "forecast_persistence_residual_time_r" in report["aggregate"]
            assert (run / "examples_val/forecast_examples.png").is_file()
            if k > 1:
                paired = report["paired_model_comparison"]["forecast_mse"]
                assert paired["seed"] == 20261021 and "simultaneous_97_5_ci" in paired
        table = summarize_history_runs(tmp_path, run_ids)
        assert table.is_file() and table.parent.name == "history_transformer_persistence"
        rows = list(csv.DictReader(table.open(encoding="utf-8-sig")))
        assert [int(r["K"]) for r in rows] == [1, 5, 10]
        assert all(int(r["H"]) == 214 for r in rows)
        first_checkpoint = tmp_path / f"outputs/E0045/runs/{run_ids[0]}/checkpoints/best.pt"
        corrupted = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
        corrupted["model"]["decay_alpha"] = corrupted["model"]["decay_alpha"]+0.01
        corrupt_path = tmp_path / "corrupt.pt"
        torch.save(corrupted, corrupt_path)
        with pytest.raises(ValueError, match="frozen persistence buffer"):
            _load_model(config, 83, corrupt_path, stats, torch.device("cpu"), tmp_path / "ae.pt")
        bad = deepcopy(config)
        bad["artifacts"]["history_persistence"]["sha256"] = "0"*64
        with pytest.raises(ValueError, match="checksum"):
            _load_model(bad, 83, tmp_path / f"outputs/E0045/runs/{run_ids[0]}/checkpoints/best.pt",
                        stats, torch.device("cpu"), tmp_path / "ae.pt")
    finally:
        torch.set_num_threads(old_threads)
