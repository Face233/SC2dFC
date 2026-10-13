from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scdfc.config import load_config
from scdfc.data import DFCSequenceDataset, precompute_dfc, read_cached
from scdfc.evaluation import _load_model
from scdfc.history_forecast import edge_correlations, forecast_metrics, paired_interval
from scdfc.managed_cli import command_run, command_evaluate_run, command_history_ridge
from scdfc.management import config_sha256, file_sha256, manifest_digest, validate_experiment_config
from scdfc.models.autoencoder import FCAutoencoder
from scdfc.models.history import FCHistoryGRU
from scdfc.history_summary import summarize_history_runs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def history_config(tmp_path):
    for name in ("sc", "timeseries", "manifests"):
        (tmp_path / "data" / name).mkdir(parents=True)
    subjects = [str(100001+i) for i in range(6)]
    rng = np.random.default_rng(9)
    for subject in subjects:
        np.savetxt(tmp_path / "data/sc" / f"{subject}.csv", np.ones((4, 4))-np.eye(4), delimiter=",")
        frame = pd.DataFrame(rng.normal(size=(1200, 4)), columns=list("ABCD"))
        frame.insert(0, "timepoint", np.arange(1200))
        frame.to_csv(tmp_path / "data/timeseries" / f"{subject}_AAL90_timeseries.csv", index=False)
    split = tmp_path / "data/manifests/split_v1.csv"
    pd.DataFrame({"subject_id": subjects, "split": ["train"]*3+["val"]*2+["test"]}).to_csv(split, index=False)
    manifest = {"dataset_version": "dataset_v1", "preprocessing_version": "preprocess_v1", "files": []}
    manifest["manifest_sha256"] = manifest_digest(manifest)
    (tmp_path / "data/manifests/dataset_v1.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifact = tmp_path / "ae.pt"
    torch.manual_seed(17)
    torch.save({"model": FCAutoencoder(6, 4, dropout=0).state_dict()}, artifact)
    config = load_config(ROOT / "configs/experiments/E0040_history_k1_full_forecast_v1.yaml")
    config["paths"].update(root=str(tmp_path), sc_dir="data/sc", timeseries={"LR": "data/timeseries"},
                           split_csv="data/manifests/split_v1.csv", cache_dir="data/cache", output_dir="outputs")
    config["data"].update(n_nodes=4, dataset_version="dataset_v1", preprocessing_version="preprocess_v1",
                          manifest_path="data/manifests/dataset_v1.json", manifest_sha256=manifest["manifest_sha256"],
                          split_version="split_v1", split_sha256=file_sha256(split))
    config["model"].update(hidden_dim=4, fc_latent_dim=4, dropout=0)
    config["experiment"].update(level=0)
    config["artifacts"]["fc_autoencoder"].update(path="ae.pt", sha256=file_sha256(artifact))
    config["training"].update(epochs=1, batch_size=2, patience=2, learning_rate_schedule={"name": "constant"})
    config["evaluation"]["bootstrap_replicates"] = 50
    config["baselines"]["history_ridge"].update(pca_components=2, cv_folds=3, alpha_grid=[1, 10])
    precompute_dfc(config, 83)
    return config


def test_common_cutoff_all_k_and_no_sc_loading(history_config, tmp_path, monkeypatch):
    from scdfc.data import fit_training_statistics
    import zarr
    fc, starts = read_cached(history_config, 83, "100001", "LR")
    cache = zarr.open_group(str(tmp_path / "data/cache/window_83.zarr"), mode="a")
    other = cache.require_group("subjects/100001/RL")
    other.create_dataset("fc_z", data=fc+10)
    other.create_dataset("window_starts", data=starts)
    stats = tmp_path / "stats.npz"
    fit_training_statistics(history_config, 83, stats)
    expected = np.mean([read_cached(history_config, 83, str(100001+i), "LR")[0][1:] for i in range(3)], axis=0)
    np.testing.assert_allclose(np.load(stats)["group_template"], expected, rtol=1e-6, atol=1e-6)
    monkeypatch.setattr("scdfc.data.load_sc", lambda *args: pytest.fail("FC-only sample loaded SC"))
    targets, last = [], []
    for k in (1, 5, 10, 20):
        config = deepcopy(history_config)
        config["data"]["warmup_windows"] = k
        dataset = DFCSequenceDataset(config, 83, "train", stats, "fc1_only")
        assert len(dataset) == 3 and all(run == "LR" for _, run in dataset.samples)
        fc, _ = read_cached(config, 83, *dataset.samples[0])
        sample = dataset[0]
        np.testing.assert_array_equal(sample["fc_warmup"], fc[20-k:20])
        np.testing.assert_array_equal(sample["fc_future"], fc[20:])
        assert sample["fc_future"].shape == (204, 6)
        targets.append(sample["fc_future"].numpy())
        last.append(sample["last_fc"].numpy())
    for value in targets[1:]:
        np.testing.assert_array_equal(value, targets[0])
    for value in last[1:]:
        np.testing.assert_array_equal(value, last[0])


def test_model_is_parameter_matched_frozen_and_residual_not_cumulative():
    states = []
    for k in (1, 5, 10, 20):
        torch.manual_seed(42)
        model = FCHistoryGRU(FCAutoencoder(6, 4, 0.5), torch.zeros(204, 6),
                             n_nodes=4, hidden_dim=4, warmup_windows=k)
        states.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        model.train()
        assert not model.fc_autoencoder.training
        history = torch.randn(2, k, 6)
        result = model(None, None, history)
        torch.testing.assert_close(result.fc_z_edges, history[:, -1, None]+result.dynamic_residual)
        result.fc_z_edges.square().mean().backward()
        assert all(p.grad is None for p in model.fc_autoencoder.parameters())
        assert model.queries.grad is not None
        assert model.history_gru.weight_ih_l0.grad is not None
        model.eval()
        first = model(torch.zeros(2, 4, 4), torch.zeros(2, 6), history)
        second = model(torch.ones(2, 4, 4)*999, torch.ones(2, 6)*999, history)
        torch.testing.assert_close(first.fc_z_edges, second.fc_z_edges)
        matrices = first.fc_matrices
        assert matrices.shape == (2, 204, 4, 4)
        torch.testing.assert_close(matrices, matrices.transpose(-1, -2))
        torch.testing.assert_close(torch.diagonal(matrices, dim1=-2, dim2=-1), torch.ones(2, 204, 4))
    for state in states[1:]:
        for name, parameter in state.items():
            torch.testing.assert_close(parameter, states[0][name], rtol=0, atol=0)


def test_correlations_axes_constants_and_subject_bootstrap():
    rng = np.random.default_rng(3)
    true = rng.normal(size=(204, 6))
    metrics = forecast_metrics(2*true+0.2, true, np.zeros_like(true), 17)
    assert metrics["forecast_space_r"] == pytest.approx(1)
    assert metrics["forecast_time_r"] == pytest.approx(1)
    constant = np.broadcast_to(true[0], true.shape)
    metrics = forecast_metrics(constant, true, np.zeros_like(true), 17)
    assert metrics["forecast_time_r"] is None
    assert metrics["forecast_time_valid_fraction"] == 0
    assert metrics["forecast_predicted_flat_fraction"] == 1
    first, valid, _, _ = edge_correlations(true, true)
    assert valid.all() and np.allclose(first, 1)
    # Subjects, not runs, get equal weight: subject A's mean=2, B's mean=10.
    result = paired_interval([1, 3, 10], [0, 0, 0], ["A", "A", "B"], replicates=100)
    assert result["mean_difference"] == pytest.approx(6)


def test_real_configs_are_identical_except_k_and_identity():
    configs = [validate_experiment_config(load_config(next((ROOT / "configs/experiments").glob(f"E00{i}_history*.yaml"))))
               for i in range(40, 44)]
    assert [config["data"]["warmup_windows"] for config in configs] == [1, 5, 10, 20]
    for config in configs:
        assert config["data"]["forecast_windows"] == 204
        assert config["data"]["forecast_origins"] == [19]
        assert config["training"]["seeds"] == [42, 43, 44]
        assert config["model"] == configs[0]["model"]
        assert config["training"] == configs[0]["training"]
    invalid = deepcopy(configs[0])
    invalid["data"]["forecast_windows"] = 40
    with pytest.raises(ValueError, match="remaining"):
        validate_experiment_config(invalid)
    invalid = deepcopy(configs[0])
    invalid["data"]["warmup_windows"] = 21
    with pytest.raises(ValueError, match="history"):
        validate_experiment_config(invalid)


def test_managed_history_train_reload_and_paired_evaluation(history_config, tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scdfc.management.git_metadata", lambda root: {
        "commit": "a"*40, "branch": "main", "remote": "", "dirty": False, "status": []})
    run_ids, protocols = [], []
    try:
        for number, k in enumerate((1, 5, 10, 20), 40):
            config = deepcopy(history_config)
            config["data"]["warmup_windows"] = k
            config["experiment"].update(id=f"E00{number}", name=f"history_k{k}_smoke")
            path = tmp_path / "configs/experiments" / f"E00{number}_history.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command_run(SimpleNamespace(experiment=str(path), seed=42, device="cpu"))
            run = next((tmp_path / f"outputs/E00{number}/runs").iterdir())
            run_ids.append(run.name)
            protocol = json.loads((run / "forecast_protocol.json").read_text())
            protocols.append(protocol)
            assert (run / "checkpoints/last.pt").exists()
            assert json.loads((run / "metadata.json").read_text())["status"] == "COMPLETED"
            checkpoint = run / "checkpoints/best.pt"
            frozen = load_config(run / "config_resolved.yaml")
            assert config_sha256(frozen) == json.loads((run / "metadata.json").read_text())["config_sha256"]
            stats = tmp_path / "outputs/shared/dataset_v1/window_83/history_training_stats.npz"
            model, payload = _load_model(frozen, 83, checkpoint, stats, torch.device("cpu"), tmp_path / "ae.pt")
            sample = DFCSequenceDataset(frozen, 83, "val", stats, "fc1_only")[0]
            restored = model(None, None, sample["fc_warmup"][None]).fc_z_edges
            assert restored.shape == (1, 204, 6)
            assert payload["validation_metrics"]["forecast_nonoverlap_mse"] >= 0
        assert len({p["initial_parameters_sha256"] for p in protocols}) == 1
        assert len({p["sample_manifest_sha256"] for p in protocols}) == 1
        assert len({p["trainable_parameters"] for p in protocols}) == 1
        for i, run_id in enumerate(run_ids):
            command_evaluate_run(SimpleNamespace(run_id=run_id, split="val", final_test=False,
                                                device="cpu", baseline_run_id=run_ids[0] if i else None,
                                                save_predictions=False))
        table = summarize_history_runs(tmp_path, run_ids)
        frame = pd.read_csv(table)
        assert frame["K"].tolist() == [1, 5, 10, 20]
        assert (frame["H"] == 204).all()
        assert frame["forecast_time_r_mean"].notna().all()
        assert (table.parent / "history_comparison.png").exists()
        report = json.loads((tmp_path / "outputs/E0043/runs" / run_ids[-1] / "evaluation_val.json").read_text())
        assert report["n_subjects"] == 2
        assert report["forecast_windows"] == 204
        assert report["aggregate"]["forecast_time_r"] is not None
        assert report["analytic_baselines"]["last_fc"]["aggregate"]["forecast_time_r"] is None
        assert report["paired_model_comparison"]["forecast_nonoverlap_mse"]["n_subjects"] == 2
        selection = (checkpoint.parent.parent / "metrics_best.json").read_bytes()
        command_history_ridge(SimpleNamespace(run_id=run_ids[-1], device="cpu"))
        ridge_dir = checkpoint.parent.parent / "baselines/history_ridge_v1"
        ridge_report = json.loads((ridge_dir / "evaluation_val.json").read_text())
        assert ridge_report["selection_split"] == "grouped_train_cv"
        assert ridge_report["selected_alpha"] in [1, 10]
        assert ridge_report["neural_minus_ridge"]["forecast_nonoverlap_mse"]["n_subjects"] == 2
        assert selection == (checkpoint.parent.parent / "metrics_best.json").read_bytes()
        from scdfc.history_ridge import predict_history_ridge
        ridge_payload = torch.load(ridge_dir / "ridge.pt", weights_only=False)
        encoded = model.encode_history(sample["fc_warmup"][None]).detach().numpy().reshape(1, -1)
        ridge_prediction = predict_history_ridge(ridge_payload, encoded, sample["last_fc"][None].numpy())
        assert ridge_prediction.shape == (1, 204, 6)
        replay_mse = float(np.mean((ridge_prediction[0, 17:]-sample["fc_future"].numpy()[17:])**2))
        assert replay_mse == pytest.approx(ridge_report["per_sample"][0]["forecast_nonoverlap_mse"], rel=1e-5)
        wrong = deepcopy(frozen)
        wrong["data"]["warmup_windows"] = 1
        with pytest.raises(ValueError, match="Checkpoint forecast protocol"):
            _load_model(wrong, 83, checkpoint, stats, torch.device("cpu"), tmp_path / "ae.pt")
    finally:
        torch.set_num_threads(old_threads)
