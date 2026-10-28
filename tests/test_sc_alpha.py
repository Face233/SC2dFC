"""Synthetic checks only: no formal participants, model runs or validation access."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
import yaml

from scdfc.config import load_config
from scdfc.management import config_sha256, file_sha256, validate_experiment_config
from scdfc.models.sc_alpha import SCAlphaGenerator, alpha_basis, sufficient_mse
from scdfc.sc_alpha import (AlphaCohort, fit_templates, fit_sc_transform, transform_sc,
    statistics, fit_one, run_sc_alpha, load_sc_alpha_component, tensor_digest, write_json)

ROOT = Path(__file__).resolve().parents[1]


def config_path(number):
    return next((ROOT/"configs/experiments").glob(f"E{number:04d}_*.yaml"))


@pytest.mark.parametrize("encoder", ["linear", "mlp", "gcn"])
def test_head_width_preserves_encoder_initialization_and_training_rng(encoder):
    states, rngs = [], []
    for k in (1, 6, 223):
        torch.manual_seed(42)
        model = SCAlphaGenerator(encoder, k, 223, n_nodes=4, pca_components=2,
                                mlp_hidden=3, gcn_hidden=5, gcn_output=3)
        states.append(tensor_digest(model.encoder)); rngs.append(torch.get_rng_state())
        x = torch.ones(2, 4, 4) if encoder == "gcn" else torch.ones(2, 2)
        assert torch.equal(model(x), torch.zeros(2, 223))
    assert len(set(states)) == 1
    assert all(torch.equal(rngs[0], r) for r in rngs)


def test_bases_and_unbounded_gradient_on_all_223_outputs():
    one, six, direct = (alpha_basis(223, k) for k in (1, 6, 223))
    assert one.shape == (223, 1) and np.all(one == 1)
    assert six.shape == (223, 6) and np.linalg.matrix_rank(six) == 6
    np.testing.assert_allclose(six.sum(1), 1, atol=2e-7)
    np.testing.assert_array_equal(direct, np.eye(223))
    model = SCAlphaGenerator("linear", 223, 223, pca_components=2)
    with torch.no_grad():
        model.head.bias.copy_(torch.linspace(-2, 2, 223))
    output = model(torch.zeros(2, 2))
    assert output.min() < 0 and output.max() > 1
    (output-3).square().mean().backward()
    assert torch.all(model.head.bias.grad != 0)  # Including the first 17 future windows.


def test_sufficient_statistics_match_full_edge_error():
    rng = np.random.default_rng(42)
    cohort = AlphaCohort([str(i) for i in range(4)], np.zeros((4, 4, 4)),
                         rng.normal(size=(4, 224, 6)).astype(np.float32), "synthetic")
    context = rng.normal(size=6)
    template = rng.normal(size=(223, 6))
    q, u, v = statistics(cohort, context, template)
    alpha = rng.normal(size=(4, 223))
    pred = template[None]+alpha[:, :, None]*(cohort.fc[:, 0].astype(float)-context)[:, None]
    expected = ((pred-cohort.fc[:, 1:])**2).mean(-1)
    actual = sufficient_mse(torch.tensor(alpha), torch.tensor(q), torch.tensor(u), torch.tensor(v)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fold_templates_and_sc_transform_ignore_heldout_data():
    rng = np.random.default_rng(2)
    fc = rng.normal(size=(8, 224, 6)).astype(np.float32)
    sc = rng.uniform(.1, 2, size=(8, 4, 4)).astype(np.float32)
    sc = (sc+sc.transpose(0, 2, 1))/2
    cohort = AlphaCohort([str(i) for i in range(8)], sc, fc, "synthetic")
    cfg = load_config(config_path(52)); cfg["model"]["pca_components"] = 2
    indices = np.arange(5)
    first = fit_templates(cohort, indices)
    transform = fit_sc_transform(sc, indices, cfg)
    fc[5:] += 1000; sc[5:] *= 1000
    second = fit_templates(cohort, indices)
    next_transform = fit_sc_transform(sc, indices, cfg)
    for a, b in zip(first, second): np.testing.assert_array_equal(a, b)
    for key in ("mean", "scale", "components", "pca_mean"):
        np.testing.assert_array_equal(transform[key], next_transform[key])


def synthetic_config(tmp_path, encoder="linear", k=6):
    cfg = load_config(config_path(52))
    cfg["paths"]["root"] = str(tmp_path)
    cfg["experiment"]["level"] = 0
    cfg["data"]["n_nodes"] = 4
    cfg["model"].update(sc_encoder=encoder, alpha_parameters=k, pca_components=2,
                        mlp_hidden=3, gcn_hidden=5, gcn_output=3)
    cfg["training"].update(epochs=2, patience=1, cv_folds=2, batch_size=4,
                          alpha_lambda_grid=[0., .1], alpha_expected_subjects={"train": 12, "val": 4})
    cfg["comparison_plan"].update(require_suite_ready=False, bootstrap_replicates=40,
                                  secondary_bootstrap_replicates=40, swap_seeds=[1, 2])
    return cfg


def make_cohorts():
    rng = np.random.default_rng(4)
    def make(n, prefix):
        sc = rng.uniform(.1, 2, size=(n, 4, 4)).astype(np.float32)
        sc = (sc+sc.transpose(0, 2, 1))/2
        first = rng.normal(size=(n, 6)).astype(np.float32)
        alpha = .3+.05*sc.mean((1, 2))[:, None]+.02*np.sin(np.arange(223)/30)[None]
        future = alpha[:, :, None]*first[:, None]+rng.normal(scale=.01, size=(n, 223, 6))
        fc = np.concatenate([first[:, None], future], axis=1).astype(np.float32)
        return AlphaCohort([f"{prefix}{i:02d}" for i in range(n)], sc, fc, prefix)
    return {"train": make(12, "fit"), "val": make(4, "hold")}


def bind_references(cfg, cohorts, tmp_path):
    context, template, alpha = fit_templates(cohorts["train"], np.arange(12))
    stats_path, alpha_path = tmp_path/"stats.npz", tmp_path/"alpha.npz"
    np.savez(stats_path, context_template=context, group_template=template)
    np.savez(alpha_path, alpha_raw=alpha, alpha=np.clip(alpha, 0, 1))
    cfg["artifacts"] = {"alpha_fit": {"path": "alpha.npz", "sha256": file_sha256(alpha_path)},
                        "training_stats": {"path": "stats.npz", "sha256": file_sha256(stats_path)}}


@pytest.mark.parametrize("encoder", ["linear", "mlp", "gcn"])
@pytest.mark.parametrize("parameters", [1, 6, 223])
def test_synthetic_train_save_reload_and_validation(tmp_path, monkeypatch, encoder, parameters):
    from scdfc import sc_alpha_evaluation as ev
    cfg = synthetic_config(tmp_path, encoder, parameters)
    cohorts = make_cohorts(); bind_references(cfg, cohorts, tmp_path)
    validate_experiment_config(cfg)
    accesses = []
    def load(config, split):
        accesses.append(split)
        return cohorts[split]
    monkeypatch.setattr("scdfc.sc_alpha.load_cohort", load)
    run_dir = tmp_path/"outputs/E0052/runs/synthetic"; (run_dir/"checkpoints").mkdir(parents=True)
    metadata = {"experiment_id": "E0052", "run_id": "synthetic", "config_sha256": config_sha256(cfg),
                "status": "COMPLETED", "seed": 42, "level": 0}
    write_json(run_dir/"metadata.json", metadata)
    (run_dir/"config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint = run_sc_alpha(cfg, run_dir, metadata, "cpu")
    assert accesses == ["train"]
    record = json.loads((run_dir/"metrics_best.json").read_text())
    assert record["split"] == "train_cv" and record["best_epoch"] is None
    selection_before = file_sha256(run_dir/"metrics_best.json")
    first = load_sc_alpha_component(checkpoint, cfg)
    second = load_sc_alpha_component(checkpoint)
    a = first.predict_alpha(cohorts["val"].sc)
    np.testing.assert_array_equal(a, second.predict_alpha(cohorts["val"].sc))
    pred = first.predict(cohorts["val"].sc, cohorts["val"].fc[:, 0])
    assert pred.shape == (4, 223, 6)
    monkeypatch.setattr(ev, "load_cohort", load)
    monkeypatch.setattr(ev, "verify_data_bindings", lambda c: {})
    path = ev.evaluate_sc_alpha_run(cfg, run_dir)
    report = json.loads(path.read_text())
    assert accesses == ["train", "val"]
    assert report["test_data_read"] is False and "per_sample" not in report
    assert report["baseline_aggregate"]["dynamic_time_pearson"] is None
    assert report["baseline_aggregate"]["dynamic_valid_fraction"] == 0
    assert file_sha256(run_dir/"metrics_best.json") == selection_before
    assert len(list((tmp_path/"outputs/E0052/visual").glob("*.png"))) == 5
    assert report["sc_swap"]["no_self_pairs"]
    checkpoint.write_bytes(checkpoint.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_sc_alpha_component(checkpoint)


def test_simple_sc_signal_reduces_heldout_error(tmp_path):
    cfg = synthetic_config(tmp_path, "linear", 1)
    cfg["training"].update(epochs=150, patience=150, learning_rate=.03, weight_decay=0.)
    cfg["model"]["pca_components"] = 1
    x = np.linspace(-1, 1, 12).astype(np.float32)[:, None]
    target = np.broadcast_to(.2+.4*x, (12, 223)).copy()
    stat = (np.ones(12), target, target**2)
    model, record = fit_one(cfg, x, stat, np.full(223, .2), np.arange(8), np.arange(8, 12), 0., torch.device("cpu"))
    assert record["best_score"] < .01
    assert record["best_score"] < float(np.mean((target[8:]-.2)**2))*.2


def test_nonfinite_gradient_is_reported_as_a_failed_fit(tmp_path, monkeypatch):
    from scdfc.sc_alpha import generator as make_generator
    cfg = synthetic_config(tmp_path, "linear", 1)
    def bad_gradient(config):
        model = make_generator(config)
        model.head.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        return model
    monkeypatch.setattr("scdfc.sc_alpha.generator", bad_gradient)
    features = np.ones((12, 2), dtype=np.float32)
    stat = (np.ones(12), np.zeros((12, 223)), np.ones((12, 223)))
    with pytest.raises(FloatingPointError, match="gradient"):
        fit_one(cfg, features, stat, np.full(223, .2), np.arange(8), np.arange(8, 12),
                0., torch.device("cpu"))


def test_failed_lambda_does_not_abort_remaining_cv_candidates(tmp_path, monkeypatch):
    cfg = synthetic_config(tmp_path, "linear", 1)
    cohorts = make_cohorts(); bind_references(cfg, cohorts, tmp_path)
    monkeypatch.setattr("scdfc.sc_alpha.load_cohort", lambda c, split: cohorts[split])
    def fail_one(config, features, stat, alpha, fitting, heldout, penalty, device, **kwargs):
        if penalty == 0:
            raise FloatingPointError("synthetic nonfinite gradient")
        return fit_one(config, features, stat, alpha, fitting, heldout, penalty, device, **kwargs)
    monkeypatch.setattr("scdfc.sc_alpha.fit_one", fail_one)
    run_dir = tmp_path/"run"; (run_dir/"checkpoints").mkdir(parents=True)
    metadata = {"experiment_id": "E0052", "run_id": "synthetic", "config_sha256": config_sha256(cfg)}
    run_sc_alpha(cfg, run_dir, metadata, "cpu")
    results = json.loads((run_dir/"cv_results.json").read_text())["candidates"]
    assert all(r["status"] == "FAILED" for r in results["0.0"])
    assert all(r["status"] == "COMPLETED" for r in results["0.1"])
    assert json.loads((run_dir/"alpha_selection.json").read_text())["selected_lambda"] == .1


def test_suite_gate_precedes_any_validation_read(tmp_path, monkeypatch):
    from scdfc import sc_alpha_evaluation as ev
    cfg = load_config(config_path(52)); cfg["paths"]["root"] = str(tmp_path)
    def forbidden(*args, **kwargs): pytest.fail("Incomplete suite must not read validation")
    monkeypatch.setattr(ev, "load_cohort", forbidden)
    with pytest.raises(RuntimeError, match="all nine"):
        ev.require_suite_ready(cfg)


@pytest.mark.parametrize("number", range(52, 61))
def test_declared_configs_and_reference_protocol(number):
    cfg = validate_experiment_config(load_config(config_path(number)))
    assert cfg["experiment"]["execution_ready"]
    assert cfg["training"]["seeds"] == [42]
    assert cfg["model"]["alpha_clip"] is False
    assert cfg["comparison_plan"]["primary_family_size"] == 9
    for name in ("alpha_fit", "training_stats"):
        ref = cfg["artifacts"][name]
        assert file_sha256(ROOT/ref["path"]) == ref["sha256"]


def test_swaps_are_repeatable_no_self_and_preserve_all_subjects():
    from scdfc.sc_alpha_evaluation import derangement
    for seed in range(10):
        order = derangement(10, seed)
        assert np.all(order != np.arange(10))
        assert sorted(order) == list(range(10))
        np.testing.assert_array_equal(order, derangement(10, seed))


def test_invalid_alpha_protocol_is_rejected():
    cfg = load_config(config_path(52)); cfg["model"]["alpha_clip"] = True
    with pytest.raises(ValueError, match="unbounded"):
        validate_experiment_config(cfg)


def test_complete_nine_run_suite_gate_evaluation_and_summary(tmp_path, monkeypatch):
    from scdfc import sc_alpha_evaluation as ev
    from scdfc.metric_records import selection_record
    cohorts = make_cohorts()
    prototype = synthetic_config(tmp_path)
    bind_references(prototype, cohorts, tmp_path)
    context, template, alpha = fit_templates(cohorts["train"], np.arange(12))
    root_output = tmp_path/"outputs"
    ids = []
    for number, (encoder, k) in enumerate(((e, k) for e in ("linear", "mlp", "gcn") for k in (1, 6, 223)), 52):
        cfg = deepcopy(prototype)
        cfg["experiment"].update(id=f"E{number:04d}", level=1)
        cfg["model"].update(sc_encoder=encoder, alpha_parameters=k)
        cfg["comparison_plan"].update(require_suite_ready=True, swap_seeds=list(range(10)), test_access=False)
        validate_experiment_config(cfg)
        run_id = f"E{number:04d}-synthetic"
        ids.append(run_id)
        path = root_output/f"E{number:04d}/runs"/run_id
        (path/"checkpoints").mkdir(parents=True)
        metadata = {"experiment_id": cfg["experiment"]["id"], "run_id": run_id,
                    "config_sha256": config_sha256(cfg), "status": "COMPLETED", "seed": 42, "level": 1}
        write_json(path/"metadata.json", metadata)
        (path/"config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        from scdfc.sc_alpha import generator
        model = generator(cfg)
        transform = fit_sc_transform(cohorts["train"].sc, np.arange(12), cfg)
        selection = {"cv_folds_sha256": "shared", "selected_lambda": .1, "final_epochs": 1, "cv_long_edge_mse": .01}
        payload = {"schema": "sc_conditioned_alpha/v1", **metadata, "config": cfg,
                   "model": model.state_dict(), "sc_transform": transform,
                   "context_template": context, "group_template": template, "alpha_raw": alpha,
                   "selection": selection, "training_cohort_sha256": "shared"}
        checkpoint = path/"checkpoints/best.pt"
        torch.save(payload, checkpoint)
        write_json(path/"component_manifest.json", {"schema": "sc_conditioned_alpha/v1",
            "sha256": file_sha256(checkpoint), "config_sha256": metadata["config_sha256"]})
        write_json(path/"metrics_best.json", selection_record({"long_edge_mse": .01}, "long_edge_mse", None,
            definition={"implementation": "sc_alpha_training_cv/v1", "split": "train_cv"},
            source="sc_alpha_training_cv", split="train_cv"))
        write_json(path/"train.log", {"event": "epoch_complete", "epoch": 1, "train_mse": .01})
        if number == 52:
            with pytest.raises(RuntimeError, match="all nine"):
                ev.require_suite_ready(cfg)
    monkeypatch.setattr(ev, "verify_data_bindings", lambda c: {})
    accesses = []
    def load(c, split):
        accesses.append(split)
        assert split == "val"
        return cohorts[split]
    monkeypatch.setattr(ev, "load_cohort", load)
    for run_id in ids:
        path = root_output/run_id[:5]/"runs"/run_id
        cfg, _ = ev.run_config(path)
        assert len(ev.require_suite_ready(cfg)) == 9
        ev.evaluate_sc_alpha_run(cfg, path)
    assert accesses == ["val"]*9
    output = ev.summarize_sc_alpha_runs(tmp_path, ids)
    summary = json.loads(output.read_text())
    assert len(summary["conditions"]) == 9
    assert len(summary["secondary_exploratory_contrasts"]) == 18
    assert len(list(output.parent.glob("sc_alpha_suite*.png"))) == 3
    # A stale evaluation must not be included after the private metric table changes.
    private = root_output/"E0052/runs"/ids[0]/"private/evaluation_val_samples.npz"
    private.write_bytes(private.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="changed"):
        ev.summarize_sc_alpha_runs(tmp_path, ids)


def test_managed_cli_keeps_training_and_validation_separate(tmp_path, monkeypatch):
    from scdfc import managed_cli, sc_alpha_evaluation as ev
    from scdfc.management import summarize_experiment, register_experiment
    cfg = synthetic_config(tmp_path)
    cohorts = make_cohorts(); bind_references(cfg, cohorts, tmp_path)
    config_file = tmp_path/"configs/E0052.yaml"; config_file.parent.mkdir()
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    register_experiment(config_file)
    accesses = []
    def load(c, split):
        accesses.append(split)
        return cohorts[split]
    monkeypatch.setattr("scdfc.sc_alpha.load_cohort", load)
    monkeypatch.setattr(managed_cli, "verify_data_bindings", lambda c: {})
    monkeypatch.setattr("scdfc.management.git_metadata", lambda root: {"commit": "a"*40, "branch": "synthetic", "dirty": False, "status": []})
    monkeypatch.setattr("scdfc.management.environment_snapshot", lambda: "synthetic test")
    monkeypatch.setattr(managed_cli, "_ensure_stats", lambda *a: pytest.fail("SC alpha must not refresh shared templates"))
    managed_cli.command_run(SimpleNamespace(experiment=str(config_file), seed=42, device="cpu"))
    assert accesses == ["train"]
    run_dir = next((tmp_path/"outputs/E0052/runs").iterdir())
    metadata = json.loads((run_dir/"metadata.json").read_text())
    assert metadata["status"] == "COMPLETED" and metadata["selection_split"] == "train_cv"
    summary = summarize_experiment(tmp_path, "E0052")
    assert summary["split"] == "train_cv" and summary["metric_source"] == "training_cv_selection"
    monkeypatch.setattr(managed_cli, "_root", lambda: tmp_path)
    monkeypatch.setattr(ev, "load_cohort", load)
    monkeypatch.setattr(ev, "verify_data_bindings", lambda c: {})
    args = SimpleNamespace(run_id=run_dir.name, split="val", final_test=False, baseline_run_id=None,
                           save_predictions=False, device="cpu")
    managed_cli.command_evaluate_run(args)
    assert accesses == ["train", "val"]
    args.split = "train"
    with pytest.raises(ValueError, match="validation-only"):
        managed_cli.command_evaluate_run(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
@pytest.mark.parametrize("encoder", ["linear", "mlp", "gcn"])
def test_cuda_synthetic_training_path(tmp_path, encoder):
    cfg = synthetic_config(tmp_path, encoder, 223)
    cohorts = make_cohorts(); context, template, alpha = fit_templates(cohorts["train"], np.arange(12))
    transform = fit_sc_transform(cohorts["train"].sc, np.arange(8), cfg)
    model, record = fit_one(cfg, transform_sc(cohorts["train"].sc, transform),
        statistics(cohorts["train"], context, template), alpha, np.arange(8), np.arange(8, 12),
        .1, torch.device("cuda"))
    assert np.isfinite(record["best_score"])
    assert record["initial"]["device_rng_sha256"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
