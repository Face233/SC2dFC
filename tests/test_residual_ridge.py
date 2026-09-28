import numpy as np
import torch

from scdfc.residual_ridge import _choose_alpha, _report, evaluate_residual_ridge, run_residual_ridge, stable_baseline


def test_stable_correction_changes_level_but_preserves_temporal_shape():
    template = np.array([[0.1, 0.2], [0.2, 0.0], [0.0, 0.3]])
    baseline = stable_baseline(np.array([0.5, 0.4]), template, np.array([0.1, 0.1]),
                               np.array([1.0, 0.5, 0.2]))
    corrected = baseline + np.array([0.04, -0.08])[None]
    np.testing.assert_allclose(np.diff(corrected, axis=0), np.diff(baseline, axis=0))
    np.testing.assert_allclose(corrected.mean(0) - baseline.mean(0), [0.04, -0.08])


def test_grouped_ridge_selection_uses_subject_groups_and_training_targets():
    rng = np.random.default_rng(8)
    subject_feature = rng.normal(size=(12, 4))
    features = np.repeat(subject_feature, 2, axis=0)
    stable = features @ rng.normal(size=(4, 3))
    long_stable = stable.copy()
    subjects = [str(index) for index in np.repeat(np.arange(12), 2)]
    chosen, scores = _choose_alpha(features, stable, long_stable, subjects, [0.1, 100.0], 3)
    assert chosen == 0.1
    assert scores["0.1"] < scores["100.0"]


def test_report_detects_stable_gain_without_claiming_dynamic_gain():
    rng = np.random.default_rng(12)
    n, steps, edges = 8, 223, 6
    baseline = np.zeros((n, steps, edges), dtype=np.float32)
    stable = rng.normal(size=(n, edges)).astype(np.float32)
    dynamic = rng.normal(scale=0.01, size=(n, steps, edges)).astype(np.float32)
    target = baseline + stable[:, None] + dynamic
    residual = target - baseline
    segments = ((0, 17), (17, 85), (85, 154), (154, 223))
    data = {
        "stable": residual.mean(axis=1), "long_stable": residual[:, 17:].mean(axis=1),
        "baseline_long_mean": baseline[:, 17:].mean(axis=1),
        "baseline_long_mse": np.mean(residual[:, 17:] ** 2, axis=(1, 2)),
        "segment_residual_mean": np.stack([residual[:, start:stop].mean(axis=1)
                                            for start, stop in segments], axis=1),
        "segment_baseline_mse": np.stack([np.mean(residual[:, start:stop] ** 2, axis=(1, 2))
                                          for start, stop in segments], axis=1),
        "subjects": [str(index) for index in range(n)], "runs": ["LR"] * n,
    }
    report = _report(data, stable, np.zeros((steps, edges)), 17,
                     {"evaluation": {"bootstrap_replicates": 30}, "seed": 42}, "val")
    direct_mse = float(np.mean((baseline[:, 17:] + stable[:, None] - target[:, 17:]) ** 2))
    assert np.isclose(report["aggregate"]["long_edge_mse"], direct_mse, atol=1e-6)
    assert report["aggregate"]["long_edge_mse"] < report["aggregate"]["e0032_long_edge_mse"]
    assert report["paired_long_mse"]["ci_high"] < 0
    assert "centered temporal trajectory identical" in report["dynamic_note"]


def test_managed_ridge_fit_and_replay_on_synthetic_subjects(tmp_path, monkeypatch):
    rng = np.random.default_rng(19)
    inputs = {split: rng.normal(size=(count, 6)).astype(np.float32)
              for split, count in (("train", 12), ("val", 6))}

    class FakeDataset:
        def __init__(self, config, window_length, split, stats_path, ablation):
            assert ablation == "fc1_only"
            self.split = split

        def __len__(self):
            return len(inputs[self.split])

        def __getitem__(self, index):
            warmup = inputs[self.split][index]
            stable = 0.2 * warmup
            future = 0.2 * warmup[None] + np.zeros((223, 6), dtype=np.float32)
            return {"fc_warmup": torch.from_numpy(warmup), "fc_future": torch.from_numpy(future),
                    "subject_id": str(index), "run_name": "LR"}

    class IdentityEncoder(torch.nn.Module):
        def encode(self, value):
            return value

    monkeypatch.setattr("scdfc.residual_ridge.DFCSequenceDataset", FakeDataset)
    monkeypatch.setattr("scdfc.residual_ridge.load_autoencoder", lambda *args: IdentityEncoder())
    stats_path = tmp_path / "stats.npz"
    alpha_path = tmp_path / "alpha.npz"
    encoder_path = tmp_path / "encoder.pt"
    np.savez(stats_path, group_template=np.zeros((223, 6)), context_template=np.zeros(6))
    np.savez(alpha_path, alpha=np.zeros(223))
    encoder_path.write_bytes(b"synthetic frozen encoder")
    config = {"seed": 42, "data": {"warmup_windows": 1, "stride": 5},
              "model": {"ridge_alpha_grid": [0.1, 10.0], "cv_folds": 3},
              "evaluation": {"bootstrap_replicates": 30}}
    run_dir = tmp_path / "run"
    checkpoint = run_residual_ridge(config, 83, stats_path, encoder_path, alpha_path, run_dir,
                                    {"experiment_id": "E0033", "run_id": "synthetic", "config_sha256": "abc"}, "cpu")
    assert checkpoint.exists()
    replay = evaluate_residual_ridge(config, 83, stats_path, encoder_path, alpha_path, checkpoint,
                                     run_dir, "val", "cpu")
    import json
    report = json.loads(replay.read_text(encoding="utf-8"))
    assert report["aggregate"]["long_edge_mse"] < report["aggregate"]["e0032_long_edge_mse"]
    assert report["paired_long_mse"]["ci_high"] < 0
