import json

import numpy as np
import torch

from scdfc.management import file_sha256
from scdfc.residual_dynamic_ridge import evaluate_residual_dynamic_ridge, run_residual_dynamic_ridge


def test_dynamic_pair_fit_and_replay_preserve_temporal_correlation(tmp_path, monkeypatch):
    rng = np.random.default_rng(34)
    inputs = {split: rng.normal(size=(count, 6)).astype(np.float32)
              for split, count in (("train", 15), ("val", 7))}
    time = np.arange(223, dtype=np.float32)
    wave_a = np.sin(2 * np.pi * time / 40)
    wave_b = np.cos(2 * np.pi * time / 67)
    wave_a -= wave_a.mean()
    wave_b -= wave_b.mean()
    spatial_a = np.array([1, .5, .2, -.4, -.8, .3], dtype=np.float32)
    spatial_b = np.array([.1, -.4, .9, .5, -.2, -.7], dtype=np.float32)
    template = 0.02 * np.sin(2 * np.pi * time / 100)[:, None] * np.ones((1, 6), dtype=np.float32)

    class FakeDataset:
        def __init__(self, config, window_length, split, stats_path, ablation):
            assert ablation == "fc1_only"
            self.split = split

        def __len__(self):
            return len(inputs[self.split])

        def __getitem__(self, index):
            x = inputs[self.split][index]
            stable = 0.1 * x
            dynamic = (0.3 * x[0] * wave_a[:, None] * spatial_a[None] +
                       0.25 * x[1] * wave_b[:, None] * spatial_b[None])
            return {"fc_warmup": torch.from_numpy(x),
                    "fc_future": torch.from_numpy((template + stable[None] + dynamic).astype(np.float32)),
                    "subject_id": str(index), "run_name": "LR"}

    class IdentityEncoder(torch.nn.Module):
        def encode(self, value):
            return value

    monkeypatch.setattr("scdfc.residual_dynamic_ridge.DFCSequenceDataset", FakeDataset)
    monkeypatch.setattr("scdfc.residual_dynamic_ridge.load_autoencoder", lambda *args: IdentityEncoder())
    stats_path = tmp_path / "stats.npz"
    alpha_path = tmp_path / "alpha.npz"
    encoder_path = tmp_path / "encoder.pt"
    stable_path = tmp_path / "stable.pt"
    np.savez(stats_path, group_template=template, context_template=np.zeros(6))
    np.savez(alpha_path, alpha=np.zeros(223))
    encoder_path.write_bytes(b"frozen encoder")
    torch.save({"model_name": "encoded_stable_residual_ridge",
                "ridge_coef": np.eye(6) * .1, "ridge_intercept": np.zeros(6),
                "scaler_mean": np.zeros(6), "scaler_scale": np.ones(6),
                "stats_sha256": file_sha256(stats_path), "alpha_sha256": file_sha256(alpha_path),
                "autoencoder_sha256": file_sha256(encoder_path)}, stable_path)
    base = {"seed": 42, "paths": {"root": str(tmp_path)},
            "data": {"warmup_windows": 1, "window_length": 83, "stride": 5},
            "model": {"pca_components_grid": [2, 3], "pca_fit_windows_per_subject": 12,
                      "ridge_alpha_grid": [.01, 1.0], "cv_folds": 3},
            "evaluation": {"bootstrap_replicates": 50},
            "artifacts": {"stable_ridge": {"path": str(stable_path), "sha256": file_sha256(stable_path)}}}
    reports = {}
    checkpoints = {}
    for include_stable in (False, True):
        config = {**base, "model": {**base["model"], "include_stable": include_stable}}
        run_dir = tmp_path / ("with_m" if include_stable else "without_m")
        checkpoint = run_residual_dynamic_ridge(config, 83, stats_path, encoder_path, alpha_path, run_dir,
            {"experiment_id": "E0035" if include_stable else "E0034", "run_id": "synthetic",
             "config_sha256": "abc"}, "cpu")
        checkpoints[include_stable] = torch.load(checkpoint, map_location="cpu", weights_only=False)
        replay = evaluate_residual_dynamic_ridge(config, 83, stats_path, encoder_path, alpha_path,
                                                  checkpoint, run_dir, "val", "cpu")
        reports[include_stable] = json.loads(replay.read_text(encoding="utf-8"))
    no_m = reports[False]["aggregate"]
    with_m = reports[True]["aggregate"]
    assert no_m["dynamic_mse"] < no_m["zero_dynamic_mse"]
    assert no_m["time_pearson"] > no_m["zero_dynamic_time_pearson"]
    assert no_m["dynamic_time_pearson"] > no_m["shifted_dynamic_time_pearson"]
    assert with_m["long_edge_mse"] < no_m["long_edge_mse"]
    np.testing.assert_allclose(with_m["time_pearson"], no_m["time_pearson"], atol=1e-6)
    np.testing.assert_array_equal(checkpoints[False]["pca_components"], checkpoints[True]["pca_components"])
    np.testing.assert_array_equal(checkpoints[False]["ridge_coef"], checkpoints[True]["ridge_coef"])
