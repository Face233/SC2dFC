from pathlib import Path

import numpy as np
import pandas as pd

from scdfc.data import audit_dataset, iter_cached_samples, precompute_dfc, read_cached
from scdfc.config import load_config


def test_default_config_is_lr_only_and_has_no_direction_switch():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert set(config["paths"]["timeseries"]) == {"LR"}
    assert "allow_missing_runs" not in config["data"]


def test_precompute_cache_is_offline_and_reproducible(tmp_path: Path):
    sc_dir = tmp_path / "sc"
    ts_dir = tmp_path / "lr"
    sc_dir.mkdir()
    ts_dir.mkdir()
    subject = "100001"
    np.savetxt(sc_dir / f"{subject}.csv", np.eye(4), delimiter=",")
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(30, 4)), columns=["A", "B", "C", "D"])
    frame.insert(0, "timepoint", np.arange(30))
    frame.to_csv(ts_dir / f"{subject}_AAL90_timeseries.csv", index=False)
    config = {
        "paths": {"root": str(tmp_path), "sc_dir": "sc", "timeseries": {"LR": "lr"}, "cache_dir": "cache"},
        "data": {"n_nodes": 4, "stride": 5, "fisher_clip": 0.999999},
    }
    result = precompute_dfc(config, 10)
    assert result == {"written": 1, "skipped": 0}
    assert list(iter_cached_samples(config, 10)) == [(subject, "LR")]
    fc, starts = read_cached(config, 10, subject, "LR")
    assert fc.shape == (5, 6)
    np.testing.assert_array_equal(starts, [0, 5, 10, 15, 20])
    assert precompute_dfc(config, 10) == {"written": 0, "skipped": 1}


def test_audit_records_malformed_csv_without_aborting(tmp_path: Path):
    sc_dir = tmp_path / "sc"
    ts_dir = tmp_path / "lr"
    sc_dir.mkdir()
    ts_dir.mkdir()
    np.savetxt(sc_dir / "100001.csv", np.eye(2), delimiter=",")
    (ts_dir / "100001_AAL90_timeseries.csv").write_text("timepoint\t,A,B\n0,1,2\n1,3,4,5\n", encoding="utf-8")
    atlas = tmp_path / "atlas.txt"
    atlas.write_text("1\tA\n2\tB\n", encoding="utf-8")
    config = {
        "paths": {"root": str(tmp_path), "atlas_labels": "atlas.txt", "sc_dir": "sc", "timeseries": {"LR": "lr"}},
        "data": {"n_nodes": 2, "n_timepoints": 2},
    }
    report = audit_dataset(config)
    assert report["invalid_subjects"] == ["100001"]
    assert any("cannot be parsed" in error for error in report["errors"])


def test_precompute_respects_frozen_split(tmp_path: Path):
    sc_dir = tmp_path / "sc"
    ts_dir = tmp_path / "lr"
    sc_dir.mkdir()
    ts_dir.mkdir()
    for subject in ["100001", "100002"]:
        np.savetxt(sc_dir / f"{subject}.csv", np.eye(4), delimiter=",")
        frame = pd.DataFrame(np.random.default_rng(int(subject)).normal(size=(30, 4)), columns=list("ABCD"))
        frame.insert(0, "timepoint", np.arange(30))
        frame.to_csv(ts_dir / f"{subject}_AAL90_timeseries.csv", index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame({"subject_id": ["100001"], "split": ["train"]}).to_csv(split, index=False)
    config = {
        "paths": {"root": str(tmp_path), "sc_dir": "sc", "timeseries": {"LR": "lr"}, "cache_dir": "cache", "split_csv": "split.csv"},
        "data": {"n_nodes": 4, "stride": 5, "fisher_clip": 0.999999},
    }
    assert precompute_dfc(config, 10) == {"written": 1, "skipped": 0}
    assert list(iter_cached_samples(config, 10)) == [("100001", "LR")]
