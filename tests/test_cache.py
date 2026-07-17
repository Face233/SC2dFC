from pathlib import Path

import numpy as np
import pandas as pd

from scdfc.data import iter_cached_samples, precompute_dfc, read_cached


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
