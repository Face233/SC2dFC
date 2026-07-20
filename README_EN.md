# SC-dFC

Deterministic prediction of a future resting-state dynamic functional-connectivity sequence from an individual structural connectome and the first FC window.

中文详细说明请见 [README_CN.md](README_CN.md)。

## Environment

```powershell
conda activate GCN_mri
python -m pip install -e ".[dev]"
```

Place source data under `data/raw/`: atlas labels in `atlas/`, SC files in `sc/`, and ROI time series in `timeseries_lr/` and `timeseries_rl/`. Derived splits and statistics belong in `data/interim/`; regenerable dFC caches belong in `data/cache/`.

## Pipeline

```powershell
scdfc audit --config configs/default.yaml
scdfc split --config configs/default.yaml
scdfc precompute --config configs/default.yaml --windows 83 42 125
scdfc train-ae --config configs/default.yaml --window 83
scdfc train --config configs/default.yaml --window 83 --model tcn
scdfc train --config configs/default.yaml --window 83 --model transformer
scdfc train --config configs/default.yaml --window 83 --model direct_mlp
scdfc train --config configs/default.yaml --window 83 --model gcn_gru
scdfc evaluate --config configs/default.yaml --window 83 --checkpoint outputs/window_83/tcn_full/best.pt
```

The precomputation step writes chunked Zarr data. Training never computes sliding-window FC online.
