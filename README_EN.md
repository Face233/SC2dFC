# SC-dFC

SC-dFC studies deterministic prediction of future resting-state dynamic functional connectivity from an individual's structural connectome and initial FC window(s). The default AAL90/LR task uses an 83-TR sliding window and a 5-TR stride; with one warm-up window, the model predicts the remaining 223 FC windows. The detailed method is documented in [Chinese](README_CN.md).

## Current status

The repository contains experiments through E0024, but no confirmed Level 2 result. Registered experiments are Level 1 and declare only seed 42. E0003 is marked `KEEP`; the other 19 registered experiments await conclusions. Later dynamic-only models recover some overall variation, yet validation diagnostics still show weak subject specificity and poor timing of the changes. See the [status snapshot](docs/project_status.md) and [diagnostic report](reports/research/2026-09-07/时间塌缩与特异性缺失_诊断与方法调研.md).

The frozen dataset records 1,055 paired SC/LR subjects and a 738/158/159 train/validation/test split. Raw imaging data, private split files, and regenerable caches are not included in this checkout. Experiment artifacts, some evaluation records with subject identifiers, and Git LFS checkpoint pointers are tracked; their release permissions should be reviewed. The actual [checkpoint archive](docs/checkpoint_archive.md) is not yet consistent with the intended lightweight-main policy.

## Environment and workflow

Use Python 3.11 or later and install the project in a compatible environment:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

PyTorch is pinned to 2.6.0 and the cache uses the Zarr 2.x API. `GCN_mri` in older instructions is an example Conda environment name, not an environment created by this repository. Full tests and training require the project dependencies; retraining also requires the private frozen data and the complete E0003 autoencoder checkpoint.

Evidence-producing work uses the [managed experiment workflow](docs/experiment_management.md): freeze data, create a new immutable E#### configuration, run declared seeds, evaluate only permitted splits, summarize, and record a human conclusion. The old `audit`, `precompute`, `train-ae`, `train`, and `evaluate` commands remain available for debugging; `evaluate` is restricted to train/validation, not formal test evaluation. New work must not reuse an existing experiment ID with changed parameters.

Sequence checkpoints are selected by each experiment's declared validation metric. E0004–E0007 used a historical Huber edge-plus-difference objective; E0024 uses MSE difference-plus-variance. There is no universal eight-term training objective, and objective values with different loss definitions are not directly comparable. `metrics_best.json` preserves checkpoint selection; `evaluation_<split>.json` contains general evaluation metrics.

Existing `val_test` visualizations for E0018 and E0022–E0024 count as exploratory test-set exposure. A future final claim needs an explicitly locked confirmation design; those results must not be described as coming from an untouched test set.

The precomputation step writes chunked Zarr data. Training never computes sliding-window FC online.
