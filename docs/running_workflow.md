# 运行流程与实时日志

以下命令均应在项目根目录和 `GCN_mri` Conda 环境中执行：

```powershell
conda activate GCN_mri
$env:PYTHONUNBUFFERED = "1"
```

正式科研流程使用受管理入口。Level 1/2 实验要求 Git 工作区干净；因此每次创建实验配置、生成工件清单或修改代码后，先审阅并提交这些变更，再运行 `scdfc run`。

## 1. 数据审计、冻结与 dFC 缓存

首次使用某一批数据时：

```powershell
scdfc audit --config configs/default.yaml
scdfc freeze-data --config configs/default.yaml `
  --dataset-version dataset_lr_v1 `
  --preprocessing-version preprocess_lr_v1 `
  --split-version split_lr_v1
```

已冻结的版本不可覆盖。原始数据或预处理发生改变时，使用新的三个版本名，并为新版本创建新的实验配置。

先构建主分析的 83 TR 缓存；42 和 125 TR 仅用于后续敏感性分析：

```powershell
scdfc precompute --config configs/default.yaml --windows 83
# 后续敏感性分析时再运行：
scdfc precompute --config configs/default.yaml --windows 42 125
```

预计算会每处理 25 个 subject/run 输出一条 `precompute_progress` 事件。缓存位于 `data/cache/dfc/window_<window>.zarr`；若明确要重建同窗长缓存，增加 `--overwrite`。

## 2. 基线、自编码器和模型

按以下顺序运行，以避免把测试集用于选择模型：

1. 解析基线：`E0001`（group mean）和 `E0002`（FC1 persistence）。
2. `E0003` 的 FC 自编码器；成功后会生成 `configs/artifacts/A0003.yaml`。提交该清单后，后续 sequence 实验显式引用它。
3. 简单学习型基线：`pca_ridge`、`mlp`、`lstm`；SC-only 对照：`direct_mlp`、`gcn_gru`。
4. 主模型：`tcn`，然后 `transformer`；再对候选主模型运行 `fc1_only`、`sc_only`、`mean_sc`、`shuffled_sc` 消融。

现有已登记实验可直接运行：

```powershell
scdfc run --experiment configs/experiments/E0001_group_mean_baseline.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0002_fc1_persistence_baseline.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0003_fc_autoencoder_w83_v1.yaml --seed 42 --device cuda
```

对于新模型，先通过 `scdfc experiment create` 生成不可变配置并登记，然后提交 `configs/experiments/` 和 `reports/experiment_registry.csv`，再以 `scdfc run --experiment ... --seed ... --device cuda` 启动。每个 seed 都应单独运行；验证集聚合后才决定是否创建 Level 2 正式确认实验。

## 3. 评估和决策

Level 0/1 仅可看训练/验证集：

```powershell
scdfc evaluate-run --run-id <run_id> --split val --device cuda
scdfc summarize --experiment E0003
```

在预先定义的规则下确认候选模型后，才运行 Level 2 的唯一一次最终测试：

```powershell
scdfc evaluate-run --run-id <level-2-run-id> --final-test --device cuda
```

主比较使用 `long_residual_pearson`，并且应将主模型与 `fc1_only` 对照比较；不能仅依据 `raw_edge_pearson` 宣称 SC 带来增益。

## 4. 查看运行状态

所有长任务会将 JSON 事件立即打印到控制台，前缀为 `[scdfc]`；训练每个 epoch 输出一次 `epoch_complete`，其中包括训练损失、验证主指标、最佳值和早停计数。运行目录也会保存同样的 JSONL 文件：

```powershell
Get-Content outputs\E0003\runs\<run_id>\train.log -Wait
Get-Content outputs\E0003\runs\<run_id>\metadata.json
```

若将输出重定向到文件，保留 `$env:PYTHONUNBUFFERED = "1"`。完成后，检查 `metrics_best.json`、`metrics_last.json`、`evaluation_val.json`（或最终的 `evaluation_test.json`）以及 `metadata.json`。
