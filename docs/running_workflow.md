# 运行流程与实时日志

以下命令均应在项目根目录和符合依赖要求的环境中执行；`GCN_mri` 是示例 Conda 环境名，并非仓库自带环境：

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

以下是首次建立研究流程时的执行顺序。当前项目已经推进到 E0024，状态见[项目快照](project_status.md)；不要把已完成的单 seed 实验误认为尚未启动，也不要覆写它们的冻结配置：

1. 解析基线：`E0001`（group mean）和 `E0002`（FC1 persistence）。
2. `E0003` 的 FC 自编码器；成功后会生成 `configs/artifacts/A0003.yaml`。提交该清单后，后续 sequence 实验显式引用它。
3. 主模型：E0004（GCN+GRU）、E0005（Hybrid+GRU）、E0006（GCN+Transformer）、E0007（Hybrid+Transformer）。
4. 四个模型先各运行 seed 42；比较验证集后，再为候选模型创建 FC1-only、SC-only 和多 seed 确认实验。

以下命令作为早期受管理流程的示例；运行前须有相同版本的私人数据、缓存和完整 A0003 checkpoint。新问题应创建新 ID，不能借旧 ID 修改参数：

```powershell
scdfc run --experiment configs/experiments/E0001_group_mean_baseline.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0002_fc1_persistence_baseline.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0003_fc_autoencoder_w83_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0004_gcn_gru_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0005_hybrid_gru_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0006_gcn_transformer_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0007_hybrid_transformer_full_v1.yaml --seed 42 --device cuda
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

序列实验按冻结配置中的完整 `CompositeLoss` 得到验证集 `objective_loss`，并据此选择 checkpoint。该目标最多启用三个非零分量，不能统一解释为“边 MSE + 0.25 × 差分 MSE”。不同损失定义下的 `objective_loss` 不能直接横向排名。E0004–E0007、E0009–E0012 为历史 Huber 运行；E0008 显式使用 MSE，后续实验也应逐项查看 `loss_type`、权重和 checkpoint 选模记录。

对探索性模型还应执行 `scdfc dynamic-audit --run-id <run_id> --split val`，并分别解释幅度、差分方向、频谱、个体检索和矩阵合法性。E0018/E0022/E0023/E0024 已有 `val_test` 图，测试数据并非完全未见过；后续正式确认需要另定锁定方案。

## 4. 查看运行状态

所有长任务会将 JSON 事件立即打印到控制台，前缀为 `[scdfc]`；训练每个 epoch 输出一次 `epoch_complete`，其中包括训练损失、验证主指标、最佳值和早停计数。运行目录也会保存同样的 JSONL 文件：

```powershell
Get-Content outputs\E0003\runs\<run_id>\train.log -Wait
Get-Content outputs\E0003\runs\<run_id>\metadata.json
```

若将输出重定向到文件，保留 `$env:PYTHONUNBUFFERED = "1"`。完成后，检查 `metrics_best.json`、`metrics_last.json`、`evaluation_val.json`（或最终的 `evaluation_test.json`）以及 `metadata.json`。`metrics_best.json` 只保存验证集选模记录，评价命令不会覆盖它；通用评价指标只写入对应的 `evaluation_<split>.json`。选模记录包含 `kind=selection`、`split=val`、零基 `best_epoch` 和损失定义。

历史运行若仍使用旧格式，可执行：

```powershell
python scripts/repair_metric_records.py          # 只预览
python scripts/repair_metric_records.py --apply  # 备份旧文件后，从 checkpoint 恢复
```

迁移会把旧文件保存为 `metrics_best.pre_schema_v2.json`。若注册实验的 checkpoint 不在本机，或本地 run 的配置哈希与注册表不一致，汇总会拒绝猜测或混合结果。

`epoch_complete` 还记录 `train_seconds`、`validation_seconds`、`epoch_seconds`、累计平均 epoch 时间、按最大 epoch 估算的剩余时间、假设不再改善时距早停的估算时间、训练吞吐量和峰值 GPU 显存。前 3 个 epoch 后应优先使用这些实测字段更新总耗时预估。

## 5. 可视化交付

后续实验可视化统一写入 `outputs/E####/visual/`，该目录与该实验的 `runs/` 同级。每个图须由同目录下可重复运行的 `.py` 脚本生成，并提交静态图片（默认 PNG）；不以 HTML 或网页作为交付物。完整目录、命名和可追溯性要求见[实验可视化合约](visualization_contract.md)。

在验证评估与动态审计完成后，可生成一个实验的四图验证报告：

```powershell
python -m scdfc.visualization --root . --experiment E0004
```
