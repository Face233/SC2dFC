# 可复现实验管理操作手册

本项目区分两类入口：

- `audit/split/precompute/train-ae/train/evaluate` 是兼容旧流程的调试命令；旧训练与评价结果不作为正式证据。
- `freeze-data/experiment/run/evaluate-run/summarize/conclude` 是受管理的科研实验入口。

受管理运行严格区分两类验证产物：`metrics_best.json` 是训练阶段产生的 checkpoint 选模证据；`evaluation_val.json` 是最佳 checkpoint 的通用评价。两者可能都包含 `objective_loss`，但只有定义一致且可重放时才能比较；旧 checkpoint 无法重放时，评价会明确标记不可用，而不能用通用边分数冒充训练目标。评价过程不得覆盖选模文件。实验汇总只读取带 `kind=selection`、`split=val` 的新版选模记录，并核对 run 与注册实验的配置哈希。

所有命令均在项目根目录、满足 `pyproject.toml` 依赖的环境中执行；`GCN_mri` 是示例 Conda 环境名。现有 E0001–E0024 的进度及未收口项见[项目状态](project_status.md)。以下 E0003/E0004 的创建命令展示**项目初建时的流程**；当前编号和工件已被使用，新研究不要重新创建或改写它们，应分配新实验 ID。

## 1. 冻结数据和划分

```powershell
scdfc freeze-data `
  --config configs/default.yaml `
  --dataset-version dataset_lr_v1 `
  --preprocessing-version preprocess_lr_v1 `
  --split-version split_lr_v1
```

该命令会完整审计数据、计算逐文件 SHA256，并在 `data/manifests/` 生成数据清单、审计报告和被试级划分。该目录包含被试标识，因此不会进入 Git。版本名一旦生成不得覆盖；数据变化后使用 `dataset_v2` 或新的预处理/划分版本。

当前 `dataset_lr_v1` 只读取 LR 时间序列：SC/LR 可配对 1055 名被试，完整审计无错误，`split_lr_v1` 包含 train/val/test 738/158/159 名被试。RL 原始文件仍可保留在本地，但不会进入数据清单、审计、划分、缓存、训练或评价。

生成正式缓存时，`precompute` 会自动只处理冻结划分中的合格被试：

```powershell
scdfc precompute --config configs/default.yaml --windows 83
# 需要窗长敏感性分析时再运行：
scdfc precompute --config configs/default.yaml --windows 42 125
```

## 2. 建立冻结的 FC 自编码器

```powershell
scdfc experiment create `
  --name fc_autoencoder_w83_v1 `
  --level 1 `
  --task autoencoder `
  --model fc_autoencoder `
  --research-question "FC 是否能稳定压缩到共享潜空间" `
  --hypothesis "256 维潜空间能够保持主要 FC 结构" `
  --primary-change "建立首个冻结 FC 自编码器" `
  --owner researcher_name `
  --seeds 42
```

创建命令会同时生成实验 YAML 和 `reports/experiment_registry.csv` 的 PLANNED 行。Level 1/2 要求工作区干净，因此运行前先审核并提交：

```powershell
git add configs/experiments reports/experiment_registry.csv
git commit -m "exp(E0003): register FC autoencoder artifact"
scdfc run --experiment configs/experiments/E0003_fc_autoencoder_w83_v1.yaml --seed 42
```

成功后生成 `configs/artifacts/A0003.yaml`。该小型清单进入 Git，checkpoint 在 `outputs/E0003/runs/<run_id>/checkpoints/best.pt`；需按[归档说明](checkpoint_archive.md)按需取得完整模型，不能把 LFS 指针当作可加载权重。在服务器之间复制 checkpoint 后必须保持清单中的相对路径和 SHA256 一致。E0003 有效训练目标是 Fisher-z 边 MSE，相关和 PSD 自编码器损失未启用。

## 3. 建立基线或主模型实验

解析基线不依赖自编码器：

```powershell
scdfc experiment create `
  --name group_mean_baseline `
  --level 1 `
  --task analytic `
  --model group_mean `
  --research-question "群体均值基线有多强" `
  --hypothesis "复杂模型应稳定优于群体均值" `
  --primary-change "建立群体均值下界" `
  --owner researcher_name
```

学习型实验显式引用冻结工件：

```powershell
scdfc experiment create `
  --name gcn_gru_full_v1 `
  --level 1 `
  --task sequence `
  --model gru `
  --sc-encoder hcp_gcn `
  --artifact configs/artifacts/A0003.yaml `
  --research-question "SC 是否为首窗 FC 提供增量预测信息" `
  --hypothesis "完整 GCN+GRU 在验证集目标上优于 FC1 persistence" `
  --primary-change "建立 GCN+GRU 主模型" `
  --baseline E0002 `
  --owner researcher_name `
  --seeds 42
```

可选条件模型包括 `gru`、`tcn` 和 `transformer`。当前 E0004–E0007 使用 `gru` 或 `transformer`、`hybrid` 或 `hcp_gcn` SC encoder，并固定 `e0003_reconstruction_decoder` 输出头。其他名称保留用于历史基线。所有学习型模型均不接收扫描方向编码。

每个已声明 seed 单独运行：

```powershell
scdfc run --experiment configs/experiments/E0004_gcn_gru_full_v1.yaml --seed 42 --device cuda
```

## 4. 评价、汇总和科研结论

Level 0/1 只能评价 train/val；用真实运行结束时输出的 run ID，下面是格式示意，不是当前仓库中的实际运行：

```powershell
scdfc evaluate-run --run-id E0004-s42-20260724T120000Z-abcdef0 --split val
```

只有 Level 2 能显式执行一次最终测试；成功后生成实验级锁文件。已有部分 `val_test` 图属于探索性测试集接触，不能把同一测试数据视为完全未见过的最终确认集：

```powershell
scdfc evaluate-run --run-id <run_id> --final-test
```

全部 seed 完成后再汇总并记录真实人工结论；以下 `KEEP` 文本是格式示意，**不是 E0004 的现有结论**：

```powershell
scdfc summarize --experiment E0004
scdfc conclude `
  --experiment E0004 `
  --status KEEP `
  --conclusion "<填写验证集证据和局限>" `
  --next-step "<填写下一步实验或停止理由>"
git add reports
git commit -m "docs: conclude E0004"
```

合法结论为 `KEEP`、`REJECT`、`INCONCLUSIVE`、`FAILED` 和 `ARCHIVED`。

## 5. 运行目录与恢复

正式运行写入 `outputs/E####/runs/<run_id>/`，目录不可覆盖。checkpoint 中记录 experiment/run ID、配置哈希、代码版本、数据/划分版本和依赖工件校验值；评价始终读取该 run 的 `config_resolved.yaml`，不会使用后来被修改的默认配置。

Level 0 允许脏工作区并保存 Git diff；Level 1/2 检测到任何未提交文件都会拒绝启动。语义参数必须写入实验 YAML，正式入口只允许通过 `--seed` 和 `--device` 选择已声明 seed 与运行设备。
