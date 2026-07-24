# 可复现实验管理操作手册

本项目区分两类入口：

- `audit/split/precompute/train-ae/train/evaluate` 是兼容旧流程的调试命令；旧训练与评价结果不作为正式证据。
- `freeze-data/experiment/run/evaluate-run/summarize/conclude` 是受管理的科研实验入口。

所有命令均在项目根目录、`GCN_mri` 环境中执行。

## 1. 冻结数据和划分

```powershell
scdfc freeze-data `
  --config configs/default.yaml `
  --dataset-version dataset_v1 `
  --preprocessing-version preprocess_v1 `
  --split-version split_v1
```

该命令会完整审计数据、计算逐文件 SHA256，并在 `data/manifests/` 生成数据清单、审计报告和被试级划分。该目录包含被试标识，因此不会进入 Git。版本名一旦生成不得覆盖；数据变化后使用 `dataset_v2` 或新的预处理/划分版本。

当前 `dataset_v1` 审计发现 6 名被试的 RL 文件长度或 CSV 结构异常，这些被试已从 `split_v1` 排除，原文件保持不变。

生成正式缓存时，`precompute` 会自动只处理冻结划分中的合格被试：

```powershell
scdfc precompute --config configs/default.yaml --windows 83 42 125
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

成功后生成 `configs/artifacts/A0003.yaml`。该小型清单进入 Git，checkpoint 本身继续保留在 `outputs/`；在服务器之间复制 checkpoint 后必须保持清单中的相对路径和 SHA256 一致。

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
  --name tcn_full_v1 `
  --level 1 `
  --task sequence `
  --model tcn `
  --artifact configs/artifacts/A0003.yaml `
  --research-question "SC 是否为首窗 FC 提供增量预测信息" `
  --hypothesis "完整 TCN 在长时距残差相关上优于 FC1-only" `
  --primary-change "加入个体 SC 条件" `
  --baseline E0002 `
  --owner researcher_name `
  --seeds 42
```

可选模型包括 `pca_ridge`、`mlp`、`lstm`、`direct_mlp`、`gcn_gru`、`tcn` 和 `transformer`。其中 `direct_mlp` 与 `gcn_gru` 明确保留为 SC-only v1 基线；`mlp`、`lstm` 和 `pca_ridge` 使用与主模型相同的 SC、首窗 FC 和 run 信息。

每个 seed 单独提交：

```powershell
scdfc run --experiment configs/experiments/E0004_tcn_full_v1.yaml --seed 42 --device cuda
```

## 4. 评价、汇总和科研结论

Level 0/1 只能评价 train/val：

```powershell
scdfc evaluate-run --run-id E0004-s42-20260724T120000Z-abcdef0 --split val
```

只有 Level 2 能显式执行一次最终测试；成功后生成实验级锁文件：

```powershell
scdfc evaluate-run --run-id <run_id> --final-test
```

全部 seed 完成后：

```powershell
scdfc summarize --experiment E0004
scdfc conclude `
  --experiment E0004 `
  --status KEEP `
  --conclusion "多个 seed 均稳定优于 E0002" `
  --next-step "进入 Level 2 正式确认"
git add reports
git commit -m "docs: conclude E0004"
```

合法结论为 `KEEP`、`REJECT`、`INCONCLUSIVE`、`FAILED` 和 `ARCHIVED`。

## 5. 运行目录与恢复

正式运行写入 `outputs/E####/runs/<run_id>/`，目录不可覆盖。checkpoint 中记录 experiment/run ID、配置哈希、代码版本、数据/划分版本和依赖工件校验值；评价始终读取该 run 的 `config_resolved.yaml`，不会使用后来被修改的默认配置。

Level 0 允许脏工作区并保存 Git diff；Level 1/2 检测到任何未提交文件都会拒绝启动。语义参数必须写入实验 YAML，正式入口只允许通过 `--seed` 和 `--device` 选择已声明 seed 与运行设备。
