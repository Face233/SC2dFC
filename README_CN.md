# SC-dFC：基于结构连接与首窗 FC 的动态功能连接预测

SC-dFC 是一个用于静息态 fMRI 的确定性预测框架。给定某名被试的结构连接矩阵（SC）和该 run 的第一个动态功能连接窗口（FC warm-up），模型预测后续所有动态功能连接（dFC）窗口，并重建为完整的相关矩阵序列。

项目当前面向 AAL90 分区和 HCP 风格的 ROI BOLD 时间序列实现；模型、数据缓存和评价逻辑均为配置驱动。

> 隐私与数据声明：本仓库只应包含代码、配置、测试和文档。任何 MRI、BOLD 时间序列、SC/FC 矩阵、行为表、缓存、训练输出和模型权重都不应提交到 GitHub。

## 1. 研究任务

主任务为：

\[
(SC_s, FC_{s,1}, run_s) \longrightarrow \hat{FC}_{s,2:T}
\]

其中：

- `SC_s`：第 `s` 名被试的 `90×90` 加权、对称结构连接矩阵；
- `FC_{s,1}`：同一 fMRI run 的第一个 dFC 窗口；
- `run_s`：LR 或 RL run 标记；
- `\hat{FC}_{s,2:T}`：预测得到的后续完整 `90×90` dFC 矩阵序列。

本版本是**确定性条件预测**：同一输入只输出一条后续轨迹。它评估的是 SC 和当前功能状态能否约束后续 dFC，而不是完整建模 \(p(dFC\mid SC,FC_1)\)。条件扩散、流匹配或状态空间生成模型属于后续扩展。

## 2. 方法概览

```text
SC matrix ──┬─ Graph Transformer ─┐
            └─ edge MLP ──────────┼─ condition encoder ─┐
FC warm-up ── FC encoder ─────────┘                     │
run (LR/RL) ─ embedding ─────────────────────────────────┤
                                                         ▼
                     TCN 或 Transformer 轨迹解码器 → FC latent sequence
                                                         ▼
                          FC decoder → Fisher-z edges → 完整 90×90 FC 序列
```

模型输出在 Fisher-z 边空间中显式分为：

\[
\hat{FC}_{s,t}=FC^{group}_t+\Delta FC^{static}_s+\Delta FC^{dynamic}_{s,t}
\]

- `FC_group` 是训练集中计算的群体模板；
- `ΔFC_static` 是个体稳定偏差；
- `ΔFC_dynamic` 是时间均值为零的个体动态残差。

这项分解配合“去群体模板后的长时距相关”作为主指标，用于避免模型只输出组平均的平直序列。

## 3. 目录与数据要求

默认配置文件为 [configs/default.yaml](configs/default.yaml)。请在项目根目录准备如下结构：

```text
SC2dFC/
├── AAL_atlas/
│   └── ROI_MNI_V4.txt
├── CSV_Files/
│   └── HCP_Structure/AAL90/<subject_id>.csv
├── TimeSeries_LR/
│   └── <subject_id>_AAL90_timeseries.csv
├── TimeSeries_RL/                         # 推荐；可暂缺
│   └── <subject_id>_AAL90_timeseries.csv
├── configs/default.yaml
└── src/
```

### 3.1 SC 矩阵

- 文件名：`CSV_Files/HCP_Structure/AAL90/<subject_id>.csv`；
- 格式：无表头的 `90×90` CSV；
- 要求：数值有限、对称、对角线为零或接近零；
- 不在程序中进行阈值化；训练时对 SC 上三角做 `log1p`，再按训练集逐边标准化。

### 3.2 BOLD ROI 时间序列

- 文件名：`TimeSeries_LR/<subject_id>_AAL90_timeseries.csv`；RL 文件规则相同；
- 默认形状：`1200×91`，第一列为 `timepoint`，后 90 列为 AAL90 ROI；
- 要求：ROI 名称与 `ROI_MNI_V4.txt` 前 90 个标签的顺序严格一致；
- 默认假设 HCP TR 为 0.72 秒，时间序列来自 HCP minimal preprocessing + ICA-FIX；如不符合，请修改配置并记录实际预处理。

## 4. 安装环境

推荐使用已具备 CUDA PyTorch 的 Conda 环境：

```powershell
conda activate GCN_mri
python -m pip install -e ".[dev]"
```

主要依赖：Python 3.11、PyTorch、NumPy、Pandas、SciPy、scikit-learn、Zarr、PyYAML 和 pytest。

验证安装：

```powershell
scdfc --help
pytest
```

> Windows 下如果 `scdfc` 命令不可用，可使用 `python -m scdfc.cli` 替代。例如：`python -m scdfc.cli audit --config configs/default.yaml`。

## 5. 完整运行流程

所有命令都在项目根目录执行。建议先使用 `--window 83` 完成主分析，再单独运行 42 和 125 TR 敏感性分析。

### 步骤 1：数据审计

```powershell
scdfc audit --config configs/default.yaml
```

审计报告写入 `outputs/audit.json`，包括：SC/时间序列数目、可配对被试数、ROI 顺序、矩阵形状、有限值、SC 对称性和 LR/RL 可用性。

在执行后续步骤前，应确保 `errors` 为空。`warnings` 中的 RL 缺失应按研究设计处理。

### 步骤 2：生成被试级划分

```powershell
scdfc split --config configs/default.yaml
```

输出为 `outputs/splits.csv`，默认比例为 70% 训练、15% 验证、15% 测试。划分以 `subject_id` 为单位，同一被试的 LR/RL 始终属于同一分区；划分由 `seed` 固定，可在配置文件中修改。

### 步骤 3：离线计算 dFC 缓存

```powershell
scdfc precompute --config configs/default.yaml --windows 83 42 125
```

该步骤使用矩形窗 Pearson 相关，取上三角并 Fisher-z 变换，写入 `cache/dfc/window_<window_length>.zarr`。训练阶段只读取这些缓存，**不会在线计算滑窗相关**。

默认参数：

| 设置 | 主分析 | 敏感性 1 | 敏感性 2 |
| --- | ---: | ---: | ---: |
| 窗长 | 83 TR | 42 TR | 125 TR |
| 对应时长（TR=0.72 s） | 59.76 s | 30.24 s | 90.00 s |
| 步长 | 5 TR | 5 TR | 5 TR |
| 每 run 窗口数 | 224 | 232 | 216 |
| 首窗后无 BOLD 重叠的预测步 | 17 | 9 | 25 |

如果需要重新计算同一窗长的缓存，请显式指定：

```powershell
scdfc precompute --config configs/default.yaml --windows 83 --overwrite
```

### 步骤 4：训练 FC 自编码器

```powershell
scdfc train-ae --config configs/default.yaml --window 83
```

自编码器将 4005 条 FC 上三角边编码为 256 维潜变量，再解码回边空间。检查点写入：

```text
outputs/window_83/fc_autoencoder.pt
```

训练 dFC 时，FC 编码器被冻结；FC 解码器在默认前 20 个 epoch 冻结，之后以更小学习率微调。

### 步骤 5：训练主模型与学习型基线

```powershell
# 主模型
scdfc train --config configs/default.yaml --window 83 --model tcn
scdfc train --config configs/default.yaml --window 83 --model transformer

# 学习型基线
scdfc train --config configs/default.yaml --window 83 --model direct_mlp
scdfc train --config configs/default.yaml --window 83 --model gcn_gru
```

模型输出目录格式为：

```text
outputs/window_83/<model>_<ablation>/best.pt
```

例如主 TCN 的检查点为：

```text
outputs/window_83/tcn_full/best.pt
```

### 步骤 6：SC 贡献消融

`FC1-only` 是最重要的对照：它输入首窗 FC，但移除个体 SC 信息。主模型只有在长时距个体残差指标上优于它，才能支持 SC 提供增量信息的结论。

```powershell
scdfc train --config configs/default.yaml --window 83 --model tcn --ablation fc1_only
scdfc train --config configs/default.yaml --window 83 --model tcn --ablation mean_sc
scdfc train --config configs/default.yaml --window 83 --model tcn --ablation shuffled_sc
scdfc train --config configs/default.yaml --window 83 --model tcn --ablation sc_only
```

消融含义：

| 参数 | 含义 |
| --- | --- |
| `full` | SC + 首窗 FC + run，主模型 |
| `fc1_only` | 首窗 FC + run；SC 输入置零 |
| `sc_only` | SC + run；首窗 FC 置零 |
| `mean_sc` | 使用训练集平均 SC |
| `shuffled_sc` | 将 SC 与被试错配 |

### 步骤 7：评价与成功门槛

```powershell
scdfc evaluate --config configs/default.yaml --window 83 `
  --checkpoint outputs/window_83/tcn_full/best.pt `
  --baseline-checkpoint outputs/window_83/tcn_fc1_only/best.pt `
  --save-predictions
```

评价结果写入检查点目录下的 `evaluation.json`。指定 `--save-predictions` 后，每个测试样本还会保存：

- Fisher-z 边预测和真实标签；
- 原始重建 FC 矩阵；
- 最近相关矩阵投影版；
- PSD 投影误差。

## 6. 评价指标与结果解释

主指标为 `long_residual_pearson`：

1. 只取首窗与目标窗口不再共享 BOLD 样本的长时距区间；
2. 分别从预测和真实序列中减去训练集群体模板；
3. 计算每个窗口的边模式 Pearson 相关并平均。

因此，单纯复制群体平均 dFC 不会取得高主指标分数。

`evaluation.json` 同时包含：

| 指标 | 含义 |
| --- | --- |
| `mse` / `mae` | Fisher-z 上三角边的重建误差 |
| `raw_edge_pearson` / `raw_edge_spearman` | 未去除群体模板的边模式相关 |
| `long_residual_pearson` | 主指标，个体化长时距边相关 |
| `difference_mse` | 相邻窗口变化量误差 |
| `variance_mae` | 各边时间方差差异 |
| `fcd_pearson` / `fcd_wasserstein` | FCD 矩阵与其分布的相似性 |
| `state_*_mae` | 动态状态占有率、转移和停留时间误差 |
| `retrieval_top1` / `retrieval_top5` | 预测未来对本人真实未来的检索表现 |
| `projection_*` | 预测矩阵 PSD 违规比例与投影改变量 |

当提供 `--baseline-checkpoint` 时，报告会额外给出以被试为重采样单位的 2000 次 bootstrap 差异置信区间。若同一被试有 LR/RL，两个 run 会先聚合为该被试的一项差异：

```json
"success_gate": {
  "mean_difference": 0.012,
  "ci_low": 0.004,
  "ci_high": 0.021,
  "passes": true
}
```

只有 `ci_low > 0` 时，`passes` 才为 `true`，表示主模型在主指标上可靠优于指定的基线。

## 7. 关键配置项

| 配置路径 | 默认值 | 说明 |
| --- | ---: | --- |
| `data.window_length` | 83 | 主分析滑窗长度（TR） |
| `data.stride` | 5 | 滑窗步长（TR） |
| `split.train/val/test` | 0.70/0.15/0.15 | 被试级分区比例 |
| `model.fc_latent_dim` | 256 | FC 自编码器潜变量维度 |
| `model.hidden_dim` | 256 | 时序解码器隐藏维度 |
| `model.tcn_dilations` | 1–32 | TCN 膨胀卷积感受野设置 |
| `training.batch_size` | 4 | dFC 序列训练批大小 |
| `training.patience` | 20 | 验证集早停耐心值 |
| `evaluation.bootstrap_replicates` | 2000 | 被试 bootstrap 次数 |

建议将每个实验复制一份配置文件，例如 `configs/tcn_83.yaml`，并将实际的预处理、随机种子、窗长、模型和损失设置与结果一同保存。

## 8. 常见问题

### `ROI order mismatch`

时间序列列名或顺序与 AAL90 标签不一致。不要只重命名列；应确认 SC、ROI BOLD 和 AAL 标签是否来自完全相同的分区定义与节点顺序。

### 没有 RL 数据能否先运行？

可以。缺少 RL 不会阻止 LR 管线运行，但会降低同一被试重复 run 的评估能力。

### 模型输出接近组平均、个体差异很弱

请优先检查：

1. `long_residual_pearson` 是否优于 `FC1-only`；
2. `variance_mae`、`difference_mse` 和 FCD 是否明显变差；
3. warm-up 后的短时距与无重叠长时距是否被混在一起报告；
4. 是否使用了正确的训练集群体模板和被试级划分。

不要仅凭较高的 `raw_edge_pearson` 声称存在个体化预测，因为群体共同 FC 成分通常很强。

### 显存不足

先将 `training.batch_size` 从 4 减到 2 或 1；不要修改 FC 边数或 AAL90 节点顺序。也可先训练 TCN，再训练 Transformer。默认 90 节点、83 TR 主分析在单张 8 GB GPU 上设计为可运行。

## 9. 开发与测试

运行全部单元测试：

```powershell
conda activate GCN_mri
pytest
```

测试覆盖矩阵上三角往返、滑窗 FC 计算、相关矩阵投影、被试级划分、Zarr 缓存、TCN/Transformer 输出形状、复合损失反传、动态评价与检索逻辑。

## 10. 当前边界与后续工作

- 当前仅输出单条确定性未来轨迹；
- 主分析不加入年龄、性别、头动等协变量；
- 默认 FC 重建通过对称化和单位对角保证矩阵形式，PSD 以软惩罚和后处理投影监控；
- 概率生成、条件扩散、神经 SDE、多尺度联合目标和协变量增量实验是后续阶段。

如果用本项目开展正式研究，请在论文或报告中单独说明数据许可、HCP 预处理版本、ROI 提取流程、被试级划分、所有窗长、模型选择规则和未通过的消融结果。
