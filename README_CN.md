# SC-dFC：基于结构连接与首窗 FC 的动态功能连接预测

SC-dFC 是一个用于静息态 fMRI 的确定性预测框架。给定某名被试的结构连接矩阵（SC）和该 run 的第一个动态功能连接窗口（FC warm-up），模型预测后续所有动态功能连接（dFC）窗口，并重建为完整的相关矩阵序列。

项目当前面向 AAL90 分区和 HCP 风格的 ROI BOLD 时间序列实现；模型、数据缓存和评价逻辑均为配置驱动。

> 数据与发布边界：原始 MRI/BOLD、逐被试 SC/FC、行为表、私人划分清单和可重算缓存不得提交。仓库目前确实跟踪了部分实验配置、指标、日志、可视化、含被试标识的评价记录，以及若干 Git LFS checkpoint 指针；不能把它描述成“只有代码和文档”。继续发布任何逐被试结果前需核查数据许可和隐私。checkpoint 的实际归档状态见[归档说明](docs/checkpoint_archive.md)。

## 1. 研究任务

主任务（单窗口 warm-up，K=1）为：

$$
(SC_s, FC_{s,1}) \longrightarrow \hat{FC}_{s,2:T}
$$

其中：

- $SC_s$：第 $s$ 名被试的 `90×90` 加权、对称结构连接矩阵；
- $FC_{s,1}$：同一 fMRI run 的第一个 dFC 窗口；
- $\hat{FC}_{s,2:T}$：预测得到的后续完整 `90×90` dFC 矩阵序列。

当前数据版本只使用 LR 时间序列。`LR` 只是数据来源名称，不会转换为数值、one-hot 或 embedding，也不会作为模型条件。原始 RL 文件可以继续保存在本地，但当前配置、审计、划分、缓存、训练和评价均不会读取它们。

本版本是**确定性条件预测**：同一输入只输出一条后续轨迹。它评估的是 SC 和当前功能状态能否约束后续 dFC，而不是完整建模 $p(dFC\mid SC,FC_1)$。条件扩散、流匹配或状态空间生成模型属于后续扩展。

E0014/E0015 还实现了 K=1/K=5 的 warm-up GRU 对照：输入严格为前 K 个 FC 窗口，标签从 `FC[K:]` 开始。比较不同 K 时必须按共同的绝对预测时间对齐。当前实验进度和已知局限见[项目状态](docs/project_status.md)。

## 2. 方法概览

```text
SC matrix ── GCN 或 Hybrid(Attention + edge MLP) ─┐
FC warm-up ── frozen E0003 FC encoder ────────────┼─ 256-d global condition
                                                   ▼
                                  GRU 或时间 Transformer → FC latent sequence
                                                   ▼
               frozen E0003 reconstruction decoder → Fisher-z edges → 完整 90×90 FC 序列
```

E0004–E0007 不再叠加群体模板或额外的 4005 维 static head；最终 Fisher-z FC 直接来自冻结的 E0003 reconstruction decoder。E0008 使用 `direct_edge_linear` 输出头，但同时把历史 Huber 改为 MSE，因此现有 E0004↔E0008 结果不能视作严格的“只改 decoder”单因素对照。

### 2.1 从原始文件到训练批次：形状与处理方法

下表以主分析设置 `window_length=83`、`stride=5`、AAL90、1200 个 TR 为例。所有统计量只在训练集拟合；验证集和测试集只使用已拟合的参数。

| 阶段 | 输入形状 | 处理 | 输出形状 |
| --- | --- | --- | --- |
| 原始 SC | `90×90` | 检查对称性；上三角向量化；`log1p`；按训练集逐边标准化 | 图分支：`90×90`；边分支：`4005` |
| 原始 BOLD | `1200×90` | 矩形滑窗 Pearson 相关；去对角线；Fisher-z | `224×4005` |
| warm-up/标签 | `224×4005` | 第 1 窗作为条件，其余窗口作为标签 | `FC1: 4005`；未来：`223×4005` |
| FC 自编码器 | `4005` | 编码、解码 Fisher-z 上三角边 | 潜变量：`256`；重建：`4005` |
| 主模型 | SC + FC1 | 条件编码后并行生成全部未来时距 | 潜轨迹：`223×256` |
| 最终输出 | `223×4005` | `tanh`、上下三角填充、单位对角线 | `223×90×90` |

#### SC 处理

每个被试的 SC 文件首先保留原始 `90×90` 矩阵，供图注意力分支使用。模型从它计算每个 ROI 的节点强度和非零连接度，并叠加可学习的 ROI 身份 embedding。

同时，SC 上三角被向量化为 4005 条无向边：

$$
x^{SC}_s=\mathrm{zscore}_{\mathrm{train}}(\log(1+SC_{s,\mathrm{upper}}))
$$

该向量进入独立的 MLP 分支，以保留图消息传递可能平滑掉的全局边模式。训练集保存 `sc_mean` 和 `sc_std`；推理接口在未显式传入边向量时会自动使用它们标准化。

#### BOLD 到 dFC

程序不会使用工作区中已有的静态 FC CSV 作为监督标签，而是直接从 `data/raw/timeseries_lr` 的 ROI BOLD 重新计算 dFC。对第 `k` 个滑窗起点 `a_k=k×5`：

$$
FC_k=\mathrm{corr}(BOLD[a_k:a_k+83,:])
$$

随后取上三角并做 Fisher-z 变换：

$$
z_k=\mathrm{arctanh}(\mathrm{clip}(FC_{k,\mathrm{upper}},-0.999999,0.999999))
$$

1200 个 TR 在 83 TR 窗长、5 TR 步长下得到 224 个窗口。第一个窗口 `z_0` 作为 warm-up；`z_1` 到 `z_223` 是模型必须预测的 223 个未来标签。每个窗长独立缓存在：

```text
data/cache/dfc/window_83.zarr/
└── subjects/<subject_id>/LR/
    ├── fc_z             # [224, 4005]，float32
    └── window_starts    # [224]
```

这种离线缓存避免每个 epoch 重复计算数十万次滑窗相关。缓存保存生成参数哈希；若同一目录采用不同窗长、步长或估计器，必须用 `--overwrite` 明确重建。

#### 训练集统计量与标签构造

在训练分区中，程序聚合每个 subject/run 的未来窗口 `z_1:T`，计算：

- `sc_mean`、`sc_std`：SC 上三角的逐边标准化参数；
- `fc_mean`、`fc_std`：未来 FC 边的描述性统计；
- `group_template[t,e]`：训练集在未来时距 `t` 的群体平均 FC，形状为 `223×4005`。

群体模板只用训练集计算，用于 group-mean 基线和“个体残差”等评价/可选损失；当前 E0004–E0024 的条件模型不会把它加到最终输出上。未来标签的长度若由 K=1 改为 K>1，模板会相应截取对齐。

### 2.2 模型内部的数据流

#### FC 自编码器

FC 自编码器的默认结构为：

```text
4005 → Linear(1024) → LayerNorm → GELU → Dropout
     → Linear(512)  → LayerNorm → GELU
     → Linear(256)  → LayerNorm
     → Linear(512)  → LayerNorm → GELU
     → Linear(1024) → LayerNorm → GELU → Dropout
     → Linear(4005)
```

它先在训练窗口上预训练，用于把高维 FC 边模式压缩为 256 维潜变量。E0004–E0007 训练时，E0003 的 FC encoder 和 reconstruction decoder 全程冻结并保持 eval 状态；只训练 SC encoder、条件融合和时序模型。

#### 条件编码器

主模型将三类信息融合为 256 维条件向量：

1. **SC 图分支**：ROI embedding、节点强度和节点度经过 3 层结构偏置 Graph Attention；正 SC 权重经 `log1p` 加入每个注意力头的 score 偏置。
2. **SC 边分支**：标准化后的 4005 条 SC 边经过 `4005→512→128` MLP。
3. **首窗 FC 分支**：`FC1` 经预训练 FC 编码器映射为 256 维状态。
`model.sc_encoder` 控制 SC 编码方式：默认 `hybrid` 使用上述图分支与边分支；`hcp_gcn` 使用单位矩阵 ROI 特征、$D^{-1/2}(A+I)D^{-1/2}$ 对称归一化、两层 `90→128→64` GCN 和 max pooling。两种编码器随后都投影到相同条件维度，并共享 FC1、时序解码器和损失函数，以便公平比较。

三者拼接后经门控融合：

$$
c=\mathrm{Linear}(u)\odot\sigma(\mathrm{Linear}(u))
$$

其中 `c` 是统一的 256 维全局条件。它用于 GRU 的时间输入和初始 hidden state，或加到 Transformer 的每个未来时间 query。

#### 两种并列时序解码器

- **GRU**：2 层、hidden size 256；对可学习的未来时间 query 进行序列建模。
- **Transformer**：4 层、256 维、8 头、FFN 1024；对未来时间 query 做时间 self-attention，不使用 SC token cross-attention。

两者均为**非自回归**：一次性输出全部未来窗口，不把真实未来 FC 输入给模型，不使用 teacher forcing 或 scheduled sampling，因此训练和测试条件完全一致。

#### 从潜轨迹恢复 FC

时序模型输出潜轨迹 `q[t]`，冻结的 E0003 reconstruction decoder 将其直接映射为边空间：

$$
\hat z_t=decoder_{E0003}(q_t)
$$

最后经 `tanh` 回到相关系数范围，再填充上下三角并将对角线固定为 1。输出始终对称且对角为 1，但不保证半正定。PSD 惩罚是可选损失，目前主要实验未启用；评价阶段会报告最近相关矩阵投影及改变量。

### 2.3 损失函数设计

损失由每个实验冻结配置中的 `training.loss_weights` 组合，**最多同时启用三项非零损失**，并没有一个固定的八项“主模型总损失”。默认调试配置为 `edge: 1.0`、`difference: 0.25`；E0004–E0007 的历史 checkpoint 使用同权重的 Huber/Smooth L1；E0018、E0022–E0024 则使用 `difference: 1.0`、`variance: 1.0` 的 MSE 动态压力测试。不同定义的 `objective_loss` 不可直接横向排名。具体实验应查看其 `config_resolved.yaml` 和 `metrics_best.json.metric_definition`。

例如，E0024 的训练目标是：

$$
L_{\mathrm{E0024}}=L_{\mathrm{diff,MSE}}+L_{\mathrm{var}}.
$$

| 可选损失 | 配置键 | 实现方式 | 作用与当前状态 |
| --- | --- | --- | --- |
| $L_{\mathrm{edge}}$ | `edge` | Fisher-z 边的 MSE 或 Huber | 每窗连接重建；E0004 启用，E0024 未启用。 |
| $L_{\mathrm{residual}}$ | `residual_corr` | 无重叠时距、减群体模板后的 $1-\mathrm{Pearson}$ | 个体边模式约束；当前主要实验未启用。 |
| $L_{\mathrm{diff}}$ | `difference` | 相邻预测窗一阶差分的 MSE 或 Huber | E0004 和 E0024 均启用；不包含最后 warm-up 窗到首个预测窗的边界差分。 |
| $L_{\mathrm{static}}$ | `static` | 预测/真实序列时间均值的 MSE 或 Huber | 个体平均水平锚点；当前主要实验未启用。 |
| $L_{\mathrm{var}}$ | `variance` | 逐边时间方差的 MSE | E0024 启用；只约束幅度，不保证相位或个体性。 |
| $L_{\mathrm{long-var}}$ | `long_horizon_variance` | 无重叠区间分三段的逐边方差 MSE | E0011 曾测试。 |
| $L_{\mathrm{FCD}}$ | `fcd` | 最多抽样 32 窗的归一化边向量 Gram 矩阵 MSE | 已实现，当前主要实验未启用。 |
| $L_{\mathrm{contrast}}$ | `contrastive` | 批内长时距平均边向量 InfoNCE | 已实现，当前主要实验未启用。 |
| $L_{\mathrm{PSD}}$ | `psd` | 最多抽样 4 窗的负特征值平方 | 已实现，当前主要实验未启用。 |

令 $p_{b,t}\in\mathbb{R}^{E}$、$y_{b,t}\in\mathbb{R}^{E}$ 分别表示第 $b$ 个样本、未来第 $t$ 个窗口的预测与真实 Fisher-z 上三角边向量，$g_t$ 表示训练集群体模板，$E=4005$；$\operatorname{MSE}(a,b)$ 表示所有元素的平均平方误差。下式写出 MSE 版本；对设置了 `loss_type: huber` 的历史训练，`edge`、`difference`、`static` 的逐点 MSE 换成 Smooth L1。令 $\tau$ 为保守的无重叠区间起点（K=1、83 TR 主分析中为未来标签索引 17）：

$$
L_{\mathrm{edge}}=\operatorname{MSE}(p_{b,t},y_{b,t})
$$

$$
L_{\mathrm{residual}}
=\underset{b,\,t\ge\tau}{\operatorname{mean}}
\left[1-\operatorname{corr}_e\left(p_{b,t}-g_t,\;y_{b,t}-g_t\right)\right]
$$

$$
L_{\mathrm{diff}}
=\operatorname{MSE}\left(p_{b,t}-p_{b,t-1},\;y_{b,t}-y_{b,t-1}\right)
$$

$$
L_{\mathrm{static}}
=\operatorname{MSE}\left(\frac{1}{T}\sum_t p_{b,t},\;\frac{1}{T}\sum_t y_{b,t}\right)
$$

$$
L_{\mathrm{var}}
=\operatorname{MSE}\left(\operatorname{Var}_t(p_{b,t}),\;\operatorname{Var}_t(y_{b,t})\right)
$$

其中 $\operatorname{corr}_e$ 是在边维度 $e$ 上计算的 Pearson 相关；`difference` 对 $t=1,\ldots,T-1$ 求均值，`variance` 使用总体方差（`unbiased=False`）。其余三项的实现细节为：

$$
\tilde p_{b,i}
=\frac{p_{b,s_i}-\operatorname{mean}_e(p_{b,s_i})}
 {\left\|p_{b,s_i}-\operatorname{mean}_e(p_{b,s_i})\right\|_2},
\qquad
\tilde y_{b,i}
=\frac{y_{b,s_i}-\operatorname{mean}_e(y_{b,s_i})}
 {\left\|y_{b,s_i}-\operatorname{mean}_e(y_{b,s_i})\right\|_2}
$$

$$
L_{\mathrm{FCD}}=\operatorname{MSE}(\tilde P\tilde P^\top,\;\tilde Y\tilde Y^\top)
$$

$$
a_b=\frac{\operatorname{mean}_{t\ge\tau}p_{b,t}}
 {\left\|\operatorname{mean}_{t\ge\tau}p_{b,t}\right\|_2},
\qquad
b_j=\frac{\operatorname{mean}_{t\ge\tau}y_{j,t}}
 {\left\|\operatorname{mean}_{t\ge\tau}y_{j,t}\right\|_2}
$$

$$
L_{\mathrm{contrast}}
=\frac{1}{2}\left[
\operatorname{CE}\left(\frac{AB^\top}{0.1},\operatorname{diag}\right)
+\operatorname{CE}\left(\frac{BA^\top}{0.1},\operatorname{diag}\right)
\right]
$$

$$
L_{\mathrm{PSD}}
=\operatorname{mean}_{b,i,k}
\left[\max\left(0,-\lambda_k\left(C(p_{b,r_i})\right)\right)^2\right]
$$

这里 $s_i$ 是从全序列均匀抽取的至多 32 个窗口，$\tilde P$、$\tilde Y$ 的行分别是对应窗口的标准化边向量，因此其 Gram 矩阵是 FCD 的近似；$A$、$B$ 的行分别为 $a_b$、$b_b$，`diag` 表示 batch 内同一被试预测/真实样本为正对；$r_i$ 是至多 4 个均匀抽样窗口；$C(\cdot)$ 将预测边先经 $\tanh$ 变为相关系数，再恢复为对称矩阵并令对角线为 1。

对于 K=1、83 TR 主窗，理论上未来标签索引 16 起已不再与首窗共享 BOLD 样本。当前实现采用保守切片 `[:, 17:]`，即从未来标签索引 17 开始计算 `residual_corr` 与 `long_residual_pearson`。无 BOLD 样本重叠不等于统计独立；相邻窗口仍有强自相关。

### 2.4 训练、验证与检查点

训练分两阶段：

1. **FC 自编码器阶段**：训练集中每个 subject/run 可复现地抽取 32 个窗口，验证集抽取 8 个窗口；E0003 的有效配置仅启用 `autoencoder_loss_weights.edge: 1.0`，即 Fisher-z 边 MSE。相关与 PSD 自编码器损失虽已实现，但 E0003 未启用。
2. **序列预测阶段**：每个 batch 包含完整未来序列，不泄漏未来 FC；预训练 FC 编码器与 E0003 重建解码器在整个序列训练过程中均保持冻结。使用 `direct_edge_linear` 输出头的实验不经过 E0003 重建解码器，该线性输出头自身参与训练。

默认优化与稳定策略：

- `AdamW`，学习率 `3e-4`，weight decay `1e-4`；
- 主模型 batch size 为 4，FC 自编码器 batch size 为 256；
- 全局梯度范数裁剪为 1.0；
- 最多训练 200 epoch，FC 自编码器最多 100 epoch；
- 默认 `patience=20`；固定轮数压力测试可在冻结配置中覆盖该值；
- 最佳 checkpoint 按实验声明的 `evaluation.primary_metric` 在验证集选择：E0003 为 `validation_loss`，E0004–E0024 序列实验主要为 `objective_loss`。默认非受管理配置才使用 `long_residual_pearson`；
- `seed` 同时固定 Python、NumPy 和 PyTorch 随机源。

受管理运行的最佳检查点及选模记录保存为：

```text
outputs/E####/runs/<run_id>/checkpoints/best.pt
outputs/E####/runs/<run_id>/metrics_best.json
```

未受管理的旧 `train-ae/train` 调试命令仍使用 `outputs/window_83/`，其产物不能替代受管理实验记录。消融类型保存在冻结配置、run 元数据和 checkpoint 中，不由受管理运行的目录名推断。

## 3. 目录与数据要求

默认配置文件为 [configs/default.yaml](configs/default.yaml)。请在项目根目录准备如下结构：

```text
SC2dFC/
├── data/
│   ├── raw/
│   │   ├── atlas/ROI_MNI_V4.txt
│   │   ├── sc/HCP_Structure/AAL90/<subject_id>.csv
│   │   └── timeseries_lr/<subject_id>_AAL90_timeseries.csv
│   ├── manifests/                       # 私有数据清单、审计和冻结划分
│   └── cache/dfc/                        # 可重新计算的 Zarr dFC 缓存
├── outputs/                              # 训练统计量、run、checkpoint、评价和图
├── reports/                              # 注册表、实验总结和研究记录
├── configs/default.yaml
└── src/
```

### 3.1 SC 矩阵

- 文件名：`data/raw/sc/HCP_Structure/AAL90/<subject_id>.csv`；
- 格式：无表头的 `90×90` CSV；
- 要求：数值有限、对称、对角线为零或接近零；
- 不在程序中进行阈值化；训练时对 SC 上三角做 `log1p`，再按训练集逐边标准化。

### 3.2 BOLD ROI 时间序列

- 文件名：`data/raw/timeseries_lr/<subject_id>_AAL90_timeseries.csv`；
- 默认形状：`1200×91`，第一列为 `timepoint`，后 90 列为 AAL90 ROI；
- 要求：ROI 名称与 `ROI_MNI_V4.txt` 前 90 个标签的顺序严格一致；
- 默认假设 HCP TR 为 0.72 秒，时间序列来自 HCP minimal preprocessing + ICA-FIX；如不符合，请修改配置并记录实际预处理。

## 4. 安装环境

推荐使用已具备 CUDA PyTorch 的 Conda 环境：

```powershell
conda activate GCN_mri
python -m pip install -e ".[dev]"
```

主要依赖：Python ≥3.11、固定为 2.6.0 的 PyTorch、NumPy、Pandas、SciPy、scikit-learn、Zarr 2.x、PyYAML 和 pytest。`GCN_mri` 是文档示例环境名，不由仓库自动提供；需先创建该环境，或激活符合依赖要求的其他环境。

验证安装：

```powershell
scdfc --help
pytest
```

> Windows 下如果 `scdfc` 命令不可用，可使用 `python -m scdfc.cli` 替代。例如：`python -m scdfc.cli audit --config configs/default.yaml`。

## 5. 完整运行流程

> **正式科研实验请先阅读 [`docs/experiment_management.md`](docs/experiment_management.md)。** 当前章节中的旧训练命令仅保留用于 Level 0 调试；需要形成可追溯结果时，应使用 `scdfc experiment create`、`scdfc run`、`scdfc evaluate-run`、`scdfc summarize` 和 `scdfc conclude`。

所有命令都在项目根目录执行。下述 `train-ae/train/evaluate` 属于旧的 Level 0 调试入口；正式实验应按[受管理流程](docs/experiment_management.md)使用冻结配置。建议先使用 83 TR 完成主分析，再单独运行 42 和 125 TR 敏感性分析。

### 步骤 1：数据审计

```powershell
scdfc audit --config configs/default.yaml
```

审计报告写入 `outputs/audit.json`，包括：SC/LR 时间序列数目、可配对被试数、ROI 顺序、矩阵形状、有限值和 SC 对称性。

在执行后续步骤前，应确保 `errors` 为空。当前配置不会检查或要求 RL 数据。

### 步骤 2：确认被试级冻结划分

当前默认配置使用 `data/manifests/split_lr_v1.csv`，包含 738/158/159 名 train/val/test 被试。不要用旧 `scdfc split` 重建它；该命令在目标文件已存在时会拒绝覆盖。若数据或划分规则发生变化，应使用 `scdfc freeze-data` 创建新的数据与划分版本，具体命令见实验管理操作手册。

### 步骤 3：离线计算 dFC 缓存

```powershell
scdfc precompute --config configs/default.yaml --windows 83
# 仅在开展窗长敏感性分析时：
scdfc precompute --config configs/default.yaml --windows 42 125
```

该步骤使用矩形窗 Pearson 相关，取上三角并 Fisher-z 变换，写入 `data/cache/dfc/window_<window_length>.zarr`。训练阶段只读取这些缓存，**不会在线计算滑窗相关**。

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

上述是旧调试路径；正式 E0003 使用 `scdfc run --experiment configs/experiments/E0003_fc_autoencoder_w83_v1.yaml --seed 42`，并将 checkpoint 记录在其独立 run 目录中。E0004–E0024 序列训练中的 FC encoder 和 reconstruction decoder 均全程冻结；`direct_edge_linear` 输出实验则绕开该重建 decoder。

### 步骤 5：训练主模型与学习型基线

```powershell
# 旧 Level 0 调试入口；正式运行请使用各 E#### 冻结配置与 scdfc run
scdfc train --config configs/default.yaml --window 83 --model gru --sc-encoder hcp_gcn
scdfc train --config configs/default.yaml --window 83 --model gru --sc-encoder hybrid
scdfc train --config configs/default.yaml --window 83 --model transformer --sc-encoder hcp_gcn
scdfc train --config configs/default.yaml --window 83 --model transformer --sc-encoder hybrid

# 学习型基线
scdfc train --config configs/default.yaml --window 83 --model direct_mlp
scdfc train --config configs/default.yaml --window 83 --model gcn_gru
```

旧调试入口的模型输出目录格式为：

```text
outputs/window_83/<model>_<ablation>/best.pt
outputs/window_83/<model>_<sc_encoder>_<ablation>/best.pt
```

例如主 TCN 的检查点为：

```text
outputs/window_83/tcn_full/best.pt
outputs/window_83/tcn_hcp_gcn_full/best.pt
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
| `full` | SC + 首窗 FC，主模型 |
| `fc1_only` | 仅首窗 FC；SC 输入置零 |
| `sc_only` | 仅 SC；首窗 FC 置零 |
| `mean_sc` | 使用训练集平均 SC |
| `shuffled_sc` | 将 SC 与被试错配 |

### 步骤 7：评价与成功门槛

正式运行用 `scdfc evaluate-run --run-id <run_id> --split val` 评价，另用 `scdfc dynamic-audit --run-id <run_id> --split val` 审计动态。Level 0/1 受管理入口不能评价 test；仅预先确认的 Level 2 运行能通过一次性 `--final-test` 访问测试集。

下面的 `evaluate` 是**仅用于 train/val 的旧调试命令**：

```powershell
scdfc evaluate --config configs/default.yaml --window 83 `
  --checkpoint outputs/window_83/tcn_full/best.pt `
  --baseline-checkpoint outputs/window_83/tcn_fc1_only/best.pt `
  --save-predictions
```

该示例默认评价 `val`，结果写入检查点目录下的 `evaluation_val.json`。指定 `--save-predictions` 后，导出的是当前所选 split 的逐样本预测，可能包含受限被试数据，不应发布；它并非正式测试集评价。导出内容包括：

- Fisher-z 边预测和真实标签；
- 原始重建 FC 矩阵；
- 最近相关矩阵投影版；
- PSD 投影误差。

## 6. 评价指标与结果解释

E0004–E0007 的 checkpoint 按验证集 `objective_loss` 最小选择；其历史训练使用 Huber/Smooth L1：

$$
L_{objective}=L_{Huber(edge)}+0.25L_{Huber(first\ difference)}.
$$

E0018/E0022–E0024 使用 MSE 差分＋方差且不启用 edge 项；E0019–E0021 的权重又各不相同。训练选模目标见 `metrics_best.json`，通用评价见 `evaluation_<split>.json`，二者不可混用。`long_residual_pearson` 是独立诊断指标；这些已登记序列实验并未用它选模。详细指标口径及历史修复见[修复记录](reports/research/2026-09-07/指标口径修复结果.md)。

`evaluation_<split>.json` 还包含：

| 指标 | 含义 |
| --- | --- |
| `objective_loss` | 可重放 checkpoint 时，按该实验完整 `CompositeLoss` 计算的目标；历史不可重放时明确标为不可用 |
| `edge_difference_score` | 通用边误差＋加权差分误差；只是局部诊断分数，不一定等于训练目标 |
| `edge_huber` / `difference_huber` | Huber 边误差与差分误差，供历史实验对照 |
| `mse` / `mae` | Fisher-z 上三角边的重建误差 |
| `raw_edge_pearson` / `raw_edge_spearman` | 未去除群体模板的边模式相关 |
| `long_residual_pearson` | 诊断指标，个体化长时距边相关 |
| `node_strength_pearson` / `node_strength_mae` | 节点强度拓扑一致性 |
| `difference_mse` | 相邻窗口变化量误差 |
| `variance_mae` | 各边时间方差差异 |
| `fcd_pearson` / `fcd_wasserstein` | FCD 矩阵与其分布的相似性 |
| `state_*_mae` | 动态状态占有率、转移和停留时间误差 |
| `retrieval_top1` / `retrieval_top5` | 预测未来对本人真实未来的检索表现 |
| `projection_*` | 预测矩阵 PSD 违规比例与投影改变量 |

旧调试评价提供 `--baseline-checkpoint` 时，报告会额外给出以被试为重采样单位的 2000 次 bootstrap `long_residual_pearson` 差异置信区间：

```json
"success_gate": {
  "mean_difference": 0.012,
  "ci_low": 0.004,
  "ci_high": 0.021,
  "passes": true
}
```

只有 `ci_low > 0` 时，`passes` 才为 `true`。这只支持指定基线与该长时距残差相关指标的配对比较，不自动证明逐窗动态相位或 SC 因果贡献。

## 7. 关键配置项

| 配置路径 | 默认值 | 说明 |
| --- | ---: | --- |
| `data.window_length` | 83 | 主分析滑窗长度（TR） |
| `data.stride` | 5 | 滑窗步长（TR） |
| `split.train/val/test` | 0.70/0.15/0.15 | 被试级分区比例 |
| `model.fc_latent_dim` | 256 | FC 自编码器潜变量维度 |
| `model.hidden_dim` | 256 | 时序解码器隐藏维度 |
| `model.sc_encoder` | `hybrid` | SC 编码器：`hybrid` 或 `hcp_gcn` |
| `model.hcp_gcn_hidden_dim` | 128 | HCP_GCN 第一层隐藏维度 |
| `model.hcp_gcn_output_dim` | 64 | HCP_GCN 池化前节点表示维度 |
| `model.gru_layers` | 2（代码回退值） | GRU 层数；正式实验可覆盖 |
| `model.transformer_layers/heads` | 4/8 | 时间 Transformer 深度与头数 |
| `model.output_head` | `e0003_reconstruction_decoder`（代码回退值） | 冻结输出 decoder；E0008 使用 `direct_edge_linear` |
| `training.batch_size` | 4 | dFC 序列训练批大小 |
| `training.patience` | 20 | 验证集早停耐心值 |
| `training.loss_weights` | `edge: 1.0, difference: 0.25` | 默认调试目标；正式实验以冻结配置为准 |
| `evaluation.primary_metric` | `long_residual_pearson` | 默认调试选模指标；正式序列实验多数覆盖为 `objective_loss` |
| `evaluation.bootstrap_replicates` | 2000 | 被试 bootstrap 次数 |

正式研究应为每个新问题分配新的 `E####` 实验 ID，并在 `configs/experiments/` 冻结配置；不要修改已经产生 run 的配置来重解释历史结果。受管理 run 同时保存解析后的配置、代码版本、数据清单、环境和选模记录。

## 8. 常见问题

### `ROI order mismatch`

时间序列列名或顺序与 AAL90 标签不一致。不要只重命名列；应确认 SC、ROI BOLD 和 AAL 标签是否来自完全相同的分区定义与节点顺序。

### 为什么当前不读取 RL 数据？

当前正式版本固定为 LR-only，以避免把 HCP 特有的相位编码方向写进模型接口。RL 原始文件不会被删除，但不在 `configs/default.yaml` 中声明，因此不会进入审计、数据清单、划分、缓存、训练或评价。以后若重新纳入 RL，应创建新的数据/划分版本和实验 ID，并先决定它是额外样本、独立复现集还是域变量；不应恢复二值方向编码作为默认模型输入。

### 模型输出接近组平均、个体差异很弱

请优先检查：

1. 在同一验证集和共同预测区间，是否优于 group mean、FC1 persistence 及 FC1-only；
2. 除总体 `temporal_std_ratio` 外，`difference_std_ratio`、`difference_temporal_pearson` 和中频功率是否仍接近零；
3. 去时间均值后，不同被试的预测动态是否仍高度相似，以及交换 SC/FC 条件后输出是否发生有意义的变化；
4. warm-up 后的短时距与无重叠长时距是否分开报告，训练集模板与被试级划分是否正确。

不要仅凭较高的 `raw_edge_pearson` 或接近 1 的 `temporal_std_ratio` 声称存在个体化动态预测。E0024 的验证集总体幅度已接近真实，但差分时间相关和个体检索仍接近零；详见[项目状态](docs/project_status.md)。

### 显存不足

先将 `training.batch_size` 从 4 减到 2 或 1；不要修改 FC 边数或 AAL90 节点顺序。可先训练 GRU，再训练 Transformer。

## 9. 开发与测试

运行全部单元测试：

```powershell
conda activate GCN_mri
pytest
```

测试覆盖矩阵上三角往返、滑窗 FC、被试级划分、Zarr 缓存、模型形状与反传、损失组合、选模记录与评价隔离、动态指标、检索和受管理解析基线端到端流程。2026-09-07 的修复记录报告 `81 passed`；这是当时环境的结果，不能代替新环境中的重新运行。

## 10. 当前边界与后续工作

- 当前仅输出单条确定性未来轨迹；
- 主分析不加入年龄、性别、头动等协变量；
- FC 重建通过对称化和单位对角保证矩阵形式，但不保证半正定；PSD 软惩罚可选，当前主要实验以评价和后处理投影监控；
- 截至 E0024，仅完成 Level 1 单 seed 探索，模型仍存在跨被试公共轨迹、时间错位和弱 SC 条件敏感性；尚无通过多 seed、个体化消融与锁定测试确认的主结果；
- 下一轮优先验证静态个体锚点、SC/FC 条件消融与短时距/更多历史的可预测性；概率模型和多尺度目标应在这些诊断之后决定是否投入；
- 已发布的部分 `val_test` 可视化属于探索性测试集接触，不能再把这些测试结果当作未见过的最终确认集。

如果用本项目开展正式研究，请在论文或报告中单独说明数据许可、HCP 预处理版本、ROI 提取流程、被试级划分、所有窗长、模型选择规则和未通过的消融结果。
