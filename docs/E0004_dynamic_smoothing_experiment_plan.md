# E0004 动态过平滑改进实验计划

更新日期：2026-08-17

## 1. 背景与目标

当前 E0004 使用个体 SC 与第一个 FC 窗口作为条件，通过 GCN、门控融合和两层 GRU 预测完整未来 dFC 序列。现有结果表明，模型能够改善平均 FC 边结构，但对时间动态的恢复有限：

- E0004 验证集 `objective_loss = 0.09299`，优于 group-mean baseline 的 `0.09628`；
- E0004 `difference_mse = 0.0094509`，group-mean baseline 为 `0.0094617`，改善很小；
- E0004 `dynamic_amplitude_mae = 0.09514`，group-mean baseline 为 `0.09348`；
- E0004 `long_residual_pearson = 0.0835`，说明长时距个体动态仍较弱。

direct-edge 输出头实验的改善有限，因此下一阶段优先验证两个假设：

1. 当前损失函数缺少直接约束动态幅度或频谱结构的成分；
2. `SC + FC1` 不足以确定未来动态状态，需要多个初始 FC 窗口进行 warmup。

两条路线应先分别进行单因素实验。只有各自有效后，才组合多窗口输入与新损失函数。

## 2. 路线 A：增加动态损失

### 2.1 当前损失的局限

E0004 当前使用：

```text
L = L_edge + 0.25 L_difference
```

其中：

```text
L_edge = mean Huber(prediction[t,e] - target[t,e])

L_difference = mean Huber(
    (prediction[t,e] - prediction[t-1,e])
    - (target[t,e] - target[t-1,e])
)
```

E0004 最佳 checkpoint 中，`edge_huber = 0.09181`，`difference_huber = 0.004725`。差分项加权后约为 `0.001181`，只占总目标的约 1.27%。

同时，差分损失要求预测波动在具体时间点和方向上与真实序列一致。如果未来波动的相位不能由 `SC + FC1` 唯一确定，确定性模型会倾向于输出条件均值，即时间上较平滑的轨迹。

### 2.2 首选实验：逐边时间方差损失

对 batch 中每个样本和每条 FC 边，计算整个预测时段的时间方差：

```text
var_pred[b,e] = Var_t(prediction[b,t,e])
var_true[b,e] = Var_t(target[b,t,e])
```

方差损失为：

```text
L_variance = mean Huber(var_pred[b,e] - var_true[b,e])
```

该损失只要求预测具有合理的波动幅度，不要求每次波动的相位完全正确。项目已在 `src/scdfc/training.py` 中实现 `variance_loss`，可以直接通过配置启用。

推荐的第一个配置为：

```yaml
training:
  loss_weights:
    edge: 1.0
    difference: 0.25
    variance: 1.0
```

当前 `CompositeLoss` 最多允许三个非零成分，上述组合刚好满足限制。

### 2.3 方差损失权重

第一轮建议使用 `variance: 1.0`。如果需要做小范围权重比较，可以依次测试：

```text
0.3, 1.0, 3.0
```

权重不应只按原始损失数值选择，还应观察各损失对共享 GRU latent 的梯度大小。理想情况下，训练初期加权后的方差项可占总目标约 5%–10%，或其梯度范数达到 edge loss 梯度范数的约 10%–30%。

若权重过小，方差仍会坍缩；若权重过大，模型可能通过产生无意义噪声来满足方差约束，同时损害边重建和个体动态相关。

### 2.4 方差损失实验的评价指标

不同损失配置的 `objective_loss` 定义不同，因此不能仅比较 objective 数值。应统一报告：

- `edge_huber`；
- `difference_huber` 与 `difference_mse`；
- `variance_mae`；
- `dynamic_amplitude_mae`；
- 预测差分标准差与真实差分标准差的比值；
- `long_residual_pearson`；
- `fcd_pearson` 与 `fcd_wasserstein`；
- state occupancy、transition 和 dwell 指标。

建议把以下条件作为初步成功标准：

1. `variance_mae` 和动态幅度误差明显改善；
2. FCD 或长时残差相关至少一项同步改善；
3. `edge_huber` 相对 E0004 的恶化不超过约 2%；
4. 改善不是由高频随机噪声产生。

### 2.5 第二阶段：频域功率损失

若方差损失能够恢复动态幅度，但 FCD 或动态时间结构仍不理想，可进一步测试频域损失。

不建议直接比较复数 FFT，因为复数 FFT 同时约束幅度和相位，仍要求模型预测未来波动的准确时间位置。更合适的是比较去均值后的功率谱：

```text
x_centered[t,e] = x[t,e] - mean_t(x[t,e])
P[f,e] = abs(RFFT(HannWindow[t] * x_centered[t,e])) ** 2
```

频域损失可以定义为：

```text
L_frequency = mean Huber(
    log(1 + P_prediction[f,e])
    - log(1 + P_target[f,e])
)
```

实现注意事项：

- 排除频率 0，避免重复约束 FC 时间均值；
- 使用 Hann window，减少有限序列造成的频谱泄漏；
- 使用 `log1p`，避免少数高功率边支配梯度；
- 优先约束低频区间；
- 不要把每条边的功率谱归一化为总和 1，否则会丢失动态幅度信息；
- 当前 dFC 窗长约 59.76 秒、stride 约 3.6 秒，相邻窗口重叠约 94%，高频 dFC 成分的解释应保持谨慎。

推荐的损失实验顺序为：

1. `edge + difference + variance`；
2. `edge + difference + frequency`；
3. 若两者分别有效，再测试 `edge + variance + frequency`。

不要在第一轮同时加入 variance 和 frequency，否则无法确定改善来自哪一项。

## 3. 路线 B：多窗口 warmup

### 3.1 数据定义

当前 E0004 使用：

```text
input  = FC[0]
target = FC[1:]
```

多窗口 warmup 应改为：

```text
input  = FC[0:K]
target = FC[K:]
```

例如 `K=5`：

```python
fc_warmup = fc[:5]   # [5, 4005]
fc_future = fc[5:]   # [219, 4005]
```

配置中可增加：

```yaml
data:
  warmup_windows: 5
```

已经作为输入的 warmup 窗口不得再次进入预测标签。

### 3.2 推荐的 warmup encoder

不建议简单平均多个 FC embedding，因为平均会消除初始变化方向。推荐使用一个专门的 warmup GRU：

```text
K x 4005 FC edges
        |
        v
frozen E0003 FC encoder
        |
        v
K x 256 latent sequence
        |
        v
warmup GRU
        |
        v
256-d initial dynamic state
        |
        +---- gated fusion with SC embedding
        |
        v
original E0004 future GRU
        |
        v
future dFC sequence
```

伪代码：

```python
batch_size, warmup_steps, n_edges = fc_warmup.shape

latent = frozen_fc_encoder(
    fc_warmup.reshape(batch_size * warmup_steps, n_edges)
)
latent = latent.reshape(batch_size, warmup_steps, 256)

_, hidden = warmup_gru(latent)
warmup_state = hidden[-1]
```

随后用 `warmup_state` 替代当前单个 FC1 embedding，与 SC embedding 进行现有门控融合。未来预测仍使用 E0004 的 query-based GRU，不在这一实验中同时改成自回归预测。

### 3.3 隔离架构变化与信息变化

增加 warmup GRU 本身也会增加模型参数。如果只比较原 E0004 与 K=5，无法判断改善来自更多输入，还是来自新增 GRU。

因此应建立以下对照：

| 实验 | warmup 窗口数 | warmup encoder | 目的 |
|---|---:|---|---|
| E0004 | 1 | 无 | 原始基线 |
| K1-control | 1 | GRU | 测量 warmup encoder 本身的影响 |
| K5 | 5 | 与 K1 相同的 GRU | 隔离增加输入窗口的影响 |

主要因果比较应为 `K1-control vs K5`。`E0004 vs K1-control` 只用于说明新增 encoder 是否改变了基线能力。

### 3.4 warmup 窗口数

当前窗口长 83 TR、stride 5 TR，相邻窗口重叠约 94%。K 个 warmup 窗口实际覆盖的原始时间为：

```text
83 + (K - 1) * 5 TR
```

因此：

- K=1：59.76 秒；
- K=5：74.16 秒；
- K=10：92.16 秒。

推荐先测试 K=5。如果有效，再测试 K=10；如果 K=5 完全无效，也可直接测试 K=10，不必逐一尝试 K=2、3、4。

虽然多个 FC 窗口增加的独立原始时间有限，但它们能够提供 FC 的初始变化方向：

```text
FC[1] - FC[0]
FC[2] - FC[1]
...
FC[K-1] - FC[K-2]
```

这是单个 FC1 无法提供的信息。

### 3.5 预测长度和模板对齐

当前模型默认按 `group_template` 长度生成全部未来窗口。引入 K 个 warmup 窗口后，目标长度变为 `224 - K`，训练和评价时必须显式使用目标长度：

```python
steps = batch["fc_future"].shape[1]
output = model(..., steps=steps)
```

同时需要处理 group template 的偏移。若现有 template 对应原始 `FC[1:]`，则 K-window 模型应使用：

```text
group_template[(K - 1):]
```

否则预测、目标和群体模板会错位。

### 3.6 公平评价与重叠窗口

K=1 与 K=5 的预测起点和预测长度不同，不能直接对各自全部未来窗口取平均后比较。应在相同绝对时间后缀上评价。

例如最大 warmup 为 K=10 时，所有模型都只评价 `FC[10:]`：

```text
K=1 model:  丢弃预测 FC[1:10]
K=5 model:  丢弃预测 FC[5:10]
K=10 model: 从第一步预测开始评价
```

另外，相邻 dFC 窗口共享大量原始 BOLD 时间点。83 TR 窗长、5 TR stride 意味着约 17 个窗口后，目标 FC 才与最后一个 warmup FC 不再共享原始时间点。因此应分别报告：

- 全预测区间指标；
- 相对最后一个 warmup 窗口至少 17 步后的 non-overlap 指标；
- 所有模型共同绝对时间后缀上的指标。

长期预测能力的主要结论应以后两类指标为准。

### 3.7 需要修改的代码位置

实现多窗口 warmup 时预计涉及：

- `src/scdfc/data.py`：返回 `[K, 4005]` 的 `fc_warmup` 和 `fc[K:]` 标签；
- `src/scdfc/models/sequence.py`：增加 warmup GRU，并让 `ConditionEncoder` 接收 FC 序列；
- `src/scdfc/training.py`：根据标签长度传入 `steps`；
- `src/scdfc/evaluation.py`：对齐预测、标签、group template 和共同评价后缀；
- 配置与管理代码：登记 `warmup_windows`；
- 单元测试：验证 K=1/K=5 shape、目标切片、模板偏移和无数据泄漏。

## 4. 推荐实验矩阵

第一阶段保持 E0004 的 GCN、GRU、数据划分、seed 和输出头不变：

| 阶段 | 实验 | 唯一主要变化 |
|---|---|---|
| A0 | E0004 baseline | 无 |
| A1 | variance-v1 | 增加 `variance: 1.0` |
| A2 | variance-low/high | 仅在 A1 有趋势但权重不合适时测试 0.3/3.0 |
| B0 | K1-control | 增加 warmup GRU，但 K=1 |
| B1 | K5-warmup | K=5，其余与 B0 相同 |
| B2 | K10-warmup | 仅在 K5 有效或信息仍可能不足时运行 |

第二阶段：

- 若 variance 有效而 multi-window 无效：优先继续频域/FCD 目标；
- 若 multi-window 有效而 variance 无效：主要瓶颈是条件信息；
- 若两者都有效：建立 `K5 + variance` 组合实验；
- 若两者都无效：考虑未来 dFC 存在不可约不确定性，转向概率预测、多个可能轨迹或动态分布预测，而不是继续增加确定性损失。

## 5. 最终判断原则

本阶段目标不是单纯降低某个复合损失，而是验证模型是否从“平均 FC 结构预测”转向“具有真实动态幅度和时间结构的个体预测”。

一个有价值的改进至少应满足：

1. 平均边重建能力没有明显退化；
2. 动态幅度不再严重坍缩；
3. FCD、长时残差相关或动态状态指标出现一致改善；
4. 改善在共同评价时间段和 non-overlap 时段仍然存在；
5. 初步方向确定后，用多个 seed 验证稳定性。
