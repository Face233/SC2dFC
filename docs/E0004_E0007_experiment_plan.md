# E0004–E0007 实验计划

更新日期：2026-08-12

## 1. 当前目标

第一阶段只回答两个架构问题：

1. SC encoder：`HCP GCN` 与 `Hybrid (Graph Attention + edge MLP)` 哪个更合适；
2. 时序模型：`GRU` 与 `Transformer` 哪个更合适。

四个主实验构成严格的 2 × 2 比较：

| 实验 | SC encoder | 时序模型 | 输入 | 输出 decoder |
|---|---|---|---|---|
| E0004 | HCP GCN | GRU | SC + FC1 | 冻结 E0003 reconstruction decoder |
| E0005 | Hybrid | GRU | SC + FC1 | 冻结 E0003 reconstruction decoder |
| E0006 | HCP GCN | Transformer | SC + FC1 | 冻结 E0003 reconstruction decoder |
| E0007 | Hybrid | Transformer | SC + FC1 | 冻结 E0003 reconstruction decoder |

所有实验首先只运行 `seed=42`。四个模型都完成后，仅根据验证集选择候选模型；测试集继续隔离。确认候选架构后，再决定是否补多 seed 正式实验。

## 2. 统一的数据流

### 2.1 FC1 分支

每个被试/静息态 run 的第一张 FC 窗口输入冻结的 E0003 FC encoder：

`FC1 (4005 edges) -> frozen E0003 encoder -> 256-d FC embedding`

E0003 encoder 的权重与 Dropout 状态均冻结。

### 2.2 SC 分支

HCP GCN：

`90 × 90 SC -> normalized 2-layer GCN (128 -> 64) -> max pool -> Linear(64, 256)`

Hybrid：

`90 × 90 SC -> 3-layer graph attention -> mean pool -> 128-d`

`4005 SC edges -> MLP(4005, 512, 128) -> 128-d`

两个 128 维表示拼接为 256 维 SC embedding。因此两种 SC encoder 向后续模块提供完全相同的 256 维接口。

### 2.3 条件融合与时序建模

`SC embedding (256) + FC1 embedding (256) -> gated fusion -> global condition (256)`

- GRU：2 层、hidden size 256；全局条件同时用于时间 query 和初始 hidden state。
- Transformer：4 层、8 头、FFN 1024；对未来时间窗口做 self-attention，全局条件加到每个时间 query。

两类时序模型均输出 `T_future × 256` 的潜轨迹。`hidden_dim` 必须等于 E0003 的 `fc_latent_dim=256`，配置错误时模型会立即报错。

### 2.4 当前输出 decoder

E0004–E0007 使用：

`future latent (256) -> frozen E0003 reconstruction decoder -> predicted FC (4005 Fisher-z edges)`

E0003 reconstruction decoder 在整个训练过程中保持参数冻结和 eval 状态，不启用 Dropout。模型不再叠加群体模板或额外的 4005 维 static head，因此最终 FC 确实由 E0003 decoder 产生。

## 3. 损失函数与模型选择

令预测与真实 Fisher-z FC 分别为 `ŷ[t,e]` 和 `y[t,e]`，Huber 阈值为 `β=1`：

```text
             0.5 r² / β,       |r| < β
Huberβ(r) =
             |r| - 0.5 β,      |r| >= β
```

边重建项：

```text
L_edge = mean[t,e] Huberβ(ŷ[t,e] - y[t,e])
```

一阶差分项：

```text
Δŷ[t,e] = ŷ[t,e] - ŷ[t-1,e]
Δy[t,e] = y[t,e] - y[t-1,e]

L_diff = mean[t,e] Huberβ(Δŷ[t,e] - Δy[t,e])
```

总损失：

```text
L = L_edge + λ_diff L_diff
λ_diff = 0.25
```

Huber 不是套在 MSE 外面，而是替代纯平方误差：小误差区间保持二次惩罚，大误差区间改为线性惩罚，减少极端边误差对梯度的支配。

最佳 checkpoint 按验证集 `objective_loss = L_edge + 0.25 L_diff` 最小选择。`MSE`、边 Pearson、node-strength、FCD 和动态幅度等只作为解释性指标报告，不参与反向传播和 checkpoint 选择。

## 4. 对照实验的后续安排

不要在 E0004–E0007 之前一次性铺开全部对照。先完成四个主模型，再按验证结果建立：

1. FC1-only：GRU 与 Transformer 各一个，共 2 个；SC embedding 在编码后置零。
2. SC-only：四种 SC encoder × 时序模型组合各一个，共 4 个；FC1 embedding 在编码后置零。

消融必须发生在 encoder 输出之后，而不是把原始矩阵置零。这样可以避免 bias、LayerNorm 或 ROI embedding 重新产生伪信息。

暂不安排 shuffled-SC。只有完整模型明显优于 FC1-only、但需要进一步证明 SC–被试对应关系时再加入。

## 5. 后续 direct-output decoder 计划

四个主模型和必要的信息消融完成后，再进行 decoder 对照。第一步只对验证集胜出的架构增加一个实验：

`future hidden (256) -> trainable Linear(256, 4005) -> predicted FC edges`

该实验使用 `output_head=direct_edge_linear`，替换 E0003 reconstruction decoder；其余输入、SC encoder、时序模型、split、seed、训练损失和早停规则全部保持不变。E0003 encoder 仍可用于编码 FC1，变化仅限于输出 decoder。

如果 direct head 在验证集带来稳定改善，再把它扩展到其余架构，形成完整的 decoder 因子比较。当前版本只记录该计划，尚未实现或创建 direct-head 实验配置，避免未经验证就扩大实验数量。

## 6. 推荐执行顺序

```powershell
scdfc run --experiment configs/experiments/E0004_gcn_gru_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0005_hybrid_gru_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0006_gcn_transformer_full_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0007_hybrid_transformer_full_v1.yaml --seed 42 --device cuda
```

每次运行前应保持 Git 工作区干净并推送对应 commit。每个实验完成后先执行 `scdfc summarize --experiment E####`，四个实验全部完成后再统一比较验证集，不提前打开测试集。
