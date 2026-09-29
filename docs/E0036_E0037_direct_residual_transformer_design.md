# E0036 / E0037：直接预测 E0032 基线之外的完整残差

状态：**代码和受管配置已实现，待训练**。`sequence` 训练入口现在可以在 Transformer 前向中重放冻结的 E0032 `alpha[t]`，并按长期边 MSE 选模。沿用 `dataset_lr_v1`、`split_lr_v1`、首窗 FC 输入、223 个未来窗口及训练折拟合的 E0032 模板和衰减工件。仅使用验证集选择和比较模型。

## 1. 问题与两个实验

E0034/E0035 的线性低维动态头未检出有意义的动态收益：E0034 长期 MSE 为 `0.185785`，E0032 为 `0.185801`；E0035 为 `0.182920`，略差于 E0033 的 `0.182871`。E0034 的预测动态标准差约为真实动态的 `3.3%`。这不排除从输入到完整残差的非线性映射，但意味着新模型必须直接证明其收益，不能只凭曲线看起来更有波动。

| 编号 | 输入到学习分支 | 研究问题 |
| --- | --- | --- |
| **E0036** | 冻结 E0003 encoder 的 FC1 表示；SC 表示在编码后置零 | FC1 中是否有线性 Ridge/PCA 动态头未提取到的、可由 Transformer 利用的未来残差信息？ |
| **E0037** | 相同 FC1 表示＋被试匹配的 SC-GCN 表示 | 在同一模型和训练规则下，SC 是否提供 FC1 之外的长期误差或真实时间轨迹信息？ |

两项均预测**完整残差**，不显式分离稳定偏移 `m` 和动态项 `d`，也不加载 E0033 的 Ridge 头或 E0034 的 PCA 模式。E0037 的唯一预定输入差别是被试匹配的 SC 编码进入融合层；不让 SC 调整解析基线的 `alpha[t]`。SC 调制 `alpha` 是以后单独研究的问题。

## 2. 实际参与前向计算的公式

令 `x_s` 为被试首窗 FC 的 4005 维 Fisher-z 边，`g_t` 为训练折未来窗口模板，`g_0` 为训练折首窗模板，`alpha_t` 为 E0032 冻结工件中的衰减曲线。对 `t=1,...,223`：

\[
B_{s,t}=g_t+\alpha_t(x_s-g_0),\qquad r_{s,t}=y_{s,t}-B_{s,t}.
\]

每次前向预测都先用输入 `x_s` 和冻结工件**实际计算** `B`。E0003 冻结 encoder 产生 `z^FC_s∈R^{256}`；SC-GCN 产生全局 SC 向量，经线性投影得到 `z^SC_s∈R^{256}`。E0036 把投影后的 `z^SC_s` 置为零；E0037 保留真实被试的 `z^SC_s`。然后：

\[
c_s=\operatorname{LayerNorm}\!\left(W_f[z^FC_s;z^SC_s]+b_f\right),\quad
h_{s,1:223}=\operatorname{Transformer}(q_{1:223}+c_s),
\]

\[
\hat r_{s,t}=W_o h_{s,t}+b_o,\qquad \hat y_{s,t}=B_{s,t}+\hat r_{s,t}.
\]

`q_t` 是可学习未来时间查询；输出头逐窗直接给出 4005 条残差边。`B`、`r`、融合、Transformer 和线性输出头均是本实验真实计算路径，不是概念示意。冻结 E0003 **encoder**，不调用其 FC reconstruction decoder；SC-GCN、融合层、Transformer 和残差边头从头训练。FC1-only 的 SC 消融必须在 SC 投影**之后**进行，以免 GCN bias 和 ROI embedding 从零原始 SC 造出非零条件。两个实验的张量形状、网络设置和初始化 seed 一致。

## 3. 训练协议

| 项目 | 冻结方案 |
| --- | --- |
| 数据 | 与 E0032–E0035 相同的数据版本、被试划分、Fisher-z 4005 边、`warmup_windows=1`，同一 E0003 和 E0032 工件哈希。 |
| 模型 | SC 路径用现有 `hcp_gcn`；FC latent 256、融合后 256、Transformer 4 层/8 头、FFN 1024、dropout 0.1；`concat → Linear(512,256) → LayerNorm`；直接 `Linear(256,4005)` 头。 |
| 目标 | `L=MSE(hat r, y-B)`，对训练批次中全部 223 个未来窗和全部边取均值。由于 `B` 冻结，这在数值上等于 `MSE(hat y,y)`。损失仅这一项，差分、方差、Pearson、对比和额外稳定项权重均为零。 |
| 优化 | AdamW、batch size 4、基础学习率 `3e-4`、weight decay `1e-4`、gradient clip `1.0`、150 epoch、沿用 E0026 的 warm-up cosine 日程；先用 seed 42 做配对探索。两组预算完全一致。 |
| 选模 | 预定以验证集 `t=17:223` 的边 MSE 最小保存 checkpoint；完整 150 epoch 训练，不按验证曲线提前终止。验证集不用于改损失或单独调整两组超参。 |

实现中 `loss_inputs` 在新分解下收到 `hat r` 和 `y-B`，并用验证集**长期边 MSE**选模；旧的 `objective_loss` 覆盖全时段，不作为本实验选模指标。E0032 模板、首窗模板和 `alpha` 从已声明工件加载；`alpha` 哈希在启动与 checkpoint 重建时核对，模型状态保存 `alpha` 数组，训练和推理走同一公式。

## 4. 验证报告与解释规则

主要配对比较：E0036 对 E0032，E0037 对 E0036；另列 E0033 和 E0035 的长期 MSE 供定位。按**被试**计算长期 `t=17:223` 的边 MSE 差及 bootstrap 95% 区间，报告 `0:17`、`17:85`、`85:154`、`154:223` 分段。不能让首窗重叠区的收益掩盖长期结果。

用户关心的相关系数是**真实与预测 FC 信号序列在时间轴上的 Pearson**：每位被试、每条非平坦边，计算 `corr_t(hat y[:,e], y[:,e])`，再汇总边与被试；报告有效边比例。并列报告减去训练折未来模板 `g_t` 后的时间相关，防止共同群体时间趋势冒充个体预测。再报告残差的去时间均值动态相关、相邻窗差分相关、预测/真实动态标准差比。所有时间指标同样先看 `t=17:223`。不要用增加方差当作时间对齐的证据。

E0037 需做冻结模型的 SC 条件诊断：保持 FC1 不变，将 SC 换成另一被试的 SC，并比较输出、长期 MSE 与时间相关；同时检查输出对真正 SC 的依赖是否因被试而异。该诊断只能证明条件敏感性；若主比较有改善，后续再考虑**单独登记**训练时使用置乱 SC 的容量对照，排除多一条可训练分支本身的影响。初轮仍只安排上述两项实验。

判读顺序：

1. E0036 若仅降低 MSE 而时间相关无改善，说明模型可能学到稳定/低频修正，不能宣称提高动态时间预测；可检查 `mean_t(hat r)` 与去均值残差分别贡献多少。
2. E0036 若时间相关提高但 MSE 变差，记录明确权衡，下一轮才考虑损失消融；本轮保持纯 MSE。
3. E0037 只有在同被试配对的长期 MSE 或真实时间相关相对 E0036 改善，且 SC 交换诊断显示匹配 SC 有效，才支持 SC 提供额外信息。只因方差增加或输出对 SC 有反应不算预测改善。
4. 两项若都无法超过 E0032/E0033 的对应指标，优先审视单窗输入的信息上限，考虑多窗 FC 历史或更短预测时距；再决定是否尝试差分/方差损失。

本机保存的 E0034/E0035 结果为验证集探索结果。测试集已有历史探索性接触，不作为本轮选模依据；seed 42 若有明确增益，再以至少 3 个 seed 检查稳定性并报告逐被试配对不确定性。

## 5. 运行与产物

配置为 `configs/experiments/E0036_fc1_direct_residual_transformer_mse_v1.yaml` 和 `configs/experiments/E0037_fc1_sc_direct_residual_transformer_mse_v1.yaml`。在拥有 E0003 checkpoint、E0032 `alpha_fit.npz`、私人数据及 FC 缓存的训练机器上，提交当前代码后运行：

```powershell
scdfc run --experiment configs/experiments/E0036_fc1_direct_residual_transformer_mse_v1.yaml --seed 42 --device cuda
scdfc evaluate-run --run-id <E0036-run-id> --split val --device cuda
scdfc run --experiment configs/experiments/E0037_fc1_sc_direct_residual_transformer_mse_v1.yaml --seed 42 --device cuda
scdfc evaluate-run --run-id <E0037-run-id> --baseline-run-id <E0036-run-id> --split val --device cuda
```

最后一条命令要求同 seed、同数据与模型协议的 E0036 run，并输出被试级配对长期 MSE、逐边时间相关及 SC 交换诊断。两组均可单独评价并输出相对 E0032 的配对长期 MSE。训练前入口会核对 E0003 与 E0032 工件哈希；本机缺少 E0003 checkpoint 和私人训练数据，尚未执行真实训练。
