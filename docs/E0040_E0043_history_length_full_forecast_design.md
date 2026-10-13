# K=1/5/10/20：历史长度与全剩余 FC 序列预测实验设计

日期：2026-10-13。依据当前代码 `f405ca5` 及 E0032–E0039 的验证结果制定。

**状态：代码及冻结配置已实现，E0040–E0043 已登记为 PLANNED，尚未进行真实科研训练。** 已使用合成 FC 数据验证四种 K 的受管理训练、保存重载和配对评价。本文件是协议说明；正式运行应使用下述 `configs/experiments/E0040–E0043` YAML。

实现验证：全项目 `98 passed`，测试使用本机 Python 3.11 / PyTorch 2.4.1 CPU 环境，尚未验证 CUDA 或正式训练环境的 PyTorch 2.6。新增测试覆盖共同截止点、四组相同标签和初始化、无 SC 前向、冻结 encoder、常量时间相关、训练保存重载、K=1 配对比较、Ridge 保存重放、CSV/图汇总，以及旧 RL 缓存不进入新 LR-only 实验。新实验使用独立的 `history_training_stats.npz`，不复用旧模型的共享统计文件。生产模型各 K 的可训练参数均为 1,542,565。

## 1. 研究问题与实验矩阵

问题：在预测起点、未来目标、模型参数量和训练协议相同的情况下，增加连续历史 FC 窗口，是否改善未来 FC 的空间相关与时间相关？改善是否仍存在于不与输入共享 BOLD 样本的预测区间？

第一轮只研究 FC 历史，不输入 SC。沿用 AAL90、`dataset_lr_v1`、`preprocess_lr_v1`、`split_lr_v1` 和同一 E0003 冻结 encoder 工件。根据用户要求，预测**全部剩余未来窗口**。四组统一在扫描第 20 窗结束观测，预测第 21–224 窗，因此 **H=204**。

此前 H=40 是用于优先检查短期可预测性的建议，不是模型限制。本版将全剩余序列设为主任务；短、中、远期分析均从同一条完整预测中截取，不另训练 H=40 模型。

| 编号 | 历史 K | 实际覆盖 TR | 实际覆盖秒数 | 未来 H | 对照 |
| --- | ---: | ---: | ---: | ---: | --- |
| E0040 | 1 | 83 | 59.76 | 204 | 本轮参数匹配参照 |
| E0041 | 5 | 103 | 74.16 | 204 | E0040 |
| E0042 | 10 | 128 | 92.16 | 204 | E0040，并列 E0041 |
| E0043 | 20 | 178 | 128.16 | 204 | E0040，并列 E0041/E0042 |

覆盖时长为 `(83 + (K-1)*5)*0.72` 秒。窗口高度重叠，不把 K 个窗口视为 K 次独立观测。H=204 表示末个目标窗相对最后输入窗平移 1020 TR，即 734.4 秒；不是 204 个独立的约 60 秒片段。

四组新实验之间只改变 K。它们与 E0036–E0039 的输出分解、预测区间和模型不同，旧实验只提供背景，不能用旧指标与新指标直接排名。

## 2. 输入和标签：固定共同预测起点

设某被试的 FC 缓存为 `X[0:224]`，每窗为 4005 维 Fisher-z 边向量。令 `a` 是最后一个观测窗的零基索引：

```text
history = X[a-K+1 : a+1]       # [K, 4005]
last_fc = X[a]                # [4005]
target  = X[a+1 : ]            # [204, 4005] when a=19
```

四组必须使用同一 `(subject, run, a)` 样本表。不能分别从扫描首窗输入 K 个窗口后立即开始预测，因为这会使四组预测不同的真实时间段。

例如 `a=19`：

| K | 输入零基索引 | 标签零基索引 |
| ---: | --- | --- |
| 1 | 19 | 20–223 |
| 5 | 15–19 | 20–223 |
| 10 | 10–19 | 20–223 |
| 20 | 0–19 | 20–223 |

**预先固定的共同起点：**每个完整 run 只使用 `a=19`，不依据表现选择起点。本轮不生成五个滚动起点片段，也不将短期训练样本数沿用到全序列方案。

- 先按既有被试划分确定 train/val，再在每名被试内部生成片段。
- 在每名被试有一个合格 LR run 时，738 名训练被试形成 738 个完整预测样本，158 名验证被试形成 158 个完整预测样本。正式启动时核对实际合格 run，不用这个算式替代数据审计。
- 每个 epoch 遍历相同片段表，训练只打乱片段顺序；验证固定顺序。
- 所有 K 在每个片段中都有相同的 `last_fc` 和真实未来。
- 评价先对每个 run 计算，再对同一被试的 run 平均，最后对被试平均。本轮标准数据每位被试只有一个 LR run。
- 学习数据统计、PCA 或模板仅使用训练被试；评价不使用真实未来进行输入标准化或校正。

这是基于既有离线预处理缓存的预测研究。若后续声称在线前瞻预测，还须单独审计滤波、去趋势等预处理是否使用了预测起点后的原始 BOLD。

### 与“从扫描起点输入前 K 窗”的区别

现有代码默认输入 `X[:K]`，预测 `X[K:]`，对应 H 分别为 223/219/214/204。这同样能输出全部剩余序列，但四组结束观测的时间不同，最后一窗也不同；收益同时包含历史长度、当前状态的新旧和预测时距差异。

因此，本轮用上表的共同截止时间作为主协议。若后续要求研究“从扫描开始，需要观测多久”，可另立前缀输入协议，保持 H=224-K，并在共同绝对目标 `X[20:]` 上对齐报告；不能把它当作纯历史长度消融，也不能与本轮同编号混用。

## 3. 共享模型：逐窗 FC encoder → 历史 GRU → 未来变化量

全部 K 使用完全相同的模块和参数形状。K=1 也经过历史 GRU，不为它设置另一个简化模型。

```text
[B,K,4005] Fisher-z 历史
    ↓ 同一个冻结 E0003 encoder 逐窗编码
[B,K,256]
    ↓ 单层、单向历史 GRU，hidden=256，初始状态为零
[B,256] 最后 hidden state c
    ↓ 与 204 个可学习的相对未来时距 query 逐一相加
[B,204,256]
    ↓ 所有未来窗共享的小型非线性预测头
[B,204,4005] 相对最后一窗的变化量
    ↓ 加回 last_fc[:,None,:]
[B,204,4005] 最终 FC 边预测
```

对历史逐窗编码得到 `z_k=f_E0003(X_k)`，按时间顺序得到 `c=GRU(z_1:K).hidden_last`。给每个未来时距 `h=1,...,204` 一个 256 维 query `q_h`：

\[
u_h=\operatorname{LayerNorm}(c+q_h),
\quad v_h=\operatorname{Dropout}_{0.1}\left(\operatorname{GELU}(W_1u_h+b_1)\right),
\]

\[
\widehat{\Delta}_h=W_2v_h+b_2,
\quad \widehat X_{a+h}=X_a+\widehat{\Delta}_h.
\]

`W1:256→256`，`W2:256→4005`。预测头含非线性，使历史条件可以在不同 query 下产生不同的变化模式；不使用仅有线性变换的 `W(c+q_h)` 作为最终方案，因为那会将历史条件的贡献限制为跨未来窗相同的加性项。

变化量均相对于 **同一个最后观测窗 X_a**，不是逐步差分，不累计输出，不将预测或真实未来再次输入网络。204 窗一次前向并行输出，不使用 teacher forcing。

- E0003 encoder 冻结且保持 eval；不加载其 reconstruction decoder 参与输出。
- 训练历史 GRU、query 和共享预测头。没有 SC 分支、SC bias 或被试 ID embedding。
- query 仅表示相对预测时距，不提供未来 FC、测试信息或可学习被试身份。
- 第一轮不微调 FC encoder、不加 cross-attention、不改窗长、不加入相关/差分/方差损失。
- 初始化同 seed 下四组可训练参数完全相同；模型初始化与数据打乱使用分别记录的随机源，避免输入长度改变随机数消费顺序导致参数不匹配。
- 预测在 Fisher-z 边空间计算；恢复 FC 矩阵时再 tanh、对称填充和设置单位对角。相关矩阵合法性另行报告。

## 4. 训练和选模协议

真实监督变化量为 `Delta_h = target_h - last_fc`。训练目标：

\[
L=\frac{1}{B\cdot204\cdot4005}\sum_{b,h,e}
(\widehat\Delta_{b,h,e}-\Delta_{b,h,e})^2.
\]

这与最终 FC 预测的全 204 窗边 MSE 数值等价。损失只有这一项。

| 参数 | 四组统一设置 |
| --- | --- |
| 训练等级 | Level 1 方向筛选 |
| seed | 配置预先声明 `[42,43,44]`；首轮只运行 42，进入复验后运行 43/44 |
| batch size | 16 个片段；若统一设备的 Level 0 检查无法承受，正式冻结前统一改为 8，不能只改 K=20 |
| optimizer | AdamW，lr=3e-4，weight decay=1e-4 |
| 日程 | 150 epoch，前 5 epoch 线性 warm-up，再 cosine 至 lr=0 |
| 梯度裁剪 | 全局范数 1.0 |
| early stopping | 关闭；所有组完整 150 epoch |
| 主选模指标 | 验证集保守无重叠区间 `[17:204]` 的边 MSE，越低越好 |
| 选模汇总 | 先每片段 MSE，再同被试平均，再跨被试平均 |
| 并列最优 | 保留最早 epoch；每次改善保存 best checkpoint，另保存最后一轮 |

训练覆盖全部 204 窗，选模只使用预定的后 187 窗，不因哪一组时间 r 较高而重新选 checkpoint。四组训练样本数量、更新步数、优化器与参数量相同；历史长度不同会改变运算量，不宣称 GPU 时间相等。

## 5. H=204 的区间口径与评价

future 采用零基数组 `target[0:204]`，对应一基未来步 `h=1:204`。

- **全预测段：**`[0:204]`，即 h=1–204，用户要求的全时段空间 r 和时间 r 都要报告。
- **近端：**`[0:17]`，即 h=1–17，含与最后输入窗共享 BOLD 样本的窗口。
- **保守无重叠段：**`[17:204]`，即 h=18–204，为主选模和主要动态增益判断区间，共 187 窗。
- **附加分段：**`[0:40]` 为短期诊断；`[17:68]`、`[68:136]`、`[136:204]` 为早/中/晚无重叠区间。分段窗口数不同，各自计算指标，不先将三个均值无权平均冒充完整长期结果。

理论上 h=17 的窗口起点已经平移 85 TR，超过 83 TR 窗长，因此 h=17 已无直接 BOLD 样本重叠。这里刻意沿用现有项目的保守数组切片起点 17，额外跳过这一窗。无样本重叠不代表时间独立。

### 必报指标

| 指标 | 计算方式与作用 |
| --- | --- |
| `forecast_mse` / `forecast_nonoverlap_mse` | 全 204 窗/后 187 窗的 Fisher-z 边 MSE |
| `forecast_space_r` / `forecast_nonoverlap_space_r` | 每个窗口沿 4005 条边计算 Pearson，片段内对窗平均 |
| `forecast_time_r` / `forecast_nonoverlap_time_r` | 每条边沿所声明的未来时间段计算 Pearson，片段内对有效边平均 |
| `forecast_adjusted_time_r` / `forecast_nonoverlap_adjusted_time_r` | 预测与真实分别减去仅训练集拟合的同起点未来群体模板后，沿时间计算 Pearson |
| `forecast_nonoverlap_residual_space_r` | 减去同起点训练群体模板后，逐窗空间 Pearson，诊断个体残差信息 |
| `forecast_nonoverlap_difference_time_r` | 后 187 窗中相邻预测/真实 FC 差分的逐边时间 Pearson；不把保守区间之前的差分混入 |
| `forecast_nonoverlap_std_ratio` | 后 187 窗逐边时间标准差的预测/真实比，报告被试汇总及分布 |
| 未来均值与去均值动态检索 | 对同一 origin 的 158 名验证被试做本人/他人匹配，分别检验静态身份和动态本人匹配 |
| 矩阵质量 | 负特征值比例、最小特征值及投影改变量，报告原始预测指标 |

模板 `g[a,h]` 由训练被试在该 origin 的对应真实未来窗平均得到，只用于解析基线和诊断，不从验证集估计，不把它加入本轮主模型输出。

空间 r 与时间 r 都使用 Fisher-z 边值。逐窗空间 r、逐边时间 r 的计算维度明确记录，不能将全序列展平后的相关或去均值后重算的指标替代它们。报告采用直接算术平均，同时保存逐被试分数和有效比例。

时间 Pearson 用 float64 做中心化与范数计算。预测或真实序列时间标准差低于固定 `1e-6` Fisher-z 的边视为常量，r 为不可定义；报告真实有效边比例、预测常量比例和共同有效边比例。不能靠分母加 epsilon 把常量输出伪装为有效动态相关，也不能仅报告剔除大量预测常量后的高 r。不同方法应另报共同有效边集合的配对结果。

187 个高度自相关的点不足以被当成 187 次独立观测。统计重采样单位为被试；不将窗口、边或同被试的 run 当作独立样本扩大置信度。

### 预测时距曲线

每个未来步 h=1–204 报告 MSE、空间 r，以及预测/真实 FC 差分的幅度。单个 h 无法计算沿时间 Pearson，因此时间 r 在整个 `[0:204]`、近端 `[0:17]`、保守段 `[17:204]` 及上面的预定短期/早中晚区间内计算；不能画“单窗时间 r”曲线。

## 6. 同目标的基线与机制对照

### 首轮基线

1. **Last-FC persistence：**每个未来窗复制 `X_a`。所有 K 的结果相同，是共同的当前状态基线。
2. **History-mean persistence：**复制当前 K 个输入窗的平均值。检验多历史收益是否只来自估计稳定 FC 更准。
3. **训练群体模板：**预测 `g[a,h]`。四组共享同一模板，不能复用不同区间的旧 E0001 汇总分数。
4. **低维 Ridge：**输入同一 K 个冻结 E0003 latent 的按时序拼接；预测未来变化量的低维表示，再还原为 4005 边并加回 X_a。目标 PCA 固定 128 维，只用训练被试的变化量拟合；每名训练被试等间距取 16 个未来窗用于拟合 PCA，204 窗都用于 Ridge 标签和评价。Ridge alpha 网格 `[1,10,100,1000,10000,100000]`，在训练被试内部五折 Group CV，按保守段 MSE 选择。每折独立拟合目标 PCA、特征标准化与 Ridge，片段按被试成组。E0003 表示工件始终固定。此参考算法及超参数已声明在四组冻结配置的 `baselines.history_ridge` 中，通过 `history-ridge` 为完成的 run 生成派生对照工件；不增加主模型的 completed_runs，不改动主模型选模记录。其算法标识为 `encoded_history_pca_ridge/v1`，同时记录父 checkpoint 与 FC encoder 哈希。

两种 persistence 预测在时间上为常量，原始时间 r 不可定义；展示为 `NA` 并报告有效比例 0，而不是填成实测 r=0。它们仍可比较 MSE、空间相关和静态检索。时间相关重点与有变化的群体模板、Ridge 及参数匹配的 K=1 模型比较。

### 筛选后的机制诊断

对 K>1 的冻结候选模型，保持最后一窗 X_a 不变，分别置乱或替换前 K-1 窗；标签和最后一窗仍然不变。它可以检查模型是否依赖新增历史，但置乱输入可能偏离训练分布，因此仅作为敏感性诊断。

若要证明时间顺序本身有价值，后续另登记一个训练时就置乱前 K-1 窗的容量匹配模型；不能仅凭推理置乱后的下降宣称顺序具有因果作用。

## 7. 对比、统计与进入下一阶段的规则

### 首轮 seed 42

- 预定主要比较：K=5、10、20 分别对 K=1；额外并列相邻 K 的差异。
- 每项差异先在同一 `(subject,run,origin)` 上配对，再对同被试片段平均。
- 对 158 名验证被试做 10,000 次被试级 bootstrap，固定重采样 seed=20261013。
- 三个主要 K 对 K=1 的比较并列报告 95% 区间；若做显著性声明，另报 Bonferroni 调整后的 98.333% 区间。seed 42 的方向筛选不写成正式确认结果。
- 全 204 窗空间 r、时间 r 与后 187 窗对应指标全部报告，包括无效或不利结果；附加短期诊断不替代主结果。

### 候选去留

以下为本轮预先提出的内部筛选标准，不是领域通用的“可发表 r 门槛”：

| 观察 | 解释与下一步 |
| --- | --- |
| 空间 r/MSE 改善，保守段时间 r 未改善 | 多历史可能只改善稳定 FC，不能宣称动态预测成功；可进入稳定个体信息研究 |
| 全段时间 r 改善，但保守段无增益 | 暂限于近端预测；检查共享 BOLD、持续性和线性基线 |
| 保守段时间 r 与去模板时间 r 均有配对改善，MSE 增幅不超过 2%，预测非平坦有效比例不低于 99% | 值得进入多 seed 验证，并同时检查差分相关、本人匹配与幅度 |
| 时间 r 改善但 MSE 增幅超过 2% | 记录相关/精度权衡；另行预登记研究，不按本轮规则直接 KEEP |
| 四组都不能超过合理基线 | 优先测试 FC 表示或 BOLD 历史、缩短目标；暂缓新增 SC 与复杂损失 |

“改善”必须同时给出绝对值与区间，不把从近零变成略高于零视为已经足以支撑动态应用。幅度和差分指标不能严重塌缩；区间跨零或方法有效边覆盖不足时归为 INCONCLUSIVE。

### 多 seed 与后续任务

筛选有候选后，补齐四组 seed 43/44，以相同协议重新运行。报告各 seed 及均值/标准差；被试级不确定性与 seed 稳定性分别展示，不把 `158×3` 当作独立被试。对每位被试可先平均三个 seed 的分数，再做被试 bootstrap，但不得先平均预测轨迹后替代各模型的动态评价。

随后才单独研究 SC 增量、encoder 微调、历史 cross-attention 或相关损失。如果完整序列只有短期有收益，先报告时距边界，再另登记只训练短期目标的实验；不能在同一 ID 下改标签长度重解释旧 run。既有测试集存在探索性接触，本轮不评价 test；正式确认需要如实记录历史接触并另定独立确认方案。

## 8. 固定的结果表和交付物

主表字段建议如下；未运行时留空，不填预期值：

| ID | K | H | seed | 全段空间 r | 全段时间 r | 无重叠空间 r | 无重叠时间 r | 无重叠去模板时间 r | 无重叠 MSE | 有效比例 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0040 | 1 | 204 | 42 | | | | | | | | 待训练 |
| E0041 | 5 | 204 | 42 | | | | | | | | 待训练 |
| E0042 | 10 | 204 | 42 | | | | | | | | 待训练 |
| E0043 | 20 | 204 | 42 | | | | | | | | 待训练 |

每个 run 保存冻结配置、代码版本、工件哈希、同一片段清单哈希、参数量、训练/选模日志和 per-subject/per-origin 指标。可视化包括：K 对两种 r 的曲线、MSE/空间 r 随 h 的曲线、固定被试与边的未来轨迹、静态/动态本人匹配及幅度诊断。验证预测如保存，应留在受限的本地输出目录；公开表不含被试标识或逐边真实 FC。

## 9. 实现边界与实施顺序

实现以独立模型和新评价模块接入既有受管理入口，保留旧配置未声明 forecast origin 时的行为：

| 位置 | 历史行为 | 已实现的扩展 |
| --- | --- | --- |
| `src/scdfc/data.py` | 从第 0 窗取 K 个 warm-up，剩余全部为标签 | 共同 origin=19、尾部 K 窗切片、全部剩余 H=204、last_fc |
| `src/scdfc/models/history.py` | 旧模型位于 sequence.py；旧残差路径限定 K=1 | 新 FC-history 模型与共享 query 预测头，输出 last-FC residual；不复用 E0032 alpha |
| `src/scdfc/training.py` | 输出长度随完整剩余模板变化 | 明确 H，片段/被试汇总选模，保存新输出分解和初始化信息 |
| `src/scdfc/history_forecast.py` | 旧评价位于 evaluation.py | 全段/保守段时间 r、常量边有效比例、同 origin 基线、矩阵诊断、检索与配对统计 |
| `src/scdfc/management.py` | 尚无本轮模型与主指标 | 新任务/模型及 `forecast_nonoverlap_mse` 验证，规范注册与汇总 |
| `src/scdfc/history_ridge.py` | 无共同 origin 的 Ridge 对照 | 训练集分组 CV、每折独立 PCA/标准化、对照保存与重放 |
| `src/scdfc/history_summary.py` | 无本轮 K 的专用主表 | CSV 双相关汇总和 K 对比图，跨 seed 标准差与被试 bootstrap 分开 |

已支持的语义参数（完整冻结设置以四份实验配置为准）：

```yaml
data:
  warmup_windows: 1  # 四组分别为 1/5/10/20
  forecast_windows: 204  # 同时检查等于 n_dfc_windows - forecast_origin - 1
  forecast_origins: [19]
model:
  name: fc_history_gru_query_residual
  output_decomposition: last_observed_fc_residual
  warmup_encoder: gru
  warmup_gru_layers: 1
  hidden_dim: 256
  dropout: 0.1
evaluation:
  primary_metric: forecast_nonoverlap_mse
  nonoverlap_start: 17
training:
  seeds: [42, 43, 44]
  epochs: 150
  batch_size: 16
  loss_type: mse
  loss_weights: {edge: 1.0}
```

数据、模型、指标与 Level 0 检查已完成，四组已登记。下一步是在训练机器上核对真实工件/缓存，提交代码及配置使 Level 1 工作区干净，再依次运行四个 seed 42，完成基线和配对报告，按规则决定是否补 seed 43/44。

正式配置：

- [E0040：K=1](../configs/experiments/E0040_history_k1_full_forecast_v1.yaml)
- [E0041：K=5](../configs/experiments/E0041_history_k5_full_forecast_v1.yaml)
- [E0042：K=10](../configs/experiments/E0042_history_k10_full_forecast_v1.yaml)
- [E0043：K=20](../configs/experiments/E0043_history_k20_full_forecast_v1.yaml)

每组通过同一入口训练，下面示范 E0040；其余三组替换配置路径：

```powershell
python -m scdfc.cli run --experiment configs/experiments/E0040_history_k1_full_forecast_v1.yaml --seed 42 --device cuda
```

记录各组实际返回的 run ID，然后执行验证评价。E0040 不传 baseline-run-id；其他组传同 seed 的 E0040：

```powershell
python -m scdfc.cli evaluate-run --run-id <E0040-run-id> --split val --device cuda
python -m scdfc.cli evaluate-run --run-id <E0041-run-id> --baseline-run-id <E0040-run-id> --split val --device cuda
python -m scdfc.cli evaluate-run --run-id <E0042-run-id> --baseline-run-id <E0040-run-id> --split val --device cuda
python -m scdfc.cli evaluate-run --run-id <E0043-run-id> --baseline-run-id <E0040-run-id> --split val --device cuda
```

`--save-predictions` 可显式导出验证预测到该 run 的本地目录；含受限数据，保持私有。常规评价不保存完整逐时预测。

对每组完成的 run 执行低维参考算法：

```powershell
python -m scdfc.cli history-ridge --run-id <run-id> --device cpu
```

结果写入父 run 的 `baselines/history_ridge_v1/`：`ridge.pt` 保存可重放的 scaler、PCA、Ridge，`evaluation_val.json` 包含 train-CV 选 alpha、验证指标与神经模型减 Ridge 的配对区间。重复执行会拒绝覆盖已有对照。对照生成后再次 `evaluate-run`，主报告会加入已验证的 Ridge 参考摘要。

完成四组评价后生成双相关主表和图：

```powershell
python -m scdfc.cli history-summary --run-ids <E0040-run-id> <E0041-run-id> <E0042-run-id> <E0043-run-id>
```

默认输出 `reports/history_full_forecast/history_comparison.csv`、`summary.json`、`history_comparison.png`。可一起传入其他 seed 的 run ID；每个实验与 seed 只允许一个 run，误差线是 seed 间标准差，不是被试 bootstrap 区间。该汇总不改动选模记录或历史指标。

## 10. 方法来源与限定

- [Cho 等，2014：RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078)：支持“历史序列编码为条件”的方法来源，原论文任务是机器翻译，不是本项目 dFC 的有效性证据。
- [Hindriks 等，2016：滑窗相关与 dFC 统计检验](https://doi.org/10.1016/j.neuroimage.2015.11.055)：提示滑窗估计的采样波动与统计检验问题，不能据此直接断言本项目未来动态不可预测。
- 项目既有 [E0014/E0015 配置](../configs/experiments/E0014_gcn_gru_warmup_k1_control_v1.yaml)、[残差预测计划](residual_prediction_experiment_plan.md) 和 [实验管理操作手册](experiment_management.md) 提供实现与管理背景；新计划以本文件的相同 origin、独立 H 和双相关评价为准。
