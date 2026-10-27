# E0050/E0051：在 E0036/E0037 中替换冻结 FC 输入编码组件

执行更新（2026-10-27）：E0050、E0051 seed42 均已完成 150 轮，完成对
E0036/E0037 的配对验证及 E0051−E0050 次要比较，登记为 AWAITING_CONCLUSION。
详见[四模型可视化与结果](../reports/experiment_notes/E0050_E0051_comparison_20261027.md)
和[E0050 运行记录](../reports/experiment_notes/E0050_seed42_validation_20261027.md)。
下文“没有启动正式训练”等描述对应 2026-10-26 的设计与适配阶段。

日期：2026-10-26。状态：**运行适配已完成，登记仍为 PLANNED；没有启动正式训练或评价。**
用户随后授权进行代码适配。两份配置现为 `experiment.execution_ready: true`，
可在后续授权训练时使用；Level 1 仍要求提交代码/配置并使用干净工作区。
后端验证只使用 pytest 临时目录中的小型合成数据和临时运行，不产生正式
E0050/E0051 结果。通用 `execution_ready: false` 防误执行锁仍然有效。

## 1. 研究问题与四个比较条件

冻结 PCA 在同期 FC 重建上的优势能否转化为未来预测优势？
本轮保留 E0036/E0037 的单窗条件、解析基线和直接残差 Transformer。

| 条件 | FC1 输入编码 | SC 条件 | 作用 |
| --- | --- | --- | --- |
| 既有 E0036 | 冻结 A0003 AE，256 维 | 投影后置零 | FC1-only AE 参照 |
| 既有 E0037 | 同一 A0003 AE | 匹配被试的 HCP-GCN 表示 | FC1＋SC AE 参照 |
| **E0050** | 冻结 A0044 PCA，256 维 | 投影后置零 | 对 E0036 的组件替换 |
| **E0051** | 同一 A0044 PCA | 匹配被试的 HCP-GCN 表示 | 对 E0037 的组件替换 |

主要比较为 E0050−E0036、E0051−E0037。次要比较 E0051−E0050，
检验 PCA 条件下的 SC 增量；E0037−E0036 可作为历史定位。
登记表当前两项 AE seed42 的选模长期 MSE 分别为 0.1842942062、0.1836168174，
这些是既有验证记录，不是新 PCA 实验结果或验收阈值。

## 2. 固定数据、工件与实际前向路径

沿用 dataset_lr_v1 / preprocess_lr_v1 / split_lr_v1：738/158/159 名
train/val/test，被试隔离、LR-only、AAL90、1200 TR、窗长 83 TR、stride 5、
每人 224 个窗口、每窗 Fisher-z 上三角 4005 边。训练只访问 train，
验证只选模和比较，不读取 test。

PCA 绑定 `configs/artifacts/A0044.yaml` 与
`outputs/components/A0044/pca.npz`，分别校验登记文件和参数文件 SHA256。
原 PCA 在 738 名训练被试的全部 165,312 窗上拟合；本轮不重新拟合，
不白化、不增加 latent 标准化或输入 LayerNorm、不改 FC 缓存。

首窗记为 `x=FC[0]`，PCA 表示为：

```text
z_FC = (x - PCA.mean) @ PCA.components.T               # 4005 -> 256
z_SC = SC_norm(HCP-GCN(SC) 经现有线性投影)              # 256
E0050: z_SC 在投影和归一化之后置零；E0051: 保留 z_SC
c = LayerNorm(Linear(concat[z_SC, z_FC]))              # 512 -> 256
h = 原 Transformer(可学习的未来查询 + c)
r_hat = 原 Linear(h)                                 # 256 -> 4005
y_hat = 原 E0032 解析基线 B(x) + r_hat
```

解析基线仍使用**真实首窗 x**和原训练折模板、alpha；不把 PCA 重建的首窗
代入 B。alpha 工件继承 E0036 的原路径与 SHA256
`dcb8a74c2a06d753791a7b2dbe1be582c57c3e6a77e7562a957b575359689efa`。
输出仍由可训练的 4005 边线性头产生；不调用 PCA 逆投影或 AE decoder。
E0051 相对 E0050 只改变 SC 消融开关。

输入/输出索引均为零基、左闭右开：输入 `FC[0]`，预测 `FC[1:224]`
共 223 窗。沿用现有保守长期区间 `target[17:223]`，对应原始
`FC[18:224]` 共 206 窗；不改成 E0045–E0047 的第 10 窗共同截止点。

## 3. 训练、选模和初始化

| 项目 | 固定方案 |
| --- | --- |
| 输入 | FC1；E0051 额外使用原 SC 条件路径 |
| 网络 | latent/hidden 256；Transformer 4 层、8 头、FFN 1024、dropout 0.1；concat 融合和直接残差头均继承 E0036 |
| 损失 | 全部 223 个未来窗口的纯边 MSE；差分、方差、static、contrastive 权重为 0 |
| 优化 | AdamW，batch 4，lr 3e-4，weight decay 1e-4，clip 1 |
| 日程 | 150 轮、patience 151；warmup cosine，warmup 5 轮、eta_min 1e-5 |
| 选模 | 仅按验证 `target[17:223]` 长期边 MSE 选 best；不按相关性事后挑 checkpoint |
| 首轮 seed | 42，与历史 E0036/E0037 对应；本轮不增设其他 seed |

所有可训练预测层从头训练。PCA 的均值和主成分是冻结 buffers。
原 AE 构造函数本身会消耗随机数，PCA 加载则不会；仅声明同一 seed 不保证
共有预测层初始化相同。适配现采用 `historical_ae_constructor_rng/v1`：在创建
共有预测层之前，按原加载器的 AE 结构临时执行一次 CPU 初始化以匹配 RNG
消耗，然后释放临时对象。此过程不加载 A0003 权重，不参与前向或优化。
测试核对同环境下 AE/PCA 两条构造路径的所有共有可训练参数和后续 CPU RNG
状态完全相同。训练记录初始参数/CPU与设备 RNG 摘要、样本清单摘要和独立
DataLoader seed。E0050/E0051 使用同一初始化协议。
历史运行未保存初始状态摘要，跨 PyTorch 版本的历史实际初始化不能仅凭
相同 seed 认证；与 E0036/E0037 的既有结果仍按探索性历史参照解释。

PCA 输出的尺度与 A0003 最后一个 LayerNorm 的输出不同；本轮不额外消除
该差别。因此检验的是“整套固定输入组件的替换”效果。
历史 A0003 的 SmoothL1/抽样预训练与 PCA 的全窗平方目标也不同，
不能把结果直接归因为线性优于非线性，或单独归因于训练覆盖率。
如后续需要定位这些因素，A0049 全窗 MSE AE 和 latent 归一化应分别另立实验。

## 4. 评价与预定判读

主指标保留 `long_edge_mse`。新旧模型必须在同一批 158 名验证被试的
同一真实目标上比较，先得到每位被试的差，再做被试 bootstrap：10,000 次，
seed 20261026。报告均值、绝对差与相对降幅。
已有通用单次评价的 bootstrap 参数保持继承值；此处 10,000 次用于新旧运行
的独立配对比较报告，不能改变预测、选模和训练协议。

两个主要 MSE 对比作为一个比较族：并列报告 95% 边际区间和各自 97.5%
Bonferroni 区间（两项合计至少 95% 族覆盖）。SC 对比、分段和相关性为次要
探索性结果，完整报告，不能按表现替换主要对比。

除完整 223 窗外，保留 `0:17`、`17:85`、`85:154`、`154:223` 分段；
相关性结论优先看无重叠长期区间。需并列报告：

- 每窗跨边空间 Pearson。
- 每条边跨时间的原始 FC Pearson；先按边、再按被试汇总，报告有效边比例。
- 减去训练折群体时间模板后的逐边时间 Pearson。
- 去时间均值动态相关、相邻窗差分相关、预测/真实时间标准差比。
- 完整残差 `r_hat` 的稳定偏移与去均值动态部分误差，帮助定位 MSE 改善来源。

E0051 还需冻结模型的 SC 交换诊断：保持首窗 FC 不变，以固定 seed20261026
生成没有自配对的验证被试 SC 置换；分别报告匹配与交换后的长期 MSE、
时间相关及输出变化。条件敏感性本身不能证明 SC 提高预测。

判读规则：主要 MSE 差的校正区间上界低于 0，支持该 seed 下数值误差改善；
若只改善 MSE，优先解释为稳定/低频修正。只有长期原始与去模板时间相关、
差分相关共同支持，才讨论动态时间预测增益。PCA 同期重建更好不保证预测更好。
首轮仅为 seed42 验证探索；多 seed 复验需另登记，并保留验证集已被多次使用的限制。

## 5. 已完成的运行适配与后续执行

模型工厂现按 `model.fc_representation` 加载 A0044，核验参数和登记文件双哈希、
数据绑定；条件模型使用组件的 `latent_dim`。PCA 输入仅允许本组 FC1、
直接残差头协议，不借此启用 PCA 解码预测或其他历史模型。
组件仍保存在历史属性名 `fc_autoencoder` 下，以兼容旧 AE checkpoint 的
共有预测层命名；PCA 实例的实际类型为 `FrozenFCPCA`，不加载 AE 权重。

训练冻结完整组件，保存 `component_protocol.json`、best 和 last。
PCA checkpoint 包含参数/登记文件摘要、初始化协议与摘要；重放时核验
两处组件别名下的主成分/均值及冻结基线/统计 buffers，拒绝 checkpoint
覆盖已核验的 PCA 或解析基线。数据入口限定 LR 且核对完整样本、缓存预处理、
FC 形状、有限值与窗口时间轴。

配对评价分别加载双方各自的冻结配置和组件。E0050 允许对 E0036；E0051
允许对 E0037 或 E0050，检查同 seed、数据、训练协议及预定 SC 条件。
模型比较输出 95% 边际区间，主要 MSE 另输出 97.5% 个别区间。
SC 交换使用 seed20261026 的被试循环置换，无自配对；报告匹配/交换的长期
MSE、时间相关、差分相关和有效边比例。

每项对比单独保存 `comparison_<baseline-experiment-id>_val.json`，
保留 baseline run ID 和逐样本指标，避免 E0051 的两项比较相互覆盖。
`evaluation_val.json` 为最近一次评价的完整报告。对同一参照实验换另一个
baseline run 时，旧比较文件须先另存，不会静默覆盖不同 run 的结果。

下面仅为后续执行示例，本次没有执行这些正式命令：

```powershell
scdfc run --experiment configs/experiments/E0050_pca_fc1_direct_residual_transformer_mse_v1.yaml --seed 42 --device cuda
scdfc run --experiment configs/experiments/E0051_pca_fc1_sc_direct_residual_transformer_mse_v1.yaml --seed 42 --device cuda
scdfc evaluate-run --run-id <E0050-run-id> --baseline-run-id <E0036-seed42-run-id> --split val --device cuda
scdfc evaluate-run --run-id <E0051-run-id> --baseline-run-id <E0037-seed42-run-id> --split val --device cuda
scdfc evaluate-run --run-id <E0051-run-id> --baseline-run-id <E0050-run-id> --split val --device cuda
```

本次没有创建正式 E0050/E0051 run。后续若已产生 run，再变更语义协议须分配新实验 ID。

## 6. 适配验证记录

全项目 133 项测试通过，覆盖组件损坏拒绝、AE/PCA 共有层初始化和 RNG 一致、
冻结参数梯度、原首窗解析基线、PCA 双别名 buffers 的篡改拒绝、checkpoint
重放、LR 白名单、分组 bootstrap、无自配对 SC 交换，以及临时合成数据下
AE/PCA 两种组件分别加载的受管训练和评价。所有合成 run 均位于 pytest
临时目录，不是本项目正式实验结果。

RTX 5080、PyTorch 2.13.0+cu130 上还加载了真实 A0044 和现有数据/统计工件，
用人工输入完成 E0050/E0051 的前向与反向检查；输出均为 `[1,223,4005]`，
没有执行优化器更新或生成正式 checkpoint。两组共有预测层初始参数摘要均为
`3e56c17c8b9e494269760a9021cfce9b4faefe27c046dcbd7c4f2f9a337ea6f3`，
与同环境下 E0036 参照构造一致，CPU/CUDA 初始 RNG 状态也一致。

此 GPU 检查出现 CUDA memory-efficient attention 反向非确定性提示。
现有 `seed_everything` 使用 deterministic 的 warn-only 模式，适配保留该协议；
初始化匹配不等于已认证整次 CUDA 训练逐位可复现。项目声明依赖为 PyTorch 2.6，
本机验证为 2.13；正式结果需记录实际运行环境，并保留这些复现范围限制。
