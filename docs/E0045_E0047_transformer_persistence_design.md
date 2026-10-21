# E0045–E0047：多窗历史、持续性基线与 Transformer 未来残差预测

确认及归档日期：2026-10-21。

**状态：方案已确认，模型、配置、准备入口及评估支持已实现；尚未进行正式训练。** 本文记录本次最终选择，不将前面对第20窗起点或 cross-attention 的建议当作已确认方案。E0044 仍保留给独立 AE 重建诊断。

## 1. 目的和固定实验矩阵

研究问题：在相同观察截止点、持续性基线、Transformer 未来解码器、训练预算及标签下，增加历史 FC 是否改善未来数值预测和真实时间变化？

| 实验 | K | 输入（从1计数） | 目标 | 主要对照 |
| --- | ---: | --- | --- | --- |
| E0045 | 1 | 第10窗 | 第11–224窗，H=214 | 解析持续性基线 |
| E0046 | 5 | 第6–10窗 | 第11–224窗，H=214 | E0045 |
| E0047 | 10 | 第1–10窗 | 第11–224窗，H=214 | E0045，另列E0046 |

共同零基 origin 为9：`history = FC[10-K:10]`，`last = FC[9]`，`target = FC[10:]`。不把K个高度重叠窗当成独立观察。K=1/5/10历史实际覆盖59.76/74.16/92.16秒，末目标相对FC10平移214×5×0.72=770.4秒。

固定使用 LR-only、AAL90、Fisher-z上三角4005边、1200 TR、83 TR窗、stride=5、TR=0.72秒，沿用 dataset_lr_v1 / preprocess_lr_v1 / split_lr_v1。训练/验证预期738/158名被试，以真实数据审计为准，不读取test。保留现有离线预处理协议；不据此宣称已完成在线预处理因果性验证。

本轮只改变K，不加入SC、历史Transformer、cross-attention、差分输入、AE微调、相关/差分/方差损失。与E0040–E0043的截止点不同，旧指标不能直接排名并归因于解码器。

## 2. 持续性基线：第10窗个体偏移的共享时距系数

以训练集对应窗口平均FC为 `g_t`，对被试s、未来h=1…214：

\[
B_{s,h}=g_{10+h}+\alpha_h(X_{s,10}-g_{10}),
\qquad \widehat X_{s,10+h}=B_{s,h}+\widehat R_{s,h}.
\]

三组都使用FC10建立个体偏移，不用各自历史均值改变基线。每个h的系数按训练被试与全部边最小二乘拟合：

\[
\alpha_h^{raw}=\frac{\sum_{s,e}(X_{s,10,e}-g_{10,e})(X_{s,10+h,e}-g_{10+h,e})}
{\sum_{s,e}(X_{s,10,e}-g_{10,e})^2+10^{-12}},
\quad \alpha_h=\operatorname{clip}(\alpha_h^{raw},0,1).
\]

每个预测时距一个标量，所有被试和边共用，拟合后冻结；允许局部回升，不强制单调。仅训练集LR缓存参与，不读SC，也不读取验证/测试被试的FC。三组引用同一个工件及SHA256；不能直接截取E0032的首窗alpha。

准备工件一次保存训练集完整FC模板（内部沿用`FC[1:]`存储约定）、第10窗模板、214个alpha_raw/alpha、数据与split协议、拟合被试清单和训练FC摘要。需要每个训练被试恰好一个完整LR run；不完整数据、形状错误、非有限值、拟合期间缓存改变应失败。训练和重载检查协议、形状、取值范围、哈希及训练样本清单。

## 3. 模型：历史GRU与未来Transformer各司其职

```text
[B,K,4005] 历史FC
    ↓ 冻结A0003编码器，逐窗4005→1024→512→256
[B,K,256]
    ↓ 单层单向GRU，hidden=256
[B,256] 最后隐藏状态
    ↓ LayerNorm，得到c
214个可学习未来query分别加c
    ↓ 4层TransformerEncoder式未来self-attention
    ↓ 8头、FFN1024、GELU、dropout0.1、pre-norm、末端LayerNorm
[B,214,256]
    ↓ 共享Linear(256,4005)
[B,214,4005] 预测残差
    ↓ 加回持续性基线B
最终预测
```

未来query之间可以交换信息，但不读取真实未来。214窗一次输出，不递归、不累计残差、不使用teacher forcing。既有AE解码器不参与预测。全部K具有同一参数形状与同seed初始化，可训练参数4,638,885；AE保持冻结及eval。模型初始化和样本打乱分开控制随机源。

训练使用完整214窗逐边纯MSE；优化残差对`target-B`与优化最终预测对target等价。AdamW lr3e-4、wd1e-4、batch4、clip1、150轮、patience151，5轮warm-up后cosine至0。**以完整214窗验证被试平均MSE选best，平分保留最早epoch**，保存best和last。

配置预先声明seed42/43/44：先完成三组42，复验统一补三组43/44并保留全部结果。当前不执行任何正式训练。

## 4. 评价与解释

主要报告完整214窗；另列近端`[0:17]`与保守无重叠`[17:214]`共197窗，不用后者替代完整指标。附加short`[0:40]`；无重叠早/中/晚段分别`[17:82]`、`[82:148]`、`[148:214]`。

必报：MSE、空间Pearson、逐边时间Pearson、去群体时间模板时间Pearson、相邻差分时间Pearson、预测/真实时间标准差比、有效边/近常数边比例。另将预测残差与真实`target-B`比较时间相关、差分相关、幅度和MSE，区分基线和学习分支的作用。

共同对照为解析持续性基线、群体时间模板、FC10恒定保持；历史均值是各K的补充参考，不能称作三组共同基线。固定seed预选被试与边，画真实、基线和预测的原始曲线，保存选择清单，不按验证表现挑图。

先run、再被试等权聚合；相关以float64计算，近常数阈值1e-6，不可定义时为null并报告覆盖率。被试bootstrap10000次、seed20261021；主比较K5−K1和K10−K1报95%区间及两项Bonferroni调整的97.5%区间。调整只覆盖这两个K比较，不宣称同时校正所有指标和基线。不同seed变异另列，不把seed当作更多独立被试。

- 仅MSE改善：只支持数值预测改善。
- 仅标准差比提高：不能证明时间预测成功。
- 时间及差分相关、幅度和误差需共同解释；保留实际权衡，不事后改选模规则。
- 持续性残差时间相关不能替代原始FC时间相关。

## 5. 配置和准备操作（不启动神经训练）

配置：

- `configs/experiments/E0045_history_k1_transformer_persistence_v1.yaml`
- `configs/experiments/E0046_history_k5_transformer_persistence_v1.yaml`
- `configs/experiments/E0047_history_k10_transformer_persistence_v1.yaml`

新模型标识`fc_history_gru_transformer_decay_residual`，分解`history_decay_residual`。旧E0036/E0040路径保持原有校验，不通过放开旧配置约束混用新协议。

真实工件尚未拟合，三个配置中的`artifacts.history_persistence.sha256`明确为null，登记PLANNED。配置结构校验可通过，但`run`会在创建运行前拒绝未绑定工件；不得填写虚假哈希绕过。数据机器上准备步骤如下：

```powershell
# 1. 只拟合训练集解析模板/alpha；不加载AE，不训练神经网络
scdfc fit-history-persistence --experiment configs/experiments/E0045_history_k1_transformer_persistence_v1.yaml

# 2. 显式绑定真实哈希到三组配置，并刷新未运行PLANNED登记行的配置哈希
scdfc bind-history-persistence --artifact outputs/shared/dataset_lr_v1/window_83/history_persistence_origin9_v1.npz --experiments configs/experiments/E0045_history_k1_transformer_persistence_v1.yaml configs/experiments/E0046_history_k5_transformer_persistence_v1.yaml configs/experiments/E0047_history_k10_transformer_persistence_v1.yaml
```

两个命令在本次留档工作中均不对真实数据执行。工件拟合拒绝覆盖已有文件；绑定拒绝修改已有run或登记哈希不一致的实验。工件默认位于忽略的`outputs/shared/`，需随数据机器保存；跨机器复现须复制原工件并核对哈希。不能在另一台机器重新拟合后静默替换。

后续真正获准训练时，先检查数据和A0003真实权重、提交完成绑定的配置与代码，保证Level1工作区干净，再使用既有`scdfc run --experiment <yaml> --seed <seed> --device cuda`。

训练后的`evaluate-run`会输出完整评价、解析基线、持续性残差诊断、置信区间及固定曲线图。K5/K10可用`--baseline-run-id <同seed的E0045-run-id>`，调用前严格检查协议。`history-summary --run-ids ...`可生成跨K/seed表与图，新族默认写`reports/history_transformer_persistence/`，不与旧族混合。

## 6. 验证边界

本次只进行配置、形状/梯度、合成数据及代码回归检查，不生成真实科研训练结果。检查包含：共同起点与标签、训练集专用拟合和旧RL排除、零残差退化为基线、AE冻结、同seed参数匹配、真实未来不进入前向、工件绑定与重载拒绝错误、完整214窗选模、常量相关及配对评价、末段覆盖至214、图表与汇总。

测试CPU环境为本机Python3.11 / PyTorch2.4.1，不能据此宣称已验证CUDA或项目目标PyTorch2.6。正式执行状态以run及数据审计为准，不以这份设计文档代替结果。

本次实现验证：全项目103项测试通过；新增5项测试使用临时合成缓存、随机模型推理与检查点，不运行新实验的优化训练。既有回归测试包含其原有的小型合成训练流程，均不产生真实科研run。真实数据和A0003完整权重本机均不存在，真实持续性工件仍待在数据机器准备。
