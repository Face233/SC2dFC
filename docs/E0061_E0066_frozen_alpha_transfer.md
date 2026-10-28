# E0061–E0066：冻结223参数α接入AE残差Transformer

2026-10-28。按用户要求，保留原 E0036/E0037 残差结构，分别接入
E0054（Linear-223）、E0057（MLP-223）、E0060（GCN-223）。
E0061–E0066现均按早停耐心20轮运行，总预算仍为150轮。最初按151轮耐心
启动的E0061在第17轮停止并记为FAILED；该checkpoint不参与评价，新E0061从头重跑。

| 编号 | 残差骨干 | 冻结α来源 |
| --- | --- | --- |
| E0061 | E0036：FC1-only残差 | E0054 |
| E0062 | E0036：FC1-only残差 | E0057 |
| E0063 | E0036：FC1-only残差 | E0060 |
| E0064 | E0037：FC1+SC-GCN残差 | E0054 |
| E0065 | E0037：FC1+SC-GCN残差 | E0057 |
| E0066 | E0037：FC1+SC-GCN残差 | E0060 |

FC输入继续使用冻结 E0003/A0003 AE encoder（4005边→256维）。输出是直接
4005边残差头，不使用 AE reconstruction decoder。没有替换成 FC-PCA；
E0054/E0057的PCA64只处理SC，E0060处理原始SC图。

## 前向与训练

`prediction = mu[t] + (alpha_raw[t] + delta_alpha(SC,t)) * (FC1-mu0) + residual[t]`。

冻结整个α组件，包括SC标准化/PCA、网络、基矩阵及已有模板。α不截断。
从头训练原融合层、Transformer及直接残差头；E0037同时训练原残差SC-GCN。
保留原纯MSE、batch4、学习率3e-4、AdamW、warmup-cosine、150轮、不早停，
验证长期target[17:223]选模。没有加载原残差模型训练权重进行微调。

E0036派生条件需要真实SC供α使用，因此数据层不再清零原始SC；残差条件
仍在SC投影后清零。不得把α分支一起消融。所有条件的FC encoder保持冻结。
父模型train模式不会启用α MLP的dropout，冻结参数不进入优化器。

构建α组件用独立RNG上下文，避免改变残差初始化或后续dropout流。记录六组
初始可训练参数、CPU/CUDA RNG、训练样本哈希并逐项比较。零α修正恢复原模型。
冻结组件与残差checkpoint绑定SHA256；重载校验α参数未随训练改变。

## 评价与比较

主要比较：六组分别对其原E0036/E0037骨干。预先固定六项比较家族，100000次
被试配对bootstrap，报告95%及99.1667%个别区间；后者对应Bonferroni家族95%。
仅使用既有158人验证集，结果仍是探索性证据，不能称为全新独立验证。

同次推理列出原骨干、原骨干直接叠加delta_alpha*d的posthoc组合、重训组合、
共享E0032、独立个体alpha五个条件。所有动态残差指标统一减共享E0032，
不能各减自己的解析基线。报告长期/全未来/分段MSE、原始/去模板时间相关、
去共享基线动态相关、差分相关、动态标准差比，以及α与残差分支的动态抵消程度。

冻结SC交换共10次，种子20261028–20261037，被试无自配对；只交换SC，
FC1及目标保持原被试。E0036条件交换α路径；E0037分别交换α、残差、两路径。
先按被试平均十次误差再bootstrap。α跨被试离散度、被试内波动、越界比例单列。
幅度增加和真实时序对齐分别判读。验证后脚本生成聚合比较表及可重绘指标图。
逐被试清单与真实/预测例子仅保存于本地private目录。

## 命令与执行

```powershell
$python = 'C:\ProgramData\miniconda3\envs\fmri\python.exe'
& $python scripts/run_alpha_transfer_suite.py --phase check
& $python scripts/run_alpha_transfer_suite.py --phase all --device cuda --experiments E0061 E0062 E0063 E0064 E0065 E0066
```

支持check/train/evaluate/summary/all阶段；默认是E0061–E0063；完成的同协议run可复用；遇到RUNNING
记录停止，避免重复训练。all先完成六组训练，再评价和汇总。

正式训练要求干净Git目录。主目录已有改动，使用独立代码快照
`D:/Code/Canlab/_codex_runs/E0061_E0066`，包含实际执行源码和配置的本地Git提交；
数据和输出目录以junction指向主工作区，保留源工件并让结果直接出现在主目录。
不提交或重置主工作区。控制台日志位于`outputs/shared/alpha_transfer_suite/`。

实现验证：合成测试覆盖三种α重放、冻结/dropout/梯度、残差路径隔离、数据层SC
保留、零修正、checkpoint重载与篡改拒绝、六个配置、bootstrap及两种骨干完整评价。

六组训练、验证和三种适用SC交换诊断已完成。E0061–E0063分别在36/26/26轮早停，
E0064–E0066均在35轮早停。联合聚合报告位于
`outputs/E0066/visual/E0061_E0066_comparison_val.md`；E0065/E0066长期MSE均值略低于
原E0037，但六项主比较的被试配对区间均覆盖零。本轮仍为单seed、重复使用验证集的
探索性结果。最佳checkpoint通过`checkpoint-archive`分支与catalog按需获取。
六组真实训练被试的CUDA前后向预检及初始化/RNG/被试清单匹配通过，见
`reports/sc_conditioned_alpha/alpha_transfer_cuda_preflight.json`。预检不构成科研结果。
