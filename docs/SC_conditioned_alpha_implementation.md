# E0052–E0060：SC 个体化 α 的实现与运行

实现依据：[完整实验计划 v3](SC_conditioned_alpha_experiment_plan_v3.md)。
2026-10-27：已实现并登记为 PLANNED；没有启动真实被试训练或读取验证集进行科研评价。

## 实验编号

| 编号 | SC 映射 | 参数数 | 展开方式 |
| --- | --- | ---: | --- |
| E0052 | Linear | 1 | 常量广播到223窗 |
| E0053 | Linear | 6 | 固定三次B样条 |
| E0054 | Linear | 223 | 时间点一一对应 |
| E0055 | MLP | 1 | 常量广播 |
| E0056 | MLP | 6 | 固定三次B样条 |
| E0057 | MLP | 223 | 时间点一一对应 |
| E0058 | HCP-GCN | 1 | 常量广播 |
| E0059 | HCP-GCN | 6 | 固定三次B样条 |
| E0060 | HCP-GCN | 223 | 时间点一一对应 |

B0不重复训练，在每项验证评价时以同一模板和alpha_raw重放。
九项配置共享 `configs/sc_alpha_v1.yaml`，各配置只覆盖编号、研究描述、编码器和参数数。

## 学习与隔离

- 新受管任务为 `experiment.task: sc_alpha`，模型名 `sc_conditioned_alpha`。
- 全部223窗训练；未加正则的长期206窗MSE用于训练内五折选参和选轮次。
- 每折独立估计模板、未截断共享α及SC变换；Linear/MLP用标准化log1p SC-PCA64，
  GCN沿用现有两层图编码器。修正头初始化为零；同编码器不同K保持相同初始编码器/RNG。
- 搜索λ=0/0.01/0.1/1，五折加权选参，最终全训练轮数为所选λ的折内最佳轮次中位数。
- α=alpha_raw+Δα，没有clip/sigmoid边界。幅度惩罚作用于展开后的Δα，三种参数化尺度可比。
- 使用精确充分统计量计算逐边MSE，避免每步生成大尺寸FC预测；训练只载入训练被试。
- 正式评价前要求九项均完成，协议、训练内容哈希、折划分哈希与3×3条件矩阵一致。
  Level1/2不能绕过该要求；Level0只供合成数据端到端检查。
- 标准化/PCA和检查点原样保存。组件推理仅依赖SC和首窗FC，不要求原始训练数据。

## 分阶段命令

在仓库根目录使用项目Python环境；Level1正式训练仍要求干净Git工作区。
以下命令只作为运行说明，本次实现未执行 train/evaluate/summary 阶段。

```powershell
$python = 'C:\ProgramData\miniconda3\envs\fmri\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'

# 默认check只验证九项登记配置、数据清单及冻结工件，不拟合FC或模型。
& $python scripts/run_sc_alpha_suite.py --phase check

# 九项只完成训练内选参与全训练拟合，不自动评价验证集。
& $python scripts/run_sc_alpha_suite.py --phase train --device cuda

# 全部完成后统一评价；随后生成十条件比较与次要对照。
& $python scripts/run_sc_alpha_suite.py --phase evaluate --device cuda
& $python scripts/run_sc_alpha_suite.py --phase summary
```

套件脚本会复用同协议的已完成训练；遇到RUNNING记录会停止，避免重复启动。
单项仍可用既有 `scdfc run --experiment <配置> --seed 42 --device cuda`。
单项 `evaluate-run --run-id <run> --split val` 会自动比较B0并做十次冻结SC交换，
但九项完成要求仍生效。它拒绝test、外部baseline参数及私有预测数组导出选项。

新命令：

```text
scdfc alpha-summary --run-ids <九个run ID> [--output-dir <可视化目录>]
scdfc alpha-visual --run-id <已评价run>
```

默认套件汇总位于 `outputs/E0060/visual/`。指定其他输出目录时，应遵守项目图像目录合约，
使用某个参与实验的 `outputs/E####/visual/`。

## 产物和指标含义

每项run保存：

- `cv_results.json`：所有λ和各折结果/失败记录；每折λ日志记录逐轮训练及内部留出MSE。
- `alpha_selection.json`：选中λ、最终轮数、训练内容/折划分/初始参数/RNG哈希。
- `train.log`：最终固定轮数的全训练过程。
- `checkpoints/best.pt`：该固定轮数最终模型，不是在外部验证集挑选的best。
- `component_manifest.json`：组件schema、校验和、协议绑定与无截断说明。
- `metrics_best.json`：**训练内CV选参指标，split=train_cv**。不冒充外部验证集结果。
- `evaluation_val.json`：评价后单独生成的外部验证聚合值、B0、九重校正区间、SC交换及α诊断。
- `private/`：折内被试清单、逐被试指标及绘图示例，已加入Git忽略规则。

既有 `scdfc summarize --experiment E0052` 汇总的是selection记录，明确标为train_cv；
外部验证效果以 `evaluation_val.json` 和 `alpha-summary` 为准。重复评价不覆盖CV选模记录。

九项主比较使用100,000次被试bootstrap；同时报告95%与99.4444%个别区间。
次要比较及SC交换使用10,000次，交换的十次误差先按被试平均，不当作十倍样本。
组内自由度及同K编码路线对照共18项，均为探索性比较。

验证图每项5张：训练概览、α/修正分布、固定四边真实/预测、动态校准、分段MSE；
套件另有指标、主比较区间、SC交换/α曲线3张。均生成同目录重绘脚本与来源记录。
绘图使用匿名的预先固定验证样本，真实蓝/预测橙，同一条边跨列共享y轴。
相关性无定义时记null及有效比例，绘图容许缺失；基线之外的残差统一相对B0计算。

## 独立组件重放

复制 `checkpoints/best.pt` 和原层级的 `component_manifest.json` 即可组成独立组件目录。
加载时校验参数文件SHA256、配置哈希和固定基矩阵；受管评价额外核验外部冻结工件。

```python
from scdfc.sc_alpha import load_sc_alpha_component

component = load_sc_alpha_component('.../checkpoints/best.pt', device='cuda')
alpha = component.predict_alpha(sc_matrices)   # [batch, 90, 90] -> [batch, 223]
future_fc = component.predict(sc_matrices, first_fc_edges)  # -> [batch, 223, 4005]
```

实现检查使用临时目录中的小型合成数据，涵盖三编码器、参数/RNG配对、无截断梯度、
充分统计量等价、内部折隔离、训练/验证隔离、九项就绪门槛、SC置换、冻结重载、
结果篡改检测及汇总绘图；不创建正式E0052–E0060结果。

2026-10-27实现检查结果：全套161项测试通过，其中新增27项SC-alpha检查；
包含三编码器的CUDA合成训练路径、受管CLI运行/评价/汇总和完整九条件套件。
九项登记配置、冻结清单和工件的只读预检通过；`git diff --check`通过。
测试绘图使用合成数据，仅检查程序与布局，不构成科研结果。

2026-10-28补充检查：端到端合成训练/保存/重载/评价/绘图已扩展到完整3×3九组组合；
全套169项测试通过。九组均以正式64维特征或90×90图的合成输入通过CUDA前向/反向检查。
非有限梯度统一记录为当前λ的拟合失败，交叉验证继续检查其余候选，避免异常中断整个实验。
