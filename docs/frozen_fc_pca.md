# 冻结 FC PCA 组件 A0044

A0044 将 E0044 已拟合的 PCA-256 保存为独立组件。参数文件为
`outputs/components/A0044/pca.npz`，登记文件为
`configs/artifacts/A0044.yaml`。参数字节与原 E0044 的 `pca.npz` 完全一致，
SHA256 为 `53319615564f4123762664173dd2d2fce51bfd39bf3fe2f035355dbe5e7d4a6e`。
本次导出没有重新拟合 PCA，也没有读取原始 FC 或 train/val/test 缓存。

该组件适用于 dataset_lr_v1 / preprocess_lr_v1 / split_lr_v1，
AAL90、83 TR 窗、stride 5、Fisher-z 上三角 4005 条边。原 PCA 只在
738 名训练被试的 LR 全部 224 窗（165,312 个窗口）上拟合；维数为 256，
不白化、不按边标准化。数据/预处理/划分摘要、原诊断与配置文件摘要、
训练窗口清单摘要均写入登记文件。原 E0044 运行时工作区为 dirty，
该来源限制随组件保留，不能称为干净提交的原始训练。

## 加载与编码

在项目根目录并使用项目 Python 环境：

```python
from scdfc.fc_pca import load_frozen_fc_pca

pca = load_frozen_fc_pca("configs/artifacts/A0044.yaml", device="cuda")
z = pca.encode(fc_z_edges)   # [..., 4005] -> [..., 256]
x_hat = pca.decode(z)       # [..., 256] -> [..., 4005]
x_hat, z = pca(fc_z_edges)
```

`fc_z_edges` 应为与组件相同设备和 dtype 的浮点 PyTorch 张量。
`encode(x) = (x - mean) @ components.T`；
`decode(z) = z @ components + mean`。
`decode` 重建完整 FC，不能直接当作“零均值残差解码”而重复加均值。
模型暴露 `latent_dim`、`n_edges`、`encode`、`decode`；
主成分和训练均值以 buffers 保存，随 `.to(device)` 和 `state_dict()` 移动/保存，
没有可训练参数，也没有拟合方法。

正式接入新预测配置时应传入 `config=load_config(...)`：加载器将核对
数据版本、预处理参数、实际数据 manifest/split 的摘要与潜维数。
单独使用时只需要登记文件及其相对路径指向的参数文件，无需原 E0044 目录
或私人数据。登记文件按 `configs/artifacts/` 的标准目录定位仓库根目录；
从其他工作目录使用时传入登记文件绝对路径。

## 后续预测实验

E0050/E0051 的单窗直接残差 Transformer 已适配组件加载与 `latent_dim` 接口，
支持以 A0044 替换 AE 输入编码，并分别加载 PCA/AE 的参照 checkpoint。
设计和执行方法见 [E0050/E0051 设计](E0050_E0051_pca_input_substitution_design.md)。
现有 E0045–E0047 历史模型仍使用 AE；本轮没有启动正式预测训练。

持续性基线及真实 FC 输出头可继续使用。PCA 输入投影可在线执行或按
组件哈希缓存；在相同数据/预处理/划分下无需再次拟合 PCA。

## 导出复现

```powershell
python scripts/export_fc_pca.py
```

脚本先检查 E0044 归档 PCA、数据审计和冻结配置的原始文件摘要，
再复制参数并写登记文件。已有 A0044 时会拒绝覆盖；更改参数或来源时须
分配新的工件 ID。导出目录与登记文件不包含原始 FC、重建序列或被试列表。
