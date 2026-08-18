# Checkpoint 归档与按需拉取

`main` 只保存代码、配置、指标、日志、评估结果和可视化；模型 checkpoint 不应随普通实验提交进入 `main`。完整模型保存在同一远端的 `checkpoint-archive` 分支，并由 Git LFS 按需下载。

当前可用模型的路径、LFS SHA256 与文件大小均登记在 [`reports/checkpoint_catalog.json`](../reports/checkpoint_catalog.json)。该索引也是实验记录与完整模型之间的稳定连接。

## 日常轻量使用

首次在一台机器上使用仓库时，关闭 LFS 的检出期自动下载：

```powershell
git lfs install --skip-smudge
git pull
```

这不会影响代码、实验配置、训练日志、指标或可视化；仅会让 checkpoint 保留为轻量 LFS 指针，直到显式请求下载。

## 拉取一个 checkpoint

在项目根目录执行，例如拉取 E0011：

```powershell
.\scripts\pull-checkpoint.ps1 -ExperimentId E0011
```

脚本会在项目同级新建 `SC2dFC-checkpoints` worktree，从 `checkpoint-archive` 获取目录树，但只通过 LFS 下载请求的 `best.pt`。完成后会输出模型的绝对路径。若某实验将来有多个 run，则额外指定 run：

```powershell
.\scripts\pull-checkpoint.ps1 `
  -ExperimentId E0011 `
  -RunId E0011-s42-20260818T083706Z-febe983
```

可用 `-Destination D:\models\SC2dFC` 自定义该独立 worktree 的位置。不要把该 worktree 直接用于代码开发；主工作区仍应保持在 `main`。

## 归档未来的 checkpoint

完成一次训练后，默认只提交轻量实验记录。只有决定长期保留、复现或共享的 run 才归档：

1. 将对应 `best.pt` 放入 `checkpoint-archive` 分支的同一路径并以 Git LFS 提交；
2. 在 `reports/checkpoint_catalog.json` 新增 experiment ID、run ID、相对路径、LFS OID 和字节数；
3. `main` 只提交上述索引及轻量的 config、metadata、metrics、log、evaluation 和可视化；
4. 以独立提交推送 archive 分支和 `main`，避免普通实验提交意外携带模型。

归档分支会保留模型存储用量；这是“在线可按需获取”所必需的代价。不要执行 `git lfs fetch --all`，否则会下载全部归档模型。
