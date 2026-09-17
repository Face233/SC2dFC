# Checkpoint 归档与按需拉取

目标约定是：普通实验提交只发布轻量记录，决定长期保留的 checkpoint 放入 `checkpoint-archive` 分支并由 Git LFS 按需下载。**当前仓库尚未完全遵守该约定**：截至 `main` 的 `6ea9cb1`，E0014–E0024 共 11 个 `best.pt` LFS 指针也被跟踪在 `main`；归档分支与下述索引只包含 E0003、E0004–E0007、E0009–E0011 共 8 个 checkpoint。`.gitignore` 不会自动取消这些已跟踪文件。

[`reports/checkpoint_catalog.json`](../reports/checkpoint_catalog.json) 只索引上述 8 个归档模型的路径、LFS SHA256 与文件大小；不能用它断言 E0014–E0024 已归档。E0013 的登记运行 checkpoint 当前不可用；没有相应模型文件时，文档、图和旧汇总不足以重建其训练选模结果。

## 日常轻量使用

首次在一台机器上使用仓库时，关闭 LFS 的检出期自动下载：

```powershell
git lfs install --skip-smudge
git pull
```

这不会影响代码、配置、训练日志、指标或可视化；对 `main` 中已跟踪的 E0014–E0024 模型以及归档分支中的模型，只会先保留 LFS 指针，直到显式下载。指针本身不是可供 PyTorch 加载的完整 checkpoint。

## 拉取一个 checkpoint

脚本只支持 catalog 中登记的归档模型。在项目根目录执行，例如拉取 E0011：

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

完成一次新的训练后，先核查 `git status` 和 `git ls-files`，避免再次把 checkpoint 指针加入 `main`。只有决定长期保留、复现或共享的 run 才归档：

1. 将对应 `best.pt` 放入 `checkpoint-archive` 分支的同一路径并以 Git LFS 提交；
2. 在 `reports/checkpoint_catalog.json` 新增 experiment ID、run ID、相对路径、LFS OID 和字节数；
3. 新实验的 `main` 提交只放上述索引及经许可审阅的轻量 config、metadata、metrics、log、evaluation 和可视化；既有 E0014–E0024 指针需另行迁移与核对，不能仅修改文档就认为它们已从 `main` 移除；
4. 以独立提交推送 archive 分支和 `main`，避免普通实验提交意外携带模型。

归档分支会保留模型存储用量；这是“在线可按需获取”所必需的代价。不要执行 `git lfs fetch --all`，否则会下载全部归档模型。把现有 11 个 `main` 指针迁移到归档分支需要先核对模型哈希、更新 catalog，再审慎修改 Git 跟踪；本次文档同步不执行这些操作。
