"""Illustrate the actual six-basis expansion; coefficients/FC edge are hypothetical."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from scdfc.models.sc_alpha import alpha_basis

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
basis = alpha_basis(223, 6).astype(float)
beta = np.array([.08, .06, .02, -.01, -.03, -.04])
with np.load(ROOT/'outputs/E0032/runs/E0032-s42-20260923T052241Z-8d7ca92/alpha_fit.npz') as archive:
    common = archive['alpha_raw']
delta = basis @ beta
individual = common + delta
x = np.arange(1, 224)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
colors = plt.cm.tab10.colors
for k in range(6):
    axes[0, 0].plot(x, basis[:, k], color=colors[k], label=f'B{k+1}(t)')
axes[0, 0].set(title='① 固定的六个平滑形状：相邻形状的影响范围重叠', xlabel='未来窗口 t', ylabel='基函数权重')
axes[0, 0].legend(ncol=3, frameon=False)
for k in range(6):
    axes[0, 1].plot(x, basis[:, k]*beta[k], color=colors[k], alpha=.6, lw=1)
axes[0, 1].plot(x, delta, color='black', lw=2.2, label='修正曲线：六条加权曲线相加')
axes[0, 1].axhline(0, color='gray', ls='--', lw=.8)
axes[0, 1].set(title='② SC 输出六个系数后，生成整条修正曲线', xlabel='未来窗口 t', ylabel='修正量 Δα(t)')
axes[0, 1].legend(frameon=False, fontsize=9)
axes[1, 0].plot(x, common, color='tab:blue', label='现有通用 α0(t)')
axes[1, 0].plot(x, individual, color='tab:orange', label='个体 α(t) = 通用 α0(t) + 修正 Δα(t)')
axes[1, 0].set(title='③ 在通用 α 上加修正，不对最终 α 做截断', xlabel='未来窗口 t', ylabel='α')
axes[1, 0].legend(frameon=False, fontsize=9)
axes[1, 1].plot(x, .1+.4*common, color='tab:blue', label='通用 α 预测')
axes[1, 1].plot(x, .1+.4*individual, color='tab:orange', label='个体 α 预测')
axes[1, 1].set(title='④ 影响一条 FC 边的预测：μ(t)=0.10，首窗偏移 d=0.40', xlabel='未来窗口 t', ylabel='预测 FC（Fisher-z）')
axes[1, 1].legend(frameon=False)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('B 样条如何影响预测（示意，不是新实验结果）\n假设 SC 输出 β = [+0.08, +0.06, +0.02, −0.01, −0.03, −0.04]', fontsize=14)
out = Path(__file__).resolve().parent
for extension in ('png', 'svg'):
    fig.savefig(out/f'bspline_alpha_explanation.{extension}', dpi=170, facecolor='white')
plt.close(fig)
for index in (0, 111, 222):
    print('t=',index+1,'B=',np.round(basis[index],4).tolist(),'delta=',float(delta[index]),'delta_FC=',float(.4*delta[index]))
