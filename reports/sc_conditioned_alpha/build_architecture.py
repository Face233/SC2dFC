"""Draw the proposed SC-conditioned alpha architecture; no data or training."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(19, 13))
fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.025)
ax.set(xlim=(0, 19), ylim=(0, 13))
ax.axis('off')
GRAY = '#64748b'
BLUE = '#2878b5'
ORANGE = '#c86b24'
GREEN = '#26836b'

def box(x, y, w, h, text, fill='#f1f5f9', edge='#cbd5e1', color='#172b44', size=12, dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle='round,pad=0.025,rounding_size=.10',
                              linewidth=1.3,facecolor=fill,edgecolor=edge,
                              linestyle='--' if dashed else '-'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=color,linespacing=1.65)

def arrow(points, color=GRAY, dashed=False):
    for a,b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.6,ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,
                                linewidth=1.6,color=color,linestyle='--' if dashed else '-'))

ax.text(.2,12.65,'SC 条件化个体 α：1 / 6 / 223 参数对照（计划 v3）',fontsize=22,weight='bold',color='#172b44')
ax.text(.2,12.18,'α 组件已实现 · 尚未正式训练    |    μ(t)、μ0 与通用 α0(t) 保持固定    |    t = 1…223，FC 为 4005 维 Fisher-z 边向量',fontsize=11,color=GRAY)

box(.2,9.65,18.6,2.1,'',fill='#f6f8fb')
ax.text(.45,11.4,'① 通用部分：沿用现有训练集拟合结果，后续冻结',fontsize=14,weight='bold',color='#334155')
box(.5,9.95,3.0,.95,'训练集 FC\n首窗 + 未来窗口',fill='white')
box(4.35,9.95,4.0,.95,'群体模板（固定）\nμ0：首窗均值；μ(t)：未来各窗均值',fill='#e9eff5',size=11)
box(9.25,9.95,4.2,.95,'训练集最小二乘拟合\n首窗偏移 → 未来 FC 偏移',fill='white',size=11)
box(14.4,9.95,3.95,.95,'固定通用 α0(t) = alpha_raw\n现有工件与截断值完全相同',fill='#e9eff5',size=11)
arrow([(3.55,10.43),(4.3,10.43)])
arrow([(8.4,10.43),(9.2,10.43)])
arrow([(13.5,10.43),(14.35,10.43)])

ax.text(.35,9.13,'② 个体化 α 组件：只训练 SC → 修正曲线的映射',fontsize=15,weight='bold',color=ORANGE)
box(.5,7.4,2.35,1.0,'被试 SC_s\n90 × 90',fill='#fff3e7',edge='#e7b27f')
box(3.6,7.4,3.7,1.0,'SC 特征 → 参数头 gθ\n线性 / 小型 MLP / HCP-GCN',fill='#fff3e7',edge='#e7b27f',size=11)
box(8.05,7.4,2.5,1.0,'个体参数 β_s\nK = 1 / 6 / 223',fill='#fff3e7',edge='#e7b27f')
box(11.3,7.4,3.3,1.0,'1：常量；6：平滑样条\n223：逐时间点直接输出',fill='#fff3e7',edge='#e7b27f',size=11)
box(15.35,7.4,3.0,1.0,'α_s(t) = α0(t) + Δα_s(t)\n无硬截断，可小于0或大于1',fill='#ffebd5',edge=ORANGE,size=11)
for a,b in [(2.9,3.55),(7.35,8.0),(10.6,11.25),(14.65,15.3)]:
    arrow([(a,7.9),(b,7.9)],ORANGE)
arrow([(16.4,9.9),(16.4,8.45)],GRAY)
ax.text(5.45,7.02,'可学习 θ；修正输出初始化为 0',ha='center',color=ORANGE,fontsize=10)
ax.text(12.95,7.02,'少量参数展开为完整 223 点序列',ha='center',color=ORANGE,fontsize=10)

box(.5,5.15,3.1,1.0,'被试首窗 FC_s(0)\n推理时可观测',fill='#e8f2fc',edge='#95bedf')
box(4.45,5.15,3.9,1.0,'首窗个体偏移\nd_s = FC_s(0) − μ0',fill='#e8f2fc',edge='#95bedf')
box(9.2,5.15,4.25,1.0,'个体化解析预测基线\nFC_base(t) = μ(t) + α_s(t) d_s',fill='#e5f5ef',edge=GREEN,size=12)
box(14.4,5.15,3.95,1.0,'阶段一输出\n未来 FC 预测 [223 × 4005]',fill='#e5f5ef',edge=GREEN,size=11)
arrow([(3.65,5.65),(4.4,5.65)],BLUE)
arrow([(8.4,5.65),(9.15,5.65)],BLUE)
arrow([(13.5,5.65),(14.35,5.65)],GREEN)
arrow([(16.85,7.35),(16.85,6.65),(11.35,6.65),(11.35,6.2)],ORANGE)
ax.text(6.35,6.45,'固定 μ0',ha='center',color=GRAY,fontsize=10)
ax.text(10.1,4.83,'μ(t) 来自训练集模板，保持不变',color=GREEN,fontsize=10)

box(.5,3.05,3.1,.95,'训练被试真实未来 FC_s(t)\n仅作为训练监督',fill='#f5edfa',edge='#baa2cd',size=11)
box(4.45,3.05,9.0,.95,'训练：全部 223 窗 MSE + λ mean(Δα²)；选参：长期 206 窗 MSE\n训练内五折；seed42；九组个体模型 + 通用 α 基线',fill='#f5edfa',edge='#baa2cd',size=12)
arrow([(3.65,3.52),(4.4,3.52)],'#87639e',True)
arrow([(16.4,5.1),(16.4,3.52),(13.5,3.52)],'#87639e',True)
arrow([(5.45,4.05),(5.45,4.65),(3.95,4.65),(3.95,7.35)],ORANGE,True)
ax.text(.55,2.6,'虚线：监督 / 反向更新 θ。验证和新被试的 α 仅由 SC 生成；不使用其未来 FC 拟合。',fontsize=11,color='#87639e')

box(.2,.35,18.6,1.8,'',fill='#fbfcfe',dashed=True)
ax.text(.45,1.82,'③ 后续接入完整预测模型（组件验证有效后，再单独训练）',fontsize=13,weight='bold',color='#334155')
box(.55,.62,4.2,.72,'首窗 FC 编码（AE / PCA）+ SC',fill='white',size=11)
box(5.6,.62,4.35,.72,'条件融合 → Transformer → 残差 R_pred(t)',fill='white',size=11)
box(10.8,.62,3.05,.72,'冻结个体化 α 组件\n计算 FC_base(t)',fill='#e5f5ef',edge=GREEN,size=10)
box(14.7,.62,3.7,.72,'最终 FC = FC_base + R_pred',fill='#e8f2fc',edge=BLUE,size=12)
arrow([(4.8,.98),(5.55,.98)],BLUE)
arrow([(10.0,.98),(10.35,.98),(10.35,.48),(16.5,.48),(16.5,.58)],BLUE)
arrow([(13.9,.98),(14.65,.98)],GREEN)

for extension in ('png','svg'):
    fig.savefig(OUT / f'sc_conditioned_alpha_architecture_v3.{extension}',dpi=180,facecolor='white')
plt.close(fig)
print(OUT / 'sc_conditioned_alpha_architecture_v3.png')
