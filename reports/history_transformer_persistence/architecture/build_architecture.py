"""Draw the implemented E0045--E0047 architecture; no experiment inference."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(16, 13), facecolor="white")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 16), ylim=(0, 13))
ax.axis("off")
INK = "#183047"
BLUE = "#2467A7"
GRAY = "#64748B"
ORANGE = "#B96B18"
GREEN = "#237552"


def text(x, y, value, size=12, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, value, ha="center", va="center", fontsize=size,
                   color=color, weight=weight, linespacing=1.45, **kwargs)


def box(x, y, w, h, title, detail="", color=BLUE, fill="#F0F7FE", ts=13, ds=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                               facecolor=fill, edgecolor=color, linewidth=1.35))
    if detail:
        text(x+w/2, y+h*.70, title, ts, color, "bold")
        text(x+w/2, y+h*.30, detail, ds)
    else:
        text(x+w/2, y+h/2, title, ts, color, "bold")


def arrow(points, color=GRAY, width=1.6):
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                                linewidth=width, color=color, zorder=1))


text(8, 12.65, "E0045–E0047  |  FC 历史 → GRU → 未来 Transformer → 残差预测", 20, weight="bold")
text(8, 12.17, "三组仅改变历史长度 K；共同截止于第 10 窗，预测第 11–224 窗（H = 214）", 12, GRAY)

for x, eid, kval, history in [(.9,"E0045",1,"第 10 窗"), (5.8,"E0046",5,"第 6–10 窗"),
                              (10.7,"E0047",10,"第 1–10 窗")]:
    box(x,10.95,4.4,.82,f"{eid}   ·   K = {kval}",f"输入：{history}",ts=14)
    arrow([(x+2.2,10.95),(x+2.2,10.42)])
box(.9,9.65,14.2,.75,"历史 FC： [N, K, 4005]    ·    AAL90 / Fisher-z 边    ·    三组最后一窗均为 FC₁₀",
    color=GRAY,fill="#F5F7FA",ts=12.5)

arrow([(4.55,9.65),(4.55,9.06)],BLUE)
arrow([(12.7,9.65),(12.7,9.06)],ORANGE)
box(1.05,8.1,7,.95,"逐窗共享 A0003 编码器  ·  冻结 / eval",
    "4005 → 1024 → 512 → 256     输出 [N, K, 256]",GRAY,"#F5F7FA")
arrow([(4.55,8.1),(4.55,7.51)],BLUE)
box(1.05,6.55,7,.95,"历史 GRU  ·  单层 / 单向 / hidden = 256",
    "按时间顺序读取 K 个编码；取最后 hidden：[N, 256]")
arrow([(4.55,6.55),(4.55,6.01)],BLUE)
box(1.05,5.25,7,.75,"c = LayerNorm(hidden_last)；未来输入 = qₕ + c",
    "214 个可学习时距 query     →     [N, 214, 256]",ts=12.2,ds=10.7)
arrow([(4.55,5.25),(4.55,4.71)],BLUE)
box(1.05,3.55,7,1.15,"未来 Transformer  ·  self-attention",
    "4 层 / 8 头 / FFN 1024 / GELU / dropout 0.1\npre-norm + 末端 LayerNorm；输出 [N, 214, 256]",ts=13,ds=10.8)
arrow([(4.55,3.55),(4.55,2.91)],BLUE)
box(1.05,2.05,7,.85,"共享线性残差头：Linear(256, 4005)",
    "预测残差 R̂：[N, 214, 4005]",ts=12.8)

box(10.3,8.1,4.8,.95,"取最后观测窗 FC₁₀",
    "三组使用相同锚点：[N, 4005]",ORANGE,"#FFF6E9",ts=12.7,ds=10.6)
box(10.3,6.3,4.8,1.15,"训练集拟合的共享工件  ·  冻结",
    "首窗锚点模板 g₁₀、未来模板 g₁₀₊ₕ\n每个时距一个 αₕ ∈ [0, 1]",GRAY,"#F5F7FA",ts=11.8,ds=10.7)
arrow([(12.7,6.3),(12.7,5.06)],ORANGE)
arrow([(15.1,8.575),(15.5,8.575),(15.5,4.30),(15.1,4.30)],ORANGE)
box(10.3,3.55,4.8,1.5,"解析持续性基线 B  ·  冻结",
    "Bₕ = g₁₀₊ₕ + αₕ (FC₁₀ − g₁₀)\n[N, 214, 4005]\n只由训练集工件与最后观测窗计算",ORANGE,"#FFF6E9",ts=12.4,ds=10.5)
text(12.7,5.57,"三组共用同一工件与 SHA256",10,GRAY)

arrow([(8.05,2.475),(8.98,2.475)],BLUE)
arrow([(12.7,3.55),(12.7,2.475),(9.62,2.475)],ORANGE)
ax.add_patch(Circle((9.3,2.475),.29,facecolor="white",edgecolor=GREEN,linewidth=1.7))
text(9.3,2.475,"+",22,GREEN)
arrow([(9.3,2.185),(9.3,1.46)],GREEN)
box(5.8,.55,7,.9,"最终预测 Ŷ = B + R̂",
    "第 11–224 窗 FC：[N, 214, 4005]",GREEN,"#ECF8F1",ts=15,ds=11)
box(1.05,.55,4.15,.9,"纯 MSE 训练",
    "MSE(R̂, Y − B) = MSE(Ŷ, Y)",GRAY,"#F5F7FA",ts=12.3,ds=10.2)
text(8,.17,"窗口编号从 1 开始；N 为 batch size。214 窗并行输出；不输入 SC，不使用 cross-attention，AE 重建解码器不参与输出。",10,GRAY)

OUT.mkdir(parents=True,exist_ok=True)
for suffix in ("png","svg"):
    path=OUT/f"E0045_E0046_E0047_architecture.{suffix}"
    fig.savefig(path,dpi=190,facecolor="white")
    print(path)
plt.close(fig)
