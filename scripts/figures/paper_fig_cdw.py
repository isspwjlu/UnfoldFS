"""论文图 7: 体相 1T-TaS2 **CDW 态**的反折叠能带(有隙)。

CDW 态把原胞的平移对称性破坏掉, 于是:
  * 它**不是**原胞的整数倍超胞(只是相容), 所以没有"原胞直算"可以对照;
  * 反折叠的原子对应不再精确(原子被挪开了)⇒ 用**未畸变**的同胞结构做对应。

两个面板, 同一份 VASP 计算、同一套 k 路径、同一个 E_F:

(a) **assign 模式**: 每条超胞态只归到权重最大的那个候选原胞动量
    ⇒ 这就是通常说的"展开能带", 带是干净的;
(b) **full 模式**: 每条态在**全部**候选原胞动量上的权重都画出来
    ⇒ 展开**谱**。CDW 态不是原胞平移的本征态, 它的权重摊在 13 个候选动量上
    (单-G 主导度 0.49, 金属态是 0.74)⇒ 谱被抹开。

CDW 态的 13 个副本是**简并**的 ⇒ "每条态只归到一个原胞动量"(assign)的画法
对这个态没有意义(会碎成片段), 所以本图**只画 full 模式的一幅**:
每个态在全部候选原胞动量上的权重, 呈横向条纹。

用法:
  python paper_fig_cdw.py --ef 5.370113 \
      --assign cdw39_unfolded_assign.dat --full cdw39_unfolded.dat \
      --labels "$\Gamma$:0,M:...,K:..." --out fig6_cdw.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
from plot_band import read_dat, heatmap_panel

CMAP = "turbo"


def kpath_marks(ax, labels):
    """k 路径竖线 + 高对称点标签(标签在坐标区外, 一律黑色)。"""
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    for name, d in labels:
        ax.axvline(d, color="w", linewidth=0.6, alpha=0.5, zorder=2)
        if d - x0 < 0.03 * span:
            ha = "left"
        elif x1 - d < 0.03 * span:
            ha = "right"
        else:
            ha = "center"
        # 高对称点标记**贴着下边框**(不要再留一大截空白)
        ax.text(d, -0.022, name, transform=ax.get_xaxis_transform(),
                ha=ha, va="top", fontsize=S.FS_TICK, color="k")


def build(ef, assign_dat, full_dat, labels, out, png=None, e_min=-1.0,
          e_max=1.0, sigma=0.05, height=3.0, width=None):
    S.apply_prb_style()
    df, efull, wf = read_dat(full_dat)
    width = width or S.FULL_WIDTH

    fig = plt.figure(figsize=(width, height))
    # 只画一幅 ⇒ gridspec 必须是**一行**: 之前分了 2 行却只用第 0 行,
    # 空白的那一行白吃掉一半画布(导出的 PDF 只有内容那一半高)。
    gs = fig.add_gridspec(1, 1)

    # CDW 态的 13 个副本是**简并**的, 所以"每条态只归到一个原胞动量"的画法
    # (assign) 会碎成片段, 没有意义 ⇒ 只画 full 模式(每个态在所有候选动量上
    # 的权重)。只画一幅: 能量窗取能隙附近, 既有隙又有上下文的带。
    rows = [(df, efull, wf, None)]

    for row, (d, e, w, title) in enumerate(rows):
        ax = fig.add_subplot(gs[row, 0])
        # heatmap_panel 的 lo/hi 是**绝对能量**(画的是 E-E_F) ⇒ 相对窗口加回 E_F
        heatmap_panel(ax, d, e, w, ef, ef + e_min, ef + e_max, sigma,
                      cmap=CMAP, rasterized=True)
        if title:
            ax.set_title(title, fontsize=S.FS_TITLE)
        ax.set_ylabel(r"$E-E_F$  (eV)")
        ax.tick_params(labelsize=S.FS_TICK)
        # k 路径的横坐标**只要高对称点标记**: 不要数字刻度, 也不要 "k-path" 轴标题
        ax.set_xticks([])
        kpath_marks(ax, labels)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.96, bottom=0.075)
    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=float, required=True, help="费米能级(绝对能量, eV)")
    ap.add_argument("--assign", required=True, help="assign 模式的 .dat")
    ap.add_argument("--full", required=True, help="full 模式的 .dat")
    ap.add_argument("--zoom", type=float, default=0.8,
                    help="(b) 面板在 E_F 附近的放大窗半宽 (eV)")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--e-min", type=float, default=-3.0, help="相对 E_F 的下限")
    ap.add_argument("--e-max", type=float, default=1.5, help="相对 E_F 的上限")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--width", type=float, default=0.0,
                    help="画布宽度 (in); 0 = 全栏 7.0 in, 3.4 = 单栏")
    args = ap.parse_args()

    labels = []
    for item in args.labels.split(","):
        if item.strip():
            name, pos = item.split(":")
            labels.append((name.strip(), float(pos)))

    build(args.ef, args.assign, args.full, labels, args.out,
          png=args.png or None, e_min=args.e_min, e_max=args.e_max,
          sigma=args.sigma, height=args.height,
          width=(args.width or None))


if __name__ == "__main__":
    main()
