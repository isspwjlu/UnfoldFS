"""论文图 1: 展开能带(PRB 版式)。

口径(与 ``scripts/figures/README.md`` 一致):
  * y 轴 = E − E_F, E_F = 0;
  * 超胞面板 = ``full`` 模式 + **按画窗归一化的线性色标**;
  * 原胞面板 = **原 band 本身、纯蓝线**(标准答案, 用最朴素的线图画)。

PRB 版式(与 ``prb_style.py`` 一致):
  * 画布 7.0 in 宽 = PRB 全栏 \\linewidth ⇒ 图里写的字号就是印刷磅值;
  * 图内**不写总标题、不写长注释**(caption 承担);
  * 底部一条横向色标(左 Low 右 High), 标签在色标**下面**, 不压任何轴标签。

用法:
  python paper_fig_bands.py --ef -1.501119 \
      --prim-dat tas2_prim.dat --unfold tas2_full_sc.dat,tas2_full_scsqrt3.dat \
      --labels "0:0,1.078:1,1.700:2,2.945:3" \
      --titles "primitive cell (direct calculation),3×1×1 supercell,√3×√3 supercell" \
      --out fig1_TaS2_bands.pdf
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
from plot_band import read_dat, heatmap_panel, lines_panel

# 展开能带的色标: 与论文里"权重从小到大"的语义一致, 但用感知均匀的 turbo
# (比 jet 均匀, 灰度打印也不塌)。
CMAP = "turbo"


def kpath_marks(ax, labels, dark: bool):
    """k 路径竖线 + 高对称点标签。

    标签画在坐标区**外面**(白底)⇒ 文字一律黑色; 热图(深底)上竖线用白线。
    两端的标签改成左/右对齐, 否则会被裁掉。
    """
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    for name, d in labels:
        ax.axvline(d, color="w" if dark else "0.75", linewidth=0.6,
                   alpha=0.55, zorder=2)
        if d - x0 < 0.06 * span:
            ha = "left"
        elif x1 - d < 0.06 * span:
            ha = "right"
        else:
            ha = "center"
        # 高对称点标记**贴着下边框**(不要再留一大截空白)
        ax.text(d, -0.022, name, transform=ax.get_xaxis_transform(),
                ha=ha, va="top", fontsize=S.FS_TICK, color="k")


def add_hbar(fig, rect, label):
    """底部横向色条: 左 Low 右 High, 标签在色条下方(不压轴标签)。"""
    cax = fig.add_axes(rect)
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1),
                                      cmap=plt.get_cmap(CMAP))
    sm.set_array(np.linspace(0, 1, 256))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_visible(False)
    cax.text(-0.010, 0.5, "Low", transform=cax.transAxes, ha="right",
             va="center", fontsize=S.FS_ANNOT)
    cax.text(1.010, 0.5, "High", transform=cax.transAxes, ha="left",
             va="center", fontsize=S.FS_ANNOT)
    cax.set_xlabel(label, fontsize=S.FS_ANNOT, labelpad=3)
    return cax


def build(ef, prim_dat, unfold_dats, titles, labels, out, png=None,
          e_min=-4.2, e_max=1.0, sigma=0.05, height=3.15):
    S.apply_prb_style()
    dp, ep, wp = read_dat(prim_dat)
    # titles[0] 属于原胞面板, 后面的依次属于各超胞面板
    figs = [(t.strip(), read_dat(p.strip()))
            for t, p in zip(titles[1:], unfold_dats)]

    n = len(figs) + 1
    fig, axes = plt.subplots(1, n, figsize=(S.FULL_WIDTH, height), squeeze=False)
    axes = axes[0]

    lines_panel(axes[0], dp, ep, ef, e_min, e_max)
    axes[0].set_title(f"(a) {titles[0]}", fontsize=S.FS_TITLE)
    kpath_marks(axes[0], labels, dark=False)
    # 原胞是画线图 ⇒ 边框**四周都有**(house style 默认只留左/下)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(True)

    for i, (ax, (title, (d, e, w))) in enumerate(zip(axes[1:], figs)):
        # 网格本身是数据 ⇒ 存 PDF 时栅格化(否则一个 pcolormesh 就是几 MB),
        # 坐标轴/文字/色标仍然是矢量。
        heatmap_panel(ax, d, e, w, ef, e_min, e_max, sigma, cmap=CMAP,
                      rasterized=True)
        ax.set_title(f"({S.letter(i + 1)}) {title}", fontsize=S.FS_TITLE)
        kpath_marks(ax, labels, dark=True)

    for i, ax in enumerate(axes):
        ax.set_ylabel(r"$E-E_F$  (eV)")
        ax.tick_params(labelsize=S.FS_TICK)
        # k 路径的横坐标**只要高对称点标记**: 不要数字刻度, 也不要 "k-path" 轴标题
        ax.set_xticks([])

    # 不画底部色标(颜色 = 谱权重, 按画窗归一化; 说明写在图注里)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.13,
                        wspace=0.30)
    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=float, required=True)
    ap.add_argument("--prim-dat", required=True)
    ap.add_argument("--unfold", required=True, help="逗号分隔的 .dat")
    ap.add_argument("--titles", required=True, help="逗号分隔的三个面板标题")
    ap.add_argument("--labels", default="", help="名:距离,逗号分隔")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--e-min", type=float, default=-4.2)
    ap.add_argument("--e-max", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--height", type=float, default=3.15)
    args = ap.parse_args()

    labels = []
    for item in args.labels.split(","):
        if item.strip():
            name, pos = item.split(":")
            labels.append((name.strip(), float(pos)))

    build(args.ef, args.prim_dat, [p.strip() for p in args.unfold.split(",")],
          args.titles.split(","), labels, args.out,
          png=args.png or None, e_min=args.e_min, e_max=args.e_max,
          sigma=args.sigma, height=args.height)


if __name__ == "__main__":
    main()
