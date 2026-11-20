"""论文图 4: 反折叠费米面(PRB 版式)。

画的是**真费米面**: E_b(k) 恰过 E_F 的那条线(脊顶), 不是 A(k,E_F) 的等值线。
见 ``scripts/figures/README.md`` 的画法约定。

版式:
  * 7.0 in 宽 = PRB 全栏; 字号即印刷磅值;
  * 三个面板 = (a) 原胞直算(蓝线) / (b) 3×1×1 / (c) √3×√3(颜色 = 过 E_F
    那条带的谱权重, 按原胞完整点群轨道平均);
  * 右侧一条竖色标, Low/High 在色标两端, 标签在色标上方;
  * 图内无总标题、无长注释(caption 承担)。

用法:
  python paper_fig_fs.py --prim PRIMCELL.vasp --method peak \
      --npz a.npz,b.npz,c.npz --titles "primitive cell (direct calculation),3×1×1 supercell,√3×√3 supercell" \
      --blue-panels 0 --out fig4a_TaS2_fs.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P

from unfoldfs.geometry import read_poscar, reciprocal

CMAP = "turbo"
COLOR_DIRECT = "#1f3fbf"          # 原胞直算: 蓝线
COLOR_BZ = S.PALETTE["neutral"]    # 布里渊区边界: 中性灰虚线


def draw_panel(ax, segs, wts, blue, title, hexv, lims, lw):
    if blue:
        lc = LineCollection(segs, colors=COLOR_DIRECT, linewidths=lw)
    else:
        lc = LineCollection(segs, cmap=CMAP, linewidths=lw,
                            norm=Normalize(0.0, float(wts.max())
                                           if wts.size else 1.0))
        lc.set_array(wts)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_facecolor((0.02, 0.03, 0.35))
    ax.add_patch(Polygon(hexv, closed=True, fill=False,
                          edgecolor=COLOR_BZ, lw=0.9, ls="--", alpha=0.85))
    ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
    ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
    ax.set_title(title, fontsize=S.FS_TITLE)
    ax.tick_params(labelsize=S.FS_TICK)


def add_colorbar(fig, rect):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([0.0, 1.0])
    cb.ax.tick_params(labelsize=S.FS_TICK, length=2, pad=2)
    cb.outline.set_visible(False)
    cax.set_title(r"$w/w_{\max}$", fontsize=S.FS_ANNOT, pad=6)
    return cax


def build(prim_poscar, npz_paths, titles, out, png=None, method="peak",
          blue_panels=(0,), lw=1.8, height=2.75):
    S.apply_prb_style()
    prim = read_poscar(prim_poscar)
    b_prim = np.asarray(reciprocal(prim.lattice))
    b1, b2 = b_prim[0][:2], b_prim[1][:2]
    b2d = b_prim[:2, :2]
    hexv = P.bz_hexagon(b_prim)
    hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
    hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
    mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
    lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))
    ops, _ = P.get_ops(prim)

    A = SimpleNamespace(no_sym=False, blur=1.0, sigma_cells=1.5, mask=0.25,
                        wmin=0.02, method=method, kz=None)

    n = len(npz_paths)
    fig, axes = plt.subplots(1, n, figsize=(S.FULL_WIDTH, height), squeeze=False)
    axes = axes[0]
    for i, (path, title) in enumerate(zip(npz_paths, titles)):
        segs, wts, kzinfo = P.panel_data(path, b_prim, b2d, b1, b2, ops, A)
        draw_panel(axes[i], segs, wts, i in blue_panels,
                   f"({S.letter(i)}) {title}", hexv, lims, lw)
        print(f"{title}: {len(segs)} 段; 权重 中位 "
              f"{np.median(wts) if wts.size else 0:.4f} max "
              f"{wts.max() if wts.size else 0:.4f}")
    fig.subplots_adjust(left=0.055, right=0.915, top=0.86, bottom=0.15,
                        wspace=0.30)
    add_colorbar(fig, [0.930, 0.21, 0.013, 0.48])
    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True, help="逗号分隔的 npz")
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--method", choices=["peak", "band"], default="peak")
    ap.add_argument("--blue-panels", default="0")
    ap.add_argument("--lw", type=float, default=1.8)
    ap.add_argument("--height", type=float, default=2.75)
    args = ap.parse_args()

    build(args.prim, [p.strip() for p in args.npz.split(",")],
          args.titles.split(","), args.out, png=args.png or None,
          method=args.method,
          blue_panels={int(x) for x in args.blue_panels.split(",")
                       if x.strip() != ""},
          lw=args.lw, height=args.height)


if __name__ == "__main__":
    main()
