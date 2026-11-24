"""论文图 5: 反折叠费米面的**验证协议**。

核心硬判据: 反折叠得到的费米面必须与原胞直接计算的费米面**重合**
(反折叠只重排权重, 不该移动费米面)。

(a)(b) 原胞直算的 FS (蓝, 粗) 与反折叠 FS (红, 细) 叠在一起 —— 完全重合;
(c) 逐点的"点到线段"距离 (以网格间距为单位);
(d) 三种情况下费米面上谱权重的分布。

⚠ 数字(中位数/最大值)**不画在图里** —— 它们在正文里; 画在图里必然和
数据打架(曾经压住整片散点)。

数据: 1T-TaS2, 96×96×1 原胞 + 3×1×1 / √3×√3 反折叠。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P

from unfoldfs.geometry import read_poscar, reciprocal

COLOR_DIRECT = S.PALETTE["blue_main"]
COLOR_UNFOLD = S.PALETTE["red_strong"]
COLOR_BZ = S.PALETTE["bz_edge"]


class _A:
    no_sym = False
    blur = 1.0
    sigma_cells = 1.5
    mask = 0.25
    wmin = 0.02
    method = "peak"
    kz = None


def fs(npz, prim, b, b2d, ops):
    segs, wts, _ = P.panel_data(npz, b, b2d, b[0][:2], b[1][:2], ops, _A())
    return np.array(segs), wts


def curve_distance(curve_a, curve_b):
    """双向的"曲线到曲线"距离(以网格间距为单位)。"""
    va = curve_a.reshape(-1, 2)
    vb = curve_b.reshape(-1, 2)
    d_ab = cKDTree(vb).query(va)[0]
    d_ba = cKDTree(va).query(vb)[0]
    return np.concatenate([d_ab, d_ba]), np.concatenate([va[:, 0], vb[:, 0]])


def build(poscar, prim_npz, sc3_npz, scs_npz, out, png=None, height=2.6):
    S.apply_prb_style()
    prim = read_poscar(poscar)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    hexv = P.bz_hexagon(b)
    ops, _ = P.get_ops(prim)

    s_prim, w_prim = fs(prim_npz, prim, b, b2d, ops)
    s_3, w_3 = fs(sc3_npz, prim, b, b2d, ops)
    s_s, w_s = fs(scs_npz, prim, b, b2d, ops)
    n1 = P.load(prim_npz)["n1"]
    h = float(np.linalg.norm(b2d[0])) / n1

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, 1.05], wspace=0.42)

    for k, (s_un, ttl) in enumerate([(s_3, r"3$\times$1$\times$1"),
                                     (s_s, r"$\sqrt{3}\times\sqrt{3}$")]):
        ax = fig.add_subplot(gs[0, k])
        ax.add_collection(LineCollection(s_prim, colors=COLOR_DIRECT,
                                         linewidths=3.2))
        ax.add_collection(LineCollection(s_un, colors=COLOR_UNFOLD,
                                         linewidths=1.2))
        ax.set_aspect("equal")
        ax.set_xlim(hexv[:, 0].min(), hexv[:, 0].max())
        ax.set_ylim(hexv[:, 1].min(), hexv[:, 1].max())
        ax.add_patch(Polygon(hexv, closed=True, fill=False,
                             edgecolor=COLOR_BZ, lw=0.9, ls="--"))
        ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
        if k == 0:
            ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
        ax.set_title(f"({S.letter(k)}) {ttl}", fontsize=S.FS_TITLE)

    # (c) 距离分布
    ax = fig.add_subplot(gs[0, 2])
    d3, x3 = curve_distance(s_3, s_prim)
    d3 = d3 / h
    ds, xs = curve_distance(s_s, s_prim)
    ds = ds / h
    ax.scatter(x3, np.maximum(d3, 1e-7), s=2.0, color=COLOR_DIRECT,
               label=r"3$\times$1$\times$1", alpha=0.5, rasterized=True)
    ax.scatter(xs, np.maximum(ds, 1e-7), s=2.0, color=COLOR_UNFOLD,
               label=r"$\sqrt{3}\times\sqrt{3}$", alpha=0.5, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
    ax.set_ylabel("distance (grid cells)")
    ax.legend(fontsize=S.FS_ANNOT, loc="upper right")
    ax.set_title(f"({S.letter(2)}) point-to-curve distance", fontsize=S.FS_TITLE)

    # (d) 权重分布
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(w_prim, bins=bins, color=COLOR_DIRECT, alpha=0.55,
            label="direct", density=True)
    ax.hist(w_3, bins=bins, color=COLOR_UNFOLD, alpha=0.55,
            label=r"3$\times$1$\times$1", density=True)
    ax.hist(w_s, bins=bins, color=S.PALETTE["green_3"], alpha=0.55,
            label=r"$\sqrt{3}\times\sqrt{3}$", density=True)
    ax.set_xlabel(r"spectral weight $w$ on the FS")
    ax.set_ylabel("probability density")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=S.FS_ANNOT, loc="upper right")
    ax.set_title(f"({S.letter(3)}) spectral weight on the FS",
                 fontsize=S.FS_TITLE)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.24,
                        wspace=0.46)
    # 图例放在画布**底部**(不在坐标区里) ⇒ 绝不遮费米面
    fig.legend(handles=[
        plt.Line2D([], [], color=COLOR_DIRECT, lw=3.2,
                   label="direct calculation"),
        plt.Line2D([], [], color=COLOR_UNFOLD, lw=1.2, label="unfolded"),
    ], loc="lower left", bbox_to_anchor=(0.055, -0.005), ncol=2,
        fontsize=S.FS_ANNOT, frameon=False, handlelength=1.8,
        columnspacing=1.6)
    S.save(fig, out, png_preview=png)
    plt.close(fig)
    print(f"  3×1×1 : 距离中位 {np.median(d3):.5f} 格, 最大 {d3.max():.3f} 格")
    print(f"  √3×√3 : 距离中位 {np.median(ds):.5f} 格, 最大 {ds.max():.3f} 格")
    print(f"  权重中位: direct {np.median(w_prim):.4f}, "
          f"3×1×1 {np.median(w_3):.4f}, √3×√3 {np.median(w_s):.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--sc3-npz", required=True)
    ap.add_argument("--scs-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=2.6)
    args = ap.parse_args()
    build(args.prim, args.prim_npz, args.sc3_npz, args.scs_npz, args.out,
          png=args.png or None, height=args.height)


if __name__ == "__main__":
    main()
