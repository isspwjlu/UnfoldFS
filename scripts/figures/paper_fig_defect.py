"""论文图: 体相 1T-TaS2 缺陷超胞的反折叠费米面(PRB 版式, 平面波路线)。

三个面板:
  (a) 原胞直算的费米面 —— 自己的 E_F = 4.405 eV
  (b) **完美 3×3×1** 超胞反折叠的费米面 —— 同一个 E_F, 与 (a) 重合
  (c) **缺陷 Ta9S16** 超胞反折叠的费米面 —— 自己的 E_F = 3.799 eV

画法: 深蓝底 + 费米面线, 线**按谱权重着色**(与旧 paper 同款)。不用背景场
A(k,E_F): 这个算例的网格只有 24×24(间距 0.090 与费米面尺度相当),
插值出来的谱脊与插值出来的费米面线**位置对不上**(看着像画错了);
验证图那些 96×96 的算例没有这个问题, 那边保留背景场。

⚠ 面板 (c) 与 (a)(b) 的费米能级不同(去掉 2 个 S 少了 12 个电子 ⇒ E_F 低
  0.61 eV, 且缺陷体系的 E_F 落在带隙内 ⇒ 它本来就没有费米面); 拿两个
  不同 E_F 的费米面直接比是跨能量比较。

用法:
  python paper_fig_defect.py --prim PRIMCELL.vasp \
      --prim-npz prim.npz --perfect-npz perfect.npz --defect-npz defect.npz \
      --kz 0 --out fig_defect.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

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
import fs_field as F

from unfoldfs.geometry import read_poscar, reciprocal

BG = (0.02, 0.03, 0.35)            # 深蓝底(与旧 paper 同款)
CMAP = "turbo"


class _A:
    no_sym = False
    blur = 1.0
    sigma_cells = 1.5
    mask = 0.25
    wmin = 0.02
    method = "peak"
    kz = None
    nearest = True
    # 24×24 的网格太粗(间距 0.090 与费米面尺度相当), 直接提等值线得到的是
    # 折线碎片 ⇒ 用协变插值重建到细网格(与 plot_fs.panel_data 同一口径)。
    refine = 96


def fs_of(npz, b, b2d, ops, kz):
    segs, wts, info = P.panel_data(npz, b, b2d, b[0][:2], b[1][:2], ops,
                                    _A())
    segs = np.array(segs) if len(segs) else np.zeros((0, 2, 2))
    return segs, wts


def build(poscar, prim_npz, perfect_npz, defect_npz, out, png=None,
          height=2.55, kz=None, smooth_len=0.035):
    S.apply_prb_style()
    prim = read_poscar(poscar)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    hexv = P.bz_hexagon(b)
    ops, _ = P.get_ops(prim)
    hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
    hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
    mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
    lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))

    panels = [(prim_npz, "primitive cell (direct)"),
              (perfect_npz, r"perfect $3\times3\times1$"),
              (defect_npz, r"vacancy supercell Ta$_9$S$_{16}$")]
    segs_all, wts_all = [], []
    for npz, _ in panels:
        segs, wts = fs_of(npz, b, b2d, ops, kz)
        # 画图层面的平滑: 24×48 的网格太粗, 插值重建出来的曲线带高频毛刺;
        # 沿弧长做一次高斯平滑把毛刺抹掉(不动数据: 谱权重与重合距离由提取
        # 本身给出, 与这里无关)。
        segs = F.smooth_chains(segs, sigma_len=smooth_len)
        segs_all.append(segs)
        wts_all.append(wts)

    norm = Normalize(0.0, 1.0)      # 三个面板共用

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.055, right=0.865,
                          top=0.85, bottom=0.17)

    for k, ((npz, ttl), segs, wts) in enumerate(zip(panels, segs_all, wts_all)):
        ax = fig.add_subplot(gs[0, k])
        if len(segs):
            lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidths=1.8)
            lc.set_array(np.asarray(wts, dtype=float))
            ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_facecolor(BG)
        ax.add_patch(Polygon(hexv, closed=True, fill=False,
                             edgecolor="#55659f", lw=0.8, ls="--", alpha=0.6))
        ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
        if k == 0:
            ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
        ax.set_title(f"({S.letter(k)}) {ttl}", fontsize=S.FS_TITLE)
        ax.tick_params(labelsize=S.FS_TICK)
        d = P.load(npz, kz)
        print(f"  {ttl}: E_F = {d['e_f']:.4f} eV, FS {len(segs)} 段"
              + (f", 权重中位 {np.median(wts):.4f}" if wts.size else ""))

    # 竖直色标(自己画: fig.colorbar 在 bbox_inches="tight" 下会把色条压扁)
    cax = fig.add_axes([0.915, 0.22, 0.012, 0.50])
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=CMAP, origin="lower",
               extent=(0.0, 1.0, 0.0, 1.0), interpolation="nearest")
    cax.set_xticks([])
    cax.set_yticks([0.0, 0.5, 1.0])
    cax.set_yticklabels(["0", "0.5", "1"])
    cax.tick_params(labelsize=S.FS_TICK, length=2, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.set_ylabel(r"weight $w$", fontsize=S.FS_ANNOT, rotation=270,
                   labelpad=14, va="bottom")

    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--perfect-npz", required=True)
    ap.add_argument("--defect-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=2.55)
    ap.add_argument("--kz", type=float, default=None)
    ap.add_argument("--smooth-len", type=float, default=0.035,
                    help="画图层面的弧长高斯平滑半宽 (2π/Å)")
    args = ap.parse_args()
    build(args.prim, args.prim_npz, args.perfect_npz, args.defect_npz,
          args.out, png=args.png or None, height=args.height, kz=args.kz,
          smooth_len=args.smooth_len)


if __name__ == "__main__":
    main()
