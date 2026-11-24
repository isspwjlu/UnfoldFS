"""论文图: 平面波反折叠的费米面 —— **旧 paper 的版式**(深蓝底 + 蓝线/权重色)。

版式与 ``paper/figs/fig4a_TaS2_fs.png`` 一致:
  * 三个面板: (a) 原胞直算 = **纯蓝线**; (b)(c) 超胞 = **按谱权重着色**(turbo);
  * 深蓝底; BZ 边界是暗色虚线六边形; 右侧一条竖直色标 $w/w_{\max}$。

⚠ 为什么线按**权重**着色而不是按 A(k,E_F):
  A(k,E_F) 在网格点之间会被高斯尾巴压低, 沿费米面的变化是**采样效应**
  (网格间距 vs 带的斜率), 不是物理量 —— 原胞直算的谱权重处处为 1, 它的
  费米面必须是**一个颜色**。谱权重才是能着色的那个量。

两行: 上排 1T-TaS2, 下排 1H-NbSe2; 每行三个面板(直算 / 3×1×1 / √3×√3)。
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

BG = (0.02, 0.03, 0.35)            # 深蓝底(旧 paper 同款)
CMAP = "turbo"
COLOR_DIRECT = "#1f3fbf"            # 原胞直算: 蓝线(旧 paper 同款)
COLOR_BZ = "#55659f"                # BZ 边界: 暗色虚线(深底上不刺眼)

# 三个列的标题(en = 原样, zh = 中文)。体系名(1T-TaS2 / 1H-NbSe2)是化学式,
# 不翻译。
LBL = {
    "ttl_prim": {"en": "primitive cell (direct calculation)",
                 "zh": "原胞（直接计算）"},
    "ttl_sc":   {"en": r"$3\times1\times1$ supercell",
                 "zh": r"$3\times1\times1$超胞"},
    "ttl_sqrt": {"en": r"$\sqrt{3}\times\sqrt{3}$ supercell",
                 "zh": r"$\sqrt{3}\times\sqrt{3}$超胞"},
}


class _A:
    no_sym = False
    blur = 1.0
    sigma_cells = 1.5
    mask = 0.25
    wmin = 0.02
    method = "peak"
    kz = None


def fs_of(npz, b, b2d, ops):
    segs, wts, _ = P.panel_data(npz, b, b2d, b[0][:2], b[1][:2], ops, _A())
    return np.array(segs), np.asarray(wts)


def build(rows, out, png=None, height=5.0, lang="en"):
    """rows: [(体系名, POSCAR, [直算 npz, 3x1x1 npz, sqrt3 npz]), ...]"""
    S.apply_prb_style(lang=lang)
    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(len(rows), 3, wspace=0.28, hspace=0.30,
                          left=0.055, right=0.895, top=0.92, bottom=0.115)
    # 色标范围: 从**数据的实际下限**起, 而不是从 0 起。
    # 严格权重下费米面上的权重全挤在 0.99 附近(直算恒为 1), 若色标从 0 起,
    # 整条曲线都是同一个颜色(旧 paper 的图不是这样 —— 那里色标虽然标 0→1,
    # 但轨道投影的权重在费米面上本来就跨 0.4~1.0, 于是颜色铺满了色标)。
    # ⇒ 这里按实际下限拉伸, 颜色的铺开程度与旧 paper 一致, 而色标的刻度
    #   标的是**真实的权重值**, 读者能看到权重的绝对大小。

    wmins = []
    for r, (tag, primfile, files) in enumerate(rows):
        # 这一行所有超胞面板的权重下限
        wmin = 1.0
        for npz in files[1:]:
            prim0 = read_poscar(primfile)
            b0 = np.asarray(reciprocal(prim0.lattice))
            ops0, _ = P.get_ops(prim0)
            _, w = fs_of(npz, b0, b0[:2, :2], ops0)
            if w.size:
                wmin = min(wmin, float(np.percentile(w, 0.5)))
        norm = Normalize(wmin, 1.0)
        wmins.append(wmin)
        prim = read_poscar(primfile)
        b = np.asarray(reciprocal(prim.lattice))
        b2d = b[:2, :2]
        hexv = P.bz_hexagon(b)
        ops, _ = P.get_ops(prim)
        hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
        hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
        mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
        lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))
        for c, npz in enumerate(files):
            segs, wts = fs_of(npz, b, b2d, ops)
            ax = fig.add_subplot(gs[r, c])
            if len(segs):
                if c == 0:                      # 原胞直算: 纯蓝线
                    ax.add_collection(LineCollection(segs, colors=COLOR_DIRECT,
                                                     linewidths=2.4))
                else:                           # 超胞: 按谱权重着色
                    # 沿曲线链做**轻微**平滑(11 段)去掉逐段取色的锯齿,
                    # 但保留权重的真实结构: 平面波路线的权重是**分块**的
                    # —— 某个 k 点的态属于哪个候选动量会跳变, 于是权重沿费
                    # 米面不是连续函数。强平滑会把这些真实的块抹成彩色的
                    # 斑块, 那才是失真。
                    svals = F.smooth_weights(segs, wts, window=11)
                    lc = LineCollection(segs, cmap=CMAP, norm=norm,
                                        linewidths=2.4)
                    lc.set_array(svals)
                    ax.add_collection(lc)
            ax.set_aspect("equal")
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_facecolor(BG)
            ax.add_patch(Polygon(hexv, closed=True, fill=False,
                                 edgecolor=COLOR_BZ, lw=0.9, ls="--",
                                 alpha=0.7))
            ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
            if c == 0:
                ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
            ttl = {0: LBL["ttl_prim"][lang],
                   1: LBL["ttl_sc"][lang],
                   2: LBL["ttl_sqrt"][lang]}[c]
            # 标题里**不写**体系名(否则相邻面板的标题会撞在一起);
            # 体系名作为行标签画在每行第一个面板的左上角(旧 paper 每张图
            # 一个体系, 所以标题里没有它)。
            ax.set_title(f"({S.letter(r * 3 + c)}) {ttl}", fontsize=S.FS_TITLE)
            if c == 0:
                ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha="left",
                        va="top", fontsize=S.FS_TITLE, color="w")
            ax.tick_params(labelsize=S.FS_TICK)
            print(f"  {tag} ({ttl}): {len(segs)} 段, 权重中位 "
                  f"{np.median(wts) if wts.size else 0:.4f}")

    # 竖直色标(自己画: fig.colorbar 在 bbox_inches="tight" 下会压扁色条)
    cax = fig.add_axes([0.915, 0.30, 0.012, 0.40])
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=CMAP, origin="lower",
               extent=(0.0, 1.0, 0.0, 1.0), interpolation="nearest")
    cax.set_xticks([])
    wmin_all = min(wmins)
    cax.set_yticks([wmin_all, 1.0])
    cax.set_yticklabels(["%.3f" % wmin_all, "1"])
    cax.tick_params(labelsize=S.FS_TICK, length=2, pad=2)
    # 刻度标签放在色标右侧: 放在左侧会压住色条本身(中英文版都有此问题)
    if True:
        # 英文版里 "0.878" 这个刻度标签压在色条上(色条右边没有空白)⇒
        # 中文版把刻度标签挪到色条**右侧**(画布右边有空白)。
        cax.yaxis.set_ticks_position("right")
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.set_title(r"$w$", fontsize=S.FS_ANNOT, pad=6)

    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tas2-prim", required=True)
    ap.add_argument("--tas2-npz", required=True,
                    help="逗号分隔: 直算, 3x1x1, sqrt3")
    ap.add_argument("--nbse2-prim", required=True)
    ap.add_argument("--nbse2-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="图内固定文字的语言")
    args = ap.parse_args()
    build([(r"1T-TaS$_2$", args.tas2_prim,
            [x.strip() for x in args.tas2_npz.split(",")]),
           (r"1H-NbSe$_2$", args.nbse2_prim,
            [x.strip() for x in args.nbse2_npz.split(",")])],
          args.out, png=args.png or None, height=args.height,
          lang=args.lang)


if __name__ == "__main__":
    main()
