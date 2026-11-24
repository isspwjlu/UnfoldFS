"""论文图: 费米速度画在**原版费米面**与**反折叠费米面**上(整个布里渊区)。

**版式与论文其它 FS 图一致**(见 paper_fig_fs.py): 深蓝底 + 中性灰虚线 BZ 轮廓
+ turbo 色图 + 竖直色标; 画满整个第一布里渊区。

三面板(共用同一色标 ⇒ 颜色可直接比):
  (a) **原版**(原胞直算)费米面, 按 v_F 着色 —— 参照;
  (b) **反折叠**(3×1×1 超胞)费米面, 按 v_F 着色 —— 图案应与 (a) 一致
      (验证协议在 v_F 上的版本: 展开只重排权重, 不该动费米面);
  (c) 反折叠的**等值面**(A(k,E_F) 的两壁), 按 v_F 着色 —— 图案被打乱。

数据与口径见 paper/fact_sheet.md §2.13; 逐点数值与 paper_fig_velocity.py 同源。
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

from paper_fig_velocity import curve_samples

CMAP = "turbo"                     # 与 paper_fig_fs.py 同一张色图
BG = (0.02, 0.03, 0.35)           # 与 paper_fig_fs.py 同一底色
COLOR_BZ = S.PALETTE["bz_edge"]     # BZ 边界: 暗蓝(浅灰在深底上像"一圈白线")

# 图内固定文字(en = 原样, zh = 中文)。轴标签与色标标签全是公式/单位, 不译。
LBL = {
    "a_title": {"en": r"(a) direct calculation", "zh": "(a) 直接计算"},
    "b_title": {"en": r"(b) unfolded, $3\times1\times1$",
                "zh": r"(b) 展开，$3\times1\times1$"},
    "c_title": {"en": r"(c) unfolded isosurface", "zh": "(c) 展开后的等值面"},
}


def pieces(curves, v_all):
    """把逐点 v 按曲线切回 (曲线, 该曲线的 v)。"""
    out, i = [], 0
    for c in curves:
        n = len(c)
        out.append((np.asarray(c, dtype=float), np.asarray(v_all[i:i + n])))
        i += n
    return out


def bz_hexagon(b_prim):
    """第一布里渊区的六边形 —— 直接用论文其它 FS 图那一份(口径必须一致)。"""
    import plot_fs as P
    return P.bz_hexagon(b_prim)


def draw_panel(ax, pp, hexv, lims, norm, lw=2.0):
    """按 v_F 着色画费米面(深蓝底 + 灰虚线 BZ)。"""
    for curve, vals in pp:
        if len(curve) < 2 or not np.isfinite(vals).any():
            continue
        segs = np.stack([curve[:-1], curve[1:]], axis=1)
        mid = 0.5 * (vals[:-1] + vals[1:])
        lc = LineCollection(segs, cmap=CMAP, linewidths=lw, norm=norm)
        lc.set_array(mid)
        ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_facecolor(BG)
    ax.add_patch(Polygon(hexv, closed=True, fill=False,
                          edgecolor=COLOR_BZ, lw=0.9, ls="--", alpha=0.85))
    ax.set_xlabel(r"$k_1$ (2$\pi$/Å)")
    ax.set_ylabel(r"$k_2$ (2$\pi$/Å)")
    ax.tick_params(labelsize=S.FS_TICK)


def build(npz_prim, npz_unf, out, png=None, height=2.4, lang="en"):
    S.apply_prb_style(lang=lang)

    d_p = curve_samples(npz_prim)
    d_u = curve_samples(npz_unf)
    # 倒格矢: npz 是 kz=0 的二维切片 ⇒ 只有面内部分可用(z 分量置零;
    # BZ 六边形只用面内分量, 与论文其它 FS 图一致)
    b_prim = np.zeros((3, 3))
    b_prim[0, :2] = d_p["b2d"][0]
    b_prim[1, :2] = d_p["b2d"][1]
    hexv = bz_hexagon(b_prim)
    lim = float(np.max(np.abs(hexv))) * 1.08
    lims = ((-lim, lim), (-lim, lim))

    pc_p = pieces(d_p["crest_curves"], d_p["crest"][0])
    pf_p = pieces(d_p["flank_curves"], d_p["flank"][0])
    pc_u = pieces(d_u["crest_curves"], d_u["crest"][0])
    pf_u = pieces(d_u["flank_curves"], d_u["flank"][0])

    allv = np.concatenate([v for _, v in pc_p] + [v for _, v in pc_u]
                          + [v for _, v in pf_u])
    allv = allv[np.isfinite(allv)]
    clim = (float(np.percentile(allv, 2)), float(np.percentile(allv, 98)))
    norm = Normalize(*clim)
    print(f"  color range {clim[0] / 1e5:.2f} ~ {clim[1] / 1e5:.2f} x 1e5 m/s")

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.34)

    panels = [
        (pc_p, LBL["a_title"][lang]),
        (pc_u, LBL["b_title"][lang]),
        (pf_u, LBL["c_title"][lang]),
    ]
    for k, (pp, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        draw_panel(ax, pp, hexv, lims, norm)
        if k == 2:
            # 把**脊顶**叠上去(浅色细线)⇒ 一眼看出两壁是**夹着**真费米面的
            for curve, vals in pc_u:
                if len(curve) >= 2:
                    ax.plot(curve[:, 0], curve[:, 1], lw=0.7,
                            color="#E8E8E8", alpha=0.85, zorder=5)
        ax.set_title(title, fontsize=S.FS_TITLE)

    # 色标: 竖直; **刻度在左、竖排标签在右** —— 横向的标签会压到 (c) 面板
    cax = fig.add_axes([0.930, 0.24, 0.010, 0.46])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=S.FS_TICK, length=2, pad=2)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e5:.1f}"))
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(r"$v_F$ ($10^{5}$ m/s)", rotation=90, labelpad=10,
                  fontsize=S.FS_ANNOT)
    cb.ax.yaxis.set_label_position("right")

    fig.subplots_adjust(left=0.055, right=0.895, top=0.88, bottom=0.15,
                        wspace=0.34)
    S.save(fig, out, png_preview=png)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz-prim", required=True, help="primitive-cell A(k,E).npz")
    p.add_argument("--npz-unf", required=True, help="unfolded A(k,E).npz (supercell)")
    p.add_argument("--out", required=True)
    p.add_argument("--png", default="")
    p.add_argument("--height", type=float, default=2.4)
    p.add_argument("--lang", choices=("en", "zh"), default="en",
                   help="图内固定文字的语言")
    a = p.parse_args()
    build(a.npz_prim, a.npz_unf, a.out, png=a.png or None, height=a.height,
          lang=a.lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
