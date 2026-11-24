"""论文图 2: 费米面是 A(k,E) 高斯脊的**脊顶**, 不是 A(k,E_F) 等值线的"两壁"。

(a) 沿一条过 Γ 的切线画 A(k,E) 谱图: 费米面处是一条高斯脊;
    A(k,E_F) 的等值水平(0.5·A_max)与脊的**两条交点** = "两壁"(红线);
    脊的中心(E_b = E_F)才是费米面(蓝线)。
(b) k 平面上: A(k,E_F) 的 0.5 A_max 等值线 = 两壁(红)。
(c) 真费米面: E_b(k) = E_F 的轨迹(蓝) —— 一条曲线, 在两壁中间。

数据: 1H-NbSe2 原胞 A(k,E).npz (48x48)。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P

from unfoldfs.geometry import read_poscar, reciprocal

COLOR_WALL = S.PALETTE["red_strong"]    # 两壁(误判的构造)
COLOR_FS = S.PALETTE["blue_main"]       # 真费米面
COLOR_BZ = S.PALETTE["bz_edge"]         # BZ 边界: 暗蓝
BG = (0.02, 0.03, 0.35)                 # 费米面面板的深底(与权重色标配套)

# 图内固定文字(en = 原样, zh = 中文)。Γ 与公式保持不变。
LBL = {
    "cut_x":    {"en": r"distance along the cut  (2$\pi$/Å)",
                 "zh": r"沿切线的距离（2$\pi$/Å）"},
    "a_title":  {"en": r"(a) $A(\mathbf{k},E)$ along a cut through $\Gamma$",
                 "zh": r"(a) 过$\Gamma$切线上的$A(\mathbf{k},E)$"},
    "b_title":  {"en": r"(b) $A(\mathbf{k},E_F)$ contour at $0.5A_{\max}$",
                 "zh": r"(b) $A(\mathbf{k},E_F)$在$0.5A_{\max}$处的等值线"},
    "c_title":  {"en": r"(c) true Fermi surface", "zh": "(c) 真费米面"},
    "leg_flank": {"en": r"$0.5A_{\max}$ crossings (flanks)",
                  "zh": r"$0.5A_{\max}$交点（两壁）"},
    "leg_crest": {"en": r"band crossing $E_F$ (crest)",
                  "zh": r"能带过$E_F$（脊顶）"},
}


def wall_segments(a_ef, fx, fy, b, b2d, level):
    """A(k,E_F) 在 level 处的等值线 -> 折回第一布里渊区的笛卡尔折线。

    做法: 把场在分数坐标上平铺 3×3 再取等值线(否则边界处的等值线会被
    截断), 然后把顶点换算到笛卡尔并折回胞内。
    """
    tile = np.tile(a_ef, (3, 3))
    tx = np.concatenate([fx - 1, fx, fx + 1])
    ty = np.concatenate([fy - 1, fy, fy + 1])
    GX, GY = np.meshgrid(tx, ty, indexing="ij")
    fig, ax = plt.subplots()
    cs = ax.contour(GX, GY, tile, levels=[level])
    plt.close(fig)
    polys = []
    for path in cs.get_paths():
        v, codes = path.vertices, path.codes
        if codes is None:
            polys.append(np.asarray(v))
            continue
        starts = list(np.nonzero(codes == 1)[0]) + [len(v)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            if s1 - s0 > 1:
                polys.append(v[s0:s1])
    folded = []
    for s in polys:
        cart = np.asarray(s) @ b2d
        best, bd = None, None
        for i in range(-2, 3):
            for j in range(-2, 3):
                sh = (i * b[0] + j * b[1])[:2]
                dd = float(np.linalg.norm(cart.mean(axis=0) - sh))
                if bd is None or dd < bd:
                    bd, best = dd, sh
        folded.append(cart - best[None, :])
    return folded


def fs_segments(g, fx, fy, b2d, b1, b2):
    """真费米面(笛卡尔坐标的折线)。"""
    segs = []
    for c in P.zero_contours(g, fx, fy, b2d, b1, b2):
        for piece in P.split_at_jumps(c):
            segs += [piece[k:k + 2] for k in range(len(piece) - 1)]
    return segs


def build(npz, poscar, out, png=None, ewin=0.48, height=2.6, lang="en"):
    S.apply_prb_style(lang=lang)
    prim = read_poscar(poscar)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    hexv = P.bz_hexagon(b)
    hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
    hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
    mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
    lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))

    d = P.load(npz)
    n1, n2, fx, fy = d["n1"], d["n2"], d["fx"], d["fy"]
    e_f, ew, sig = d["e_f"], d["e_grid"], d["sigma"]
    a, be = d["a"], d["be"]
    i_ef = int(np.argmin(np.abs(ew - e_f)))

    j0 = int(np.argmin(np.abs(fy)))
    half = n1 // 2
    ks = np.arange(half + 1)
    line_A = a[ks, j0, :]
    line_E = np.array([be[bb][ks, j0] for bb in range(be.shape[0])])

    i_lo = int(np.argmin(np.abs(ew - (e_f - ewin))))
    i_hi = int(np.argmin(np.abs(ew - (e_f + ewin))))

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.0, 1.0], wspace=0.34)

    # ---------------- (a) 谱图 ----------------
    ax = fig.add_subplot(gs[0, 0])
    kcart = np.column_stack([ks / n1, np.full(len(ks), fy[j0])]) @ b2d
    dist = np.linalg.norm(kcart - kcart[0], axis=1)
    V = line_A[:, i_lo:i_hi + 1]
    im = ax.pcolormesh(dist, ew[i_lo:i_hi + 1] - e_f, V.T, cmap="magma",
                       shading="auto", vmin=0.0, vmax=float(np.nanmax(V)))
    ax.axhline(0.0, color="w", ls="--", lw=0.9)
    for bb in range(be.shape[0]):
        ok = np.isfinite(line_E[bb]) & (np.abs(line_E[bb] - e_f) < ewin)
        if ok.sum() > 2:
            ax.plot(dist[ok], line_E[bb][ok] - e_f, color="w", lw=0.7,
                    alpha=0.55)
    ax.set_xlabel(LBL["cut_x"][lang])
    ax.set_ylabel(r"$E-E_F$  (eV)")
    ax.set_title(LBL["a_title"][lang], fontsize=S.FS_TITLE)

    a_line = line_A[:, i_ef]
    lev = 0.5 * float(np.nanmax(a_line))
    cross = []
    for i in range(len(dist) - 1):
        if (a_line[i] - lev) * (a_line[i + 1] - lev) < 0:
            t = (lev - a_line[i]) / (a_line[i + 1] - a_line[i])
            cross.append(dist[i] + t * (dist[i + 1] - dist[i]))
    dbest = np.full(len(ks), np.inf)
    sgn = np.zeros(len(ks))
    for bb in range(be.shape[0]):
        ok = np.isfinite(line_E[bb])
        d_ = np.where(ok, np.abs(line_E[bb] - e_f), np.inf)
        m = d_ < dbest
        dbest = np.where(m, d_, dbest)
        sgn = np.where(m, line_E[bb] - e_f, sgn)
    x_fs = float(dist[int(np.argmin(np.abs(sgn)))])
    for xc in cross:
        ax.axvline(xc, color=COLOR_WALL, lw=1.6)
    ax.axvline(x_fs, color=COLOR_FS, lw=2.0)
    ax.legend(handles=[
        plt.Line2D([], [], color=COLOR_WALL, lw=1.6,
                   label=LBL["leg_flank"][lang]),
        plt.Line2D([], [], color=COLOR_FS, lw=2.0,
                   label=LBL["leg_crest"][lang]),
    ], loc="lower left", fontsize=S.FS_ANNOT, frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.85,
        handlelength=1.4, borderpad=0.3)

    # ---------------- (b) A(k,E_F) 等值线 = 两壁 ----------------
    ax = fig.add_subplot(gs[0, 1])
    a_ef = a[:, :, i_ef]
    segs = wall_segments(a_ef, fx, fy, b, b2d, lev)
    ax.add_collection(LineCollection(segs, colors=COLOR_WALL, linewidths=2.0))
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_facecolor(BG)
    ax.add_patch(Polygon(hexv, closed=True, fill=False, edgecolor=COLOR_BZ,
                         lw=0.9, ls="--"))
    ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
    ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
    ax.set_title(LBL["b_title"][lang], fontsize=S.FS_TITLE)

    # ---------------- (c) 真费米面 ----------------
    ax = fig.add_subplot(gs[0, 2])
    e_star = P.blur_cartesian(P.peak_energy_field(a, ew, e_f, sig), b2d, 1.0)
    g = np.where(np.abs(e_star - e_f) <= 0.25, e_star, np.nan)
    g = np.where(a_ef >= 0.02 * np.nanmax(a_ef), g, np.nan)
    segs = fs_segments(g - e_f, fx, fy, b2d, b[0][:2], b[1][:2])
    ax.add_collection(LineCollection(segs, colors=COLOR_FS, linewidths=2.0))
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_facecolor(BG)
    ax.add_patch(Polygon(hexv, closed=True, fill=False, edgecolor=COLOR_BZ,
                         lw=0.9, ls="--"))
    ax.set_xlabel(r"$k_1$  (2$\pi$/Å)")
    ax.set_ylabel(r"$k_2$  (2$\pi$/Å)")
    ax.set_title(LBL["c_title"][lang], fontsize=S.FS_TITLE)

    fig.subplots_adjust(left=0.065, right=0.905, top=0.90, bottom=0.16,
                        wspace=0.34)
    # 色标放在**整幅图最右边**: 只有 (a) 是上色图, 但色标挨着 (b) 会和 (b) 的
    # y 轴标签打架 —— 放最右就没有左边的东西跟它抢位置。
    cax = fig.add_axes([0.925, 0.22, 0.012, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$A(\mathbf{k},E)$", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK, pad=1)
    cb.outline.set_visible(False)
    S.save(fig, out, png_preview=png)
    plt.close(fig)
    print(f"  两壁在切线上的位置: {np.round(cross, 4)}")
    print(f"  真费米面位置: {x_fs:.4f}  (两壁中点 {np.mean(cross):.4f})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--prim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--ewin", type=float, default=0.48)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="图内固定文字的语言")
    args = ap.parse_args()
    build(args.npz, args.prim, args.out, png=args.png or None,
          ewin=args.ewin, height=args.height, lang=args.lang)


if __name__ == "__main__":
    main()
