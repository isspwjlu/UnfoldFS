"""论文图: 费米速度 —— 按参考文献(bands4vasp, Dirnberger 2021)的画法。

**画法照参考**: 三维费米面(轨迹沿 kz 扫成面) + BZ 线框 + 高对称点标注 +
**用标量着色**(他们用 Bloch/轨道特征, 这里用费米速度 v_F)。

三面板:
  (a) 三维费米面, 按 **脊顶**上的 v_F 着色(= 真费米面);
  (b) 三维费米面, 按 **两壁**上的 v_F 着色(= 社区惯用的等值面);
      (a)(b) **共用同一个色标** ⇒ 颜色的错位直接可见;
  (c) 沿过 Γ 切线的带色散: 说明为什么两壁处的带不在 E_F。

数据与口径见 paper/fact_sheet.md §2.13; 逐点数值与 paper_fig_velocity.py 同源。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P
from plot_fs_3d import bz_prism, sheet_mesh
from paper_fig_velocity import curve_samples

import pyvista as pv
from pyvista import Plotter

CMAP = "turbo"          # 与 paper_fig_fs.py 同一张色图
BLUE = S.PALETTE["blue_main"]
RED = S.PALETTE["red_strong"]
DARK = S.PALETTE["gray_dark"]
GRAY = S.PALETTE["gray_mid"]


def render_vf_panel(pieces, hexv, z_half, clim, out_png, labels,
                    size=(760, 700)):
    """渲染一个 3D 面板: 费米面按 v_F 着色 + BZ 线框 + 高对称点。

    ``pieces`` = [(轨迹 N×2, v_F N), ...]
    相机由 BZ 决定(与数据无关)⇒ 不同面板**天然同视角**, 颜色可直接比较。
    """
    pl = Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    meshes = []
    for curve, vals in pieces:
        c3 = np.column_stack([np.asarray(curve)[:, :2],
                              np.zeros(len(curve))])
        meshes.append(sheet_mesh(c3, np.asarray(vals, dtype=float), z_half))
    if meshes:
        merged = meshes[0]
        for m in meshes[1:]:
            merged = merged.merge(m)
        pl.add_mesh(merged, scalars="w", cmap=CMAP, smooth_shading=True,
                    specular=0.25, show_scalar_bar=False, clim=list(clim))
    pl.add_mesh(bz_prism(hexv, z_half), color=GRAY, line_width=2.0)
    # ⚠ VTK 的内置字体**没有 LaTeX 字形** ⇒ 高对称点只能用 ASCII
    #   (Γ 要写成 "G"; 论文里的 Γ 标注由 matplotlib 那一侧来做)
    pl.add_point_labels(
        np.column_stack([hexv, np.full(len(hexv), -z_half)]),
        [str(s).replace("$\\Gamma$", "G").replace("$", "")
         for s in list(labels)[:len(hexv)]],
        font_size=14, text_color="black", shape=None,
        always_visible=True, show_points=False)
    # 相机: 与 plot_fs_3d.py 同一套(由 BZ 定, 与数据无关)
    rmax = float(np.max(np.abs(hexv)))
    half = float(np.hypot(rmax, z_half))
    va = 30.0
    dist = 1.30 * half / np.tan(np.radians(va) / 2.0)
    direction = np.array([0.62, -0.70, 0.52])
    direction /= np.linalg.norm(direction)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = tuple(direction * dist)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.view_angle = va
    pl.camera.clipping_range = (0.01 * dist, 20.0 * dist)
    pl.screenshot(out_png)
    pl.close()
    return out_png


def build(npz_prim, out, png=None, height=2.7, kz_scale=3.0):
    S.apply_prb_style()
    d = curve_samples(npz_prim)
    sl, b2d, e_f = d["sl"], d["b2d"], d["e_f"]

    # BZ 六边形(倒格矢的 Voronoi 胞)
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pts.append(i * b2d[0] + j * b2d[1])
    pts = np.asarray(pts)
    hexv = pts[np.argsort(np.linalg.norm(pts, axis=1))[:6]]
    hexv = hexv[np.argsort(np.arctan2(hexv[:, 1], hexv[:, 0]))]
    # 单层体系: kz 半高 = |b3|/2; 这里没有 c 轴, 用面内尺度的固定比例
    z_half = 0.16 * float(np.max(np.linalg.norm(hexv, axis=1))) * kz_scale

    v_crest, _, _, pc = d["crest"]
    v_flank, _, _, pf = d["flank"]
    good_c = np.isfinite(v_crest)
    good_f = np.isfinite(v_flank)
    vmin = float(np.percentile(np.concatenate([v_crest[good_c],
                                               v_flank[good_f]]), 2))
    vmax = float(np.percentile(np.concatenate([v_crest[good_c],
                                               v_flank[good_f]]), 98))
    clim = (vmin, vmax)

    # 曲线与逐点 v_F 要按顺序对应 ⇒ 重新逐曲线取点
    def pieces_with_v(curves, v_all):
        """把逐点 v 按曲线重新切分(curve_samples 已按曲线顺序展平)。"""
        out, i = [], 0
        for c in curves:
            n = len(c)
            out.append((c, v_all[i:i + n]))
            i += n
        return out

    pc_pieces = pieces_with_v(d["crest_curves"], v_crest)
    pf_pieces = pieces_with_v(d["flank_curves"], v_flank)

    tmp = tempfile.mkdtemp()
    png_a = os.path.join(tmp, "a.png")
    png_b = os.path.join(tmp, "b.png")
    labels = [r"$\Gamma$", "M", "K", r"$\Gamma$", "M", "K"]
    render_vf_panel(pc_pieces, hexv, z_half, clim, png_a, labels)
    render_vf_panel(pf_pieces, hexv, z_half, clim, png_b, labels)

    # ---------------------------------------------------------------- 拼版
    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)

    for k, (png, title) in enumerate(((png_a, r"(a) crest: $v_F$ on the true "
                                                r"Fermi surface"),
                                      (png_b, r"(b) flanks: $v_F$ read on "
                                                r"the isosurface"))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(plt.imread(png))
        ax.set_axis_off()
        ax.set_title(title, fontsize=S.FS_TITLE)

    # (c) 带色散: 两壁处带不在 E_F
    ax = fig.add_subplot(gs[0, 2])
    n1, n2 = sl.n1, sl.n2
    j0 = int(np.argmin(np.abs(sl.fy)))
    ks = np.arange(n1 // 2 + 1)
    kcart = sl.k_cart.reshape(n1, n2, 3)[ks, j0, :2]
    dist = np.linalg.norm(kcart - kcart[0], axis=1)
    be = sl.band_e[:, ks, j0]
    va = sl.valid[:, ks, j0]
    band = np.full(len(ks), np.nan)
    for i in range(len(ks)):
        ok = va[:, i] & np.isfinite(be[:, i])
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        band[i] = be[idx[int(np.argmin(np.abs(be[idx, i] - e_f)))], i]
    ax.axhline(e_f, color=DARK, lw=1.0, ls="--")
    ax.plot(dist, band - e_f, color=BLUE, lw=S.LW_DATA)
    a_cut = sl.a[:, :, int(np.argmin(np.abs(sl.e_grid - e_f)))][ks, j0]
    lev = 0.5 * float(np.nanmax(a_cut))
    walls = []
    for i in range(len(dist) - 1):
        if (a_cut[i] - lev) * (a_cut[i + 1] - lev) < 0:
            t = (lev - a_cut[i]) / (a_cut[i + 1] - a_cut[i])
            walls.append(dist[i] + t * (dist[i + 1] - dist[i]))
    good = np.isfinite(band - e_f)
    d_c, b_c = dist[good], (band - e_f)[good]
    x_crest = []
    for i in range(len(d_c) - 1):
        if b_c[i] * b_c[i + 1] < 0:
            t = -b_c[i] / (b_c[i + 1] - b_c[i])
            x_crest.append(d_c[i] + t * (d_c[i + 1] - d_c[i]))
    for x in walls:
        ax.axvline(x, color=RED, lw=1.2)
        j = int(np.argmin(np.abs(dist - x)))
        right = x > 0.5 * (dist.min() + dist.max())
        ax.annotate(r"%+.3f eV" % (band[j] - e_f), xy=(x, band[j] - e_f),
                    xytext=(-4 if right else 4, 5),
                    ha="right" if right else "left",
                    textcoords="offset points", fontsize=S.FS_ANNOT, color=RED)
    for x in x_crest:
        ax.axvline(x, color=BLUE, lw=1.2)
    ax.set_xlabel("distance along the cut (2$\\pi$/Å)")
    ax.set_ylabel(r"$E_b-E_F$ (eV)")
    ax.set_title(r"(c) the band at the flanks is not at $E_F$",
                 fontsize=S.FS_TITLE)
    ax.set_ylim(-0.20, 0.20)
    if walls and x_crest:
        lo = min(min(walls), min(x_crest))
        hi = max(max(walls), max(x_crest))
        pad = 0.30 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)

    # 共用色标(放在 (c) 右侧)
    cax = fig.add_axes([0.955, 0.28, 0.010, 0.42])
    sm = plt.cm.ScalarMappable(norm=Normalize(*clim), cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$v_F$ ($10^{5}$ m/s)", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK, pad=1)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e5:.1f}"))

    fig.subplots_adjust(left=0.045, right=0.945, top=0.86, bottom=0.16,
                        wspace=0.28)
    S.save(fig, out, png_preview=png)
    plt.close(fig)
    print(f"  已写出 {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz-prim", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png", default="")
    p.add_argument("--height", type=float, default=2.7)
    p.add_argument("--kz-scale", type=float, default=3.0)
    a = p.parse_args()
    build(a.npz_prim, a.out, png=a.png or None, height=a.height,
          kz_scale=a.kz_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
