"""三维反折叠费米面（1T-TaS₂ / 1H-NbSe₂ 这类**单层**体系）。

**物理前提（决定了"三维费米面"该画成什么）**

1. 单层（c = 17.66 Å，真空层很厚）⇒ 带在 kz 方向**几乎无色散** ⇒
   三维布里渊区里的费米面 = 二维费米面轨迹沿 kz 拉伸成的**带**，
   在 kz = 0 与 kz = ±π/c 各有一份（周期性等价）。
   我们的算例只在 kz = 0 取样（A(k,E) 只有 kz = 0）⇒ 三维 FS 的"面"没有
   额外的色散信息。
2. 因此 3D 渲染取**物理版**：FS 轨迹沿 kz 扫成一条**带**（kz ∈ [-π/c, π/c]），
   颜色 = 谱权重（``--mode sheet``，默认）。kz 方向按比例拉伸显示（单层的 BZ 只有
   面内的 1/5 高，等比例画几乎看不出厚度）。
   另备 ``--mode tube``：FS 轨迹吹成**管**、**管半径 ∝ 谱权重**（强副本粗、弱副本细），
   ⚠ 此时管在 kz 方向的粗细**不代表色散**，只表示权重（图注里写明）。
3. 反折叠的 3D 价值：原胞费米面与它的折叠副本**同时**出现在原胞 BZ 里。

**实现**: VTK/pyvista 负责 3D 渲染（每个面板出一张 PNG），matplotlib 负责
拼版与中文标注（VTK 的内置字体**没有中文字形** ⇒ 中文一律放 matplotlib 里画）。

用法:
    python plot_fs_3d.py --prim PRIMCELL.vasp \
        --npz "tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz,tas2_fs_scsqrt3.npz" \
        --titles "原胞直通 (direct),3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
        --blue-panels 0 --out fs3d.png [--mode sheet|tube] [--kz-scale 3]
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import pyvista as pv
from pyvista import Plotter

from unfoldfs.geometry import read_poscar, reciprocal
import plot_fs as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 费米面数据

def fs_pieces(npz, b2d, b1, b2, ops, args):
    """-> [(笛卡尔轨迹 N×2, 权重 N), ...]  (与 plot_fs.py 同一口径)"""
    d = P.load(npz, getattr(args, "kz", None))
    n1, n2, fx, fy = d["n1"], d["n2"], d["fx"], d["fy"]
    e_f = d["e_f"]
    e = P.peak_energy_field(d["a"], d["e_grid"], e_f, d["sigma"])
    w = P.weight_at_peak(d["be"], d["bw"], d["va"], e)
    if not args.no_sym:
        w = P.symmetrize(w, ops, n1, n2)
    if args.blur > 0:
        e = P.blur_cartesian(e, b2d, args.blur)
    g = np.where(np.abs(e - e_f) <= args.mask, e, np.nan)
    i_ef = int(np.argmin(np.abs(d["e_grid"] - e_f)))
    aef = d["a"][:, :, i_ef]
    g = np.where(aef >= args.wmin * float(np.nanmax(aef)), g, np.nan)
    out = []
    for c in P.zero_contours(g - e_f, fx, fy, b2d, b1, b2):
        for piece in P.split_at_jumps(c):
            if len(piece) < 3:
                continue
            wts = P.sample_equiv(w, piece, b2d, args.sigma_cells)
            out.append((np.asarray(piece), np.asarray(wts)))
    return out


# ---------------------------------------------------------------- 网格构造

def _flat(cells):
    """VTK 的 cell 数组格式: [n0, i0, i1, ..., n1, ...]"""
    out = []
    for f in cells:
        out.append(len(f))
        out.extend(int(i) for i in f)
    return np.asarray(out, dtype=np.int64)


def tube_mesh(curve3, wts, r0, nring=16):
    """把一条**平面**轨迹吹成管：截面 = 面内法向 × kz 方向, 半径 ∝ 权重。

    轨迹在 kz = 0 平面上 ⇒ 法向直接取 z×切向, 不会有扭转问题。
    """
    p = curve3
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
    n = np.cross(t, np.array([0.0, 0.0, 1.0]))
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    wmax = float(np.max(wts)) if len(wts) else 1.0
    rad = r0 * (0.18 + 0.82 * np.asarray(wts) / max(wmax, 1e-12))
    th = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    pts = np.empty((len(p) * nring, 3))
    for i in range(len(p)):
        ring = (np.cos(th)[:, None] * n[i][None, :]
                + np.sin(th)[:, None] * np.array([0.0, 0.0, 1.0])[None, :])
        pts[i * nring:(i + 1) * nring] = p[i] + rad[i] * ring
    faces = []
    for i in range(len(p) - 1):
        for k in range(nring):
            faces.append([i * nring + k,
                          i * nring + (k + 1) % nring,
                          (i + 1) * nring + (k + 1) % nring,
                          (i + 1) * nring + k])
    mesh = pv.PolyData(pts, faces=_flat(faces))
    mesh["w"] = np.repeat(np.asarray(wts), nring)
    return mesh


def sheet_mesh(curve3, wts, z_half):
    """把轨迹沿 kz 扫成一条**带**(kz ∈ [-z_half, z_half])—— 单层体系的物理版 FS。"""
    p = np.asarray(curve3)[:, :2]
    n = len(p)
    pts = np.vstack([np.column_stack([p, np.full(n, -z_half)]),
                     np.column_stack([p, np.full(n, z_half)])])
    faces = [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)]
    mesh = pv.PolyData(pts, faces=_flat(faces))
    mesh["w"] = np.concatenate([wts, wts])
    return mesh


def bz_prism(bz_xy, z_half):
    """六棱柱 BZ 线框(单层的 BZ 沿 c 方向很扁)"""
    n = len(bz_xy)
    allp = np.vstack([np.column_stack([bz_xy, np.full(n, -z_half)]),
                      np.column_stack([bz_xy, np.full(n, z_half)])])
    edges = []
    for i in range(n):
        edges.append([i, (i + 1) % n])                 # 底面六边形
        edges.append([n + i, n + (i + 1) % n])         # 顶面六边形
        edges.append([i, n + i])                       # 竖棱
    return pv.PolyData(allp, lines=_flat(edges))


# ---------------------------------------------------------------- 渲染

def render_panel(pieces, hexv, z_half, blue, args, out_png, clim=None,
                  size=(760, 700)):
    """用 pyvista 出一个面板(不含中文 —— VTK 内置字体没有中文字形)"""
    pl = Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    meshes = []
    for curve, wts in pieces:
        c3 = np.column_stack([curve, np.zeros(len(curve))])
        if args.mode == "tube":
            meshes.append(tube_mesh(c3, wts, args.radius))
        else:
            meshes.append(sheet_mesh(c3, wts, z_half))
    if meshes:
        merged = meshes[0]
        for m in meshes[1:]:
            merged = merged.merge(m)
        if blue:
            pl.add_mesh(merged, color="#1f3fbf", smooth_shading=True,
                        specular=0.25, show_scalar_bar=False)
        else:
            pl.add_mesh(merged, scalars="w", cmap="jet", smooth_shading=True,
                        specular=0.25, show_scalar_bar=False,
                        clim=clim or [0, float(np.max(merged["w"]))])
    pl.add_mesh(bz_prism(hexv, z_half), color="crimson", line_width=2.0)
    pl.add_point_labels(
        np.column_stack([hexv, np.full(len(hexv), -z_half)]),
        ["G", "M", "K", "G", "M", "K"][:len(hexv)],
        font_size=13, text_color="black", shape=None,
        always_visible=True, show_points=False)
    # 相机手动给: reset_camera() 会被点标注的包围盒带偏(缩到很小)。
    # 距离要按视场角算, 否则会"贴脸"(看不全 BZ)。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["sheet", "tube"], default="sheet")
    ap.add_argument("--no-sym", action="store_true")
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--sigma-cells", type=float, default=1.5)
    ap.add_argument("--mask", type=float, default=0.25)
    ap.add_argument("--wmin", type=float, default=0.02)
    ap.add_argument("--blue-panels", default="")
    ap.add_argument("--kz-scale", type=float, default=3.0,
                    help="kz 方向的显示拉伸倍数(单层 BZ 很扁, 等比例看不出厚度)")
    ap.add_argument("--kz", type=float, default=None,
                    help="要画的 kz 切片(分数坐标, 1.0 = 2π/c)")
    ap.add_argument("--radius", type=float, default=0.10,
                    help="管的最大半径(2π/Å)")
    ap.add_argument("--suptitle", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    pv.OFF_SCREEN = True
    prim = read_poscar(args.prim)
    b_prim = np.asarray(reciprocal(prim.lattice))
    b1, b2 = b_prim[0][:2], b_prim[1][:2]
    b2d = b_prim[:2, :2]
    ops, _ = P.get_ops(prim)
    blue_panels = {int(x) for x in args.blue_panels.split(",") if x.strip() != ""}

    hexv = P.bz_hexagon(b_prim)
    # b_prim[2] 就是沿 c 方向的倒格矢(长度 = 2π/c)⇒ BZ 的 kz 半高 = |b3|/2
    c_axis = float(np.linalg.norm(b_prim[2]))
    z_half = 0.5 * c_axis * args.kz_scale

    paths = args.npz.split(",")
    titles = args.titles.split(",")
    n = len(paths)
    tmpdir = tempfile.mkdtemp(prefix="fs3d_")
    all_pieces = [fs_pieces(p.strip(), b2d, b1, b2, ops, args) for p in paths]
    # 共享色标: 各超胞面板的权重可直接互相比较(否则每栏各自归一化, 颜色不可比)
    wmax = 0.0
    for i, pieces in enumerate(all_pieces):
        if i in blue_panels:
            continue
        for _, w in pieces:
            if len(w):
                wmax = max(wmax, float(np.max(w)))
    clim = [0.0, wmax] if wmax > 0 else None
    pngs = []
    for i, (pieces, title) in enumerate(zip(all_pieces, titles)):
        png = os.path.join(tmpdir, f"p{i}.png")
        render_panel(pieces, hexv, z_half, i in blue_panels, args, png, clim=clim)
        pngs.append(png)
        print(f"  面板 {chr(ord('a') + i)}: {len(pieces)} 段轨迹 -> {png}")

    # ---- matplotlib 拼版(中文标题/图注都在这里画) ----
    fig = plt.figure(figsize=(4.6 * n, 5.0))
    for i, (png, title) in enumerate(zip(pngs, titles)):
        ax = fig.add_subplot(1, n, i + 1)
        ax.imshow(plt.imread(png))
        ax.set_axis_off()
        ax.set_title(f"({chr(ord('a') + i)}) {title.strip()}", fontsize=11,
                     pad=4)
    # 颜色条(共享色标): 谱权重
    if clim is not None and any(i not in blue_panels for i in range(n)):
        cax = fig.add_axes([0.42, 0.095, 0.20, 0.014])
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(*clim), cmap=plt.get_cmap("jet"))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([])
        cb.outline.set_visible(False)
        cax.text(-0.012, 0.5, "Low", transform=cax.transAxes, ha="right",
                 va="center", fontsize=10)
        cax.text(1.012, 0.5, "High", transform=cax.transAxes, ha="left",
                 va="center", fontsize=10)
        cax.set_xlabel("spectral weight  $w$", fontsize=9, labelpad=2)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.975)
    if args.note:
        fig.text(0.5, 0.022, args.note, ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.13,
                        wspace=0.0)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
