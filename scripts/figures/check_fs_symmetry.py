"""核对费米面图的对称性(几何 + 颜色)。

**几何**: 曲线到它在点群各操作下的像的**点到线段**距离(以网格间距为单位)。
    ⚠ 不要用"光栅化比像素"(亚像素位移就整片翻红, 会把 0.001 格的偏差报成 10%),
      也不要用"顶点配对"距离(量的是顶点采样, 不是几何)。
**颜色**: 在对称相关的两点上取色, 比较两边的值。

用法:
    python check_fs_symmetry.py --prim <原胞POSCAR> <npz> [<npz> ...]
    (不 --prim 时按 npz 路径猜: 含 "nbse2" → NbSe2_prim.vasp, 否则 PRIMCELL.vasp;
     也可用环境变量 TAS2_PRIM / NBSE2_PRIM 指定)
"""
import os
import sys

import numpy as np

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from unfoldfs.geometry import read_poscar, reciprocal, rotation_inverse
import plot_fs as P


def prim_for(npz):
    """按 npz 路径猜对应的原胞 POSCAR(可用 --prim / 环境变量覆盖)"""
    if "nbse2" in npz:
        return os.environ.get("NBSE2_PRIM", "NbSe2_prim.vasp")
    return os.environ.get("TAS2_PRIM", "PRIMCELL.vasp")


def build(npz, prim_file, blur=1.0, sigma_cells=1.5):
    """与 plot_fs.py 完全同口径地重建一个面板的曲线"""
    prim = read_poscar(prim_file)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    ops, _ = P.get_ops(prim)
    d = P.load(npz)
    n1, n2, fx, fy = d["n1"], d["n2"], d["fx"], d["fy"]
    e_f = d["e_f"]
    e = P.peak_energy_field(d["a"], d["e_grid"], e_f, d["sigma"])
    w = P.weight_at_peak(d["be"], d["bw"], d["va"], e)
    w = P.symmetrize(w, ops, n1, n2)
    if blur > 0:
        e = P.blur_cartesian(e, b2d, blur)
    g = np.where(np.abs(e - e_f) <= 0.25, e, np.nan)
    i_ef = int(np.argmin(np.abs(d["e_grid"] - e_f)))
    aef = d["a"][:, :, i_ef]
    g = np.where(aef >= 0.02 * float(np.nanmax(aef)), g, np.nan)
    segs = []
    for c in P.zero_contours(g - e_f, fx, fy, b2d, b[0][:2], b[1][:2]):
        for piece in P.split_at_jumps(c):
            segs += [piece[k:k + 2] for k in range(len(piece) - 1)]
    return np.array(segs), w, ops, b2d, n1


def seg_dist(pts, segs, block=1024):
    """点到一串线段的最小距离(分块, 否则 n² 的内存会爆)"""
    a = segs[:, 0]
    b = segs[:, 1]
    ab = b - a
    denom = np.einsum("jk,jk->j", ab, ab)
    out = np.empty(len(pts))
    for s in range(0, len(pts), block):
        p = pts[s:s + block]
        d = p[:, None, :] - a[None, :, :]
        t = np.einsum("ijk,jk->ij", d, ab) / np.maximum(denom[None, :], 1e-30)
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        out[s:s + block] = np.linalg.norm(p[:, None, :] - proj, axis=-1).min(axis=1)
    return out


def check(npz, prim_file):
    segs, w, ops, b2d, n1 = build(npz, prim_file)
    inv = np.linalg.inv(b2d)
    h = float(np.linalg.norm(b2d[0])) / n1          # 网格间距
    verts = np.vstack([s for s in segs])
    print(f"== {npz}   顶点 {len(verts)}   网格间距 {h:.5f} (2π/Å)")
    worst_g, worst_c = 0.0, 0.0
    for k, r in enumerate(ops):
        v3 = np.concatenate([verts @ inv, np.zeros((len(verts), 1))], 1)
        img = (v3 @ rotation_inverse(r))[:, :2] @ b2d
        dd = seg_dist(img, segs) / h                  # 几何偏差(格)
        wi = P.sample_equiv(w, img, b2d, 1.5)
        w0 = P.sample_equiv(w, verts, b2d, 1.5)
        dc = float(np.abs(wi - w0).max())
        worst_g = max(worst_g, float(dd.max()))
        worst_c = max(worst_c, dc)
        print(f"   op{k:2d}: 几何偏差 中位 {np.median(dd):.4f} 格 / "
              f"90% {np.percentile(dd, 90):.4f} / 最大 {dd.max():.4f} 格"
              f"   颜色最大差 {dc:.2e}")
    print(f"   >>> 汇总: 几何最大 {worst_g:.4f} 格 (BZ 的 {worst_g * h:.5f} 2π/Å)"
          f"   颜色最大 {worst_c:.2e}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    pf = None
    if "--prim" in argv:
        i = argv.index("--prim")
        pf = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for npz in argv:
        check(npz, pf or prim_for(npz))
