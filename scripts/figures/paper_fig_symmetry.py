"""论文图 3: 展开谱权重的**点群协变性破缺**及其修法。

用 npz 里**未经对称化**的逐带数据 (band_e, band_w, valid) 按
    A(k,E) = Σ_n w_n(k) exp(−(E−E_n)²/2σ²)
重建原始 A(k,E_F), 再分别用两种点群做轨道平均:
  (a) 原始(未对称化)          δ = 9.99×10⁻¹
  (b) 原胞**完整点群**(12 个)  δ = 4.4×10⁻¹⁶
  (c) 只用原胞∩超胞**公共子群** δ = 9.61×10⁻¹
  (d) 原胞直接计算(参考)       δ = 2.6×10⁻³

δ 写在**面板标题的第二行**(不是压在 x 轴标签下面 —— 那样必然重叠)。

数据: 1T-TaS2, 3×1×1 超胞 (M = diag(3,1,1)), 96×96×1, σ = 0.06 eV。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P

from unfoldfs.geometry import read_poscar, reciprocal, rotation_inverse


def raw_A(npz, e_f):
    """按谱函数公式从**原始逐带数据**重建 A(k,E_F)。"""
    d = P.load(npz)
    n1, n2 = d["n1"], d["n2"]
    e = d["e_grid"]
    i_ef = int(np.argmin(np.abs(e - e_f)))
    be, bw, va = d["be"], d["bw"], d["va"]
    sig = d["sigma"]
    e_valid = np.where(va, be, 0.0)
    w_valid = np.where(va, bw, 0.0)
    out = np.zeros((n1, n2))
    for b in range(be.shape[0]):
        dd = (e[i_ef] - e_valid[b]) ** 2
        out += w_valid[b] * np.exp(-dd / (2.0 * sig * sig))
    return out, d


def orbit_average(field, ops, n1, n2):
    """按点群做轨道平均(把**原点**的值加到**像**上)。"""
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)], 1)
    out = np.zeros_like(field)
    cnt = np.zeros_like(field)
    vals = field.ravel()
    for r in ops:
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        np.add.at(out, (i2, j2), vals)
        np.add.at(cnt, (i2, j2), 1.0)
    return out / np.maximum(cnt, 1)


def common_ops(ops, M):
    """保留 M⁻ᵀ R Mᵀ 为整数矩阵的操作(原胞∩超胞的公共子群)。"""
    MinvT = np.linalg.inv(M).T
    keep = []
    for r in ops:
        t = MinvT @ np.asarray(r, float) @ M.T
        if np.allclose(t, np.rint(t), atol=1e-8):
            keep.append(r)
    return np.array(keep)


def dispersion(field, ops, n1, n2):
    """A(k,E_F) 在点群下的相对离散度 —— 六重对称的量化度量。"""
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)], 1)
    vals = field.ravel()
    ds = []
    for r in ops:
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        ds.append(np.abs(vals - vals[i2 * n2 + j2]).max())
    mx = float(field.max())
    return float(np.max(ds) / mx), float(np.mean(ds) / mx)


def fmt_delta(v: float) -> str:
    """把 9.99e-01 / 4.40e-16 写成 $9.99\\times10^{-1}$ 这样的形式。"""
    if v <= 0:
        return r"$0$"
    exp = int(np.floor(np.log10(v)))
    return r"$\delta = %.2f\times10^{%d}$" % (v / 10.0 ** exp, exp)


def build(sc_npz, prim_npz, poscar, out, png=None, height=2.45):
    S.apply_prb_style()
    M = np.diag([3.0, 1.0, 1.0])          # 3×1×1 超胞(破坏面内 C3)

    prim = read_poscar(poscar)
    b = np.asarray(reciprocal(prim.lattice))
    ops_full, _ = P.get_ops(prim)
    ops_sub = common_ops(ops_full, M)
    print(f"原胞完整点群: {len(ops_full)} 个操作; 3×1×1 超胞的公共子群: "
          f"{len(ops_sub)} 个")

    e_f_sc = P.load(sc_npz)["e_f"]
    e_f_pr = P.load(prim_npz)["e_f"]
    a_raw, d = raw_A(sc_npz, e_f_sc)
    n1, n2 = d["n1"], d["n2"]
    a_full = orbit_average(a_raw, ops_full, n1, n2)
    a_sub = orbit_average(a_raw, ops_sub, n1, n2)
    a_ref, dref = raw_A(prim_npz, e_f_pr)
    n1r, n2r = dref["n1"], dref["n2"]

    fields = [a_raw, a_full, a_sub, a_ref]
    titles = ["raw unfolded", "full point group",
              "common subgroup", "direct calculation"]

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.30)
    vmax = float(np.nanmax(a_full))
    for k, (f, title) in enumerate(zip(fields, titles)):
        ax = fig.add_subplot(gs[0, k])
        mx, mn = dispersion(f, ops_full, f.shape[0], f.shape[1])
        im = ax.imshow(f.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(f"({S.letter(k)}) {title}\n{fmt_delta(mx)}",
                     fontsize=S.FS_TITLE)
        ax.set_xlabel(r"$k_1$ (frac.)")
        if k == 0:
            ax.set_ylabel(r"$k_2$ (frac.)")
        ax.tick_params(labelsize=S.FS_TICK)
        S.strip_decorations(ax)
        print(f"  ({S.letter(k)}) {title}: dispersion max {mx:.2e} mean {mn:.2e}")

    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$A(\mathbf{k},E_F)$", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.outline.set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.17)
    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-npz", required=True, help="3×1×1 超胞的 npz")
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=2.45)
    args = ap.parse_args()
    build(args.sc_npz, args.prim_npz, args.prim, args.out,
          png=args.png or None, height=args.height)


if __name__ == "__main__":
    main()
