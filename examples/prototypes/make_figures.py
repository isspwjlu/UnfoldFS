#!/usr/bin/env python3
"""由 unfolded_bands.dat 生成反折叠谱函数图件:
  1) A(kx,ky) 在若干能量处的二维切片 (费米面演化)
  2) 价带顶 (VBM) 处的"反折叠费米面"
  3) A(k,E) 的 k 分辨态密度图
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def load(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(x) for x in p[:5]])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="unfolded_bands.dat")
    ap.add_argument("--doscar", default="DOSCAR")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--ef", type=float, default=None)
    ap.add_argument("--eigenval", default="EIGENVAL")
    args = ap.parse_args()

    arr = load(args.bands)
    print(f"[read] {len(arr)} 条 (k, E, w)")

    # DOSCAR -> E_F
    ef = args.ef
    if ef is None:
        with open(args.doscar) as fh:
            for _ in range(5):
                fh.readline()
            ef = float(fh.readline().split()[3])
    print(f"[ef] E_F = {ef:.4f} eV")

    kk = np.unique(np.round(arr[:, 0:3], 6), axis=0)
    kidx = {tuple(k): i for i, k in enumerate(kk)}
    ik = np.array([kidx[tuple(np.round(r[0:3], 6))] for r in arr])

    # 能量轴
    emin = float(arr[:, 3].min()) - 3 * args.sigma
    emax = float(arr[:, 3].max()) + 3 * args.sigma
    nE = 1400
    Egrid = np.linspace(emin, emax, nE)
    dE = Egrid[1] - Egrid[0]
    A = np.zeros((len(kk), nE))
    for r, i in zip(arr, ik):
        j = int(np.argmin(np.abs(Egrid - r[3])))
        A[i, j] += r[4]
    A = gaussian_filter1d(A, args.sigma / dE, axis=1, mode="nearest")
    print(f"[spec] {A.shape[0]} k x {A.shape[1]} E,  max A = {A.max():.4f}")

    # VBM / CBM: 由 EIGENVAL 的占据数确定(费米面在带隙内时无 FS)
    vbm, cbm = None, None
    try:
        with open(args.eigenval) as fh:
            lines = fh.readlines()
        nb = int(lines[5].split()[-1])
        idx, occ_e, occ_o = 6, [], []
        for ik in range(len(lines) - 1):
            if idx >= len(lines):
                break
            if not lines[idx].strip():
                idx += 1
                continue
            idx += 1
            for ib in range(nb):
                if idx >= len(lines):
                    break
                v = lines[idx].split()
                occ_e.append(float(v[1]))
                occ_o.append(float(v[2]))
                idx += 1
            idx += 1
        oe = np.array(occ_e)
        oo = np.array(occ_o)
        vbm = float(oe[oo > 0.5].max())
        cbm = float(oe[oo < 0.5].min())
        print(f"[gap] VBM = {vbm:.3f} eV, CBM = {cbm:.3f} eV, "
              f"带隙 = {cbm - vbm:.3f} eV")
    except Exception as exc:
        print(f"[warn] 读 EIGENVAL 失败({exc}), 用权重峰代替")
        tot = A.sum(axis=0)
        occ = np.where(tot > 0.02 * tot.max())[0]
        vbm = Egrid[occ.max()]
    print(f"[vbm] 价带顶 ≈ {vbm:.3f} eV   (E_F = {ef:.3f} eV, "
          f"{'半导体: E_F 在带隙内' if abs(ef - vbm) > 0.2 else '金属'})")

    x, y = kk[:, 0], kk[:, 1]
    lim = max(abs(x).max(), abs(y).max()) * 1.03

    def draw(ax, ei, title, vmax=None):
        j = int(np.argmin(np.abs(Egrid - ei)))
        v = A[:, j]
        vmax = vmax or max(v.max(), 1e-12)
        lv = np.linspace(0, vmax, 21)
        sc = ax.tricontourf(x, y, v, levels=lv, cmap="inferno", extend="max")
        ax.tricontour(x, y, v, levels=lv[::4], colors="w", linewidths=0.25, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$k_x$ (2$\pi$/$\AA$)", fontsize=8)
        ax.set_ylabel(r"$k_y$ (2$\pi$/$\AA$)", fontsize=8)
        ax.tick_params(labelsize=7)
        return sc

    # 图 1: 能量切片的费米面演化
    es = np.linspace(vbm - 2.0, vbm + 0.5, 6)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    vmax = max(A.max(), 1e-12)
    for ax, e in zip(axes.ravel(), es):
        sc = draw(ax, e, f"E = {e:.2f} eV", vmax)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8)
    cb.set_label(r"$A(\mathbf{k}, E)$")
    fig.suptitle("Unfolded spectral function of the 3x3x1 MoS2 supercell "
                 "(primitive BZ)", fontsize=11)
    fp = os.path.join(args.outdir, "unfolded_specfunc_slices.pdf")
    fig.savefig(fp, dpi=350, bbox_inches="tight")
    print(f"[write] {fp}")

    # 图 2: VBM 处的反折叠费米面
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sc = draw(ax, vbm, f"Unfolded Fermi surface at VBM ({vbm:.2f} eV)", vmax)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$A(\mathbf{k}, E_F)$")
    fp = os.path.join(args.outdir, "unfolded_fermi_surface.pdf")
    fig.savefig(fp, dpi=350, bbox_inches="tight")
    print(f"[write] {fp}")

    # 图 3: k 分辨态密度 A(k,E)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    jslice = slice(max(0, int(np.argmin(np.abs(Egrid - vbm))) - 3),
                   int(np.argmin(np.abs(Egrid - vbm))) + 4)
    im = ax.tricontourf(x, y, A[:, jslice].mean(axis=1), cmap="inferno",
                        levels=np.linspace(0, vmax, 21), extend="max")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$A(\mathbf{k}, E)$")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (2$\pi$/$\AA$)")
    ax.set_ylabel(r"$k_y$ (2$\pi$/$\AA$)")
    ax.set_title("Unfolded spectral function (2D BZ)", fontsize=10)
    fp = os.path.join(args.outdir, "unfolded_specfunc_2d.pdf")
    fig.savefig(fp, dpi=350, bbox_inches="tight")
    print(f"[write] {fp}")

    # 逐带能量谱(带权重): 每个 k 点按权重挑出"有效带"
    # 输出 band.dat 供 FermiSurfer
    bdir = os.path.join(args.outdir, "bands_unfolded")
    os.makedirs(bdir, exist_ok=True)
    per_k = {}
    for r in arr:
        per_k.setdefault(tuple(np.round(r[0:3], 6)), []).append((r[3], r[4]))
    nmax = max(len(v) for v in per_k.values())
    for ib in range(min(nmax, 120)):
        fn = os.path.join(bdir, f"band{ib+1}.dat")
        with open(fn, "w") as fh:
            fh.write("# kx ky kz  E(eV)  weight\n")
            for kk2, lst in per_k.items():
                lst = sorted(lst, key=lambda t: -t[1])
                if ib < len(lst):
                    fh.write(f"{kk2[0]:.8f} {kk2[1]:.8f} {kk2[2]:.8f} "
                             f"{lst[ib][0]:.6f} {lst[ib][1]:.8f}\n")
    print(f"[write] {bdir}/band1..{min(nmax,120)}.dat (FermiSurfer 格式)")


if __name__ == "__main__":
    main()
