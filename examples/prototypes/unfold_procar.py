#!/usr/bin/env python3
"""投影法超胞能带反折叠 (Popescu, PRL 104, 216401 (2012)) + E_F 谱函数 / 费米面。

需要 LORBIT=12 的 PROCAR (VASP 5.4/6.x 写复投影系数 Re/Im; LORBIT=11 不写相位!)。

PROCAR (LORBIT=12) 每个 band 的结构:
    ion  s py pz px dxy dyz dz2 dxz dx2-y2  tot      <- 模方块 (11 字段/行)
    <nions 行>
    tot ...
    ion  s py pz px dxy dyz dz2 dxz dx2-y2           <- 复投影块 (20 字段/行)
    <nions 行>   ion + 9*(Re,Im) + |c|^2

公式:
    A_i(lm) = sum_{超胞中 i 的副本 j} c_{j,lm} * exp(s1 * i K.(R_j - r_i))
    C_i     = sum_lm A_i(lm)
    w(k')   = |sum_i C_i * exp(s2 * i k'.r_i)|^2 ,   k' = K + G
相位符号约定 (s1,s2) 由"谱守恒"自检自动确定: sum_G w 应 ~1。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np

TWOPI = 2.0 * math.pi


# ------------------------------------------------------------------ 读取
def read_poscar(path: str):
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip()
    idx = 8
    if mode.lower().startswith("s"):
        idx = 9
    cart = mode.lower().startswith("c")
    inv = np.linalg.inv(lat)
    pos = []
    species = []
    for s, n in zip(symbols, counts):
        for _ in range(n):
            p = np.array([float(x) for x in lines[idx].split()[:3]])
            if cart:
                p = p @ inv
            pos.append(p)
            species.append(s)
            idx += 1
    return lat, species, np.array(pos)


def reciprocal(lat: np.ndarray) -> np.ndarray:
    """3x3, 行=倒格矢, 含 2pi (单位 2pi/A)"""
    v = np.dot(lat[0], np.cross(lat[1], lat[2]))
    return TWOPI * np.array([
        np.cross(lat[1], lat[2]), np.cross(lat[2], lat[0]), np.cross(lat[0], lat[1])
    ]) / v


def read_transmat(path: str) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            p = line.split()
            if len(p) == 3:
                rows.append([int(float(x)) for x in p])
    return np.array(rows, dtype=float)


def read_eigenval(path: str):
    with open(path) as fh:
        lines = fh.readlines()
    hdr = lines[5].split()
    nkpts, nbands = int(hdr[-2]), int(hdr[-1])
    kpts = np.zeros((nkpts, 3))
    ener = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        kpts[ik] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
        for ib in range(nbands):
            ener[ik, ib] = float(lines[idx].split()[1])
            idx += 1
        idx += 1
    return kpts, ener


def read_procar(path: str):
    """-> (cplx[NK][NB][NIONS][NORB] 复投影, w2[NK][NB][NIONS][NORB] 模方)"""
    with open(path) as fh:
        lines = fh.readlines()
    nums = [int(x) for x in re.findall(r"\d+", lines[1])]
    nk, nb, nions = nums[0], nums[1], nums[2]
    norb = None
    for i in range(4, min(len(lines), 80)):
        if lines[i].lstrip().startswith("ion"):
            norb = len(lines[i].split()) - 2
            break
    cplx = np.zeros((nk, nb, nions, norb), dtype=complex)
    w2 = np.zeros((nk, nb, nions, norb))
    has_cplx = False
    idx = 0
    for ik in range(nk):
        while idx < len(lines) and not lines[idx].lstrip().startswith("k-point"):
            idx += 1
        if idx >= len(lines):
            break
        idx += 1
        for ib in range(nb):
            while idx < len(lines) and not lines[idx].lstrip().startswith("band"):
                idx += 1
            if idx >= len(lines):
                break
            # 收集本 band 的所有数据行 直到下一个 band/k-point
            rows = []
            j = idx + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("band") or s.startswith("k-point"):
                    break
                if s and s[0].isdigit():
                    rows.append((j, len(s.split()), s.split()))
                j += 1
            idx = j
            for (ln, nf, v) in rows:
                ia = int(v[0]) - 1
                if nf == 11 and ia < nions:
                    for m in range(norb):
                        w2[ik, ib, ia, m] = float(v[1 + m])
                elif nf == 20 and ia < nions:
                    has_cplx = True
                    for m in range(norb):
                        re_ = float(v[1 + 2 * m])
                        im_ = float(v[2 + 2 * m])
                        cplx[ik, ib, ia, m] = re_ + 1j * im_
    print(f"[PROCAR] NKPTS={nk} NBANDS={nb} NIONS={nions} NORB={norb} "
          f"complex_block={has_cplx}")
    if not has_cplx:
        raise SystemExit("PROCAR 中没有复投影块 —— 需要 LORBIT=12 (LORBIT=11 不写相位)")
    return cplx, w2


def read_ef(path: str) -> float:
    """DOSCAR 第 6 行: EMAX EMIN NEDOS EFERMI ..."""
    with open(path) as fh:
        for _ in range(5):
            fh.readline()
        return float(fh.readline().split()[3])


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--ef", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--emax", type=float, default=None)
    ap.add_argument("--emin", type=float, default=None)
    args = ap.parse_args()
    d = args.dir

    lat_sc, _, pos_sc = read_poscar(os.path.join(d, "POSCAR"))
    lat_pr, _, pos_pr = read_poscar(os.path.join(d, "PRIMCELL.vasp"))
    M = read_transmat(os.path.join(d, "TRANSMAT.in"))
    Minv = np.linalg.inv(M)
    ndet = int(round(abs(np.linalg.det(M))))
    b_pr = reciprocal(lat_pr)
    b_sc = reciprocal(lat_sc)
    cart_pr = pos_pr @ lat_pr
    cart_sc = pos_sc @ lat_sc

    # 原胞原子 -> 超胞副本 (平移 T 为原胞格矢单位)
    reps = []
    for i, p in enumerate(pos_pr):
        lst = []
        for j, q in enumerate(pos_sc):
            dq = (q @ lat_sc - p @ lat_pr) @ np.linalg.inv(lat_sc)
            Ti = np.round(dq @ M)
            if np.allclose(dq @ M, Ti, atol=1e-3):
                lst.append((j, Ti))
        reps.append(lst)
    ncopy = [len(r) for r in reps]
    print(f"[atoms] 原胞 {len(pos_pr)} 原子 / 超胞 {len(pos_sc)} 原子; "
          f"每原胞原子副本数 {ncopy}")
    if any(n == 0 for n in ncopy):
        raise SystemExit("[!] 原子副本匹配失败: 检查 POSCAR/PRIMCELL/TRANSMAT 是否自洽")

    kpts, ener = read_eigenval(os.path.join(d, "EIGENVAL"))
    cplx, w2 = read_procar(os.path.join(d, "PROCAR"))
    nk = min(len(kpts), cplx.shape[0])
    nband = min(ener.shape[1], cplx.shape[1])
    print(f"[data] {nk} k 点, {nband} 带")

    # 候选 G: g·M^-1 ∈ [0,1)^3
    span = int(max(abs(M).max(), 1)) + 2
    Gset = []
    for g in np.ndindex(*([3 * (span + 1) - 1] * 3)):
        gg = np.array(g) - span
        t = Minv @ gg
        if np.all(t >= -1e-9) and np.all(t < 1 - 1e-9):
            Gset.append(gg)
    if len(Gset) != ndet:
        raise SystemExit(f"[!] 候选 G 数 {len(Gset)} != det(M) {ndet}")
    print(f"[G] {len(Gset)} 个")

    # 每个超胞 k 的原胞 k 候选
    kprim_list = []
    for ik in range(nk):
        kprim_list.append([(Minv @ (kpts[ik] + G)) @ b_pr for G in Gset])

    # 相位符号约定搜索: 使 sum_G w ~ 1
    # p_i(k') = (1/N_copies) sum_T c_{i+T} exp(s1 * i (k'-K).(T))
    # w(k')   = |sum_i p_i(k') exp(s2 * i k'.r_i)|^2
    best = None
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            peaks, ratios = [], []
            for ik in range(min(4, nk)):
                kc = kpts[ik] @ b_sc
                for ib in range(min(nband, 40)):
                    ws = []
                    for gi, kp in enumerate(kprim_list[ik]):
                        dG = kp - kc
                        cvec = np.zeros((len(pos_pr), cplx.shape[3]), dtype=complex)
                        for i, lst in enumerate(reps):
                            acc = np.zeros(cplx.shape[3], dtype=complex)
                            for (j, Ti) in lst:
                                acc += cplx[ik, ib, j] * np.exp(
                                    1j * s1 * float(dG @ (Ti @ lat_pr)))
                            cvec[i] = acc / max(len(lst), 1)
                        ph = np.exp(1j * s2 * (kp @ cart_pr.T))
                        ws.append(float(np.abs(np.sum(cvec * ph[:, None])) ** 2))
                    ws = np.array(ws)
                    sw = ws.sum()
                    if sw > 1e-12:
                        peaks.append(ws.max() / sw)
                    tw = float(w2[ik, ib].sum())
                    if tw > 1e-6:
                        ratios.append(sw / tw)
            p = float(np.mean(peaks))
            print(f"[conv] s1={s1:+d} s2={s2:+d} -> 单G主导度 {p:.4f}  "
                  f"归一化 {np.mean(ratios):.4f}")
            if best is None or p > best[0]:
                best = (p, s1, s2)
    _, S1, S2 = best
    print(f"[conv] 采用 s1={S1:+d}, s2={S2:+d} (单G主导度最高)")

    # 正式展开
    rows = []
    for ik in range(nk):
        kc = kpts[ik] @ b_sc
        for ib in range(nband):
            E = ener[ik, ib]
            if args.emax is not None and E > args.emax:
                continue
            if args.emin is not None and E < args.emin:
                continue
            for gi, kp in enumerate(kprim_list[ik]):
                dG = kp - kc
                # 每个原胞原子、每个轨道的复投影 (副本平均 + 相位还原)
                cvec = np.zeros((len(pos_pr), cplx.shape[3]), dtype=complex)
                for i, lst in enumerate(reps):
                    acc = np.zeros(cplx.shape[3], dtype=complex)
                    for (j, Ti) in lst:
                        acc += cplx[ik, ib, j] * np.exp(
                            1j * S1 * float(dG @ (Ti @ lat_pr)))
                    cvec[i] = acc / max(len(lst), 1)
                # Bloch character: 对 (原子, 轨道) 求和, 原子位置给相位
                ph = np.exp(1j * S2 * (kp @ cart_pr.T))          # (natoms,)
                w = float(np.abs(np.sum(cvec * ph[:, None])) ** 2)
                rows.append((kp[0], kp[1], kp[2], E, w))
    arr = np.array(rows)
    print(f"[unfold] {len(rows)} 条 (k, E, w)")

    # 谱守恒自检
    sums = []
    for ik in range(min(6, nk)):
        for ib in range(nband):
            sel = arr[np.isclose(arr[:, 3], ener[ik, ib])]
            if len(sel) == ndet:
                sums.append(sel[:, 4].sum())
    if sums:
        s = np.array(sums)
        print(f"[check] sum_G w: mean={s.mean():.4f} median={np.median(s):.4f} "
              f"min={s.min():.4f} max={s.max():.4f}")

    ef = args.ef if args.ef is not None else read_ef(os.path.join(d, "DOSCAR"))
    print(f"[ef] E_F = {ef:.4f} eV")

    out = os.path.join(args.outdir, "unfolded_bands.dat")
    with open(out, "w") as fh:
        fh.write("# kx ky kz (primitive BZ, 2pi/A)  E(eV)  weight\n")
        for r in arr:
            fh.write(f"{r[0]:.8f} {r[1]:.8f} {r[2]:.8f} {r[3]:.6f} {r[4]:.8f}\n")
    print(f"[write] {out}")

    # 谱函数
    emin = float(arr[:, 3].min()) - 5 * args.sigma
    emax = float(arr[:, 3].max()) + 5 * args.sigma
    nE = 1600
    Egrid = np.linspace(emin, emax, nE)
    kk = np.unique(np.round(arr[:, 0:3], 6), axis=0)
    kidx = {tuple(k): i for i, k in enumerate(kk)}
    A = np.zeros((len(kk), nE))
    for r in arr:
        i = kidx[tuple(np.round(r[0:3], 6))]
        j = int(np.argmin(np.abs(Egrid - r[3])))
        A[i, j] += r[4]
    from scipy.ndimage import gaussian_filter1d
    A = gaussian_filter1d(A, args.sigma / (Egrid[1] - Egrid[0]), axis=1, mode="nearest")
    sp = os.path.join(args.outdir, "spectral_function.dat")
    with open(sp, "w") as fh:
        fh.write("# kx ky kz  E(eV)  A(k,E)\n")
        for i, k in enumerate(kk):
            for j, E in enumerate(Egrid):
                if A[i, j] > 1e-4:
                    fh.write(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f} {E:.6f} {A[i,j]:.8f}\n")
    print(f"[write] {sp}")

    jf = int(np.argmin(np.abs(Egrid - ef)))
    aef = A[:, jf]
    print(f"[FS] A(k,E_F): max={aef.max():.4f}, >0.1 的点数 {(aef>0.1).sum()}, "
          f">0.5 的点数 {(aef>0.5).sum()}")
    af = os.path.join(args.outdir, "fermi_surface_map.dat")
    with open(af, "w") as fh:
        fh.write("# kx ky kz  A(k,E_F)\n")
        for i, k in enumerate(kk):
            fh.write(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f} {aef[i]:.8f}\n")
    print(f"[write] {af}")

    # 二维费米面图 A(k, E_F)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x, y = kk[:, 0], kk[:, 1]
        lim = max(abs(x).max(), abs(y).max()) * 1.03
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        sc = ax.tricontourf(x, y, aef, levels=20, cmap="inferno")
        ax.tricontour(x, y, aef, levels=20, colors="w", linewidths=0.3, alpha=0.7)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$A(\mathbf{k}, E_F)$")
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$k_x$ (2$\pi$/$\AA$)")
        ax.set_ylabel(r"$k_y$ (2$\pi$/$\AA$)")
        ax.set_title("Unfolded Fermi surface (spectral function at $E_F$)", fontsize=10)
        fp = os.path.join(args.outdir, "fermi_surface_map.pdf")
        fig.savefig(fp, dpi=400, bbox_inches="tight")
        fig.savefig(fp.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"[write] {fp}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 绘图失败: {exc}")

    with open(os.path.join(args.outdir, "UNFOLD_GEOM.json"), "w") as fh:
        json.dump({"M": M.tolist(), "det": ndet,
                   "supercell_lattice": lat_sc.tolist(),
                   "primitive_lattice": lat_pr.tolist(),
                   "primitive_reciprocal": b_pr.tolist(),
                   "supercell_reciprocal": b_sc.tolist()}, fh, indent=2)


if __name__ == "__main__":
    main()
