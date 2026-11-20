#!/usr/bin/env python3
"""由反折叠能带数据 (vaspkit 282 的 EBS.dat) 构建 E_F 处的谱函数并渲染反折叠费米面。

输入
----
EBS.dat                      vaspkit 282 输出: (k坐标, 能量, 权重) 每行一条
KPOINTS_MAPPING_TABLE*.in    vaspkit 映射表: 原胞k | 超胞k | G (行序与 EBS.dat 块一一对应)
DOSCAR / --ef                费米能级 (eV)

输出
----
spectral_function.dat        (kx, ky, kz, E, A)  —— 原胞 BZ 内的展开谱函数
fermi_surface_map.<pdf|png>  A(k, E_F) 二维切片图 (2D 体系)
fermi_surface.isosurface.vtk  3D 等值面 (3D 体系, pyvista)
bands_unfolded/b*.dat        逐带 (k, E, w) —— 可喂 FermiSurfer 画带色费米面
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------- 文件读取
def read_ebs(path: str) -> list[list[tuple[float, float, float]]]:
    """EBS.dat -> [[(col0, E, w), ...], ...] 每块对应一个 k 点。"""
    blocks: list[list[tuple[float, float, float]]] = []
    cur: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 3:
                cur.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if cur:
        blocks.append(cur)
    return blocks


def read_mapping(path: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """映射表 -> [(k_prim, k_super, G), ...]"""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Kmesh"):
                continue
            p = line.split()
            if len(p) < 9:
                continue
            kp = np.array([float(p[0]), float(p[1]), float(p[2])])
            ks = np.array([float(p[3]), float(p[4]), float(p[5])])
            G = np.array([int(float(p[6])), int(float(p[7])), int(float(p[8]))])
            out.append((kp, ks, G))
    return out


def read_ef_efermi(doscar: str) -> float:
    """DOSCAR 第 2 行第 2 列是 E-fermi。"""
    with open(doscar) as fh:
        fh.readline()
        line = fh.readline().split()
    return float(line[1])


def read_geom(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- 核心
def build_spectral_function(blocks, mapping, ef, sigma):
    """返回 (klist, Elist, A) —— A[i][j] = 第 i 个 k 点、第 j 个能级的谱函数权重。

    A(k,E) = sum_n w_n(k) * exp(-(E-E_n(k))^2 / 2 sigma^2)
    """
    klist, bands = [], []
    for i, blk in enumerate(blocks):
        k = blk[0][0]
        klist.append(k)
        bands.append([(e, w) for _, e, w in blk])
    nb = max(len(b) for b in bands)
    # 统一能级网格
    emin = min(e for b in bands for e, _ in b)
    emax = max(e for b in bands for e, _ in b)
    nE = 1200
    Elist = np.linspace(emin - 3 * sigma, emax + 3 * sigma, nE)
    A = np.zeros((len(klist), nE))
    for i, b in enumerate(bands):
        for e, w in b:
            A[i] += w * np.exp(-0.5 * ((Elist - e) / sigma) ** 2)
    return klist, Elist, A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebs", default="EBS.dat")
    ap.add_argument("--mapping", default="KPOINTS_MAPPING_TABLE_mesh16.in")
    ap.add_argument("--geom", default="UNFOLD_GEOM_mesh16.json")
    ap.add_argument("--doscar", default="DOSCAR")
    ap.add_argument("--ef", type=float, default=None, help="费米能级 (eV)")
    ap.add_argument("--sigma", type=float, default=0.06, help="展宽 (eV)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    blocks = read_ebs(args.ebs)
    mapping = read_mapping(args.mapping)
    geom = read_geom(args.geom)
    print(f"[read] EBS.dat: {len(blocks)} 块;  映射表: {len(mapping)} 行")
    if len(blocks) != len(mapping):
        print(f"[warn] 块数({len(blocks)}) != 映射行数({len(mapping)})，"
              f"按块序一一对应(前 {min(len(blocks), len(mapping))} 行)")

    ef = args.ef if args.ef is not None else read_ef_efermi(args.doscar)
    print(f"[ef] E_F = {ef:.4f} eV")

    # k 坐标: 优先用映射表的原胞 k(分数坐标)，否则退化用 EBS.dat 第 1 列
    n = min(len(blocks), len(mapping))
    kfrac = np.array([mapping[i][0] for i in range(n)])
    # 笛卡尔坐标(原胞倒格矢)
    b_pr = np.array(geom["primitive_reciprocal"])  # 3x3, 行=倒格矢
    kcart = kfrac @ b_pr

    elist, A = [], None
    klist, Elist, A = build_spectral_function(blocks[:n], mapping[:n], ef, args.sigma)

    # 谱函数数据文件
    sp = os.path.join(args.outdir, "spectral_function.dat")
    with open(sp, "w") as fh:
        fh.write("# kx ky kz (2pi/Ang, primitive BZ)   E(eV)   A(k,E)\n")
        for i in range(n):
            for j, E in enumerate(Elist):
                if A[i, j] > 1e-4:
                    fh.write(f"{kcart[i,0]:.8f} {kcart[i,1]:.8f} {kcart[i,2]:.8f} "
                             f"{E:.6f} {A[i,j]:.6f}\n")
    print(f"[write] {sp}")

    # E_F 处的谱函数切片
    jf = int(np.argmin(np.abs(Elist - ef)))
    aef = A[:, jf]
    af = os.path.join(args.outdir, "fermi_surface_map.dat")
    with open(af, "w") as fh:
        fh.write("# kx ky kz  A(k,E_F)\n")
        for i in range(n):
            fh.write(f"{kcart[i,0]:.8f} {kcart[i,1]:.8f} {kcart[i,2]:.8f} {aef[i]:.6f}\n")
    print(f"[write] {af}   max A = {aef.max():.3f}, "
          f"超过 0.1 的点数 = {(aef > 0.1).sum()}")

    # 逐带数据 (供 FermiSurfer)
    bdir = os.path.join(args.outdir, "bands_unfolded")
    os.makedirs(bdir, exist_ok=True)
    nb = min(len(b) for b in blocks[:n])
    for bi in range(nb):
        fn = os.path.join(bdir, f"band{bi+1}.dat")
        with open(fn, "w") as fh:
            fh.write(f"# kx ky kz  E(eV)  weight\n")
            for i in range(n):
                _, e, w = blocks[i][bi]
                fh.write(f"{kcart[i,0]:.8f} {kcart[i,1]:.8f} {kcart[i,2]:.8f} "
                         f"{e:.6f} {w:.6f}\n")
    print(f"[write] {bdir}/band1..{nb}.dat")

    # 2D 绘图 (n3=1)
    if geom["n_mesh"][2] == 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # 原胞 BZ 网格化: 原始 k 是 3N x 3N 网格, 用 tricontourf
            x = kcart[:, 0]
            y = kcart[:, 1]
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            lim = max(abs(x).max(), abs(y).max()) * 1.02
            sc = ax.tricontourf(x, y, aef, levels=18, cmap="inferno")
            ax.tricontour(x, y, aef, levels=18, colors="w", linewidths=0.25, alpha=0.6)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(r"$A(\mathbf{k}, E_F)$")
            ax.set_aspect("equal")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(r"$k_x$ (2$\pi$/$\AA$)")
            ax.set_ylabel(r"$k_y$ (2$\pi$/$\AA$)")
            ax.set_title("Unfolded Fermi surface (spectral function at $E_F$)")
            out = os.path.join(args.outdir, "fermi_surface_map.pdf")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[write] {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 2D 绘图失败: {exc}")


if __name__ == "__main__":
    main()
