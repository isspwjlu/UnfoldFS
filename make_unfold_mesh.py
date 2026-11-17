#!/usr/bin/env python3
"""为超胞反折叠费米面生成 (KPOINTS + KPOINTS_MAPPING_TABLE.in)。

原理
----
超胞 = M x 原胞 (TRANSMAT.in)。折叠关系(已用 vaspkit 官方算例核对)：

    k_prim_frac = (k_super_frac + G) * M^-1

其中 G 为整数矢量。固定一个超胞 k 点，有 |det M| 个候选 G，
对应原胞 BZ 内 |det M| 个不同 k 点——这正是谱函数
    A(k,E) = sum_G sum_n w(k+G,n) delta(E - E_n(k+G))
所需的全部信息。

输出
----
KPOINTS_mesh                  超胞 Gamma 中心网格(坐标=超胞分数坐标)
KPOINTS_MAPPING_TABLE_mesh.in 每个超胞 k 对应 |det M| 行映射
"""
from __future__ import annotations

import argparse
import itertools
import math
import os


def read_transmat(path: str) -> list[list[int]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 3:
                rows.append([int(float(x)) for x in parts])
    if len(rows) != 3:
        raise SystemExit(f"TRANSMAT.in 解析失败: {rows}")
    return rows


def read_poscar_lattice(path: str) -> list[list[float]]:
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    scale = float(lines[1].split()[0])
    lat = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    return [[v * scale for v in row] for row in lat]


def reciprocal(lattice: list[list[float]]) -> list[list[float]]:
    a = lattice
    cross = lambda u, v: [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
    b1 = cross(a[1], a[2])
    b2 = cross(a[2], a[0])
    b3 = cross(a[0], a[1])
    vol = sum(a[i][i] for i in range(3)) if False else (
        a[0][0] * b1[0] + a[0][1] * b1[1] + a[0][2] * b1[2])
    return [[v / vol for v in row] for row in (b1, b2, b3)]


def mat_mul_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def mat_inverse3(m):
    d = det3(m)
    c = [
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d,
         (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d,
         (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d,
         (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d,
         (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d,
         (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d,
         (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d],
    ]
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", default="POSCAR", help="超胞 POSCAR")
    ap.add_argument("--primcell", default="PRIMCELL.vasp", help="原胞 POSCAR")
    ap.add_argument("--transmat", default="TRANSMAT.in")
    ap.add_argument("--n1", type=int, default=16)
    ap.add_argument("--n2", type=int, default=16)
    ap.add_argument("--n3", type=int, default=1)
    ap.add_argument("--tag", default="mesh")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    M = read_transmat(args.transmat)
    Minv = mat_inverse3(M)
    ndet = abs(int(round(det3(M))))
    print(f"[M] = {M}   det = {det3(M):.0f}  -> 每超胞 k 对应 {ndet} 个原胞 k")

    lat_sc = read_poscar_lattice(args.poscar)
    lat_pr = read_poscar_lattice(args.primcell)
    b_sc = reciprocal(lat_sc)
    b_pr = reciprocal(lat_pr)
    print(f"[supercell] a = {lat_sc}")
    print(f"[primitive] a = {lat_pr}")

    # 候选整数 G: 满足 g·M^-1 ∈ [0,1)^3 (即把原胞 BZ 切成 |det M| 块的平移矢量)
    # 注意: G 必须取非负整数 —— VASPKIT 用定宽格式读该表, 负号会破坏列对齐
    gmax = [max(1, abs(M[i][j]) * 2 + 2) for i in range(3) for j in range(3)]
    gmax = [max(gmax), max(gmax), max(gmax)]
    Gset = []
    for G in itertools.product(range(0, gmax[0] + 1),
                               range(0, gmax[1] + 1),
                               range(0, gmax[2] + 1)):
        t = mat_mul_vec(Minv, list(G))
        if all(-1e-9 <= t[d] < 1.0 - 1e-9 for d in range(3)):
            Gset.append((list(G), t))
    if len(Gset) != ndet:
        raise SystemExit(f"候选 G 数 {len(Gset)} != det(M) = {ndet}")
    print(f"[G] {len(Gset)} 个候选: " + ", ".join(str(g[0]) for g in Gset))

    # --- 生成超胞 Gamma 中心网格 ---
    ks = []
    for i in range(args.n1):
        for j in range(args.n2):
            for l in range(args.n3):
                ks.append([i / args.n1, j / args.n2, l / args.n3])

    kpoints_file = os.path.join(args.outdir, f"KPOINTS_{args.tag}")
    with open(kpoints_file, "w") as fh:
        fh.write(f"Parameters to Generate KPOINTS (Don't Edit This Line):"
                 f"   0.040  0.040   0 {len(ks)}  {len(ks)}   0   0\n")
        fh.write(f"   {len(ks)}\n")
        fh.write("Reciprocal lattice\n")
        for k in ks:
            fh.write("   %.12f   %.12f   %.12f       1\n" % tuple(k))
    print(f"[write] {kpoints_file}  ({len(ks)} k 点)")

    # --- 映射表 (列宽与 VASPKIT 官方格式逐字节一致) ---
    table = os.path.join(args.outdir, f"KPOINTS_MAPPING_TABLE_{args.tag}.in")
    nrows = 0
    with open(table, "w") as fh:
        fh.write("   Kmesh for primitive cell                             "
                 "Kmesh for super cell                                 "
                 "G vectors for super cell\n")
        for ksuper in ks:
            for G, _t in Gset:
                kprim = mat_mul_vec(Minv, [ksuper[d] + G[d] for d in range(3)])
                fh.write("   %.12f   %.12f   %.12f     %.12f  %.12f   %.12f"
                         "     %d   %d   %d\n"
                         % (kprim[0], kprim[1], kprim[2],
                            ksuper[0], ksuper[1], ksuper[2],
                            G[0], G[1], G[2]))
                nrows += 1
    print(f"[write] {table}  ({nrows} 行 = {len(ks)} x {ndet})")

    # --- 标准 VASP 格式的网格 KPOINTS (供 vaspkit 26x 费米面模块使用) ---
    import math
    std = os.path.join(args.outdir, f"KPOINTS_{args.tag}_std")
    with open(std, "w") as fh:
        fh.write(f"Unfolded Fermi Surface mesh {args.n1}x{args.n2}x{args.n3}\n")
        fh.write(f"   {len(ks)}\n")
        fh.write("Reciprocal lattice\n")
        for row in b_sc:
            fh.write("   %18.12f %18.12f %18.12f   1.000000000000\n"
                     % (row[0] * 2 * math.pi, row[1] * 2 * math.pi, row[2] * 2 * math.pi))
        for k in ks:
            fh.write("   %.12f %.12f %.12f   1.000000000000\n" % tuple(k))
    print(f"[write] {std}  (标准 VASP 网格格式, 含超胞倒格矢)")

    # --- 供后处理使用的几何信息 ---
    geo = os.path.join(args.outdir, f"UNFOLD_GEOM_{args.tag}.json")
    import json
    with open(geo, "w") as fh:
        json.dump({
            "M": M,
            "M_inv": Minv,
            "det": ndet,
            "n_mesh": [args.n1, args.n2, args.n3],
            "supercell_lattice": lat_sc,
            "primitive_lattice": lat_pr,
            "supercell_reciprocal": b_sc,
            "primitive_reciprocal": b_pr,
        }, fh, indent=2)
    print(f"[write] {geo}")


if __name__ == "__main__":
    main()
