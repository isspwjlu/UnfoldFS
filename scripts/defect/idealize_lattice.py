"""把结构文件的晶格"理想化"为理想六方晶格(保持分数坐标不变)。

为什么需要: VASP 弛豫后的晶格带 ~1e-5 A 的**数值倾斜**(应力有限差分的
噪声)。体相 1T-TaS2 的真实晶格是六方的, 倾斜不是物理。但它会让 spglib
在 symprec=1e-5 下**只识别出 4 个点群操作**而不是 12 个(实测: symprec>=1e-4
才恢复 12 个)⇒ 轨道平均只恢复部分对称性, 费米面的对称图案被压扁。

⚠ 原胞与超胞必须**一起**理想化: `unfoldfs` 的原子副本匹配要求超胞原子
  落在原胞原子的整数平移上, 容差只有 1e-6; 若只改原胞的晶格, 两边的
  笛卡尔位置会错开 ~1e-5 A ⇒ 匹配直接失败。
  做法: 原胞用理想六方晶格, 超胞用 M @ 原胞晶格, 两边都保持分数坐标不变
  (原子随晶格微移 ~1e-5 A, 相对 PROCAR 的物理内容可忽略)。

用法:
  python idealize_lattice.py --prim PRIMCELL.vasp \
      --supercell sc3x3x1_defect.vasp --transmat TRANSMAT.in \
      --out-prim PRIMCELL_ideal.vasp --out-super sc3x3x1_defect_ideal.vasp
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar, read_transmat, resolve_transform


def write_vasp(path: str, title: str, lattice: np.ndarray, species: list[str],
               frac: np.ndarray) -> None:
    groups: list[list] = []
    for sp in species:
        if groups and groups[-1][0] == sp:
            groups[-1][1] += 1
        else:
            groups.append([sp, 1])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(title + "\n1.0\n")
        for row in lattice:
            fh.write("  %.16f %.16f %.16f\n" % tuple(row))
        fh.write("  " + " ".join(g[0] for g in groups) + "\n")
        fh.write("  " + " ".join(str(g[1]) for g in groups) + "\n")
        fh.write("Direct\n")
        for row in frac:
            fh.write("  %.16f %.16f %.16f\n" % tuple(row))


def ideal_hexagonal(lattice: np.ndarray) -> np.ndarray:
    """由晶格矢量的长度与 c 分量构造理想六方晶格(120 度)。"""
    a = float(np.linalg.norm(lattice[0]))
    c = float(lattice[2][2])
    return np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, 0.5 * a * np.sqrt(3.0), 0.0],
        [0.0, 0.0, c],
    ])


def symmetrize_positions(frac: np.ndarray, species: list[str]) -> np.ndarray:
    """把 1T-TaS2 的原子位置对称化到严格的 D3d。

    VASP 弛豫后 S 的分数坐标会偏离理想的 (1/3, 2/3) 约 1e-5(数值噪声),
    刚好越过 spglib 的 symprec=1e-5 ⇒ 点群从 12 个操作掉到 4 个 ⇒
    轨道平均只恢复部分对称性。单层的 PRIMCELL.vasp 是精确的 1/3, 所以
    没有这个问题。

    这里把 Ta 放到 (0,0,0), 两个 S 放到 (1/3,2/3,z) 与 (2/3,1/3,1-z),
    z 取较小者(靠近 Ta 平面)。物理上这与弛豫结果的差别 < 1e-4 A。
    """
    out = np.array(frac, dtype=float)
    ta = [i for i, sp in enumerate(species) if sp == "Ta"]
    s = [i for i, sp in enumerate(species) if sp == "S"]
    if not ta or len(s) != 2:
        return out
    ta0 = ta[0]
    out[ta0] = 0.0
    z = min(out[s[0]][2], out[s[1]][2])
    z = z % 1.0
    if z > 0.5:
        z = 1.0 - z
    out[s[0]] = [1.0 / 3.0, 2.0 / 3.0, z]
    out[s[1]] = [2.0 / 3.0, 1.0 / 3.0, (1.0 - z) % 1.0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--supercell", required=True)
    ap.add_argument("--transmat", required=True)
    ap.add_argument("--out-prim", required=True)
    ap.add_argument("--out-super", required=True)
    ap.add_argument("--symmetrize-atoms", action="store_true",
                    help="同时把 1T 结构的原子位置对称化到严格 D3d")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    sup = read_poscar(args.supercell)
    tm = read_transmat(args.transmat)

    # 倾斜度量: 理想六方晶格下 a1 只沿 x, a2 在 xy 平面内, a3 沿 z
    ideal_probe = ideal_hexagonal(prim.lattice)
    tilt = float(np.abs(prim.lattice - ideal_probe).max())
    print(f"原胞晶格相对理想六方的最大偏差: {tilt:.3e} A")

    prim_ideal = ideal_hexagonal(prim.lattice)
    # 超胞晶格必须 = M @ 原胞晶格(逐位), 否则原子副本匹配会失败
    tr = resolve_transform(sup, prim, tm)
    sup_ideal = tr.m_eff @ prim_ideal

    prim_pos = prim.positions
    sup_pos = sup.positions
    if args.symmetrize_atoms:
        prim_pos = symmetrize_positions(prim_pos, list(prim.species))
        # 超胞的"理想位置" = 原胞理想位置 + 该原子所在的原胞平移
        # 超胞分数坐标 q 与原胞分数坐标 p 的关系: q @ M ≡ p (mod 1)
        # (与 unpack/unfold 的 k' = (K+g)·M^{-T} 对偶)
        sup_pos = np.array(sup_pos)
        for i, sp in enumerate(sup.species):
            f = sup_pos[i] @ tr.m_eff.astype(float)   # 原胞分数坐标
            shift = np.round(f)             # 原胞平移(整数)
            base = f - shift                # 原胞内的位置
            if sp == "S":
                z = base[2] % 1.0
                z = min(z, 1.0 - z)
                if abs(base[0] - 1 / 3) < abs(base[0] - 2 / 3):
                    ideal = np.array([1 / 3, 2 / 3, z])
                else:
                    ideal = np.array([2 / 3, 1 / 3, (1 - z) % 1.0])
                sup_pos[i] = (ideal + shift) % 1.0
            elif sp == "Ta":
                sup_pos[i] = (base + shift) % 1.0

    write_vasp(args.out_prim, "1T-TaS2 bulk primitive (ideal hexagonal lattice)",
               prim_ideal, list(prim.species), prim_pos)
    write_vasp(args.out_super, "3x3x1 supercell (ideal hexagonal lattice)",
               sup_ideal, list(sup.species), sup_pos)

    # 自检: 原子副本匹配必须仍然成立
    from unfoldfs.unfold import build_atom_copies
    p2 = read_poscar(args.out_prim)
    s2 = read_poscar(args.out_super)
    tr2 = resolve_transform(s2, p2, tm)
    counts = [len(c) for c in build_atom_copies(p2, s2, tr2).copies]
    print(f"理想化后: 每个原胞原子的副本数 = {counts} (总计 {sum(counts)})")
    print(f"超胞晶格 = M @ 原胞晶格, 偏差 "
          f"{np.abs(sup_ideal - tr2.m_eff @ p2.lattice).max():.3e}")
    print(f"写好了:\n   {args.out_prim}\n   {args.out_super}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
