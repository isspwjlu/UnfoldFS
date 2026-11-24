"""体相 1T-TaS2 的 3x3x1 超胞 + S 空位缺陷 (Ta9S16) 的结构生成。

背景: 论文需要"缺陷如何改变反折叠费米面"的算例 —— 用**体相** 1T-TaS2 的
3x3x1 超胞(9 Ta + 18 S), 随机去掉 2 个 S => Ta9S16 (25 原子)。

输出的文件:
  PRIMCELL.vasp          体相 1T-TaS2 原胞 (1 Ta + 2 S) —— 反折叠的目标原胞
  TRANSMAT.in            M = diag(3, 3, 1) (行约定: A_sc = M @ A_prim)
  sc3x3x1_perfect.vasp   完美 3x3x1 超胞 (27 原子) —— 对照组
  sc3x3x1_defect.vasp    缺陷 3x3x1 超胞 (25 原子) —— **理想位置**版:
                         既是 DFT 的初始结构, 也是反折叠做**原子对应**的
                         `--supercell` 文件(弛豫后晶格不变、只有离子动过,
                         严格匹配需要理想位置 —— 见 unfoldfs/unfold.py::
                         build_atom_copies)

为什么要"理想位置"做原子对应: 弛豫后离子偏离理想格点, 超胞原子不再位于
原胞原子的整数平移上, `build_atom_copies` 的整数性判据(_COPY_TOL = 1e-6)
会直接失败。反折叠的相位项用的是**原胞**原子的位置, 超胞位置只用于
"哪个超胞原子属于哪个原胞原子"的匹配 ⇒ 用理想位置匹配是物理上正确的
做法(与 CDW 算例用 IDEAL_SUPERCELL 同理)。

用法:
  python build_defect_tas2.py --out-dir <dir> [--seed 20261123]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar, read_transmat, resolve_transform
from unfoldfs.unfold import build_atom_copies

# 体相 1T-TaS2 的晶格常数 (A): 取自用户自己的 CDW 超胞反推值
# (scripts/cdw/build_bulk_cdw_prim.py: a = 3.36006, 超胞 c = 29.9106 = 5c)
DEFAULT_A = 3.36006
DEFAULT_C = 5.98212

# 初始 Ta-S 键长 (A): 只用来定 S 的初始 z 高度, 随后由 VASP 弛豫修正
DEFAULT_TA_S = 2.58

# 超胞变换 (行约定: A_sc = M @ A_prim)
M_SUPER = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=int)


def hex_lattice(a: float, c: float) -> np.ndarray:
    """六方晶格 (行 = 实矢量): a1=(a,0,0), a2=(-a/2, a*sqrt3/2,0), a3=(0,0,c)。"""
    return np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, 0.5 * a * np.sqrt(3.0), 0.0],
        [0.0, 0.0, c],
    ])


def z_from_bond_length(a: float, c: float, d_ta_s: float) -> float:
    """由 Ta-S 键长定 S 到 Ta 平面的高度 (A)。"""
    in_plane = a / np.sqrt(3.0)          # Ta 到 S 的面内投影
    h = np.sqrt(d_ta_s ** 2 - in_plane ** 2)
    return h


def write_vasp(path: str, title: str, lattice: np.ndarray, species: list[str],
               frac: np.ndarray) -> None:
    """写 VASP-5 格式 POSCAR (Direct 分数坐标, 物种按出现顺序分组)。"""
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


def build_primcell(a: float, c: float, z_s: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """体相 1T-TaS2 原胞: Ta 在 (0,0,0), S 在 (1/3,2/3,+z) 与 (2/3,1/3,-z)。

    (S 的两个子晶格上下关系按用户的 CDW 结构取 A 在下、B 在上。)
    """
    lat = hex_lattice(a, c)
    z = z_s / c
    frac = np.array([
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, z],
        [2.0 / 3.0, 1.0 / 3.0, (-z) % 1.0],
    ])
    species = ["Ta", "S", "S"]
    return lat, frac, species


def build_supercell(prim_lat: np.ndarray, prim_frac: np.ndarray,
                    prim_species: list[str], m: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """把原胞按 M 展开成超胞 (原子顺序: 先所有原胞平移, 每个平移内按原胞顺序)。

    ⚠ 平移必须以**原胞格矢**为单位: q_cart = p_cart + n1·a1 + n2·a2 (n ∈ {0,1,2})。
    用**分数坐标**平移是错的 —— 分数平移 (n1,n2) 对应的笛卡尔位移是
    (n1,n2)·A_sc = 3(n1,n2)·a_prim, 而超胞里的原胞副本只差 1 个 a_prim
    ⇒ 反折叠的原子副本匹配(build_atom_copies)会直接失败。

    平移取 m1, m2 ∈ {0..M-1} (面内), m3 = 0 (kz 方向不折叠)。
    """
    sc_lat = m @ prim_lat
    inv_sc = np.linalg.inv(sc_lat)
    prim_cart = prim_frac @ prim_lat
    shifts = []
    for i in range(m[0, 0]):
        for j in range(m[1, 1]):
            shifts.append(i * prim_lat[0] + j * prim_lat[1])
    frac_all, species_all = [], []
    for s in shifts:
        for p, sp in zip(prim_cart, prim_species):
            frac_all.append(np.mod((p + s) @ inv_sc, 1.0))
            species_all.append(sp)
    frac_all = np.array(frac_all)
    # 按物种分组排序 (Ta 在前, S 在后): POSCAR 的元素计数行写成 "9 16",
    # POTCAR 只需拼接 Ta_pv + S 两段; 也让缺陷版与完美版的原子编号一一对应
    order = ([i for i, sp in enumerate(species_all) if sp == "Ta"]
             + [i for i, sp in enumerate(species_all) if sp == "S"])
    return sc_lat, frac_all[order], [species_all[i] for i in order]


def symmetrize_1t(frac: np.ndarray, species: list[str]) -> np.ndarray:
    """把 1T-TaS2 的原子位置对称化到严格的 D3d。

    VASP 弛豫后 S 的分数坐标会偏离理想的 (1/3, 2/3) 约 1e-5(数值噪声),
    刚好越过 spglib 的 symprec=1e-5 ⇒ 点群从 12 个操作掉到 4 个 ⇒
    轨道平均只恢复部分对称性(费米面变得不对称)。单层的 PRIMCELL.vasp
    是精确的 1/3, 所以没有这个问题。

    Ta 放 (0,0,0), 两个 S 放 (1/3,2/3,z) 与 (2/3,1/3,1-z)(z 取较小者)。
    与弛豫结果的差别 < 1e-4 A。
    """
    out = np.array(frac, dtype=float)
    ta = [i for i, sp in enumerate(species) if sp == "Ta"]
    s = [i for i, sp in enumerate(species) if sp == "S"]
    if not ta or len(s) != 2:
        return out
    out[ta[0]] = 0.0
    z = min(out[s[0]][2] % 1.0, out[s[1]][2] % 1.0)
    z = z if z <= 0.5 else 1.0 - z
    out[s[0]] = [1.0 / 3.0, 2.0 / 3.0, z]
    out[s[1]] = [2.0 / 3.0, 1.0 / 3.0, (1.0 - z) % 1.0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a", type=float, default=DEFAULT_A)
    ap.add_argument("--c", type=float, default=DEFAULT_C)
    ap.add_argument("--ta-s", type=float, default=DEFAULT_TA_S)
    ap.add_argument("--n-vac", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--from-primcell", default=None,
                    help="直接用一个已弛豫的原胞 POSCAR 作原胞 "
                         "(晶格与原子位置全取自它, 忽略 --a/--c/--ta-s)")
    ap.add_argument("--symmetrize-atoms", action="store_true",
                    help="把原胞原子位置对称化到严格 D3d (VASP 弛豫后 S 的 "
                         "分数坐标会偏离 1/3 约 1e-5, 恰好越过 spglib 的 "
                         "symprec=1e-5 ⇒ 点群从 12 个操作掉到 4 个)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.from_primcell:
        st = read_poscar(args.from_primcell)
        prim_lat = st.lattice
        # VASP 把弛豫后的原子写成 (1,1,1) 这类等价坐标 -> 折回胞内
        prim_frac = np.mod(st.positions, 1.0)
        prim_species = list(st.species)
        a = float(np.linalg.norm(prim_lat[0]))
        c = float(prim_lat[2, 2])
        print(f"体相 1T-TaS2 原胞取自 {args.from_primcell}: "
              f"a = {a:.5f} A, c = {c:.5f} A, {len(prim_species)} 原子")
    else:
        a, c = args.a, args.c
        z_s = z_from_bond_length(args.a, args.c, args.ta_s)
        print(f"体相 1T-TaS2: a = {args.a:.5f} A, c = {args.c:.5f} A, "
              f"Ta-S = {args.ta_s:.3f} A -> S 的面间距 {z_s:.4f} A "
              f"(z_frac = {z_s / args.c:.5f})")
        prim_lat, prim_frac, prim_species = build_primcell(args.a, args.c, z_s)
    if args.symmetrize_atoms:
        prim_frac = symmetrize_1t(prim_frac, prim_species)
        print("原胞原子位置已对称化到严格 D3d (S 精确在 1/3, 2/3)")
    sc_lat, sc_frac, sc_species = build_supercell(
        prim_lat, prim_frac, prim_species, M_SUPER)

    p_prim = os.path.join(args.out_dir, "PRIMCELL.vasp")
    p_tm = os.path.join(args.out_dir, "TRANSMAT.in")
    p_perf = os.path.join(args.out_dir, "sc3x3x1_perfect.vasp")
    p_def = os.path.join(args.out_dir, "sc3x3x1_defect.vasp")

    write_vasp(p_prim, "1T-TaS2 bulk primitive (1 Ta + 2 S)", prim_lat,
               prim_species, prim_frac)
    write_vasp(p_perf, "1T-TaS2 bulk 3x3x1 supercell (perfect, 27 atoms)",
               sc_lat, sc_species, sc_frac)

    with open(p_tm, "w", encoding="utf-8") as fh:
        fh.write("Read transformation matrix from the TRANSMAT.in file if it exists.\n")
        for row in M_SUPER:
            fh.write("  %d  %d  %d   # must be three integers\n" % tuple(row))

    # ---- 挑空位 ----
    rng = np.random.default_rng(args.seed)
    s_idx = [i for i, sp in enumerate(sc_species) if sp == "S"]
    vac = sorted(rng.choice(s_idx, size=args.n_vac, replace=False).tolist())
    keep = [i for i in range(len(sc_species)) if i not in set(vac)]
    def_frac = sc_frac[keep]
    def_species = [sc_species[i] for i in keep]
    write_vasp(p_def, "1T-TaS2 bulk 3x3x1 supercell with 2 S vacancies "
                       "(Ta9S16, 25 atoms, ideal positions)", sc_lat,
               def_species, def_frac)

    n_ta = sum(1 for sp in def_species if sp == "Ta")
    n_s = sum(1 for sp in def_species if sp == "S")
    print(f"\n缺陷超胞: Ta {n_ta} + S {n_s} = {len(def_species)} 原子 "
          f"(化学式 Ta{n_ta}S{n_s})")

    # ---- 诊断: 空位的位置与间距 ----
    cart = sc_frac @ sc_lat
    print(f"空位 (seed={args.seed}):")
    for i in vac:
        f = sc_frac[i]
        # 折回原胞分数坐标判 S 子晶格: A = (1/3,2/3), B = (2/3,1/3)
        fp = f @ M_SUPER
        subl = "A" if abs(fp[0] - 1 / 3) < 0.1 else "B"
        print(f"  #{i}: 超胞 frac = ({f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f})  "
              f"原胞 frac = ({fp[0] % 1:.4f}, {fp[1] % 1:.4f}, {fp[2]:.4f})  "
              f"S 子晶格 {subl}  z = {f[2] * c:.3f} A")
    if len(vac) == 2:
        d = np.linalg.norm(cart[vac[0]] - cart[vac[1]])
        # 最短镜像距离 (面内 3x3 周期)
        best = d
        for m1 in (-1, 0, 1):
            for m2 in (-1, 0, 1):
                shift = m1 * sc_lat[0] + m2 * sc_lat[1]
                best = min(best, np.linalg.norm(
                    cart[vac[0]] - cart[vac[1]] + shift))
        print(f"  两空位的最短周期距离 = {best:.3f} A")

    # ---- 自检: unfoldfs 的原子对应 ----
    st_prim = read_poscar(p_prim)
    st_perf = read_poscar(p_perf)
    st_def = read_poscar(p_def)
    tr = resolve_transform(st_perf, st_prim, read_transmat(p_tm))
    print(f"\nM = {tr.m_eff.tolist()}  det = "
          f"{int(round(abs(np.linalg.det(tr.m_eff.astype(float)))))}")

    for name, st in (("完美超胞", st_perf), ("缺陷超胞", st_def)):
        copies = build_atom_copies(st_prim, st, tr)
        counts = [len(c) for c in copies.copies]
        print(f"  {name}: 每个原胞原子的副本数 = {counts} "
              f"(总计 {sum(counts)} = 9 Ta + 2x9 S = 27 的缺陷版少 {27 - sum(counts)})")

    print(f"\n写好了:\n   {p_prim}\n   {p_tm}\n   {p_perf}\n   {p_def}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
