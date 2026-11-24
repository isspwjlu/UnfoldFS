"""协变性缺陷的归因: 逐操作 δ(R) × "该操作是否保持超胞晶格"。

论文 §II.B 的证据表。做法:

1. 对原胞点群的**每一个**操作 R, 判定它是否**保持超胞晶格**
   (判据 R_sc = M^-T R M^T 为整数矩阵, 与 geometry.common_symmetry_ops_detailed
   同口径; 平移用 t_sc = M^-T t)。
   —— 不保持 ⇒ 对称相关的原胞动量被折到**不同的**超胞动量集合上,
      展开谱必然不协变(**任何**展开方法都如此)。
2. 再对照该操作的 δ(R) (A(k,E_F) 在 R 下的相对离散度)。
   —— 保持超胞晶格的操作下 δ 仍不为 0 ⇒ 缺陷来自**代表原子平均**,
      是**本实现特有**的。

判决性算例: 1T-TaS2 的 √3×√3 R30 超胞, 空间群与原胞相同 ⇒ 12 个操作
**全部**保持超胞晶格(折叠完全协变), 而 δ(R) 仍达 0.99 ⇒ 该算例的缺陷
100% 来自权重构造。

另外还给出两个超胞的**点群交集**(CDW 态 vs pristine √3×√3), 用于
论文 §VI.C: CDW 态自己就破坏了原胞 12 个操作中的 6 个 ⇒ CDW 情形下
δ(R) 必须对着**超胞的点群**读。

用法:
    python scripts/analysis/covariance_attribution.py \
        --prim  .fsdata/TaS2_PRIMCELL.vasp \
        --npz   .fsdata/tas2_fs96_scsqrt3.npz \
        --matrix "2 1 0;-1 1 0;0 0 1" \
        --supercell-poscar <超胞 POSCAR>          # 可选: 算点群交集
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from unfoldfs.fs import (load_spectral_npz, spectral_at_ef_from_bands,  # noqa: E402
                         rotation_inverse)
from unfoldfs.geometry import primitive_symmetry_ops, read_poscar  # noqa: E402

Z = {"Ta": 73, "S": 16, "Nb": 41, "Se": 34, "Mo": 42, "V": 23}


def parse_matrix(text: str) -> np.ndarray:
    """'3 0 0;0 1 0;0 0 1' 或多行写法 -> 3x3 矩阵。"""
    rows = []
    for chunk in text.replace("\n", ";").split(";"):
        vals = chunk.split()
        if vals:
            rows.append([float(x) for x in vals])
    return np.array(rows, dtype=float)


def per_op_delta(field: np.ndarray, ops) -> np.ndarray:
    """每个操作的 δ(R) = max_k |A(k) - A(Rk)| / max_k A(k)。"""
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)], 1)
    vals = field.ravel()
    mx = float(np.nanmax(field))
    out = []
    for r in np.asarray(ops.rotations):
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        out.append(float(np.abs(vals - vals[i2 * n2 + j2]).max() / mx))
    return np.array(out)


def preserves_supercell_lattice(ops, M: np.ndarray) -> np.ndarray:
    """R_sc = M^-T R M^T 为整数矩阵 <=> R 保持超胞晶格。"""
    minv_t = np.linalg.inv(M).T
    flags = []
    for r in np.asarray(ops.rotations):
        r_sc = minv_t @ np.asarray(r, float) @ M.T
        flags.append(bool(np.allclose(r_sc, np.round(r_sc), atol=1e-6)))
    return np.array(flags)


def point_group(st) -> str:
    import spglib
    cell = (np.asarray(st.lattice), np.asarray(st.positions),
            np.array([Z[x] for x in st.species]))
    d = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    return "%s (%d 操作)" % (d.international, len(d.rotations))


def space_group(st) -> tuple:
    """(国际符号, 操作数)。点群包含关系由符号直接读出 (P-3 ⊂ P-3m1)。"""
    import spglib
    cell = (np.asarray(st.lattice), np.asarray(st.positions),
            np.array([Z[x] for x in st.species]))
    d = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    return d.international, len(d.rotations)


def point_group(st) -> str:
    sym, n = space_group(st)
    return "%s (%d 操作)" % (sym, n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--npz", required=True, help="展开后的 A(k,E).npz")
    ap.add_argument("--matrix", required=True,
                    help="超胞变换矩阵, 例如 '3 0 0;0 1 0;0 0 1'")
    ap.add_argument("--supercell-poscar", default=None,
                    help="超胞 POSCAR (给则额外算点群交集)")
    ap.add_argument("--label", default="", help="算例名")
    ap.add_argument("--compare-npz", default=None,
                    help="第二个 A(k,E).npz: 与第一个**逐操作对照**打印 "
                         "(例如 pristine vs CDW, 同一超胞同一赝势)")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    ops = primitive_symmetry_ops(prim, 1e-4)
    M = parse_matrix(args.matrix)
    s = load_spectral_npz(args.npz)
    field = spectral_at_ef_from_bands(s.band_e, s.band_w, s.valid,
                                       s.e_grid, s.e_f, s.sigma)
    d = per_op_delta(field, ops)
    keep = preserves_supercell_lattice(ops, M)

    print("### %s" % (args.label or args.npz))
    print("    原胞点群 %s, 超胞网格 %dx%d, det(M)=%g"
          % (point_group(prim), s.n1, s.n2, abs(np.linalg.det(M))))
    print("    op   delta(R)      保持超胞晶格")
    for i in range(len(d)):
        print("     op%-2d  %.4e      %s" % (i, d[i], "是" if keep[i] else "否"))
    print("    保持超胞晶格的操作 n=%d: delta = %s"
          % (int(keep.sum()), np.round(d[keep], 6)))
    if (~keep).any():
        print("    其余 n=%d: delta max=%.4f min=%.4f"
              % (int((~keep).sum()), d[~keep].max(), d[~keep].min()))
    if args.compare_npz:
        s2 = load_spectral_npz(args.compare_npz)
        f2 = spectral_at_ef_from_bands(s2.band_e, s2.band_w, s2.valid,
                                       s2.e_grid, s2.e_f, s2.sigma)
        d2 = per_op_delta(f2, ops)
        keep2 = preserves_supercell_lattice(ops, M)
        print("    --- 对照: %s ---" % args.compare_npz)
        print("    op   delta(算例1)  delta(算例2)   保持超胞晶格")
        for i in range(len(d)):
            print("     op%-2d  %.4e      %.4e      %s"
                  % (i, d[i], d2[i], "是" if keep2[i] else "否"))
        print("    保持超胞晶格的操作: 算例1 delta=%s"
              % np.round(d[keep2], 4))
        print("                       算例2 delta=%s"
              % np.round(d2[keep2], 4))
        if (~keep2).any():
            print("    其余操作:           算例1 max=%.4f / 算例2 max=%.4f"
                  % (d[~keep2].max(), d2[~keep2].max()))
        else:
            print("    其余操作: (无 —— 所有操作都保持超胞晶格)")

    if args.supercell_poscar:
        sc = read_poscar(args.supercell_poscar)
        sym_sc, n_sc = space_group(sc)
        sym_pc, n_pc = space_group(prim)
        print("    原胞空间群 %s (%d 操作)   超胞空间群 %s (%d 操作)"
              % (sym_pc, n_pc, sym_sc, n_sc))
        if n_sc < n_pc:
            print("    => 超胞破坏原胞点群的 %d 个操作(态本身少了对称性)"
                  % (n_pc - n_sc))


if __name__ == "__main__":
    main()
