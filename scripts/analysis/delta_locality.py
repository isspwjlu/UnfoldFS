"""δ(R) 的"局域性"诊断: 残差是弥散的还是集中在少数 k 上?

以及: 逐操作判定"该操作是否是**超胞晶体**的对称操作"(用 spglib 的空间群),
用来把"态的破缺"与"方法的残留"分开。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from unfoldfs.fs import (load_spectral_npz, rotation_inverse,  # noqa: E402
                         spectral_at_ef_from_bands)
from unfoldfs.geometry import primitive_symmetry_ops, read_poscar  # noqa: E402

Z = {"Ta": 73, "S": 16, "Nb": 41, "Se": 34, "Mo": 42, "V": 23}


def field_of(path: str) -> np.ndarray:
    s = load_spectral_npz(path)
    return spectral_at_ef_from_bands(s.band_e, s.band_w, s.valid,
                                     s.e_grid, s.e_f, s.sigma)


def space_group(st) -> tuple:
    import spglib
    cell = (np.asarray(st.lattice), np.asarray(st.positions),
            np.array([Z[x] for x in st.species]))
    d = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    return d.international, len(d.rotations), np.asarray(d.rotations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--supercell", default=None, help="超胞 POSCAR(判定态的破缺)")
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--frac-thresh", type=float, default=0.1,
                    help="超过 max 的这个比例才算'显著不同的点'")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    ops = primitive_symmetry_ops(prim, 1e-4)
    n1 = n2 = None

    sc_ops = None
    if args.supercell:
        sym, n_sc, sc_ops = space_group(read_poscar(args.supercell))
        print("超胞空间群 %s (%d 个操作); 原胞点群 %d 个操作"
              % (sym, n_sc, len(ops.rotations)))
    M = np.array([[float(x) for x in chunk.split()]
                  for chunk in args.matrix.replace("\n", ";").split(";")
                  if chunk.split()])
    minv_t = np.linalg.inv(M).T

    for path in args.npz:
        f = field_of(path)
        n1, n2 = f.shape
        idx = np.arange(n1 * n2)
        frac = np.stack([(idx // n2) / n1, (idx % n2) / n2,
                         np.zeros_like(idx, float)], 1)
        vals = f.ravel()
        mx = float(np.nanmax(f))
        print("\n### %s   (网格 %dx%d, A_max=%.4e)" % (pathlib.Path(path).name,
                                                       n1, n2, mx))
        print("  op   delta(R)     相对差中位   >1%%的点   >5%%的点   "
              "该操作是否超胞晶体的对称")
        for oi, r in enumerate(np.asarray(ops.rotations)):
            f2 = frac @ rotation_inverse(r)
            i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
            j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
            diff = np.abs(vals - vals[i2 * n2 + j2]) / mx
            d = float(diff.max())
            med = float(np.median(diff))
            f1 = float(np.mean(diff > 0.01))
            f5 = float(np.mean(diff > 0.05))
            tag = "-"
            if sc_ops is not None:
                # 原胞的操作 R 在**超胞基**下的表示是 R_sc = M^-T R M^T
                r_sc = minv_t @ np.asarray(r, float) @ M.T
                same = np.any(np.all(np.isclose(sc_ops, r_sc, atol=1e-6),
                                     axis=(1, 2)))
                tag = "是" if same else "否"
            print("  %-3d  %.4e     %.2e    %6.4f    %6.4f      %s"
                  % (oi, d, med, f1, f5, tag))


if __name__ == "__main__":
    main()
