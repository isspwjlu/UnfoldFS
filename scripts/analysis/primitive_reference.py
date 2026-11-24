"""构造**权重 = 1** 的原胞直接计算谱函数(平面波路线的参考)。

平面波路线的谱权重是**严格**的谱权重: 每个超胞态的权重在候选之间归一化,
对单个候选求和等于 1 ⇒ 展开谱在原胞点 k' 上的权重应当与"原胞自己的带"
的权重(每条带 1)可比。轨道投影路线的权重不是这个定义(分波不完备 ⇒ 权重
只在单次计算内相对可比), 所以两条路线要各自配自己的直接参考。

网格与参考 npz 完全一致(用参考 npz 的 k_frac 作网格, 逐点找 EIGENVAL 里
最近的 k), 保证逐点可比。

用法:
    python scripts/analysis/primitive_reference.py \
        --eigenval  .fsdata/run96_prim_pw/EIGENVAL \
        --ref-npz   .fsdata/tas2_fs96_sc3x1x1_pw.npz \
        --prim      .fsdata/TaS2_PRIMCELL.vasp \
        --ef -1.501110 --out .fsdata/tas2_fs96_prim_pwref.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from unfoldfs.expand import ExpandedData                      # noqa: E402
from unfoldfs.fsout import write_spectral_npz                 # noqa: E402
from unfoldfs.geometry import reciprocal, read_poscar        # noqa: E402
from unfoldfs.grid import GridSpec                            # noqa: E402
from unfoldfs.vaspio import read_eigenval                     # noqa: E402


def _periodic_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    return np.linalg.norm(np.minimum(d, 1.0 - d), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eigenval", required=True)
    ap.add_argument("--ref-npz", required=True,
                    help="参考 npz(决定网格、点序、E 网格与 sigma)")
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--ef", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    ref = np.load(args.ref_npz)
    k_frac_ref = np.asarray(ref["k_frac"])
    e_grid = np.asarray(ref["e_grid"])
    sigma = float(ref["sigma"])

    band = read_eigenval(args.eigenval)
    kf = np.asarray(band.kpoints)
    en = np.asarray(band.energies)
    nb = en.shape[1]

    # 参考 npz 的网格点序: 直接用它的 k_frac
    n_points = k_frac_ref.shape[0]
    band_e = np.full((n_points, nb), np.inf)
    band_w = np.zeros((n_points, nb))
    valid = np.zeros((n_points, nb), dtype=bool)
    for i in range(n_points):
        d = _periodic_dist(kf, k_frac_ref[i])
        j = int(np.argmin(d))
        if d[j] > args.tol:
            raise ValueError("第 %d 个网格点在 EIGENVAL 的 k 列表里找不到"
                             "(最近距离 %.2e)" % (i, d[j]))
        band_e[i] = en[j]
        band_w[i] = 1.0
        valid[i] = True

    prim = read_poscar(args.prim)
    b_prim = reciprocal(prim.lattice)
    k_cart = k_frac_ref @ b_prim
    # 网格规格只用于元信息(npz 不存它); 展宽用的 e_grid 已经是参考 npz 的
    spec = GridSpec(p=(1, 1, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    expanded = ExpandedData(k_cart=k_cart, band_e=band_e, band_w=band_w,
                            valid=valid, spec=spec)
    write_spectral_npz(args.out, expanded, b_prim, args.ef,
                       vbm=float("nan"), cbm=float("nan"), e_grid=e_grid,
                       sigma=sigma)
    print("已写出 %s (%d 个网格点, %d 条带, 权重 = 1)"
          % (args.out, n_points, nb))


if __name__ == "__main__":
    main()
