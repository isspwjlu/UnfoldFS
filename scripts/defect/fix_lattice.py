"""用参考 POSCAR 的晶格 + CONTCAR 的离子位置, 重建一个 POSCAR。

为什么需要: 缺陷超胞弛豫必须**固定晶格** (ISIF=2), 否则晶格一变,
超胞与原胞的平移关系 (TRANSMAT) 就不再自洽 —— `unfoldfs` 的
`resolve_transform` 要求 `M @ PRIMCELL.lattice == SUPERCELL.lattice`
(容差 1e-4), `build_atom_copies` 的整数平移判据也会随之失效。

用法:
  python fix_lattice.py --ref POSCAR --contcar CONTCAR --out POSCAR_fixed

会打印晶格的偏差 (A), 供核对。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考 POSCAR (提供晶格与物种顺序)")
    ap.add_argument("--contcar", required=True, help="弛豫后的 CONTCAR (提供离子位置)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = read_poscar(args.ref)
    con = read_poscar(args.contcar)
    if len(ref.species) != len(con.species):
        raise SystemExit(f"原子数不一致: ref {len(ref.species)} vs "
                         f"con {len(con.species)}")
    if tuple(ref.species) != tuple(con.species):
        raise SystemExit("物种顺序不一致 —— VASP 的 CONTCAR 保持 POSCAR 的顺序, "
                         "请检查是不是用了不同的 POSCAR")

    dev = np.abs(con.lattice - ref.lattice).max()
    print(f"晶格最大偏差: {dev:.3e} A "
          f"({'OK' if dev < 1e-3 else '⚠ 已用参考晶格强制替换'})")

    # CONTCAR 的分数坐标是按它自己的晶格算的 -> 换成参考晶格下的分数坐标
    # ⚠ **不能用 np.mod(x, 1.0)**: 它把 -0.005 变成 0.995, 于是同一个物理位置
    #   (0 与 0.995 等价) 会被当成相差一个晶格常数 => 假的 6.4 A 位移。
    #   这里居中到 [-0.5, 0.5), 保持"离原点最近"的代表元。
    cart = con.positions @ con.lattice
    frac = cart @ np.linalg.inv(ref.lattice)
    frac = frac - np.round(frac)
    write_vasp(args.out, "relaxed ions on the reference lattice", ref.lattice,
               list(ref.species), frac)
    print(f"写好了: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
