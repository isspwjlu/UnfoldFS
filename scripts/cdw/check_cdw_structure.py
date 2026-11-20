"""检查 √3×√3 CDW 超胞的结构(弛豫前后都能用)。

报告三件事:
1. **三聚化**: 三个 Ta 之间的键长(CDW 态下应当"一长两短", 即三聚体收缩);
2. **S 的位移**: 相对于**未畸变**的 √3×√3 位置, 每个 S 移了多少;
3. **三重对称**: 三个 Ta 的位移模长是否相等(CDW 态应保持 C3)。

用法:
  python check_cdw_structure.py --poscar POSCAR [--ideal sc_sqrt3.vasp]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--ideal", default="", help="未畸变的 √3×√3 POSCAR(给位移用)")
    args = ap.parse_args()

    st = read_poscar(args.poscar)
    lat = st.lattice
    cart = st.positions @ lat
    ta = [i for i, s in enumerate(st.species) if s == "Ta"]
    s_idx = [i for i, s in enumerate(st.species) if s == "S"]

    print(f"晶格: a1={np.round(lat[0],4)}  a2={np.round(lat[1],4)}")
    print(f"  |a1|={np.linalg.norm(lat[0]):.4f}  |a2|={np.linalg.norm(lat[1]):.4f}  "
          f"c={lat[2,2]:.4f}")

    print("\nTa-Ta 键长 (A):")
    for i in range(len(ta)):
        for j in range(i + 1, len(ta)):
            d = float(np.linalg.norm(cart[ta[i]] - cart[ta[j]]))
            print(f"   Ta{i}-Ta{j}: {d:.4f}")

    if args.ideal:
        ref = read_poscar(args.ideal)
        ref_cart = ref.positions @ ref.lattice
        d = np.linalg.norm(cart - ref_cart, axis=1)
        print(f"\n相对未畸变结构的位移 (A):")
        for i, s in enumerate(st.species):
            print(f"   {s:2s} #{i}: {d[i]:.4f}")
        print(f"   S 位移范围: {d[s_idx].min():.4f} - {d[s_idx].max():.4f} A")
        print(f"   Ta 位移范围: {d[ta].min():.4f} - {d[ta].max():.4f} A")


if __name__ == "__main__":
    main()
