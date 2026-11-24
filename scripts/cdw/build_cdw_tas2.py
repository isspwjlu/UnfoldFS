"""1T-TaS2 的 √3×√3 R30° CDW 超胞(反折叠算例用)。

⚠ **本脚本的 F3m 位移场有 bug, 生成的结构不是 C3 协变的**(2026-11-23 实测):
M 点星 {M1,M2,M3} **不对 -M 闭合**(M1+M2+M3=0, 但 -M1 不在星里),
而场写成 Σ_j Re[û_j exp(i M_j·r)] —— 取实部后得到的场不是 C3 协变的,
生成的空间群是 **P1**(连 C3 都没有) ⇒ 不能用来做"CDW 破坏哪些对称操作"的判据。
**要 C3 协变的三聚化, 用 `build_trimer_cdw.py`**(径向收缩模型, 空间群 P31m/C3v,
恰好破坏原胞 D3d 的 12 个操作中的 6 个)。


CDW 位移场
----------
√3×√3 的三阶星 {M1, M2, M3} (原胞倒格矢基下的三个 M 点):
    M1 = (b1+b2)/3,  M2 = (2b1-b2)/3,  M3 = (b1-2b2)/3
位移场取**三重协变**的矢量形式(F3m 型 CDW 的标准写法):
    u(R) = Σ_j Re[ δ · (C3^-j û) · exp(iθ) · exp(i M_j·(R - r0)) ]
  * C3^-j û: 振幅矢量随波矢一起转 ⇒ 场在 C3 下协变;
  * θ: 唯一的自由相位, 用来挑 CDW 态;
  * r0: 调制原点。取成 Ta 三聚体的中心 ⇒ Ta 往聚体中心收 ⇒ 三聚化。

上层的 S 与下层的 S 取**相反**的面内位移(1T 的 "SO(3)" 星形: 上下两层
反向旋转), 并各加一个反向的 z 分量。

用法:
  python build_cdw_tas2.py --poscar sc_sqrt3.vasp --out POSCAR_CDW \
      --delta-ta 0.10 --delta-s 0.15 --delta-z 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar, reciprocal

# 1T-TaS2 原胞(与 PRIMCELL.vasp 一致)
A1 = np.array([3.3649, 0.0, 0.0])
A2 = np.array([-1.68245, 2.9140888811942576, 0.0])
C = 17.6585355706663840
PRILAT = np.array([A1, A2, [0.0, 0.0, C]])


def rot3(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s], [s, c]])


def cdw_field(r_cart: np.ndarray, m_star, delta: float, theta: float,
              r0: np.ndarray) -> np.ndarray:
    """三重协变的 √3×√3 位移场(面内两分量)。"""
    u = np.zeros(2)
    for j, mj in enumerate(m_star):
        amp = rot3(-2 * np.pi * j / 3) @ np.array([1.0, 0.0])
        u += delta * np.real(amp * np.exp(1j * (mj[:2] @ (r_cart - r0)
                                                 + theta)))
    return u


def species_groups(species):
    """按顺序把 (元素符号, 个数) 压成组(VASP-5 的写法)。"""
    groups = []
    for s in species:
        if groups and groups[-1][0] == s:
            groups[-1][1] += 1
        else:
            groups.append([s, 1])
    return groups


def write_vasp(path: str, title: str, lattice: np.ndarray, groups,
                coords: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(title + "\n1.0\n")
        for row in lattice:
            fh.write("  %.16f %.16f %.16f\n" % tuple(row))
        fh.write("  " + " ".join(g[0] for g in groups) + "\n")
        fh.write("  " + " ".join(str(g[1]) for g in groups) + "\n")
        fh.write("Direct\n")
        for row in coords:
            fh.write("  %.16f %.16f %.16f\n" % tuple(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True, help="未畸变的 √3×√3 超胞 POSCAR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="1T-TaS2 sqrt3xsqrt3 R30 CDW")
    ap.add_argument("--delta-ta", type=float, default=0.10, help="Å")
    ap.add_argument("--delta-s", type=float, default=0.15, help="Å")
    ap.add_argument("--delta-z", type=float, default=0.05, help="Å")
    ap.add_argument("--theta", type=float, default=0.0, help="度")
    args = ap.parse_args()

    st = read_poscar(args.poscar)
    lattice = st.lattice
    coords = st.positions.copy()
    groups = species_groups(st.species)

    b = reciprocal(PRILAT)
    m_star = [(b[0] + b[1]) / 3, (2 * b[0] - b[1]) / 3, (b[0] - 2 * b[1]) / 3]

    ta_idx = [i for i, s in enumerate(st.species) if s == "Ta"]
    s_idx = [i for i, s in enumerate(st.species) if s == "S"]

    # Ta 三聚体(超胞分数坐标 (0,0), (1/3,2/3), (2/3,1/3))的中心
    ta_cart = coords[ta_idx] @ lattice
    center = ta_cart.mean(axis=0)[:2]
    theta = np.radians(args.theta)

    new = coords.copy()
    for i in s_idx:
        r = coords[i] @ lattice
        # 上层 S(z > 0.5c) 与下层 S(z < 0.5c) 反向
        sign = 1.0 if r[2] > 0.5 * lattice[2, 2] else -1.0
        u = sign * cdw_field(r[:2], m_star, args.delta_s, theta, center)
        new[i, 0] += u[0] / lattice[0, 0]
        new[i, 1] += u[1] / lattice[1, 1]
        new[i, 2] += sign * args.delta_z / lattice[2, 2]
    for i in ta_idx:
        r = coords[i] @ lattice
        u = cdw_field(r[:2], m_star, args.delta_ta, theta, center)
        new[i, 0] += u[0] / lattice[0, 0]
        new[i, 1] += u[1] / lattice[1, 1]

    write_vasp(args.out, args.title, lattice, groups, new)
    print("wrote", args.out)

    # ---- 几何自检 ----
    new_ta = new[ta_idx] @ lattice
    print("Ta-Ta 键长 (Å):")
    for i in range(len(ta_idx)):
        for j in range(i + 1, len(ta_idx)):
            d0 = float(np.linalg.norm(ta_cart[i] - ta_cart[j]))
            d1 = float(np.linalg.norm(new_ta[i] - new_ta[j]))
            print(f"   Ta{i}-Ta{j}: {d0:.4f} -> {d1:.4f} "
                  f"({(d1 - d0) / d0 * 100:+.1f}%)")
    print("Ta 位移 (Å):", np.round(new_ta - ta_cart, 4).tolist())
    disp_s = np.linalg.norm(new[s_idx] @ lattice - coords[s_idx] @ lattice,
                            axis=1)
    print(f"S 位移: {disp_s.min():.3f} - {disp_s.max():.3f} Å")


if __name__ == "__main__":
    main()
