"""体相 1T-TaS2 的 CDW 超胞(39 原子) -> 反折叠需要的三个文件。

输入: CDW 超胞的 POSCAR(13 Ta + 26 S, 与 1T 原胞的平移关系未知 —— CDW
把原胞平移对称性破掉了, 所以超胞**不是**原胞的整数倍超胞, 只是晶格相容)。

输出:
  * ``PRIMCELL.vasp``   —— 1T-TaS2 原胞(1 Ta + 2 S), 面内常数由超胞晶格反推;
  * ``TRANSMAT.in``     —— 超胞 = M @ 原胞 的整数变换(行约定);
  * ``IDEAL_SUPERCELL.vasp`` —— **未畸变**的 39 原子超胞(原子顺序与 CDW POSCAR
    一致)。反折叠时用它做**原子对应**: CDW 把原子挪开了, 原胞原子不再位于超胞
    原子的整数平移上, 严格匹配会失败(见 unfoldfs/unfold.py::build_atom_copies)。

用法:
  python build_bulk_cdw_prim.py --cdw-poscar POSCAR --out-dir <dir>
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from unfoldfs.geometry import read_poscar, resolve_transform


def find_M(A: np.ndarray, det_target: int, ta_frac=None):
    """找整数矩阵 M, 使 M^{-1}A 是等长、120° 的面内基矢。

    多个 M 给出同样的 |a|, 但**取向不同** —— 只有让 CDW 的 Ta 落在原胞的
    整数平移上(偏移最小)的那个才是对的(否则"理想结构"会差 ~2 A)。
    ta_frac: Ta 的超胞分数坐标(n,2), 用来挑取向。
    """
    cands = []
    for m11 in range(-6, 7):
        for m12 in range(-6, 7):
            for m21 in range(-6, 7):
                for m22 in range(-6, 7):
                    M = np.array([[m11, m12], [m21, m22]], dtype=int)
                    det = int(round(np.linalg.det(M)))
                    if abs(det) != det_target:
                        continue
                    ap = np.linalg.inv(M.astype(float)) @ A
                    l1, l2 = np.linalg.norm(ap[0]), np.linalg.norm(ap[1])
                    if abs(l1 - l2) > 1e-3 or not (2.5 < l1 < 4.0):
                        continue
                    ang = np.degrees(np.arccos(np.dot(ap[0], ap[1]) / (l1 * l2)))
                    if abs(ang - 120.0) > 1e-2:
                        continue
                    score = 0.0
                    if ta_frac is not None:
                        fp = ta_frac @ M
                        dev = fp - np.rint(fp)
                        score = float(np.max(np.linalg.norm(dev @ ap, axis=1)))
                    cands.append((M, ap, l1, score))
    if not cands:
        raise SystemExit(f"找不到 det={det_target} 的整数变换")
    # 先按"CDW 的 Ta 离理想格点的偏移最小"排, 再按 det>0
    cands.sort(key=lambda c: (round(c[3], 4), -np.sign(np.linalg.det(c[0]))))
    return cands[0]


def write_vasp(path: str, title: str, lattice: np.ndarray, species,
                coords: np.ndarray) -> None:
    """写 VASP-5 格式 POSCAR(按 species 的出现顺序分组)。"""
    groups = []
    for s in species:
        if groups and groups[-1][0] == s:
            groups[-1][1] += 1
        else:
            groups.append([s, 1])
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
    ap.add_argument("--cdw-poscar", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    st = read_poscar(args.cdw_poscar)
    lat = st.lattice
    cart = st.positions @ lat
    ta_idx = [i for i, s in enumerate(st.species) if s == "Ta"]
    s_idx = [i for i, s in enumerate(st.species) if s == "S"]
    n_cell = len(ta_idx)
    print(f"CDW 超胞: {len(ta_idx)} Ta + {len(s_idx)} S = {len(st.species)} 原子")

    # ---- 1. 面内变换 M ----
    ta_frac = np.array([cart[i] @ np.linalg.inv(lat) for i in ta_idx])
    M2, ap, a_prim, score = find_M(lat[:2, :2], n_cell, ta_frac[:, :2])
    print(f"  选 M 的判据: CDW 的 Ta 离理想格点最大偏移 {score:.4f} A")
    M = np.array([[M2[0, 0], M2[0, 1], 0],
                  [M2[1, 0], M2[1, 1], 0],
                  [0, 0, 1]], dtype=int)
    print(f"原胞面内常数 a = {a_prim:.5f} A;  M =\n{M}\n  det = {int(round(np.linalg.det(M)))}")

    # ---- 2. 原胞(1 Ta + 2 S) ----
    # Ta 的 z: 取 CDW 里 Ta 的平均 z(相对于晶胞原点); S 的 z 同理
    # Ta 层在 z ~ 0; S 分两层: z in [0, c/2) 是"上一层 Ta 之上"的 S,
    # z in [c/2, c) 的换算成负值是"上一层 Ta 之下"的 S。
    z_ta = float(np.mean([cart[i][2] for i in ta_idx]))
    z_s_all = np.array([cart[i][2] for i in s_idx])
    z_mid = 0.5 * lat[2, 2]
    z_s_top = float(np.mean(z_s_all[z_s_all < z_mid]))
    z_s_bot = float(np.mean(z_s_all[z_s_all >= z_mid])) - lat[2, 2]
    # 原胞晶格直接取 M^{-1} @ 超胞晶格: 这样 M @ PRIMCELL.lattice 与超胞晶格
    # **逐位相同**(弛豫过的超胞有 ~1e-5 A 的 z 倾斜, 手工拼会差 1e-4,
    # 超过 unfoldfs.resolve_transform 的 1e-4 容差)
    prim_lat = np.linalg.inv(M.astype(float)) @ lat
    c_lat = lat[2, 2]
    # S 的 z **不假设**哪一子晶格在上: 按原胞分数坐标把 S 分成 (1/3,2/3) 与
    # (2/3,1/3) 两类, 各自取平均 z(这一版的 CDW 与 1T 的 S 上下关系是反的!)
    inv_lat0 = np.linalg.inv(lat)
    cls = {"A": [], "B": []}
    for i in s_idx:
        f2 = (cart[i] @ inv_lat0)[:2] @ M2
        d = f2 - np.rint(f2)
        # 归到 (1/3,2/3) 或 (2/3,1/3) 两个子晶格中更近的一个(周期距离)
        cand = {"A": np.array([1.0 / 3.0, 2.0 / 3.0]),
                "B": np.array([2.0 / 3.0, 1.0 / 3.0])}
        dd = {k: (d - v) - np.rint(d - v) for k, v in cand.items()}
        key = min(dd, key=lambda k: np.linalg.norm(dd[k]))
        cls[key].append(cart[i][2])
    z_a = float(np.mean(cls["A"]))
    z_b = float(np.mean(cls["B"]))
    print(f"  S 子晶格 A(1/3,2/3): z = {z_a:.4f} A ({len(cls['A'])} 个); "
          f"B(2/3,1/3): z = {z_b:.4f} A ({len(cls['B'])} 个)")
    # ⚠ POSCAR 的 "Direct" 是**分数坐标** ⇒ z 要除以晶格常数
    prim_pos = np.array([
        [0.0, 0.0, (z_ta / c_lat) % 1.0],
        [1.0 / 3.0, 2.0 / 3.0, (z_a / c_lat) % 1.0],
        [2.0 / 3.0, 1.0 / 3.0, (z_b / c_lat) % 1.0],
    ])
    prim_spec = ("Ta", "S", "S")
    write_vasp(os.path.join(args.out_dir, "PRIMCELL.vasp"),
                "1T-TaS2 primitive (from the CDW supercell lattice)",
                prim_lat, prim_spec, prim_pos)

    # 自洽性: M @ 原胞晶格 == 超胞晶格
    assert np.allclose(M @ prim_lat, lat, atol=1e-5), "M 与晶格不自洽"
    print("  自洽性检查: M @ PRIMCELL.lattice == CDW.lattice  OK")

    with open(os.path.join(args.out_dir, "TRANSMAT.in"), "w", encoding="utf-8") as fh:
        fh.write("Read transformation matrix from the TRANSMAT.in file if it exists.\n")
        for row in M:
            fh.write("  %d  %d  %d   # must be three integers\n" % tuple(row))

    # ---- 3. 未畸变的 39 原子超胞(给原子对应用) ----
    # 生成: 原胞 3 个原子 × 13 个平移(平移矢量 = 超胞晶格矢量 A1, A2)
    prim_cart = prim_pos @ prim_lat
    # 超胞自身的面内晶格矢量(注意: 不能用 prim_cart[0], 那是第一个**原子**)
    A1 = lat[0]
    A2 = lat[1]
    inv_lat = np.linalg.inv(lat)
    # 原胞平移点 m*a1 + n*a2 **对超胞晶格取模** -> 13 个不同的等价类
    a1_prim, a2_prim = prim_lat[0], prim_lat[1]
    shifts, seen = [], set()
    for m in range(-20, 21):
        for n in range(-20, 21):
            t = m * a1_prim + n * a2_prim
            f = np.mod(t @ inv_lat, 1.0)
            key = tuple(np.round(f, 5) % 1.0)
            if key in seen:
                continue
            seen.add(key)
            shifts.append(f @ lat)
    ideal = []
    for shift in shifts:
        for p in prim_cart:
            q = p + shift
            f = np.mod(q @ inv_lat, 1.0)
            ideal.append(f @ lat)
    ideal = np.array(ideal)
    # 去重(折回胞内时可能有边界重复)
    _, uniq = np.unique(np.round(ideal @ inv_lat, 5) % 1.0, axis=0,
                        return_index=True)
    ideal = ideal[np.sort(uniq)]
    n_ta_ideal = int(np.sum(np.abs(ideal[:, 2] - z_ta) < 1e-3))
    print(f"  未畸变超胞: {len(ideal)} 个理想原子 "
          f"(Ta {n_ta_ideal}, S {len(ideal) - n_ta_ideal}; "
          f"期望 {len(ta_idx)}+{len(s_idx)})")

    # 按 CDW 的原子顺序逐个匹配(同物种 + 最近)
    # 理想原子按物种分开索引(匹配时**只匹配同物种**)
    ideal_sp = []
    for j, q in enumerate(ideal):
        f = np.mod(q @ inv_lat, 1.0)
        ideal_sp.append(None)
    # 重新生成: 每个原胞原子 -> 13 个副本, 逐个按物种匹配
    order = []
    used = set()
    for i, sp in enumerate(st.species):
        best, bj = 1e9, None
        for j, q in enumerate(ideal):
            if j in used:
                continue
            # 物种判定: 理想原子离 z_ta 近的是 Ta, 否则是 S
            is_ta = abs(q[2] - z_ta) < 1e-3
            if is_ta != (sp == "Ta"):
                continue
            q_frac = q @ inv_lat
            p_frac = cart[i] @ inv_lat
            d = p_frac - q_frac
            d -= np.rint(d)
            dist = float(np.linalg.norm(d @ lat))
            if dist < best:
                best, bj = dist, j
        order.append(bj)
        used.add(bj)
    ideal_ordered = ideal[order]
    write_vasp(os.path.join(args.out_dir, "IDEAL_SUPERCELL.vasp"),
                "1T-TaS2 sqrt13 CDW supercell - ideal (undistorted) positions",
                lat, st.species, ideal_ordered @ inv_lat)

    # ---- 4. 诊断 ----
    ta_c = cart[ta_idx]
    print("\nCDW 结构的 Ta-Ta 键长 (A):")
    for i in range(len(ta_idx)):
        for j in range(i + 1, len(ta_idx)):
            d = float(np.linalg.norm(ta_c[i] - ta_c[j]))
            print(f"   Ta{i}-Ta{j}: {d:.4f}")
    print(f"原胞 Ta-Ta = {a_prim:.4f} A  -> 三聚化收缩 {(a_prim-3.16527)/a_prim*100:.1f}%")
    print("\n写好了:")
    for f in ("PRIMCELL.vasp", "TRANSMAT.in", "IDEAL_SUPERCELL.vasp"):
        print("   ", os.path.join(args.out_dir, f))


if __name__ == "__main__":
    main()
