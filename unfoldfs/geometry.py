"""几何与对称性：结构解析、变换矩阵、候选折叠矢量、公共对称操作集。

约定(全模块统一):
  * 晶格矩阵 lattice: 行 = 实矢量(Å);  倒格矢 reciprocal(): 行 = 倒矢量(含 2π)
  * 变换矩阵 m_eff: 行约定 A_sc = m_eff @ A_prim
  * 分数坐标: 原胞/超胞各自基底下的无量纲坐标, 行向量写法 k @ M
  * spglib 的对称操作按列向量约定存储: x' = R x + t
    (故对行向量 k 的作用为 k @ R.T + t)
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Structure:
    lattice: np.ndarray
    positions: np.ndarray
    species: Tuple[str, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.species)


def read_poscar(path: str) -> Structure:
    """读 POSCAR/PRIMCELL.vasp -> Structure(分数坐标)。"""
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    ) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip()
    idx = 8
    if mode.lower().startswith("s"):          # Selective dynamics
        idx = 9
    cartesian = mode.lower().startswith("c")
    inv = np.linalg.inv(lattice)
    positions, species = [], []
    for sym, n in zip(symbols, counts):
        for _ in range(n):
            p = np.array([float(x) for x in lines[idx].split()[:3]])
            positions.append(p @ inv if cartesian else p)
            species.append(sym)
            idx += 1
    return Structure(lattice, np.array(positions), tuple(species))


def reciprocal(lattice: np.ndarray) -> np.ndarray:
    """行=倒格矢, 含 2π (2π/Å)。"""
    vol = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    return 2.0 * np.pi * np.array([
        np.cross(lattice[1], lattice[2]),
        np.cross(lattice[2], lattice[0]),
        np.cross(lattice[0], lattice[1]),
    ]) / vol


def read_transmat(path: str) -> np.ndarray:
    """读 TRANSMAT.in -> (3,3) 整数矩阵(跳过注释行)。"""
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            rows.append([int(float(x)) for x in parts])
    if len(rows) != 3:
        raise ValueError(f"TRANSMAT 解析失败, 得到 {len(rows)} 行")
    return np.array(rows, dtype=int)


@dataclass(frozen=True)
class Transform:
    m_eff: np.ndarray
    m_inv: np.ndarray

    @property
    def det(self) -> int:
        return int(round(abs(np.linalg.det(self.m_eff))))

    def prim_from_super(self, k_super: np.ndarray, g: np.ndarray) -> np.ndarray:
        """k'_prim = (K + G) · M_eff^-1"""
        return (k_super + g) @ self.m_inv

    def super_from_prim(self, k_prim: np.ndarray) -> np.ndarray:
        """K = k'_prim · M_eff (mod 1)"""
        return np.mod(k_prim @ self.m_eff, 1.0)


def resolve_transform(sc: Structure, prim: Structure, m: np.ndarray) -> Transform:
    """探测 TRANSMAT 的行/列约定, 统一到行约定 (A_sc = M_eff @ A_prim)。"""
    for cand in (m, m.T):
        if np.allclose(cand @ prim.lattice, sc.lattice, atol=1e-4):
            m_eff = np.array(cand, dtype=int)
            return Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    raise ValueError(
        "TRANSMAT/POSCAR/PRIMCELL 不自洽: "
        f"M={m.tolist()} 与超胞晶格 {sc.lattice.tolist()} 不匹配"
    )


def matrix_denominator_lcm(m_eff: np.ndarray) -> int:
    """comm(M_eff^-1): 使 M_eff^-1 所有元素分母整除的最小正整数。"""
    inv = np.linalg.inv(np.asarray(m_eff, dtype=float))
    dens = [Fraction(float(x)).limit_denominator(1000).denominator for x in inv.ravel()]
    out = 1
    for d in dens:
        out = out * d // _gcd(out, d)
    return int(out)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def candidate_g_vectors(m_eff: np.ndarray) -> np.ndarray:
    """{g ∈ Z³ : g·M_eff^-1 ∈ [0,1)³}, 恰 det(M) 个。"""
    inv = np.linalg.inv(np.asarray(m_eff, dtype=float))
    det = int(round(abs(np.linalg.det(np.asarray(m_eff, dtype=float)))))
    span = int(np.abs(np.asarray(m_eff)).max()) + 2
    out = []
    for g in np.ndindex(*([2 * span + 1] * 3)):
        gg = np.array(g) - span
        t = gg @ inv
        if np.all(t >= -1e-9) and np.all(t < 1.0 - 1e-9):
            out.append(gg)
    if len(out) != det:
        raise ValueError(f"候选 G 数 {len(out)} != det(M) {det}")
    return np.array(out, dtype=int)


@dataclass(frozen=True)
class SymmetryOps:
    rotations: np.ndarray
    translations: np.ndarray

    def apply(self, k_prim: np.ndarray, op_index: int) -> np.ndarray:
        r = self.rotations[op_index]
        t = self.translations[op_index]
        return np.mod(k_prim @ r.T + t, 1.0)   # r 以列向量约定存储 -> 转置


def common_symmetry_ops(prim: Structure, m_eff: np.ndarray,
                        symprec: float = 1e-5) -> SymmetryOps:
    """原胞对称操作中, 同时也是超胞对称的那些。

    判据 (行约定):  M^-1 R M 为整数矩阵  且  t·M^-1 为整数矢量。

    注意 t 必须先归一化到 [0,1)³ 再判: 平移只在原胞格矢意义下确定, spglib
    返回的代表元可能是 (1,0,0) 而非 (0,0,0) (同一个操作, 不同代表元),
    直接拿未归一化的 t 去判会误删操作, 得到不成群的集合。
    """
    import spglib

    # spglib 的 cell = (lattice, positions, numbers): lattice 行=基矢, positions 分数坐标
    cell = (prim.lattice, prim.positions, _species_ids(prim.species))
    # spglib >= 2.0: get_symmetry 返回 dict(rotations/translations/equivalent_atoms)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    rotations = sym["rotations"]
    translations = sym["translations"]
    m = np.asarray(m_eff, dtype=float)
    inv = np.linalg.inv(m)
    keep_r, keep_t = [], []
    for r, t in zip(rotations, translations):
        r_sc = inv @ r.astype(float) @ m
        if not np.allclose(r_sc, np.round(r_sc), atol=1e-6):
            continue                     # 在超胞基底非整数 -> 不是超胞对称
        t_prim = _canonical_translation(t)
        t_sc = t_prim @ inv
        if not np.allclose(t_sc, np.round(t_sc), atol=1e-6):
            continue                     # 平移不落在超胞晶格 -> 不是超胞对称
        keep_r.append(r.astype(int))
        keep_t.append(t_prim)
    if not keep_r:
        raise ValueError("未找到任何公共对称操作")
    return SymmetryOps(np.array(keep_r), np.array(keep_t))


def _canonical_translation(t: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """平移归一化到 [0,1)³。

    贴近 0 的分量是数值残差(spglib 返回 ~1e-8), 贴近 1 的分量与 0 等价
    (同一个操作的另一个代表元), 两者都归零。
    """
    t_mod = np.mod(np.asarray(t, dtype=float), 1.0)
    t_mod = np.where(np.abs(t_mod - 1.0) < tol, 0.0, t_mod)
    return np.where(np.abs(t_mod) < tol, 0.0, t_mod)


def _species_ids(species: Tuple[str, ...]) -> np.ndarray:
    uniq = sorted(set(species))
    index = {s: i + 1 for i, s in enumerate(uniq)}
    return np.array([index[s] for s in species], dtype=int)
