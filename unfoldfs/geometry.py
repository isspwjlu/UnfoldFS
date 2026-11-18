"""几何与对称性：结构解析、变换矩阵、候选折叠矢量、公共对称操作集。

约定(全模块统一, Ruling 8: 直接空间用 M, 倒空间用 M^-T):
  * 晶格矩阵 lattice: 行 = 实矢量(Å);  倒格矢 reciprocal(): 行 = 倒矢量(含 2π)
  * 变换矩阵 m_eff: 行约定 A_sc = m_eff @ A_prim
    -> 倒格矢   B_sc = (m_eff^-1)^T @ B_prim   (由 A_sc B_sc^T = 2πI 推得)
    -> 位置     r_prim = r_sc @ m_eff          (分数坐标换基)
    -> k        k_prim = k_sc @ (m_eff^-1)^T   (分数坐标换基)
  * 分数坐标: 原胞/超胞各自基底下的无量纲坐标, 行向量写法 k @ M
  * spglib 的对称操作按**列向量**约定存储: x' = R x + t
    (故对**直接空间**的行向量作用是 x' = x @ R.T + t);
    **倒空间**的分数基算符的伴随是 **R^-1**(不是 R.T!) —— 分数基不是正交阵,
    "R 的转置 = R 的逆"在这里不成立:
        k' = k @ R^-1   (行向量约定)
    笛卡尔几何真值(12 个操作 × 多个 k)裁决: k@R^-1 与真值 48/48 一致,
    k@R.T 只有 16/48、k@R 只有 32/48 —— 见 tests/test_geometry.py 的
    `test_symmetry_ops_apply_matches_the_cartesian_geometry_truth` 与
    docs/algorithms.md §1.2。
    超胞基底下的旋转 = M^-T R M^T, 整数性判据即 M^-T R M^T ∈ Z³
    (行向量约定下才是 M R M^-1; 对称 M 时两者恒等)
"""
from __future__ import annotations

import os
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
    """读 POSCAR/PRIMCELL.vasp -> Structure(分数坐标)。

    **只支持 VASP-5 格式**(第 6 行是元素符号、第 7 行是各元素的原子数)。
    VASP-4 格式(没有元素符号行, 第 6 行直接是原子数)在这里给出明确报错 ——
    否则它会掉进 ``int("Direct")`` 这种看不懂的 ValueError。
    """
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    if len(lines) < 8:
        raise ValueError(
            f"{os.path.basename(path)} 被截断: 只有 {len(lines)} 行, "
            "至少需要 8 行(晶格 3 行 + 缩放 + 元素符号 + 原子数 + 坐标模式)")
    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    ) * scale
    if _looks_like_counts(lines[5]):
        raise ValueError(
            f"{os.path.basename(path)} 看起来是 **VASP-4 格式**(第 6 行是原子数"
            "计数, 没有元素符号行) —— 本工具只支持 **VASP-5 格式**: "
            "第 6 行必须是元素符号(如 'Mo S'), 第 7 行才是各元素的原子数。"
            "请在 POSCAR 里补上元素符号行(VASP-5 格式)。")
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


def _looks_like_counts(line: str) -> bool:
    """该行是不是"原子数计数"行(VASP-4 格式的第 6 行)。"""
    parts = line.split()
    if not parts:
        return False
    try:
        for p in parts:
            int(p)
    except ValueError:
        return False
    return True


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
    with open(path) as fh:
        for line in fh:
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
        """k'_prim = (K + G) · (M_eff^-1)^T   (倒空间用 M^-T)"""
        return (k_super + g) @ self.m_inv.T

    def super_from_prim(self, k_prim: np.ndarray) -> np.ndarray:
        """K = k'_prim · M_eff^T (mod 1)   (倒空间用 M^-T 的逆变换)"""
        return np.mod(k_prim @ self.m_eff.T, 1.0)


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


def periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐点周期距离: (n,3) x (n,3) -> (n,)。

    周期距离 = min(|Δ|, 1-|Δ|): 0.99999997 与 0.0 的距离是 3e-8 而不是 1。
    (spglib 的平移代表元带 ~3e-8 残差, 且取模后 0.99999997 与 0.0 是同一个点
    —— 用非周期距离会选错候选。)
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    d = np.abs(a - b)
    d = np.minimum(d, 1.0 - d)
    return np.linalg.norm(d, axis=1)


def periodic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组分数坐标点之间的周期距离(L2): (n,3) x (m,3) -> (n,m)。"""
    d = np.abs(np.atleast_2d(a)[:, None, :] - np.atleast_2d(b)[None, :, :])
    d = np.minimum(d, 1.0 - d)
    return np.linalg.norm(d, axis=2)


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
    """{g ∈ Z³ : g·(M_eff^-1)^T ∈ [0,1)³}, 恰 det(M) 个。

    基本区域判据用 (M^-1)^T (倒空间用 M^-T), 与 prim_from_super 保持一致。

    搜索范围由 **M 的行和界**给出: 候选 g 满足 g·M^-T = f ∈ [0,1)³,
    即 g = f·M^T, 故 g_i = Σ_j f_j·M_ij 的取值范围是
    [Σ_j min(0,M_ij), Σ_j max(0,M_ij)] (M 的第 i 行)。
    用 max|M| 给界会漏掉离对角元素大的矩阵(莫尔型超胞)的候选 ——
    实测随机整数 M 的失败率随 |M| 增大: |M| ~ 3 时 ~9%, |M| ~ 30 时 ~93%;
    换成行和界后降到 0。(unfold.candidate_g_for_k 用同一界, 两处必须保持一致。)
    """
    m = np.asarray(m_eff)
    m_f = m.astype(float)
    inv = np.linalg.inv(m_f)
    det = int(round(abs(np.linalg.det(m_f))))
    hi = np.clip(m, 0, None).sum(axis=1)     # 每行正部分之和 -> g_i 上界
    lo = np.clip(-m, 0, None).sum(axis=1)    # 每行负部分之和 -> g_i 下界
    # 逐轴区间外扩一格作数值余量(边界候选本身会被分数判据挡掉)
    axes = [np.arange(-lo[i] - 1, hi[i] + 2, dtype=int) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    frac = grid @ inv.T
    keep = np.all(frac >= -1e-9, axis=1) & np.all(frac < 1.0 - 1e-9, axis=1)
    out = grid[keep]
    if out.shape[0] != det:
        rng_desc = "×".join(f"[{-lo[i] - 1},{hi[i] + 1}]" for i in range(3))
        raise ValueError(
            f"候选 G 数 {out.shape[0]} != 期望 {det} 个 "
            f"(逐轴搜索区间 {rng_desc}, 由 M_eff 行和界给出) —— "
            "整数可逆的 m_eff 不该出现此错: 请检查 m_eff 是否为整数矩阵、是否奇异")
    return out.astype(int)


@dataclass(frozen=True)
class SymmetryOps:
    rotations: np.ndarray
    translations: np.ndarray

    def apply(self, k_prim: np.ndarray, op_index: int) -> np.ndarray:
        """对称操作作用到**原胞分数坐标**的 k 上(行向量约定)。

        k' = k @ R^-1 + t

        ⚠ 倒空间的伴随是 **R^-1**, 不是 R.T: 分数基下的算符不是正交阵,
        "R 的转置 = R 的逆"只对笛卡尔正交阵成立。笛卡尔几何真值裁决
        (12 个操作 × 多个 k): k@R^-1 一致 48/48, k@R.T 只有 16/48。
        见 docs/algorithms.md §1.2 与 tests/test_geometry.py 的回归测试。
        """
        r = self.rotations[op_index]
        t = self.translations[op_index]
        return np.mod(k_prim @ rotation_inverse(r) + t, 1.0)


def rotation_inverse(r: np.ndarray) -> np.ndarray:
    """整数阵 R 的**精确**逆(仍是整数阵)。

    R 是幺模整数阵(对称操作的旋转部分)⇒ R^-1 也是整数阵, 故用
    "浮点 inv 再取整"求它是精确的(不是近似)。不用 ``np.linalg.inv`` 直接返回
    浮点: 大分量下会丢精度, 而这里的下游全是整数性判据。
    """
    r_int = np.asarray(r)
    inv = np.round(np.linalg.inv(r_int.astype(float))).astype(int)
    if not np.allclose(inv @ r_int, np.eye(3, dtype=int), atol=1e-6):
        raise ValueError(
            f"R 不是幺模整数阵, 求不出整数逆: R={r_int.tolist()}")
    return inv


def common_symmetry_ops_detailed(prim: Structure, m_eff: np.ndarray,
                                 symprec: float = 1e-5):
    """原胞对称操作中, 同时也是超胞对称的那些 —— 并报告**被剔除**的操作。

    判据 (行约定):  M^-T R M^T 为整数矩阵 (该旋转把超胞晶格映到自身)。
    spglib 的 R 是列向量约定 (x' = R x), 故超胞基底下的旋转是 M^-T R M^T;
    行向量约定下才是 M R M^-1 (对称 M 时两者恒等)。

    平移**不参与**筛选: spglib 给的 t 是模原胞格矢的代表元 (C3^2 的代表元是
    (1,0,0)), 换基底后 t·M^-1 未必是整数 —— 用它筛选会误删操作, 得到不成群的
    集合 (C3 在内而 C3^2 不在)。平移只需是原胞晶格平移 (spglib 已保证),
    而 A(k,E) 对整个原胞点群不变。

    返回 (ops, dropped): dropped 是 [(旋转, 平移, 原因), ...], 供 `info`
    打印"保留 N / 剔除 M(原因: …)"(spec §7)。
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
    dropped = []
    for r, t in zip(rotations, translations):
        r_sc = inv.T @ r.astype(float) @ m.T
        if not np.allclose(r_sc, np.round(r_sc), atol=1e-6):
            dropped.append((r.astype(int), _harden_translation(t),
                            "M^-T·R·M^T 在超胞基底非整数 —— 该旋转不保持超胞晶格"))
            continue        # 旋转在超胞基底非整数 -> 该操作不保持超胞晶格
        keep_r.append(r.astype(int))
        keep_t.append(_harden_translation(t))
    if not keep_r:
        raise ValueError("未找到任何公共对称操作")
    return SymmetryOps(np.array(keep_r), np.array(keep_t)), dropped


def common_symmetry_ops(prim: Structure, m_eff: np.ndarray,
                        symprec: float = 1e-5) -> SymmetryOps:
    """原胞对称操作中, 同时也是超胞对称的那些(只返回保留的操作)。"""
    ops, _dropped = common_symmetry_ops_detailed(prim, m_eff, symprec)
    return ops


def _harden_translation(t: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """平移归一化到 [0,1)³, 并把接近 0/1 的分量吸附到 0。

    spglib 给 C3^2 的代表元是 t=(0.9999999999999999, 0, 0) —— 与 0 等价但
    数值上 != 0; 原样存下, 下游按 int(mod(t*n,1)*n) 做网格检查会静默错一格。
    """
    t_mod = np.mod(np.asarray(t, dtype=float), 1.0)
    return np.where(t_mod > 1.0 - tol, 0.0, t_mod)


def _species_ids(species: Tuple[str, ...]) -> np.ndarray:
    uniq = sorted(set(species))
    index = {s: i + 1 for i, s in enumerate(uniq)}
    return np.array([index[s] for s in species], dtype=int)
