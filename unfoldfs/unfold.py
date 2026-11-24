"""Popescu 投影法超胞能带反折叠(Popescu, PRL 104, 216401 (2012))。

展开公式(移植自已实测验证的原型 ``unfold_procar.py``):

    p_i(lm) = (1/N_i) · Σ_T c_{i+T,lm} · exp(s1·i·(k'−K)·T)     副本相位还原平均
    W(k')   = | Σ_i exp(s2·i·k'·r_i) · Σ_lm p_i(lm) |²           候选 G 上的谱权重

两处**不可改**的语义(原型做对了, 改了就错):

1. **不可**假设同一原胞原子的各副本复值相同 —— 实测同一 (k, 带, 轨道) 上
   9 个副本可取 3 个不同值(按 (m+n) mod 3 分组), 故必须做**相位还原**的副本
   平均, 不能取单个副本、也不能假设副本相等;
2. 相位符号约定 (s1, s2) 的两个符号**物理地位不同**(修复轮 9):
   * **s1(副本相位)由物理唯一确定 = −1** —— 副本相位必须抵消超胞 Bloch
     因子 ``e^{+iK·R_T}``(把超胞态投影到原胞 Bloch 态时, 每个副本带
     ``e^{−i(k'−K)·T·A}``)。实测「单 G 主导度」对 s1 几乎没有分辨力
     (四个约定只差 ~1e-3), 而选错 s1 会让 55%~65% 的 (k,带) 换候选归属
     ⇒ 整张费米面错形 ⇒ **s1 不能靠打分定**;
   * **s2(Bloch character 符号)是真正的约定** —— 在 s1 固定后的两个候选
     里按「单 G 集中度」择优(实测相对差 2.0%~2.1%, 有判别力)。

**两种模式(权重定义不同, 不可混用)**:

* **超胞展开模式**(det(M) > 1): 上面的 Popescu 公式。权重是**相干**的 ——
  先在原胞原子上相干求和再取模方, 它是"这个态落在哪个候选 G 上"的权重。
* **原胞直通模式**(det(M) == 1): "超胞"就是原胞, 没有折叠也没有多份副本,
  Popescu 路径**在数学上退化**(候选 G 只有一个 ⇒ 4 种相位约定的主导度
  必然完全相同 ⇒ 择优守卫会拒掉原胞自己的计算)。此时**不做展开**, 直接用
  原胞的带能量与**轨道投影权重** ``w_n(k) = Σ_{ion,lm} |c_{ion,lm,n}(k)|²``
  (**非相干**和, 教科书定义, 也是 vaspkit 282 用的那个量)。
  ⇒ 两种模式的**费米面形状不可直接比较**。

约定(与 geometry.py 一致):

* 晶格行 = 实矢量; 倒格矢行 = 倒矢量(含 2π);
* 超胞 k(分数) K 与原胞 k(分数) k' 的关系: ``k' = (K + g)·(M^-1)^T``
  (倒空间用 M^-T); 候选 g **逐 k 计算** —— 一般 M 下候选集随 K 平移,
  用固定 gset 会给出错误的候选;
* 副本平移 T 以**原胞格矢**为单位;
* 原胞 k 全部用 EIGENVAL 的分数坐标(PROCAR 的 k 与之逐位相同)。

**相位约定的择优口径必须与原型一致**: 原型只在 首 min(4, n_k) 个 k ×
首 min(n_band, 40) 条带 上评估 4 种约定(见 ``CONVENTION_SAMPLE_N_K`` /
``CONVENTION_SAMPLE_N_BAND``)。用"全部 k × 全部带"择优在 fixture 上会选出
**不同**的约定(评审实测: 164/240 个 (k,带) 的候选归属改变 -> 重组后的能带
结构, 即费米面, 随之改变)。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from .geometry import Structure, Transform, periodic_delta, periodic_distance
from .pwunfold import plane_wave_weights
from .vaspio import BandData, ProcarData
from .wfc import iter_wavecar_kpoints, read_wavecar_header

__all__ = [
    "AtomCopies", "UnfoldedResult", "DominanceReport", "build_atom_copies",
    "candidate_g_for_k", "unfold_at_k", "select_phase_convention",
    "unfold_supercell",
]

# 相位符号约定的 4 种组合 (s1 作用于副本平移项, s2 作用于原胞 k 相位项)
CONVENTIONS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# ---------------------------------------------------------------- 相位符号的物理地位
#
# 修复轮 9 的实测结论: **两个符号的物理地位完全不同**, 必须分开处理。
#
# * ``s1``(副本相位 ``exp(s1·i·(k'−K)·T)`` 的符号)由**物理唯一确定**:
#   把超胞态 ψ_K 投影到原胞 Bloch 态 |k'⟩ 时, 每个超胞原子
#   R_j = r_i + T·A 带相位 ``e^{i(K−k')·R_j}``(超胞 Bloch 因子 e^{+iK·R}
#   与原胞 Bloch character e^{−ik'·r} 合成的); 把它拆成"原子项"
#   ``e^{i(K−k')·r_i}`` 与"副本项" ``e^{i(K−k')·(T·A)} = e^{−i(k'−K)·(T·A)}``
#   ⇒ **s1 = −1**, 唯一解。
#
#   ⇒ s1 **不靠打分**: 实测"单 G 集中度"对 s1 几乎没有分辨力
#   (1T-TaS₂ 的 √3×√3 R30° 超胞 48×48×1: 四个约定
#   0.7653/0.7723/0.7701/0.7680(全网格)与 0.6728/0.6885/0.6873/0.6733
#   (原型口径), 差 ~1e-3), 而选错 s1 会让 **55%~65% 的 (k,带) 换候选
#   归属** ⇒ 整张费米面错形(实测 Pearson 0.1556 vs 0.9941, 带能量最近
#   距离 0.154 eV vs 2.05e-05 eV)⇒ **打分分辨不出对错, 必须靠物理**。
#
# * ``s2``(原胞 Bloch character ``exp(s2·i·k'·r_i)`` 的符号)是**真正的约定**:
#   e^{+ik·r} 与 e^{−ik·r} 互为共轭, 投影本身定不出哪一个 ⇒ 只能靠
#   "态应当集中在唯一一个候选上"(单 G 集中度)来分辨。
#   实测三组真实数据上, s1 固定之后两个候选的集中度相对差
#   **2.13% / 1.91% / 2.04%** ⇒ 这一步**有判别力**(比旧判据的噪声
#   量级 1e-3 大 20 倍)。
PHASE_COPY_SIGN = -1

# 单 G 主导度下限: 低于此值说明态没有被集中到唯一一个候选 G 上 -> 展开不可信
MIN_DOMINANCE = 0.5

# 相位约定择优的**采样口径**(与原型 unfold_procar.py 一致):
# 只在 首 min(4, n_k) 个 k × 首 min(n_band, 40) 条带 上评估 4 种约定。
CONVENTION_SAMPLE_N_K = 4
CONVENTION_SAMPLE_N_BAND = 40

# 统计每态主导度时的最小总权重(原型: sw > 1e-12 才计入)
_MIN_STATE_WEIGHT = 1e-12

# winner 与 runner-up 的**相对**主导度差小于此值 -> 约定选择落在退化区
# (由噪声驱动, 换一批采样就可能换一个约定) -> 必须告警, 不能静默
MIN_CONVENTION_MARGIN_REL = 1e-3

# "好"的每态主导度阈值(真实 VASP 数据实测 0.85~1.0)
DOMINANCE_GOOD = 0.85

# 原子副本匹配容差(分数坐标下的整数性判据)
_COPY_TOL = 1e-6

# 平面波路线: 平面波归入候选等价类的残差容差(分数坐标的周期距离)
_PW_CLASS_TOL = 1e-6

# 平面波路线: 完备性 Σ_j w_j = 1 的容差(候选集是平面波基的划分 ⇒ 必为 1)
_PW_COMPLETENESS_TOL = 1e-6

# k 点匹配容差(分数坐标, 周期距离)
_K_TOL = 1e-5


@dataclass(frozen=True)
class AtomCopies:
    """原胞原子 -> 超胞副本表。

    copies[i] = [(超胞原子下标, 原胞平移 T), ...]; T 以原胞格矢为单位。
    """

    copies: List[List[Tuple[int, np.ndarray]]]

    @property
    def n_atoms(self) -> int:
        return len(self.copies)


@dataclass(frozen=True)
class DominanceReport:
    """单 G 主导度的分布 + 相位约定择优的判别性诊断。

    分布(全量权重 pass 上, 每个 (k,带) 是一个"态"):
        mean / min / median     主导度的均值 / 最小 / 中位数
        frac_above_good         主导度 > DOMINANCE_GOOD(=0.85) 的态占比
    择优诊断(采样口径上, 顺序同 CONVENTIONS):
        scores                  4 种约定的主导度
        selection_score         winner 的主导度
        runner_up_score         runner-up 的主导度
        margin / margin_rel     winner - runner_up 及其相对值
        degenerate              margin_rel < MIN_CONVENTION_MARGIN_REL
                               -> 选择由噪声驱动(已告警)

    ⚠ 修复轮 9: winner / runner-up 的比较**只在两个 s1 = −1 的候选之间**
    (s1 由物理唯一确定, 见 ``PHASE_COPY_SIGN``)⇒ margin 描述的是
    "s2 有没有选对", 而不是"整个约定有没有选对"。
    """

    mean: float = float("nan")
    min: float = float("nan")
    median: float = float("nan")
    frac_above_good: float = float("nan")
    scores: Tuple[float, ...] = ()
    selection_score: float = float("nan")
    runner_up_score: float = float("nan")
    margin: float = float("nan")
    margin_rel: float = float("nan")
    degenerate: bool = False


@dataclass(frozen=True)
class UnfoldedResult:
    """一个超胞 k 点集合的展开结果。

    k_cand   (n_k, detM, 3)  每个计算点的候选原胞 k(分数坐标)
    energies (n_k, detM, n_band_eff)  候选内按能量升序排列的带能量
    weights  (n_k, detM, n_band_eff)  对应权重; 补位处权重 0、能量 +inf
    valid    (n_k, detM, n_band_eff)  有效位掩码: 补位槽为 False
             (下游若对带轴做 argmin/求最小值, 必须用它排除 +inf 补位槽)
    """

    k_cand: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    orbital_weights: Optional[np.ndarray] = None
    """逐轨道谱权重 (n_k, detM, n_band_eff, norb); 默认 None(不计算)。

    只有 ``unfold_supercell(..., with_orbitals=True)`` 时才有。
    轨道成分请按 W_orb / Σ_n W_orb 归一化后使用(见 :func:`_weight_matrix`)。
    """
    s1: int = 1               # 采用的相位符号约定
    s2: int = 1
    dominance: float = float("nan")   # 选定约定下的单 G 主导度(均值)
    valid: Optional[np.ndarray] = None
    dominance_report: Optional[DominanceReport] = None
    # --- 经典 Popescu 展开能带图需要的**完整**带表(可选) ---
    band_energies: Optional[np.ndarray] = None   # (n_k, n_band) 全部超胞带能量
    band_weights: Optional[np.ndarray] = None    # (n_k, detM, n_band)
    #   每条超胞带在**每个**候选通道上的权重(不是只保留归属到该候选的那些带)


@dataclass(frozen=True)
class _CopyArrays:
    """副本表的扁平化形式(供向量化计算用)。"""

    ion: np.ndarray        # (Ntot,) 超胞原子下标
    parent: np.ndarray     # (Ntot,) 所属原胞原子下标
    trans: np.ndarray      # (Ntot,3) 原胞平移 T
    n_copies: np.ndarray   # (n_prim,) 每个原胞原子的副本数
    starts: np.ndarray     # (n_prim,) 副本分组起点(reduceat 用)
    cart_trans: np.ndarray  # (Ntot,3) 笛卡尔平移 T·A_prim


def build_atom_copies(prim: Structure, sc: Structure, tr: Transform) -> AtomCopies:
    """原胞原子 -> 超胞副本 [(超胞原子下标, 原胞平移 T)]。

    平移 T 以原胞格矢为单位::

        T = [(q·A_sc − p·A_prim)·A_sc^-1] · M_eff

    差值必须在**笛卡尔**下算: 分数坐标直接相减会把跨胞的原子混在一起。
    """
    inv_sc = np.linalg.inv(np.asarray(sc.lattice, dtype=float))
    out: List[List[Tuple[int, np.ndarray]]] = []
    for p in np.asarray(prim.positions, dtype=float):
        p_cart = p @ prim.lattice
        lst: List[Tuple[int, np.ndarray]] = []
        for j, q in enumerate(sc.positions):
            q_cart = np.asarray(q, dtype=float) @ sc.lattice
            frac = (q_cart - p_cart) @ inv_sc
            t = frac @ tr.m_eff
            t_int = np.round(t)
            if np.allclose(t, t_int, atol=_COPY_TOL):
                lst.append((int(j), t_int.astype(int)))
        if not lst:
            raise ValueError(
                "原子副本匹配失败: 原胞原子在超胞里找不到整数平移 —— "
                "检查 POSCAR / PRIMCELL / TRANSMAT 是否自洽")
        out.append(lst)
    return AtomCopies(out)


def candidate_g_for_k(k_super_frac: np.ndarray, m_eff: np.ndarray,
                      m_inv_T: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """使 (K+g)·(M^-1)^T 落在原胞第一布里渊区 [0,1)³ 的整数 g(恰 det(M) 个)。

    一般 M 下候选集**随 K 平移**(fixture 的 M=diag(3,3,1) 恰好与 K 无关),
    故不能用固定的全局 gset。

    搜索范围用 |M| 的**行和**给界: 由 (K+g)·(M^-1)^T ∈ [0,1)³ 可得
    K+g = v·M^T (v ∈ [0,1)³), 即 u_j = Σ_i v_i·M_ji, 故 |g_j| ≤ Σ_i|M_ji| + 1。
    用 max|M| 给界会漏掉离对角元素大的矩阵的候选(实测 300 个随机 M 中有 ~11%
    漏候选, 行和界后降到 0)。

    入口先把 K 折回 [0,1)³(``k = k % 1``): 上面那条界隐含 K ∈ [0,1)³ 的假设,
    K 落在第一布里渊区之外时(如 -0.2 / 10.3)候选会落在搜索范围之外而找不到。
    折回不改变候选的**分数值**: K -> K+n(整数) 时候选集平移 g -> g-n, 而
    (K+n+g-n)·(M^-1)^T ≡ (K+g)·(M^-1)^T; 相位项也用同一个折回后的 K
    (见 ``_candidate_k``), 保证候选集合与相位自洽。
    """
    m = np.asarray(m_eff, dtype=float)
    k = np.mod(np.asarray(k_super_frac, dtype=float), 1.0)
    m_inv_T = np.asarray(m_inv_T, dtype=float)
    spans = np.abs(m).sum(axis=1).astype(int) + 1          # 每轴的行和界
    axes = [np.arange(-s, s + 1, dtype=int) for s in spans]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"),
                    axis=-1).reshape(-1, 3)
    frac = (k + grid) @ m_inv_T
    keep = np.all(frac >= -tol, axis=1) & np.all(frac < 1.0 - tol, axis=1)
    out = grid[keep]
    det = int(round(abs(np.linalg.det(m))))
    if out.shape[0] != det:
        raise ValueError(
            f"k={k.tolist()} 的候选 G 数 {out.shape[0]} != det(M)={det}")
    return out.astype(int)


def _copy_arrays(copies: AtomCopies, a_prim: np.ndarray) -> _CopyArrays:
    """把副本表摊平成可向量化的数组。"""
    ion, parent, trans, counts = [], [], [], []
    for i, lst in enumerate(copies.copies):
        for j, t in lst:
            ion.append(int(j))
            parent.append(i)
            trans.append(np.asarray(t, dtype=float))
        counts.append(len(lst))
    counts_arr = np.array(counts, dtype=int)
    trans_arr = np.array(trans, dtype=float).reshape(-1, 3)
    starts = np.concatenate([[0], np.cumsum(counts_arr)[:-1]]).astype(int)
    return _CopyArrays(
        ion=np.array(ion, dtype=int),
        parent=np.array(parent, dtype=int),
        trans=trans_arr,
        n_copies=counts_arr,
        starts=starts,
        cart_trans=trans_arr @ np.asarray(a_prim, dtype=float),
    )


def _candidate_k(k_super_frac: np.ndarray, m_eff: np.ndarray,
                 m_inv_T: np.ndarray, b_prim: np.ndarray,
                 b_super: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """候选 g、候选原胞 k(分数) 与 (k'−K) 的笛卡尔差。"""
    g = candidate_g_for_k(k_super_frac, m_eff, m_inv_T)
    # 与 candidate_g_for_k 用**同一个** K(折回 [0,1)³) —— 否则候选集合与
    # 相位项不自洽(VASP 的 KPOINTS 本身就在 [0,1)³ 内, 此时折回是恒等变换)。
    k = np.mod(np.asarray(k_super_frac, dtype=float), 1.0)
    k_frac = (k + g) @ m_inv_T
    k_cart = k_frac @ b_prim
    k_super_cart = k @ b_super
    return g, k_frac, k_cart - k_super_cart


def _weight_matrix(k_super_frac: np.ndarray, coeffs: np.ndarray,
                   cp: _CopyArrays, tr: Transform, m_eff: np.ndarray,
                   a_prim: np.ndarray, prim_cart: np.ndarray,
                   b_prim: np.ndarray, b_super: np.ndarray,
                   s1: int, s2: int,
                   with_orbital: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """候选原胞 k(分数) 与谱权重 W(k') (detM, n_band)。

    ``with_orbital=True`` 时额外返回**逐轨道**权重 (detM, n_band, norb)::

        W_orb[g, b, n] = |Σ_i bloch_i · p[i, b, n]|²

    ⚠ 逐轨道权重之和**不等于**总权重: |Σ_n c_n|² 与 Σ_n |c_n|² 之间有交叉项。
       所以轨道成分要按 W_orb / Σ_n W_orb 归一化后才是"成分占比"。
    """
    _, k_frac, d_g = _candidate_k(k_super_frac, m_eff, tr.m_inv.T,
                                  b_prim, b_super)
    # 副本平移项的相位: exp(s1·i·(k'−K)·T)
    phase = np.exp(1j * s1 * (d_g @ cp.cart_trans.T))              # (detM, Ntot)
    # 原胞 k 的 Bloch 相位: exp(s2·i·k'·r_i)
    bloch = np.exp(1j * s2 * ((k_frac @ b_prim) @ prim_cart.T))     # (detM, n_prim)
    c = np.asarray(coeffs)[:, cp.ion, :]                           # (nb, Ntot, norb)
    p = np.einsum("gc,bcn->gbcn", phase, c)
    p /= cp.n_copies[cp.parent][None, None, :, None]
    p = np.add.reduceat(p, cp.starts, axis=2)                      # (detM, nb, n_prim, norb)
    # ⚠ **总权重必须走原来的求和顺序** —— 换成"先算逐轨道再求和"在浮点上
    # **不逐位相同**(求和次序不同), 而论文里的数字依赖逐位可复现。
    # 所以逐轨道那条另算一条路径, 只在 with_orbital 时才做。
    total = np.einsum("gi,gbi->gb", bloch, p.sum(axis=3))
    w_total = np.abs(total) ** 2
    if not with_orbital:
        return k_frac, w_total
    # 逐轨道: (detM, nb, norb)。不先对 norb 求和, 才能保留轨道分辨
    total_orb = np.einsum("gi,gbin->gbn", bloch, p)
    return k_frac, w_total, np.abs(total_orb) ** 2


def _dominance(w: np.ndarray) -> np.ndarray:
    """每条带在候选 G 上的单 G 主导度 max(w)/sum(w); w 的形状是 (..., detM, nb)。"""
    tot = w.sum(axis=-2)
    mx = w.max(axis=-2)
    out = np.zeros_like(tot)
    ok = tot > 0
    out[ok] = mx[ok] / tot[ok]
    return out


def _check_transform(tr: Transform, m_eff: np.ndarray) -> None:
    if not np.allclose(np.asarray(tr.m_eff, dtype=float),
                       np.asarray(m_eff, dtype=float)):
        raise ValueError(
            "tr.m_eff 与 m_eff 不一致 —— 候选 G 与相位都依赖变换矩阵, "
            "两者必须相同")


def unfold_at_k(k_super_frac: np.ndarray, coeffs: np.ndarray,
                energies: np.ndarray, tr: Transform, copies: AtomCopies,
                m_eff: np.ndarray, a_prim: np.ndarray, prim_cart: np.ndarray,
                b_prim: np.ndarray, b_super: np.ndarray,
                s1: int, s2: int,
                with_orbital: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """展开一个超胞 k 点。

    coeffs   (n_band, n_ions, norb) 复投影系数(晶胞周期规范)
    energies (n_band,) 该 k 点的超胞带能量

    返回 (k_cand (detM,3) 候选原胞 k(分数), E (detM,n_band), w (detM,n_band))。
    同一 k 的所有候选共享同一批超胞能量, 故 E 是 energies 的广播。
    """
    _check_transform(tr, m_eff)
    coeffs = np.asarray(coeffs)
    energies = np.asarray(energies, dtype=float)
    if coeffs.shape[0] != energies.shape[0]:
        raise ValueError(
            f"coeffs 的带数 {coeffs.shape[0]} != energies 的带数 "
            f"{energies.shape[0]}")
    cp = _copy_arrays(copies, a_prim)
    out = _weight_matrix(k_super_frac, coeffs, cp, tr, m_eff,
                          a_prim, prim_cart, b_prim, b_super, s1, s2,
                          with_orbital=with_orbital)
    e = np.repeat(energies[None, :], out[0].shape[0], axis=0)
    if with_orbital:
        return out[0], e, out[1], out[2]
    return out[0], e, out[1]


def _evaluate_conventions(samples: List[Tuple[np.ndarray, np.ndarray]],
                          cp: _CopyArrays, tr: Transform, m_eff: np.ndarray,
                          a_prim: np.ndarray, prim_cart: np.ndarray,
                          b_prim: np.ndarray, b_super: np.ndarray) -> np.ndarray:
    """在给定的采样上评估 4 种相位约定的平均单 G 主导度。

    samples: [(k_frac (3,), coeffs (n_band, n_ions, norb)), ...]
        每个元素是一组 (k, 该 k 上参与统计的带)。
    返回 (4,) 数组, 顺序同 ``CONVENTIONS``。

    每个 (k, 带) 的主导度 = max_g W / Σ_g W(总权重 > 1e-12 才计入, 与原型
    ``unfold_procar.py`` 的 ``if sw > 1e-12`` 一致), 再对全部样本取平均。
    """
    scores = np.zeros(len(CONVENTIONS))
    for ci, (s1, s2) in enumerate(CONVENTIONS):
        doms = []
        for k_frac, coeffs in samples:
            _, w = _weight_matrix(k_frac, coeffs, cp, tr, m_eff, a_prim,
                                  prim_cart, b_prim, b_super, s1, s2)
            keep = w.sum(axis=0) > _MIN_STATE_WEIGHT
            if keep.any():
                doms.append(_dominance(w)[keep])
        scores[ci] = float(np.concatenate(doms).mean()) if doms else 0.0
    return scores


def _choose_convention(scores: np.ndarray) -> Tuple[int, int, DominanceReport]:
    """在 4 种相位约定里择优 —— **s1 物理唯一解, s2 由集中度定**。

    修复轮 9(实测依据见 ``PHASE_COPY_SIGN`` 的注释):

    * **s1(副本相位符号)= PHASE_COPY_SIGN = −1**, 由超胞 Bloch 因子物理
      唯一确定, **不靠打分** —— 实测"单 G 集中度"对 s1 的分辨力只有 ~1e-3
      (四个约定 0.7653/0.7723/0.7701/0.7680), 而选错 s1 会让 55%~65% 的
      (k,带) 换候选归属 ⇒ 整张费米面错形(Pearson 0.1556 vs 0.9941);
    * **s2(Bloch character 符号)** 在剩下的两个候选里按**单 G 集中度**
      择优 —— 实测相对差 2.13% / 1.91% / 2.04%(三组真实数据)⇒ 有判别力。

    三项检查:

    1. winner 集中度 < MIN_DOMINANCE -> 抛错: 展开没有把态集中到唯一一个
       候选 G 上, 此时的谱函数不可信;
    2. 两个 s1=−1 的候选给不出区分(完全相同) -> 抛错: 相位约定没有意义;
    3. winner 与 runner-up 的相对差 < MIN_CONVENTION_MARGIN_REL -> **告警**,
       且告警必须说明**后果**: 退化区里选出的约定由噪声主导, 而选错约定
       会让整张费米面错形(不是"不可复现"那么轻)。
    """
    scores = np.asarray(scores, dtype=float)
    # 只有 s1 = PHASE_COPY_SIGN 的两个约定是物理上可能的候选
    cand = [i for i, (s1, _s2) in enumerate(CONVENTIONS)
            if s1 == PHASE_COPY_SIGN]
    # 稳定排序: 分数相同时取 CONVENTIONS 里靠前的那个(确定性)
    order = sorted(cand, key=lambda i: (-scores[i], i))
    best, second = order[0], order[1]
    s1, s2 = CONVENTIONS[best]
    if scores[best] < MIN_DOMINANCE:
        raise ValueError(
            f"单 G 主导度 {scores[best]:.4f} < {MIN_DOMINANCE} —— "
            "展开没有把态集中到唯一一个候选 G 上, 相位约定/副本平均不可信")
    if scores[best] - scores[second] <= 1e-12:
        raise ValueError(
            f"s1={PHASE_COPY_SIGN:+d} 的两个相位约定没有区分度(主导度完全"
            "相同) —— 相位约定无法确定, 展开结果没有意义")
    margin = float(scores[best] - scores[second])
    margin_rel = margin / scores[best] if scores[best] > 0 else 0.0
    degenerate = bool(margin_rel < MIN_CONVENTION_MARGIN_REL)
    report = DominanceReport(
        scores=tuple(float(x) for x in scores),
        selection_score=float(scores[best]),
        runner_up_score=float(scores[second]),
        margin=margin,
        margin_rel=float(margin_rel),
        degenerate=degenerate,
    )
    if degenerate:
        warnings.warn(
            f"相位约定择优落在退化区: winner=({s1:+d},{s2:+d}) 主导度 "
            f"{scores[best]:.6f}, runner-up 主导度 {scores[second]:.6f}, "
            f"相对差仅 {margin_rel:.2e} < {MIN_CONVENTION_MARGIN_REL:.0e} —— "
            "选出的约定由噪声主导(换一批采样就可能换一个约定)。"
            "⚠ 后果**不是**'可复现性'问题: 实测选错相位约定会让 55%~65% 的 "
            "(k,带) 换候选归属 ⇒ **整张费米面错形**(实测 Pearson 0.16 vs "
            "0.99, 带能量最近距离 0.154 eV vs 2.05e-05 eV) ⇒ 本次结果的"
            "能带归属与费米面形状**都不可信**; 请加密网格(超胞 BZ 至少被 "
            "4×4 个点采样)或人工核对候选归属",
            UserWarning, stacklevel=3)
    return int(s1), int(s2), report


def select_phase_convention(k_super_frac: np.ndarray, coeffs: np.ndarray,
                            energies: np.ndarray, tr: Transform,
                            copies: AtomCopies, m_eff: np.ndarray,
                            a_prim: np.ndarray, prim_cart: np.ndarray,
                            b_prim: np.ndarray, b_super: np.ndarray
                            ) -> Tuple[int, int, float]:
    """在 4 种 (s1, s2) 相位约定里择优 —— **s1 物理唯一解, s2 靠集中度定**。

    返回 (s1, s2, 主导度)。主导度 < 0.5 视为失败(抛错): 展开没有把态集中到
    唯一一个候选 G 上, 此时的谱函数不可信。两个 s1=−1 的候选给不出区分时同样
    抛错 —— 那说明相位约定本身没有意义。

    判据的物理依据与实测依据见 ``_choose_convention`` 与 ``PHASE_COPY_SIGN``
    的注释(以及 docs/algorithms.md §7.2): ``s1`` 由超胞 Bloch 因子唯一确定,
    ``s2`` 由"态应当集中在唯一一个候选上"分辨。

    口径: **单个 k**、该 k 的全部带(能量非有限的带不参与统计)。
    生产路径(``unfold_supercell``)用的是原型口径 —— 首 min(4, n_k) 个 k ×
    首 min(n_band, 40) 条带 —— 见 ``CONVENTION_SAMPLE_N_K``。
    两者共用同一套评分/择优实现(``_evaluate_conventions`` + ``_choose_convention``)。
    """
    _check_transform(tr, m_eff)
    coeffs = np.asarray(coeffs)
    energies = np.asarray(energies, dtype=float)
    if coeffs.shape[0] != energies.shape[0]:
        raise ValueError(
            f"coeffs 的带数 {coeffs.shape[0]} != energies 的带数 "
            f"{energies.shape[0]}")
    cp = _copy_arrays(copies, a_prim)
    # 能量非有限的带是坏数据, 不参与主导度统计
    good = np.isfinite(energies)
    samples = [(np.asarray(k_super_frac, dtype=float),
                coeffs[good] if good.any() else coeffs)]
    scores = _evaluate_conventions(samples, cp, tr, m_eff, a_prim, prim_cart,
                                   b_prim, b_super)
    s1, s2, _ = _choose_convention(scores)
    return s1, s2, float(scores[CONVENTIONS.index((s1, s2))])


def _match_k(target: np.ndarray, source: np.ndarray,
             tol: float = _K_TOL) -> np.ndarray:
    """把 target 的每个 k 匹配到 source 中周期距离最近的 k(下标)。

    匹配必须**单射**: 两个不同的目标 k 不能落到同一个源 k 上 ——
    否则会静默复用同一组系数, 展开结果悄悄错掉。源 k 点集合缺一个点时,
    这里报"塌缩"而不是笼统的"找不到"。
    """
    t = np.mod(np.atleast_2d(np.asarray(target, dtype=float)), 1.0)
    s = np.mod(np.atleast_2d(np.asarray(source, dtype=float)), 1.0)
    dist = periodic_distance(t, s)                # (n_target, n_source)
    idx = dist.argmin(axis=1)
    best = dist[np.arange(t.shape[0]), idx]
    bad = np.nonzero(best > tol)[0]
    if bad.size:
        i = int(bad[0])
        # 塌缩检测: 是否有**另一个**目标 k 也匹配到了同一个源 k?
        others = np.nonzero((idx == idx[i]) & (np.arange(t.shape[0]) != i))[0]
        if others.size:
            j = int(others[0])
            raise ValueError(
                f"两个不同的目标 k ({t[j].tolist()} 与 {t[i].tolist()}) 匹配到了"
                f"同一个源 k(下标 {int(idx[i])}) —— 匹配发生塌缩, 会静默复用同一"
                "组系数; 检查 EIGENVAL/PROCAR 的 k 点集合是否有重复或缺失")
        raise ValueError(
            f"k 点 {np.asarray(target, dtype=float)[i].tolist()} 在 "
            "EIGENVAL/PROCAR 里找不到(周期距离 "
            f"{best[i]:.2e} > {tol:.0e}) —— k 点集合与计算结果不一致")
    return idx


def _unfold_primitive_direct(band: BandData, procar: ProcarData,
                              band_idx: np.ndarray, pro_idx: np.ndarray,
                              n_band: int, k_list: np.ndarray) -> UnfoldedResult:
    """det(M)=1 的**直通分支**: 原胞自己的能带 + 轨道投影权重。

    det(M)=1 ⇒ "超胞"就是原胞: 没有折叠、没有多份副本。Popescu 路径在这里
    **数学上退化** —— 候选 G 只有一个(g=0), 于是 4 种相位约定的主导度必然
    都等于 1.0(完全相同), 择优守卫会把原胞自己的计算也拒掉。

    ⇒ 这里**不做展开**: 能量 = EIGENVAL 的带能量(逐 k 按能量升序),
    权重 = ``Σ_{ion,lm} |c_{ion,lm,n}(k)|²``(**非相干**和)。

    ⚠ **权重定义与超胞展开模式不同**:
      * 直通模式: Σ_{ion,lm}|c|² —— 逐轨道模方相加(教科书定义);
      * 展开模式: Popescu 的 ``|Σ_i e^{i k'·r_i} Σ_lm p_i(lm)|²`` ——
        先在原胞原子上**相干**求和再取模方。
      两者在 det(M)=1 时给出**不同**的数 ⇒ 两种模式的费米面形状不可直接比较。

    相位约定择优与它的守卫在这里**整体跳过**(没有展开就没有约定可择优);
    ``DominanceReport`` 用 NaN 填充 —— **不伪造分数**。
    """
    n_k = k_list.shape[0]
    w = np.zeros((n_k, n_band))
    for ik in range(n_k):
        c = np.asarray(procar.coeffs[pro_idx[ik], :n_band])   # (nb, n_ions, norb)
        w[ik] = np.sum(np.abs(c) ** 2, axis=(1, 2))
    e = np.asarray(band.energies, dtype=float)[band_idx, :n_band]
    # 逐 k 按能量升序(权重跟着一起重排)—— 与展开模式"候选内按能量升序"同口径,
    # 下游(fsout / band)不需要区分两种模式。
    order = np.argsort(e, axis=1, kind="stable")
    e = np.take_along_axis(e, order, axis=1)
    w = np.take_along_axis(w, order, axis=1)

    k_cand = np.mod(k_list, 1.0)[:, None, :]        # (n_k, 1, 3) 候选只有 g=0
    energies = e[:, None, :]                       # (n_k, 1, n_band)
    weights = w[:, None, :]
    valid = np.ones(energies.shape, dtype=bool)
    report = DominanceReport()                      # 全 NaN: 不适用, 不伪造
    return UnfoldedResult(k_cand=k_cand, energies=energies, weights=weights,
                           s1=1, s2=1, dominance=float("nan"),
                           valid=valid, dominance_report=report)


def unfold_supercell(band: BandData, procar: ProcarData, tr: Transform,
                     copies: AtomCopies, m_eff: np.ndarray,
                     a_prim: np.ndarray, prim_cart: np.ndarray,
                     b_prim: np.ndarray, b_super: np.ndarray,
                     k_super_list: np.ndarray,
                     strict_dominance: bool = False,
                     with_orbitals: bool = False,
                     min_dominance: float = MIN_DOMINANCE) -> UnfoldedResult:
    """展开整个超胞 k 点集合, 并做候选重组。

    k_super_list = grid_map.k_super(长度 n_k)

    候选重组: 每条超胞带归属到权重最大的候选; 候选内按能量升序排列 ——
    于是重组后的第 n 条带就是该候选上的第 n 条原胞带(原胞带序号在对称
    变换下不变, 故逐带输出自洽)。不足 n_band_eff 的候选用权重 0、能量 +inf 补位,
    并在 ``UnfoldedResult.valid`` 里标出哪些槽位是有效的。

    相位约定择优用**原型口径**(首 min(4, n_k) 个 k × 首 min(n_band, 40) 条带,
    见 ``CONVENTION_SAMPLE_N_K`` / ``CONVENTION_SAMPLE_N_BAND``): 已验证的
    原型 ``unfold_procar.py`` 就是这么选的; 用"全部 k × 全部带"择优在 fixture 上
    会选出**不同**的约定(164/240 个 (k,带) 的候选归属改变 -> 费米面改变)。
    选定约定后只做**一次**全量权重 pass。

    ⚠ **判据本身**(修复轮 9): ``s1`` 由物理唯一确定(``PHASE_COPY_SIGN``),
    ``s2`` 按单 G 集中度择优 —— 旧的"四个约定里取集中度最大"分辨不出对错
    (四个分数只差 ~1e-3, 而选错会让 55%~65% 的 (k,带) 换候选归属 =>
    整张费米面错形)。见 ``_choose_convention``。

    **det(M) == 1 时走直通分支**(见 ``_unfold_primitive_direct``): 不做展开,
    权重是轨道投影 Σ_{ion,lm}|c|²; 相位约定择优与它的守卫整体跳过。
    ``strict_dominance`` 在直通模式下无意义(没有主导度可查), 忽略。

    strict_dominance: True 时要求**每个态**的单 G 主导度都 >= MIN_DOMINANCE
        (简报自检①的严格版)。默认关闭 —— 合成 fixture 上有 10.8% 的态低于
        0.5, 严格版会误报; 真实 VASP 数据上应当打开。
    """
    _check_transform(tr, m_eff)
    k_list = np.atleast_2d(np.asarray(k_super_list, dtype=float))
    n_k = k_list.shape[0]
    det_m = tr.det
    band_idx = _match_k(k_list, band.kpoints)
    pro_idx = _match_k(k_list, procar.kpoints)
    n_band_eig = band.energies.shape[1]
    n_band_pro = procar.coeffs.shape[1]
    n_band = min(n_band_eig, n_band_pro)
    if n_band == 0:
        raise ValueError("EIGENVAL / PROCAR 里没有可用的带")
    if n_band_eig != n_band_pro:
        warnings.warn(
            f"EIGENVAL 有 {n_band_eig} 条带而 PROCAR 有 {n_band_pro} 条 —— "
            f"只展开前 {n_band} 条(取两者的最小值); 两个文件的带数不一致, "
            "请检查计算输出", UserWarning, stacklevel=2)

    if det_m == 1:
        return _unfold_primitive_direct(band, procar, band_idx, pro_idx,
                                        n_band, k_list)

    cp = _copy_arrays(copies, a_prim)

    # --- 相位约定择优: 原型口径的采样(首 4 个 k × 首 40 条带)
    k_sel = list(range(min(CONVENTION_SAMPLE_N_K, n_k)))
    band_sel = np.arange(min(CONVENTION_SAMPLE_N_BAND, n_band))
    samples = [(k_list[ik], procar.coeffs[pro_idx[ik], band_sel])
               for ik in k_sel]
    scores = _evaluate_conventions(samples, cp, tr, m_eff, a_prim, prim_cart,
                                   b_prim, b_super)
    s1, s2, report = _choose_convention(scores)

    # --- 单次全量权重 pass(只算选中的约定)
    w_best = np.zeros((n_k, det_m, n_band))
    norb = int(np.asarray(procar.coeffs).shape[2])
    w_orb_best = np.zeros((n_k, det_m, n_band, norb)) if with_orbitals else None
    for ik in range(n_k):
        out = _weight_matrix(k_list[ik], procar.coeffs[pro_idx[ik], :n_band],
                             cp, tr, m_eff, a_prim, prim_cart, b_prim, b_super,
                             s1, s2, with_orbital=with_orbitals)
        w_best[ik] = out[1]
        if with_orbitals:
            w_orb_best[ik] = out[2]

    return _pack_unfolded(w_best, band, band_idx, k_list, m_eff, tr, b_prim,
                          b_super, report, strict_dominance, w_orb_best, norb,
                          n_band, s1, s2, with_orbitals,
                          min_dominance=min_dominance)


def _pack_unfolded(w_best, band, band_idx, k_list, m_eff, tr, b_prim, b_super,
                   report, strict_dominance, w_orb_best, norb, n_band, s1, s2,
                   with_orbitals: bool = False,
                   min_dominance: float = MIN_DOMINANCE) -> UnfoldedResult:
    """主导度统计 + 候选重组 + 打包(两条展开路线共用)。

    w_best : (n_k, det_m, n_band) 每个候选原胞动量上的谱权重

    ``min_dominance``: 单 G 主导度的下限(硬错)。默认 0.5。对**破坏平移对称**
    的态(CDW、缺陷超胞)要放宽 —— 那里的主导度低是**态的性质**(态不是原胞
    平移的本征态), 不是展开的数值失败。
    """
    n_k, det_m, _ = w_best.shape
    # --- 每态主导度的分布 + 聚合下限(硬错)
    dom = _dominance(w_best)                       # (n_k, n_band)
    states = dom[w_best.sum(axis=1) > 0]
    if states.size == 0:
        raise ValueError("所有候选的权重都是 0 —— 展开结果为空")
    dominance = float(states.mean())
    if dominance < min_dominance:
        raise ValueError(
            f"全量展开的平均单 G 主导度 {dominance:.4f} < {min_dominance} —— "
            "展开没有把态集中到唯一一个候选 G 上, 相位约定/副本平均不可信"
            "(对破坏平移对称的态可放宽 --min-dominance)")
    report = replace(report, mean=dominance, min=float(states.min()),
                      median=float(np.median(states)),
                      frac_above_good=float(np.mean(states > DOMINANCE_GOOD)))
    if strict_dominance and float(states.min()) < min_dominance:
        raise ValueError(
            f"严格每态主导度下限: 最差的态只有 {states.min():.4f} < "
            f"{min_dominance} —— 展开把态摊到了多个候选 G 上, 谱函数不可信")

    e_all = np.asarray(band.energies, dtype=float)[band_idx, :n_band]
    k_cand = np.zeros((n_k, det_m, 3))
    for ik in range(n_k):
        _, k_frac, _ = _candidate_k(k_list[ik], m_eff, tr.m_inv.T,
                                    b_prim, b_super)
        k_cand[ik] = k_frac

    # 候选重组: 每条带归属到权重最大的候选, 候选内按能量升序
    assign = w_best.argmax(axis=1)                  # (n_k, n_band)
    nonzero = w_best.sum(axis=1) > 0
    per_cand: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    counts = np.zeros((n_k, det_m), dtype=int)
    for ik in range(n_k):
        row = []
        for j in range(det_m):
            sel = np.nonzero((assign[ik] == j) & nonzero[ik])[0]
            order = sel[np.argsort(e_all[ik, sel], kind="stable")]
            row.append((e_all[ik, order], w_best[ik, j, order]))
            counts[ik, j] = order.size
        per_cand.append(row)
    n_band_eff = int(counts.max())
    energies = np.full((n_k, det_m, n_band_eff), np.inf)
    weights = np.zeros((n_k, det_m, n_band_eff))
    orbital_weights = (np.zeros((n_k, det_m, n_band_eff, norb))
                       if with_orbitals else None)
    valid = np.zeros((n_k, det_m, n_band_eff), dtype=bool)
    for ik in range(n_k):
        for j in range(det_m):
            e, w = per_cand[ik][j]
            energies[ik, j, :e.size] = e
            weights[ik, j, :e.size] = w
            valid[ik, j, :e.size] = True
            if with_orbitals:
                # 同一套 (候选归属 + 能量排序) ⇒ 逐带挑出同一批带
                sel = np.nonzero((assign[ik] == j) & nonzero[ik])[0]
                order = sel[np.argsort(e_all[ik, sel], kind="stable")]
                orbital_weights[ik, j, :order.size] = w_orb_best[ik, j, order]
    return UnfoldedResult(k_cand=k_cand, energies=energies, weights=weights,
                           orbital_weights=orbital_weights,
                           s1=int(s1), s2=int(s2), dominance=dominance,
                           valid=valid, dominance_report=report,
                           band_energies=e_all, band_weights=w_best)



def _plane_wave_primitive_coords(gidx: np.ndarray,
                                 m_inv_T: np.ndarray) -> np.ndarray:
    """平面波的整数索引 -> **原胞倒空间分数坐标**。

    超胞倒格矢基 B_sc = B_prim · M^{-T}(行向量约定) ⇒ 整数索引 n 的平面波,
    在原胞倒空间基下的分数坐标 = n · M^{-T}。
    """
    return gidx.astype(float) @ m_inv_T


def unfold_supercell_plane_waves(band: BandData, wavecar_path: str, tr: Transform,
                                  m_eff: np.ndarray, a_prim: np.ndarray,
                                  prim_cart: np.ndarray, b_prim: np.ndarray,
                                  b_super: np.ndarray, k_super_list: np.ndarray,
                                  strict_dominance: bool = False,
                                  min_dominance: float = MIN_DOMINANCE
                                  ) -> UnfoldedResult:
    """平面波路线: 严格谱权重 w(k') = Σ_g |C(g + k' - K)|²。

    公式出自 Popescu & Zunger, PRB **85**, 085201 (2012) ��录 Eq. (A8);
    实现细节见 :mod:`unfoldfs.pwunfold`。

    与 PROCAR 路线的差别(论文 §II):
    - 权重是**严格的谱权重**(平面波基正交完备)而不是相对量;
    - **没有相位约定**: 权重是模方, 与超胞态的整体相位无关 ⇒ 不做约定择优,
      也就没有"约定落在简并区"的告警;
    - **没有"代表原子平均"这一步** ⇒ 没有那一步带来的协变性破缺。

    WAVECAR 的 k 点顺序**不假定**与 EIGENVAL 相同: 用 WAVECAR 自带的笛卡尔 k
    换算回超胞分数坐标, 再与 EIGENVAL 的 k 逐点匹配(周期距离)。

    det(M) == 1 时没有展开可言, 直接报错(原胞直算另走 PROCAR 路线)。
    """
    _check_transform(tr, m_eff)
    if tr.det == 1:
        raise ValueError("det(M) == 1 没有展开可言; 原胞直算请用 PROCAR 路线")
    k_list = np.atleast_2d(np.asarray(k_super_list, dtype=float))
    n_k = k_list.shape[0]
    n_band_eig = int(band.energies.shape[1])
    header = read_wavecar_header(wavecar_path, nkpts=n_k, nspin=1)
    if header.nbands < n_band_eig:
        raise ValueError(
            "WAVECAR 只有 %d 条带, 少于 EIGENVAL 的 %d 条 —— WAVECAR 与 "
            "EIGENVAL 必须出自同一次计算" % (header.nbands, n_band_eig))
    n_band_use = min(header.nbands, n_band_eig)
    det_m = int(tr.det)
    m_inv_T = np.asarray(tr.m_inv, dtype=float).T
    a_super = np.asarray(a_prim, float) @ m_eff.T   # 超胞晶格(行向量)
    w_best = np.zeros((n_k, det_m, n_band_use))
    completeness = np.zeros((n_k, n_band_use))
    n_matched = 0
    for kp in iter_wavecar_kpoints(wavecar_path, header):
        if kp.index >= n_k:
            break
        # WAVECAR 头里的 k 是**分数** k(超胞晶体坐标, 与 EIGENVAL 同一约定),
        # 直接与 EIGENVAL 的 k 逐点比对(不依赖两个文件的 k 顺序一致)
        k_wc = np.mod(np.asarray(kp.k_frac, dtype=float), 1.0)
        dd_k = np.abs(k_list - k_wc[None, :])
        dd_k = np.minimum(dd_k, 1.0 - dd_k)
        d = np.linalg.norm(dd_k, axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-4:
            raise ValueError(
                "WAVECAR 的第 %d 个 k 点在 EIGENVAL 的 k 列表里找不到"
                "(最近距离 %.2e)" % (kp.index, d[j]))
        # 候选: 每个超胞 k 有 det(M) 个候选原胞 k'
        g_cand, _k_frac, _d_g = _candidate_k(k_wc, m_eff, m_inv_T, b_prim, b_super)
        # 等价类判定: 平面波 n 属于候选 j <=> (K+n)·M^{-T} ≡ k'_j·M^{-T} (mod 1)
        u = np.mod(_plane_wave_primitive_coords(kp.gvec, m_inv_T), 1.0)
        u_cand = np.mod(_plane_wave_primitive_coords(g_cand, m_inv_T), 1.0)
        dd = np.abs(u[:, None, :] - u_cand[None, :, :])
        dd = np.minimum(dd, 1.0 - dd).max(axis=2)          # (nplw, detM)
        labels = np.argmin(dd, axis=1)
        if float(dd.min(axis=1).max()) > _PW_CLASS_TOL:
            raise ValueError(
                "第 %d 个 k 点有平面波无法归入任何一个候选等价类(最大残差 "
                "%.2e) —— 变换矩阵与 WAVECAR 的晶胞不自洽" % (kp.index,
                                                             dd.min(axis=1).max()))
        w, comp = plane_wave_weights(kp.coeffs[:n_band_use].T, labels,
                                     np.arange(det_m))
        w_best[j] = w
        completeness[j] = comp
        n_matched += 1
    if n_matched != n_k:
        raise ValueError("WAVECAR 匹配到 %d 个 k 点, 期望 %d" % (n_matched, n_k))
    # 完备性: Σ_j w_j = 1(候选集是平面波基的划分)。偏离即说明分类有错。
    dev = np.abs(completeness - 1.0)
    if float(dev.max()) > _PW_COMPLETENESS_TOL:
        raise ValueError(
            "完备性 Σ_j w_j = 1 不成立(最大偏差 %.2e) —— 候选等价类的构造有误"
            % float(dev.max()))
    band_idx = np.arange(n_k)
    report = DominanceReport(scores=(), degenerate=False)
    return _pack_unfolded(w_best, band, band_idx, k_list, m_eff, tr, b_prim,
                          b_super, report, strict_dominance, None, 0,
                          n_band_use, s1=1, s2=1, with_orbitals=False,
                          min_dominance=min_dominance)
