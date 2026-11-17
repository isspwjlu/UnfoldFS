"""Popescu 投影法超胞能带反折叠(Popescu, PRL 104, 216401 (2012))。

展开公式(移植自已实测验证的原型 ``unfold_procar.py``):

    p_i(lm) = (1/N_i) · Σ_T c_{i+T,lm} · exp(s1·i·(k'−K)·T)     副本相位还原平均
    W(k')   = | Σ_i exp(s2·i·k'·r_i) · Σ_lm p_i(lm) |²           候选 G 上的谱权重

两处**不可改**的语义(原型做对了, 改了就错):

1. **不可**假设同一原胞原子的各副本复值相同 —— 实测同一 (k, 带, 轨道) 上
   9 个副本可取 3 个不同值(按 (m+n) mod 3 分组), 故必须做**相位还原**的副本
   平均, 不能取单个副本、也不能假设副本相等;
2. 相位符号约定 (s1, s2) 由 4 种组合的「单 G 主导度」自动择优(原型实测
   主导度 0.85~1.0, 非主导候选的权重掉到 ~1e-33)。

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

from .geometry import Structure, Transform
from .vaspio import BandData, ProcarData

__all__ = [
    "AtomCopies", "UnfoldedResult", "DominanceReport", "build_atom_copies",
    "candidate_g_for_k", "unfold_at_k", "select_phase_convention",
    "unfold_supercell",
]

# 相位符号约定的 4 种组合 (s1 作用于副本平移项, s2 作用于原胞 k 相位项)
CONVENTIONS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

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

    def n_copies(self) -> np.ndarray:
        """每个原胞原子的副本数。"""
        return np.array([len(lst) for lst in self.copies], dtype=int)


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
    s1: int = 1               # 采用的相位符号约定
    s2: int = 1
    dominance: float = float("nan")   # 选定约定下的单 G 主导度(均值)
    valid: Optional[np.ndarray] = None
    dominance_report: Optional[DominanceReport] = None


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
                   s1: int, s2: int) -> Tuple[np.ndarray, np.ndarray]:
    """候选原胞 k(分数) 与谱权重 W(k') (detM, n_band)。"""
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
    total = np.einsum("gi,gbi->gb", bloch, p.sum(axis=3))
    return k_frac, np.abs(total) ** 2


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
                s1: int, s2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    k_frac, w = _weight_matrix(k_super_frac, coeffs, cp, tr, m_eff,
                                a_prim, prim_cart, b_prim, b_super, s1, s2)
    e = np.repeat(energies[None, :], k_frac.shape[0], axis=0)
    return k_frac, e, w


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
    """从 4 种约定里择优, 并做三项检查。

    1. 主导度 < MIN_DOMINANCE -> 抛错: 展开没有把态集中到唯一一个候选 G 上,
       此时的谱函数不可信;
    2. 4 种约定给不出区分(主导度完全相同) -> 抛错: 相位约定本身没有意义;
    3. winner 与 runner-up 的相对差 < MIN_CONVENTION_MARGIN_REL -> **告警**:
       选择落在退化区(噪声驱动), 必须让用户看见而不是静默接受。
    """
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    best, second, worst = int(order[0]), int(order[1]), int(order[-1])
    s1, s2 = CONVENTIONS[best]
    if scores[best] < MIN_DOMINANCE:
        raise ValueError(
            f"单 G 主导度 {scores[best]:.4f} < {MIN_DOMINANCE} —— "
            "展开没有把态集中到唯一一个候选 G 上, 相位约定/副本平均不可信")
    if scores[best] - scores[worst] <= 1e-12:
        raise ValueError(
            "4 种相位约定没有区分度(主导度完全相同) —— 相位约定无法确定, "
            "展开结果没有意义")
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
            "选出的约定由噪声主导(换一批采样就可能换一个约定), 结果不可复现; "
            "请检查展开输入或扩大采样",
            UserWarning, stacklevel=3)
    return int(s1), int(s2), report


def select_phase_convention(k_super_frac: np.ndarray, coeffs: np.ndarray,
                            energies: np.ndarray, tr: Transform,
                            copies: AtomCopies, m_eff: np.ndarray,
                            a_prim: np.ndarray, prim_cart: np.ndarray,
                            b_prim: np.ndarray, b_super: np.ndarray
                            ) -> Tuple[int, int, float]:
    """在 4 种 (s1, s2) 相位约定里选「单 G 主导度」最高的那个。

    返回 (s1, s2, 主导度)。主导度 < 0.5 视为失败(抛错): 展开没有把态集中到
    唯一一个候选 G 上, 此时的谱函数不可信。4 种约定给不出区分时同样抛错 ——
    那说明相位约定本身没有意义。

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
    d = np.abs(t[:, None, :] - s[None, :, :])
    d = np.minimum(d, 1.0 - d)                     # 周期距离
    dist = np.linalg.norm(d, axis=2)              # (n_target, n_source)
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


def unfold_supercell(band: BandData, procar: ProcarData, tr: Transform,
                     copies: AtomCopies, m_eff: np.ndarray,
                     a_prim: np.ndarray, prim_cart: np.ndarray,
                     b_prim: np.ndarray, b_super: np.ndarray,
                     k_super_list: np.ndarray,
                     strict_dominance: bool = False) -> UnfoldedResult:
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
    for ik in range(n_k):
        _, w = _weight_matrix(k_list[ik], procar.coeffs[pro_idx[ik], :n_band],
                              cp, tr, m_eff, a_prim, prim_cart, b_prim, b_super,
                              s1, s2)
        w_best[ik] = w

    # --- 每态主导度的分布 + 聚合下限(硬错)
    dom = _dominance(w_best)                       # (n_k, n_band)
    states = dom[w_best.sum(axis=1) > 0]
    if states.size == 0:
        raise ValueError("所有候选的权重都是 0 —— 展开结果为空")
    dominance = float(states.mean())
    if dominance < MIN_DOMINANCE:
        raise ValueError(
            f"全量展开的平均单 G 主导度 {dominance:.4f} < {MIN_DOMINANCE} —— "
            "展开没有把态集中到唯一一个候选 G 上, 相位约定/副本平均不可信")
    report = replace(report, mean=dominance, min=float(states.min()),
                      median=float(np.median(states)),
                      frac_above_good=float(np.mean(states > DOMINANCE_GOOD)))
    if strict_dominance and float(states.min()) < MIN_DOMINANCE:
        raise ValueError(
            f"严格每态主导度下限: 最差的态只有 {states.min():.4f} < "
            f"{MIN_DOMINANCE} —— 展开把态摊到了多个候选 G 上, 谱函数不可信")

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
    valid = np.zeros((n_k, det_m, n_band_eff), dtype=bool)
    for ik in range(n_k):
        for j in range(det_m):
            e, w = per_cand[ik][j]
            energies[ik, j, :e.size] = e
            weights[ik, j, :e.size] = w
            valid[ik, j, :e.size] = True
    return UnfoldedResult(k_cand=k_cand, energies=energies, weights=weights,
                           s1=int(s1), s2=int(s2), dominance=dominance,
                           valid=valid, dominance_report=report)
