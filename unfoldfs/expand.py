"""对称扩展: 把展开结果映射回**完整均匀原胞网格**, 并组装谱函数 A(k,E)。

展开(unfold.py)的结果是**按已算超胞 k 索引**的: 每个超胞 k 上有 det(M) 个
候选原胞 k(分数坐标), 每个候选上有一组带能量与谱权重。本模块把它们重新
铺回满网格的每个点 —— 一个网格点可以由**多个** (超胞 k, 候选) 对覆盖
(多源求和, 修复轮 8)。

核心映射(**正向**: 候选 -> 网格点):

    g     = candidate_g_for_k(K)                   该 k 的**逐 k**候选折叠矢量
    cand  = (K+g)·M^-T                             候选原胞 k(落在 [0,1)³)
    i     = 网格点里周期距离最近的那个(cand)        多个源可以落到同一点
    band  = energies/weights/valid[k, j]           该 (超胞 k, 候选) 的贡献

**多源求和(修复轮 8)**: 原胞的谱函数是**所有**落到同一个原胞 k' 的超胞态的
投影之和 —— 每个超胞 k 点 K 各贡献一份(经由自己的 G 分量 G_K = k'·Mᵀ − K)。
⇒ 同一个网格点被几个超胞 k 覆盖, 权重就该是几份之和; 只取一份就丢掉
(n_k−1)/n_k 的权重。多个源落在同一点时, 它们的带**按能量对齐**后合并
(能量差 <= merge_tol 的带权重相加)。⚠ 当前 ``unfoldfs gen`` 写出的 KPOINTS
是**折叠网格**(每个网格点恰好折到一个超胞 k) ⇒ 实测每个网格点的源数
min = max = 1 ⇒ 多源求和在当前流程里是**恒等变换** —— 见 docs/algorithms.md
§8.7。

三处**不可改**的语义(本项目已踩过的坑):

1. **候选 G 必须逐 k 计算** —— 一般 M 下逐 k 的候选集与全局 ``gset`` 的
   **成员**不同, 用 gset 定索引会静默取错带(见 contract-v2.md §2);
2. **比较必须用周期距离** —— spglib 的平移代表元带 ~3e-8 残差, 且取模后
   0.99999997 与 0.0 是同一个点(非周期距离会选错候选);
3. **补位槽必须排除** —— ``valid=False`` 的槽能量是 +inf, 参与展宽会把
   谱函数污染成 NaN。

k 侧约定: ``k'_prim = (K + g)·(M^-1)^T`` —— 倒空间用 M^-T, **注意转置**。

对称操作作用到倒空间分数坐标上是 ``k @ R^-1 + t``(分数基的伴随是 R^-1,
不是 R^T)—— 与 ``geometry.SymmetryOps.apply`` 同口径。

``k_cart`` 取**候选的**分数坐标(落在 [0,1)³ 内)而不是轨道像 p_i: 后者经
``ops.apply`` 取模后可能出现 0.99999997(真实格点是 0.0), 直接用会让
笛卡尔坐标整整差一个倒格矢。

**点群轨道对称化(修复轮 5)**: 展开得到的 A(k,E) 在晶体点群下**不协变** ——
同一轨道上的对称相关点给出不同的谱权重(机制: Popescu 副本平均用的是**一个
代表原子**的位置, 对称操作把代表原子映到另一个原子(附带原胞晶格平移 Δᵢ)时
相位多出 (k'−K)·Δᵢ 项)。``symmetrize_spectral_function`` 把 A(k,E) 按轨道
平均, 正好抵消这一项(实测最大相对偏差 1.000e+00 -> 0.000e+00), 即把展开谱
**投影到对称不变子空间** —— 见 docs/algorithms.md §15。

**用哪个点群(修复轮 6)**: 投影的目标子空间是"**原胞点群**下不变"的子空间
(原胞的谱函数按构造就在原胞点群下不变 ⇒ 这个投影有物理依据)。因此对称化
必须用**原胞的完整点群**(:func:`geometry.primitive_symmetry_ops`), 而不是
"原胞与超胞的公共对称操作"(``grid_map.ops`` —— 后者对破坏对称性的超胞只剩
4/12 个操作, 六重对称恢复不了)。实测(3×1×1 的 1T-TaS2): 完整点群下最大相对
偏差 1.000e+00 -> 0.000e+00, 而只用公共子群仍是 1.000e+00 —— 见
docs/algorithms.md §15.8。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import SymmetryOps, periodic_delta, rotation_inverse
from .grid import (GridMap, GridSpec, _PeriodicIndex, filter_ops_by_grid,
                   primitive_grid_points)
from .unfold import UnfoldedResult, candidate_g_for_k

__all__ = ["ExpandedData", "expand_to_full_grid", "build_spectral_function",
           "point_group_orbits", "symmetrize_spectral_function",
           "full_point_group"]

# 分数坐标下的匹配容差: 远大于 spglib 平移残差(~3e-8), 又远小于网格间距
_MATCH_TOL = 1e-6

# 多源合并时"同一条带"的能量容差(eV): 两个源里能量差小于它的带视为同一条,
# 权重相加(简并带本来就该在谱函数里相加)
_MERGE_TOL = 1e-6

# 谱函数分块计算时每块的最大元素数(控制峰值内存)
_SPECTRAL_BLOCK = 2_000_000


@dataclass(frozen=True)
class ExpandedData:
    """铺回满网格后的能带数据。

    k_cart (n_points,3) 原胞笛卡尔坐标(2π/Å), 顺序 = 满网格点序
    band_e (n_points, n_band_eff) 带能量; **补位槽是 +inf**
    band_w (n_points, n_band_eff) 谱权重; 补位槽是 0
    valid (n_points, n_band_eff) 有效位掩码; 补位槽为 False
             (下游若对带轴做 argmin/求最小值, 必须用它排除 +inf 补位槽)
    spec  网格规格(与 grid_map.spec 一致)
    """

    k_cart: np.ndarray
    band_e: np.ndarray
    band_w: np.ndarray
    valid: np.ndarray
    spec: GridSpec


def _grid_frac_points(gm: GridMap) -> np.ndarray:
    """每个满网格点的原胞分数坐标(轨道像)。

    轨道代表点 ``ir_k_prim[ir]`` 经 ``rot_of_full[i]`` 号对称操作即得网格
    点 i; ``rot_of_full[i] == -1`` 表示该点就是代表点本身。

    ⚠ 对称操作作用到**倒空间**分数坐标上是 ``k @ R^-1 + t``(不是 ``k @ R^T``) ——
    分数基下的算符不是正交阵。与 ``SymmetryOps.apply`` 同口径(两者必须一致,
    否则轨道像与网格点对不上, 展开会报"没有候选覆盖")。
    """
    ir = np.asarray(gm.full_to_ir, dtype=int)
    rot = np.asarray(gm.rot_of_full, dtype=int)
    idx = np.where(rot < 0, 0, rot)
    rots = gm.ops.rotations[idx]            # (n,3,3)
    trans = gm.ops.translations[idx]        # (n,3)
    invs = np.stack([rotation_inverse(r) for r in rots])   # (n,3,3) 整数
    p = np.einsum("oj,ojk->ok", gm.ir_k_prim[ir], invs) + trans
    p = np.mod(p, 1.0)
    fixed = rot < 0
    if fixed.any():
        p[fixed] = gm.ir_k_prim[ir[fixed]]
    return p


def _wrap01(v: np.ndarray) -> np.ndarray:
    """把分数坐标折回 [0,1)³。

    ⚠ ``np.mod`` 对**极小的负数**会返回精确的 1.0(而不是 0)⇒ 直接用它做
    折叠会让"折回后的 K"偶尔带着一个 1.0 分量, 于是候选坐标整整差一个倒格矢
    (实测: √3×√3 超胞的 K=(0.0625,0,0) 反推成 (0.0625,1,0) ⇒ 候选的分数值
    偏 0.47)。⇒ 这里把 1.0 显式改回 0.0。
    """
    out = np.mod(np.asarray(v, dtype=float), 1.0)
    return np.where(out >= 1.0, 0.0, out)


def _merge_source_bands(sources: List[Tuple[np.ndarray, np.ndarray]],
                        merge_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """把多个源的带按**能量对齐**合并: 能量相同(差 <= merge_tol)的带权重相加。

    ``sources`` = [(能量数组, 权重数组), ...]; 每个数组已按能量升序、且**不含**
    补位槽(能量 +inf)。

    * 合并只发生在**不同源之间**: 同一个源里的简并带保持分开(逐带输出依赖
      这个区分 —— 两条简并带是两条带, 不是一条);
    * 同一个源内部的匹配是**单调**的(第 n 条带只会匹配到第 n 条已合并带之后
      的位置)⇒ 简并带在源之间也能一一对上;
    * 合并后的列表按能量升序(稳定排序 ⇒ 源 0 的顺序不变)。
    """
    merged_e: List[float] = []
    merged_w: List[float] = []
    merged_src: List[set] = []
    for s, (e_src, w_src) in enumerate(sources):
        cursor = 0
        for e, w in zip(np.asarray(e_src, float), np.asarray(w_src, float)):
            hit = -1
            for i in range(cursor, len(merged_e)):
                if abs(float(e) - merged_e[i]) <= merge_tol and \
                        s not in merged_src[i]:
                    hit = i
                    break
            if hit < 0:
                merged_e.append(float(e))
                merged_w.append(float(w))
                merged_src.append({s})
            else:
                merged_w[hit] += float(w)
                merged_src[hit].add(s)
                cursor = hit + 1
    if not merged_e:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    e = np.array(merged_e, dtype=float)
    w = np.array(merged_w, dtype=float)
    order = np.argsort(e, kind="stable")
    return e[order], w[order]


def expand_to_full_grid(unfolded: UnfoldedResult, grid_map: GridMap,
                        *, tol: float = _MATCH_TOL,
                        merge_tol: float = _MERGE_TOL) -> ExpandedData:
    """把展开结果铺回完整均匀原胞网格(**多源求和**)。

    每个超胞 k 点 K 的每个候选 G 都落到**一个**原胞网格点上; **同一个网格点
    可以被多个源覆盖** —— 那些源的带按能量对齐后合并、权重相加。

    **为什么需要多源**: 原胞的谱函数是**所有**落到同一个原胞 k' 的超胞态的
    投影之和; 每个超胞 k 点 K 各贡献一份(经由自己的 G 分量 G_K = k'·Mᵀ − K)。
    当超胞 k 列表覆盖了原胞第一布里渊区里的**多个超胞 BZ 副本**(k 列表含有
    相差一个超胞倒格矢的 k 点)时, 同一个原胞 k' 会被 det(M) 个超胞 k 覆盖 ——
    只取其中一个就丢掉 (n_k−1)/n_k 的谱权重。

    **单源情形逐位等价于旧实现**: ``grid_map.k_super`` 给出的折叠网格里每个
    原胞网格点恰好被一个 (K, 候选) 覆盖(见 ``grid.supercell_kpoints`` 的折叠
    K = k'·Mᵀ) ⇒ 多源列表每个只有一项 ⇒ 合并是恒等操作。

    三项自检(任一不通过即抛错, 不静默取错带):

    1. 每个候选都落在某个满网格点上(距离 < tol), 每个满网格点至少被一个
       源覆盖;
    2. 候选与展开结果的候选轴**索引一致**(用 ``unfolded.k_cand`` 交叉核对)
       —— 候选顺序一旦错位就会在这里被抓到;
    3. 前 ``n_k`` 行(``grid_map.k_super`` 给出的那些 k)的 (K, 候选) 与满网格点
       构成双射(不多不少)—— 匹配塌缩会在这里被抓到。

    参数
    ----
    unfolded : 展开结果; k 列表必须**包含** ``grid_map.k_super`` 的全部 k 点
                (前 ``n_k`` 行按顺序对应它们), 额外的行是超胞 BZ 的其它副本
                (它们的 K 由 ``k_cand`` 反推: K = k_cand·Mᵀ mod 1)
    grid_map : 网格与变换
    tol      : 候选与网格点之间的匹配容差(分数坐标下的周期距离)
    merge_tol: 多源合并时"同一条带"的能量容差(eV)
    """
    gm = grid_map
    n_points = int(gm.full_to_k.shape[0])
    n_k = int(gm.k_super.shape[0])
    det = int(gm.transform.det)
    n_rows = int(unfolded.energies.shape[0])
    if n_rows < n_k:
        raise ValueError(
            f"展开结果的 k 数 {n_rows} < grid_map 的 {n_k} —— 展开时传入的 "
            "k 列表必须至少包含 grid_map.k_super 的全部 k 点")
    if unfolded.energies.shape[1] != det:
        raise ValueError(
            f"展开结果的候选数 {unfolded.energies.shape[1]} != det(M)={det}")
    if n_points != n_k * det:
        raise ValueError(
            f"满网格点数 {n_points} != n_k·det(M) = {n_k}×{det} —— 网格与超胞"
            "不自洽(检查网格点数是否为 comm(M^-1) 的倍数)")

    m_eff = np.asarray(gm.transform.m_eff, dtype=float)
    m_inv_T = np.asarray(gm.transform.m_inv, dtype=float).T
    p = _grid_frac_points(gm)
    # 网格点的周期索引: 候选 -> 网格点 的正向匹配(替代旧的"按 k 分组反查")
    index = _PeriodicIndex(p, tol=tol)

    energies = np.asarray(unfolded.energies, dtype=float)
    weights = np.asarray(unfolded.weights, dtype=float)
    if unfolded.valid is not None:
        valid = np.asarray(unfolded.valid, dtype=bool)
    else:
        valid = np.isfinite(energies)
    kcand = np.asarray(unfolded.k_cand, dtype=float)

    # 每个网格点的源列表: [(能量, 权重, 候选分数坐标), ...]
    src_e: List[List[np.ndarray]] = [[] for _ in range(n_points)]
    src_w: List[List[np.ndarray]] = [[] for _ in range(n_points)]
    src_frac: List[List[np.ndarray]] = [[] for _ in range(n_points)]
    primary: Dict[int, Tuple[int, int]] = {}      # 网格点 -> (K, 候选)(前 n_k 行)
    for k in range(n_rows):
        # 前 n_k 行的 K 取自 grid_map.k_super; 额外的行(BZ 副本)由 k_cand
        # 反推: K = k_cand·Mᵀ mod 1(g 是整数矢量, 取模把它消掉)
        if k < n_k:
            kk = _wrap01(gm.k_super[k])
        else:
            kk = _wrap01(kcand[k, 0] @ m_eff.T)
        # 候选 G **逐 k 计算**: 一般 M 下候选集随 K 平移, 用全局 gset 会取错带
        g = candidate_g_for_k(kk, m_eff, m_inv_T)
        cand = (kk + g) @ m_inv_T            # (det,3) 候选原胞 k, 落在 [0,1)³

        # 候选索引必须与展开结果的候选轴一致(k_cand 是展开时算出的候选分数值)
        d = periodic_delta(kcand[k], cand)
        bad = np.nonzero(d > tol)[0]
        if bad.size:
            j = int(bad[0])
            raise ValueError(
                f"候选索引与展开结果不一致: 第 {k} 行的候选 {j} 与其原胞 k "
                f"坐标相差 {d[j]:.2e} —— 展开时用的 k 列表顺序必须与 "
                "grid_map.k_super 一致, 否则会静默取错带")

        hit = index.find(cand)               # (det,) 命中的网格点(-1 = 没命中)
        miss = np.nonzero(hit < 0)[0]
        if miss.size:
            j = int(miss[0])
            raise ValueError(
                f"第 {k} 行的候选 {j} 没有落在任何满网格点上 —— grid_map 与"
                "展开结果的 k 集合不一致?")
        for j in range(det):
            i = int(hit[j])
            v = valid[k, j]
            src_e[i].append(energies[k, j][v])
            src_w[i].append(weights[k, j][v])
            src_frac[i].append(cand[j])
            if k < n_k:
                if i in primary:
                    raise ValueError(
                        f"(超胞 k, 候选) 映射不是单射: 满网格点 {i} 同时被 "
                        f"({primary[i][0]}, {primary[i][1]}) 和 ({k}, {j}) 命中 "
                        "—— 匹配发生塌缩")
                primary[i] = (k, j)

    # 每个满网格点至少被一个源覆盖
    empty = [i for i in range(n_points) if not src_e[i]]
    if empty:
        i = int(empty[0])
        raise ValueError(
            f"满网格点 {i} 没有被任何候选覆盖 —— grid_map 与展开结果的 k "
            "集合不一致?")

    # 多源合并: 同一网格点的各源的带按能量对齐后相加
    merged = [_merge_source_bands(list(zip(src_e[i], src_w[i])), merge_tol)
              for i in range(n_points)]
    # 带轴长度: 不小于展开结果原来的带轴长度(补位槽的口径保持不变)
    n_band_eff = int(energies.shape[2])
    for e, _ in merged:
        n_band_eff = max(n_band_eff, int(e.size))
    band_e = np.full((n_points, n_band_eff), np.inf)
    band_w = np.zeros((n_points, n_band_eff))
    out_valid = np.zeros((n_points, n_band_eff), dtype=bool)
    frac = np.zeros((n_points, 3))
    for i in range(n_points):
        e, w = merged[i]
        m = int(e.size)
        band_e[i, :m] = e
        band_w[i, :m] = w
        out_valid[i, :m] = True
        # k_cart 取**候选的**分数坐标(落在 [0,1)³ 内)而不是轨道像 p_i
        frac[i] = src_frac[i][0]
    k_cart = frac @ np.asarray(gm.b_prim, dtype=float)
    return ExpandedData(k_cart=k_cart, band_e=band_e, band_w=band_w,
                         valid=out_valid, spec=gm.spec)


def build_spectral_function(expanded: ExpandedData, e_grid: np.ndarray,
                            sigma: float) -> np.ndarray:
    """谱函数 A(k,E) = Σ_n w_n(k)·exp(−(E−E_n)²/2σ²)。

    返回 (n_points, n_E)。**补位槽(valid=False, 能量 +inf)必须排除** ——
    否则 +inf 能量会把展宽污染成 NaN。
    """
    e = np.atleast_1d(np.asarray(e_grid, dtype=float))
    sigma = float(sigma)
    if not sigma > 0.0:
        raise ValueError(f"sigma 必须为正, 给出 {sigma}")
    band_e = np.asarray(expanded.band_e, dtype=float)
    band_w = np.asarray(expanded.band_w, dtype=float)
    valid = np.asarray(expanded.valid, dtype=bool)
    # 补位槽: 能量置 0(不再有 +inf)、权重置 0(本就不贡献)
    e_valid = np.where(valid, band_e, 0.0)
    w_valid = np.where(valid, band_w, 0.0)

    n_points = e_valid.shape[0]
    out = np.empty((n_points, e.size), dtype=float)
    block = max(1, int(_SPECTRAL_BLOCK // max(e_valid.shape[1] * e.size, 1)))
    for start in range(0, n_points, block):
        sl = slice(start, min(start + block, n_points))
        d = e[None, None, :] - e_valid[sl][:, :, None]
        out[sl] = np.einsum("pn,pne->pe", w_valid[sl],
                            np.exp(-d * d / (2.0 * sigma * sigma)))
    return out


# ------------------------------------------------------------------ 点群轨道对称化

def point_group_orbits(grid: np.ndarray, ops: SymmetryOps, *,
                       tol: Optional[float] = None) -> List[np.ndarray]:
    """把网格点按**原胞点群轨道**分组(修复轮 5)。

    轨道 = 用 ``ops`` 的每个操作把点 i 映到网格上的所有点(**去重**): 动量作用
    是 ``k·R⁻¹ + t``(见 ``geometry.SymmetryOps.apply`` —— 倒空间的伴随是 R⁻¹,
    不是 Rᵀ)。返回每个轨道的成员下标数组(升序)。

    * 只有**把网格映射到自身**的操作才参与(``grid.filter_ops_by_grid`` 已
      做过筛选); 映不回网格上的像(周期距离 > tol)不参与;
    * 去重是必须的: 不同操作可能落到同一个格点(恒等操作永远落在自己身上),
      不去重会把同一个点重复计入平均。
    * ``tol=None`` 时由网格自身推出(与 ``grid.decompose_grid`` 同一口径)。

    ⇒ 本函数给出的轨道与 ``grid.decompose_grid`` 的轨道分解**一致**
    (``grid_map.full_to_ir`` 正是按它分组的) —— 见 tests/test_expand.py 的
    ``test_point_group_orbits_match_the_grid_map_decomposition``。
    """
    pts = np.mod(np.atleast_2d(np.asarray(grid, dtype=float)), 1.0)
    n = int(pts.shape[0])
    index = _PeriodicIndex(pts, tol=tol)
    n_ops = int(ops.rotations.shape[0])
    match = np.empty((n, n_ops), dtype=int)
    for o in range(n_ops):
        match[:, o] = index.find(ops.apply(pts, o))
    seen = np.zeros(n, dtype=bool)
    orbits: List[np.ndarray] = []
    for i in range(n):
        if seen[i]:
            continue
        members = {i}
        for o in range(n_ops):
            j = int(match[i, o])
            if j >= 0:
                members.add(j)
        idx = np.array(sorted(members), dtype=int)
        seen[idx] = True
        orbits.append(idx)
    return orbits


def full_point_group(grid_map: GridMap) -> SymmetryOps:
    """从 grid_map 复原**原胞的完整点群**(不做 M 过滤, 也不做网格过滤)。

    ``grid_map.ops`` 只装了"原胞与超胞的**公共**对称操作"(判据
    ``M^-T·R·M^T ∈ Z³``); 被剔除的那些 —— 原胞点群里**不保持超胞晶格**的
    操作(破坏对称性的超胞会剔掉一大半), 或不把当前网格映到自身的操作 ——
    **带着旋转与平移**记在 ``grid_map.dropped`` 里(``unfoldfs gen`` 就是
    这么记的, 见 ``grid.save_grid_map``)⇒ 两者合起来 = 原胞的完整点群。

    ⚠ 若 grid_map 不是由 ``unfoldfs gen`` 生成的(``dropped`` 为空), 复原
    出来的就只有公共子群 —— 调用方应当显式传 ``ops``
    (CLI 就是这么做的: ``geometry.primitive_symmetry_ops(prim)``)。
    """
    rots = [np.asarray(r, dtype=int) for r in grid_map.ops.rotations]
    trans = [np.asarray(t, dtype=float) for t in grid_map.ops.translations]
    for entry in (getattr(grid_map, "dropped", ()) or ()):
        rots.append(np.asarray(entry[0], dtype=int))
        trans.append(np.asarray(entry[1], dtype=float))
    seen = set()
    keep_r, keep_t = [], []
    for r, t in zip(rots, trans):
        key = (tuple(np.asarray(r, dtype=int).ravel()),
               tuple(np.round(np.mod(np.asarray(t, dtype=float), 1.0), 6)))
        if key in seen:
            continue
        seen.add(key)
        keep_r.append(r)
        keep_t.append(t)
    return SymmetryOps(np.array(keep_r), np.array(keep_t))


def symmetrize_spectral_function(a: np.ndarray, grid_map: GridMap,
                                 grid: Optional[np.ndarray] = None, *,
                                 ops: Optional[SymmetryOps] = None,
                                 tol: Optional[float] = None) -> np.ndarray:
    """按**原胞完整点群**的轨道平均谱函数 A(k,E) —— 投影到对称不变子空间。

    **为什么需要它**: 展开得到的 A(k,E) 在晶体点群下**不协变** —— 同一轨道上
    的对称相关点给出不同的谱权重(实测最大相对偏差 MoS2 9.186e-01 /
    1T-TaS2 1.000e+00, 即"一个谱峰被映到 0")。机制见 docs/algorithms.md §15:
    Popescu 副本平均用的是**一个代表原子**的位置, 对称操作把代表原子映到
    另一个原子(附带原胞晶格平移 Δᵢ)时相位会多出 (k'−K)·Δᵢ 项 ⇒ 权重不协变。
    **按轨道平均正好抵消这一项** ⇒ 偏差降到机器精度。

    **为什么用原胞的*完整*点群**(修复轮 6): 原胞的谱函数按构造就在原胞点群
    下不变 ⇒ 把展开谱投影到"在这个点群下不变"的子空间是**有物理依据**的。
    若只用"超胞还保留的那部分操作"(= ``grid_map.ops``, 判据
    ``M^-T·R·M^T ∈ Z³``), 破坏对称性的超胞(如 3×1×1)只剩 4/12 个操作
    ⇒ 六重对称恢复不了。**任何**超胞都应当能折回原胞的对称性 ⇒ 用完整点群。

    参数
    ----
    a        : (n_points,) 或 (n_points, n_E) 的谱函数(点序 = 网格点序)
    grid_map : 网格与对称操作的来源
    grid     : 网格点的原胞分数坐标; 缺省时由 ``grid_map.spec`` 重建
               (``primitive_grid_points``)
    ops      : 用来做轨道平均的对称操作; **缺省 = 原胞完整点群**
               (由 ``full_point_group(grid_map)`` 复原)。显式传入
               ``ops=grid_map.ops`` 则只用"原胞与超胞的公共子群"。
               无论哪种来源, 都会**按网格筛选**(只有把网格映到自身的操作
               才能参与 —— 否则轨道不是群轨道, 平均后的结果也不再协变)。
    tol      : 匹配容差(分数坐标下的周期距离); None = 由网格自身推出

    返回
    ----
    与 ``a`` 同形状的新数组: 每个点取它在**轨道**上的平均值。

    ⚠ **这是投影, 不是美化**: 原胞的谱函数本来就该是对称的, 所以把它投影到
    "对称不变子空间"是**无偏**的; 但它确实改动了权重(能带**能量**本来就已经
    是对称的, 见 docs/algorithms.md §15.3)。

    ⚠ **"本来就没有对称性破缺"时是逐位无操作**: 轨道上各点完全相同时直接取
    该值(而不是先求平均), 保证结果**逐位不变** —— 例如保留全部 12 个操作的
    √3×√3 R30° 莫尔超胞(见 docs/algorithms.md §15.7)。
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        raise ValueError("a 必须是 (n_points,) 或 (n_points, n_E) 的数组")
    pts = (primitive_grid_points(grid_map.spec) if grid is None
           else np.atleast_2d(np.asarray(grid, dtype=float)))
    pts = np.mod(pts, 1.0)
    if int(arr.shape[0]) != int(pts.shape[0]):
        raise ValueError(
            f"A(k,E) 的点数 {arr.shape[0]} != 网格点数 {pts.shape[0]} —— "
            "对称化要求 A 的点序与网格点序一致")
    if ops is None:
        ops = full_point_group(grid_map)
    # 按网格筛选: 只有把网格映到自身的操作才参与(否则轨道不是群轨道)
    ops = filter_ops_by_grid(ops, pts, tol=tol)
    out = np.array(arr, dtype=float, copy=True)
    for members in point_group_orbits(pts, ops, tol=tol):
        vals = arr[members]
        mean = vals.mean(axis=0)
        # 轨道上各点完全相同时直接取该值: 保证"无对称性破缺"的情形逐位不变
        same = np.all(vals == vals[0], axis=0)
        out[members] = np.where(same, vals[0], mean)
    return out
