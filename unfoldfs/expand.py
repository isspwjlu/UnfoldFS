"""对称扩展: 把展开结果映射回**完整均匀原胞网格**, 并组装谱函数 A(k,E)。

展开(unfold.py)的结果是**按已算超胞 k 索引**的: 每个超胞 k 上有 det(M) 个
候选原胞 k(分数坐标), 每个候选上有一组带能量与谱权重。本模块把它们重新
铺回满网格的每个点 —— 每个网格点恰好由一个 (超胞 k, 候选) 对覆盖。

核心映射(逐满网格点 i):

    k_idx = full_to_k[i]                           该点折到哪个已算超胞 k
    p_i   = ops.apply(ir_k_prim[ir], rot_of_full)  该点的原胞分数坐标(轨道像)
    gset  = candidate_g_for_k(k_super[k_idx])      该 k 的**逐 k**候选折叠矢量
    j     = argmin_j 周期距离( (k_super+g_j)·M^-T , p_i )
    band  = energies/weights/valid[k_idx, j]

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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import periodic_delta, periodic_distance, rotation_inverse
from .grid import GridMap, GridSpec
from .unfold import UnfoldedResult, candidate_g_for_k

__all__ = ["ExpandedData", "expand_to_full_grid", "build_spectral_function"]

# 分数坐标下的匹配容差: 远大于 spglib 平移残差(~3e-8), 又远小于网格间距
_MATCH_TOL = 1e-6

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


def expand_to_full_grid(unfolded: UnfoldedResult, grid_map: GridMap,
                        *, tol: float = _MATCH_TOL) -> ExpandedData:
    """把展开结果铺回完整均匀原胞网格。

    对每个满网格点 i: 找到唯一覆盖它的 (超胞 k, 候选 j), 取该候选的能带块。
    三项自检(任一不通过即抛错, 不静默取错带):

    1. 每个网格点命中一个候选, 且命中距离 < tol;
    2. 命中的候选与展开结果的候选轴**索引一致**(用 unfolded.k_cand 交叉核对)
       —— 候选顺序一旦错位就会在这里被抓到;
    3. (超胞 k, 候选) 与满网格点构成双射(不多不少)。
    """
    gm = grid_map
    n_points = int(gm.full_to_k.shape[0])
    n_k = int(gm.k_super.shape[0])
    det = int(gm.transform.det)
    if unfolded.energies.shape[0] != n_k:
        raise ValueError(
            f"展开结果的 k 数 {unfolded.energies.shape[0]} != grid_map 的 "
            f"{n_k} —— 展开时传入的 k 列表必须就是 grid_map.k_super")
    if unfolded.energies.shape[1] != det:
        raise ValueError(
            f"展开结果的候选数 {unfolded.energies.shape[1]} != det(M)={det}")
    if n_points != n_k * det:
        raise ValueError(
            f"满网格点数 {n_points} != n_k·det(M) = {n_k}×{det} —— 网格与超胞"
            "不自洽(检查网格点数是否为 comm(M^-1) 的倍数)")

    m_eff = np.asarray(gm.transform.m_eff, dtype=float)
    m_inv_T = np.asarray(gm.transform.m_inv, dtype=float).T
    full_to_k = np.asarray(gm.full_to_k, dtype=int)
    p = _grid_frac_points(gm)

    hits = np.full(n_points, -1, dtype=int)
    frac = np.zeros((n_points, 3))
    for k in range(n_k):
        sel = np.nonzero(full_to_k == k)[0]
        if sel.size == 0:
            continue
        kk = np.mod(gm.k_super[k], 1.0)
        # 候选 G **逐 k 计算**: 一般 M 下候选集随 K 平移, 用全局 gset 会取错带
        g = candidate_g_for_k(kk, m_eff, m_inv_T)
        cand = (kk + g) @ m_inv_T            # (det,3) 候选原胞 k, 落在 [0,1)³
        dist = periodic_distance(p[sel], cand)
        j = dist.argmin(axis=1)
        best = dist[np.arange(sel.size), j]
        bad = np.nonzero(best > tol)[0]
        if bad.size:
            i = int(sel[int(bad[0])])
            raise ValueError(
                f"满网格点 {i} 没有被任何候选覆盖(最近距离 "
                f"{best[int(bad[0])]:.2e} > {tol:.0e}) —— grid_map 与展开结果"
                "的 k 集合不一致?")
        hits[sel] = j
        frac[sel] = cand[j]

    # 候选索引必须与展开结果的候选轴一致(k_cand 是展开时算出的候选分数值)
    kc = np.asarray(unfolded.k_cand)[full_to_k, hits]
    d = periodic_delta(kc, p)
    bad = np.nonzero(d > tol)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"候选索引与展开结果不一致: 满网格点 {i} 命中的候选与其原胞 k "
            f"坐标相差 {d[i]:.2e} —— 展开时用的 k 列表顺序必须与 "
            "grid_map.k_super 一致, 否则会静默取错带")

    slots = full_to_k * det + hits
    if np.unique(slots).size != n_points:
        raise ValueError(
            f"(超胞 k, 候选) 映射不是单射: {np.unique(slots).size} 个不同"
            f"组合覆盖了 {n_points} 个满网格点 —— 匹配发生塌缩")

    e = np.asarray(unfolded.energies)[full_to_k, hits]
    w = np.asarray(unfolded.weights)[full_to_k, hits]
    if unfolded.valid is not None:
        v = np.asarray(unfolded.valid, dtype=bool)[full_to_k, hits]
    else:
        v = np.isfinite(e)
    k_cart = frac @ np.asarray(gm.b_prim, dtype=float)
    return ExpandedData(k_cart=k_cart, band_e=e, band_w=w, valid=v,
                         spec=gm.spec)


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
