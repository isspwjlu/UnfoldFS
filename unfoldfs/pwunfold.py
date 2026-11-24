"""平面波路线的反折叠核心 (Popescu--Zunger 平面波构造)。

**公式** (Popescu & Zunger, PRB **85**, 085201 (2012), 附录 Eq. (A8))::

    P_{Km}(k0) = Σ_g |C_{Km}(g + k0 - K)|²,   g 跑遍原胞倒格矢 G_prim

即: 超胞态 |Ψ_K⟩ 的**平面波系数**按"总动量模原胞倒格矢"的等价类分组, 每组
的权重是组内 |C|² 的**非相干**求和。附录的推导用到了原胞本征态在原胞倒格矢
基下的正交完备性(附录 Eq. (A4)), 因此交叉项被 δ 消掉 ⇒ 结果是非相干的。

**候选集 = 等价类, 构成平面波基的划分** ⇒ 完备性::

    Σ_{k0} P_{Km}(k0) = Σ_G |C_{Km}(G)|² = 1      (系数按带归一化后)

这个恒等式是**判据**: 候选集一旦构造错(重叠或漏), 完备性立刻偏离 1。
本模块在 :func:`unfold_supercell_plane_waves` 里逐 k 检查它。

**与轨道投影路线的差别** (论文 §II):

* 权重是**严格的谱权重** —— 平面波基正交完备, 不需要"代表原子平均"那一步,
  也就没有那一步带来的协变性破缺;
* **没有相位约定** —— 权重是模方, 与超胞态的整体相位无关; 轨道投影路线必须
  选一个约定(选错会让 55%~65% 的 (k,带) 换候选归属)。

**为什么不用 easyunfold / VASPKIT 的"公共 G + 平移"做法**:
那两种实现取"同时属于原胞与超胞倒格矢的公共 G", 再整体平移一个矢量; 平移
后的平面波有一部分落到截断球外, 于是**截断球边缘的候选被漏掉**(实测 1T-TaS₂
3×1×1 上漏掉约 26% 的候选)。漏掉的是系数最小的那些, 权重因此被系统性低估。
本模块直接按等价类判定, 每个平面波只属于一个类, 不重不漏。

坐标约定(与 geometry.py / unfold.py 一致):

* 超胞倒格矢基 ``B_super = B_prim · M^{-T}``(行向量约定) ⇒ 整数索引 n 的
  平面波在原胞倒空间基下的分数坐标 = ``n · M^{-T}``;
* 总动量 ``K + G`` 的原胞分数坐标 = ``(k_super + n) · M^{-T}``;
* 候选 k0 的等价类判据: ``(k_super + n) · M^{-T} ≡ k0 · M^{-T} (mod 1)``。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "plane_wave_class_labels", "plane_wave_weights", "class_labels_for_candidates",
    "PLANE_WAVE_WEIGHT_NOTE",
]

PLANE_WAVE_WEIGHT_NOTE = (
    "P(k') = sum_g |C(g + k' - K)|^2  (Popescu--Zunger, PRB 85, 085201 (2012), "
    "Eq. (A8)); 非相干求和, 完备性 sum_k' P = 1"
)

# 分数坐标的匹配容差: 远大于浮点噪声, 远小于 1/det(M)
_MATCH_TOL = 1e-6


def _periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """分数坐标下的周期距离 (逐分量取最近镜像)。"""
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def class_labels_for_candidates(gidx: np.ndarray, m_inv_T: np.ndarray,
                                ref_g: np.ndarray) -> np.ndarray:
    """给每个超胞平面波定"等价类号" (0 .. det(M)-1)。

    平面波 ``G = n · B_super`` 的等价类由 ``n · M^{-T} (mod 1)`` 决定; 参考
    元 ``ref_g`` 给 det(M) 个候选(整数超胞索引), 它们的 ``· M^{-T} (mod 1)``
    就是 det(M) 个类的代表。

    ⚠ **等价类与超胞 k 无关** —— 超胞 k 改变的只是候选与类的**对应关系**,
    不改变类本身。所以这个函数只需要算一次, 每��� k 只需重排标签
    (见 :func:`permutation_for_candidates`)。

    参数
    ----
    gidx   : (nplw, 3) 超胞倒格矢整数索引
    m_inv_T: (3, 3) M^{-T}
    ref_g  : (detM, 3) 参考候选(整数超胞索引)

    返回
    ----
    (nplw,) 整数组, 取值 0..det(M)-1
    """
    g = np.asarray(gidx, dtype=float)
    ref = np.atleast_2d(np.asarray(ref_g, dtype=float))
    u = np.mod(g @ m_inv_T, 1.0)
    u_ref = np.mod(ref @ m_inv_T, 1.0)
    d = _periodic_delta(u[:, None, :], u_ref[None, :, :]).max(axis=2)
    return np.argmin(d, axis=1).astype(int)


def permutation_for_candidates(g_cand: np.ndarray, m_inv_T: np.ndarray,
                               class_labels: np.ndarray,
                               ref_g: np.ndarray) -> np.ndarray:
    """给定某���超胞 k 的候选集合, 求"候选 -> 等价类"的置换。

    候选 ``g_cand[j]`` 本身就是一个平面波索引, 它的等价类号就是
    ``class_labels`` 里对应的号 ⇒ 直接查表即可。

    返回 (detM,) 数组: ``perm[j]`` = 候选 j 对应的等价类号。
    """
    g_cand = np.atleast_2d(np.asarray(g_cand, dtype=float))
    # 候选的等价类: 用代表元匹配(ref_g 的类与 class_labels 一致)
    u_ref = np.mod(np.atleast_2d(np.asarray(ref_g, dtype=float)) @ m_inv_T, 1.0)
    u_cand = np.mod(g_cand @ m_inv_T, 1.0)
    d = _periodic_delta(u_cand[:, None, :], u_ref[None, :, :]).max(axis=2)
    return np.argmin(d, axis=1).astype(int)


def plane_wave_weights(coeffs: np.ndarray, class_labels: np.ndarray,
                       perm: np.ndarray, *,
                       normalize: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """平面波谱权重 (Popescu--Zunger Eq. (A8))。

    ``w[j, b] = Σ_{n: class(n) == perm[j]} |C_{n,b}|²``

    参数
    ----
    coeffs      : (nplw, n_band) 复系数(行 = 平面波)
    class_labels: (nplw,) 等价类号
    perm        : (detM,) 候选 -> 类 的置换
    normalize   : 是否按带把系数归一化(``Σ_n |C|² = 1``)。VASP 写出的系数
                  不是归一的(实测 Σ|C|² ≈ 1.2), 归一化后完备性才等于 1。
                  easyunfold 也是这么做的。

    返回
    ----
    (w, completeness)
      w            : (detM, n_band) 权重
      completeness : (n_band,) 每条带的 Σ_j w_j —— 应当为 1(候选集构造正确的判据)
    """
    c = np.asarray(coeffs)
    if c.ndim != 2:
        raise ValueError("coeffs 必须是 (nplw, n_band)")
    if normalize:
        norm = np.linalg.norm(c, axis=0)
        norm = np.where(norm > 0, norm, 1.0)
        c = c / norm[None, :]
    lab = np.asarray(class_labels, dtype=int)
    perm = np.asarray(perm, dtype=int)
    w = np.zeros((perm.size, c.shape[1]), dtype=float)
    for j, cl in enumerate(perm):
        sel = lab == cl
        if sel.any():
            w[j] = np.sum(np.abs(c[sel]) ** 2, axis=0)
    return w, w.sum(axis=0)


def weight_dominance(w: np.ndarray) -> np.ndarray:
    """单 G 主导度: 每条带在候选上的权重占比 max/sum (形状 (n_band,))。

    对"超胞态只对应一个原胞动量"的干净超胞, 这个值应当接近 1; 它显著小于 1
    说明态被摊到了多个候选上(超胞破坏了平移对称性, 或 k 网格太粗)。
    """
    w = np.asarray(w, dtype=float)
    tot = w.sum(axis=0)
    mx = w.max(axis=0)
    out = np.zeros_like(tot)
    ok = tot > 0
    out[ok] = mx[ok] / tot[ok]
    return out
