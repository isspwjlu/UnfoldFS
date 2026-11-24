"""费米面提取与诊断 —— 从谱函数到费米面的"最后一步"。

这个模块把原先只存在于论文出图脚本(``scripts/figures/``)里的三段逻辑收进包里:

1. **脊顶提取**(:func:`crest_field` / :func:`crest_curves`)
   A(k,E) 在 E_F 处是一条**高斯脊**; 真费米面是**脊顶**(恰过 E_F 的那条带),
   而 A(k,E_F) 的等值线给出的是脊的**两壁**。追踪主峰必须过三道闸门
   (:func:`crest_field` 的三个门限), 否则会凭空画出 A(k,E_F)=0 的"幽灵小圈"。
2. **协变性诊断**(:func:`dispersion`)
   A(k,E_F) 在点群各操作下的相对离散度 δ(R)。展开谱权重**天然不协变**
   (副本平均只用单个代表原子), 这个量把它变成可计算的数。
3. **验证协议**(:func:`verify_fermi_surface`)
   反折叠只重排权重、不该移动费米面 ⇒ 反折叠 FS 必须与原胞直算 FS 重合。

外加费米矢量与费米速度(:func:`fermi_vectors` / :func:`fermi_velocity`)——
它们给"等值面 vs 脊顶"这个几何差别一个**物理后果**: 取错了面, |v_F| 就系统性地错。

⚠ **等值面 ≠ 费米面**: 本模块默认输出的是**脊顶**。若要复现社区惯用的
A(k,E_F) 等值面(两壁), 用 :func:`spectral_curves`, 并注意那不是费米面。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import SymmetryOps, rotation_inverse

__all__ = [
    "SpectralSlice",
    "load_spectral_npz",
    "crest_field",
    "crest_curves",
    "spectral_curves",
    "dispersion",
    "point_to_segment_distance",
    "vertex_distance",
    "verify_fermi_surface",
    "fermi_vectors",
    "fermi_velocity",
]

# 三道闸门的默认门限(与论文一致, 见 docs/algorithms.md §15)
DEFAULT_WIN_EV = 0.20          # 主峰搜索窗(半宽, eV)
DEFAULT_MAX_SHIFT_EV = 0.25    # 闸门②: |E*(k) − E_F| 的上限
DEFAULT_MIN_WEIGHT_FRAC = 0.02  # 闸门③: A(k,E_F) / max A 的下限


# --------------------------------------------------------------------------
# 数据载入
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralSlice:
    """``A(k,E).npz`` 的一个 kz 切片(展开成二维网格)。

    n1, n2      网格点数(第一、二维)
    fx, fy      分数坐标轴(升序)
    a           谱函数 A(k,E), 形状 (n1, n2, nE)
    band_e      逐带能量 (n_band, n1, n2); 补位槽是 +inf
    band_w      逐带谱权重 (n_band, n1, n2); 补位槽是 0
    valid       有效位掩码 (n_band, n1, n2)
    e_grid      能量网格 (nE,)
    e_f         费米能级 (eV)
    sigma       展宽 (eV)
    k_frac      本切片的分数坐标 (n1*n2, 3)
    """

    n1: int
    n2: int
    fx: np.ndarray
    fy: np.ndarray
    a: np.ndarray
    band_e: np.ndarray
    band_w: np.ndarray
    valid: np.ndarray
    e_grid: np.ndarray
    e_f: float
    sigma: float
    k_frac: np.ndarray
    k_cart: np.ndarray
    orbital_weights: Optional[np.ndarray] = None
    """(n_band, n1, n2, norb) 逐轨道谱权重; npz 里没有则为 None。"""
    orbital_labels: tuple = ()

    @property
    def a_ef(self) -> np.ndarray:
        """A(k, E_F), 形状 (n1, n2)。"""
        i_ef = int(np.argmin(np.abs(self.e_grid - self.e_f)))
        return self.a[:, :, i_ef]


def load_spectral_npz(path: str, kz: Optional[float] = None) -> SpectralSlice:
    """读 ``A(k,E).npz`` 并取指定 kz 层的 kx-ky 切片。

    网格点序为 ``i1·n2·n3 + i2·n3 + i3``(见 :mod:`unfoldfs.expand`)。

    ⚠ 只有 kz = 0 一层时(单层算例就是 2D k 网格), 任意 kz 都只能返回那一层 ——
       单层体系带在 kz 上是平的(层间耦合 ≈ 0.003), 各 kz 切片本来就该完全相同。
    """
    d = np.load(path)
    kf = np.asarray(d["k_frac"])
    n_orb = int(np.asarray(d["band_orb"]).shape[2]) if "band_orb" in d.files else 0
    fx = np.unique(np.round(kf[:, 0], 6))
    fy = np.unique(np.round(kf[:, 1], 6))
    fz = np.unique(np.round(kf[:, 2], 6))
    n1, n2, n3 = len(fx), len(fy), len(fz)
    k3 = n3 // 2 if kz is None else int(np.argmin(np.abs(fz - float(kz))))
    sel = np.arange(n1 * n2) * n3 + k3
    return SpectralSlice(
        n1=n1,
        n2=n2,
        fx=fx,
        fy=fy,
        a=np.asarray(d["A"])[sel].reshape(n1, n2, -1),
        band_e=np.asarray(d["band_e"])[sel].T.reshape(-1, n1, n2),
        band_w=np.asarray(d["band_w"])[sel].T.reshape(-1, n1, n2),
        valid=np.asarray(d["valid"])[sel].T.reshape(-1, n1, n2),
        e_grid=np.asarray(d["e_grid"]),
        e_f=float(d["ef"]),
        sigma=float(d["sigma"]),
        k_frac=kf[sel],
        k_cart=np.asarray(d["k_cart"])[sel],
        # npz 里 band_orb 的约定: (n_points, n_band, norb)(与 band_e 同布局)
        orbital_weights=(np.asarray(d["band_orb"])[sel]
                         .reshape(n1, n2, -1, n_orb).transpose(2, 0, 1, 3)
                         if "band_orb" in d.files else None),
        orbital_labels=(tuple(str(x) for x in d["orb_labels"])
                        if "orb_labels" in d.files else ()),
    )


# --------------------------------------------------------------------------
# 1. 脊顶提取
# --------------------------------------------------------------------------
def crest_field(a: np.ndarray, e_grid: np.ndarray, e_f: float, sigma: float,
                *, win: float = DEFAULT_WIN_EV,
                max_shift: float = DEFAULT_MAX_SHIFT_EV,
                min_weight_frac: float = DEFAULT_MIN_WEIGHT_FRAC,
                b2d: Optional[np.ndarray] = None,
                blur_cells: float = 0.0) -> np.ndarray:
    """每个 k 上 A(k,·) 在 E_F 附近**主峰**的能量 E*(k), 并过三道闸门。

    参数
    ----
    a : (n1, n2, nE) 谱函数
    win : 主峰搜索窗的半宽(eV)。窗外的带不参与竞争。
    max_shift : 闸门② —— |E*(k) − E_F| 的上限(eV)。超出者不是过 E_F 的带。
    min_weight_frac : 闸门③ —— A(k,E_F) / max A 的下限。没有谱权重就没有费米面。
    b2d : (2,2) 倒格矢; 给了 **blur_cells** 时用来做笛卡尔等距模糊。
    blur_cells : 模糊的核宽(单位网格格数)。**取等值线前必须 >0**, 否则主峰
        换带处的跳变会拉出横穿布里渊区的**直线**(假象)。

    返回
    ----
    (n1, n2) 的 E*(k); 被闸门挡掉的地方给 NaN(等值线自然断开)。

    三道闸门各自防一种失效:

    ① 峰必须是**真的局域极大**(A[j-1] < A[j] > A[j+1])。否则会把"搜索窗
       边缘上强带的尾巴"当成峰, 那种"峰"离 E_F 有 0.3 eV, E* 在那里乱跳。
    ② |E*(k) − E_F| < max_shift: 排除与 E_F 无关的强带。
    ③ A(k,E_F) > min_weight_frac · max A: 排除 E_F 处根本没有谱权重的区域
       (否则会画出 A(k,E_F) 恒为 0 的孤立小圈)。
    """
    ew = np.asarray(e_grid, dtype=float)
    a = np.asarray(a, dtype=float)
    sel = np.abs(ew - e_f) <= win
    if not sel.any():
        raise ValueError("搜索窗内没有能量点: 检查 e_grid 与 e_f")
    aw = a[..., sel]
    ews = ew[sel]
    j = np.argmax(aw, axis=-1)
    e0 = ews[j]

    # 闸门①: 在**全能量网格**上检验是不是真局域极大
    sel_idx = np.nonzero(sel)[0]
    jfull = sel_idx[j][..., None]
    left = np.take_along_axis(a, np.clip(jfull - 1, 0, None), axis=-1)[..., 0]
    right = np.take_along_axis(a, np.clip(jfull + 1, 0, None), axis=-1)[..., 0]
    peak = np.take_along_axis(a, jfull, axis=-1)[..., 0]
    is_peak = (peak > left) & (peak > right) & (peak > 0.0)

    # 峰的能量的加权平均(在 2σ 邻域内) ⇒ E* 是光滑场, 不随网格抖
    dist = np.abs(ew[None, None, :] - e0[..., None])
    near = (dist <= 2.0 * sigma) * a
    num = (near * ew[None, None, :]).sum(axis=-1)
    den = near.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_star = np.where(den > 0, num / den, np.nan)

    # 模糊(强烈建议): 抹平"主峰在副本之间跳变"造成的折角, 同时保持点群协变
    # (笛卡尔核 ⇒ 严格对称)。**顺序与论文一致: 先模糊, 再过闸门**。
    if blur_cells and b2d is not None:
        e_star = blur_cartesian(e_star, np.asarray(b2d, dtype=float),
                               float(blur_cells))

    # 闸门②③
    keep = is_peak & (np.abs(e_star - e_f) <= max_shift)
    i_ef = int(np.argmin(np.abs(ew - e_f)))
    a_ef = a[:, :, i_ef]
    finite_max = np.nanmax(a_ef[np.isfinite(a_ef)]) if np.isfinite(a_ef).any() else 0.0
    keep &= a_ef > min_weight_frac * finite_max
    return np.where(keep, e_star, np.nan)


def crest_curves(field: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                 b2d: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                 *, level: float = 0.0) -> List[np.ndarray]:
    """标量场的等值线 -> 折回第一 BZ 的曲线列表(每条是 (n,2) 的笛卡尔坐标)。

    ``field`` 是 E*(k) − E_F(零等值线 = 脊顶); 传别的场可取别的等值线。
    NaN 处被遮蔽 ⇒ 等值线自然断开(不会跨过没有数据的区域)。

    ⚠ 需要 matplotlib(``pip install 'unfoldfs[plot]'``)。
    """
    return _contours(field, fx, fy, b2d, b1, b2, levels=[level])


def spectral_curves(a_ef: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                    b2d: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                    *, level: Optional[float] = None) -> List[np.ndarray]:
    """A(k,E_F) 的等值线 —— **两壁**, 不是费米面。

    ``level`` 默认取场的最大值的一半(社区惯用的做法)。⚠ 这条曲线与
    :func:`crest_curves` 给出的真费米面**不是同一个曲面**, 差别见论文 §III.A。
    """
    field = np.asarray(a_ef, dtype=float)
    if level is None:
        finite = field[np.isfinite(field)]
        level = 0.5 * float(np.nanmax(finite)) if finite.size else 0.0
    return _contours(field, fx, fy, b2d, b1, b2, levels=[float(level)])


def _contours(field: np.ndarray, fx: np.ndarray, fy: np.ndarray,
              b2d: np.ndarray, b1: np.ndarray, b2: np.ndarray,
              *, levels: Sequence[float]) -> List[np.ndarray]:
    """等值线提取(3×3 平铺 → 取等值线 → 折回第一 BZ)。

    平铺是必须的: 否则边界处的等值线会被截断。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "取等值线需要 matplotlib: pip install 'unfoldfs[plot]'") from exc

    tile = np.ma.masked_invalid(np.tile(np.asarray(field, dtype=float), (3, 3)))
    tx = np.concatenate([np.asarray(fx) - 1.0, np.asarray(fx), np.asarray(fx) + 1.0])
    ty = np.concatenate([np.asarray(fy) - 1.0, np.asarray(fy), np.asarray(fy) + 1.0])
    gx, gy = np.meshgrid(tx, ty, indexing="ij")
    fig, ax = plt.subplots()
    cs = ax.contour(gx, gy, tile, levels=list(levels))
    plt.close(fig)

    out: List[np.ndarray] = []
    for path in cs.get_paths():
        verts, codes = path.vertices, path.codes
        if codes is None:
            if len(verts) > 1:
                out.extend(_split_jumps(
                    fold_to_bz(np.asarray(verts) @ b2d, b1, b2)))
            continue
        starts = list(np.nonzero(codes == 1)[0]) + [len(verts)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            if s1 - s0 > 1:
                out.extend(_split_jumps(
                    fold_to_bz(np.asarray(verts[s0:s1]) @ b2d, b1, b2)))
    return out


def _split_jumps(curve: np.ndarray, factor: float = 3.0) -> List[np.ndarray]:
    """折回 BZ 后, 把"跨越边界"的那些段切开(否则会画出横穿 BZ 的直线)。

    折回第一 BZ 时, contour 路径在网格边界处的一跳(k=0.99 → 0.01)会被折成
    一条横跨整个 BZ 的直线 —— 那是**假象**, 不是费米面。判据: 相邻顶点
    距离远大于该曲线的中位距离 ⇒ 该处跨越了边界。
    """
    if len(curve) < 2:
        return []
    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    thr = factor * float(np.median(d)) if len(d) else np.inf
    pieces: List[np.ndarray] = []
    cur: List[np.ndarray] = [curve[0]]
    for k, dist in enumerate(d):
        if dist > thr:
            if len(cur) > 1:
                pieces.append(np.asarray(cur))
            cur = [curve[k + 1]]
        else:
            cur.append(curve[k + 1])
    if len(cur) > 1:
        pieces.append(np.asarray(cur))
    return pieces


def fold_to_bz(cart: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """把每个顶点折回第一 BZ(倒格矢的 Voronoi 胞, 对六边形晶格即六边形)。"""
    cart = np.asarray(cart, dtype=float)
    best = np.zeros_like(cart)
    bestd = np.full(cart.shape[0], np.inf)
    for i in range(-2, 3):
        for j in range(-2, 3):
            shift = i * np.asarray(b1, dtype=float) + j * np.asarray(b2, dtype=float)
            dd = np.linalg.norm(cart - shift[None, :], axis=1)
            m = dd < bestd
            bestd = np.where(m, dd, bestd)
            best[m] = shift
    return cart - best


def split_at_jumps(curve: np.ndarray, factor: float = 3.0) -> List[np.ndarray]:
    """折回后相邻顶点距离异常大 = 该段跨越了 BZ 边界 ⇒ 切开。"""
    if len(curve) < 2:
        return []
    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    thr = factor * float(np.median(d)) if len(d) else np.inf
    pieces: List[np.ndarray] = []
    cur: List[np.ndarray] = [curve[0]]
    for k, dist in enumerate(d):
        if dist > thr:
            if len(cur) > 1:
                pieces.append(np.asarray(cur))
            cur = [curve[k + 1]]
        else:
            cur.append(curve[k + 1])
    if len(cur) > 1:
        pieces.append(np.asarray(cur))
    return pieces


# --------------------------------------------------------------------------
# 2. 协变性诊断
# --------------------------------------------------------------------------
def closed_loops(pieces: Sequence[np.ndarray], tol: float = 1e-6) -> List[np.ndarray]:
    """把折回 BZ 后被切碎的曲线段**拼回闭合回路**(算包围面积时需要)。

    :func:`crest_curves` 会把跨越 BZ 边界的段切开(否则会画出横穿 BZ 的直线),
    于是每条闭合的费米面变成一串首尾相接的碎段。面积只能用**闭合回路**算
    (shoelace 对开口弧没有意义)。

    做法: 用 KD-tree 找"起点离当前终点最近"的未用段, 一直拼到回到起点。
    """
    pieces = [np.asarray(p, dtype=float) for p in pieces if len(p) >= 2]
    if not pieces:
        return []
    from scipy.spatial import cKDTree
    starts = np.array([p[0] for p in pieces])
    tree = cKDTree(starts)
    used: set = set()
    loops: List[np.ndarray] = []
    for i in range(len(pieces)):
        if i in used:
            continue
        chain = [pieces[i]]
        used.add(i)
        start = pieces[i][0]
        cur = pieces[i][-1]
        while np.linalg.norm(cur - start) > tol:
            d, j = tree.query(cur)
            if j in used or d > tol:
                break
            chain.append(pieces[j])
            used.add(j)
            cur = pieces[j][-1]
        loops.append(np.vstack(chain))
    return loops


def loop_area(loops: Sequence[np.ndarray]) -> float:
    """闭合回路围出的总面积(shoelace, 逐回路取绝对值后求和)。"""
    total = 0.0
    for c in loops:
        if len(c) < 3:
            continue
        x, y = c[:, 0], c[:, 1]
        total += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                              - np.dot(y, np.roll(x, -1))))
    return total


def spectral_at_ef_from_bands(band_e: np.ndarray, band_w: np.ndarray,
                              valid: np.ndarray, e_grid: np.ndarray,
                              e_f: float, sigma: float) -> np.ndarray:
    """从逐带数据重建 A(k,E_F)(**不做**点群轨道平均)。

    npz 里存的 ``A`` 已经是对称化后的结果, 用它算协变性恒得 0 —— 想看
    展开本身的相位缺陷, 必须从逐带数据重建。

    参数
    ----
    band_e, band_w, valid : 形状 (n_band, n1, n2) 的逐带能量/权重/有效位掩码
    """
    e = np.asarray(e_grid, dtype=float)
    i_ef = int(np.argmin(np.abs(e - e_f)))
    be = np.where(valid, band_e, 0.0)
    bw = np.where(valid, band_w, 0.0)
    out = np.zeros((be.shape[1], be.shape[2]))
    for b in range(be.shape[0]):
        d2 = (e[i_ef] - be[b]) ** 2
        out += bw[b] * np.exp(-d2 / (2.0 * sigma * sigma))
    return out


def orbital_fraction_at_peak(band_orb: Optional[np.ndarray],
                              band_e: np.ndarray, valid: np.ndarray,
                              e_star: np.ndarray) -> Optional[np.ndarray]:
    """每个 k 上"主峰那条带"的**轨道成分**(归一化)。

    参数
    ----
    band_orb : (n_band, n1, n2, norb) 逐轨道权重, 或 None
    band_e   : (n_band, n1, n2) 逐带能量
    valid    : (n_band, n1, n2) 有效位掩码
    e_star   : (n1, n2) 脊顶场

    返回 (n1, n2, norb) 的成分(每个 k 上权重和为 1); band_orb 为 None 时返回 None。

    ⚠ 逐轨道权重之和**不等于**总权重(交叉项)⇒ 成分按逐轨道权重自身归一化。
    """
    if band_orb is None:
        return None
    bo = np.asarray(band_orb, dtype=float)
    be = np.asarray(band_e, dtype=float)
    va = np.asarray(valid, dtype=bool)
    es = np.asarray(e_star, dtype=float)
    n_band, n1, n2, n_orb = bo.shape
    out = np.full((n1, n2, n_orb), np.nan)
    for i in range(n1):
        for j in range(n2):
            if not np.isfinite(es[i, j]):
                continue
            ok = va[:, i, j] & np.isfinite(be[:, i, j])
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            b = idx[int(np.argmin(np.abs(be[idx, i, j] - es[i, j])))]
            w = bo[b, i, j]
            tot = float(w.sum())
            if tot > 0:
                out[i, j] = w / tot
    return out


def symmetrize_field(field: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                     ops: SymmetryOps, tol: float = 1e-6) -> np.ndarray:
    """按点群轨道平均一个二维场(NaN-safe)。

    ⚠ **为什么脊顶场 E*(k) 也要轨道平均**: 不只是权重。嵌套体系(如体相
       1T-TaS2)在嵌套向量处两条带在 E_F 附近交换, 而主峰取"最高"的那个峰
       ⇒ 该 k 处 E* 偏离 E_F 达 ~0.09 eV ⇒ ``E*-E_F = 0`` 的等值线在那里
       **不存在** ⇒ 费米面的臂在"腰"处整段消失(实测体相 54 个断点)。
       轨道平均后 E* 在对称相关点上一致 ⇒ 曲线连续。

    ⚠ **NaN-safe**: 只在**有限**的像之间平均; 所有像都是 NaN 时保持 NaN。
       写成 0 会把"这里没有峰"当成"峰在 E=0", 随后被闸门整片删掉。

    只有把网格**映到自身**的操作才参与(否则轨道不是群轨道, 平均后也不再协变)。
    """
    field = np.asarray(field, dtype=float)
    n1, n2 = field.shape
    gx, gy = np.meshgrid(np.asarray(fx), np.asarray(fy), indexing="ij")
    frac = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n1 * n2)])
    vals = field.ravel()
    n_ops = int(np.asarray(ops.rotations).shape[0])
    idx = np.zeros((n1 * n2, n_ops), dtype=int)
    ok = np.zeros((n1 * n2, n_ops), dtype=bool)
    for o in range(n_ops):
        img = ops.apply(frac, o)
        ii = np.rint(img[:, 0] * n1).astype(int) % n1
        jj = np.rint(img[:, 1] * n2).astype(int) % n2
        dx = np.abs(img[:, 0] - ii / n1)
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.abs(img[:, 1] - jj / n2)
        dy = np.minimum(dy, 1.0 - dy)
        ok[:, o] = np.hypot(dx, dy) < tol
        idx[:, o] = ii * n2 + jj
    out = np.full(n1 * n2, np.nan)
    for i in range(n1 * n2):
        # ⚠ **恒含自身**: 若没有操作把网格映到自身(例如网格与点群不兼容),
        #   轨道会是空的 ⇒ 全场变 NaN。自身就是恒等操作的作用。
        members = np.unique(np.concatenate(([i], idx[i, ok[i]])))
        v = vals[members]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = float(v.mean())
    return out.reshape(n1, n2)


def dispersion(field: np.ndarray, ops: SymmetryOps,
               b_prim: np.ndarray) -> Tuple[float, float]:
    """A(k,E_F) 在点群各操作下的相对离散度 δ(R) —— (最大值, 平均值)。

    δ(R) = max_k |A(k,E_F) − A(Rk,E_F)| / max_k A(k,E_F)

    对每个操作 R 取全部 k 上的最大偏差, 再除以场的最大值。
    δ=0 ⇒ 场在该点群下协变; δ=1 ⇒ 有一个谱峰被映到零权重。

    ⚠ 约定与论文一致: 操作作用到**分数坐标**是 k @ R^-1 + t(倒空间的伴随是
       R^-1 而不是 R.T —— 分数基下的算符不是正交阵)。
    """
    field = np.asarray(field, dtype=float)
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)], 1)
    vals = field.ravel()
    devs: List[float] = []
    for r in np.asarray(ops.rotations):
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        devs.append(float(np.abs(vals - vals[i2 * n2 + j2]).max()))
    mx = float(np.nanmax(field))
    if not mx > 0:
        return 0.0, 0.0
    return float(np.max(devs) / mx), float(np.mean(devs) / mx)


# --------------------------------------------------------------------------
# 3. 验证协议
# --------------------------------------------------------------------------
def point_to_segment_distance(points: np.ndarray, curve: np.ndarray,
                               chunk: int = 4096) -> np.ndarray:
    """每个点到折线的**点到线段**距离(不是顶点配对距离)。

    点到顶点会高估距离(顶点是采样点, 曲线在顶点之间才是连续的)⇒
    做几何核对时必须用点到线段。

    分块计算, 避免 O(N·M) 的显存峰值。
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    c = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(c) < 2:
        raise ValueError("折线至少要两个顶点")
    a = c[:-1]
    ab = c[1:] - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0, denom, 1.0)
    out = np.empty(len(p), dtype=float)
    for start in range(0, len(p), chunk):
        sl = slice(start, min(start + chunk, len(p)))
        q = p[sl][:, None, :] - a[None, :, :]          # (chunk, n_seg, 2)
        t = np.einsum("qsi,si->qs", q, ab) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d = np.linalg.norm(p[sl][:, None, :] - proj, axis=2)
        out[sl] = d.min(axis=1)
    return out


def vertex_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> np.ndarray:
    """双向的**顶点配对**距离(最近顶点)。

    这是论文图 5(c) 用的度量: 它量的是两套**采样**的对应, 不是几何本身。
    几何核对请用 :func:`point_to_segment_distance`。
    """
    from scipy.spatial import cKDTree

    va = np.asarray(curve_a, dtype=float).reshape(-1, 2)
    vb = np.asarray(curve_b, dtype=float).reshape(-1, 2)
    d_ab = cKDTree(vb).query(va)[0]
    d_ba = cKDTree(va).query(vb)[0]
    return np.concatenate([d_ab, d_ba])


def verify_fermi_surface(curve_unfolded: np.ndarray, curve_direct: np.ndarray,
                          mesh_spacing: float) -> dict:
    """反折叠费米面 vs 原胞直算费米面的验证协议。

    参数
    ----
    mesh_spacing : 一个网格间距的**笛卡尔长度**(2π/Å)。曲线顶点的坐标也是
        2π/Å, 所以距离必须**除以**它才是"格"—— 直接拿 2π/Å 的绝对值当阈值
        会把明显的错位误报成 100 % 重合。

    返回字典:

    * ``coincidence`` —— 两曲线**重合的长度占比**(0~1)。判据: 到对方曲线的
      距离 ≤ 1 个网格间距;
    * ``median`` / ``max`` —— 距离统计, 单位是**网格间距**("格");
    * ``n_points`` —— 参与统计的点数(双向)。
    """
    if not mesh_spacing > 0:
        raise ValueError("mesh_spacing 必须为正")
    d_ab = point_to_segment_distance(curve_unfolded, curve_direct) / mesh_spacing
    d_ba = point_to_segment_distance(curve_direct, curve_unfolded) / mesh_spacing
    d = np.concatenate([d_ab, d_ba])
    return {
        "coincidence": float(np.mean(d <= 1.0)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
        "n_points": int(d.size),
    }


# --------------------------------------------------------------------------
# 4. 费米矢量与费米速度
# --------------------------------------------------------------------------
def _grid_spacing(fx: np.ndarray, fy: np.ndarray) -> tuple[float, float]:
    """均匀网格的步长(分数坐标)。"""
    fx = np.asarray(fx, dtype=float)
    fy = np.asarray(fy, dtype=float)
    dfx = float(fx[1] - fx[0]) if len(fx) > 1 else 1.0
    dfy = float(fy[1] - fy[0]) if len(fy) > 1 else 1.0
    return dfx, dfy


def _bilinear(field: np.ndarray, fx: np.ndarray, fy: np.ndarray,
              x: float, y: float) -> float:
    """周期双线性插值(分数坐标)。

    ⚠ 参与的四个格点里**只要有一个是 NaN**, 结果就是 NaN —— 这样被闸门
       挡掉的区域不会在插值后"漏回来"。
    """
    field = np.asarray(field, dtype=float)
    n1, n2 = field.shape
    dfx, dfy = _grid_spacing(fx, fy)
    u = (float(x) % 1.0) / dfx
    v = (float(y) % 1.0) / dfy
    i0 = int(np.floor(u)) % n1
    j0 = int(np.floor(v)) % n2
    i1 = (i0 + 1) % n1
    j1 = (j0 + 1) % n2
    fu = u - np.floor(u)
    fv = v - np.floor(v)
    c00 = field[i0, j0]
    c10 = field[i1, j0]
    c01 = field[i0, j1]
    c11 = field[i1, j1]
    val = (c00 * (1 - fu) * (1 - fv) + c10 * fu * (1 - fv)
           + c01 * (1 - fu) * fv + c11 * fu * fv)
    if not all(np.isfinite(v) for v in (c00, c10, c01, c11)):
        return float("nan")
    return float(val)


def lattice_kernel(b2d: np.ndarray, sigma: float,
                   rmax: float = 3.0):
    """落在笛卡尔半径 rmax*sigma 内的格点偏移 + 高斯权重(归一化)。

    支撑集用**笛卡尔距离**定义 ⇒ 对倒格矢的每一次等距变换(旋转/镜像)都封闭
    ⇒ 核在点群下**严格协变**。若用"分数坐标里的方形窗口", 剪切型的 C3 会把
    方窗变成平行四边形 ⇒ 协变性被破坏。
    """
    r_max = int(np.ceil(rmax * sigma / np.linalg.norm(b2d[0]))) + 1
    di, dj, w = [], [], []
    for a in range(-r_max, r_max + 1):
        for b in range(-r_max, r_max + 1):
            d = np.array([a, b], dtype=float) @ b2d
            r = float(np.linalg.norm(d))
            if r <= rmax * sigma:
                di.append(a)
                dj.append(b)
                w.append(np.exp(-r * r / (2.0 * sigma * sigma)))
    w = np.asarray(w)
    return np.array(di, dtype=int), np.array(dj, dtype=int), w / w.sum()


def blur_cartesian(field: np.ndarray, b2d: np.ndarray, sigma_cells: float,
                   rmax: float = 3.0) -> np.ndarray:
    """笛卡尔等距高斯模糊(周期边界, NaN 安全)⇒ 保持点群协变。

    ⚠ **必须做这一步**: 脊顶场 E*(k) 在"主峰换带"处会**跳变**(不同的 k 上
       主峰属于不同的副本), 不平滑就直接 contour 零等值线, 会在跳变处拉出
       **横穿整个布里渊区的直线** —— 那是假象, 不是费米面。

    核的支撑集按**笛卡尔距离**选(不是分数坐标里的方窗)⇒ 严格协变。
    """
    h = float(np.linalg.norm(b2d[0]))
    di, dj, w = lattice_kernel(b2d, sigma_cells * h, rmax)
    num = np.zeros_like(field)
    den = np.zeros_like(field)
    for a, b, wt in zip(di, dj, w):
        sh = np.roll(np.roll(field, -a, axis=0), -b, axis=1)
        ok = np.isfinite(sh)
        num += wt * np.where(ok, sh, 0.0)
        den += wt * ok
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def crest_gradient(crest: np.ndarray, fx: np.ndarray,
                   fy: np.ndarray) -> np.ndarray:
    """脊顶场在网格上的梯度(中心差分, 周期边界) → (n1, n2, 2) 分数坐标。

    ⚠ 必须在**网格点**上做差分再插值: 直接在任意点用最近邻做有限差分,
       梯度会恒为 0 或出现数值爆炸(最近邻在网格尺度内是常数)。
    """
    crest = np.asarray(crest, dtype=float)
    dfx, dfy = _grid_spacing(fx, fy)
    d1 = (np.roll(crest, -1, axis=0) - np.roll(crest, 1, axis=0)) / (2.0 * dfx)
    d2 = (np.roll(crest, -1, axis=1) - np.roll(crest, 1, axis=1)) / (2.0 * dfy)
    return np.stack([d1, d2], axis=-1)


def fermi_vectors(crest: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                  b2d: np.ndarray, n_rays: int = 180,
                  e_f: float = 0.0) -> np.ndarray:
    """从脊顶场 E*(k) 沿 Γ 出发的射线求费米矢量(过 E_F 处的 k 矢量)。

    做法: 对每个方向 n̂, 在射线上找 E*(r·n̂) − E_F 的过零点(线性插值),
    再把它折回第一 BZ。返回 (n_found, 3) 的**笛卡尔**坐标(2π/Å)。

    ⚠ 用脊顶场而不是逐带能量: 展开谱里"带号"在跨 k 时会跳变(同一带号在不同
       k 是不同副本), 逐带求过零点会碎成竖条; 脊顶场 E*(k) 是连续的。

    ⚠ 只覆盖以 Γ 为起点的射线 ⇒ 拿不到不包含 Γ 的口袋(那部分要用
       :func:`verify_fermi_surface` 的曲线距离去核对)。
    """
    crest = np.asarray(crest, dtype=float)
    if crest.shape != (len(fx), len(fy)):
        raise ValueError("crest 的形状应为 (len(fx), len(fy))")
    # 射线方向在**分数坐标**里取, 便于与网格对齐
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False)
    dirs = np.column_stack([np.cos(ang), np.sin(ang)])
    b2d = np.asarray(b2d, dtype=float)

    out: List[np.ndarray] = []
    for d in dirs:
        ts = np.linspace(0.0, 1.0, 512)
        vals = np.array([_bilinear(crest, fx, fy, (t * d[0]) % 1.0,
                                   (t * d[1]) % 1.0) for t in ts])
        vals = vals - e_f
        good = np.isfinite(vals)
        if good.sum() < 2:
            continue
        tsg, vsg = ts[good], vals[good]
        sign = np.sign(vsg)
        cross = np.nonzero(np.diff(sign) != 0)[0]
        for c in cross:
            if vsg[c] == vsg[c + 1]:
                continue
            t0 = tsg[c] + (tsg[c + 1] - tsg[c]) * (-vsg[c]) / (vsg[c + 1] - vsg[c])
            k_frac = (t0 * d) % 1.0
            out.append(np.append(np.asarray(k_frac[:2], dtype=float) @ b2d, 0.0))
    return np.asarray(out) if out else np.zeros((0, 3))


def fermi_velocity(crest: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                   b2d: np.ndarray, e_f: float = 0.0,
                   *, n_rays: int = 180) -> dict:
    """费米速度: 沿 Γ 射线, v_F = (1/ħ)|∇_k E| 在 E*(k) = E_F 处。

    返回字典:

    * ``vectors`` (n,3) 笛卡尔费米矢量(2π/Å)
    * ``grad_frac`` (n,2) 脊顶场在分数坐标下的梯度 (eV)
    * ``v_f`` (n,) |v_F| 的**分数坐标**模长(未换算; 见 :func:`velocity_to_si`)

    ⚠ **为什么用脊顶场而不是逐带能量**: 展开谱里"带号"在跨 k 时会跳变
       (同一带号在不同 k 是不同副本), 逐带求过零点会碎成竖条; 脊顶场
       E*(k) 是连续函数, 它的梯度才是良定义的费米速度。
    """
    crest = np.asarray(crest, dtype=float)
    b2d = np.asarray(b2d, dtype=float)
    vecs = fermi_vectors(crest, fx, fy, b2d, n_rays=n_rays, e_f=e_f)
    if len(vecs) == 0:
        return {"vectors": vecs, "grad_frac": np.zeros((0, 2)),
                "v_f": np.zeros(0)}
    grad_field = crest_gradient(crest, fx, fy)
    b_inv = np.linalg.inv(b2d)
    grads = []
    for v in vecs:
        k_frac = v[:2] @ b_inv
        grads.append(_sample_gradient(grad_field, fx, fy, k_frac))
    grad_frac = np.asarray(grads, dtype=float)
    return {
        "vectors": vecs,
        "grad_frac": grad_frac,
        "v_f": np.linalg.norm(grad_frac, axis=1),
    }


def _sample_gradient(grad_field: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                     k_frac: np.ndarray) -> np.ndarray:
    """在 (k_frac) 处对**网格梯度场**做双线性插值(两个分量分别插值)。"""
    g1 = _bilinear(grad_field[:, :, 0], fx, fy, k_frac[0], k_frac[1])
    g2 = _bilinear(grad_field[:, :, 1], fx, fy, k_frac[0], k_frac[1])
    return np.array([g1, g2], dtype=float)


def velocity_to_si(grad_frac: np.ndarray, b2d: np.ndarray) -> np.ndarray:
    """把脊顶场在**分数坐标**下的梯度换算成 m/s。

    参数
    ----
    grad_frac : (n, 2) 分数坐标下的梯度 (eV)
    b2d : (2,2) 原胞倒格矢的前两行(单位 1/Å)

    E(k_frac) = E(k_frac · B) ⇒ ∇_k E = ∇_k_frac · B⁻¹(**不是**乘 |B|:
    分数基下的倒格矢不正交, 只有正交基才成立)。

    ⚠ **单位**: ``b2d`` 的数值是 1/Å(见 ``geometry.reciprocal``: B = 2π·(叉积/体积)
    ⇒ 数值就是物理的 1/Å), 故 ∇_k E 的单位是 eV·Å。于是

        v = (1/ħ)·|∇_k E|   [Å/s]  ⇒  ×1e-10 换到 m/s

    (ħ = 6.582119569e-16 eV·s)。**注意 Å→m 是 1e-10 而不是 1e10** ——
    反过来写会得到大 20 个数量级的荒谬速度。
    """
    b2d = np.asarray(b2d, dtype=float)
    grad = np.asarray(grad_frac, dtype=float)
    grad = grad.reshape(1, 2) if grad.ndim == 1 else grad.reshape(-1, 2)
    b_inv = np.linalg.inv(b2d)
    grad_cart = grad @ b_inv                      # (n, 2) 单位 eV·Å
    hbar = 6.582119569e-16                          # eV·s
    m_per_angstrom = 1.0e-10                       # 1 Å = 1e-10 m
    return np.linalg.norm(grad_cart, axis=1) / hbar * m_per_angstrom
