"""费米面(**真费米面**): 画的是 E_b(k) 恰过 E_F 的那条线, 不是 A(k,E_F) 的等值线。

为什么不能用 A(k, E_F) 的等值线:
    A(k, E) = Σ_b w_b(k)·exp(−(E−E_b)²/2σ²)。在 k 平面上 A(k, E_F) 沿费米面是
    一条**高斯脊**(脊高 = w, 垂直于脊的宽度 ∝ σ/|dE/dk|)。因此**任何低于脊高的
    等值线都会给出脊两侧各一条线**(两壁), 而费米面是**脊顶**(E_b = E_F 处)。
    实测 1H-NbSe2: 沿 Γ–M 切线, A 的 0.5max 等值线在 k1_frac = 0.214 / 0.245,
    真费米面在 0.229 —— 正好在两壁中间 ⇒ 画 A 的等值线 = 画了管子的壁。

两种画法:
  * ``band``  逐带的 (E_b − E_F) 零等值线 —— **精确**(线性插值到网格内),
             适合原胞(带序在 k 上连续)。
             ⚠ 反折叠超胞不适用: 超胞带在展开后**带号会跳**(同一带号在不同 k 上
               可能是不同的 Bloch-character 副本)⇒ 零等值线碎成竖条/碎片。
  * ``peak``  A(k,·) 在 E_F 附近**主峰的能量** E*(k) 的零等值线 —— 光滑、对称
             (A 已按原胞点群对称化), 对原胞与反折叠超胞**同一口径**。

数据口径: npz 里 band_e/band_w 的布局是 **(点, 带)** 点优先 ⇒ 要 (带, n1, n2)
必须先转置:  d["band_e"].T.reshape(-1, n1, n2)(直接 reshape 会把点带搅乱)。
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from unfoldfs.geometry import read_poscar, reciprocal

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图内固定文字的中英对照(论文用 --lang en)
LBL = {
    "x": {"zh": "$k_1$  (2π/Å)", "en": r"$k_1$  (2$\pi$/Å)"},
    "y": {"zh": "$k_2$  (2π/Å)", "en": r"$k_2$  (2$\pi$/Å)"},
    "cb": {"zh": "$w/w_{max}$", "en": r"$w/w_{\max}$"},
    "low": {"zh": "Low", "en": "Low"},
    "high": {"zh": "High", "en": "High"},
}


def bz_hexagon(b_prim):
    b1, b2 = b_prim[0], b_prim[1]
    v = np.array([(b1 + b2) / 3, (2 * b1 - b2) / 3, (b1 - 2 * b2) / 3,
                  -(b1 + b2) / 3, -(2 * b1 - b2) / 3, -(b1 - 2 * b2) / 3])
    ang = np.arctan2(v[:, 1], v[:, 0])
    return v[np.argsort(ang)][:, :2]


def load(npz_path, kz=None):
    """读 A(k,E).npz, 取**指定 kz 层**的 kx-ky 切片。

    kz 用**分数坐标**(沿 c 的倒格矢为单位, 1.0 = 2π/c)。网格点序
    i1·n2·n3 + i2·n3 + i3(见 unfoldfs/expand.py::_grid_indices)。

    ⚠ 只有 kz = 0 一层时(现在的单层算例就是 2D k 网格), 任意 kz 都只能返回
       那一层 —— 单层体系带在 kz 上是平的(层间耦合 e^{-c/3} ≈ 0.003),
       所以**各 kz 切片本来就该完全相同**; 要看到真正不同的切片需要
       带 kz 采样的算例(--grid n1 n2 n3, n3 > 1)。
    """
    d = np.load(npz_path)
    kf = d["k_frac"]
    fx = np.unique(np.round(kf[:, 0], 6))
    fy = np.unique(np.round(kf[:, 1], 6))
    fz = np.unique(np.round(kf[:, 2], 6))
    n1, n2, n3 = len(fx), len(fy), len(fz)
    k3 = n3 // 2 if kz is None else int(np.argmin(np.abs(fz - float(kz))))
    sel = np.arange(n1 * n2) * n3 + k3           # 取第 k3 层
    be = d["band_e"][sel].T.reshape(-1, n1, n2)   # (带, n1, n2) ← 必须先转置
    bw = d["band_w"][sel].T.reshape(-1, n1, n2)
    va = d["valid"][sel].T.reshape(-1, n1, n2)
    a = d["A"][sel].reshape(n1, n2, -1)
    return dict(n1=n1, n2=n2, fx=fx, fy=fy, be=be, bw=bw, va=va, a=a,
                e_grid=d["e_grid"], e_f=float(d["ef"]), sigma=float(d["sigma"]),
                fz=fz, kz_used=float(fz[k3]), n3=n3)


def split_paths(cs):
    """matplotlib 的 Path -> 一条条**连续的**顶点序列(marching squares 的连通性)"""
    out = []
    for path in cs.get_paths():
        v, codes = path.vertices, path.codes
        if codes is None:
            if len(v) > 1:
                out.append(np.asarray(v))
            continue
        starts = list(np.nonzero(codes == 1)[0]) + [len(v)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            if s1 - s0 > 1:
                out.append(v[s0:s1])
    return out


def fold_to_bz(cart, b1, b2):
    """把每个顶点折回第一 BZ(倒格矢的 Voronoi 胞 = 六边形 BZ)"""
    best = np.zeros_like(cart)
    bestd = np.full(cart.shape[0], np.inf)
    for i in range(-2, 3):
        for j in range(-2, 3):
            shift = i * b1 + j * b2
            dd = np.linalg.norm(cart - shift[None, :], axis=1)
            m = dd < bestd
            bestd = np.where(m, dd, bestd)
            best[m] = shift
    return cart - best


def zero_contours(field2d, fx, fy, b2d, b1, b2):
    """标量场的零等值线 -> 折回第一 BZ 的曲线列表(笛卡尔)。

    ⚠ 必须用**逐点**的 Voronoi 折叠(`fold_to_bz`), 不能改成"按段整体平移":
      逐点折叠对每个点独立选最近像 ⇒ 与点群**协变** ⇒ 画出来的费米面对称。
      按段平移虽然能让臂连续伸出边界, 但每段独立选平移量 ⇒ **破坏协变性**
      ⇒ 费米面变得不对称(实测 2026-11-23)。对称比连续更重要:
      越界的部分本来就要由周期性折回, 臂在边界处结束是标准画法。
    """
    tile = np.ma.masked_invalid(np.tile(field2d, (3, 3)))
    tx = np.concatenate([fx - 1.0, fx, fx + 1.0])
    ty = np.concatenate([fy - 1.0, fy, fy + 1.0])
    GX, GY = np.meshgrid(tx, ty, indexing="ij")
    fig, ax = plt.subplots()
    cs = ax.contour(GX, GY, tile, levels=[0.0])
    paths = split_paths(cs)
    plt.close(fig)
    out = []
    for p in paths:
        cart = np.asarray(p) @ b2d
        out.append(fold_to_bz(cart, b1, b2))
    return out


def split_at_jumps(curve, factor=3.0):
    """折回后相邻顶点距离异常大 = 该段跨越了 BZ 边界 => 切开(丢掉那一段)"""
    if len(curve) < 2:
        return []
    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    thr = factor * float(np.median(d)) if len(d) else np.inf
    pieces, cur = [], [curve[0]]
    for k, dist in enumerate(d):
        if dist > thr:
            if len(cur) > 1:
                pieces.append(np.array(cur))
            cur = [curve[k + 1]]
        else:
            cur.append(curve[k + 1])
    if len(cur) > 1:
        pieces.append(np.array(cur))
    return pieces


# ---- 对称等变的核: 支撑集按**笛卡尔距离**定义(倒格矢的等距变换保它) ----

def lattice_kernel(b2d, sigma, rmax=3.0):
    """落在笛卡尔半径 rmax*sigma 内的格点偏移 + 高斯权重(归一化)。

    支撑集用**笛卡尔距离**定义 ⇒ 对倒格矢的每一次等距变换(旋转/镜像)都封闭
    ⇒ 核在点群下**严格协变**。若用"分数坐标里的方形窗口", 剪切型的 C3 会把
    方窗变成平行四边形 ⇒ 协变性被破坏(实测权重图案残差 ~10%)。
    """
    R = int(np.ceil(rmax * sigma / np.linalg.norm(b2d[0]))) + 1
    di, dj, w = [], [], []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            d = np.array([a, b], dtype=float) @ b2d
            r = float(np.linalg.norm(d))
            if r <= rmax * sigma:
                di.append(a)
                dj.append(b)
                w.append(np.exp(-r * r / (2.0 * sigma * sigma)))
    w = np.asarray(w)
    return np.array(di, dtype=int), np.array(dj, dtype=int), w / w.sum()


def blur_cartesian(field, b2d, sigma_cells, rmax=3.0):
    """笛卡尔等距高斯模糊(周期边界, NaN 安全)⇒ 保持点群协变。"""
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


def sample_equiv(field, cart_pts, b2d, sigma_cells, rmax=3.0):
    """在笛卡尔坐标上做高斯核插值 ⇒ **严格**协变(对称相关的点拿到相同的值)。

    双线性插值在**分数坐标**的平行四边形格子里做, 而点群里的 C3 是剪切型变换
    ⇒ 平行四边形被映射成另一个平行四边形, 插值权重不对应 ⇒ 插值结果不协变
    (实测颜色残差可达场值跨度的 10%)。

    ⚠ 三个必须同时避开的坑:
      (a) 核的支撑集按**笛卡尔距离**选(按分数坐标选方窗 ⇒ 剪切会把方窗变成
          平行四边形 ⇒ 核不再封闭);
      (b) "最近格点"按**周期笛卡尔距离**找(分数四舍五入 ≠ 菱形 Voronoi);
      (c) 行列约定: cart→frac 是 `p @ inv(b2d)`(行向量); 查询点先折回胞内,
          且**距离要用折回后的笛卡尔位置**(点群的像点可能落在胞外好几个胞之外)。
    """
    h = float(np.linalg.norm(b2d[0]))
    di, dj, wk = lattice_kernel(b2d, sigma_cells * h, rmax)
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    gi = (idx // n2) / n1
    gj = (idx % n2) / n2
    # 格点的笛卡尔坐标 = frac @ b2d  (行向量约定; b2d 的第 0/1 行是 b1/b2)
    grid_cart = np.stack([gi * b2d[0, 0] + gj * b2d[1, 0],
                          gi * b2d[0, 1] + gj * b2d[1, 1]], axis=1)
    vals = field.ravel()
    inv = np.linalg.inv(b2d)
    sigma = sigma_cells * h
    shifts = np.array([(ta * b2d[0] + tb * b2d[1])
                       for ta in (-1, 0, 1) for tb in (-1, 0, 1)])
    p = np.atleast_2d(np.asarray(cart_pts, dtype=float))
    # 折回原胞(分数坐标 mod 1)—— 只差整数个倒格矢, 周期距离不变
    fr = p @ inv
    fr = fr - np.floor(fr)
    p = fr @ b2d
    # 周期笛卡尔最近的格点(3×3 邻域 × 9 个周期平移)
    ci = np.rint(fr[:, 0] * n1).astype(int) % n1
    cj = np.rint(fr[:, 1] * n2).astype(int) % n2
    best = np.full(len(p), np.inf)
    bi = ci.copy()
    bj = cj.copy()
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            ii = (ci + a) % n1
            jj = (cj + b) % n2
            base = grid_cart[ii * n2 + jj]                # (n_pts, 2)
            for sh in shifts:
                dd = ((base + sh - p) ** 2).sum(axis=1)
                m = dd < best
                best = np.where(m, dd, best)
                bi = np.where(m, ii, bi)
                bj = np.where(m, jj, bj)
    # 候选点 = 中心 + 核偏移
    sel = ((bi[:, None] + di[None, :]) % n1) * n2 + ((bj[:, None] + dj[None, :]) % n2)
    cand = grid_cart[sel]                                   # (n_pts, K, 2)
    d2 = np.full(cand.shape[:2], np.inf)
    for sh in shifts:                                       # 周期距离
        d2 = np.minimum(d2, ((cand + sh[None, None, :] - p[:, None, :]) ** 2).sum(-1))
    w = np.exp(-d2 / (2.0 * sigma * sigma)) * wk[None, :]
    # NaN 安全: 只用有效候选点, 并重新归一化(否则填 0 会造出假的零交叉)
    ok = np.isfinite(vals[sel])
    w = np.where(ok, w, 0.0)
    v = np.where(ok, vals[sel], 0.0)
    num = np.sum(w * v, axis=1)
    den = np.sum(w, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def symmetrize(field, ops, n1, n2):
    """按原胞完整点群做轨道平均(与工具对 A(k,E) 同口径)。

    反折叠的逐带权重不协变(同一轨道上对称相关点给出不同权重)⇒ 必须平均,
    否则图案继承超胞的对称性(3×1×1 只剩 4/12 个操作)⇒ 看起来既不 3 重也不 6 重。

    ⚠ 要把**原点**的值加到**像**上(`field.ravel()`); 写成 `field[i2,j2]` 就是
    "把每点加给自己" ⇒ 平均退化成恒等 ⇒ 图案仍然不对称。

    ⚠ **NaN 安全**: 只在**有限**的像之间平均; 若某个点的所有像都是 NaN,
      结果保持 NaN(而不是 0)。这对含 NaN 的场(脊顶能量场在没有局域极大的
      地方是 NaN)是关键: 写成 0 会把"没有峰"当成"峰在 E=0", 随后被闸门
      整片删掉 ⇒ 曲线全消失(实测 1T-TaS2 单层直接变 0 段)。
    """
    if not len(ops):
        return field
    from unfoldfs.geometry import rotation_inverse
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)],
                    axis=1)
    out = np.zeros_like(field)
    cnt = np.zeros_like(field)
    vals = field.ravel()
    for r in ops:
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        ok = np.isfinite(vals)
        np.add.at(out, (i2, j2), np.where(ok, vals, 0.0))
        np.add.at(cnt, (i2, j2), ok.astype(float))
    # cnt == 0 => 该点的所有像都是 NaN => 保持 NaN(不是 0)
    return np.divide(out, cnt, out=np.full_like(out, np.nan), where=cnt > 0)


def get_ops(prim, symprec=1e-4):
    """原胞的完整点群(spglib)"""
    import spglib
    sp = {"Ta": 73, "S": 16, "Nb": 41, "Se": 34}
    numbers = [sp[s] for s in prim.species]
    ds = spglib.get_symmetry_dataset((prim.lattice, prim.positions, numbers),
                                    symprec=symprec)
    keep = [i for i, r in enumerate(ds.rotations)
            if np.allclose(r, np.rint(r), atol=1e-6)]
    return ds.rotations[keep], ds.translations[keep]


def peak_energy_field(a, e_grid, e_f, sigma, win=0.20, nearest=False):
    """每个 k 上 A(k,·) 在 E_F 附近**主峰**的能量 E*(k)(光滑函数)。

    ⚠ 必须确认这个峰是**真的局域极大**(A[j-1] < A[j] > A[j+1]), 否则会把
    "窗口边缘上的强带尾巴"当成峰 —— 那种"峰"离 E_F 有 0.3 eV, E* 在那里乱跳
    ⇒ 凭空画出**孤零零的小圈**(实测: 那些圈上 A(k,E_F)=0, 谱权重全在别处)。
    没有真峰的地方一律给 NaN(等值线自然断开)。

    ``nearest=True``: 在窗口内取**离 E_F 最近**的那个局域极大, 而不是
    **最高**的那个。
    ⚠ 区别很要紧(实测 2026-11-24, 体相 1T-TaS2): 嵌套的费米面上有两条带
      同时靠近 E_F, 最高的那条未必是过 E_F 的那条 ⇒ E* 偏掉 ~0.1 eV ⇒
      轨道平均后偏差中位 0.08 eV ⇒ 曲线的"腰"整段被闸门删掉, 画出来只剩
      中心附近几段小钩。用"最近的"就能全程跟住过 E_F 的那条带。
    """
    ew = np.asarray(e_grid, dtype=float)
    if nearest:
        nE = a.shape[-1]
        left = np.concatenate([a[..., :1], a[..., :-1]], axis=-1)
        right = np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)
        is_peak = (a > left) & (a > right) & (a > 0.0)
        sel = np.abs(ew - e_f) <= win
        cost = np.where(sel & is_peak, np.abs(ew - e_f)[None, None, :], np.inf)
        j = np.argmin(cost, axis=-1)
        found = np.isfinite(np.take_along_axis(cost, j[..., None], axis=-1)[..., 0])
        e0 = ew[j]
    else:
        sel = np.abs(ew - e_f) <= win
        aw = a[..., sel]
        ews = ew[sel]
        j = np.argmax(aw, axis=-1)
        e0 = ews[j]
        # 局域极大检验(在全能量网格上): 左邻 < 峰 < 右邻
        # 窗口内的下标 -> 全能量网格的下标
        sel_idx = np.nonzero(sel)[0]
        jfull = sel_idx[j][..., None]
        left = np.take_along_axis(a, np.clip(jfull - 1, 0, None), axis=-1)[..., 0]
        right = np.take_along_axis(a, np.clip(jfull + 1, 0, None), axis=-1)[..., 0]
        peak = np.take_along_axis(a, jfull, axis=-1)[..., 0]
        is_peak = (peak > left) & (peak > right) & (peak > 0.0)
    dist = np.abs(ew[None, None, :] - e0[..., None])
    near = (dist <= 2.0 * sigma) * a
    num = (near * ew[None, None, :]).sum(axis=-1)
    den = near.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_star = np.where(den > 0, num / den, np.nan)
    return np.where(found if nearest else is_peak, e_star, np.nan)


def weight_at_peak(be, bw, va, e_star):
    """每个 k 上「主峰那条带」的谱权重(离 E* 最近的带)。

    ⚠ 没有有效带的地方必须给 **NaN**, 不能给 0: 取色时用的是高斯核插值
      (``sample_equiv``), 它会把邻域里的 0 混进来 ⇒ 直接计算的费米面上
      权重从 1 被抹成 0.86(假的)。实测(2026-11-24): 原胞直算的 band_w
      恒为 1, 而这里算出来的中位数却是 0.8583 —— 那全是插值污染。
    """
    w = np.full(e_star.shape, np.nan)
    dbest = np.full(e_star.shape, np.inf)
    for b in range(be.shape[0]):
        ok = va[b] & np.isfinite(be[b]) & np.isfinite(e_star)
        db = np.where(ok, np.abs(be[b] - e_star), np.inf)
        m = db < dbest
        dbest = np.where(m, db, dbest)
        w = np.where(m, np.where(ok, bw[b], np.nan), w)
    return w


def wrap_text(text, width=52):
    """按显示宽度折行(中文字符按 1 个宽度算)"""
    import textwrap
    return chr(10).join(textwrap.wrap(text, width))


def panel_data(path, b_prim, b2d, b1, b2, ops, args):
    """一个面板要画的东西 -> (segs, wts, kz_info)

    ``args.refine = N``: 在原网格上算出脊顶能量场之后, 用**协变高斯插值**
    (``sample_equiv``) 把它重建到 N×N 的细网格上再提等值线。
    ⚠ 只在**网格太粗**时用(实测 24×24 的体相 1T-TaS2): 那个网格的间距
      (0.090 2π/Å) 与费米面的特征尺度相当, 直接提等值线得到的是**折线碎片**
      (实测: 原胞的费米面被画成 6 段小钩, 而不是六瓣花瓣 —— 那是采样不足,
      不是物理)。插值重建给出正确的六瓣, 但它终究是插值, 所以正文必须
      写明网格分辨率。
    ⚠ 在**网格够细**时不要用: 96×96 的算例上插值会引入 ~0.24 个网格间距的
      系统偏移(核宽 ~1 格 ⇒ 把脊抹平), 比不插值更差。
    """
    d = load(path.strip(), getattr(args, "kz", None))
    n1, n2, fx, fy = d["n1"], d["n2"], d["fx"], d["fy"]
    e_f = d["e_f"]
    if args.method == "peak":
        e_star = peak_energy_field(d["a"], d["e_grid"], e_f, d["sigma"],
                                   nearest=getattr(args, "nearest", False))
        if not args.no_sym:
            # ⚠ 脊顶能量场**也要**轨道平均, 不只是权重: E*(k) 在对称相关点上
            #   并不一致(嵌套处两条带交换 ⇒ "最高峰"跳到另一条 ⇒ E* 偏掉
            #   ~0.1 eV), 不平均就会在费米面的"腰"处把曲线整段切断
            #   (实测体相 1T-TaS2: 54 个断点, 全在 |k|=0.25)。
            #   平均后腰处的 E* 归零 ⇒ 曲线连续(这正是"轨道平均恢复协变性"
            #   在画费米面时的那一步)。
            e_star = symmetrize(e_star, ops, n1, n2)
        wfield = weight_at_peak(d["be"], d["bw"], d["va"], e_star)
        if not args.no_sym:
            wfield = symmetrize(wfield, ops, n1, n2)
        if args.blur > 0:
            # 笛卡尔等距模糊: 抹平"主峰在副本之间跳变"造成的折角,
            # 同时保持点群协变(笛卡尔核 ⇒ 严格对称)
            e_star = blur_cartesian(e_star, b2d, args.blur)
        i_ef = int(np.argmin(np.abs(d["e_grid"] - e_f)))
        a_ef = d["a"][:, :, i_ef]
        refine = int(getattr(args, "refine", 0) or 0)
        if refine > n1:
            # 重建到细网格(分数坐标居中, 周期延拓由 sample_equiv 处理)
            fxf = (np.arange(refine) + 0.5) / refine - 0.5
            F1, F2 = np.meshgrid(fxf, fxf, indexing="ij")
            cart = np.stack([F1 * b2d[0, 0] + F2 * b2d[1, 0],
                             F1 * b2d[0, 1] + F2 * b2d[1, 1]],
                            axis=-1).reshape(-1, 2)
            e_star = sample_equiv(e_star, cart, b2d,
                                  args.sigma_cells).reshape(refine, refine)
            wfield = sample_equiv(wfield, cart, b2d,
                                  args.sigma_cells).reshape(refine, refine)
            a_ef = sample_equiv(a_ef, cart, b2d,
                                args.sigma_cells).reshape(refine, refine)
            fx = fy = fxf
        # 只保留"峰就在 E_F 附近"的区域: 远处本来就不该有曲线
        e_gate = np.where(np.abs(e_star - e_f) <= args.mask, e_star, np.nan)
        # 再加一道物理闸门: **E_F 处确实有谱权重**的地方才算费米面
        # (否则追踪到的"峰"可能在 E_F 之外很远, 平滑后凭空过零 ⇒ 孤零零的小圈)
        keep = a_ef >= args.wmin * float(np.nanmax(a_ef))
        e_gate = np.where(keep, e_gate, np.nan)
        curves = [(c, None) for c in
                  zero_contours(e_gate - e_f, fx, fy, b2d, b1, b2)]
    else:
        wfield = None
        curves = []
        for b_i in range(d["be"].shape[0]):
            S = np.where(d["va"][b_i] & np.isfinite(d["be"][b_i]),
                         d["be"][b_i] - e_f, np.nan)
            if not np.isfinite(S).any():
                continue
            for c in zero_contours(S, fx, fy, b2d, b1, b2):
                curves.append((c, b_i))
    segs, wts = [], []
    for curve, b_i in curves:
        wfield_use = wfield if b_i is None else d["bw"][b_i]
        for piece in split_at_jumps(curve):
            # ⚠ 颜色必须**一段一个**(取段中点)—— 否则段数与颜色数不等,
            #   set_array 会把颜色整体错位(而且是"一段一错", 很难看出)
            mid = 0.5 * (piece[:-1] + piece[1:])
            for k in range(len(piece) - 1):
                segs.append(piece[k:k + 2])
            wts.extend(sample_equiv(wfield_use, mid, b2d,
                                    args.sigma_cells).tolist())
    return segs, (np.asarray(wts) if wts else np.zeros(0)), \
        dict(kz=float(d["kz_used"]), n3=int(d["n3"]))


def draw_panel(ax, segs, wts, blue, title, hexv, lims, args):
    if blue:
        lc = LineCollection(segs, colors=(0.12, 0.25, 0.95),
                            linewidths=args.lw)
    else:
        lc = LineCollection(
            segs, cmap="jet", linewidths=args.lw,
            norm=Normalize(0.0, float(wts.max()) if wts.size else 1.0))
        lc.set_array(wts)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_facecolor((0.02, 0.03, 0.35))
    ax.add_patch(Polygon(hexv, closed=True, fill=False,
                          edgecolor="crimson", lw=1.4, ls="--"))
    ax.set_xlabel(LBL["x"][args.lang])
    ax.set_ylabel(LBL["y"][args.lang])
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)


def add_colorbar(fig, rect, lang="zh"):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="jet")
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([0.0, 1.0])
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=9, length=2, pad=2)
    cb.outline.set_visible(False)
    cax.set_title(LBL["cb"][lang], fontsize=10, pad=26)
    cax.text(0.5, 1.035, LBL["high"][lang], transform=cax.transAxes,
             ha="center", va="bottom", fontsize=10)
    cax.text(0.5, -0.035, LBL["low"][lang], transform=cax.transAxes,
             ha="center", va="top", fontsize=10)
    return cax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["peak", "band"], default="peak")
    ap.add_argument("--no-sym", action="store_true",
                    help="不做原胞完整点群的轨道平均(权重会继承超胞对称性)")
    ap.add_argument("--blur", type=float, default=1.0,
                    help="画等值线前对主峰能量场做的笛卡尔等距模糊(格点)")
    ap.add_argument("--sigma-cells", type=float, default=1.5,
                    help="取色时高斯核的宽度(格点); 必须用笛卡尔等距核才对称")
    ap.add_argument("--mask", type=float, default=0.25,
                    help="只保留 |E*(k)-E_F| < mask 的区域(挡掉凭空的小圈)")
    ap.add_argument("--wmin", type=float, default=0.02,
                    help="E_F 处谱权重门槛(相对最大值); 低于它的点不参与画费米面")
    ap.add_argument("--lw", type=float, default=2.0)
    ap.add_argument("--blue-panels", default="")
    ap.add_argument("--suptitle", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--split-dir", default="",
                    help="若给出, 每个面板另存一张单独的图(小图)")
    ap.add_argument("--split-prefix", default="panel")
    ap.add_argument("--split-suptitle", default="",
                    help="小图用的标题(默认沿用 --suptitle)")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="图内固定文字的语言(标题/注释由 --titles/--suptitle 给)")
    ap.add_argument("--kz", type=float, default=None,
                    help="要画的 kz 切片(分数坐标, 1.0 = 2π/c); 默认中间那层")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    b_prim = np.asarray(reciprocal(prim.lattice))
    b1, b2 = b_prim[0][:2], b_prim[1][:2]
    b2d = b_prim[:2, :2]
    hexv = bz_hexagon(b_prim)
    hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
    hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
    mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
    lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))
    blue_panels = {int(x) for x in args.blue_panels.split(",") if x.strip() != ""}
    ops, _ = get_ops(prim)

    paths = args.npz.split(",")
    titles = args.titles.split(",")
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.8), squeeze=False)
    axes = axes[0]
    for i, (path, title) in enumerate(zip(paths, titles)):
        segs, wts, kzinfo = panel_data(path, b_prim, b2d, b1, b2, ops, args)
        draw_panel(axes[i], segs, wts, i in blue_panels,
                   f"({chr(ord('a') + i)}) {title.strip()}", hexv, lims, args)
        print(f"{title.strip()}: {len(segs)} 段; 权重 中位 "
              f"{np.median(wts) if wts.size else 0:.4f} max "
              f"{wts.max() if wts.size else 0:.4f}"
              f"   [kz = {kzinfo['kz']:.4f} (共 {kzinfo['n3']} 层)]")
        if args.split_dir:
            os.makedirs(args.split_dir, exist_ok=True)
            f1, a1 = plt.subplots(figsize=(6.4, 6.6))
            draw_panel(a1, segs, wts, i in blue_panels,
                       f"({chr(ord('a') + i)}) {title.strip()}", hexv, lims, args)
            if i not in blue_panels:
                add_colorbar(f1, [0.90, 0.16, 0.016, 0.52], args.lang)
                f1.subplots_adjust(left=0.12, right=0.85, top=0.88, bottom=0.10)
            else:
                f1.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.10)
            st = args.split_suptitle or args.suptitle
            if st:
                f1.suptitle(st, fontsize=12, y=0.975)
            if args.note:
                f1.text(0.5, 0.945, wrap_text(args.note, 52), ha="center",
                        va="top", fontsize=8.5, color="0.35")
            out1 = os.path.join(args.split_dir,
                                f"{args.split_prefix}_{chr(ord('a') + i)}.png")
            f1.savefig(out1, dpi=160)
            plt.close(f1)
            print("  wrote", out1)
    add_colorbar(fig, [0.955, 0.16, 0.013, 0.52], args.lang)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.985)
    if args.note:
        fig.text(0.5, 0.925, args.note, ha="center", fontsize=9.5, color="0.35")
    fig.subplots_adjust(left=0.055, right=0.925, top=0.865, bottom=0.10,
                        wspace=0.24)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
