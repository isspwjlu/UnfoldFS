"""费米面图的"场"画法: 背景 = A(k, E_F), 费米面 = 亮线描边。

为什么不用"把费米面按谱权重着色"(旧画法, ``paper_fig_fs.py``):

* 在**平面波严格权重**下, 一个展开态几乎完全属于**唯一一个**候选动量
  (完备性恒等式 Σ_j w_j = 1), 于是"费米面上那条带"的权重几乎是常数
  (1T-TaS2: 0.86; 1H-NbSe2: 0.94) ⇒ 色标上三个面板**颜色几乎一样**,
  读者看不出差别, 只看到"整条线一个颜色"。
* 直接计算的费米面与反折叠的费米面用**两套编码**(蓝线 vs 权重色)画在
  同一张图上时, 形状明明逐点相同, 看上去却像两条不同的曲线。

"场"画法把两件事都解决了:

* **颜色 = A(k, E_F)(绝对值, 共享色标)**: 亮度直接就是该动量在费米能级上
  携带的谱权重。直接计算的脊高 1.0, 反折叠的脊高 0.86 (TaS2) —— 差别
  看得见, 而且三个面板是**同一个量**, 可以直接比。
* **费米面用亮线描出**: 亮线是"脊顶"(E_b = E_F), 背景是"脊"
  (高斯宽度 ∝ σ/|∇E|) ⇒ 一眼能看出两者不是同一条线, 也正好呼应
  论文里"脊顶 vs 等值面"的论点。

坐标: 背景场画在**笛卡尔 k 空间**(2π/Å), 与旧画法一致; 网格本身是
倒格子的平行四边形, 所以用 ``pcolormesh`` 而不是 ``imshow``。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_fs as P

# 费米面描边的颜色: 与背景的 magma(黑→紫→橙→黄白)互补, 亮处也分得清
COLOR_FS = "#2EE6C8"
COLOR_REF = "#8FB6FF"        # 直接计算的费米面(参考): 淡蓝, **更粗**


def pad_field(field, fx, fy, pad=1):
    """把周期场延拓到 [-pad, 1+pad) 的分数坐标区间。

    返回 (field_pad, fx_pad, fy_pad)。
    """
    if not pad:
        return field, fx, fy
    n1, n2 = field.shape
    i1 = np.arange(-pad * n1, (pad + 1) * n1) % n1
    i2 = np.arange(-pad * n2, (pad + 1) * n2) % n2
    return (field[i1][:, i2],
            np.arange(-pad * n1, (pad + 1) * n1) / n1,
            np.arange(-pad * n2, (pad + 1) * n2) / n2)


def ef_field(npz_path, kz=None, pad=1):
    """A(k, E_F) 的二维场 + 分数坐标轴(**周期延拓**)。

    返回 (field, fx, fy, e_f); field 的行列是 (n1, n2)。

    ⚠ 必须延拓: 分数坐标网格只覆盖 [0,1)², 换成笛卡尔坐标后只占倒格胞的
      一个平行四边形 —— 而第一布里渊区在倒格胞**中间**, 不延拓的话背景场
      只填满 BZ 的右上角, 其余是黑的(看着像画错)。延拓到 [-1,2)² 后
      BZ 被完整覆盖。
    """
    d = P.load(npz_path, kz)
    i_ef = int(np.argmin(np.abs(d["e_grid"] - d["e_f"])))
    f = d["a"][:, :, i_ef]
    f, fx, fy = pad_field(f, d["fx"], d["fy"], pad)
    return f, fx, fy, d["e_f"]


def cart_grid(fx, fy, b2d):
    """分数坐标网格 → 笛卡尔坐标网格 (n1, n2, 2)。"""
    F1, F2 = np.meshgrid(fx, fy, indexing="ij")
    X = F1 * b2d[0, 0] + F2 * b2d[1, 0]
    Y = F1 * b2d[0, 1] + F2 * b2d[1, 1]
    return X, Y


def clip_to_bz(field, X, Y, hexv):
    """把第一布里渊区**之外**的场置 NaN。

    场是周期的, 延拓之后 BZ 外会出现费米面的**周期像**(那些斜条纹)。
    它们不是第一 BZ 里的费米面, 画出来只会让读者以为"费米面跑到 BZ 外了"。
    ⇒ 只保留 BZ 内。
    """
    from matplotlib.path import Path
    inside = Path(np.asarray(hexv, dtype=float)).contains_points(
        np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape)
    return np.where(inside, field, np.nan)


def refine_field(field2d, b2d, n, sigma_cells=1.5):
    """把 (太粗的) 网格上的场用协变高斯核插值重建到 n×n 的分数网格。

    与 ``plot_fs.panel_data`` 的 ``refine`` **同一个核、同一个口径** ⇒
    背景的脊与画出来的费米面线才落在同一个地方(否则看着是错位的)。
    返回 (field, fx, fy); 分数坐标取 [-0.5, 0.5), 正好覆盖第一布里渊区。
    """
    f = (np.arange(n) + 0.5) / n - 0.5
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    cart = np.stack([F1 * b2d[0, 0] + F2 * b2d[1, 0],
                     F1 * b2d[0, 1] + F2 * b2d[1, 1]],
                    axis=-1).reshape(-1, 2)
    out = P.sample_equiv(field2d, cart, b2d, sigma_cells).reshape(n, n)
    return out, f, f


def seg_values_by_chain(segs, field2d, b2d, sigma_cells=1.5, n_per_chain=240):
    """按**曲线链**取色(而不是逐段取) ⇒ 线上的颜色是连续的, 不锯齿。

    逐段取色时, 相邻两段的取值只要有一点差别就会显示成一节一节
    (实测 1H-NbSe2 的费米面尤其明显)。这里先把段接回成链, 沿链按弧长
    等距采样, 再把采样值线性插值回每一段。
    """
    segs = np.asarray(segs, dtype=float)
    if len(segs) == 0:
        return None
    chains = _chains(segs)
    out = np.zeros(len(segs))
    idx_all = np.arange(len(segs))
    # 段 → 链的归属: 用中点做最近邻
    mids = 0.5 * (segs[:, 0] + segs[:, 1])
    used = set()
    for ch in chains:
        d = np.linalg.norm(mids[:, None, :] - ch[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        members = np.where(assign < len(ch) - 1)[0]
        if len(members) == 0:
            continue
        used.update(members.tolist())
        pts = ch
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        if s[-1] <= 0:
            continue
        n = min(n_per_chain, max(8, len(pts)))
        query = np.linspace(0, s[-1], n)
        qi = np.clip(np.searchsorted(s, query) - 1, 0, len(pts) - 2)
        frac = (query - s[qi]) / np.maximum(s[qi + 1] - s[qi], 1e-12)
        xy = pts[qi] + frac[:, None] * (pts[qi + 1] - pts[qi])
        vals = P.sample_equiv(field2d, xy, b2d, sigma_cells)
        # 把采样值按弧长映射回每一段的中点
        mid_s = np.interp(
            0.5 * (s[assign[members]] + s[np.minimum(assign[members] + 1,
                                                      len(pts) - 1)]),
            query, vals)
        out[members] = mid_s
    if len(used) < len(segs):          # 没归到链上的(罕见)用整体取色
        rest = np.setdiff1d(idx_all, list(used))
        if len(rest):
            out[rest] = P.sample_equiv(field2d, mids[rest], b2d, sigma_cells)
    return out


def seg_values(field2d, cart_pts, b2d, sigma_cells=1.5):
    """在笛卡尔点上取场的值(协变高斯核插值, 与取色同口径)。"""
    return P.sample_equiv(field2d, np.atleast_2d(cart_pts), b2d, sigma_cells)


def draw_field_panel(ax, field, fx, fy, b2d, segs, hexv, lims, norm,
                     cmap="turbo", lw=1.5, fs_color=COLOR_FS,
                     ref_segs=None, ref_color=COLOR_REF, ref_lw=3.0,
                     rasterized=True, clip_bz=True, seg_vals=None,
                     line_norm=None, line_cmap=None):
    """画一个"背景场 + 费米面"的面板。

    ``seg_vals``: 每段费米面中点处的 A(k,E_F) —— 给定则**按它给线着色**。
    ``line_norm``: 线专用的色标(缺省与背景共用)。费米面上 A 的取值集中在
    0.5~1 ⇒ 线若用完整色标, 有一半是深蓝(与底色混在一起, 看不见);
    给线单独一个只覆盖亮端的区间, 线就既清楚又有层次。
    """
    X, Y = cart_grid(fx, fy, b2d)
    # BZ 边界先建好, 后面用它做**精确裁剪**(set_clip_path)。若改成在数据
    # 层把 BZ 外的点置 NaN, gouraud 着色会在 NaN 边界上插值出锯齿毛边。
    bz_patch = Polygon(hexv, closed=True, fill=False,
                       edgecolor="#55659f", lw=0.8, ls="--", alpha=0.6,
                       zorder=1)
    ax.add_patch(bz_patch)
    im = ax.pcolormesh(X, Y, field, cmap=cmap, norm=norm, shading="gouraud",
                        rasterized=rasterized, zorder=0)
    if clip_bz:
        im.set_clip_path(bz_patch)
    if ref_segs is not None and len(ref_segs):
        # ⚠ 参考线画成**虚线**: 两条线逐点重合(中位分离 10⁻⁴ 个网格间距),
        #   实线会被完全盖住, 读者会以为只画了一条; 虚线的缺口把主线切成
        #   一节一节 ⇒ 一眼看出"上面压着一条别的线"。
        ax.add_collection(LineCollection(ref_segs, colors=ref_color,
                                         linewidths=ref_lw, alpha=0.95,
                                         linestyles=(0, (3.0, 2.2)),
                                         zorder=2))
    if len(segs):
        if seg_vals is not None:
            lc = LineCollection(segs,
                                cmap=(line_cmap if line_cmap is not None
                                      else cmap),
                                norm=(line_norm if line_norm is not None
                                      else norm),
                                linewidths=lw, zorder=3)
            lc.set_array(np.asarray(seg_vals, dtype=float))
        else:
            lc = LineCollection(segs, colors=fs_color, linewidths=lw, zorder=3)
        ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])


def line_colorbar(fig, rect, cmap="plasma", vmin=0.33, vmax=0.72,
                  label=None):
    """费米面**线**专用的色标(与背景同一个量, 但范围窄得多)。"""
    cax = fig.add_axes(rect)
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower",
               extent=(0.0, 1.0, 0.0, 1.0), interpolation="nearest")
    cax.set_xticks([])
    cax.set_yticks([vmin, vmax])
    cax.set_yticklabels(["%.2f" % vmin, "%.2f" % vmax])
    cax.tick_params(labelsize=6, length=2, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)
    if label:
        cax.set_ylabel(label, fontsize=7, rotation=270, labelpad=10,
                       va="bottom")
    return cax



def smooth_chains(segs, sigma_len=0.02, n_per_chain=600, n_out=None):
    """把费米面曲线在**弧长**上做高斯平滑, 去掉插值带来的高频毛刺。

    ⚠ 这是**画图层面**的平滑, 不改数据: 谱权重、段数、重合距离等数字都
      由提取本身给出, 与这里无关。它只把"同一条曲线上的锯齿"抹平 ——
      网格越粗, 插值重建的曲线毛刺越多(实测 24×24 的缺陷算例)。

    ``sigma_len``: 高斯核的半宽(2π/Å), 应当 ≪ 费米面的特征尺度(~0.1)。
    """
    segs = np.asarray(segs, dtype=float)
    if len(segs) == 0:
        return segs
    out = []
    for ch in _chains(segs):
        if len(ch) < 4:
            out.extend([tuple(p) for p in ch])
            continue
        d = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(ch, axis=0), axis=1))])
        total = d[-1]
        if total <= 0:
            continue
        n = n_per_chain if n_out is None else n_out
        q = np.linspace(0.0, total, n)
        xi = np.interp(q, d, ch[:, 0])
        yi = np.interp(q, d, ch[:, 1])
        # 弧长上的高斯卷积(端点用最近的边界值)
        if sigma_len > 0:
            rad = max(1, int(3 * sigma_len / max(total / n, 1e-12)))
            ker = np.exp(-0.5 * (np.arange(-rad, rad + 1) *
                                 (total / n) / sigma_len) ** 2)
            ker /= ker.sum()
            xi = np.convolve(np.pad(xi, rad, mode="edge"), ker, mode="valid")
            yi = np.convolve(np.pad(yi, rad, mode="edge"), ker, mode="valid")
        for i in range(len(xi) - 1):
            out.append(((xi[i], yi[i]), (xi[i + 1], yi[i + 1])))
    return np.array(out) if out else segs

def field_colorbar(fig, rect, vmax=1.0, label=r"$A(\mathbf{k},E_F)$"):
    """竖直色标。⚠ 不用 ``fig.colorbar``: 它在 ``savefig(bbox_inches="tight")``
    下会把色条压扁并裁掉上端刻度 ⇒ 改成自己用 imshow 画, extent 明确。"""
    cax = fig.add_axes(rect)
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap="turbo", origin="lower",
               extent=(0.0, 1.0, 0.0, 1.0), interpolation="nearest")
    cax.set_xticks([])
    cax.set_yticks([0.0, 0.5, 1.0])
    cax.set_yticklabels(["0", "0.5", "1"])
    cax.tick_params(labelsize=7, length=2, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)
    # 标签用 ylabel(竖排, 在色标右边)而不是 title(在色标上方)——
    # title 会和最右一列面板的标题撞在一起。
    cax.set_ylabel(label, fontsize=8, rotation=270, labelpad=12,
                   va="bottom")
    return cax


def rebuild_a(d, e_grid):
    """从 npz 的**逐带**数据重建 A(k,E): (n1, n2, nE)。

    unfold 的谱函数本来就是 Σ_b w_b·exp(−(E−E_b)²/2σ²), 所以这是等价
    的。用途: npz 自带的能量窗口是以它自己的 E_F 为中心的, 要看**别的
    能量**(例如把体相原胞取到缺陷超胞的 E_F 上)就必须重建。
    """
    ev = np.where(d["va"], d["be"], 0.0)
    wv = np.where(d["va"], d["bw"], 0.0)
    sig = float(d["sigma"])
    a = np.zeros((d["n1"], d["n2"], len(e_grid)))
    for b in range(ev.shape[0]):
        a += wv[b][..., None] * np.exp(
            -(ev[b][..., None] - e_grid[None, None, :]) ** 2 / (2 * sig * sig))
    return a


def fs_from_field(a, e_grid, e_f, b2d, b1, b2, ops, blur=1.0, mask=0.25,
                   sigma_cells=1.5, wmin=0.02):
    """从**给定的** A(k,E) 场提取费米面 —— 与 ``plot_fs.panel_data`` 同一口径。

    返回 (segs, a_at_ef)。
    """
    e_star = P.peak_energy_field(a, e_grid, e_f, 0.06)
    e_star = P.symmetrize(e_star, ops, a.shape[0], a.shape[1])
    if blur > 0:
        e_star = P.blur_cartesian(e_star, b2d, blur)
    e_gate = np.where(np.abs(e_star - e_f) <= mask, e_star, np.nan)
    i_ef = int(np.argmin(np.abs(e_grid - e_f)))
    a_ef = a[:, :, i_ef]
    keep = a_ef >= wmin * float(np.nanmax(a_ef))
    e_gate = np.where(keep, e_gate, np.nan)
    fx = np.arange(a.shape[0]) / a.shape[0]
    fy = np.arange(a.shape[1]) / a.shape[1]
    segs = []
    for curve in P.zero_contours(e_gate - e_f, fx, fy, b2d, b1, b2):
        for piece in P.split_at_jumps(curve):
            segs.extend(piece[k:k + 2] for k in range(len(piece) - 1))
    return (np.array(segs) if segs else np.zeros((0, 2, 2))), a_ef


def key_of(p, tol=6):
    return (round(float(p[0]), tol), round(float(p[1]), tol))


def _chains(segs, tol=6):
    """把逐段的曲线接回成**连续的链**(段之间首尾相接)。

    ``panel_data`` 返回的是逐段数组, 顺序上本来是连续的, 但被
    ``split_at_jumps`` 切开的地方会断 ⇒ 这里按端点重新接上。
    """
    segs = np.asarray(segs, dtype=float)
    if len(segs) == 0:
        return []

    def key(p):
        return (round(float(p[0]), tol), round(float(p[1]), tol))

    ends = {}
    for i, s in enumerate(segs):
        ends.setdefault(key(s[0]), []).append(i)
        ends.setdefault(key(s[1]), []).append(i)
    used = [False] * len(segs)
    out = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        pts = [segs[i][0], segs[i][1]]
        for _ in range(len(segs)):
            tail = key(pts[-1])
            nxt = [j for j in ends.get(tail, []) if not used[j]]
            if not nxt:
                break
            j = nxt[0]
            used[j] = True
            p0, p1 = segs[j][0], segs[j][1]
            pts.append(p1 if key(p0) == tail else p0)
        out.append(np.asarray(pts))
    return out


def chain_segments(segs):
    """把逐段的曲线接回成链, 返回每条链上的**段索引**序列。"""
    segs = np.asarray(segs, dtype=float)
    if len(segs) == 0:
        return []
    ends = {}
    for i, sg in enumerate(segs):
        ends.setdefault(key_of(sg[0]), []).append(i)
        ends.setdefault(key_of(sg[1]), []).append(i)
    used = [False] * len(segs)
    chains = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = [i]
        # 向两端延伸
        for _ in range(len(segs)):
            for side in (0, 1):
                tail = segs[chain[-1] if side else chain[0]][side]
                cand = [j for j in ends.get(key_of(tail), []) if not used[j]]
                if not cand:
                    break
                j = cand[0]
                used[j] = True
                if side:
                    chain.append(j)
                else:
                    chain.insert(0, j)
        chains.append(chain)
    return chains


def smooth_weights(segs, vals, window=11):
    """沿**曲线链**平滑每段的权重 ⇒ 线上的颜色连续过渡。

    逐段取色时, 相邻两段的权重只要差一点点就会显示成一节一节(斑驳);
    旧 paper 的图是沿曲线连续渐变的。做法: 段权重 → 顶点权重 → 沿链做
    移动平均 → 回到段(两端顶点取平均)。
    """
    vals = np.asarray(vals, dtype=float)
    if len(segs) == 0 or vals.size == 0:
        return vals
    out = vals.copy()
    for chain in chain_segments(segs):
        if len(chain) < 3:
            continue
        v = vals[chain]
        # 顶点权重: 链有 len(v) 个段 => len(v)+1 个顶点
        vp = np.empty(len(v) + 1)
        vp[0] = v[0]
        vp[-1] = v[-1]
        vp[1:-1] = 0.5 * (v[:-1] + v[1:])
        w = max(3, int(window) | 1)
        kern = np.ones(w) / w
        vp = np.convolve(np.pad(vp, w // 2, mode="edge"), kern, mode="valid")
        for a, i in enumerate(chain[:-1]):
            out[i] = 0.5 * (vp[a] + vp[a + 1])
    return out


def zoom_window(segs, span=0.2, frac=0.5):
    """在费米面上挑一段做放大窗: **最长的一条链**上、按弧长取 ``frac`` 处。

    ``span`` 是窗口的边长(笛卡尔单位, 2π/Å)。用固定的边长而不是"链长的
    百分比" —— 后者在长链上会给出巨大的窗口, 放大后什么也看不出来。

    返回 (x0, x1, y0, y1) —— 笛卡尔坐标。
    """
    chains = _chains(segs)
    if not chains:
        return None
    pts = max(chains, key=len)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                         axis=1))])
    i = int(np.searchsorted(d, frac * d[-1]))
    i = min(max(i, 1), len(pts) - 1)
    c = pts[i]
    half = 0.5 * span
    return (c[0] - half, c[0] + half, c[1] - half, c[1] + half)
