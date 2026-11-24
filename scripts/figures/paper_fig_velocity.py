"""论文 §III.A 的费米速度分析: 等值面(两壁) vs 脊顶。

回答的问题: 从**展宽谱函数**取费米面时, 社区惯用的 A(k,E_F) 等值面给出的是
高斯脊的**两壁**; 这在物理上意味着什么?

本脚本给出三个量(都在**真实数据**上算, 见 paper/fact_sheet.md §2.13):

  1. **能量偏移**: 在两壁处, 过 E_F 的那条带离 E_F 有多远(中位/90 分位);
     在脊顶处同样量(只剩网格离散残差)。理论: 局部线性带下 ≈ σ√(2ln2)。
  2. **面积**: 脊顶 / 内壁 / 外壁各自围的面积, 以及内外壁均值与脊顶的差
     ⇒ 为什么这个错误**不容易被发现**(几何量几乎抵消)。
  3. **费米速度**: v_F = (1/ħ)|dE/dk|, 在脊顶轨迹与两壁轨迹上各算一遍,
     按方位角配对后给比值 ⇒ 局域量**不抵消**。

用法(需要服务器上的 A(k,E).npz 与原胞 POSCAR):
    python paper_fig_velocity.py --npz <A(k,E).npz> --prim <PRIMCELL.vasp> \
        [--label "1H-NbSe2"] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from unfoldfs.fs import (closed_loops, crest_curves, crest_field,
                         load_spectral_npz, loop_area, spectral_curves,
                         velocity_to_si)

HBAR = 6.582119569e-16      # eV*s
# 脊顶场的笛卡尔模糊宽度(单位网格格数)—— 与出图脚本的 --blur 1.0 同口径
BLUR_CELLS = 1.0
M_PER_ANG = 1.0e-10         # 1 Å = 1e-10 m


def reciprocal_from_npz(sl) -> np.ndarray:
    """由 npz 的 (k_frac, k_cart) 反推 2D 倒格矢(行=基矢, 单位 1/Å)。"""
    kf = np.asarray(sl.k_frac, dtype=float)
    kc = np.asarray(sl.k_cart, dtype=float)
    b, *_ = np.linalg.lstsq(kf, kc, rcond=None)
    return b[:2, :2]


def band_at(sl, i: int, j: int, e_ref: float) -> tuple[float, int]:
    """(i,j) 处**能量最接近 e_ref** 的有效带 → (能量, 带号)。"""
    be = sl.band_e[:, i, j]
    va = sl.valid[:, i, j]
    ok = va & np.isfinite(be)
    if not ok.any():
        return float("nan"), -1
    idx = np.nonzero(ok)[0]
    ib = idx[int(np.argmin(np.abs(be[idx] - e_ref)))]
    return float(be[ib]), int(ib)


def slope_at(sl, i: int, j: int, e_ref: float, steps) -> tuple[float, float]:
    """过 E_F 的那条带在 (i,j) 的局域斜率 (dE/dk_1, dE/dk_2), 单位 eV·Å。

    沿每个方向中心差分; 邻居处按**能量连续性**跟带(避免在能级交叉处跟错带)。
    """
    e0, _ = band_at(sl, i, j, e_ref)
    if not np.isfinite(e0):
        return float("nan"), float("nan")
    out = []
    for axis in (0, 1):
        if axis == 0:
            ep = band_at(sl, (i + 1) % sl.n1, j, e0)[0]
            em = band_at(sl, (i - 1) % sl.n1, j, e0)[0]
        else:
            ep = band_at(sl, i, (j + 1) % sl.n2, e0)[0]
            em = band_at(sl, i, (j - 1) % sl.n2, e0)[0]
        if not (np.isfinite(ep) and np.isfinite(em)):
            out.append(float("nan"))
        else:
            out.append((ep - em) / (2.0 * steps[axis]))
    return out[0], out[1]


def shoelace_area(curves) -> float:
    """折线围出的面积(shoelace, 逐段闭合)。"""
    tot = 0.0
    for c in curves:
        if len(c) < 3:
            continue
        x, y = c[:, 0], c[:, 1]
        tot += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                             - np.dot(y, np.roll(x, -1))))
    return tot


def analyse(npz_path: str, prim_path: str, label: str) -> dict:
    sl = load_spectral_npz(npz_path)
    b2d = reciprocal_from_npz(sl)
    e_f = sl.e_f
    n1, n2 = sl.n1, sl.n2
    steps = (float(np.linalg.norm(b2d[0])) / n1,
             float(np.linalg.norm(b2d[1])) / n2)
    b_inv = np.linalg.inv(b2d)

    crest = crest_field(sl.a, sl.e_grid, e_f, sl.sigma, max_shift=0.25,
                        min_weight_frac=0.02, b2d=b2d,
                        blur_cells=BLUR_CELLS)
    cc = crest_curves(crest - e_f, sl.fx, sl.fy, b2d, b2d[0], b2d[1])
    fc = spectral_curves(sl.a_ef, sl.fx, sl.fy, b2d, b2d[0], b2d[1])

    def sample(curves):
        vb, de, pts = [], [], []
        for c in curves:
            for xy in c:
                kf = (xy @ b_inv) % 1.0
                i = int(np.argmin(np.abs(sl.fx - kf[0])))
                j = int(np.argmin(np.abs(sl.fy - kf[1])))
                d1, d2 = slope_at(sl, i, j, e_f, steps)
                v = (np.hypot(d1, d2) / HBAR * M_PER_ANG
                     if np.isfinite(d1) and np.isfinite(d2) else np.nan)
                e_b, _ = band_at(sl, i, j, e_f)
                vb.append(v)
                de.append(abs(e_b - e_f))
                pts.append(xy)
        return np.asarray(vb), np.asarray(de), np.asarray(pts)

    vc, dec, pc = sample(cc)
    vf, def_, pf = sample(fc)

    # 按方位角分箱配对(不能用最近邻: 壁-顶间距小于网格间距 ⇒ 会配到同一格)
    def binned(pts, v, min_pts=5):
        r = np.hypot(pts[:, 0], pts[:, 1])
        ang = (np.arctan2(pts[:, 1], pts[:, 0]) + np.pi) / (2 * np.pi)
        b = np.clip((ang * 360).astype(int), 0, 359)
        res = {}
        for k in range(360):
            m = (b == k) & np.isfinite(v)
            if m.sum() >= min_pts:
                res[k] = (float(np.median(r[m])), float(np.median(v[m])))
        return res

    cb, fb = binned(pc, vc), binned(pf, vf)
    ratios = [fb[k][1] / cb[k][1] for k in cb
              if k in fb and cb[k][1] > 0 and fb[k][0] > cb[k][0] * 0.9]

    crest_r = float(np.median(np.hypot(pc[:, 0], pc[:, 1])))
    inner = [c for c in fc
             if float(np.median(np.hypot(c[:, 0], c[:, 1]))) < crest_r]
    outer = [c for c in fc
             if float(np.median(np.hypot(c[:, 0], c[:, 1]))) >= crest_r]
    # **径向括号**: 两壁在半径上近似对称地夹住脊顶 ⇒ 任何依赖"平均位置"的
    # 量(面积等)几乎不受影响, 而局域量不抵消。
    # ⚠ 不用"曲线围出的面积"来量化: 曲线被切开后再拼回闭合回路容易产生
    #   伪回路(算出的面积会超过布里渊区面积, 物理上不可能)。
    def _med_r(curves):
        rs = [float(np.median(np.hypot(c[:, 0], c[:, 1]))) for c in curves
              if len(c)]
        return float(np.median(rs)) if rs else float("nan")

    # Γ 口袋 = 脊顶里半径最小的那一组
    r_all = [( _med_r([c]), c) for c in cc]
    r_crest = min(r_all, key=lambda t: t[0])[0]
    inner = [c for c in fc if _med_r([c]) < r_crest]
    outer = [c for c in fc if r_crest < _med_r([c]) < 1.15 * r_crest]
    r_in, r_out = _med_r(inner), _med_r(outer)
    r_mid = 0.5 * (r_in + r_out)
    a_crest = r_crest
    a_in, a_out = r_in, r_out

    out = {
        "label": label, "e_f": e_f, "sigma": sl.sigma,
        "grid": [n1, n2],
        "v_crest_median": float(np.nanmedian(vc)),
        "v_flank_median": float(np.nanmedian(vf)),
        "dE_crest_median": float(np.nanmedian(dec)),
        "dE_flank_median": float(np.nanmedian(def_)),
        "dE_flank_p90": float(np.percentile(def_, 90)),
        "area_crest": a_crest, "area_inner": a_in, "area_outer": a_out,
        "radial_offset_rel": float(abs(r_mid - r_crest) / r_crest),
        "radial_offset_in": float((r_crest - r_in) / r_crest),
        "radial_offset_out": float((r_out - r_crest) / r_crest),
        "r_crest": r_crest, "r_inner": r_in, "r_outer": r_out,
        "ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "ratio_iqr": [float(np.percentile(ratios, 25)),
                      float(np.percentile(ratios, 75))] if ratios else None,
        "theory_dE": float(sl.sigma * np.sqrt(2 * np.log(2))),
    }
    return out


def curve_samples(npz_path: str, prim_path: str = ""):
    """逐点采样两条轨迹(脊顶 / 两壁): (v_F, |E_b−E_F|, 方位角, 笛卡尔点)。

    画图与统计**共用**这一份口径 —— 避免两处实现漂移。
    """
    sl = load_spectral_npz(npz_path)
    b2d = reciprocal_from_npz(sl)
    e_f = sl.e_f
    n1, n2 = sl.n1, sl.n2
    steps = (float(np.linalg.norm(b2d[0])) / n1,
             float(np.linalg.norm(b2d[1])) / n2)
    b_inv = np.linalg.inv(b2d)

    crest = crest_field(sl.a, sl.e_grid, e_f, sl.sigma, max_shift=0.25,
                        min_weight_frac=0.02, b2d=b2d,
                        blur_cells=BLUR_CELLS)
    cc = crest_curves(crest - e_f, sl.fx, sl.fy, b2d, b2d[0], b2d[1])
    fc = spectral_curves(sl.a_ef, sl.fx, sl.fy, b2d, b2d[0], b2d[1])

    def sample(curves):
        vb, de, ang, pts = [], [], [], []
        for c in curves:
            for xy in c:
                kf = (xy @ b_inv) % 1.0
                i = int(np.argmin(np.abs(sl.fx - kf[0])))
                j = int(np.argmin(np.abs(sl.fy - kf[1])))
                d1, d2 = slope_at(sl, i, j, e_f, steps)
                v = (np.hypot(d1, d2) / HBAR * M_PER_ANG
                     if np.isfinite(d1) and np.isfinite(d2) else np.nan)
                e_b, _ = band_at(sl, i, j, e_f)
                vb.append(v)
                de.append(abs(e_b - e_f))
                ang.append(float(np.arctan2(xy[1], xy[0])))
                pts.append(xy)
        return (np.asarray(vb), np.asarray(de), np.asarray(ang),
                np.asarray(pts))

    return {
        "sl": sl, "b2d": b2d, "e_f": e_f,
        "crest_curves": cc, "flank_curves": fc,
        "crest": sample(cc), "flank": sample(fc),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--prim", default=None, help="(未用: 倒格矢从 npz 反推)")
    p.add_argument("--label", default="")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    res = analyse(a.npz, a.prim, a.label)
    for k, v in res.items():
        print(f"{k}: {v}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
