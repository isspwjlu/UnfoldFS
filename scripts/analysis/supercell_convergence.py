#!/usr/bin/env python3
"""超胞尺寸收敛检验: 展开谱与原胞直算的差异, 是有限尺寸效应还是展开误差?

问题: 展开谱与原胞直算的 A(k,E_F) 有差异(Pearson 0.72)。这个差异有两个可能来源:
  (a) **有限尺寸**: 超胞的带与原胞的带本来就不一样(3x1x1 的超胞有 3 倍的折叠,
      其谱函数在原胞 BZ 上的投影自然不同于原胞自己的谱)
  (b) **展开误差**: 投影步骤本身引入的形状失真

区分办法: 做**超胞尺寸序列**(3x1x1 -> 6x6x1 -> ...)。若差异随超胞增大而
单调下降, 收敛到 0, 则 (a) 成立; 若停在某个平台, 那个平台就是 (b)。

本脚本只做分析与画图, 不负责跑 VASP。

用法:
  python scripts/analysis/supercell_convergence.py --prim A_prim.npz \
         --sc A_sc3x1x1.npz [--sc A_sc6x6x1.npz ...] [--label 3x1x1 ...] \
         [--out figures/]

判据说明(全部在同一原胞网格上逐点比较, 不做插值):
  * Pearson / 相对L2: 在各自 E_F 处取 A(k,E_F), 归一化到各自 max 后比较
    (Pearson 对仿射不变, 所以它反映的是**空间形状**的差异, 不含整体强度差)
  * 峰位差: 每个 k 的 A(k,.) 的能量重心之差 -> 分辨"谱结构错位"与"强度分布差异"
  * 带能量差: 逐 k 比较能带能量集合(按能量排序后配对) -> 有限尺寸效应的直接度量
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in d.keys()}
    return out


def _at_ef(data, ef=None):
    """取 A(k, E_F) 列; ef 缺省用文件自带的。"""
    e = data["e_grid"]
    ef = float(data["ef"]) if ef is None else float(ef)
    j = int(np.argmin(np.abs(e - ef)))
    return data["A"][:, j], e[j], j


def _norm(v):
    m = np.max(np.abs(v))
    return v / m if m > 0 else v


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _rel_l2(a, b):
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n > 0 else float("nan")


def _spectral_moment(data, lo=None, hi=None):
    """每个 k 的 A(k,.) 的能量重心(eV)。"""
    e = data["e_grid"]
    m = (e >= lo) & (e <= hi) if (lo is not None and hi is not None) else np.ones_like(e, bool)
    A = data["A"][:, m]
    e = e[m]
    w = A.sum(axis=1)
    mom = np.where(w > 0, (A * e[None, :]).sum(axis=1) / np.where(w > 0, w, 1.0), np.nan)
    return mom


def _band_energies(data):
    """每个 k 的有效带能量集合(升序, 去掉 valid=False 的补位槽)。"""
    be, va = data["band_e"], data["valid"]
    out = []
    for i in range(be.shape[0]):
        out.append(np.sort(be[i][va[i]]))
    return out


def compare(prim, sc, name):
    """在两者之间做全套比较, 返回一行指标字典。"""
    # 网格一致性
    kp = np.asarray(prim["k_frac"])
    ks = np.asarray(sc["k_frac"])
    if kp.shape != ks.shape:
        raise SystemExit(f"{name}: k 网格形状不同 {kp.shape} vs {ks.shape}")
    dev = np.abs(kp - ks)
    # 周期距离(避免 0.99999997 vs 0.0)
    dev = np.minimum(dev, 1.0 - dev)
    max_dev = float(dev.max())

    a_p, ef_p, _ = _at_ef(prim)
    a_s, ef_s, _ = _at_ef(sc)
    np_, ns_ = _norm(a_p), _norm(a_s)

    # 峰位(能量重心)
    mp = _spectral_moment(prim)
    ms = _spectral_moment(sc)
    ok = np.isfinite(mp) & np.isfinite(ms)
    d_mom = float(np.nanmax(np.abs(ms[ok] - mp[ok]))) if ok.any() else float("nan")
    mean_mom = float(np.nanmean(np.abs(ms[ok] - mp[ok]))) if ok.any() else float("nan")

    # 带能量(有限尺寸的直接度量): 逐 k 比较两边能量集合的最近距离
    be_p, be_s = _band_energies(prim), _band_energies(sc)
    dmin_all = []
    for bp, bs in zip(be_p, be_s):
        if len(bp) == 0 or len(bs) == 0:
            continue
        # 原胞的每条带到超胞带集合的最近距离
        d = np.abs(bs[None, :] - bp[:, None]).min(axis=1)
        dmin_all.append(d)
    if dmin_all:
        dmin_all = np.concatenate(dmin_all)
        band_med = float(np.median(dmin_all))
        band_max = float(dmin_all.max())
        band_p95 = float(np.percentile(dmin_all, 95))
    else:
        band_med = band_max = band_p95 = float("nan")

    return {
        "name": name,
        "k_max_dev": max_dev,
        "ef_prim": ef_p,
        "ef_sc": ef_s,
        "d_ef": float(ef_s - ef_p),
        "pearson_own_ef": _pearson(a_p, a_s),
        "pearson_norm_own_ef": _pearson(np_, ns_),
        "rel_l2_norm_own_ef": _rel_l2(ns_, np_),
        "pearson_norm_common_ef": _pearson(
            _norm(prim["A"][:, int(np.argmin(np.abs(prim["e_grid"] - ef_s)))]),
            _norm(sc["A"][:, int(np.argmin(np.abs(sc["e_grid"] - ef_s)))]),
        ),
        "d_moment_max_eV": d_mom,
        "d_moment_mean_eV": mean_mom,
        "band_dmin_median_eV": band_med,
        "band_dmin_p95_eV": band_p95,
        "band_dmin_max_eV": band_max,
    }


def symmetrize(data, gm_path, primcell, symprec=1e-5):
    """按**原胞完整点群**做轨道平均(与 `unfoldfs unfold` 的默认行为一致)。

    需要 grid_map.npz(unfoldfs gen 的产物) 与原胞 POSCAR。
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
    from unfoldfs.grid import load_grid_map
    from unfoldfs.geometry import primitive_symmetry_ops, read_poscar
    from unfoldfs.expand import symmetrize_spectral_function

    gm = load_grid_map(gm_path)
    ops = primitive_symmetry_ops(read_poscar(primcell), symprec)
    out = dict(data)
    out["A"] = symmetrize_spectral_function(data["A"], gm, ops=ops)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--sc", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--symmetrize", action="store_true",
                    help="对展开结果按原胞完整点群做轨道平均(需要 --gm 与 --primcell)")
    ap.add_argument("--gm", nargs="*", default=None,
                    help="每个 --sc 对应的 grid_map.npz(顺序一致)")
    ap.add_argument("--primcell", default=None, help="原胞 POSCAR(--symmetrize 时必需)")
    ap.add_argument("--symprec", type=float, default=1e-5)
    args = ap.parse_args()

    prim = load(args.prim)
    labels = args.label or [os.path.basename(p) for p in args.sc]
    rows = []
    for i, (p, lb) in enumerate(zip(args.sc, labels)):
        sc = load(p)
        if args.symmetrize:
            if not args.gm or not args.primcell:
                raise SystemExit("--symmetrize 需要 --gm 与 --primcell")
            gm = args.gm[i] if len(args.gm) > i else args.gm[0]
            sc = symmetrize(sc, gm, args.primcell, args.symprec)
        rows.append(compare(prim, sc, lb))

    cols = [
        ("name", "算例"),
        ("k_max_dev", "k网格最大偏差"),
        ("d_ef", "ΔE_F (eV)"),
        ("pearson_norm_own_ef", "Pearson(归一化)"),
        ("rel_l2_norm_own_ef", "相对L2(归一化)"),
        ("pearson_norm_common_ef", "Pearson(共同E_F)"),
        ("d_moment_max_eV", "谱重心差 max (eV)"),
        ("band_dmin_median_eV", "带能量最近距 中位 (eV)"),
        ("band_dmin_p95_eV", "带能量最近距 p95 (eV)"),
    ]
    def fmt(v):
        return v if isinstance(v, str) else f"{v:.6g}"

    widths = [max(len(c[1]), *(len(fmt(r[c[0]])) for r in rows)) for c in cols]
    print(" | ".join(c[1].ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(fmt(r[c[0]]).ljust(w) for c, w in zip(cols, widths)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
