"""缺陷超胞与体相原胞的**同能量**对照(平面波路线)。

为什么需要这个脚本
------------------
缺陷超胞与体相原胞的费米能级不同(去掉 2 个 S 少了 12 个电子 ⇒ E_F 低
0.61 eV), 拿两个不同 E_F 的费米面直接比是**跨能量比较**。要比较, 必须把
原胞的谱函数取到**缺陷的 E_F** 上。

⚠ 陷阱: 原胞的 ``A(k,E).npz`` 的能量窗口是以原胞自己的 E_F 为中心的
  (3.9057 ~ 4.9057 eV), **不覆盖**缺陷的 E_F = 3.799 eV。
  若只是把 npz 里的 ``ef`` 改小(旧做法), 提取费米面时窗口与能量网格没有
  交集 ⇒ 峰找不到 ⇒ "0 段" ⇒ 会被误读成"体相在那个能量没有费米面"。
  实际上那里**有**态(见本脚本的输出)。

做法: 从 npz 的**逐带**数据 ``band_e``/``band_w`` 在本地重建 A(k,E) ——
unfold 的谱函数本来就是 Σ_b w_b·exp(−(E−E_b)²/2σ²), 所以重建是等价
的, 而且 σ 与工具一致。重建之后用与工具相同的口径(peak + 三道闸门 +
轨道平均 + 笛卡尔模糊)提取费米面。

用法:
  python scripts/analysis/defect_same_energy.py \
      --prim-npz .fsdata/defect_prim96_pwref.npz \
      --defect-npz .fsdata/defect_vacancy_pw.npz \
      --prim .fsdata/defect_PRIMCELL_prim96.vasp
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "figures"))

import plot_fs as P
import fs_field as F
from unfoldfs.geometry import read_poscar, reciprocal


def radii(segs):
    if len(segs) == 0:
        return np.zeros(0)
    r = np.linalg.norm(segs.reshape(-1, 2), axis=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--defect-npz", required=True)
    ap.add_argument("--prim", required=True)
    ap.add_argument("--kz", type=float, default=0.0)
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    ops, _ = P.get_ops(prim)

    dp = P.load(args.prim_npz, args.kz)
    dd = P.load(args.defect_npz, args.kz)
    e_def = float(dd["e_f"])
    raw = np.load(args.defect_npz)
    vbm = float(raw["vbm"]) if "vbm" in raw.files else float("nan")
    cbm = float(raw["cbm"]) if "cbm" in raw.files else float("nan")
    print(f"体相原胞 E_F = {dp['e_f']:.4f} eV   缺陷超胞 E_F = {e_def:.4f} eV"
          f"   (差 {dp['e_f'] - e_def:+.4f} eV)")
    print(f"缺陷超胞: VBM = {vbm:.4f}, CBM = {cbm:.4f}"
          f"  ⇒ E_F {'落在带隙内(绝缘体)' if vbm < e_def < cbm else '在带内(金属)'}")

    # ---- 体相原胞在缺陷的 E_F 上的谱函数与费米面 ----
    e_grid = np.linspace(e_def - 0.5, e_def + 0.5, 401)
    a = F.rebuild_a(dp, e_grid)
    segs_p, a_p = F.fs_from_field(a, e_grid, e_def, b2d, b[0][:2],
                              b[1][:2], ops)
    print(f"\n体相原胞 @ {e_def:.4f} eV:  A 的最大值 {a_p.max():.4f}, "
          f"> 1e-3 的动量占 {(a_p > 1e-3).mean():.4f}")
    print(f"  费米面段数 {len(segs_p)}"
          + (f", 半径 {radii(segs_p).min():.3f} ~ {radii(segs_p).max():.3f}"
             if len(segs_p) else "  ⇒ 该能量上**没有**费米面"))

    # ---- 体相原胞在自己的 E_F 上的费米面(对照) ----
    e_grid0 = np.linspace(dp["e_f"] - 0.5, dp["e_f"] + 0.5, 401)
    a0 = F.rebuild_a(dp, e_grid0)
    segs_0, _ = F.fs_from_field(a0, e_grid0, float(dp["e_f"]), b2d,
                                  b[0][:2], b[1][:2], ops)
    print(f"体相原胞 @ 自己的 E_F {dp['e_f']:.4f} eV: 费米面段数 {len(segs_0)}"
          + (f", 半径 {radii(segs_0).min():.3f} ~ {radii(segs_0).max():.3f}"
             if len(segs_0) else ""))

    # ---- 缺陷超胞的费米面 ----
    class _A:
        no_sym = False
        blur = 1.0
        sigma_cells = 1.5
        mask = 0.25
        wmin = 0.02
        method = "peak"
        kz = args.kz
    sd, wd, _ = P.panel_data(args.defect_npz, b, b2d, b[0][:2], b[1][:2],
                              ops, _A())
    sd = np.array(sd)
    print(f"缺陷超胞 @ {e_def:.4f} eV: 费米面段数 {len(sd)}"
          + (f", 半径 {radii(sd).min():.3f} ~ {radii(sd).max():.3f}"
             if len(sd) else ""),
          f", 谱权重中位 {np.median(wd):.4f}" if len(wd) else "")

    # ---- 两者在同一能量上的位置关系 ----
    if len(segs_p) and len(sd):
        from scipy.spatial import cKDTree
        h = float(np.linalg.norm(b2d[0])) / dp["n1"]
        d = cKDTree(segs_p.reshape(-1, 2)).query(sd.reshape(-1, 2))[0] / h
        print(f"\n缺陷 FS 到体相@{e_def:.4f} 费米面的距离: 中位 {np.median(d):.4f} 格, "
              f"最大 {d.max():.3f} 格; 落在 0.5 格内的比例 "
              f"{100 * (d < 0.5).mean():.2f}%")
    print(f"\n网格间距 h = {float(np.linalg.norm(b2d[0])) / dp['n1']:.5f} (2π/Å)")


if __name__ == "__main__":
    main()
