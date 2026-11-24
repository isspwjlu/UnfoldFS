"""体相 1T-TaS2: 原胞 / 完美 3x3x1 / 缺陷 Ta9S16 的反折叠对比。

回答三个问题:
  1. **3x3x1 超胞的展开可信吗?** —— 完美超胞的费米面与原胞直算的费米面
     的重合率(论文的硬判据: 反折叠只重排权重, 不该移动费米面)。
  2. **缺陷把费米面改成什么样了?** —— 缺陷超胞与原胞的 FS 重合率,
     以及 E_F 处谱权重的变化 A(k, E_F)。
  3. **缺陷态落在原胞 BZ 的哪里?** —— ΔA(k, E_F) 的空间分布
     (超胞被破坏平移对称性后, 缺陷态的权重会集中到哪些 k)。

用法:
  python compare_defect.py --prim PRIMCELL_relaxed.vasp \
      --npz prim=fs/A(k,E).npz;perfect=fs/A(k,E).npz;defect=fs/A(k,E).npz \
      --kz 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "figures"))
from unfoldfs.geometry import read_poscar, reciprocal
import plot_fs as P


class _A:
    """plot_fs.panel_data 需要的参数袋(与 paper_fig_fs 的口径一致)。"""

    def __init__(self, kz):
        self.no_sym = False
        self.blur = 1.0
        self.sigma_cells = 1.5
        self.mask = 0.25
        self.wmin = 0.02
        self.method = "peak"
        self.kz = kz


def fs_curves(npz, b, b2d, ops, kz):
    """与 plot_fs / paper_fig_fs 完全同口径地重建一个面板的曲线。"""
    segs, wts, info = P.panel_data(npz, b, b2d, b[0][:2], b[1][:2], ops, _A(kz))
    return np.array(segs) if len(segs) else np.zeros((0, 2, 2)), wts, info


def overlap_ratio(curve_a, curve_b, h, tol_cells=0.5):
    """两条 FS 曲线的重合率(双向: 顶点到最近顶点的距离 < tol_cells 个网格)。

    返回 (重合率, 距离中位数(格), 距离最大值(格))。
    """
    if len(curve_a) == 0 or len(curve_b) == 0:
        return float("nan"), float("nan"), float("nan")
    va = curve_a.reshape(-1, 2)
    vb = curve_b.reshape(-1, 2)
    d = np.concatenate([cKDTree(vb).query(va)[0], cKDTree(va).query(vb)[0]]) / h
    return float((d < tol_cells).mean()), float(np.median(d)), float(d.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True,
                    help="分号分隔的 name=path (用分号不用逗号: "
                         "unfoldfs 产出的文件名 'A(k,E).npz' 里含逗号)")
    ap.add_argument("--kz", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.5, help="重合判据(网格数)")
    args = ap.parse_args()

    paths = dict(p.split("=", 1) for p in args.npz.split(";"))
    prim = read_poscar(args.prim)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]
    ops, _ = P.get_ops(prim)

    curves, weights, efs, aefs = {}, {}, {}, {}
    for name, path in paths.items():
        segs, wts, info = fs_curves(path, b, b2d, ops, args.kz)
        d = P.load(path, args.kz)
        curves[name] = segs
        weights[name] = wts
        efs[name] = d["e_f"]
        i_ef = int(np.argmin(np.abs(d["e_grid"] - d["e_f"])))
        aefs[name] = d["a"][:, :, i_ef]
        print(f"[{name}] E_F = {d['e_f']:.6f} eV;  kz = {info['kz']:.4f} "
              f"(共 {info['n3']} 层);  FS 段数 = {len(segs)};  "
              f"A(k,E_F): 中位 {np.median(aefs[name]):.5f} "
              f"max {aefs[name].max():.5f}")

    h = float(np.linalg.norm(b2d[0])) / P.load(paths[list(paths)[0]], args.kz)["n1"]
    print(f"\n网格间距 = {h:.5f} (2π/A);  重合判据 = {args.tol} 格")

    print("\n== 费米面重合率(反折叠 vs 原胞直算) ==")
    base = "prim"
    for name in paths:
        if name == base:
            continue
        r, med, mx = overlap_ratio(curves[name], curves[base], h, args.tol)
        print(f"  {name:8s} vs {base}: 重合率 {r * 100:.2f}%   "
              f"距离 中位 {med:.4f} 格 / 最大 {mx:.4f} 格")

    if "perfect" in paths and "defect" in paths:
        r, med, mx = overlap_ratio(curves["defect"], curves["perfect"], h, args.tol)
        print(f"  {'defect':8s} vs {'perfect'}: 重合率 {r * 100:.2f}%   "
              f"距离 中位 {med:.4f} 格 / 最大 {mx:.4f} 格")

    if "perfect" in paths and "defect" in paths:
        print("\n== 缺陷引起的 E_F 处谱权重变化 ==")
        da = aefs["defect"] - aefs["perfect"]
        print(f"  ΔA(k,E_F): 最小 {da.min():.5f}  中位 {np.median(da):.5f}  "
              f"最大 {da.max():.5f}")
        thr = 0.1 * max(aefs["perfect"].max(), 1e-12)
        print(f"  A_defect - A_perfect > 0.1*max(A_perfect) 的网格点: "
              f"{int((da > thr).sum())} / {da.size}")
        # 增量最大的 k 在 BZ 里的位置(分数坐标)
        d = P.load(paths["defect"], args.kz)
        i1, i2 = np.unravel_index(np.argmax(da), da.shape)
        print(f"  增量最大处: 网格索引 ({i1}, {i2});  "
              f"分数坐标 k = ({d['fx'][i1]:.4f}, {d['fy'][i2]:.4f})")
        # 谱权重(过 E_F 那条带)的分布
        for name in ("perfect", "defect"):
            w = weights[name]
            if w.size:
                print(f"  {name:8s} 的谱权重 w(k): 中位 {np.median(w):.4f} "
                      f"max {w.max():.4f}  min {w.min():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
