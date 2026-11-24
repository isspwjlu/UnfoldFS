"""缺陷超胞: 反折叠费米面与原胞直算的重合(平面波路线)。

与 :mod:`unfoldfs.fs` 的验证协议同口径: 取 kz 切片, 把原胞直算的费米面与
反折叠的费米面对照, 算逐点的"点到曲线"距离(以网格间距为单位)。

⚠ 原胞与超胞的 k 网格**不同**(原胞 48×48×6, 超胞 96×96×1)⇒ 比较只能在
笛卡尔空间做(费米面曲线本身与网格无关; 权重则各自是"过各自 E_F 的那条带"
的权重, 跨体系可比)。

用法:
    python scripts/analysis/defect_compare.py \
        --prim-npz .fsdata/defect_prim_orb.npz \
        --sc-npz   .fsdata/defect_perfect_pw.npz \
        --prim     .fsdata/defect_PRIMCELL.vasp --kz 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import plot_fs as P                                          # noqa: E402
from unfoldfs.geometry import read_poscar, reciprocal        # noqa: E402


class _A:
    def __init__(self, kz):
        self.no_sym = False
        self.blur = 1.0
        self.sigma_cells = 1.5
        self.mask = 0.25
        self.wmin = 0.02
        self.method = "peak"
        self.kz = kz


def fs_of(npz, b, b2d, ops, kz):
    segs, wts, _ = P.panel_data(npz, b, b2d, b[0][:2], b[1][:2], ops, _A(kz))
    return np.array(segs), wts


def curve_distance(a, b):
    """双向的"曲线到曲线"距离(笛卡尔坐标, 单位 2π/Å)。"""
    va = a.reshape(-1, 2)
    vb = b.reshape(-1, 2)
    d_ab = cKDTree(vb).query(va)[0]
    d_ba = cKDTree(va).query(vb)[0]
    return np.concatenate([d_ab, d_ba])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--sc-npz", required=True, help="超胞反折叠的 npz")
    ap.add_argument("--prim", required=True, help="原胞 POSCAR")
    ap.add_argument("--kz", type=float, default=0.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    b = np.asarray(reciprocal(prim.lattice))
    b2d = b[:2, :2]   # 面内的 2x2 倒格矢块(与 paper_fig_defect.py 同口径)
    ops, _ = P.get_ops(prim)

    seg_p, w_p = fs_of(args.prim_npz, b, b2d, ops, args.kz)
    seg_s, w_s = fs_of(args.sc_npz, b, b2d, ops, args.kz)
    kf = np.asarray(np.load(args.sc_npz)["k_frac"])
    n1 = int(len(np.unique(np.round(kf[:, 0], 6))))
    print("### %s" % (args.label or ""))
    print("  原胞直算: %d 段; 谱权重 中位 %.4f max %.4f"
          % (len(seg_p), np.median(w_p) if len(w_p) else float("nan"),
             np.max(w_p) if len(w_p) else float("nan")))
    print("  反折叠  : %d 段; 谱权重 中位 %.4f max %.4f"
          % (len(seg_s), np.median(w_s) if len(w_s) else float("nan"),
             np.max(w_s) if len(w_s) else float("nan")))
    if not len(seg_p):
        print("  原胞在该能量下**没有**费米面(0 段) => 无法做同能量对照")
    if len(seg_p) and len(seg_s):
        # 换算成**网格间距**(与 paper_fig_verify.py 同一口径): h = |b1| / n1
        h = float(np.linalg.norm(b2d[0])) / n1
        d = curve_distance(seg_p, seg_s) / h
        print("  网格间距 h = %.5f (2pi/Ang)" % h)
        print("  点到曲线距离: 中位 %.5f 格; 最大 %.5f 格; "
              "落在 0.5 格内的比例 %.2f%%"
              % (np.median(d), d.max(), 100.0 * np.mean(d < 0.5)))


if __name__ == "__main__":
    main()
