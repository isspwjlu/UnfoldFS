"""画**任意 kz 切片**上的 kx-ky 费米面(多切片并排)。

kz 用**分数坐标**(沿 c 的倒格矢为单位, 1.0 = 2π/c)。网格由
`unfoldfs gen --grid n1 n2 n3 --grid-type gamma` 生成 ⇒ n3 = 3 时正好是
**kz = 0, 1/3, 2/3** 三层。

⚠ **单层体系(1T-TaS2 / 1H-NbSe2)带在 kz 上是平的**(层间耦合 e^{-c/3} ≈ 0.003)⇒
   各 kz 切片的费米面**本来就该完全相同**。现在手上的算例是 2D k 网格
   (只有 kz = 0 一层), 所以任意 kz 都只能返回那一层 —— 要看到真正不同的切片,
   需要带 kz 采样的算例(见 setup_fs_kz.sh)。

用法:
    python plot_fs_kz.py --prim PRIMCELL.vasp \
        --npz "tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz" \
        --titles "3×1×1 超胞反折叠" \
        --kz-list 0,0.3333,0.6667 --out fs_kz.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from unfoldfs.geometry import read_poscar, reciprocal
import plot_fs as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True, help="逗号分隔: 每个超胞一个 npz")
    ap.add_argument("--titles", required=True)
    ap.add_argument("--kz-list", required=True, help="逗号分隔的 kz(分数坐标)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["peak", "band"], default="peak")
    ap.add_argument("--no-sym", action="store_true")
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--sigma-cells", type=float, default=1.5)
    ap.add_argument("--mask", type=float, default=0.25)
    ap.add_argument("--wmin", type=float, default=0.02)
    ap.add_argument("--lw", type=float, default=1.6)
    ap.add_argument("--blue-panels", default="")
    ap.add_argument("--suptitle", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    b_prim = np.asarray(reciprocal(prim.lattice))
    b1, b2 = b_prim[0][:2], b_prim[1][:2]
    b2d = b_prim[:2, :2]
    ops, _ = P.get_ops(prim)
    hexv = P.bz_hexagon(b_prim)
    hx0, hx1 = hexv[:, 0].min(), hexv[:, 0].max()
    hy0, hy1 = hexv[:, 1].min(), hexv[:, 1].max()
    mx, my = 0.06 * (hx1 - hx0), 0.06 * (hy1 - hy0)
    lims = ((hx0 - mx, hx1 + mx), (hy0 - my, hy1 + my))
    blue_panels = {int(x) for x in args.blue_panels.split(",") if x.strip() != ""}

    kz_list = [float(x) for x in args.kz_list.split(",") if x.strip() != ""]
    paths = args.npz.split(",")
    titles = args.titles.split(",")
    nrow, ncol = len(paths), len(kz_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow),
                             squeeze=False)
    wmax = 0.0
    data = []
    for r, path in enumerate(paths):
        row = []
        for c, kz in enumerate(kz_list):
            args.kz = kz
            segs, wts, info = P.panel_data(path.strip(), b_prim, b2d, b1, b2,
                                          ops, args)
            row.append((segs, wts, info))
            if r not in blue_panels and wts.size:
                wmax = max(wmax, float(wts.max()))
        data.append(row)

    for r, row in enumerate(data):
        for c, (segs, wts, info) in enumerate(row):
            ax = axes[r][c]
            if r in blue_panels:
                lc = LineCollection(segs, colors=(0.12, 0.25, 0.95),
                                    linewidths=args.lw)
            else:
                lc = LineCollection(segs, cmap="jet",
                                    norm=Normalize(0.0, wmax),
                                    linewidths=args.lw)
                lc.set_array(wts)
            ax.add_collection(lc)
            ax.set_aspect("equal")
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_facecolor((0.02, 0.03, 0.35))
            from matplotlib.patches import Polygon
            ax.add_patch(Polygon(hexv, closed=True, fill=False,
                                  edgecolor="crimson", lw=1.4, ls="--"))
            ax.set_xlabel("$k_1$  (2π/Å)")
            ax.set_ylabel("$k_2$  (2π/Å)")
            ax.set_title(f"kz = {info['kz']:.4f}"
                         + (f"  (共 {info['n3']} 层)" if info["n3"] > 1 else ""),
                         fontsize=10)
            ax.tick_params(labelsize=8)
            print(f"{titles[r].strip():24s} kz={info['kz']:.4f}: {len(segs)} 段"
                  f"  权重中位 {np.median(wts) if wts.size else 0:.4f}"
                  f"  (数据里共 {info['n3']} 层)")
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.985)
    if args.note:
        fig.text(0.5, 0.012, args.note, ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08,
                        wspace=0.22, hspace=0.30)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
