"""平面波路线的论文图: **同一超胞、同一代码, 只换投影基**的对照。

核心论点: 反折叠谱在晶体点群下的协变性缺陷, 到底属于"反折叠"本身, 还是
属于"轨道投影"这个近似? 受控对照: 同一个超胞计算、同一套网格、同一段下游
代码, 只有**权重**的来源不同。

面板(上排):
  (a) 平面波路线的原始 A(k,E_F)      (b) 轨道投影路线的原始 A(k,E_F)
  (c) 平面波路线的 |A(k)-A(Rk)|      (d) 轨道投影路线的 |A(k)-A(Rk)|
下排:
  (e) 逐操作的 δ(R)(两条路线); 灰色横杠标出"该操作也是**超胞晶体**的对称"

(c)(d) 用对数色标: 平面波路线几乎全黑(残差比 1e-6 还小), 轨道投影路线
在费米面上亮起 —— 这就是"缺陷在权重里、而且落在费米面上"的样子。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P
from paper_fig_symmetry import raw_A

from unfoldfs.geometry import read_poscar, reciprocal, rotation_inverse

Z = {"Ta": 73, "S": 16, "Nb": 41, "Se": 34, "Mo": 42, "V": 23}


def _image_of(field: np.ndarray, r: np.ndarray) -> np.ndarray:
    """A(Rk) 的像: |A(k) - A(Rk)|。"""
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2,
                     np.zeros_like(idx, float)], 1)
    f2 = frac @ rotation_inverse(r)
    i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
    j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
    return np.abs(field.ravel() - field.ravel()[i2 * n2 + j2]).reshape(n1, n2)


def per_op_delta(field: np.ndarray, ops) -> np.ndarray:
    """每个操作的 δ(R) = max_k |A(k) - A(Rk)| / max_k A(k)。"""
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2,
                     np.zeros_like(idx, float)], 1)
    vals = field.ravel()
    mx = float(np.nanmax(field))
    out = []
    for r in np.asarray(ops):
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        out.append(float(np.abs(vals - vals[i2 * n2 + j2]).max() / mx))
    return np.array(out)


def state_symmetry_flags(ops, M: np.ndarray, supercell_poscar: str) -> np.ndarray:
    """逐操作判定: 该操作诱导的超胞对称是否也是**超胞晶体**的对称。

    判据: 原胞的操作 R 在超胞基下的表示 R_sc = M^-T R M^T; 若 R_sc 是超胞
    点群(用 spglib 从结构算)里的一个操作, 则该操作也是**态本身**的对称
    ⇒ 它下面的 δ(R) 若不为零, 就只能是方法的问题。
    """
    import spglib
    st = read_poscar(supercell_poscar)
    cell = (np.asarray(st.lattice), np.asarray(st.positions),
            np.array([Z[x] for x in st.species]))
    sc_ops = np.asarray(spglib.get_symmetry_dataset(cell, symprec=1e-3).rotations)
    minv_t = np.linalg.inv(M).T
    flags = []
    for r in np.asarray(ops):
        r_sc = minv_t @ np.asarray(r, float) @ M.T
        flags.append(bool(np.any(np.all(np.isclose(sc_ops, r_sc, atol=1e-6),
                                       axis=(1, 2)))))
    return np.array(flags)


def build(pw_npz, orb_npz, prim_npz, prim_poscar, supercell_poscar, out,
          png=None, height=4.6, matrix="3 0 0;0 1 0;0 0 1", op_probe=1):
    S.apply_prb_style()
    M = np.array([[float(x) for x in chunk.split()]
                  for chunk in matrix.replace("\n", ";").split(";")
                  if chunk.split()])

    prim = read_poscar(prim_poscar)
    ops_full, _ = P.get_ops(prim)

    a_pw, d = raw_A(pw_npz, P.load(pw_npz)["e_f"])
    a_orb, _ = raw_A(orb_npz, P.load(orb_npz)["e_f"])
    a_ref, _ = raw_A(prim_npz, P.load(prim_npz)["e_f"])

    d_pw = per_op_delta(a_pw, ops_full)
    d_orb = per_op_delta(a_orb, ops_full)
    flags = state_symmetry_flags(ops_full, M, supercell_poscar)
    r_probe = np.asarray(ops_full)[op_probe]

    diff_pw = _image_of(a_pw, r_probe) / float(np.nanmax(a_pw))
    diff_orb = _image_of(a_orb, r_probe) / float(np.nanmax(a_orb))

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.66], wspace=0.30,
                          hspace=0.45, width_ratios=[1, 1, 1, 1, 0.055])

    vmax = float(np.nanmax([a_pw.max(), a_orb.max(), a_ref.max()]))
    maps = [(a_pw, "plane-wave unfolding"), (a_orb, "orbital projection")]
    for k, (f, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(f.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title("(%s) %s" % (S.letter(k), title), fontsize=S.FS_TITLE)
        ax.set_xlabel(r"$k_1$ (frac.)")
        if k == 0:
            ax.set_ylabel(r"$k_2$ (frac.)")
        ax.tick_params(labelsize=S.FS_TICK)
        S.strip_decorations(ax)

    # ---- (c)(d) 不对称图 |A(k) - A(Rk)| ----
    norm = LogNorm(vmin=1e-6, vmax=1.0)
    for k, (f, title) in enumerate([(diff_pw, "plane wave"),
                                    (diff_orb, "orbital")]):
        ax = fig.add_subplot(gs[0, 2 + k])
        im2 = ax.imshow(f.T, origin="lower", extent=[0, 1, 0, 1],
                        cmap="magma", norm=norm, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title("(%s) asymmetry, %s" % (S.letter(2 + k), title),
                     fontsize=S.FS_TITLE)
        ax.set_xlabel(r"$k_1$ (frac.)")
        ax.tick_params(labelsize=S.FS_TICK)
        S.strip_decorations(ax)

    # 两个色标: 叠放在最右列
    gs_c = gs[0, 4].subgridspec(2, 1, hspace=0.45)
    cax = fig.add_subplot(gs_c[0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$A(\mathbf{k},E_F)$", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.outline.set_visible(False)
    cax2 = fig.add_subplot(gs_c[1])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(r"$|A(\mathbf{k})-A(R\mathbf{k})|/A_{\max}$",
                  fontsize=S.FS_ANNOT)
    cb2.ax.tick_params(labelsize=S.FS_TICK)
    cb2.outline.set_visible(False)

    # ---- (e) 逐操作 δ(R) ----
    ax = fig.add_subplot(gs[1, :])
    ops_idx = np.arange(len(d_pw))
    w = 0.38
    ax.bar(ops_idx - w / 2, d_pw, width=w, color=S.PALETTE["blue_main"],
           label="plane wave")
    ax.bar(ops_idx + w / 2, d_orb, width=w, color=S.PALETTE["red_strong"],
           label="orbital projection")
    first = None
    for i, f in enumerate(flags):
        if f:
            ax.plot([i - 0.46, i + 0.46], [1.6, 1.6],
                    color=S.PALETTE["gray_dark"], lw=1.6, solid_capstyle="butt",
                    clip_on=False, zorder=5)
            if first is None:
                first = i
    if first is not None:
        ax.annotate("grey bar: the operation is also a symmetry\n"
                    "of the supercell crystal",
                    xy=(first, 1.6), xytext=(4.6, 4.2), fontsize=S.FS_ANNOT,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.8,
                                    color=S.PALETTE["gray_dark"]))
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 8.0)
    ax.set_xlim(-0.7, len(d_pw) - 0.3)
    ax.set_xlabel("point-group operation of the primitive cell")
    ax.set_ylabel(r"$\delta(R)$")
    ax.set_xticks(ops_idx)
    ax.set_xticklabels([str(i) for i in ops_idx])
    ax.tick_params(labelsize=S.FS_TICK)
    ax.legend(fontsize=S.FS_LEG, loc="upper left", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.08)
    S.save(fig, out, png_preview=png)
    plt.close(fig)

    print("逐操作 δ(R):")
    for i in range(len(d_pw)):
        print("  op%-2d  平面波 %.4e   轨道投影 %.4e   %s"
              % (i, d_pw[i], d_orb[i], "态的对称" if flags[i] else "-"))
    return d_pw, d_orb, flags


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw-npz", required=True)
    ap.add_argument("--orb-npz", required=True)
    ap.add_argument("--prim-npz", required=True)
    ap.add_argument("--prim", required=True)
    ap.add_argument("--supercell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=4.6)
    ap.add_argument("--matrix", default="3 0 0;0 1 0;0 0 1")
    ap.add_argument("--op-probe", type=int, default=1,
                    help="用来画不对称图的操作编号")
    args = ap.parse_args()
    build(args.pw_npz, args.orb_npz, args.prim_npz, args.prim, args.supercell,
          args.out, png=args.png or None, height=args.height,
          matrix=args.matrix, op_probe=args.op_probe)


if __name__ == "__main__":
    main()
