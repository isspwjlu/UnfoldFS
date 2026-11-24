"""论文图: 展开谱的**点群协变性**(平面波路线)。

面板:
  (a) 展开后的 A(k,E_F) —— 六重对称完好
  (b) 十二个操作的 δ(R); 灰色横杠标出"该操作也是超胞晶体的对称",
      水平虚线是**直接原胞计算**在同一网格上给出的地板

δ(R) 是本图唯一的定量证据, 地板线是它的判据: 折叠本身对被超胞破坏的
操作不等变, 那部分 δ ~ 10⁻¹ 不是展开的错; 只有"也是超胞晶体对称"的
那四个操作, δ 应当落到地板上。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
import plot_fs as P

from unfoldfs.geometry import read_poscar, rotation_inverse

Z = {"Ta": 73, "S": 16, "Nb": 41, "Se": 34, "Mo": 42, "V": 23}

# 图内固定文字(en = 原样, zh = 中文)。高对称点、公式与 δ(R) 符号保持不变。
LBL = {
    "unfolded": {"en": r"(a) unfolded $A(\mathbf{k},E_F)$",
                 "zh": r"(a) 展开后的$A(\mathbf{k},E_F)$"},
    "k1":       {"en": r"$k_1$ (frac.)", "zh": "$k_1$（分数坐标）"},
    "k2":       {"en": r"$k_2$ (frac.)", "zh": "$k_2$（分数坐标）"},
    "floor":    {"en": r"floor: direct primitive-cell calculation, "
                      r"$\delta_{\max}$ = %.1e",
                 "zh": r"地板线：直接原胞计算，$\delta_{\max}$ = %.1e"},
    "grey_bar": {"en": "grey bar: the operation is also a\n"
                      "symmetry of the supercell crystal",
                 "zh": "灰色横杠：该操作也是超胞晶体的对称"},
    "op_x":     {"en": "operation of the primitive point group",
                 "zh": "原胞点群的各个操作"},
}


def image_of(field, r):
    """A(Rk) 的像: |A(k) - A(Rk)|。"""
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2,
                     np.zeros_like(idx, float)], 1)
    f2 = frac @ rotation_inverse(r)
    i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
    j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
    return np.abs(field.ravel() - field.ravel()[i2 * n2 + j2]).reshape(n1, n2)


def per_op_delta(field, ops):
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


def state_symmetry_flags(ops, M, supercell_poscar):
    """该操作诱导的超胞对称是否也是**超胞晶体**的对称。"""
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


def ef_field_of(npz_path, kz=None):
    """A(k, E_F) —— 从**逐带**数据重建(**未**做轨道平均)。

    ⚠ 必须用逐带数据重建, 不能直接取 npz 里的 ``A``: npz 的 ``A`` 是
    **轨道平均之后**的谱函数, 而协变性测量的对象恰恰是"平均之前"的权重
    (对称化会抹掉要测的不对称)。逐带数据 ``band_e``/``band_w`` 没有被
    对称化 ⇒ 重建出来的场才是被测量的那个。
    """
    s = P.load(npz_path, kz)
    i_ef = int(np.argmin(np.abs(s["e_grid"] - s["e_f"])))
    e_valid = np.where(s["va"], s["be"], 0.0)
    w_valid = np.where(s["va"], s["bw"], 0.0)
    sig = float(s["sigma"])
    field = np.zeros((s["n1"], s["n2"]))
    for b in range(s["be"].shape[0]):
        field += w_valid[b] * np.exp(-(e_valid[b] - s["e_grid"][i_ef]) ** 2
                                      / (2 * sig * sig))
    return field


def build(npz, prim_poscar, supercell_poscar, out, png=None, height=2.45,
          matrix="3 0 0;0 1 0;0 0 1", op_probe=1, direct_npz=None,
          panels="ab", lang="en"):
    S.apply_prb_style(lang=lang)
    M = np.array([[float(x) for x in chunk.split()]
                  for chunk in matrix.replace("\n", ";").split(";")
                  if chunk.split()])
    prim = read_poscar(prim_poscar)
    ops, _ = P.get_ops(prim)
    field = ef_field_of(npz)
    d = per_op_delta(field, ops)
    flags = state_symmetry_flags(ops, M, supercell_poscar)
    floor = None
    if direct_npz:
        floor = float(np.max(per_op_delta(ef_field_of(direct_npz), ops)))

    if panels == "a":
        # 只要 (a): 展开后的 A(k,E_F) —— 六重对称一眼可见, 单栏。
        # δ(R) 的逐操作数值放在正文里(它是数字, 不需要一张图)。
        fig = plt.figure(figsize=(3.4, height))
        ax = fig.add_subplot(111)
        # turbo(低端深蓝)⇒ 与全篇的图同色系(旧 paper 的图也是蓝底)
        im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="turbo", vmin=0, vmax=float(np.nanmax(field)),
                       interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(LBL["unfolded"][lang], fontsize=S.FS_TITLE)
        ax.set_xlabel(LBL["k1"][lang])
        ax.set_ylabel(LBL["k2"][lang])
        ax.tick_params(labelsize=S.FS_TICK)
        S.strip_decorations(ax)
        cax = fig.add_axes([0.86, 0.20, 0.045, 0.55])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"$A(\mathbf{k},E_F)$", fontsize=S.FS_ANNOT)
        cb.ax.tick_params(labelsize=S.FS_TICK)
        cb.outline.set_visible(False)
        fig.subplots_adjust(left=0.16, right=0.84, top=0.90, bottom=0.14)
        S.save(fig, out, png_preview=png)
        plt.close(fig)
        print("逐操作 delta(R):")
        for i in range(len(d)):
            print("  op%-2d  %.4e   %s" % (i, d[i], "态的对称" if flags[i] else "-"))
        if floor:
            print("直接原胞计算的地板: %.4e" % floor)
        return d, flags

    fig = plt.figure(figsize=(S.FULL_WIDTH, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 0.04], wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma",
                   vmin=0, vmax=float(np.nanmax(field)),
                   interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(LBL["unfolded"][lang], fontsize=S.FS_TITLE)
    ax.set_xlabel(LBL["k1"][lang])
    ax.set_ylabel(LBL["k2"][lang])
    ax.tick_params(labelsize=S.FS_TICK)
    S.strip_decorations(ax)

    ax = fig.add_subplot(gs[0, 1])
    idx = np.arange(len(d))
    ax.bar(idx, d, width=0.62, color=S.PALETTE["blue_main"])
    if floor:
        ax.axhline(floor, color=S.PALETTE["red_strong"], lw=1.0, ls="--",
                   zorder=4)
        ax.annotate(LBL["floor"][lang] % floor,
                    xy=(len(d) - 0.4, 2.0), xytext=(len(d) - 0.4, 2.0),
                    fontsize=S.FS_ANNOT, va="bottom", ha="right",
                    color=S.PALETTE["red_strong"])
    first = None
    for i, f in enumerate(flags):
        if f:
            ax.plot([i - 0.4, i + 0.4], [0.35, 0.35],
                    color=S.PALETTE["gray_dark"], lw=1.8, solid_capstyle="butt",
                    clip_on=False, zorder=5)
            if first is None:
                first = i
    if first is not None:
        ax.annotate(LBL["grey_bar"][lang],
                    xy=(first, 0.35), xytext=(first + 1.6, 1.6),
                    fontsize=S.FS_ANNOT, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.8,
                                    color=S.PALETTE["gray_dark"]))
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 30.0)
    ax.set_xlim(-0.7, len(d) - 0.3)
    ax.set_xlabel(LBL["op_x"][lang])
    ax.set_ylabel(r"$\delta(R)$")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx])
    ax.tick_params(labelsize=S.FS_TICK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$A(\mathbf{k},E_F)$", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.outline.set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.17)
    S.save(fig, out, png_preview=png)
    plt.close(fig)
    print("逐操作 delta(R):")
    for i in range(len(d)):
        print("  op%-2d  %.4e   %s" % (i, d[i], "态的对称" if flags[i] else "-"))
    if floor:
        print("直接原胞计算的地板: %.4e" % floor)
    return d, flags


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--prim", required=True)
    ap.add_argument("--supercell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--height", type=float, default=2.45)
    ap.add_argument("--matrix", default="3 0 0;0 1 0;0 0 1")
    ap.add_argument("--op-probe", type=int, default=1)
    ap.add_argument("--direct-npz", default="",
                    help="原胞直接计算的 npz; 给了就把它的 δ(R) 当作地板画出来")
    ap.add_argument("--panels", choices=("ab", "a"), default="ab",
                    help="ab = (a) 谱图 + (b) δ(R); a = 只要 (a), 单栏")
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="图内固定文字的语言")
    args = ap.parse_args()
    build(args.npz, args.prim, args.supercell, args.out,
          png=args.png or None, height=args.height, matrix=args.matrix,
          op_probe=args.op_probe, direct_npz=args.direct_npz or None,
          panels=args.panels, lang=args.lang)


if __name__ == "__main__":
    main()
