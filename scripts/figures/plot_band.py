"""展开能带对比图(最终画法, 按用户 2026-11-19 的三条要求):

  1. y 轴 = E - E_F, E_F 处在 0(白色虚线);
  2. 超胞面板 = `--unfold-mode full` + **按画窗归一化的线性色标**
     (弱副本暗但可见, 强带亮);
  3. 原胞面板 = **原 band 本身、纯蓝线**(不画 unfolded)。

用法:
  python plot_band_final.py --ef -2.703913 \
      --prim-dat prim.dat --unfold sc3x1x1.dat,scsqrt3.dat \
      --titles "3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
      --labels "GAMMA:0,M:1.078,K:1.700,GAMMA:2.945" \
      --out out.png [--e-min -4.2] [--e-max 1.0] [--sigma 0.05]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from unfoldfs import band

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_dat(path):
    """-> (dist (n_k,), energy (n_band, n_k), weight (n_band, n_k))"""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            q = line.split()
            rows.append((float(q[0]), float(q[1]), float(q[2]), int(q[3])))
    bands = sorted({r[3] for r in rows})
    dists = sorted({round(r[0], 9) for r in rows})
    bi = {b: i for i, b in enumerate(bands)}
    ki = {d: i for i, d in enumerate(dists)}
    en = np.full((len(bands), len(dists)), np.nan)
    w = np.zeros((len(bands), len(dists)))
    for d, e, ww, b in rows:
        en[bi[b], ki[round(d, 9)]] = e
        w[bi[b], ki[round(d, 9)]] = ww
    return np.array(dists), en, w


def heatmap_panel(ax, dist, en, w, e_f, lo, hi, sigma, cmap="jet",
                  rasterized=False):
    """A(k,E) 热图: 颜色 = 按画窗归一化的线性权重, y 轴 = E - E_F。"""
    step = max((hi - lo) / 1400.0, 1e-3)
    e_grid = np.arange(lo, hi + 0.5 * step, step)
    bl = np.nanmin(en, axis=1)[:, None]
    bh = np.nanmax(en, axis=1)[:, None]
    mask = np.broadcast_to((bh >= lo) & (bl <= hi), w.shape)
    wcol = band._band_figure_colors(w, "linear", mask=mask).reshape(w.shape)
    # 路径端点被算两次 => 去重(否则 pcolormesh 画退化格)
    keep = np.concatenate([[True], np.diff(dist) > 0])
    A = band.band_spectral_function(dist[keep], en[:, keep], wcol[:, keep],
                                    e_grid, sigma)
    im = ax.pcolormesh(dist[keep], e_grid - e_f, A, cmap=cmap, vmin=0.0,
                       vmax=1.0, shading="auto", rasterized=rasterized)
    ax.axhline(0.0, color="w", linestyle="--", linewidth=0.9, zorder=3)
    ax.set_ylim(lo - e_f, hi - e_f)
    ax.set_xlim(dist[0], dist[-1])
    return im


def lines_panel(ax, dist, en, e_f, lo, hi):
    """原胞: 原 band 本身, 纯蓝线。"""
    keep = np.concatenate([[True], np.diff(dist) > 0])
    for b in range(en.shape[0]):
        e = en[b, keep] - e_f
        m = np.isfinite(e)
        if m.sum() >= 2:
            ax.plot(dist[keep][m], e[m], color="#1f3fbf", linewidth=1.0,
                    solid_capstyle="round")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.9, zorder=3)
    ax.set_ylim(lo - e_f, hi - e_f)
    ax.set_xlim(dist[0], dist[-1])


def wrap_text(text, width=56):
    """按显示宽度折行"""
    import textwrap
    return chr(10).join(textwrap.wrap(text, width))


LBL = {
    "kx": {"zh": "k-path (2π/Å)", "en": r"k-path  (2$\pi$/Å)"},
    "ey": {"zh": "E − E$_F$  (eV)", "en": r"$E-E_F$  (eV)"},
    "sw": {"zh": "spectral weight (归一化, 按画窗)", "en": "spectral weight (normalized to the plotted window)"},
    "low": {"zh": "Low", "en": "Low"},
    "high": {"zh": "High", "en": "High"},
    "pan_a": {"zh": "原胞 band (直接算)", "en": "primitive band (direct calc.)"},
    "bands": {"zh": "条带", "en": "bands"},
}


def add_hbar(fig, rect, lang="zh"):
    """底部横向色条(左 Low 右 High)"""
    cax = fig.add_axes(rect)
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1),
                                      cmap=plt.get_cmap("jet"))
    sm.set_array(np.linspace(0, 1, 256))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_visible(False)
    cax.text(-0.012, 0.5, LBL["low"][lang], transform=cax.transAxes, ha="right",
             va="center", fontsize=10)
    cax.text(1.012, 0.5, LBL["high"][lang], transform=cax.transAxes, ha="left",
             va="center", fontsize=10)
    cax.set_xlabel(LBL["sw"][lang], fontsize=8, labelpad=2)
    return cax


def kpath_marks(ax, labels, dark):
    """k 路径的竖线 + 高对称点标签。

    ⚠ 标签画在**坐标区外面**(白底), 所以文字一律黑色; 竖线在热图(深底)上
    用白线、在原胞(白底)面板上用灰线。
    ⚠ 位于坐标区两端的标签改成左/右对齐, 否则会被裁掉。
    """
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    for name, d in labels:
        ax.axvline(d, color="w" if dark else "0.6", linewidth=0.5,
                   alpha=0.6, zorder=2)
        if d - x0 < 0.06 * span:
            ha = "left"
        elif x1 - d < 0.06 * span:
            ha = "right"
        else:
            ha = "center"
        ax.text(d, -0.105, name, transform=ax.get_xaxis_transform(),
                ha=ha, va="top", fontsize=9, color="k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=float, required=True)
    ap.add_argument("--prim-dat", required=True)
    ap.add_argument("--unfold", required=True, help="逗号分隔的 .dat")
    ap.add_argument("--titles", required=True)
    ap.add_argument("--labels", default="", help="名:距离,逗号分隔")
    ap.add_argument("--out", required=True)
    ap.add_argument("--e-min", type=float, default=-4.2)
    ap.add_argument("--e-max", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--suptitle", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--split-dir", default="", help="每个面板另存一张单独的图")
    ap.add_argument("--split-prefix", default="panel")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="图内固定文字的语言")
    ap.add_argument("--split-suptitle", default="",
                    help="小图用的标题(默认沿用 --suptitle)")
    args = ap.parse_args()

    labels = []
    for item in args.labels.split(","):
        if item.strip():
            name, pos = item.split(":")
            labels.append((name.strip(), float(pos)))

    dp, ep, wp = read_dat(args.prim_dat)
    figs = [(t.strip(), read_dat(p.strip()))
            for t, p in zip(args.titles.split(","), args.unfold.split(","))]

    n = len(figs) + 1
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.6), squeeze=False)
    axes = axes[0]

    # ---- (a) 原胞: 蓝线 ----
    lines_panel(axes[0], dp, ep, args.ef, args.e_min, args.e_max)
    axes[0].set_title(f"(a) {LBL['pan_a'][args.lang]}\n"
                      f"{ep.shape[0]} {LBL['bands'][args.lang]}", fontsize=11)
    if args.split_dir:
        os.makedirs(args.split_dir, exist_ok=True)
        f1, a1 = plt.subplots(figsize=(6.4, 6.6))
        lines_panel(a1, dp, ep, args.ef, args.e_min, args.e_max)
        a1.set_title(f"(a) {LBL['pan_a'][args.lang]}\n"
                     f"{ep.shape[0]} {LBL['bands'][args.lang]}", fontsize=11)
        a1.set_xlabel("k-path (2π/Å)", labelpad=32)
        a1.set_ylabel("E − E$_F$  (eV)")
        a1.tick_params(labelsize=8)
        kpath_marks(a1, labels, dark=False)
        f1.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.20)
        st = args.split_suptitle or args.suptitle
        if st:
            f1.suptitle(st, fontsize=12, y=0.975)
        if args.note:
            f1.text(0.5, 0.945, wrap_text(args.note, 58), ha="center",
                    va="top", fontsize=8.5, color="0.35")
        out1 = os.path.join(args.split_dir, f"{args.split_prefix}_a.png")
        f1.savefig(out1, dpi=160)
        plt.close(f1)
        print("  wrote", out1)

    # ---- (b)(c)... 超胞反折叠: 热图 ----
    for i, (ax, (title, (d, e, w))) in enumerate(zip(axes[1:], figs)):
        heatmap_panel(ax, d, e, w, args.ef, args.e_min, args.e_max,
                      args.sigma)
        ax.set_title(f"({chr(ord('b') + i)}) {title}\n"
                     f"{e.shape[0]} {LBL['bands'][args.lang]}", fontsize=11)
        if args.split_dir:
            f1, a1 = plt.subplots(figsize=(6.4, 6.6))
            heatmap_panel(a1, d, e, w, args.ef, args.e_min, args.e_max,
                          args.sigma)
            a1.set_title(f"({chr(ord('b') + i)}) {title}\n"
                         f"{e.shape[0]} {LBL['bands'][args.lang]}", fontsize=11)
            a1.set_xlabel(LBL["kx"][args.lang], labelpad=32)
            a1.set_ylabel(LBL["ey"][args.lang])
            a1.tick_params(labelsize=8)
            kpath_marks(a1, labels, dark=True)
            add_hbar(f1, [0.24, 0.085, 0.46, 0.014], args.lang)
            f1.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.24)
            st = args.split_suptitle or args.suptitle
            if st:
                f1.suptitle(st, fontsize=12, y=0.975)
            if args.note:
                f1.text(0.5, 0.945, wrap_text(args.note, 58), ha="center",
                        va="top", fontsize=8.5, color="0.35")
            out1 = os.path.join(args.split_dir,
                                f"{args.split_prefix}_{chr(ord('b') + i)}.png")
            f1.savefig(out1, dpi=160)
            plt.close(f1)
            print("  wrote", out1)

    for i, ax in enumerate(axes):
        ax.set_xlabel("k-path (2π/Å)", labelpad=32)
        ax.set_ylabel("E − E$_F$  (eV)")
        ax.tick_params(labelsize=8)
        kpath_marks(ax, labels, dark=(i > 0))
    # 底部横向色条(左 Low 右 High)
    add_hbar(fig, [0.40, 0.055, 0.22, 0.016], args.lang)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.985)
    if args.note:
        fig.text(0.5, 0.008, args.note, ha="center", fontsize=9,
                 color="0.35")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.25,
                        wspace=0.24)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
