"""论文图 1: 展开能带(PRB 版式)。

口径(与 ``scripts/figures/README.md`` 一致):
  * y 轴 = E − E_F, E_F = 0;
  * 超胞面板 = ``full`` 模式 + **按画窗归一化的线性色标**;
  * 原胞面板 = **原 band 本身、纯蓝线**(标准答案, 用最朴素的线图画)。

PRB 版式(与 ``prb_style.py`` 一致):
  * 画布 7.0 in 宽 = PRB 全栏 \\linewidth ⇒ 图里写的字号就是印刷磅值;
  * 图内**不写总标题、不写长注释**(caption 承担);
  * 底部一条横向色标(左 Low 右 High), 标签在色标**下面**, 不压任何轴标签。

用法:
  python paper_fig_bands.py --ef -1.501119 \
      --prim-dat tas2_prim.dat --unfold tas2_full_sc.dat,tas2_full_scsqrt3.dat \
      --labels "0:0,1.078:1,1.700:2,2.945:3" \
      --titles "primitive cell (direct calculation),3×1×1 supercell,√3×√3 supercell" \
      --out fig1_TaS2_bands.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prb_style as S
from plot_band import read_dat, heatmap_panel, lines_panel

# 展开能带的色标: turbo。低端是**深蓝紫**, 而 A(k,E) 在带外几乎为 0 ⇒
# 底色就是深蓝(与旧 paper 的图同色系), 带被点亮成暖色。
CMAP = "turbo"
# "到原胞带的能量偏差"的色标: 偏差小 = 亮(magma_r 的高端), 偏差大 = 暗
CMAP_OFFSET = "magma_r"
CMAP_ERR = ["#d43b1b", "#1f6fb4"]

# 图内固定文字(en = 原样, zh = 中文)。面板标题 (a)(b)(c) 由 --titles 传入,
# 放在这里的是**图脚本自己写**的那些字。
LBL = {
    "path_dist": {"en": "path distance", "zh": "沿路径的距离"},
    "err":       {"en": "error (meV)", "zh": "误差（meV）"},
    "rebuild":   {"en": "(d) rebuild error", "zh": "(d) 重建误差"},
    "low":       {"en": "Low", "zh": "低"},
    "high":      {"en": "High", "zh": "高"},
}


def kpath_marks(ax, labels, dark: bool):
    """k 路径竖线 + 高对称点标签。

    标签画在坐标区**外面**(白底)⇒ 文字一律黑色; 热图(深底)上竖线用白线。
    两端的标签改成左/右对齐, 否则会被裁掉。
    """
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    for name, d in labels:
        ax.axvline(d, color="w" if dark else "0.75", linewidth=0.6,
                   alpha=0.55, zorder=2)
        if d - x0 < 0.06 * span:
            ha = "left"
        elif x1 - d < 0.06 * span:
            ha = "right"
        else:
            ha = "center"
        # 高对称点标记**贴着下边框**(不要再留一大截空白)
        ax.text(d, -0.022, name, transform=ax.get_xaxis_transform(),
                ha=ha, va="top", fontsize=S.FS_TICK, color="k")


def add_hbar(fig, rect, label, norm=None, lang="en"):
    """底部横向色条: 左 Low 右 High, 标签在色条下方(不压轴标签)。"""
    cax = fig.add_axes(rect)
    if norm is None:
        norm = matplotlib.colors.Normalize(0, 1)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP))
    sm.set_array(np.linspace(0, 1, 256))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_visible(False)
    cax.text(-0.010, 0.5, LBL["low"][lang], transform=cax.transAxes, ha="right",
             va="center", fontsize=S.FS_ANNOT)
    cax.text(1.010, 0.5, LBL["high"][lang], transform=cax.transAxes, ha="left",
             va="center", fontsize=S.FS_ANNOT)
    cax.set_xlabel(label, fontsize=S.FS_ANNOT, labelpad=3)
    return cax


def offset_colors(en, ep):
    """每个展开态到**最近**原胞带的能量偏差 (eV), 同形于 en。

    为什么用它: 平面波严格权重下每个展开态完整属于唯一一个候选动量
    (完备性恒等式 Σw=1)⇒ **权重是二值的**(99% 的点 = 1), 任何色标都拉不开
    ⇒ 整条带一个颜色。真正有层次的是**能量**: 原胞的带对应偏差 ~0, 副本
    偏离 0.1~1 eV, 且不同副本的偏离各不相同 ⇒ 颜色能把"原胞的带"与
    "各个副本"分开。
    """
    en = np.asarray(en, dtype=float)
    ep = np.asarray(ep, dtype=float)
    out = np.full(en.shape, np.nan)
    for i in range(en.shape[1]):          # 沿路径逐点
        eu = en[:, i]
        ok = np.isfinite(eu)
        if not ok.any():
            continue
        epi = ep[:, i]
        epi = epi[np.isfinite(epi)]
        if epi.size == 0:
            continue
        out[ok, i] = np.abs(eu[ok][:, None] - epi[None, :]).min(axis=1)
    return out


def rebuild_error(en, ep):
    """每个路径点上, 原胞的带与展开态的**最佳匹配误差的中位数** (eV)。

    对每条原胞带找最接近的展开态, 再取这些误差的中位数 ⇒
    "这一条路径上, 原胞的带被重建得有多准"(典型值)。
    平面波严格权重下这个量是 meV 量级 ⇒ 它是"展开重建得准不准"的
    **定量**证据(颜色给不了: 权重是二值的)。
    """
    en = np.asarray(en, dtype=float)
    ep = np.asarray(ep, dtype=float)
    out = np.full(en.shape[1], np.nan)
    for i in range(en.shape[1]):
        eu = en[:, i]
        eu = eu[np.isfinite(eu)]
        epi = ep[:, i]
        epi = epi[np.isfinite(epi)]
        if eu.size == 0 or epi.size == 0:
            continue
        d = np.abs(eu[:, None] - epi[None, :]).min(axis=0)
        out[i] = float(np.median(d))
    return out


def build(ef, prim_dat, unfold_dats, titles, labels, out, png=None,
          e_min=-4.2, e_max=1.0, sigma=0.05, height=3.15,
          color_scale="linear", overlay_prim=False, color_by="weight",
          lang="en"):
    S.apply_prb_style(lang=lang)
    dp, ep, wp = read_dat(prim_dat)
    # titles[0] 属于原胞面板, 后面的依次属于各超胞面板
    figs = [(t.strip(), read_dat(p.strip()))
            for t, p in zip(titles[1:], unfold_dats)]

    n = len(figs) + 1
    fig, axes = plt.subplots(1, n + 1, figsize=(S.FULL_WIDTH, height),
                             squeeze=False)
    axes = axes[0]

    # 色标: linear(默认)或 log(权重近二值时用, 否则复制带全压在底部看不见)
    if color_scale == "log":
        norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.0)
    elif color_scale == "symlog":
        # 权重近乎二值(93% 的点=1)时: 线性色标把复制带全压在底部,
        # 纯对数又把背景(A<1e-3)变成白色。symlog 两头都不失真。
        norm = matplotlib.colors.SymLogNorm(linthresh=1e-2, linscale=0.5,
                                           vmin=0.0, vmax=1.0)
    else:
        norm = None

    lines_panel(axes[0], dp, ep, ef, e_min, e_max)
    axes[0].set_title(f"(a) {titles[0]}", fontsize=S.FS_TITLE)
    kpath_marks(axes[0], labels, dark=False)
    # 原胞是画线图 ⇒ 边框**四周都有**(house style 默认只留左/下)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(True)

    cmap = CMAP
    for i, (ax, (title, (d, e, w))) in enumerate(zip(axes[1:], figs)):
        # 网格本身是数据 ⇒ 存 PDF 时栅格化(否则一个 pcolormesh 就是几 MB),
        # 坐标轴/文字/色标仍然是矢量。
        wcol = None
        pnorm = norm
        if color_by == "offset":
            wcol = offset_colors(e, ep)
            finite = wcol[np.isfinite(wcol)]
            vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
            pnorm = matplotlib.colors.Normalize(0.0, max(vmax, 1e-3))
            cmap = CMAP_OFFSET
        heatmap_panel(ax, d, e, w, ef, e_min, e_max, sigma, cmap=cmap,
                      rasterized=True, norm=pnorm, wcol=wcol)
        if overlay_prim:
            # 把**原胞**的带叠在展开图上(白线): 读者看到每条原胞的带都被
            # 一条亮带覆盖 ⇒ "原胞的带被重建出来了"; 多出来的是同一组带的
            # 平移副本。
            keep = np.concatenate([[True], np.diff(dp) > 0])
            for b in range(ep.shape[0]):
                eb = ep[b, keep] - ef
                m = np.isfinite(eb)
                if m.sum() >= 2:
                    # 原胞的带画成**白线**(旧 paper 同款): turbo 的高端是
                    # 深红, 白线在亮带上分得清。
                    ax.plot(dp[keep][m], eb[m], color="w", linewidth=0.9,
                            alpha=0.95, zorder=5, solid_capstyle="round")
        ax.set_title(f"({S.letter(i + 1)}) {title}", fontsize=S.FS_TITLE)
        kpath_marks(ax, labels, dark=True)

    for i, ax in enumerate(axes):
        ax.set_ylabel(r"$E-E_F$  (eV)")
        ax.tick_params(labelsize=S.FS_TICK)
        # k 路径的横坐标**只要高对称点标记**: 不要数字刻度, 也不要 "k-path" 轴标题
        ax.set_xticks([])

    # ---- 第 4 个面板: 重建误差(定量) ----
    ax = axes[3]
    for i, (title, (d, e, w)) in enumerate(figs):
        err = rebuild_error(e, ep)
        ax.plot(d, err * 1e3, color=CMAP_ERR[i % len(CMAP_ERR)], lw=1.2,
                label=f"({S.letter(i + 1)})")
    ax.set_yscale("log")
    ax.set_xlabel(LBL["path_dist"][lang])
    ax.set_ylabel(LBL["err"][lang])
    ax.set_title(LBL["rebuild"][lang], fontsize=S.FS_TITLE)
    ax.tick_params(labelsize=S.FS_TICK)
    ax.legend(fontsize=S.FS_LEG, loc="upper right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    # 中文版的 (d) 面板 y 轴标签比英文版宽("误差（meV）"), 英文版就已经压到
    # (c) 面板的热图上 ⇒ 中文版把列间距拉开一点, 让它落在空白里。
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.13,
                        wspace=0.58)
    S.save(fig, out, png_preview=png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=float, required=True)
    ap.add_argument("--prim-dat", required=True)
    ap.add_argument("--unfold", required=True, help="逗号分隔的 .dat")
    ap.add_argument("--titles", required=True, help="逗号分隔的三个面板标题")
    ap.add_argument("--labels", default="", help="名:距离,逗号分隔")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    ap.add_argument("--color-scale", choices=("linear", "log", "symlog"),
                    default="linear",
                    help="色标: linear(默认)或 log。平面波路线的权重"
                         "近乎二值(93%% 的点=1), 线性色标会把复制带"
                         "全压在底部 => 用 log")
    ap.add_argument("--e-min", type=float, default=-4.2)
    ap.add_argument("--e-max", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--height", type=float, default=3.15)
    ap.add_argument("--overlay-prim", action="store_true",
                    help="在超胞面板上叠原胞的带(白线)")
    ap.add_argument("--color-by", choices=("weight", "offset"),
                    default="weight",
                    help="颜色编码: weight(谱权重, 平面波路线下近乎二值) "
                         "或 offset(到最近原胞带的能量偏差, 有层次)")
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="图内固定文字的语言(面板标题由 --titles 传入)")
    args = ap.parse_args()

    labels = []
    for item in args.labels.split(","):
        if item.strip():
            name, pos = item.split(":")
            labels.append((name.strip(), float(pos)))

    build(args.ef, args.prim_dat, [p.strip() for p in args.unfold.split(",")],
          args.titles.split(","), labels, args.out,
          png=args.png or None, e_min=args.e_min, e_max=args.e_max,
          sigma=args.sigma, height=args.height,
          color_scale=args.color_scale, overlay_prim=args.overlay_prim,
          color_by=args.color_by, lang=args.lang)


if __name__ == "__main__":
    main()
