"""PRB 论文图的统一版式(house style)。

三条硬规矩(踩过的坑都记在这里):

1. **画布宽度 = 最终印刷宽度**。PRB 双栏全栏 ``\\linewidth`` ≈ 7.0 in。
   画布比最终尺寸大 ⇒ LaTeX 缩放 ⇒ 图里写的字号跟着变小(曾经把 9.5 pt 的
   标题画在 11.4 in 宽的画布上, 缩到 3.5 in 就只剩 2.9 pt, 印出来根本读不清)。
   所以本模块按 **7.0 in** 出图, 这里的字号就是印刷时的真实磅值。
2. **图里不写总标题、不写长注释** —— 那是 caption 的活。图内只保留:
   面板标号 (a)(b)(c)、轴标题与单位、刻度、图例、色标。
   ⚠ 文字里**不要出现 markdown 星号**(``*强调*`` / ``**强调**``): matplotlib
   不解析 markdown, 星号会原样印在图上(已犯过)。
3. **任何写在坐标区外的文字都不能和轴标签/刻度重叠**。办法: 文字要么放
   坐标区**内**并配半透明白底框, 要么用 ``fig.text`` 放在画布留白里, 绝不用
   ``ax.text(transform=ax.transAxes, y=-0.16)`` 这种"贴在轴下面"的写法 ——
   它必然和 x 轴标签打架(已犯过: fig3 的 dispersion 数字压住了 $k_1$)。

字号层级(磅)与线宽见下面的常量。
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 字号层级
FS_TITLE = 9.0      # 面板标题
FS_AXLAB = 9.0      # 轴标题
FS_TICK = 8.0       # 刻度数字
FS_LEG = 8.0        # 图例
FS_PANEL = 9.5      # 面板标号 (a)(b)(c)
FS_ANNOT = 8.0      # 图内小注(带白底框的那种)

LW_AX = 1.1        # 坐标轴
LW_DATA = 1.4      # 数据线
LW_THIN = 0.8      # 参考线/网格

# PRB 全栏宽(in)
FULL_WIDTH = 7.0

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# 房子风格的语义色(blue=主角/直接算, green=改进, red=对照/误判, neutral=辅助)
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "gray_mid": "#767676",
    "gray_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


def apply_prb_style(fontsize: float = FS_TICK) -> None:
    """把 rcParams 设成 PRB 论文图的口径。"""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": fontsize,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_AXLAB,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.linewidth": LW_AX,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "legend.fontsize": FS_LEG,
        "legend.handlelength": 1.6,
        "lines.linewidth": LW_DATA,
        "lines.solid_capstyle": "round",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        # 矢量导出: Type42 嵌入(TrueType), 编辑器里文本仍可选可编
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_label(ax, letter: str, *, dx: float = 0.0, dy: float = 1.015,
                color: str = "k", inside: bool = False) -> None:
    """在坐标区**左上角**写面板标号 (a)。

    inside=False(默认): 写在坐标区外侧上沿左端(白底), 适合白底坐标区。
    inside=True: 写在坐标区内侧左上角(适合深色热图/深色费米面)。
    """
    xy = (0.0 if not inside else 0.02, 1.0 if not inside else 0.97)
    va = "bottom" if not inside else "top"
    ax.text(dx + xy[0], dy if not inside else xy[1], f"({letter})",
            transform=ax.transAxes, ha="left", va=va,
            fontsize=FS_PANEL, color=color, fontweight="bold")


def boxed_note(ax, text: str, loc: str = "upper right", *, pad: float = 0.35,
               size: float = FS_ANNOT) -> None:
    """坐标区内的带白底框小注(不靠"贴在轴下面"来避免压轴标签)。"""
    xy = {"upper right": (0.97, 0.96), "upper left": (0.03, 0.96),
          "lower right": (0.97, 0.04), "lower left": (0.03, 0.04)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color="0.15",
            bbox=dict(boxstyle=f"round,pad={pad}", fc="white", ec="0.75",
                      alpha=0.9, lw=0.6))


def strip_decorations(ax) -> None:
    """热图/图像面板: 四条脊全去掉, 只留刻度。"""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=2.5, width=LW_AX)


def save(fig, path: str, *, png_preview: str | None = None,
         dpi: int = 300, pad_inches: float = 0.02) -> None:
    """存矢量图(论文用 PDF); 可选再存一张 PNG 供目视核查。"""
    path = os.fspath(path)
    if path.lower().endswith(".pdf"):
        fig.savefig(path, format="pdf", bbox_inches="tight",
                    pad_inches=pad_inches)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    print("wrote", path)
    if png_preview:
        fig.savefig(png_preview, dpi=160, bbox_inches="tight",
                    pad_inches=pad_inches)
        print("wrote", png_preview)


def letter(i: int) -> str:
    """0 -> 'a'。"""
    return chr(ord("a") + i)
