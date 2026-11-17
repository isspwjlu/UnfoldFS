"""费米面文件输出: FS.cube / FS.bxsf / band.N / A(k,E).npz(共 4 种)。

**格式不凭空设计**: 每一种格式的字段顺序、单位、约定都在 ``docs/formats.md``
里逐条记录了出处(XCrySDen 自带样例、vaspkit 262 的输出、FermiSurfer 2.2.1 的
读取源码), 本模块严格按那份规范实现。改动本模块前先读那份文档。

统一约定(所有文件):
  * k 用笛卡尔坐标, 单位 **2π/Å**(与 ``geometry.reciprocal`` 一致);
  * E 用 **eV**;
  * 网格 = 原胞的均匀网格, 点数 == 满网格点数 ``spec.n_points``
    (**不是**超胞 k 点数 ``n_k``);
  * ``valid == False`` 的补位槽(能量 +inf、权重 0)绝不进入任何输出。

补位槽在 BXSF / FermiSurfer 逐带文件里的处理: 该槽位对应的带在该 k 点上
根本不存在, 必须填一个数。填**该点最高有效带的能量**(该点一条有效带都没有
时填 0) —— 规则见 docs/formats.md §5。补位带与最高有效带重合, 所以最高带
过 E_F 的地方可能出现**重合的**等值面。

FermiSurfer 的 ``band.N`` 写的是**相对 E_F** 的能量(E − E_F): FermiSurfer
默认 ``EF = 0``(``fermisurfer.cpp:219``), 它的官方 .frmsf 样例也全部以 0
为中心 ⇒ 写绝对能量会画不出费米面(静默失败)。
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .expand import ExpandedData
from .vaspio import BandData

__all__ = [
    "BOHR_IN_ANGSTROM",
    "band_edges", "is_metal", "select_fermi_energy", "make_e_grid",
    "write_cube", "write_bxsf", "write_fermisurfer_bands",
    "write_spectral_npz",
    "read_cube", "read_bxsf", "read_fermisurfer_band",
]

# ---------------------------------------------------------------- 网格重建

def _grid_indices(expanded: ExpandedData,
                  b_prim: np.ndarray) -> tuple[np.ndarray, tuple]:
    """满网格点 -> (i1, i2, i3) 网格下标, 以及网格维度。

    网格下标由**分数坐标**反推(k_cart @ inv(b_prim)), 而不是靠点序:
    点序只是实现细节, 靠它会在点序被打乱时静默写错位置。
    反推后必须构成双射(每个下标恰好一个点), 否则报错。
    """
    spec = expanded.spec
    dims = tuple(int(n) for n in spec.p)
    b = np.asarray(b_prim, dtype=float)
    frac = np.asarray(expanded.k_cart, dtype=float) @ np.linalg.inv(b)
    off = np.array([0.0 if spec.grid_type == "gamma" else 0.5 / n
                    for n in dims])
    idx = np.empty((frac.shape[0], 3), dtype=int)
    for a in range(3):
        x = (frac[:, a] - spec.shift[a] - off[a]) * dims[a]
        idx[:, a] = np.mod(np.rint(x), dims[a]).astype(int)
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    if np.unique(flat).size != flat.size:
        raise ValueError(
            "k 点 -> 网格下标的映射不是双射(有下标被多个点占用) —— "
            "展开结果的 k_cart 与网格规格不自洽")
    return idx, dims


def _field_on_grid(expanded: ExpandedData, b_prim: np.ndarray,
                   values: np.ndarray) -> tuple[np.ndarray, tuple]:
    """把逐点的值摊到 (i1, i2, i3) 的三维数组上。"""
    idx, dims = _grid_indices(expanded, b_prim)
    field = np.zeros(dims, dtype=float)
    field[idx[:, 0], idx[:, 1], idx[:, 2]] = np.asarray(values, dtype=float)
    return field, dims


def _filled_band_energies(expanded: ExpandedData) -> np.ndarray:
    """逐带能量, 补位槽填**该点最高有效带的能量**(该点无有效带则填 0)。

    返回 (n_points, n_band_eff); 全部为有限值(没有有效带的点填 0)。

    实现上取"下标更大且有效的带"优先, 但 ``unfold`` 产生的补位槽恒为**连续
    尾部** ⇒ 不存在这样的带 ⇒ 实际恒等于"最高有效带"(见 docs/formats.md §5)。
    """
    e = np.asarray(expanded.band_e, dtype=float)
    valid = np.asarray(expanded.valid, dtype=bool)
    n_pts, n_band = e.shape
    if n_band == 0:
        return np.zeros((n_pts, 0))
    # 下一个有效带的下标(严格大于当前下标)
    nxt = np.full((n_pts, n_band), -1, dtype=int)
    run = np.full(n_pts, -1, dtype=int)
    for m in range(n_band - 1, -1, -1):
        nxt[:, m] = run
        run = np.where(valid[:, m], m, run)
    # 该点最高有效带的下标
    last = np.full(n_pts, -1, dtype=int)
    for m in range(n_band):
        last = np.where(valid[:, m], m, last)
    src = np.where(nxt >= 0, nxt, np.maximum(last, 0)[:, None])
    out = np.where(valid, e, np.take_along_axis(e, src, axis=1))
    dead = last < 0
    if dead.any():
        out[dead, :] = 0.0
    return np.where(np.isfinite(out), out, 0.0)


# ---------------------------------------------------------------- E_F 选取

def band_edges(band: BandData, threshold: float = 0.5) -> tuple:
    """从 EIGENVAL 的占据数求 (VBM, CBM)(eV)。

    占据判据: 占据数 > threshold(默认 0.5)。VBM = 占据态的最高能量,
    CBM = 未占据态的最低能量(没有未占据态时为 +inf)。
    """
    e = np.asarray(band.energies, dtype=float)
    occ = np.asarray(band.occupations, dtype=float)
    if e.shape != occ.shape:
        raise ValueError("EIGENVAL 的能量与占据数形状不一致")
    occupied = occ > threshold
    if not occupied.any():
        raise ValueError(
            "EIGENVAL 里没有任何占据态(占据数全为 0) —— 无法确定 VBM/CBM")
    vbm = float(e[occupied].max())
    free = ~occupied
    cbm = float(e[free].min()) if free.any() else float("inf")
    return vbm, cbm


def is_metal(band: BandData, gap_tol: float = 0.05) -> bool:
    """金属判据: **带隙 <= gap_tol**。

    ⚠ "存在分数占据"**不是**金属判据: 任何用了展宽的**有带隙**体系, 带边的
    占据数都会变成 0.98 这类小数(高斯展宽与四面体法都一样) —— 用它判金属
    会把半导体误判成金属(实测: MoS2 的 VBM=-2.077449 / CBM=-0.465849 /
    gap=1.61 eV 因带边占据 0.9866 被判成金属, auto 于是取了带隙中央的
    DOSCAR E_F ⇒ 费米面是空的)。

    E_F 的最终选择见 ``select_fermi_energy`` 的 ``auto`` 分支(它还看
    DOSCAR 的 E_F 落在带隙的哪一侧)。
    """
    vbm, cbm = band_edges(band)
    return bool(cbm - vbm <= gap_tol)


def select_fermi_energy(dos_ef: float, band: BandData, mode: str = "auto",
                         offset: float = 0.0, gap_tol: float = 0.05) -> float:
    """按 ``--ef auto|vbm|<数值>`` 的语义选费米能(eV)。

    * ``auto``    按**带隙与 DOSCAR 的 E_F 的相对位置**决定(见下)
    * ``vbm``     总是 VBM + offset
    * ``doscar``  总是 DOSCAR 的 E_F
    * 数值/数值字符串 -> 直接当作 E_F(绝对值, 不加 offset)

    ``auto`` 的判据(不依赖展宽宽度, 因此不需要额外物理参数)::

        gap = CBM - VBM
        gap <= gap_tol                 -> DOSCAR 的 E_F   (真金属/半金属)
        VBM <= dos_ef <= CBM           -> VBM + offset    (E_F 落在带隙里)
        否则                             -> DOSCAR 的 E_F   (E_F 在带隙外)

    ⚠ "带边有分数占据" **不能**用来判金属: 用了展宽的有带隙体系, 带边占据数
    都会是 0.98 这类小数(实测: MoS2 的 gap=1.61 eV 被误判成金属, auto 于是
    取了带隙中央的 DOSCAR E_F ⇒ 费米面是空的)。

    **只有 ``auto`` 与 ``vbm`` 需要 EIGENVAL**: 数值与 ``doscar`` 模式完全不
    碰 EIGENVAL —— 用户明明白白给了 E_F, 不该被一个用不上的 EIGENVAL 挡住。
    ``auto`` 模式若 EIGENVAL 里没有占据态(VBM 不可得), 退化到 DOSCAR 的 E_F
    并给出明确 warning(不静默)。
    """
    if isinstance(mode, str):
        key = mode.strip().lower()
        if key in ("doscar", "dos"):
            return float(dos_ef)
        if key == "auto":
            try:
                vbm, cbm = band_edges(band)
            except ValueError as exc:
                warnings.warn(
                    f"EIGENVAL 里读不到占据态({exc}); auto 模式退化到 DOSCAR "
                    f"的费米能 {float(dos_ef):.6f} eV。")
                return float(dos_ef)
            gap = cbm - vbm
            if gap <= gap_tol:                    # 真金属/半金属
                return float(dos_ef)
            if vbm <= float(dos_ef) <= cbm:       # E_F 落在带隙里 ⇒ 半导体
                return float(vbm) + float(offset)
            return float(dos_ef)                  # E_F 在带隙外 ⇒ 金属
        if key == "vbm":
            vbm, _cbm = band_edges(band)
            return float(vbm) + float(offset)
        try:
            return float(mode)
        except ValueError as exc:
            raise ValueError(
                f"ef 模式未知: {mode!r} —— 只接受 auto / vbm / doscar / 数值"
            ) from exc
    if isinstance(mode, (int, float, np.floating, np.integer)):
        return float(mode)
    raise ValueError(f"ef 模式未知: {mode!r}")


def make_e_grid(e_f: float, half_width: float, n: int) -> np.ndarray:
    """以 E_F 为中心、**恰好含 E_F** 的能量网格(eV)。

    n 取奇数, E_F 落在正中一个格点上 —— 这样谱函数在 E_F 处的值不必再做
    插值, 也不受网格步长影响。

    **偶数 n 会被上调一格**(n -> n+1, 使 E_F 正好落在格点上); n < 3 直接报错。
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"能量网格点数 n 至少为 3, 给出 {n}")
    if n % 2 == 0:
        n += 1
    de = 2.0 * float(half_width) / (n - 1)
    return float(e_f) + (np.arange(n) - n // 2) * de


# ---------------------------------------------------------------- 公共片段

# cube 格式的长度单位: 原子单位(Bohr)。Quantum ESPRESSO 的 cube.f90 文件头
# 注释明写 "ALL COORDINATES ARE GIVEN IN ATOMIC UNITS"; ASE 的 read_cube 也
# 按 Bohr 解析(``cell[i] = n * Bohr * vec[i]``)。故 cube 里的矢量按 Bohr 写。
BOHR_IN_ANGSTROM = 0.529177210903

_HEADER_COMMENT = [
    "unfoldfs: unfolded Fermi surface (spectral function A(k, E_F))",
    "units: E in eV; cell vectors in Bohr (cube format = atomic units); "
    "k = 2pi/Angstrom / 0.529177; reciprocal box = BZ",
]


def _isolevel_hint(vmax: float) -> str:
    """cube 头部里的"体数据最大值 + 建议等值面层级"。

    VESTA 的默认 isolevel = 0 —— 只有当高斯尾巴**恰好**下溢到 0.0 时才看得见
    等值面(实测: 换个 sigma 或能量窗口就可能什么都没有)。故给出"最大值的一
    半"作为建议层级: 稳健, 且落在 (0, max) 内。
    """
    if not np.isfinite(vmax) or vmax <= 0.0:
        return f"field max = {vmax:.3e}; (empty field: no isosurface)"
    return (f"field max = {vmax:.3e}; "
            f"use isosurface level ~ {0.5 * vmax:.3e}")


def _check_lines(lines: List[str], min_lines: int, path: str) -> None:
    """行数检查: 畸形/被截断的文件给可读信息, 而不是 IndexError。"""
    if len(lines) < min_lines:
        raise ValueError(
            f"{os.path.basename(path)} 被截断: 只有 {len(lines)} 行, "
            f"至少需要 {min_lines} 行")


def _fmt_vec(v: Sequence[float], prec: int = 8) -> str:
    """向量的定宽浮点格式(与样例文件的 %16.8f 同风格)。"""
    return "".join(f"{float(x):{prec + 8}.{prec}f}" for x in v)


def _fmt_vals(values: np.ndarray, per_line: int) -> List[str]:
    out = []
    flat = np.asarray(values, dtype=float).reshape(-1)
    for i in range(0, flat.size, per_line):
        out.append("".join(f"{x:13.5E}" for x in flat[i:i + per_line]))
    return out


# ---------------------------------------------------------------- FS.cube

def write_cube(path: str, expanded: ExpandedData, b_prim: np.ndarray,
               values: np.ndarray, *, comments: Optional[Sequence[str]] = None,
               per_line: int = 6) -> None:
    """写 Gaussian cube(消费者: VESTA / XCrySDen / VMD)。

    **假盒子 = 第一布里渊区**: cube 的三个矢量行写的是**每格点的增量矢量**
    ``b_i / n_i``(cube 格式的定义, 见 Quantum ESPRESSO 的 ``PP/src/cube.f90``
    文件头注释: "4-6 I5,3F12.6 #GRIDPOINTS, INCREMENT VECTOR"), 于是
    n_i 个格点铺满的盒子正好是 b1,b2,b3 张成的倒格矢盒子 = 原胞 BZ,
    画出来的等值面就是 k 空间里的费米面。体数据 = A(k, E_F)。

    **长度单位 = 原子单位(Bohr)**: cube 格式的坐标一律用原子单位
    (QE 的 cube.f90: "ALL COORDINATES ARE GIVEN IN ATOMIC UNITS"; ASE 的
    ``read_cube`` 也按 Bohr 解析)。故矢量按 Bohr 写(1/Bohr = 2π/Å ÷
    0.529177)。体数据 A(k,E_F) 无量纲, 不受影响。

    存储顺序: **第 3 个矢量最快**(与 QE 的 cube.f90 写循环
    ``DO i1 / DO i2 / WRITE (rho(i1,i2,i3),i3=1,nr3)``、以及 ASE 的
    ``read_cube`` 一致), 即数据线性下标 = i1·n2·n3 + i2·n3 + i3。

    ⚠ VASP 的 CHGCAR **不是**这个约定(它写整条晶格矢量, 且是 VASP 专有
    格式), 别拿 CHGCAR 当 cube 的参考。

    **头部给出体数据最大值与建议等值面层级**: VESTA 的默认 isolevel = 0,
    "画不画得出来"会取决于高斯尾巴是否恰好下溢到 0.0 ⇒ 头部必须写明该选
    什么层级(见 docs/formats.md §2)。
    """
    b = np.asarray(b_prim, dtype=float)
    field, dims = _field_on_grid(expanded, b, values)
    # cube 的长度单位是原子单位(Bohr); 盒子 = 增量 × 格点数 = BZ
    inc = b / (np.array(dims, dtype=float)[:, None] * BOHR_IN_ANGSTROM)
    data = field.reshape(-1)                             # 第 3 个轴最快
    lines = list(comments) if comments is not None else list(_HEADER_COMMENT)
    while len(lines) < 2:
        lines.append("")
    lines = lines[:2]
    # 第 2 行补上"最大值 + 建议层级"(头部只有两行, 单位说明与它挤在一起)
    lines[1] = lines[1] + "; " + _isolevel_hint(float(np.max(data)))
    lines.append(f"{0:5d}{_fmt_vec([0.0, 0.0, 0.0], prec=12)}")
    for a in range(3):
        # 增量矢量要多给几位: 盒子 = 增量 × 格点数, 位数不够会把 k 点拉偏
        lines.append(f"{dims[a]:5d}{_fmt_vec(inc[a], prec=12)}")
    lines.extend(_fmt_vals(data, per_line))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def read_cube(path: str) -> Dict:
    """读回 cube(往返自检用)。

    返回的 ``field`` 按 (i1, i2, i3) 排(与展开结果的点序一致);
    ``vectors`` 是文件里写的增量矢量, ``cell`` = 增量 × 格点数 = 盒子。
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    _check_lines(lines, 3, path)
    natoms = int(lines[2].split()[0])
    _check_lines(lines, 6 + natoms, path)
    origin = [float(x) for x in lines[2].split()[1:4]]
    dims = []
    vectors = []
    for a in range(3):
        f = lines[3 + a].split()
        dims.append(int(f[0]))
        vectors.append([float(x) for x in f[1:4]])
    vals = []
    for ln in lines[6 + natoms:]:
        vals.extend(float(x) for x in ln.split())
    n1, n2, n3 = dims
    if len(vals) != n1 * n2 * n3:
        raise ValueError(
            f"cube 体数据个数 {len(vals)} != {n1}×{n2}×{n3} = {n1 * n2 * n3}")
    # 文件里第 3 个轴最快 -> 直接 C 序 reshape 成 (n1, n2, n3)
    field = np.asarray(vals).reshape(n1, n2, n3)
    inc = np.asarray(vectors)                       # 文件里的增量矢量(Bohr)
    cell = inc * np.array(dims, dtype=float)[:, None] * BOHR_IN_ANGSTROM
    return {"comment": lines[:2], "natoms": natoms, "origin": origin,
            "dims": dims, "vectors": inc, "cell": cell,
            "data": np.asarray(vals), "field": field}


# ---------------------------------------------------------------- FS.bxsf

def write_bxsf(path: str, expanded: ExpandedData, b_prim: np.ndarray,
               e_f: float, *, comments: Optional[Sequence[str]] = None,
               per_line: int = 6) -> None:
    """写 BXSF(Band-XCRYSDEN-Structure-File; 消费者: XCrySDen / FermiSurfer)。

    结构(逐字对齐实测样例)::

         BEGIN_INFO
           # 注释行
         Fermi Energy:     <E_F>
         END_INFO
         BEGIN_BLOCK_BANDGRID_3D
         band_energies
         BEGIN_BANDGRID_3D
         <带数>
         <n1> <n2> <n3>
         <原点 x y z>
         <b1>
         <b2>
         <b3>
         BAND: <带号>
         <能量值 ...>
         ...
         END_BANDGRID_3D
         END_BLOCK_BANDGRID_3D

    要点:
      * 网格维度 = **唯一点数 + 1**(周期端点重复一次), 每个带存
        (n1+1)(n2+1)(n3+1) 个值; 存储顺序是**第 3 个轴最快**;
      * 能量是**绝对值**(eV), E_F 单独写在 INFO 块里 —— FermiSurfer 的
        ``read_bxsf`` 会自己减去 E_F;
      * 每个带一段, 带号从 1 起。

    **只支持 Γ 网格**: FermiSurfer 的 ``read_bxsf()`` 结尾写死
    ``shiftk[ii] = 0`` —— 一律按 Γ 网格解释(而 ``.frmsf`` 路径有 MP 处理)。
    MP 网格的数据会被整体平移半个倒格矢 ⇒ 静默的错误费米面, 故直接报错。
    """
    if expanded.spec.grid_type != "gamma":
        raise ValueError(
            "BXSF 只支持 Gamma 网格: FermiSurfer 的 .bxsf 读取路径一律按 Gamma "
            f"网格解释(shiftk 写死 0), MP 网格的数据会被整体平移半个倒格矢。"
            f"当前网格类型是 {expanded.spec.grid_type!r} —— 请改用 Gamma 网格, "
            "或改用 FS.cube / band.N(以及 A(k,E).npz)。")
    b = np.asarray(b_prim, dtype=float)
    idx, dims = _grid_indices(expanded, b)
    filled = _filled_band_energies(expanded)
    n_band = filled.shape[1]
    lines = list(comments) if comments is not None else list(_HEADER_COMMENT)
    out: List[str] = [" BEGIN_INFO"]
    for c in lines:
        out.append(f"   # {c}")
    out.append(f" Fermi Energy: {float(e_f):12.6f}")
    out.append(" END_INFO")
    out.append(" BEGIN_BLOCK_BANDGRID_3D")
    out.append(" band_energies")
    out.append(" BEGIN_BANDGRID_3D")
    out.append(f"{n_band:6d}")
    ext_dims = tuple(d + 1 for d in dims)
    out.append("".join(f"{d:6d}" for d in ext_dims))
    out.append(_fmt_vec([0.0, 0.0, 0.0]))
    for a in range(3):
        out.append(_fmt_vec(b[a]))
    for n in range(n_band):
        out.append(f" BAND: {n + 1:6d}")
        field = np.zeros(dims, dtype=float)
        field[idx[:, 0], idx[:, 1], idx[:, 2]] = filled[:, n]
        ext = np.empty(ext_dims, dtype=float)
        ext[:dims[0], :dims[1], :dims[2]] = field
        ext[dims[0], :dims[1], :dims[2]] = field[0]
        ext[:, dims[1], :dims[2]] = ext[:, 0, :dims[2]]
        ext[:, :, dims[2]] = ext[:, :, 0]
        out.extend(_fmt_vals(ext.reshape(-1), per_line))
    out.append(" END_BANDGRID_3D")
    out.append(" END_BLOCK_BANDGRID_3D")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def read_bxsf(path: str) -> Dict:
    """读回 BXSF(往返自检用)。

    ``bands[n]`` 是第 n+1 个带在 (n1+1)×(n2+1)×(n3+1) 网格上的能量,
    按文件顺序展平(第 3 个轴最快)。
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    _check_lines(lines, 8, path)
    i_ef = next(i for i, ln in enumerate(lines) if "Fermi Energy:" in ln)
    ef = float(lines[i_ef].split()[-1])
    i_grid = next(i for i, ln in enumerate(lines)
                  if ln.strip() in ("BEGIN_BANDGRID_3D", "BANDGRID_3D_BANDS"))
    _check_lines(lines, i_grid + 7, path)
    nb = int(lines[i_grid + 1].split()[0])
    dims = [int(x) for x in lines[i_grid + 2].split()]
    bvec = np.array([[float(x) for x in lines[i_grid + 4 + a].split()]
                     for a in range(3)])
    band_idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("BAND:")]
    bands = []
    for b in range(nb):
        start = band_idx[b] + 1
        stop = band_idx[b + 1] if b + 1 < len(band_idx) else len(lines)
        vals: List[float] = []
        for ln in lines[start:stop]:
            if ln.strip().startswith("END_"):
                break
            vals.extend(float(x) for x in ln.split())
        arr = np.asarray(vals)
        if arr.size != dims[0] * dims[1] * dims[2]:
            raise ValueError(
                f"bxsf 第 {b + 1} 个带的值个数 {arr.size} != "
                f"{dims[0]}×{dims[1]}×{dims[2]} = {dims[0] * dims[1] * dims[2]}")
        bands.append(arr)
    return {"ef": ef, "nb": nb, "dims": dims, "bvec": bvec, "bands": bands}


# ---------------------------------------------------------------- FermiSurfer

def write_fermisurfer_bands(dirpath: str, expanded: ExpandedData,
                             b_prim: np.ndarray, e_f: float, *,
                             per_line: int = 6) -> List[str]:
    """写 FermiSurfer 的逐带文件(``band.1`` / ``band.2`` / ...)。

    FermiSurfer 2.2.1 只认两种格式: ``.frmsf`` 与 ``.bxsf``
    (源码 ``src/read_file.cpp`` / ``src/read_bxsf``,
    ``src/fermisurfer.cpp`` 按扩展名分派)。简报里设想的"逐带 (k,E) 文本"
    在该版本里**没有读取器**, 故此处按源码写成 ``.frmsf`` 格式, 每个文件
    只含一条带 —— FermiSurfer 可直接打开。

    ``.frmsf`` 布局::

        <n1> <n2> <n3>      网格点数(不含重复的周期端点)
        <lshift>            0 = MP 网格, 1 = Gamma 居中, 2 = Gamma + 半格
        <nb>                带数(逐带文件恒为 1)
        <b1>
        <b2>
        <b3>
        <能量: n1·n2·n3 个, 第 3 个轴最快>
        <矩阵元: 3 轴 × n1·n2·n3 个(本模块写 0)>

    **能量是相对 E_F 的(E − E_F)**: FermiSurfer 从批处理文件读
    ``fermienergy`` 来定等值面, 缺省是 ``0.0``(``fermisurfer.cpp:219``);
    它自带的三个 .frmsf 样例(mgb2_vfz / pb_vf3D / srvo3_orb)能量也全部
    以 0 为中心(配套 ``frmsf.in`` 写 ``fermienergy 0.0``)⇒ 写绝对能量
    会让用户直接打开时**画不出费米面**(静默失败)。

    ⚠ ``.frmsf`` 格式**放不下注释行**(第一行就被 ``sscanf("%d%d%d")`` 读走),
    所以本函数没有 ``comments`` 参数 —— 单位与能量口径见 docs/formats.md §4
    与 ``A(k,E).npz``。
    """
    b = np.asarray(b_prim, dtype=float)
    e_f = float(e_f)
    idx, dims = _grid_indices(expanded, b)
    filled = _filled_band_energies(expanded)
    n_band = filled.shape[1]
    lshift = 1 if expanded.spec.grid_type == "gamma" else 2
    os.makedirs(dirpath, exist_ok=True)
    paths: List[str] = []
    for n in range(n_band):
        field = np.zeros(dims, dtype=float)
        field[idx[:, 0], idx[:, 1], idx[:, 2]] = filled[:, n]
        energies = field.reshape(-1) - e_f    # 第 3 个轴最快; 相对 E_F
        n_grid = int(np.prod(dims))
        lines = [f"{dims[0]:12d}{dims[1]:12d}{dims[2]:12d}",
                 f"{lshift:12d}",
                 f"{1:12d}"]
        for a in range(3):
            lines.append("".join(f"{float(x):16.8E}" for x in b[a]))
        lines.extend(_fmt_vals(energies, per_line))
        # 矩阵元(3 个轴): FermiSurfer 用它做费米速度着色; 本模块没有速度,
        # 写 0(确定性的, 不给 FermiSurfer 留未初始化内存)
        lines.extend(_fmt_vals(np.zeros(n_grid), per_line))
        for _ in range(2):
            lines.extend(_fmt_vals(np.zeros(n_grid), per_line))
        p = os.path.join(dirpath, f"band.{n + 1}")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        paths.append(p)
    return paths


def read_fermisurfer_band(path: str) -> Dict:
    """读回 FermiSurfer 的 ``.frmsf`` 逐带文件(往返自检用)。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    _check_lines(lines, 6, path)
    dims = [int(x) for x in lines[0].split()]
    lshift = int(lines[1].split()[0])
    nb = int(lines[2].split()[0])
    bvec = np.array([[float(x) for x in lines[3 + a].split()] for a in range(3)])
    n_grid = int(np.prod(dims))
    vals: List[float] = []
    for ln in lines[6:]:
        vals.extend(float(x) for x in ln.split())
    if len(vals) < nb * n_grid:
        raise ValueError(
            f"{os.path.basename(path)} 被截断: 数据个数 {len(vals)} < "
            f"{nb}×{n_grid} = {nb * n_grid}")
    energies = np.asarray(vals[:nb * n_grid])
    return {"dims": dims, "lshift": lshift, "nb": nb, "bvec": bvec,
            "energies": energies}


# ---------------------------------------------------------------- npz

def write_spectral_npz(path: str, expanded: ExpandedData, b_prim: np.ndarray,
                       e_f: float, *, vbm: float, cbm: float, e_grid: np.ndarray,
                       sigma: float) -> None:
    """写完整谱函数 + 元信息(自用/二次分析)。

    存: A(k,E) 体数据、E 网格、k 坐标(笛卡尔 + 分数)、逐带能量/权重/有效位
    掩码, 以及 E_F / VBM / CBM / sigma / 单位说明。

    分数坐标 ``k_frac`` 由 ``b_prim`` 反推(k_cart · b_prim⁻¹)—— 二次分析不必
    再去找倒格矢, 也不必自己算周期距离。
    """
    from .expand import build_spectral_function
    e_grid = np.atleast_1d(np.asarray(e_grid, dtype=float))
    a = build_spectral_function(expanded, e_grid, float(sigma))
    k_cart = np.asarray(expanded.k_cart, dtype=float)
    k_frac = k_cart @ np.linalg.inv(np.asarray(b_prim, dtype=float))
    np.savez_compressed(
        path,
        A=a,
        e_grid=e_grid,
        k_cart=k_cart,
        k_frac=k_frac,
        band_e=np.asarray(expanded.band_e, dtype=float),
        band_w=np.asarray(expanded.band_w, dtype=float),
        valid=np.asarray(expanded.valid, dtype=bool),
        ef=float(e_f),
        vbm=float(vbm),
        cbm=float(cbm),
        sigma=float(sigma),
        k_units=np.array("2pi/Angstrom"),
        e_units=np.array("eV"),
    )
