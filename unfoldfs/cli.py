"""命令行接口: ``unfoldfs gen`` / ``unfoldfs unfold`` / ``unfoldfs band`` / ``unfoldfs info``。

四个子命令把整条流水线接起来:

    gen     对称原胞网格 -> KPOINTS + grid_map.npz + INCAR.suggested
    unfold  读 VASP 输出 -> 反折叠 -> 对称扩展 -> 写费米面文件
    band    沿 k 路径做展开能带 -> 写 vaspkit 的 ENERGY/MOMENTUM/WEIGHT/KLABELS
    info    查看网格与对称信息

约定(与 geometry.py / grid.py 一致):
  * k 用笛卡尔坐标, 单位 2π/Å; E 用 eV;
  * E_F 的选取按 ``--ef auto|vbm|doscar|<数值>`` 的语义(见 fsout);
  * ``--resolution`` 的单位是 2π/Å, 按 |b_i| 定义(见 contract-v2 §5)。

KPOINTS 的首行: VASP 按首行的**第一个字符**选择解析分支 —— 'G'/'K' 会走
"自动生成网格"分支(此时显式 k 点列表被忽略, VASP 自己生成 k 点, 个数与
顺序都变), 'S'/'C'/'R' 分别是不可约点/笛卡尔/倒格矢分支。故首行写成
**纯注释**('u' 开头, 不落在任何分支上), 并把 G/M 标签挪进注释里 ——
零成本, 但保证 VASP 走"显式 k 点列表"分支。该首行格式由
``grid.write_kpoints(..., explicit=True)`` 给出(唯一的事实来源)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .band import (assemble_band, dedup_k_points, path_distances, path_labels,
                   path_to_super_k, plot_band, read_klabels, read_kpath,
                   symmetrize_band_weights,
                   write_band_dat, write_grd, write_klabels,
                   write_kpoints_explicit)
from .expand import (build_spectral_function, expand_to_full_grid,
                     symmetrize_spectral_function)
from .fsout import (band_edges, has_fermi_crossing, make_e_grid,
                    select_fermi_energy, write_bxsf,
                    write_cube, write_fermisurfer_bands,
                    write_spectral_npz)
from .geometry import (common_symmetry_ops_detailed, primitive_symmetry_ops,
                       read_poscar, read_transmat, reciprocal, resolve_transform)
from .grid import (GridSpec, build_grid_map, choose_grid_points,
                   filter_ops_by_grid_detailed, load_grid_map,
                   primitive_grid_points, save_grid_map, write_kpoints)
from .unfold import build_atom_copies, unfold_supercell
from .vaspio import BandData, read_doscar_fermi, read_eigenval, read_procar

__all__ = ["main", "cmd_gen", "cmd_unfold", "cmd_band", "cmd_info"]

# 谱函数 npz 的能量窗: 以 E_F 为中心 ±_NPZ_HALF_WIDTH eV, 含 E_F 本身
_NPZ_HALF_WIDTH = 0.5
_NPZ_N_POINTS = 401

# 带隙判据(E_F 语义里的"金属 vs 半导体"), 与 fsout 的默认值一致
_GAP_TOL = 0.05

# E_F 落在带隙里时的**判定阈值**: E_F 离最近的一个态(任意 k, 任意带)超过
# _EMPTY_FS_SIGMA 个 sigma 时, 谱函数在 E_F 处已经衰减到
# exp(-d²/2σ²) <= exp(-_EMPTY_FS_SIGMA²/2) ≈ 13% 的峰值 -> 费米面是空的。
_EMPTY_FS_SIGMA = 2.0

# gen 的默认分辨率(2π/Å); 显式给了 --grid 时不用它
_DEFAULT_RESOLUTION = 0.02

# 常见元素的价电子数(PAW_PBE 势的典型 ZVAL)—— 只用于**估计** NBANDS。
# 势函数未指定时无法知道真实 ZVAL, 故这里给一个表 + 未知元素的保守缺省值,
# 并允许 --nbands 覆盖。
_VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 3, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6,
    "F": 7, "Ne": 8, "Na": 7, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6,
    "Cl": 7, "Ar": 8, "K": 9, "Ca": 10, "Ti": 12, "V": 13, "Cr": 14,
    "Mn": 15, "Fe": 16, "Co": 17, "Ni": 18, "Cu": 11, "Zn": 12, "Ga": 13,
    "Ge": 14, "As": 15, "Se": 6, "Br": 7, "Kr": 8, "Rb": 9, "Sr": 10,
    "Y": 11, "Zr": 12, "Nb": 13, "Mo": 14, "Tc": 13, "Ru": 14, "Rh": 15,
    "Pd": 16, "Ag": 11, "Cd": 12, "In": 13, "Sn": 14, "Sb": 15, "Te": 6,
    "I": 7, "Xe": 8, "Cs": 9, "Ba": 10, "La": 12, "W": 14, "Re": 14,
    "Os": 14, "Ir": 15, "Pt": 18, "Au": 11, "Hg": 12, "Tl": 13, "Pb": 14,
    "Bi": 15,
}
_DEFAULT_VALENCE = 8

# gen 顺手复制到输出目录的结构文件(unfold 默认从 --dir 里找它们)
_STRUCTURE_FILES = ("POSCAR", "PRIMCELL.vasp", "TRANSMAT.in")


# ------------------------------------------------------------------ 小工具

def _valence_estimate(species: Sequence[str]) -> tuple:
    """按价电子数估计原胞所需的带数 -> (nelect, n_bands_prim, 未知元素列表)。"""
    unknown = []
    nelect = 0
    for sp in species:
        z = _VALENCE_ELECTRONS.get(sp)
        if z is None:
            unknown.append(sp)
            z = _DEFAULT_VALENCE
        nelect += z
    n_bands_prim = int(np.ceil(nelect / 2.0))
    return nelect, n_bands_prim, sorted(set(unknown))


def _nbands_with_margin(n_occ_est: int) -> int:
    """建议的 NBANDS = 估计占据带数 + 空带余量。

    余量取 ``max(10, ceil(0.1 * n_occ))``: 少一条带就会把占据态截断(费米面
    静默缺态), 而 NBANDS 恰好等于占据带数时一点余量都不留。
    """
    n_occ = max(int(n_occ_est), 1)
    return n_occ + max(10, int(np.ceil(0.1 * n_occ)))


def _naive_supercell_count(p: Sequence[int],
                            m_eff: np.ndarray) -> tuple:
    """朴素超胞网格的点数 ∏(P_i / M_ii)(算不出来时返回 (None, 原因))。

    这是"不利用对称性时, 超胞 BZ 里 P_i/M_i 个点"的那一套 —— KPOINTS 里
    写的其实就是它(权重把对称等价点合并进同一个 k)。

    返回 (点数 | None, 说明文字): 三种"算不出来"的原因分开给, 否则用户
    只会看到一句笼统的"变换矩阵不是对角的"。
    """
    m = np.asarray(m_eff, dtype=float)
    if not np.allclose(m, np.diag(np.diag(m))):
        return None, ("变换矩阵不是对角的(有非零的非对角元) —— "
                      "∏(P_i/M_i) 这个公式只适用于对角 M")
    diag = np.diag(m)
    if np.any(np.abs(diag - np.rint(diag)) > 1e-8):
        return None, "变换矩阵的对角元不是整数 —— ∏(P_i/M_i) 无意义"
    diag = np.rint(diag).astype(int)
    if np.any(diag <= 0):
        return None, "变换矩阵含非正的对角元 —— 不是合法的超胞放大"
    bad = [i for i in range(3) if int(p[i]) % int(diag[i]) != 0]
    if bad:
        axes = ", ".join(f"P_{i+1}={p[i]} (M_{i+1}={diag[i]})" for i in bad)
        return None, (f"网格点数不是超胞对应维度的整数倍({axes}) —— "
                      "∏(P_i/M_i) 不是整数")
    return int(np.prod([int(p[i]) // int(diag[i]) for i in range(3)])), ""


def _is_numeric_ef(mode) -> bool:
    """--ef 给的是不是一个具体数值(用户明给了 E_F)。"""
    if isinstance(mode, (int, float, np.floating, np.integer)):
        return True
    try:
        float(mode)
    except (TypeError, ValueError):
        return False
    return True


def _nearest_state_distance(band, e_f: float) -> float:
    """E_F 到最近的一个态(任意 k, 任意带)的距离(eV)。"""
    e = np.asarray(band.energies, dtype=float)
    if e.size == 0:
        return float("inf")
    return float(np.min(np.abs(e - float(e_f))))


@dataclass(frozen=True)
class _IncarContext:
    """INCAR 建议文本需要的"体系与计算规模"信息(gen 与 band 共用)。

    抽出来是为了让**能带路径**那条流水线复用同一套 INCAR 生成逻辑
    (ISYM=0 / LORBIT=12 / ISMEAR=1 / SIGMA=0.2 / NBANDS 余量 …) ——
    复制一份的话, 两边的硬要求迟早会走偏。
    """
    n_k: int          # 实际计算的 k 点数
    det: int          # 超胞放大倍数 det(M)
    desc: str         # 一句话说明(网格规格 / 能带路径)
    n_occ_est: int    # 估计的占据带数


def _incar_suggested(ctx: _IncarContext, n_bands: int, encut: float, kpar: int,
                     unknown_species: Sequence[str]) -> str:
    """给出建议的 INCAR 文本。

    其中 ``ISYM = 0`` 与 ``LORBIT = 12`` 是**硬要求**:
      * ISYM=0: 否则 VASP 把显式 k 点表按对称性归约到不可约部分, k 点的个数
        与顺序都会变, 反折叠的映射随之失效;
      * LORBIT=12: PROCAR 必须带复投影系数(相位), LORBIT=11 不写相位。
    """
    head = [
        "# unfoldfs 建议的 INCAR",
        f"# {ctx.desc}, 实际计算 {ctx.n_k} 个超胞 k 点",
        f"# 超胞变换: det(M) = {ctx.det}",
        "#",
        "# ISYM = 0 必须保留: 否则 VASP 会把显式 k 点表按对称性归约, "
        "k 点的个数与顺序都会变,",
        "#   反折叠的 (k 点 -> 候选 G) 映射随之失效。",
        "# LORBIT = 12 必须保留: PROCAR 需要复投影系数(相位), "
        "LORBIT=11 不写相位。",
        "# ISMEAR = 1 + SIGMA = 0.2 是 VASP 默认值: 每个 k 点只需几次 DAV "
        "迭代(实测单核机器上: 9 原子超胞 ~6 s/k 点, 27 原子超胞 ~2 min/k 点)。",
        "#   展宽只影响带边占据数(带隙体系的带边占据数必然是 0.98 这类小数), "
        "并不能让费米能更准。",
        "#   **不要**改成四面体法(ISMEAR <= -4): 四面体法要求 VASP 从 KPOINTS "
        "的**首行**自己解析 MP 网格",
        "#   (首行须为 'M P G X Y Z SHIFT'), 与本工具的**显式 k 点列表**互斥; "
        "本工具的 KPOINTS 首行是纯注释",
        "#   => VASP 报 'Error reading KPOINTS file, at line: N+1'。"
        "四面体法只与 KSPACING(VASP 自建网格)兼容。",
        "#   想用更小的展宽就改成 ISMEAR = 0 + 更小的 SIGMA(仍是展宽法, "
        "不改 k 点语义)。",
        f"# NBANDS = {n_bands} = 估计占据带数 {ctx.n_occ_est} + 空带余量 "
        f"{n_bands - ctx.n_occ_est}(至少 10 条)。",
        "#   NBANDS 由价电子数**估计**得出(表内 ZVAL, Mo=14/S=6 等), "
        "**请按实际 POTCAR 的 ZVAL 核对**,",
        "#   或直接用 unfoldfs gen --nbands <N> 覆盖。",
    ]
    if unknown_species:
        head.append(f"# NBANDS 用价电子数**估计**得出(元素 "
                    f"{', '.join(unknown_species)} 不在价电子数表里, "
                    "按缺省值计); 请按实际势函数核对。")
    body = [
        "ISTART = 0",
        "ICHARG = 2",
        "ISYM = 0",
        "ISMEAR = 1",
        "SIGMA = 0.2",
        "EDIFF = 1E-6",
        "LORBIT = 12",
        "LWAVE = .TRUE.",
        "LCHARG = .FALSE.",
        "IBRION = -1",
        "NSW = 0",
        "PREC = Normal",
        f"ENCUT = {encut:g}",
        f"KPAR = {kpar:d}",
        f"NBANDS = {n_bands:d}",
    ]
    return "\n".join(head + body) + "\n"


# ------------------------------------------------------------------ gen

def _resolve_grid_spec(prim, tr, resolution: Optional[float],
                       explicit: Optional[Sequence[int]],
                       grid_type: str) -> GridSpec:
    """网格规格: 显式 --grid 优先, 否则按 --resolution 推。

    ``--grid`` 与 ``--resolution`` 互斥(同时给会让人猜到底是哪个生效)。
    """
    if explicit is not None:
        if resolution is not None:
            raise ValueError(
                "--grid 与 --resolution 互斥: 只能给其中一个 "
                "(--grid 显式指定网格, --resolution 按分辨率推)。")
        p = tuple(int(x) for x in explicit)
        if any(x < 1 for x in p):
            raise ValueError(
                f"--grid 的每一维都必须是 >= 1 的整数, 给出 {p}")
        return GridSpec(p, grid_type, (0.0, 0.0, 0.0))
    res = _DEFAULT_RESOLUTION if resolution is None else float(resolution)
    return choose_grid_points(prim.lattice, tr.m_eff, res, grid_type)


def cmd_gen(args: argparse.Namespace) -> int:
    """对称约化网格 -> KPOINTS / grid_map.npz / INCAR.suggested。"""
    prim = read_poscar(args.primcell)
    sc = read_poscar(args.supercell)
    tr = resolve_transform(sc, prim, read_transmat(args.transmat))
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    spec = _resolve_grid_spec(prim, tr, args.resolution, args.grid,
                               args.grid_type)
    grid = primitive_grid_points(spec)
    # 公共对称集 + 被剔除的操作与原因(spec §7: info 里要列出剔除项)
    all_ops, dropped = common_symmetry_ops_detailed(prim, tr.m_eff)
    ops, dropped_grid = filter_ops_by_grid_detailed(all_ops, grid)
    gm = build_grid_map(spec, grid, ops, tr, b_prim, b_super,
                        dropped=tuple(dropped) + tuple(dropped_grid))

    # 自洽性: 满网格点数必须等于 n_k · det(M)(expand 的硬性前提)
    det = int(round(tr.det))
    if gm.spec.n_points != gm.n_k * det:
        raise ValueError(
            f"网格 {gm.spec.p} 与超胞变换不自洽: 满网格点数 "
            f"{gm.spec.n_points} != n_k·det(M) = {gm.n_k}×{det}。"
            "请换一个 --grid(每个维度都应是超胞对应维度的整数倍)。")

    os.makedirs(args.outdir, exist_ok=True)
    kpoints_path = os.path.join(args.outdir, "KPOINTS")
    gridmap_path = os.path.join(args.outdir, "grid_map.npz")
    incar_path = os.path.join(args.outdir, "INCAR.suggested")
    write_kpoints(kpoints_path, gm, explicit=True)
    save_grid_map(gridmap_path, gm)

    nelect, n_bands_prim, unknown = _valence_estimate(prim.species)
    n_occ_est = det * max(n_bands_prim, 1)
    if args.nbands is not None:
        n_bands = int(args.nbands)
    else:
        n_bands = _nbands_with_margin(n_occ_est)
    with open(incar_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(_incar_suggested(
            _IncarContext(n_k=gm.n_k, det=gm.transform.det,
                          desc=f"网格: {spec.p[0]} x {spec.p[1]} x {spec.p[2]} "
                               f"({spec.grid_type})",
                          n_occ_est=n_occ_est),
            n_bands, args.encut, args.kpar, unknown))

    # 结构文件顺手复制过去: unfold 默认从 --dir 里找它们
    sources = {"POSCAR": args.supercell, "PRIMCELL.vasp": args.primcell,
               "TRANSMAT.in": args.transmat}
    copied = []
    for name in _STRUCTURE_FILES:
        src = sources[name]
        dst = os.path.join(args.outdir, name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
            copied.append(dst)

    print(f"网格规格      : {spec.p[0]} x {spec.p[1]} x {spec.p[2]} "
          f"({spec.grid_type}), 满网格 {spec.n_points} 点")
    print(f"对称不可约点  : {gm.n_ir}")
    print(f"实际计算 k 点 : {gm.n_k} (超胞分数坐标), 权重和 = "
          f"{int(gm.k_weight.sum())} == 满网格点数")
    print(f"超胞变换      : det(M) = {tr.det}")
    print(f"估计电子数    : {nelect} (价电子数估计), 估计占据带数 = {n_occ_est}, "
          f"NBANDS = {n_bands}")
    print(f"已写出        : {kpoints_path}")
    print(f"已写出        : {gridmap_path}")
    print(f"已写出        : {incar_path}")
    for path in copied:
        print(f"已复制        : {path}")
    return 0


# ------------------------------------------------------------------ unfold

def _check_ef_in_gap(band, vbm: float, cbm: float, e_f: float,
                     sigma: float, numeric_mode: bool):
    """E_F 是否落在带隙里(落在带隙里 ⇒ 该能量上**没有任何态**)。

    返回 (status, message):
      "ok"     —— E_F 不在带隙里(或带隙/带边不可得), 无需提示;
      "near"   —— E_F 在带隙里但离最近的态很近(谱函数在 E_F 处仍有峰),
                  画的是价带顶附近的等值面;
      "warn"   —— E_F 在带隙里且谱函数在 E_F 处已可忽略 —— 但用户明给了
                  这个 E_F(数值模式), 只提示不阻断;
      "refuse" —— 同上, 但 E_F 是 CLI 自己(或 VASP)选的 => 拒绝执行。

    "离最近的态"的判据: 谱函数 A(k,E) 是高斯, 相对权重 ≈ exp(-d²/2σ²);
    d > 2σ 时只剩 13%, 画出来的"费米面"基本是空的。
    """
    if not (np.isfinite(vbm) and np.isfinite(cbm)):
        return "ok", ""
    gap = cbm - vbm
    if gap <= _GAP_TOL or not (vbm < e_f < cbm):
        return "ok", ""
    d = _nearest_state_distance(band, e_f)
    near = d <= _EMPTY_FS_SIGMA * sigma
    if near:
        return "near", (
            f"E_F = {e_f:.6f} eV 落在带隙内 (VBM={vbm:.6f}, CBM={cbm:.6f}, "
            f"gap={gap:.6f} eV), 但离最近的态只有 {d:.6f} eV "
            f"(<= {_EMPTY_FS_SIGMA:g}×sigma) —— 画的是价带顶附近的等值面。")
    msg = (
        f"选出的 E_F = {e_f:.6f} eV 落在带隙内 (VBM={vbm:.6f}, "
        f"CBM={cbm:.6f}, gap={gap:.6f} eV), 离最近的态 {d:.6f} eV "
        f"(> {_EMPTY_FS_SIGMA:g}×sigma) —— 该能量上没有任何态, "
        f"产出的费米面是空的。")
    if numeric_mode:
        return "warn", msg
    return "refuse", msg


def _read_eigenval_needed(run_dir: str, mode) -> BandData:
    """按需读 EIGENVAL, 并给出**可操作**的报错。

    * E_F 的**选取**只有 ``auto``/``vbm`` 需要 EIGENVAL 的**占据数**
      (VBM/CBM + 金属判据); 数值/``doscar`` 模式不需要它;
    * 但**反折叠本身**需要 EIGENVAL 的**带能量** —— PROCAR 里没有能量。

    ⇒ EIGENVAL 缺失时必须说清"缺的是哪一部分", 而不是甩一个裸的
    ``[Errno 2] No such file or directory: ...EIGENVAL``。
    """
    path = os.path.join(run_dir, "EIGENVAL")
    if os.path.exists(path):
        return read_eigenval(path)
    if str(mode).strip().lower() in ("auto", "vbm"):
        raise ValueError(
            f"--ef {mode} 需要 EIGENVAL 的占据数(求 VBM/CBM 与判金属), "
            f"但 {path} 不存在。")
    raise ValueError(
        f"反折叠需要 EIGENVAL 的**带能量**(PROCAR 里没有能量), "
        f"但 {path} 不存在。\n"
        f"      注意: E_F 的选取本身不需要 EIGENVAL(--ef {mode} 已经给了明确的 "
        f"E_F), 但带能量只有 EIGENVAL 里有。")


def _check_grid_map_consistency(gm, tr, prim, sc) -> None:
    """grid_map.npz 必须与当前的 POSCAR/PRIMCELL.vasp/TRANSMAT.in **自洽**。

    不自洽时(用户把别的目录的 grid_map 混进来、或拷错了 POSCAR)在展开的
    **最贵的一步之后**才会炸成一个看不懂的 IndexError
    (``coeffs[:, cp.ion, :]`` 的离子下标越界)。⇒ 在最贵的一步**之前**挡住,
    并说清是哪一项对不上。
    """
    gm_m = np.asarray(gm.transform.m_eff)
    tr_m = np.asarray(tr.m_eff)
    if not np.array_equal(gm_m, tr_m):
        raise ValueError(
            "grid_map 与 POSCAR/TRANSMAT 不自洽: 变换矩阵不同 —— "
            f"grid_map 记的是 {gm_m.tolist()}, 而当前结构给出 {tr_m.tolist()}。"
            "请确认 --dir 里的 grid_map.npz 与 POSCAR/PRIMCELL.vasp/"
            "TRANSMAT.in 出自同一次 `unfoldfs gen`。")
    for name, stored, now in (("b_prim", gm.b_prim, reciprocal(prim.lattice)),
                              ("b_super", gm.b_super, reciprocal(sc.lattice))):
        if not np.allclose(np.asarray(stored, dtype=float),
                           np.asarray(now, dtype=float), atol=1e-6):
            raise ValueError(
                f"grid_map 与 POSCAR/PRIMCELL.vasp 不自洽: {name} 不同 —— "
                f"grid_map 记的是 {np.asarray(stored).tolist()}, "
                f"而当前结构给出 {np.asarray(now).tolist()}。"
                "请确认 --dir 里的 grid_map.npz 与结构文件出自同一次 "
                "`unfoldfs gen`。")


def _print_convention_line(unfolded, det: int) -> None:
    """打印相位约定那一行 —— 直通模式(det(M)=1)下**没有**约定可择优。

    直通模式不能打印"单 G 主导度 nan"(那是个填充值, 不是分数), 也不该
    打印 s1/s2(没有择优发生过)⇒ 换成一行说清模式与权重定义的话。
    """
    if det == 1:
        print("模式          : 原胞直通 (det(M)=1 —— 不做展开; "
              "权重 = Σ_{ion,lm}|c|², 与超胞展开模式的 Popescu G 投影不是"
              "同一个定义)")
        return
    print(f"相位约定      : s1 = {unfolded.s1:+d}, s2 = {unfolded.s2:+d} "
          f"(单 G 主导度 {unfolded.dominance:.4f})")


def _write_spectral_npz(path: str, expanded, b_prim: np.ndarray, e_f: float, *,
                        vbm: float, cbm: float, e_grid: np.ndarray, sigma: float,
                        symmetrize=None, ops=None) -> None:
    """写 ``A(k,E).npz``; 给定 ``symmetrize`` 时把文件里的 A 换成**对称化后**的。

    ``fsout.write_spectral_npz`` 自己从 ``expanded`` 重算 A(k,E) —— 而对称化是
    **投影**(不可逆: 逐带数据在对称相关点上根本不是同一组带, 见
    docs/algorithms.md §15.7)⇒ 无法由对称化后的 band_e/band_w 重算 ⇒ 这里在
    写完之后把 A 换掉, 其余键原样保留。

    ⇒ npz 里的 ``A`` 是**对称化后**的谱函数, 而 ``band_e``/``band_w`` 是
    **原始**展开数据(见 docs/formats.md)。

    ``ops``: 对称化用的对称操作(缺省 = 原胞完整点群, 见
    ``expand.symmetrize_spectral_function``)。
    """
    write_spectral_npz(path, expanded, b_prim, e_f, vbm=vbm, cbm=cbm,
                       e_grid=e_grid, sigma=sigma)
    if symmetrize is None:
        return
    a = symmetrize_spectral_function(
        build_spectral_function(expanded, e_grid, sigma), symmetrize, ops=ops)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["A"] = a
    np.savez_compressed(path, **arrays)


def cmd_unfold(args: argparse.Namespace) -> int:
    """读 VASP 输出 -> 反折叠 -> 对称扩展 -> 写费米面文件。"""
    run_dir = args.dir
    supercell = args.supercell or os.path.join(run_dir, "POSCAR")
    primcell = args.primcell or os.path.join(run_dir, "PRIMCELL.vasp")
    transmat = args.transmat or os.path.join(run_dir, "TRANSMAT.in")
    gridmap_path = args.grid_map or os.path.join(run_dir, "grid_map.npz")

    gm = load_grid_map(gridmap_path)
    band = _read_eigenval_needed(run_dir, args.ef)

    # k 点数守卫: 比**真实**的 k 点数(EIGENVAL 里实际的 k 点块数), 不能拿
    # 截断后的自身比 —— 那样永远相等, 守卫形同虚设。
    n_calc = band.kpoints.shape[0]
    if n_calc != gm.n_k:
        raise ValueError(
            f"VASP 算出的 k 点数 {n_calc} != grid_map 记录 {gm.n_k}。"
            f"请确认 KPOINTS 用的是 gen 写出的文件, 且 INCAR 里 ISYM=0。")

    # NBANDS 事后检查: 最高带被占据 => NBANDS 偏小, 占据态被截断
    occ = np.asarray(band.occupations, dtype=float)
    if occ.ndim == 2 and occ.shape[1] > 0:
        top = float(np.max(occ[:, -1]))
        if top > 1e-6:
            print(f"警告: 最高带(第 {occ.shape[1]} 带)的占据数最大为 "
                  f"{top:.6f} > 0 —— NBANDS 可能偏小, 占据态被截断, "
                  f"费米面会缺态。请在 INCAR 里加大 NBANDS(建议 "
                  f"unfoldfs gen --nbands 覆盖)。", file=sys.stderr)

    procar = read_procar(os.path.join(run_dir, "PROCAR"))
    if procar.kpoints.shape[0] != gm.n_k:
        raise ValueError(
            f"PROCAR 里的 k 点数 {procar.kpoints.shape[0]} != grid_map 记录 "
            f"{gm.n_k}。请确认 PROCAR 与 EIGENVAL 出自同一次计算。")

    dos_path = os.path.join(run_dir, "DOSCAR")
    need_dos = str(args.ef).strip().lower() in ("auto", "doscar", "dos")
    if os.path.exists(dos_path):
        dos_ef = read_doscar_fermi(dos_path)
    elif need_dos:
        raise ValueError(
            f"--ef {args.ef} 需要 DOSCAR 里的费米能, 但 {dos_path} 不存在。")
    else:
        dos_ef = float("nan")

    prim = read_poscar(primcell)
    sc = read_poscar(supercell)
    tr = resolve_transform(sc, prim, read_transmat(transmat))
    _check_grid_map_consistency(gm, tr, prim, sc)
    copies = build_atom_copies(prim, sc, tr)
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    # E_F 的选取与自检放在反折叠**之前**: 落在带隙深处时要在昂贵的计算之前
    # 就中止(而不是让用户等完再被告知结果没有意义)。
    # VBM/CBM 只在 EIGENVAL 里有占据态时才有意义; --ef <数值> 不该被它挡住
    #
    # 费米穿越 = 金属判据(与 fsout.has_fermi_crossing 同一个判据): 有带的
    # 占据数在 k 上跨过 0.5。对金属而言 band_edges 给的 VBM/CBM 只是**表观值**
    # (有限 k 网格的采样假象), 不能拿它当带隙。
    fermi_crossing = has_fermi_crossing(band)
    try:
        vbm, cbm = band_edges(band)
        vbm_txt = f"{vbm:.6f} / {cbm:.6f} eV"
        if fermi_crossing:
            vbm_txt += " (金属: 表观值, 有限 k 网格的采样假象)"
    except ValueError as exc:
        vbm, cbm = float("nan"), float("nan")
        vbm_txt = f"不可用 ({exc})"
    e_f = select_fermi_energy(dos_ef, band, mode=args.ef, offset=args.delta,
                               gap_tol=_GAP_TOL)

    # ★ 自检: 选出的 E_F 落在带隙里 => 该能量上没有任何态 => 费米面是空的。
    # 这是"默认路径静默产出空费米面"的最后一道防线(不依赖 is_metal 的判定)。
    #
    # ⚠ **金属必须跳过这个检查**: 金属的表观带隙是采样假象(1T-TaS2 实测:
    # 6x6x1 网格上表观 gap = 0.209156 eV, DOSCAR 的 E_F = -1.377658 正落在
    # 里面), 而那个能量上其实**有**态(带 25/26/27 穿过费米能) ⇒ 拿它拒绝
    # 执行会把一次成功的金属计算扔掉。
    numeric_ef = _is_numeric_ef(args.ef)
    if fermi_crossing:
        print("提示: 检测到费米穿越(金属) —— 有带的占据数在 k 上跨过 0.5; "
              "金属的表观带隙是有限 k 网格的采样假象, 跳过带隙检查。",
              file=sys.stderr)
    else:
        status, message = _check_ef_in_gap(band, vbm, cbm, e_f, args.sigma,
                                            numeric_ef)
        if status == "refuse":
            raise ValueError(
                message + "\n      半导体请用 --ef vbm (VBM+delta); "
                "金属请检查 ISMEAR/展宽设置。已中止, 未写出任何费米面文件。")
        if status in ("near", "warn"):
            print(f"提示: {message}", file=sys.stderr)

    # 展开必须传 gm.k_super(去重后的超胞 k), 不是 IR 点列表
    unfolded = unfold_supercell(
        band, procar, tr, copies, tr.m_eff, prim.lattice,
        prim.positions @ prim.lattice, b_prim, b_super, gm.k_super,
        strict_dominance=args.strict_dominance)
    expanded = expand_to_full_grid(unfolded, gm)

    # ★ 点群轨道对称化(默认开): 把展开谱 A(k,E) 投影到"对称不变子空间" ——
    # 展开得到的谱权重在点群下**不协变**(docs/algorithms.md §15), 不对称化
    # 的话费米面的等值面相对晶体对称性是畸变的。
    #
    # ★ 用**原胞的完整点群**(修复轮 6): 原胞的谱函数按构造就在这个点群下
    # 不变 ⇒ 投影有物理依据; 而"超胞还保留的那部分操作"只对保持对称的超胞
    # 够用 —— 破坏对称性的超胞(如 3×1×1)只剩 4/12 个操作, 六重对称恢复不了。
    symmetrize = not getattr(args, "no_symmetrize", False)
    full_ops = primitive_symmetry_ops(prim)
    n_common = int(gm.ops.rotations.shape[0])

    _print_convention_line(unfolded, tr.det)
    print("点群对称化    : " + ("已启用(按**原胞完整点群**("
                               f"{full_ops.rotations.shape[0]} 个操作)轨道平均 "
                               "A(k,E); --no-symmetrize 可关闭)" if symmetrize
                               else "已关闭(--no-symmetrize) —— 谱权重是"
                                    "逐 (K,G) 的原始权重, 费米面带对称性破缺"))
    print(f"公共对称操作  : {n_common} 个(原胞与超胞共有的; 超胞破坏掉的"
          f" {full_ops.rotations.shape[0] - n_common} 个由完整点群补回)")
    print(f"VBM / CBM     : {vbm_txt}")
    print(f"E_F           : {e_f:.6f} eV  (--ef {args.ef}, "
          f"--delta {args.delta:g})")

    os.makedirs(args.outdir, exist_ok=True)
    values = build_spectral_function(expanded, np.array([e_f]),
                                     args.sigma)[:, 0]
    if symmetrize:
        values = symmetrize_spectral_function(values, gm, ops=full_ops)
    comments = _fs_comments(e_f, args, vbm, cbm)

    cube_path = os.path.join(args.outdir, "FS.cube")
    bxsf_path = os.path.join(args.outdir, "FS.bxsf")
    npz_path = os.path.join(args.outdir, "A(k,E).npz")
    write_cube(cube_path, expanded, b_prim, values, comments=comments)
    print(f"已写出        : {cube_path}")
    band_paths = write_fermisurfer_bands(args.outdir, expanded, b_prim, e_f)
    print(f"已写出        : {os.path.join(args.outdir, 'band.1')} ... "
          f"band.{len(band_paths)}")
    print("逐带文件      : FS.bxsf / band.N **未**对称化 —— 逐带数据在对称相关"
          "点上根本不是同一组带(实测 3×1×1 的 1T-TaS2: 19 条轨道里只有 1 条"
          "的有效带数一致), 按带序号平均会把不同的带混在一起;"
          "谱函数派生的输出(FS.cube / A(k,E).npz)是完全对称的)")
    e_grid = make_e_grid(e_f, _NPZ_HALF_WIDTH, _NPZ_N_POINTS)
    _write_spectral_npz(npz_path, expanded, b_prim, e_f, vbm=vbm, cbm=cbm,
                        e_grid=e_grid, sigma=args.sigma,
                        symmetrize=gm if symmetrize else None,
                        ops=full_ops if symmetrize else None)
    print(f"已写出        : {npz_path}")
    # BXSF 只支持 Gamma 网格(见 docs/formats.md §3): MP 网格下跳过它,
    # 但把原因说清楚 —— 其余输出(FS.cube / band.N / npz)对 MP 网格都成立
    try:
        write_bxsf(bxsf_path, expanded, b_prim, e_f, comments=comments)
    except ValueError as exc:
        print(f"已跳过        : {bxsf_path} ({exc})")
    else:
        print(f"已写出        : {bxsf_path}")
    return 0


def _fs_comments(e_f: float, args: argparse.Namespace, vbm: float,
                 cbm: float) -> List[str]:
    """写进各费米面文件头部的两行说明(含 E_F)。"""
    line1 = "unfoldfs: unfolded Fermi surface (spectral function A(k, E_F))"
    vbm_txt = "n/a" if not np.isfinite(vbm) else f"{vbm:.6f}"
    cbm_txt = "n/a" if not np.isfinite(cbm) else f"{cbm:.6f}"
    line2 = (f"E_F = {e_f:.6f} eV (--ef {args.ef}, --delta {args.delta:g}, "
             f"--sigma {args.sigma:g}); VBM = {vbm_txt} CBM = {cbm_txt} eV; "
             "k = 2pi/Angstrom; E = eV; cell vectors in Bohr (cube = a.u.); "
             "reciprocal box = BZ")
    return [line1, line2]


# ------------------------------------------------------------------ band

def _check_klabels_consistent(path: str, labels: Sequence[tuple]) -> None:
    """阶段 1 写下的 KLABELS.in 必须与当前 KPATH.in 算出的标签一致。

    不一致说明用户改了 KPATH.in 却没重跑阶段 1 ⇒ 阶段 2 复原出来的 KLABELS
    与实际算的 k 点对不上。**必须报错**: 标签与距离对不上时, MATLAB 脚本
    画出来的 x 轴刻度是错的, 而且看不出来。
    """
    stored = read_klabels(path)
    if len(stored) != len(labels):
        raise ValueError(
            f"{os.path.basename(path)} 里有 {len(stored)} 个标签, 而当前的 "
            f"KPATH.in 算出 {len(labels)} 个 —— 两者不自洽。"
            "请用同一个 KPATH.in 重跑阶段 1(`unfoldfs band --kpath ...`)。")
    for (lab_a, dist_a), (lab_b, dist_b) in zip(stored, labels):
        # KLABELS 里的距离只保留 3 位小数(与 vaspkit 一致) ⇒ 按**写出来的值**比
        if lab_a != lab_b or round(float(dist_a), 3) != round(float(dist_b), 3):
            raise ValueError(
                f"{os.path.basename(path)} 与当前的 KPATH.in 不自洽: "
                f"({lab_a}, {dist_a}) != ({lab_b}, {dist_b}) —— "
                "请用同一个 KPATH.in 重跑阶段 1(`unfoldfs band --kpath ...`)。")


def cmd_band_prepare(args: argparse.Namespace) -> int:
    """阶段 1: KPATH.in -> 超胞 k -> KPOINTS + INCAR.suggested + KLABELS.in。"""
    prim = read_poscar(args.primcell)
    sc = read_poscar(args.supercell)
    tr = resolve_transform(sc, prim, read_transmat(args.transmat))
    b_prim = reciprocal(prim.lattice)

    path = read_kpath(args.kpath)
    dist = path_distances(path, b_prim)
    labels = path_labels(path, dist)
    # K = k'·Mᵀ (mod 1): 路径点用的是**原胞**分数坐标, VASP 算的是**超胞** k
    k_super = path_to_super_k(path.k_prim, tr.m_eff)
    # ★ 写进 KPOINTS 的 k 点必须**去重**(修复轮 7 / B1): 路径的相邻段共享
    # 端点, 且不同的原胞 k 可能折到同一个超胞 k ⇒ VASP 6.6.0 读到含重复 k 点
    # 的显式列表会崩(服务器实测: 每次都在第一个 k 点算完后死掉)。表头计数
    # 必须等于实际写出的行数。
    k_unique, _index_map = dedup_k_points(k_super)

    outdir = args.outdir or "run"
    os.makedirs(outdir, exist_ok=True)
    kpoints_path = os.path.join(outdir, "KPOINTS")
    # 传**原始**列表: 去重由写文件的那一步统一做(单一出处)
    write_kpoints_explicit(kpoints_path, k_super)

    nelect, n_bands_prim, unknown = _valence_estimate(prim.species)
    n_occ_est = tr.det * max(n_bands_prim, 1)
    n_bands = int(args.nbands) if args.nbands is not None \
        else _nbands_with_margin(n_occ_est)
    incar_path = os.path.join(outdir, "INCAR.suggested")
    with open(incar_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(_incar_suggested(
            _IncarContext(n_k=int(k_unique.shape[0]), det=tr.det,
                          desc=f"能带路径: {path.n_k} 个路径点 -> "
                               f"{k_unique.shape[0]} 个待算的超胞 k"
                               "(原胞分数坐标 -> 超胞分数坐标, 重复端点只算一次)",
                          n_occ_est=n_occ_est),
            n_bands, args.encut, args.kpar, unknown))

    klabels_path = os.path.join(outdir, "KLABELS.in")
    write_klabels(klabels_path, labels)

    # 结构文件与路径文件顺手复制过去(阶段 2 从 --dir 里找它们)
    sources = {"POSCAR": args.supercell, "PRIMCELL.vasp": args.primcell,
               "TRANSMAT.in": args.transmat, "KPATH.in": args.kpath}
    copied = []
    for name, src in sources.items():
        dst = os.path.join(outdir, name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
            copied.append(dst)

    print(f"路径段数      : {len(path.segments)}(每段 {path.n_points_per_segment} "
          f"个采样点, 含两端)")
    print(f"路径点总数    : {path.n_k}")
    print(f"KPOINTS k 点  : {k_unique.shape[0]}(去重后 —— 重复端点与折到同一"
          "超胞 k 的点只算一次)")
    print(f"高对称点      : {', '.join(lab for lab, _ in labels)}")
    print(f"路径总长度    : {float(dist[-1]):.6f} (2π/Å)")
    print(f"超胞变换      : det(M) = {tr.det}")
    print(f"估计占据带数  : {n_occ_est}, NBANDS = {n_bands}")
    print(f"已写出        : {kpoints_path}")
    print(f"已写出        : {incar_path}")
    print(f"已写出        : {klabels_path}")
    for path_out in copied:
        print(f"已复制        : {path_out}")
    return 0


def cmd_band_unfold(args: argparse.Namespace) -> int:
    """阶段 2: VASP 输出 -> 展开能带 -> 写 vaspkit 的四个文件(+ PNG)。"""
    run_dir = args.dir
    supercell = args.supercell or os.path.join(run_dir, "POSCAR")
    primcell = args.primcell or os.path.join(run_dir, "PRIMCELL.vasp")
    transmat = args.transmat or os.path.join(run_dir, "TRANSMAT.in")
    kpath_path = os.path.join(run_dir, "KPATH.in")
    if not os.path.exists(kpath_path):
        raise ValueError(
            f"阶段 2 需要 {kpath_path}(路径信息在阶段 1 写的 KPATH.in 里) —— "
            "请先跑阶段 1: `unfoldfs band --supercell ... --primcell ... "
            "--transmat ... --kpath KPATH.in --outdir <dir>`")

    band_data = _read_eigenval_needed(run_dir, args.ef)
    procar = read_procar(os.path.join(run_dir, "PROCAR"))
    dos_path = os.path.join(run_dir, "DOSCAR")
    need_dos = str(args.ef).strip().lower() in ("auto", "doscar", "dos")
    if os.path.exists(dos_path):
        dos_ef = read_doscar_fermi(dos_path)
    elif need_dos:
        raise ValueError(
            f"--ef {args.ef} 需要 DOSCAR 里的费米能, 但 {dos_path} 不存在。")
    else:
        dos_ef = float("nan")

    prim = read_poscar(primcell)
    sc = read_poscar(supercell)
    tr = resolve_transform(sc, prim, read_transmat(transmat))
    copies = build_atom_copies(prim, sc, tr)
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    e_f = select_fermi_energy(dos_ef, band_data, mode=args.ef,
                               offset=args.delta, gap_tol=_GAP_TOL)
    # E_F 落在带隙里对能带图是**正常**的(半导体), 只提示, 不阻断
    try:
        vbm, cbm = band_edges(band_data)
    except ValueError:
        vbm = cbm = float("nan")
    status, message = _check_ef_in_gap(band_data, vbm, cbm, e_f, args.sigma,
                                        _is_numeric_ef(args.ef))
    if status in ("near", "warn"):
        print(f"提示: {message}", file=sys.stderr)

    path = read_kpath(kpath_path)
    dist = path_distances(path, b_prim)
    labels = path_labels(path, dist)
    klabels_in = os.path.join(run_dir, "KLABELS.in")
    if os.path.exists(klabels_in):
        _check_klabels_consistent(klabels_in, labels)
    k_super = path_to_super_k(path.k_prim, tr.m_eff)
    # ★ 计算的 k 点与**路径点**不是一回事(修复轮 7 / B1): 路径的相邻段共享
    # 端点, 且不同的原胞 k 可能折到同一个超胞 k ⇒ VASP 算的是**去重后**的
    # k 列表(与阶段 1 写进 KPOINTS 的完全一致), 路径点则保持原样(KLABELS /
    # unfolded_band.dat 仍是原始路径点数)。index_map 把路径点映射回已计算的 k。
    k_unique, index_map = dedup_k_points(k_super)

    # 相位约定择优用**原型口径**: k 列表的**前 min(4, n_k) 个点** × 前
    # min(n_band, 40) 条带(见 unfold.CONVENTION_SAMPLE_N_K)。**与 `unfold`
    # 完全同一条规则** —— 修复轮 7 / B4: `band` 曾经把路径点摊开再采样,
    # 于是两个子命令在同一组超胞 k 上选出了不同的 (s1, s2)
    # (实测 band=(-1,+1) vs unfold=(+1,-1))⇒ 结果不可复现。
    unfolded = unfold_supercell(
        band_data, procar, tr, copies, tr.m_eff, prim.lattice,
        prim.positions @ prim.lattice, b_prim, b_super, k_unique,
        strict_dominance=args.strict_dominance)
    energy, weight = assemble_band(unfolded, path.k_prim, index_map=index_map)
    # ★ 点群轨道对称化(默认开): 路径上**对称相关**的两点会给出不同的展开谱
    # 权重(docs/algorithms.md §15)⇒ 同一段路径的两半颜色不同。带能量不动。
    #
    # ★ 用**原胞的完整点群**(修复轮 6): 破坏对称性的超胞(如 3×1×1)只保留
    # 4/12 个公共操作, 用公共子群做平均会漏掉路径上其余的对称相关点。
    symmetrize = not getattr(args, "no_symmetrize", False)
    if symmetrize:
        weight = symmetrize_band_weights(
            energy, weight, path.k_prim,
            primitive_symmetry_ops(prim))

    outdir = args.outdir or "bandfs"
    os.makedirs(outdir, exist_ok=True)
    energy_path = os.path.join(outdir, "ENERGY.grd")
    momentum_path = os.path.join(outdir, "MOMENTUM.grd")
    weight_path = os.path.join(outdir, "WEIGHT.grd")
    klabels_path = os.path.join(outdir, "KLABELS")
    dat_path = os.path.join(outdir, "unfolded_band.dat")
    write_grd(energy_path, energy)
    write_grd(momentum_path, np.tile(dist, (energy.shape[0], 1)))
    write_grd(weight_path, weight)
    write_klabels(klabels_path, labels)
    write_band_dat(dat_path, dist, energy, weight)

    _print_convention_line(unfolded, tr.det)
    print("点群对称化    : " + ("已启用(按点群轨道平均路径上对称相关点的权重; "
                               "--no-symmetrize 可关闭)" if symmetrize
                               else "已关闭(--no-symmetrize)"))
    print(f"E_F           : {e_f:.6f} eV  (--ef {args.ef}, --delta {args.delta:g})")
    print(f"路径点        : {path.n_k}(高对称点 "
          f"{', '.join(lab for lab, _ in labels)})")
    print(f"展开带数      : {energy.shape[0]}")
    print(f"已写出        : {energy_path}")
    print(f"已写出        : {momentum_path}")
    print(f"已写出        : {weight_path}")
    print(f"已写出        : {klabels_path}")
    print(f"已写出        : {dat_path}")

    if args.no_plot:
        print("已跳过        : PNG(--no-plot)")
        return 0
    png_path = os.path.join(outdir, "unfolded_band.png")
    try:
        plot_band(dist, energy, weight, labels, e_f=e_f, out_path=png_path,
                  e_min=args.e_min, e_max=args.e_max, mirror=args.mirror)
    except ValueError as exc:
        raise ValueError(
            f"{exc}\n      (四个 vaspkit 文件已经写好了, 只是画不出 PNG)") from exc
    print(f"已写出        : {png_path}")
    return 0


def cmd_band(args: argparse.Namespace) -> int:
    """``unfoldfs band``: 阶段 1 准备 VASP 输入, 阶段 2 出展开能带。"""
    if args.kpath and args.dir:
        raise ValueError(
            "--kpath(阶段 1)与 --dir(阶段 2)只能给一个: "
            "阶段 1 生成 KPOINTS 供 VASP 算, 阶段 2 才读 VASP 的输出")
    if args.kpath:
        for name, value in (("--supercell", args.supercell),
                            ("--primcell", args.primcell),
                            ("--transmat", args.transmat)):
            if not value:
                raise ValueError(f"阶段 1 需要 {name}")
        return cmd_band_prepare(args)
    if args.dir:
        return cmd_band_unfold(args)
    raise ValueError(
        "`unfoldfs band` 需要 --kpath(阶段 1: 准备 VASP 输入)"
        "或 --dir(阶段 2: 从 VASP 输出得到展开能带)")


# ------------------------------------------------------------------ info

def _symmetry_summary(gm) -> str:
    """'保留 N / 剔除 M(原因: …)' 一行摘要(spec §7 要求 info 列出剔除项)。"""
    n_keep = int(gm.ops.rotations.shape[0])
    dropped = list(getattr(gm, "dropped", ()) or ())
    if not dropped:
        return f"保留 {n_keep} / 剔除 0"
    reasons: dict = {}
    for _r, _t, why in dropped:
        reasons[why] = reasons.get(why, 0) + 1
    detail = "; ".join(f"{why} × {n}" for why, n in reasons.items())
    return f"保留 {n_keep} / 剔除 {len(dropped)} (原因: {detail})"


def cmd_info(args: argparse.Namespace) -> int:
    """打印网格规模、对称性与约化率。"""
    gridmap_path = args.grid_map or os.path.join(args.dir, "grid_map.npz")
    gm = load_grid_map(gridmap_path)
    p = gm.spec.p
    ratio = gm.n_k / float(gm.spec.n_points)
    naive, naive_reason = _naive_supercell_count(p, gm.transform.m_eff)
    naive_txt = (f"{naive} 点 (∏(P_i/M_i))" if naive is not None
                 else f"n/a ({naive_reason})")
    print(f"网格规格      : {p[0]} x {p[1]} x {p[2]} ({gm.spec.grid_type})")
    print(f"满网格点数    : {gm.spec.n_points}")
    print(f"对称不可约点  : {gm.n_ir}")
    print(f"对称操作      : {_symmetry_summary(gm)}")
    print(f"实际计算 k 点 : {gm.n_k} (超胞分数坐标)")
    print(f"朴素超胞网格  : {naive_txt}")
    print(f"相对原胞满网格: {gm.n_k}/{gm.spec.n_points} = {ratio:.4f}")
    print(f"超胞变换      : det(M) = {gm.transform.det}")
    print(f"KPOINTS 权重和: {int(gm.k_weight.sum())} (应等于满网格点数 "
          f"{gm.spec.n_points})")
    return 0


# ------------------------------------------------------------------ 参数解析

def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--supercell", required=True, help="超胞 POSCAR 路径")
    p.add_argument("--primcell", required=True, help="原胞 PRIMCELL.vasp 路径")
    p.add_argument("--transmat", required=True, help="TRANSMAT.in 路径")
    p.add_argument("--resolution", type=float, default=None,
                   help="k 网格分辨率, 单位 2π/Å, 按 |b_i| 定义 "
                        f"(不给则用 {_DEFAULT_RESOLUTION}; 与 --grid 互斥)")
    p.add_argument("--grid", type=int, nargs=3, default=None,
                   metavar=("P1", "P2", "P3"),
                   help="显式指定原胞网格点数(如 --grid 6 6 1), 绕过分辨率"
                        "规则; 与 --resolution 互斥")
    p.add_argument("--grid-type", choices=("gamma", "mp"), default="gamma",
                   help="网格类型: gamma(Gamma 居中) 或 mp(Monkhorst-Pack)")
    p.add_argument("--outdir", default=".", help="输出目录")
    p.add_argument("--nbands", type=int, default=None,
                   help="建议 INCAR 里的 NBANDS(不给则按价电子数估计 + 余量)")
    p.add_argument("--encut", type=float, default=320.0,
                   help="建议 INCAR 里的 ENCUT (eV)")
    p.add_argument("--kpar", type=int, default=1,
                   help="建议 INCAR 里的 KPAR")


def _add_unfold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dir", required=True,
                   help="VASP 计算目录(含 EIGENVAL/PROCAR/grid_map.npz; "
                        "DOSCAR 仅 --ef auto|doscar 需要)")
    p.add_argument("--supercell", default=None,
                   help="超胞 POSCAR(默认 <dir>/POSCAR)")
    p.add_argument("--primcell", default=None,
                   help="原胞 PRIMCELL.vasp(默认 <dir>/PRIMCELL.vasp)")
    p.add_argument("--transmat", default=None,
                   help="TRANSMAT.in(默认 <dir>/TRANSMAT.in)")
    p.add_argument("--grid-map", default=None,
                   help="grid_map.npz(默认 <dir>/grid_map.npz)")
    p.add_argument("--ef", default="auto",
                   help="费米能: auto | vbm | doscar | 数值(eV)")
    p.add_argument("--delta", type=float, default=0.05,
                   help="半导体时 VBM + delta 的偏移 (eV)")
    p.add_argument("--sigma", type=float, default=0.06,
                   help="谱函数展宽 (eV)")
    p.add_argument("--strict-dominance", action="store_true",
                   help="要求每个态的单 G 主导度 >= 0.5(严格模式)")
    p.add_argument("--no-symmetrize", action="store_true",
                   help="**不做**点群轨道对称化: A(k,E) 保持逐 (K,G) 的原始"
                        "谱权重(此时费米面带对称性破缺, 见 docs/algorithms.md "
                        "§15)。默认是对称化后的(投影到对称不变子空间)")
    p.add_argument("--outdir", default="fs", help="费米面文件输出目录")


def _add_band_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--supercell", default=None,
                   help="超胞 POSCAR 路径(阶段 1 必给)")
    p.add_argument("--primcell", default=None,
                   help="原胞 PRIMCELL.vasp 路径(阶段 1 必给)")
    p.add_argument("--transmat", default=None,
                   help="TRANSMAT.in 路径(阶段 1 必给)")
    p.add_argument("--kpath", default=None,
                   help="vaspkit 格式的 KPATH.in(阶段 1: 读路径, 写 KPOINTS)")
    p.add_argument("--dir", default=None,
                   help="VASP 计算目录(阶段 2: 读 EIGENVAL/PROCAR/KPATH.in)")
    p.add_argument("--outdir", default=None,
                   help="输出目录(阶段 1 默认 run, 阶段 2 默认 bandfs)")
    p.add_argument("--nbands", type=int, default=None,
                   help="建议 INCAR 里的 NBANDS(不给则按价电子数估计 + 余量)")
    p.add_argument("--encut", type=float, default=320.0,
                   help="建议 INCAR 里的 ENCUT (eV)")
    p.add_argument("--kpar", type=int, default=1,
                   help="建议 INCAR 里的 KPAR")
    p.add_argument("--ef", default="auto",
                   help="费米能: auto | vbm | doscar | 数值(eV)"
                        "(只用于画图上的水平虚线; .grd 里存的是绝对能量)")
    p.add_argument("--delta", type=float, default=0.05,
                   help="半导体时 VBM + delta 的偏移 (eV)")
    p.add_argument("--sigma", type=float, default=0.06,
                   help="谱函数展宽 (eV); 只用于判断 E_F 是否落在带隙里")
    p.add_argument("--e-min", type=float, default=None,
                   help="图的能量下限 (eV, 绝对值)")
    p.add_argument("--e-max", type=float, default=None,
                   help="图的能量上限 (eV, 绝对值)")
    p.add_argument("--mirror", action="store_true",
                   help="把镜像路径(距离取负、顺序反过来)也画上"
                        "(对应 MATLAB 脚本里的 IsPlotMirror)")
    p.add_argument("--strict-dominance", action="store_true",
                   help="要求每个态的单 G 主导度 >= 0.5(严格模式)")
    p.add_argument("--no-symmetrize", action="store_true",
                   help="**不做**点群轨道对称化: 路径上对称相关点的展开谱权重"
                        "保持原始值(见 docs/algorithms.md §16.7)。默认是对称化"
                        "后的")
    p.add_argument("--no-plot", action="store_true",
                   help="不画 PNG(不需要 matplotlib)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unfoldfs",
        description="超胞能带反折叠 -> 原胞均匀网格上的费米面")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="生成对称约化网格与 KPOINTS")
    _add_gen_args(p_gen)
    p_gen.set_defaults(func=cmd_gen)

    p_unfold = sub.add_parser("unfold", help="反折叠 VASP 输出并写费米面文件")
    _add_unfold_args(p_unfold)
    p_unfold.set_defaults(func=cmd_unfold)

    p_band = sub.add_parser(
        "band", help="沿 k 路径做展开能带(输出 vaspkit 的四个文件)",
        description=(
            "沿 k 路径做展开能带(unfolded band structure)。\n"
            "分两个阶段, 与 gen -> VASP -> unfold 同一套路:\n"
            "  阶段 1: --supercell/--primcell/--transmat/--kpath -> "
            "KPOINTS + INCAR.suggested + KLABELS.in\n"
            "  阶段 2: --dir -> 展开 -> ENERGY.grd/MOMENTUM.grd/WEIGHT.grd/"
            "KLABELS + unfolded_band.dat/.png\n"
            "输出四个 vaspkit 格式的文件, 用户组现有的 MATLAB 脚本"
            "(unfold_band_plot_mirror.m)可以直接画。\n"
            "⚠ WEIGHT.grd 里的权重是 **Popescu 投影法的 G 分辨权重**"
            "(路径点上取原胞 k 等于该路径点的那个候选的投影权重),\n"
            "   **不是** vaspkit 282 用 WAVECAR 平面波展开得到的原胞带 |c(k)|² —— "
            "两者都叫权重但含义不同。\n"
            "⚠ 展开谱权重在晶体点群下**天然不协变**(带能量不受影响) => "
            "默认对路径上对称相关的点做轨道平均(--no-symmetrize 可关); "
            "能带曲线可信、能带形状仍不可信。"))
    _add_band_args(p_band)
    p_band.set_defaults(func=cmd_band)

    p_info = sub.add_parser("info", help="查看网格与对称信息")
    p_info.add_argument("--dir", required=True, help="含 grid_map.npz 的目录")
    p_info.add_argument("--grid-map", default=None,
                        help="grid_map.npz(默认 <dir>/grid_map.npz)")
    p_info.set_defaults(func=cmd_info)
    return parser


def _force_utf8_output() -> None:
    """把 stdout/stderr 切到 UTF-8。

    服务器上 LANG 可能是 C/POSIX, 此时 sys.stdout 的编码是 ASCII ——
    直接 print 中文会抛 UnicodeEncodeError, CLI 整个崩掉。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):      # pragma: no cover - 平台相关
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        # 顶层兜底: 缺文件 / 参数不自洽 这类**用户错误**不该甩一脸 traceback
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
