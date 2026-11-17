"""命令行接口: ``unfoldfs gen`` / ``unfoldfs unfold`` / ``unfoldfs info``。

三个子命令把整条流水线接起来:

    gen     对称原胞网格 -> KPOINTS + grid_map.npz + INCAR.suggested
    unfold  读 VASP 输出 -> 反折叠 -> 对称扩展 -> 写费米面文件
    info    查看网格与对称信息

约定(与 geometry.py / grid.py 一致):
  * k 用笛卡尔坐标, 单位 2π/Å; E 用 eV;
  * E_F 的选取按 ``--ef auto|vbm|doscar|<数值>`` 的语义(见 fsout);
  * ``--resolution`` 的单位是 2π/Å, 按 |b_i| 定义(见 contract-v2 §5)。

KPOINTS 的首行: VASP 按首行的**第一个字符**选择解析分支 —— 'G'/'K' 会走
"自动生成网格"分支(此时显式 k 点列表被忽略, VASP 自己生成 k 点, 个数与
顺序都变), 'S'/'C'/'R' 分别是不可约点/笛卡尔/倒格矢分支。故这里把首行写成
**纯注释**('P' 开头, 不落在任何分支上), 并把 G/M 标签挪进注释里 ——
零成本, 但保证 VASP 走"显式 k 点列表"分支。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from .expand import build_spectral_function, expand_to_full_grid
from .fsout import (band_edges, make_e_grid, select_fermi_energy, write_bxsf,
                    write_cube, write_fermisurfer_bands,
                    write_spectral_npz)
from .geometry import (common_symmetry_ops, read_poscar, read_transmat,
                       reciprocal, resolve_transform)
from .grid import (build_grid_map, choose_grid_points, filter_ops_by_grid,
                   load_grid_map, primitive_grid_points, save_grid_map,
                   write_kpoints)
from .unfold import build_atom_copies, unfold_supercell
from .vaspio import read_doscar_fermi, read_eigenval, read_procar

__all__ = ["main", "cmd_gen", "cmd_unfold", "cmd_info"]

# 谱函数 npz 的能量窗: 以 E_F 为中心 ±_NPZ_HALF_WIDTH eV, 含 E_F 本身
_NPZ_HALF_WIDTH = 0.5
_NPZ_N_POINTS = 401

# 带隙判据(E_F 语义里的"金属 vs 半导体"), 与 fsout 的默认值一致
_GAP_TOL = 0.05

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


def _write_kpoints_explicit(path: str, grid_map) -> None:
    """写 KPOINTS(显式 k 点列表)。

    k 点与权重由 ``grid.write_kpoints`` 给出(唯一的事实来源), 这里只把**首行**
    换成纯注释形式: VASP 按首行第一个字符选解析分支, 'G'/'K' 会让 VASP 自己
    生成 k 点(显式列表被忽略)。'P' 开头不落在任何分支上 -> 走显式列表分支。
    """
    write_kpoints(path, grid_map)
    with open(path) as fh:
        lines = fh.read().splitlines()
    tag = "G" if grid_map.spec.grid_type == "gamma" else "M"
    p = grid_map.spec.p
    lines[0] = (f"Parameters to Generate KPOINTS (Don't Edit This Line) -- "
                f"explicit k-point list, {tag} grid {p[0]}x{p[1]}x{p[2]}")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _incar_suggested(grid_map, n_bands: int, encut: float, kpar: int,
                     unknown_species: Sequence[str]) -> str:
    """给出建议的 INCAR 文本。

    其中 ``ISYM = 0`` 与 ``LORBIT = 12`` 是**硬要求**:
      * ISYM=0: 否则 VASP 把显式 k 点表按对称性归约到不可约部分, k 点的个数
        与顺序都会变, 反折叠的映射随之失效;
      * LORBIT=12: PROCAR 必须带复投影系数(相位), LORBIT=11 不写相位。
    """
    p = grid_map.spec.p
    head = [
        "# unfoldfs 建议的 INCAR",
        f"# 网格: {p[0]} x {p[1]} x {p[2]} ({grid_map.spec.grid_type}), "
        f"实际计算 {grid_map.n_k} 个超胞 k 点",
        f"# 超胞变换: det(M) = {grid_map.transform.det}",
        "#",
        "# ISYM = 0 必须保留: 否则 VASP 会把显式 k 点表按对称性归约, "
        "k 点的个数与顺序都会变,",
        "#   反折叠的 (k 点 -> 候选 G) 映射随之失效。",
        "# LORBIT = 12 必须保留: PROCAR 需要复投影系数(相位), "
        "LORBIT=11 不写相位。",
        "# ISMEAR = 1 + SIGMA = 0.2 是 VASP 默认值: 每个 k 点只需几次 DAV "
        "迭代(实测单核机器上 ~15 s/k 点)。",
        "#   半导体若要更锐利的费米能, 改成 ISMEAR = 0 + SIGMA = 0.05, "
        "但收敛会慢一个量级。",
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

def cmd_gen(args: argparse.Namespace) -> int:
    """对称约化网格 -> KPOINTS / grid_map.npz / INCAR.suggested。"""
    prim = read_poscar(args.primcell)
    sc = read_poscar(args.supercell)
    tr = resolve_transform(sc, prim, read_transmat(args.transmat))
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    spec = choose_grid_points(prim.lattice, tr.m_eff, args.resolution,
                              args.grid_type)
    grid = primitive_grid_points(spec)
    ops = filter_ops_by_grid(common_symmetry_ops(prim, tr.m_eff), grid)
    gm = build_grid_map(spec, grid, ops, tr, b_prim, b_super)

    os.makedirs(args.outdir, exist_ok=True)
    kpoints_path = os.path.join(args.outdir, "KPOINTS")
    gridmap_path = os.path.join(args.outdir, "grid_map.npz")
    incar_path = os.path.join(args.outdir, "INCAR.suggested")
    _write_kpoints_explicit(kpoints_path, gm)
    save_grid_map(gridmap_path, gm)

    nelect, n_bands_prim, unknown = _valence_estimate(prim.species)
    if args.nbands is not None:
        n_bands = int(args.nbands)
    else:
        # 超胞的占据带数 = det(M) x 原胞的占据带数; 少一条带就会截断占据态
        n_bands = int(tr.det) * max(n_bands_prim, 1)
    with open(incar_path, "w", newline="\n") as fh:
        fh.write(_incar_suggested(gm, n_bands, args.encut, args.kpar, unknown))

    print(f"网格规格      : {spec.p[0]} x {spec.p[1]} x {spec.p[2]} "
          f"({spec.grid_type}), 满网格 {spec.n_points} 点")
    print(f"对称不可约点  : {gm.n_ir}")
    print(f"实际计算 k 点 : {gm.n_k} (超胞分数坐标), 权重和 = "
          f"{int(gm.k_weight.sum())} == 满网格点数")
    print(f"超胞变换      : det(M) = {tr.det}")
    print(f"估计电子数    : {nelect} (价电子数估计), NBANDS = {n_bands}")
    print(f"已写出        : {kpoints_path}")
    print(f"已写出        : {gridmap_path}")
    print(f"已写出        : {incar_path}")
    return 0


# ------------------------------------------------------------------ unfold

def cmd_unfold(args: argparse.Namespace) -> int:
    """读 VASP 输出 -> 反折叠 -> 对称扩展 -> 写费米面文件。"""
    run_dir = args.dir
    supercell = args.supercell or os.path.join(run_dir, "POSCAR")
    primcell = args.primcell or os.path.join(run_dir, "PRIMCELL.vasp")
    transmat = args.transmat or os.path.join(run_dir, "TRANSMAT.in")
    gridmap_path = args.grid_map or os.path.join(run_dir, "grid_map.npz")

    gm = load_grid_map(gridmap_path)
    band = read_eigenval(os.path.join(run_dir, "EIGENVAL"))

    # k 点数守卫: 比**真实**的 k 点数(EIGENVAL 里实际的 k 点块数), 不能拿
    # 截断后的自身比 —— 那样永远相等, 守卫形同虚设。
    n_calc = band.kpoints.shape[0]
    if n_calc != gm.n_k:
        raise ValueError(
            f"VASP 算出的 k 点数 {n_calc} != grid_map 记录 {gm.n_k}。"
            f"请确认 KPOINTS 用的是 gen 写出的文件, 且 INCAR 里 ISYM=0。")

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
    copies = build_atom_copies(prim, sc, tr)
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    # 展开必须传 gm.k_super(去重后的超胞 k), 不是 IR 点列表
    unfolded = unfold_supercell(
        band, procar, tr, copies, tr.m_eff, prim.lattice,
        prim.positions @ prim.lattice, b_prim, b_super, gm.k_super,
        strict_dominance=args.strict_dominance)
    expanded = expand_to_full_grid(unfolded, gm)

    # VBM/CBM 只在 EIGENVAL 里有占据态时才有意义; --ef <数值> 不该被它挡住
    try:
        vbm, cbm = band_edges(band)
        vbm_txt = f"{vbm:.6f} / {cbm:.6f} eV"
    except ValueError as exc:
        vbm, cbm = float("nan"), float("nan")
        vbm_txt = f"不可用 ({exc})"
    e_f = select_fermi_energy(dos_ef, band, mode=args.ef, offset=args.delta,
                               gap_tol=_GAP_TOL)
    print(f"相位约定      : s1 = {unfolded.s1:+d}, s2 = {unfolded.s2:+d} "
          f"(单 G 主导度 {unfolded.dominance:.4f})")
    print(f"VBM / CBM     : {vbm_txt}")
    print(f"E_F           : {e_f:.6f} eV  (--ef {args.ef}, "
          f"--delta {args.delta:g})")

    os.makedirs(args.outdir, exist_ok=True)
    values = build_spectral_function(expanded, np.array([e_f]),
                                     args.sigma)[:, 0]
    comments = _fs_comments(e_f, args, vbm, cbm)

    cube_path = os.path.join(args.outdir, "FS.cube")
    bxsf_path = os.path.join(args.outdir, "FS.bxsf")
    npz_path = os.path.join(args.outdir, "A(k,E).npz")
    write_cube(cube_path, expanded, b_prim, values, comments=comments)
    print(f"已写出        : {cube_path}")
    # BXSF 只支持 Gamma 网格(见 docs/formats.md §3): MP 网格下跳过它,
    # 但把原因说清楚 —— 其余输出(FS.cube / band.N / npz)对 MP 网格都成立
    try:
        write_bxsf(bxsf_path, expanded, b_prim, e_f, comments=comments)
    except ValueError as exc:
        print(f"已跳过        : {bxsf_path} ({exc})")
    else:
        print(f"已写出        : {bxsf_path}")
    band_paths = write_fermisurfer_bands(args.outdir, expanded, b_prim, e_f)
    print(f"已写出        : {os.path.join(args.outdir, 'band.1')} ... "
          f"band.{len(band_paths)}")
    e_grid = make_e_grid(e_f, _NPZ_HALF_WIDTH, _NPZ_N_POINTS)
    write_spectral_npz(npz_path, expanded, b_prim, e_f, vbm=vbm, cbm=cbm,
                       e_grid=e_grid, sigma=args.sigma)
    print(f"已写出        : {npz_path}")
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


# ------------------------------------------------------------------ info

def cmd_info(args: argparse.Namespace) -> int:
    """打印网格规模、对称性与约化率。"""
    gridmap_path = args.grid_map or os.path.join(args.dir, "grid_map.npz")
    gm = load_grid_map(gridmap_path)
    p = gm.spec.p
    ratio = gm.n_k / float(gm.spec.n_points)
    print(f"网格规格      : {p[0]} x {p[1]} x {p[2]} ({gm.spec.grid_type})")
    print(f"满网格点数    : {gm.spec.n_points}")
    print(f"对称不可约点  : {gm.n_ir}")
    print(f"实际计算 k 点 : {gm.n_k} (超胞分数坐标)")
    print(f"约化率        : {gm.n_k}/{gm.spec.n_points} = {ratio:.4f} "
          f"(约化 {gm.spec.n_points / float(gm.n_k):.2f} 倍)")
    print(f"超胞变换      : det(M) = {gm.transform.det}")
    print(f"KPOINTS 权重和: {int(gm.k_weight.sum())} (应等于满网格点数 "
          f"{gm.spec.n_points})")
    return 0


# ------------------------------------------------------------------ 参数解析

def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--supercell", required=True, help="超胞 POSCAR 路径")
    p.add_argument("--primcell", required=True, help="原胞 PRIMCELL.vasp 路径")
    p.add_argument("--transmat", required=True, help="TRANSMAT.in 路径")
    p.add_argument("--resolution", type=float, default=0.02,
                   help="k 网格分辨率, 单位 2π/Å, 按 |b_i| 定义")
    p.add_argument("--grid-type", choices=("gamma", "mp"), default="gamma",
                   help="网格类型: gamma(Gamma 居中) 或 mp(Monkhorst-Pack)")
    p.add_argument("--outdir", default=".", help="输出目录")
    p.add_argument("--nbands", type=int, default=None,
                   help="建议 INCAR 里的 NBANDS(不给则按价电子数估计)")
    p.add_argument("--encut", type=float, default=320.0,
                   help="建议 INCAR 里的 ENCUT (eV)")
    p.add_argument("--kpar", type=int, default=1,
                   help="建议 INCAR 里的 KPAR")


def _add_unfold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dir", required=True,
                   help="VASP 计算目录(含 EIGENVAL/PROCAR/DOSCAR/grid_map.npz)")
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
    p.add_argument("--outdir", default="fs", help="费米面文件输出目录")


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
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
