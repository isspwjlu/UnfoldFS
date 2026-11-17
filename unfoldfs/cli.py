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
from typing import List, Optional, Sequence

import numpy as np

from .expand import build_spectral_function, expand_to_full_grid
from .fsout import (band_edges, make_e_grid, select_fermi_energy, write_bxsf,
                    write_cube, write_fermisurfer_bands,
                    write_spectral_npz)
from .geometry import (common_symmetry_ops, read_poscar, read_transmat,
                       reciprocal, resolve_transform)
from .grid import (GridSpec, build_grid_map, choose_grid_points,
                   filter_ops_by_grid, load_grid_map, primitive_grid_points,
                   save_grid_map, write_kpoints)
from .unfold import build_atom_copies, unfold_supercell
from .vaspio import read_doscar_fermi, read_eigenval, read_procar

__all__ = ["main", "cmd_gen", "cmd_unfold", "cmd_info"]

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
                            m_eff: np.ndarray) -> Optional[int]:
    """朴素超胞网格的点数 ∏(P_i / M_ii)(M 非对角或不可整除时返回 None)。

    这是"不利用对称性时, 超胞 BZ 里 P_i/M_i 个点"的那一套 —— KPOINTS 里
    写的其实就是它(权重把对称等价点合并进同一个 k)。
    """
    m = np.asarray(m_eff, dtype=float)
    if not np.allclose(m, np.diag(np.diag(m))):
        return None
    diag = np.diag(m)
    if np.any(np.abs(diag - np.rint(diag)) > 1e-8):
        return None
    diag = np.rint(diag).astype(int)
    if np.any(diag <= 0):
        return None
    if any(int(p[i]) % int(diag[i]) != 0 for i in range(3)):
        return None
    return int(np.prod([int(p[i]) // int(diag[i]) for i in range(3)]))


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


def _incar_suggested(grid_map, n_bands: int, encut: float, kpar: int,
                     unknown_species: Sequence[str], n_occ_est: int) -> str:
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
        "#   展宽只影响带边占据数(带隙体系的带边占据数必然是 0.98 这类小数), "
        "并不能让费米能更准。",
        "#   有带隙的体系更推荐 ISMEAR = -5(四面体法): 占据数接近整数, "
        "没有 DAV 收敛问题;",
        "#   若改用 ISMEAR = -5, 请同时删掉 SIGMA(四面体法不接受展宽)。",
        f"# NBANDS = {n_bands} = 估计占据带数 {n_occ_est} + 空带余量 "
        f"{n_bands - n_occ_est}(至少 10 条)。",
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
    ops = filter_ops_by_grid(common_symmetry_ops(prim, tr.m_eff), grid)
    gm = build_grid_map(spec, grid, ops, tr, b_prim, b_super)

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
        fh.write(_incar_suggested(gm, n_bands, args.encut, args.kpar, unknown,
                                  n_occ_est))

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
    copies = build_atom_copies(prim, sc, tr)
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    # E_F 的选取与自检放在反折叠**之前**: 落在带隙深处时要在昂贵的计算之前
    # 就中止(而不是让用户等完再被告知结果没有意义)。
    # VBM/CBM 只在 EIGENVAL 里有占据态时才有意义; --ef <数值> 不该被它挡住
    try:
        vbm, cbm = band_edges(band)
        vbm_txt = f"{vbm:.6f} / {cbm:.6f} eV"
    except ValueError as exc:
        vbm, cbm = float("nan"), float("nan")
        vbm_txt = f"不可用 ({exc})"
    e_f = select_fermi_energy(dos_ef, band, mode=args.ef, offset=args.delta,
                               gap_tol=_GAP_TOL)

    # ★ 自检: 选出的 E_F 落在带隙里 => 该能量上没有任何态 => 费米面是空的。
    # 这是"默认路径静默产出空费米面"的最后一道防线(不依赖 is_metal 的判定)。
    numeric_ef = _is_numeric_ef(args.ef)
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
    band_paths = write_fermisurfer_bands(args.outdir, expanded, b_prim, e_f)
    print(f"已写出        : {os.path.join(args.outdir, 'band.1')} ... "
          f"band.{len(band_paths)}")
    e_grid = make_e_grid(e_f, _NPZ_HALF_WIDTH, _NPZ_N_POINTS)
    write_spectral_npz(npz_path, expanded, b_prim, e_f, vbm=vbm, cbm=cbm,
                       e_grid=e_grid, sigma=args.sigma)
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


# ------------------------------------------------------------------ info

def cmd_info(args: argparse.Namespace) -> int:
    """打印网格规模、对称性与约化率。"""
    gridmap_path = args.grid_map or os.path.join(args.dir, "grid_map.npz")
    gm = load_grid_map(gridmap_path)
    p = gm.spec.p
    ratio = gm.n_k / float(gm.spec.n_points)
    naive = _naive_supercell_count(p, gm.transform.m_eff)
    naive_txt = ("n/a (变换矩阵不是对角的)" if naive is None
                 else f"{naive} 点 (∏(P_i/M_i))")
    print(f"网格规格      : {p[0]} x {p[1]} x {p[2]} ({gm.spec.grid_type})")
    print(f"满网格点数    : {gm.spec.n_points}")
    print(f"对称不可约点  : {gm.n_ir}")
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
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        # 顶层兜底: 缺文件 / 参数不自洽 这类**用户错误**不该甩一脸 traceback
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
