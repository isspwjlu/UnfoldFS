"""CLI 测试: unfoldfs gen / unfold / info。

**为什么用合成的小体系而不是 tests/data 的 fixture**:
fixture 是 3x3x1 MoS2 超胞, EIGENVAL 只有 2 个 k 点, 而"满网格 -> 超胞 k"
的折叠要求 k 点集合来自某个均匀网格 —— M=diag(3,3,1) 下均匀网格折出的超胞 k
的第 1 个分量只能是 j/(P1/3), 凑不出 fixture 的 (0.1, 0, 0)。故这里造一个
2 原子的正交原胞 + 3x1x1 超胞, 并**按设计**生成 PROCAR 的复投影系数, 使
相位约定有明确的区分度(见 _U / _THETA 的注释)。

设计(与 scratch 里的数值搜索同源):
  * 原胞 A = diag(3, 4, 5) Å, 2 个原子 (0,0,0) 与 (1/2,1/2,1/2);
  * M = diag(3,1,1) -> 超胞 6 个原子, 每个原胞原子 3 个副本;
  * 网格 P = (3,1,1) -> 满网格 3 点, 折成 1 个超胞 k (Gamma), det(M)=3
    个候选原胞 k;
  * 复投影系数取 c[band, ion] = Atilde_i(theta_b) * u_j: 副本相位沿 a 方向
    差 2pi/3, 因此 s1 = +/-1 给出不同的权重分布(主导度 0.655 vs 0.594,
    相对差 9% —— 远离退化区, 不会触发"约定由噪声决定"的告警)。
"""
import ast
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from unfoldfs import cli
from unfoldfs.expand import point_group_orbits
from unfoldfs.fsout import read_bxsf, read_cube, read_fermisurfer_band
from unfoldfs.geometry import (common_symmetry_ops_detailed,
                               periodic_distance, primitive_symmetry_ops,
                               read_poscar,
                               read_transmat, reciprocal, resolve_transform)
from unfoldfs.grid import (GridSpec, build_grid_map,
                           filter_ops_by_grid_detailed, load_grid_map,
                           primitive_grid_points, save_grid_map,
                           write_kpoints)
from unfoldfs.vaspio import read_eigenval, read_procar
from tests.conftest import FIXTURE_EIGENVAL

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 合成体系

_A_PRIM = np.diag([3.0, 4.0, 5.0])
_M = np.diag([3, 1, 1])
_PRIM_FRAC = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
_BAND_ENERGIES = (-10.0, -6.0, -2.0, 1.0)
_DOS_EF = -5.0

# 真金属: 末两条带在 DOSCAR 的 E_F 处简并(占据 0.6 / 0) -> VBM == CBM,
# 带隙为 0。这才是"金属"(不是"有带隙 + 带边分数占据")。
_METAL_BAND_ENERGIES = (-10.0, -6.0, _DOS_EF, _DOS_EF)

# 副本相位模式 u: 让 s1=+1/-1 的单 G 主导度明显不同
_U = np.array([0.1027125684732821 - 0.21014635803702303j,
                0.23240310131893024 - 0.7116491117596169j,
                0.5501292501189036 + 1.001533391119041j])
# 每个带的原子振幅相位(让不同带落到不同的候选 G 上)
_THETA = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
# 均匀网格: 3 个满网格点折成 1 个超胞 k
_SPEC = GridSpec(p=(3, 1, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))


def _write_poscar(path, title, lattice, frac, species, counts):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(title + "\n")
        fh.write("   1.000000000000000\n")
        for row in lattice:
            fh.write("  %22.16f %22.16f %22.16f\n" % tuple(row))
        fh.write("  " + " ".join(species) + "\n")
        fh.write("  " + " ".join(str(c) for c in counts) + "\n")
        fh.write("Direct\n")
        for p in frac:
            fh.write("  %22.16f %22.16f %22.16f\n" % tuple(p))


def _supercell_frac(m=_M):
    """超胞 6 个原子的分数坐标(原胞原子 x 沿 a 的 3 个副本)。

    m = 单位阵时就是原胞自己(2 个原子, 无副本) —— 用于 det(M)=1 的直通模式。
    """
    n_copies = int(round(abs(np.linalg.det(np.asarray(m, dtype=float)))))
    a_sc = np.asarray(m, dtype=float) @ _A_PRIM
    out = []
    for p in _PRIM_FRAC @ _A_PRIM:
        for j in range(n_copies):
            out.append((p + j * _A_PRIM[0]) @ np.linalg.inv(a_sc))
    return np.array(out)


def _write_mini_system(directory, occupations=None, band_energies=None,
                       k_points=None, occ_per_k=None, dos_ef=None, m=None):
    """写出一个自洽的迷你 VASP 计算目录。

    occupations: 4 个带的占据数(默认 1,1,0,0 -> 半导体);
        给 (1, 1, 0.4, 0) 则出现**带边分数占据**(展宽的产物)。
    band_energies: 4 个带的能量(默认 _BAND_ENERGIES);
        给 _METAL_BAND_ENERGIES 则得到零带隙的**真金属**;
        给**每个 k 一行**的二维序列时, 逐 k 用各自的能量。
    k_points: 超胞分数坐标的 k 点列表(默认只有 Gamma 一个点)。
        **多 k 点**才能测出"某带的占据在 k 上跨过阈值"(费米穿越)。
    occ_per_k: 每个 k 点一组占据数(给了 k_points 时必须给)。
    dos_ef: DOSCAR 里的 E_F(默认 _DOS_EF)。
    m: 超胞变换矩阵(默认 _M = diag(3,1,1))。给**单位阵**时"超胞"就是原胞
        —— det(M)=1, 用于原胞直通模式。
    """
    os.makedirs(directory, exist_ok=True)
    m = _M if m is None else np.asarray(m, dtype=int)
    n_copies = int(round(abs(np.linalg.det(m.astype(float)))))
    n_ions = len(_PRIM_FRAC) * n_copies
    occ = occupations if occupations is not None else (1.0, 1.0, 0.0, 0.0)
    energies = band_energies if band_energies is not None else _BAND_ENERGIES
    kpts = [(0.0, 0.0, 0.0)] if k_points is None \
        else [(float(a), float(b), float(c)) for a, b, c in k_points]
    occ_rows = [tuple(occ)] * len(kpts) if occ_per_k is None \
        else [tuple(o) for o in occ_per_k]
    assert len(occ_rows) == len(kpts)
    if energies and isinstance(energies[0], (list, tuple)):
        energies_per_k = [tuple(e) for e in energies]
    else:
        energies_per_k = [tuple(energies)] * len(kpts)
    assert len(energies_per_k) == len(kpts)
    energies = energies_per_k[0]
    ef = _DOS_EF if dos_ef is None else float(dos_ef)
    a_sc = m @ _A_PRIM
    _write_poscar(os.path.join(directory, "POSCAR"), "synthetic supercell",
                  a_sc, _supercell_frac(m), ["Si"], [n_ions])
    _write_poscar(os.path.join(directory, "PRIMCELL.vasp"),
                  "synthetic primitive", _A_PRIM, _PRIM_FRAC, ["Si"], [2])
    with open(os.path.join(directory, "TRANSMAT.in"), "w",
              encoding="utf-8") as fh:
        fh.write("Read transformation matrix from the TRANSMAT.in file "
                 "if it exists.\n")
        for row in m:
            fh.write("  %d  %d  %d   # must be three integers\n" % tuple(row))

    with open(os.path.join(directory, "EIGENVAL"), "w",
              encoding="utf-8") as fh:
        fh.write("   %d   %d    1    1\n" % (n_ions, n_ions))
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic mini system\n")
        fh.write("    2      %d    %d\n" % (len(kpts), len(energies)))
        fh.write("\n")
        for ik, kfrac in enumerate(kpts):
            fh.write("  %14.7E %14.7E %14.7E %14.7E\n"
                     % (kfrac[0], kfrac[1], kfrac[2], 1.0))
            for b, e in enumerate(energies_per_k[ik]):
                fh.write("  %5d %14.6f %10.6f\n"
                         % (b + 1, e, occ_rows[ik][b]))
            fh.write("\n")

    with open(os.path.join(directory, "DOSCAR"), "w", encoding="utf-8") as fh:
        fh.write("   %d   %d   1   0\n" % (n_ions, n_ions))
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic mini system\n")
        fh.write("  1.000000000000000E+01 -1.000000000000000E+01  301"
                 " %20.10E  1.000000000000000E+00\n" % ef)
        for i in range(301):
            fh.write("  %14.6f 0.0000E+00 0.0000E+00\n"
                     % (-10.0 + i * 20.0 / 300.0))

    coeffs = np.zeros((len(energies), n_ions, 1), dtype=complex)
    for b, theta in enumerate(_THETA[:len(energies)]):
        at = np.array([1.0 + 0j, np.exp(1j * theta)])
        for i in range(len(_PRIM_FRAC)):
            for j in range(n_copies):
                coeffs[b, i * n_copies + j, 0] = at[i] * _U[j]
    with open(os.path.join(directory, "PROCAR"), "w", encoding="utf-8") as fh:
        fh.write("PROCAR lm decomposed + phase\n")
        fh.write("# of k-points: %3d         # of bands:  %d"
                 "         # of ions: %4d\n"
                 % (len(kpts), len(energies), n_ions))
        fh.write("\n")
        for ik, kfrac in enumerate(kpts):
            fh.write(" k-point %5d :    %.8f %.8f %.8f"
                     "     weight = 1.00000000\n"
                     % (ik + 1, kfrac[0], kfrac[1], kfrac[2]))
            for b, e in enumerate(energies_per_k[ik]):
                fh.write("\n")
                fh.write("band  %5d # energy  %14.8f # occ.  1.00000000\n"
                         % (b + 1, e))
                fh.write("\n")
                fh.write("ion      s     tot\n")
                tot = 0.0
                for i in range(n_ions):
                    m2 = abs(coeffs[b, i, 0]) ** 2
                    tot += m2
                    fh.write("  %3d  %12.8f  %12.8f\n" % (i + 1, m2, m2))
                fh.write("tot   %12.8f  %12.8f\n" % (tot, tot))
                fh.write("ion      s\n")
                for i in range(n_ions):
                    c = coeffs[b, i, 0]
                    fh.write("  %3d %14.8f %14.8f %14.8f\n"
                             % (i + 1, c.real, c.imag, abs(c) ** 2))
    return directory


def _write_grid_map(directory, spec=_SPEC):
    """写出 grid_map.npz(均匀网格 -> 超胞 k 的映射)。"""
    prim = read_poscar(os.path.join(directory, "PRIMCELL.vasp"))
    sc = read_poscar(os.path.join(directory, "POSCAR"))
    tr = resolve_transform(sc, prim,
                           read_transmat(os.path.join(directory, "TRANSMAT.in")))
    grid = primitive_grid_points(spec)
    all_ops, dropped = common_symmetry_ops_detailed(prim, tr.m_eff)
    ops, dropped_grid = filter_ops_by_grid_detailed(all_ops, grid)
    gm = build_grid_map(spec, grid, ops, tr, reciprocal(prim.lattice),
                        reciprocal(sc.lattice),
                        dropped=tuple(dropped) + tuple(dropped_grid))
    save_grid_map(os.path.join(directory, "grid_map.npz"), gm)
    return gm


@pytest.fixture()
def mini_run(tmp_path):
    """一个自洽的迷你 VASP 计算目录(含 grid_map.npz)。"""
    directory = tmp_path / "run"
    _write_mini_system(str(directory))
    _write_grid_map(str(directory))
    return directory


def _read_kpoints(path):
    """解析 KPOINTS -> (首行, k 点列表[(k, 权重)])。"""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n = int(lines[1].split()[0])
    pts = []
    for line in lines[3:3 + n]:
        f = line.split()
        pts.append(([float(x) for x in f[:3]], float(f[3])))
    return lines[0], pts


def _parse_incar(text):
    """把 INCAR 文本解析成 {tag: value}(**跳过 '#' 注释行**)。

    断言必须看**生效值**而不是子串: INCAR.suggested 的注释里就写着
    "ISYM = 0 必须保留" —— 只做子串匹配的话, 把真正的 ISYM = 0 删掉
    测试照样通过(评审变异 M4/M5 就是这样活下来的)。
    """
    tags = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        tag, _, value = s.partition("=")
        tags[tag.strip().upper()] = value.strip()
    return tags


def _gen_args(src, out, *extra):
    return ["gen",
            "--supercell", os.path.join(src, "POSCAR"),
            "--primcell", os.path.join(src, "PRIMCELL.vasp"),
            "--transmat", os.path.join(src, "TRANSMAT.in"),
            "--outdir", str(out)] + list(extra)


# ------------------------------------------------------------------ gen

def test_gen_writes_kpoints_gridmap_and_incar(tmp_path):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    rc = cli.main(_gen_args(src, out, "--resolution", "1.6"))
    assert rc == 0
    assert (out / "KPOINTS").is_file()
    assert (out / "grid_map.npz").is_file()
    assert (out / "INCAR.suggested").is_file()
    incar = (out / "INCAR.suggested").read_text(encoding="utf-8")
    tags = _parse_incar(incar)
    # 生效值(不是子串匹配 —— 注释里也有 "ISYM = 0" 这几个字)
    assert tags["ISYM"] == "0"
    assert tags["LORBIT"] == "12"
    assert tags["ISMEAR"] == "1"
    assert tags["SIGMA"] == "0.2"
    assert tags["LWAVE"] == ".TRUE."


def test_gen_incar_does_not_recommend_the_tetrahedron_method(tmp_path):
    """Ruling 29: 不得推荐四面体法(ISMEAR <= -4)—— 它与本工具的 KPOINTS 互斥。

    VASP 6.6.0 实测: ISMEAR <= -4 时 VASP **不使用**显式 k 点列表, 而是要求
    从 KPOINTS 的**首行**自己解析 MP 网格(首行须为 'M P G X Y Z SHIFT'),
    而本工具的 KPOINTS 首行是纯注释(显式 k 点列表)
    => VASP 报 'Error reading KPOINTS file, at line: N+1'。
    """
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    incar = (out / "INCAR.suggested").read_text(encoding="utf-8")
    assert "ISMEAR = -5" not in incar
    # 注释里必须把"为什么不能用"说清楚
    assert "四面体" in incar
    assert "KPOINTS" in incar
    tags = _parse_incar(incar)
    assert tags["ISMEAR"] == "1"
    assert tags["SIGMA"] == "0.2"


def test_gen_incar_kpoints_is_free_of_the_vasp_auto_grid_wording(tmp_path):
    """M7: 首行别再用 "Parameters to Generate KPOINTS ..." 的措辞 ——
    这份文件是**显式 k 点列表**, 不是自动网格。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    first, _ = _read_kpoints(str(out / "KPOINTS"))
    assert "Parameters to Generate KPOINTS" not in first
    assert first.lower().startswith("unfoldfs")


def test_gen_kpoints_first_line_comes_from_grid_write_kpoints(tmp_path):
    """M2: KPOINTS 首行的唯一事实来源是 grid.write_kpoints(explicit=True),
    不是 CLI 里再改一遍。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    gm = load_grid_map(str(out / "grid_map.npz"))
    direct = str(tmp_path / "direct_KPOINTS")
    write_kpoints(direct, gm, explicit=True)
    with open(direct, encoding="utf-8") as fh:
        direct_first = fh.read().splitlines()[0]
    first, _ = _read_kpoints(str(out / "KPOINTS"))
    assert first == direct_first


def test_gen_kpoints_weight_sum_equals_full_grid_point_count(tmp_path):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    gm = load_grid_map(str(out / "grid_map.npz"))
    _, pts = _read_kpoints(str(out / "KPOINTS"))
    assert len(pts) == gm.n_k
    assert sum(w for _, w in pts) == pytest.approx(gm.spec.n_points, rel=0, abs=0)


def test_gen_kpoints_first_line_is_not_a_vasp_branch_keyword(tmp_path):
    """首行不能以 G/K/S/C/R 开头 —— 否则 VASP 会走"自动生成网格"等分支,
    把显式 k 点列表忽略掉(k 点的个数与顺序都会变)。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    first, _ = _read_kpoints(str(out / "KPOINTS"))
    assert first[0] not in "GgKkSsCcRr"


def test_gen_kpoints_has_no_duplicate_k_point(tmp_path):
    """`gen` 写出的 KPOINTS 也**不许有重复 k 点**(修复轮 7 / B1)。

    VASP 6.6.0 读到含重复 k 点的显式列表会崩(服务器实测 2026-11-18: 每次
    都在第一个 k 点算完后死掉)。网格的 k 点天然唯一 —— 这里把它**锁住**,
    免得将来有人改折叠/去重的逻辑时把它弄坏。
    """
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    _first, pts = _read_kpoints(str(out / "KPOINTS"))
    k = np.array([p[0] for p in pts])
    d = periodic_distance(k, k)
    assert np.all(d[~np.eye(len(k), dtype=bool)] > 1e-6)


def test_gen_grid_map_round_trips(tmp_path):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    gm = load_grid_map(str(out / "grid_map.npz"))
    # M = diag(3,1,1) ⇒ 逐维约束: 只有第 1 维必须是 3 的倍数, 第 2/3 维
    # (没被放大)不受约束(旧的全局 LCM 会把它们也强制成 3 的倍数)。
    assert gm.spec.p == (3, 1, 1)
    assert gm.n_k == 1
    assert gm.n_ir >= 1
    assert int(gm.k_weight.sum()) == gm.spec.n_points
    # 再存一次再读回来, 数组逐位相同
    again = str(tmp_path / "gm2.npz")
    save_grid_map(again, gm)
    gm2 = load_grid_map(again)
    assert np.array_equal(gm.k_super, gm2.k_super)
    assert np.array_equal(gm.full_to_k, gm2.full_to_k)
    assert np.array_equal(gm.full_to_ir, gm2.full_to_ir)
    assert np.array_equal(gm.rot_of_full, gm2.rot_of_full)
    assert np.array_equal(gm.ir_k_prim, gm2.ir_k_prim)
    assert np.array_equal(gm.b_prim, gm2.b_prim)


def test_gen_copies_the_structure_files(tmp_path):
    """M5: gen 顺手把 POSCAR / PRIMCELL.vasp / TRANSMAT.in 复制进 outdir ——
    unfold 默认从 --dir 里找这三个文件。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    for name in ("POSCAR", "PRIMCELL.vasp", "TRANSMAT.in"):
        assert (out / name).is_file(), name


def test_gen_module_entrypoint_runs(tmp_path):
    """`python -m unfoldfs.cli` 也要能跑(不依赖打包)。"""
    src = _write_mini_system(str(tmp_path / "src"))
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "unfoldfs.cli", "gen",
         "--supercell", os.path.join(src, "POSCAR"),
         "--primcell", os.path.join(src, "PRIMCELL.vasp"),
         "--transmat", os.path.join(src, "TRANSMAT.in"),
         "--resolution", "1.6", "--outdir", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert (out / "KPOINTS").is_file()


# ------------------------------------------------------------------ --grid

def test_gen_accepts_an_explicit_grid(tmp_path):
    """I3: --grid 绕过分辨率规则, 直接点名网格(简报要的 6x6x1 因此可达)。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    rc = cli.main(_gen_args(src, out, "--grid", "6", "6", "1"))
    assert rc == 0
    gm = load_grid_map(str(out / "grid_map.npz"))
    assert gm.spec.p == (6, 6, 1)
    assert gm.spec.n_points == 36
    # 自洽性: n_points == n_k · det(M)
    assert gm.spec.n_points == gm.n_k * int(round(gm.transform.det))
    _, pts = _read_kpoints(str(out / "KPOINTS"))
    assert len(pts) == gm.n_k
    assert sum(w for _, w in pts) == gm.spec.n_points


def test_gen_rejects_an_inconsistent_explicit_grid(tmp_path):
    """不自洽的网格必须报错并说清原因(而不是写出一份坏文件)。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    rc = cli.main(_gen_args(src, out, "--grid", "2", "2", "2"))
    assert rc != 0
    assert not (out / "KPOINTS").exists()


def test_gen_grid_and_resolution_are_mutually_exclusive(tmp_path, capsys):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    rc = cli.main(_gen_args(src, out, "--grid", "6", "6", "1",
                             "--resolution", "1.6"))
    assert rc != 0
    err = capsys.readouterr().err
    assert "互斥" in err


# ------------------------------------------------------------------ NBANDS

def test_gen_nbands_keeps_empty_band_margin(tmp_path):
    """I4: NBANDS 必须比估计占据带数多出空带余量(零余量会把占据态截断)。"""
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    tags = _parse_incar((out / "INCAR.suggested").read_text(encoding="utf-8"))
    n_bands = int(tags["NBANDS"])
    # 迷你体系: 2 个 Si(各 4 个价电子) -> 原胞 4 条带; det(M)=3 -> 12 条占据带
    n_occ_est = 3 * 4
    assert n_bands > n_occ_est
    assert n_bands == n_occ_est + max(10, -(-n_occ_est // 10))


def test_gen_nbands_can_be_overridden(tmp_path):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6", "--nbands", "77"))
    tags = _parse_incar((out / "INCAR.suggested").read_text(encoding="utf-8"))
    assert int(tags["NBANDS"]) == 77


# ------------------------------------------------------------------ unfold

def test_unfold_raises_when_k_point_count_does_not_match(tmp_path, mini_run,
                                                         capsys):
    """k 点数不符必须报错(守卫比较的是 EIGENVAL 里的**真实** k 点数)。

    这里把 fixture 的 EIGENVAL(2 个 k 点)放进一个 n_k = 1 的 grid_map 里 ——
    不改动 tests/data/ 的本体。
    """
    run = str(mini_run)
    with open(FIXTURE_EIGENVAL, encoding="utf-8") as fh:
        text = fh.read()
    with open(os.path.join(run, "EIGENVAL"), "w", encoding="utf-8") as fh:
        fh.write(text)
    rc = cli.main(["unfold", "--dir", run, "--outdir", str(tmp_path / "fs")])
    assert rc != 0
    err = capsys.readouterr().err
    # 必须正是 k 点数守卫的消息(删掉守卫后会是 PROCAR 的另一条消息)
    assert "VASP 算出的 k 点数 2 != grid_map 记录 1" in err
    assert "Traceback" not in err


def test_unfold_reports_a_missing_file_without_traceback(tmp_path, capsys):
    """M3: 缺文件是用户错误, 不该甩一脸 traceback。"""
    rc = cli.main(["unfold", "--dir", str(tmp_path / "nope"),
                   "--outdir", str(tmp_path / "fs")])
    assert rc != 0
    err = capsys.readouterr().err
    assert err.startswith("错误:")
    assert "Traceback" not in err


def test_unfold_produces_all_four_output_files(tmp_path, mini_run):
    """交付面 = 4 种格式(FS.grd 已删除: 没有任何标准读取器)。"""
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(mini_run), "--outdir", str(fs)])
    assert rc == 0
    for name in ("FS.cube", "FS.bxsf", "A(k,E).npz"):
        assert (fs / name).is_file(), name
    assert (fs / "band.1").is_file()
    assert not (fs / "FS.grd").exists()
    # 写出的文件能被自己的读取器读回来
    cube = read_cube(str(fs / "FS.cube"))
    assert cube["dims"] == [3, 1, 1]
    assert cube["field"].size == 3
    bxsf = read_bxsf(str(fs / "FS.bxsf"))
    assert bxsf["dims"] == [4, 2, 2]
    assert bxsf["nb"] >= 1
    band = read_fermisurfer_band(str(fs / "band.1"))
    assert band["dims"] == [3, 1, 1]
    npz = np.load(str(fs / "A(k,E).npz"))
    assert npz["A"].shape == (3, 401)
    assert npz["k_cart"].shape == (3, 3)


def test_unfold_prints_fermi_energy(mini_run, capsys):
    cli.main(["unfold", "--dir", str(mini_run), "--outdir",
              str(mini_run.parent / "fs")])
    out = capsys.readouterr().out
    assert "E_F" in out
    assert "-5.950000 eV" in out


# ------------------------------------------------------------------ 点群轨道对称化

def test_unfold_symmetrizes_by_default_and_no_symmetrize_disables_it(tmp_path,
                                                                       mini_run):
    """默认对称化 => 谱函数在点群轨道上逐位相同; --no-symmetrize => 恢复原始权重。

    (迷你体系是 3 个网格点, 其中两个在同一条轨道上 —— 见 mini_run 的
    full_to_ir = [0, 1, 1]; 这两个点的原始谱权重不同, 所以"默认 vs 关闭"
    必然给出不同的输出。)
    """
    out_sym = tmp_path / "fs_sym"
    out_raw = tmp_path / "fs_raw"
    assert cli.main(["unfold", "--dir", str(mini_run),
                     "--outdir", str(out_sym)]) == 0
    assert cli.main(["unfold", "--dir", str(mini_run), "--outdir", str(out_raw),
                     "--no-symmetrize"]) == 0
    a_sym = np.load(str(out_sym / "A(k,E).npz"))["A"]
    a_raw = np.load(str(out_raw / "A(k,E).npz"))["A"]
    assert not np.array_equal(a_sym, a_raw), (
        "--no-symmetrize 必须给出与默认不同的结果(对称化在这里确实有活干)")

    gm = load_grid_map(str(mini_run / "grid_map.npz"))
    grid = primitive_grid_points(gm.spec)
    orbits = point_group_orbits(grid, gm.ops)
    multi = [o for o in orbits if o.size > 1]
    assert multi, "迷你体系应当有多点轨道(否则本用例无意义)"
    for members in multi:
        for i in members[1:]:
            assert np.array_equal(a_sym[members[0]], a_sym[i]), (
                "默认输出在点群轨道上不逐位相同 —— 对称化没生效")
            assert not np.array_equal(a_raw[members[0]], a_raw[i])


def test_unfold_reports_the_symmetrization_state(mini_run, tmp_path, capsys):
    """CLI 必须**说清**默认是对称化的, 以及 --no-symmetrize 的后果。"""
    cli.main(["unfold", "--dir", str(mini_run), "--outdir",
              str(tmp_path / "fs")])
    out = capsys.readouterr().out
    assert "点群对称化" in out
    assert "已启用" in out
    cli.main(["unfold", "--dir", str(mini_run), "--outdir",
              str(tmp_path / "fs2"), "--no-symmetrize"])
    out = capsys.readouterr().out
    assert "已关闭" in out


def test_unfold_help_documents_the_symmetrization_switch(capsys):
    with pytest.raises(SystemExit):
        cli.main(["unfold", "--help"])
    out = capsys.readouterr().out
    assert "--no-symmetrize" in out
    assert "对称化" in out


# ------------------------------------------------------------------ det(M)=1 直通模式

def _primitive_run(directory):
    """det(M)=1 的迷你计算目录: "超胞"就是原胞, 3 个 k 点(3x1x1 网格)。"""
    _write_mini_system(directory, m=np.eye(3, dtype=int),
                       k_points=[(0.0, 0.0, 0.0),
                                 (1.0 / 3.0, 0.0, 0.0),
                                 (2.0 / 3.0, 0.0, 0.0)])
    _write_grid_map(directory)
    return directory


def test_unfold_works_for_a_primitive_cell_run(tmp_path):
    """det(M)=1(原胞当超胞)时 unfold 必须**照常工作**, 4 个费米面文件齐全。

    原来这里抛"4 种相位约定没有区分度"—— 用户要的正是原胞的费米面。
    """
    run = _primitive_run(str(tmp_path / "run"))
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", run, "--outdir", str(fs)])
    assert rc == 0
    for name in ("FS.cube", "FS.bxsf", "A(k,E).npz", "band.1"):
        assert (fs / name).is_file(), name
    cube = read_cube(str(fs / "FS.cube"))
    assert cube["field"].size == 3          # 3 个网格点
    npz = np.load(str(fs / "A(k,E).npz"))
    assert npz["A"].shape == (3, 401)


def test_unfold_reports_the_primitive_direct_mode(tmp_path, capsys):
    """直通模式必须在输出里说清"没有展开、权重是轨道投影"。"""
    run = _primitive_run(str(tmp_path / "run"))
    rc = cli.main(["unfold", "--dir", run, "--outdir", str(tmp_path / "fs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "原胞直通" in out
    assert "det(M)=1" in out
    # 直通模式没有相位约定可择优 ⇒ 不能打印一个假的"主导度 1.0000"
    assert "单 G 主导度" not in out


def test_primitive_direct_weights_match_the_orbital_projection_sums(tmp_path):
    """端到端核对: 直通模式的权重 = Σ_{ion,lm}|c|²(逐带逐 k)。

    参照物是 PROCAR **自己报的** |c|²(复投影块最后一列)—— 两条独立的
    路径(复系数求模方 vs VASP 直接打印)必须一致到打印精度。
    """
    run = _primitive_run(str(tmp_path / "run"))
    fs = tmp_path / "fs"
    assert cli.main(["unfold", "--dir", run, "--outdir", str(fs)]) == 0

    procar = read_procar(os.path.join(run, "PROCAR"))
    band = read_eigenval(os.path.join(run, "EIGENVAL"))
    # 从 PROCAR 文本里独立地把"每个离子那一行的最后一列"(= |c|² 总和)读出来
    printed = _procar_reported_c2(os.path.join(run, "PROCAR"))
    expect = printed.sum(axis=2)                       # (n_k, n_band) Σ_ion |c|²
    order = np.argsort(band.energies, axis=1, kind="stable")
    expect = np.take_along_axis(expect, order, axis=1)

    # 权重本身: 用 unfold 的内部接口再算一遍(与 npz 无关的独立核对)
    weights = _primitive_direct_weights(run)
    # PROCAR 只印 3 位小数 ⇒ 两条路径只能一致到打印精度(实测 ~1e-3)
    assert np.allclose(weights, expect, rtol=5e-3, atol=1e-12)
    # 而工具的权重与"从复系数现算的 Σ|c|²"是**精确**一致的
    exact = np.sum(np.abs(procar.coeffs) ** 2, axis=(2, 3))
    exact = np.take_along_axis(exact, order, axis=1)
    assert np.allclose(weights, exact, rtol=1e-12, atol=0.0)


def _procar_reported_c2(path):
    """独立解析 PROCAR: 复投影块每一行**最后一列**(= VASP 报的 |c|²)。

    只用文本解析, 不碰 unfoldfs.vaspio —— 目的是给工具的权重算一条
    **独立**的参照路径。返回 (n_k, n_band, n_ions)。
    """
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    head = lines[1]
    n_k = int(head.split("# of k-points:")[1].split()[0])
    n_band = int(head.split("# of bands:")[1].split()[0])
    n_ions = int(head.split("# of ions:")[1].split()[0])
    n_cplx = 4                       # 1 个轨道 -> (Re, Im) + |c|²
    out = []
    for line in lines:
        s = line.strip()
        if not s or not s[0].isdigit():
            continue
        parts = s.split()
        if len(parts) != n_cplx:
            continue
        out.append(float(parts[-1]))
    arr = np.array(out)
    assert arr.size == n_k * n_band * n_ions, (arr.size, n_k, n_band, n_ions)
    return arr.reshape(n_k, n_band, n_ions)


def _primitive_direct_weights(run_dir):
    """用工具自己的 unfold 路径再算一遍直通模式的权重(逐带逐 k)。"""
    from unfoldfs.geometry import (read_poscar, read_transmat, reciprocal,
                                   resolve_transform)
    from unfoldfs.unfold import build_atom_copies, unfold_supercell
    prim = read_poscar(os.path.join(run_dir, "PRIMCELL.vasp"))
    sc = read_poscar(os.path.join(run_dir, "POSCAR"))
    tr = resolve_transform(sc, prim,
                           read_transmat(os.path.join(run_dir, "TRANSMAT.in")))
    copies = build_atom_copies(prim, sc, tr)
    band = read_eigenval(os.path.join(run_dir, "EIGENVAL"))
    procar = read_procar(os.path.join(run_dir, "PROCAR"))
    gm = load_grid_map(os.path.join(run_dir, "grid_map.npz"))
    res = unfold_supercell(band, procar, tr, copies, tr.m_eff, prim.lattice,
                            prim.positions @ prim.lattice,
                            reciprocal(prim.lattice), reciprocal(sc.lattice),
                            gm.k_super)
    return res.weights[:, 0, :]


def test_unfold_warns_when_the_highest_band_is_occupied(tmp_path, capsys):
    """I4: 最高带被占据 => NBANDS 偏小, 占据态被截断, 费米面会缺态。"""
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 1.0, 1.0))
    _write_grid_map(str(run))
    rc = cli.main(["unfold", "--dir", str(run), "--outdir", str(tmp_path / "fs")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "NBANDS" in err
    assert "缺态" in err


# ------------------------------------------------------------------ --ef 语义

@pytest.mark.parametrize("ef_arg, expected", [
    ("auto", -5.95),      # 半导体: VBM(-6) + delta(0.05)
    ("vbm", -5.95),       # 强制 VBM + delta
    ("-3.5", -3.5),       # 数值: 直接当 E_F
])
def test_ef_modes_have_the_documented_semantics(tmp_path, mini_run, ef_arg,
                                                expected):
    fs = tmp_path / ("fs_" + ef_arg.replace("-", "m").replace(".", "p"))
    cli.main(["unfold", "--dir", str(mini_run), "--ef", ef_arg,
              "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(expected, abs=1e-9)
    assert float(npz["vbm"]) == pytest.approx(-6.0, abs=1e-9)
    assert float(npz["cbm"]) == pytest.approx(-2.0, abs=1e-9)


def test_ef_auto_takes_the_vbm_when_the_doscar_ef_sits_inside_the_gap(tmp_path):
    """带边的**分数占据不再意味着金属**(Ruling 25)。

    占据 (1, 1, 0.4, 0) => VBM=-6, CBM=-2(gap 4 eV), 而 DOSCAR 的 E_F=-5
    落在带隙里 => auto 取 VBM+delta。展宽体系里带边占据数本来就是 0.4/0.98
    这种小数(实测 MoS2: gap=1.61 eV 却因带边 0.9866 被判成金属, auto 于是
    取了带隙中央的 DOSCAR E_F => 费米面是空的)。
    """
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.4, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    cli.main(["unfold", "--dir", str(run), "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(-6.0 + 0.05, abs=1e-9)
    # 关键: **不是**带隙中央的 DOSCAR E_F
    assert float(npz["ef"]) != pytest.approx(_DOS_EF, abs=1e-9)


def test_ef_auto_uses_doscar_fermi_for_a_real_metal(tmp_path):
    """I2: "金属"用例必须是**真金属**。

    末两条带在 DOSCAR 的 E_F 处简并, 占据 (1,1,0.6,0) => VBM == CBM == -5
    (带隙为 0), 且带 3 的占据数 0.6 说明它就穿在 E_F 上。auto 因此取
    DOSCAR 的 E_F。
    """
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.6, 0.0),
                       band_energies=_METAL_BAND_ENERGIES)
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    cli.main(["unfold", "--dir", str(run), "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["vbm"]) == pytest.approx(_DOS_EF, abs=1e-9)
    assert float(npz["cbm"]) == pytest.approx(_DOS_EF, abs=1e-9)
    assert float(npz["ef"]) == pytest.approx(_DOS_EF, abs=1e-9)


def test_ef_auto_does_not_take_the_in_gap_doscar_ef_for_a_gapped_system(tmp_path,
                                                                        capsys):
    """I2 的另一半: 有带隙 + 带边分数占据时, auto **不得**取带隙里的
    DOSCAR E_F(那会让费米面是空的), 而且必须把这件事说出来。"""
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.4, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--outdir", str(fs)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "落在带隙内" in err
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(-5.95, abs=1e-9)


def test_ef_auto_semiconductor_uses_vbm_plus_delta(tmp_path):
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.0, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    cli.main(["unfold", "--dir", str(run), "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(-6.0 + 0.05, abs=1e-9)


# ------------------------------------------------------------------ 金属路径

# 1T-TaS2(金属)的**最小复现**: 一条带的占据数在 k 上跨过 0.5(费米穿越), 而
# 硬阈值切出来的"表观带隙"是**有限 k 网格的采样假象** —— 最高占据态与最低
# 空态**不在同一个 k** 上。
#   实测(1T-TaS2, VASP 6.6.0, 12 k 点): 表观 VBM=-1.568592 / CBM=-1.359436
#   (gap 0.209156 eV), DOSCAR E_F=-1.377658 正落在里面 ⇒ 旧代码把金属当成
#   半导体, auto 取了 VBM+0.05 = -1.518592(比真 E_F 低 0.140934 eV),
#   A(E_F)/A_max = 0.682 而不是 0.877。
_METAL_SPEC = GridSpec(p=(3, 3, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
_METAL_BAND_E = ((-10.0, -6.0, -5.0, -4.6),
                 (-10.0, -6.0, -5.8, -4.6),
                 (-10.0, -6.0, -5.0, -4.6))
_METAL_BAND_OCC = ((1.0, 1.0, 0.4232, 0.0),
                   (1.0, 1.0, 1.0198, 0.0),
                   (1.0, 1.0, 0.4232, 0.0))
_METAL_DOS_EF = -5.4         # 落在表观带隙 (-5.8, -5.0) 里

# 同样的几何, 但带 3 的占据数**整带都在 0.5 以上**(展宽的产物) ⇒ 半导体。
# 带隙 1.2 eV, DOSCAR 的 E_F 落在带隙中央。
_SEMI_BAND_E = ((-10.0, -6.0, -5.8, -4.6),
                (-10.0, -6.0, -5.8, -4.6),
                (-10.0, -6.0, -5.8, -4.6))
_SEMI_BAND_OCC = ((1.0, 1.0, 0.9866, 0.0),
                  (1.0, 1.0, 1.0267, 0.0),
                  (1.0, 1.0, 0.9866, 0.0))
_SEMI_DOS_EF = -5.2          # 落在表观带隙 (-5.8, -4.6) 里


def _write_multi_k_run(directory, band_e, occ_per_k, dos_ef):
    """多 k 点的迷你 VASP 计算目录(k 点取自 grid_map 的 k_super)。"""
    _write_mini_system(directory, band_energies=band_e[0],
                       occ_per_k=[occ_per_k[0]], dos_ef=dos_ef)
    gm = _write_grid_map(directory, spec=_METAL_SPEC)
    _write_mini_system(directory, band_energies=band_e,
                       k_points=[tuple(k) for k in gm.k_super],
                       occ_per_k=occ_per_k, dos_ef=dos_ef)
    return gm


def test_unfold_does_not_refuse_a_metal_whose_doscar_ef_sits_in_the_apparent_gap(
        tmp_path, capsys):
    """金属: 表观带隙是采样假象 ⇒ **不得**据此拒绝执行(Ruling 28)。

    那个能量上其实**有**态(带 25/26/27 穿过费米能), 拒绝执行等于把一次
    成功的金属计算扔掉。
    """
    run = tmp_path / "run"
    _write_multi_k_run(str(run), _METAL_BAND_E, _METAL_BAND_OCC, _METAL_DOS_EF)
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--ef", "doscar",
                   "--outdir", str(fs)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "落在带隙内" not in err
    assert (fs / "FS.cube").is_file()
    assert (fs / "band.1").is_file()


def test_unfold_auto_takes_the_doscar_ef_for_a_metal(tmp_path):
    """金属: --ef auto 必须取 DOSCAR 的 E_F(不是表观 VBM + delta)。"""
    run = tmp_path / "run"
    _write_multi_k_run(str(run), _METAL_BAND_E, _METAL_BAND_OCC, _METAL_DOS_EF)
    fs = tmp_path / "fs"
    cli.main(["unfold", "--dir", str(run), "--ef", "auto", "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(_METAL_DOS_EF, abs=1e-9)
    assert float(npz["ef"]) != pytest.approx(-5.8 + 0.05, abs=1e-9)


def test_unfold_still_refuses_a_semiconductor_whose_doscar_ef_sits_in_the_gap(
        tmp_path, capsys):
    """半导体: E_F 落在带隙里 ⇒ **仍然拒绝**(Ruling 28 只对金属关闭这个检查)。"""
    run = tmp_path / "run"
    _write_multi_k_run(str(run), _SEMI_BAND_E, _SEMI_BAND_OCC, _SEMI_DOS_EF)
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--ef", "doscar",
                   "--outdir", str(fs)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "落在带隙内" in err
    assert not fs.exists(), "拒绝执行时不能留下任何输出文件"


def test_unfold_auto_keeps_the_vbm_for_a_smeared_semiconductor(tmp_path):
    """展宽半导体: auto 仍然取 VBM+delta(费米穿越判据不能误伤半导体)。"""
    run = tmp_path / "run"
    _write_multi_k_run(str(run), _SEMI_BAND_E, _SEMI_BAND_OCC, _SEMI_DOS_EF)
    fs = tmp_path / "fs"
    cli.main(["unfold", "--dir", str(run), "--ef", "auto", "--outdir", str(fs)])
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(-5.8 + 0.05, abs=1e-9)
    assert float(npz["ef"]) != pytest.approx(_SEMI_DOS_EF, abs=1e-9)


# ------------------------------------------------------------------ C1 自检

def test_unfold_refuses_when_the_chosen_ef_sits_deep_inside_the_gap(tmp_path,
                                                                    capsys):
    """C1: 选出的 E_F 落在带隙**深处** => 该能量上没有任何态 => 拒绝执行。

    这里用 --ef doscar: DOSCAR 的 E_F=-5 落在 [-6,-2] 这个 4 eV 带隙里,
    离最近的态(VBM=-6)有 1 eV >> 2*sigma ⇒ 谱函数在 E_F 处已经衰减没了,
    产出的"费米面"是空的。CLI 必须**拒绝**(非零退出码), 且不写出任何文件。
    """
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.4, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--ef", "doscar",
                   "--outdir", str(fs)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "落在带隙内" in err
    assert not fs.exists(), "拒绝执行时不能留下任何输出文件"


def test_unfold_refuses_when_auto_lands_deep_inside_the_gap(tmp_path, capsys):
    """C1 的另一条路径: --ef auto + 一个把 E_F 推到带隙深处的 --delta。"""
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.4, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--ef", "auto",
                   "--delta", "1.0", "--outdir", str(fs)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "落在带隙内" in err
    assert not fs.exists()


def test_unfold_numeric_ef_inside_the_gap_warns_but_continues(tmp_path, capsys):
    """C1: 数值模式下不阻断(用户明给了 E_F, 是他自己的判断) —— 只告警。"""
    run = tmp_path / "run"
    _write_mini_system(str(run), occupations=(1.0, 1.0, 0.4, 0.0))
    _write_grid_map(str(run))
    fs = tmp_path / "fs"
    rc = cli.main(["unfold", "--dir", str(run), "--ef", "-5.0",
                   "--outdir", str(fs)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "落在带隙内" in err
    assert (fs / "FS.cube").is_file()
    npz = np.load(str(fs / "A(k,E).npz"))
    assert float(npz["ef"]) == pytest.approx(-5.0, abs=1e-9)


# ------------------------------------------------------------------ info

def test_info_reports_grid_size_and_reduction_ratio(mini_run, capsys):
    rc = cli.main(["info", "--dir", str(mini_run)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 x 1 x 1" in out
    assert "1/3" in out
    assert "0.3333" in out
    # M1: 标签要说清楚"相对什么"的约化, 并给出朴素超胞网格的对照
    assert "相对原胞满网格" in out
    assert "朴素超胞网格" in out


# ------------------------------------------------------------------ Python 3.9

def test_all_sources_parse_under_python_39():
    """master 上的解释器是 3.9 —— 语法必须能在 3.9 下解析。"""
    files = sorted((REPO_ROOT / "unfoldfs").glob("*.py")) \
        + sorted((REPO_ROOT / "tests").glob("*.py"))
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


# ------------------------------------------------------------------ I1/I3/I5

def test_unfold_without_eigenval_gives_an_actionable_message(tmp_path, mini_run,
                                                              capsys):
    """I1: EIGENVAL 缺失时的报错必须说清"缺的是哪一部分"。

    旧实现直接让 FileNotFoundError 冒到顶层 -> 用户看到
    "[Errno 2] No such file or directory: ...EIGENVAL", 完全不知道
    EIGENVAL 是干什么用的(E_F 的选取不需要它, 但**带能量**只有它有)。
    """
    run = str(mini_run)
    os.remove(os.path.join(run, "EIGENVAL"))
    rc = cli.main(["unfold", "--dir", run, "--ef", "-3.5",
                   "--outdir", str(tmp_path / "fs")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "EIGENVAL" in err
    assert "带能量" in err
    assert "Errno" not in err          # 不能是裸的 OSError
    assert "Traceback" not in err


def test_unfold_without_eigenval_in_auto_mode_says_the_ef_selection_needs_it(
        tmp_path, mini_run, capsys):
    """I1: --ef auto/vbm 的 E_F 选取**确实**需要 EIGENVAL 的占据数。

    这两种模式的报错要说清"缺的是占据数", 与数值模式的"缺带能量"区分开。
    """
    run = str(mini_run)
    os.remove(os.path.join(run, "EIGENVAL"))
    rc = cli.main(["unfold", "--dir", run, "--ef", "auto",
                   "--outdir", str(tmp_path / "fs")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "EIGENVAL" in err
    assert "占据数" in err
    assert "Traceback" not in err


def test_unfold_rejects_a_grid_map_that_does_not_match_the_structure(
        tmp_path, mini_run, capsys):
    """I3: grid_map 与 POSCAR/TRANSMAT 不自洽时必须**立刻**报错。

    旧实现没有守卫, 错误发生在最贵的相位约定择优之后, 而且是一条看不懂的
    ``IndexError: index 9 is out of bounds``。
    """
    import dataclasses
    run = str(mini_run)
    gm = load_grid_map(os.path.join(run, "grid_map.npz"))
    wrong_m = np.diag([2, 1, 1])          # 与结构文件的 M=diag(3,1,1) 不符
    tr = dataclasses.replace(
        gm.transform, m_eff=wrong_m,
        m_inv=np.linalg.inv(wrong_m.astype(float)))
    save_grid_map(os.path.join(run, "grid_map.npz"),
                  dataclasses.replace(gm, transform=tr))
    rc = cli.main(["unfold", "--dir", run, "--outdir", str(tmp_path / "fs")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "不自洽" in err
    assert "Traceback" not in err
    assert "IndexError" not in err


def test_info_lists_the_dropped_symmetry_operations(mini_run, capsys):
    """I5: info 必须列出被剔除的对称操作及原因(spec §7)。"""
    rc = cli.main(["info", "--dir", str(mini_run)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "对称操作" in out
    assert "保留" in out and "剔除" in out
    assert "原因:" in out                # 剔除原因必须写出来


def test_naive_supercell_count_gives_a_reason_for_each_failure_mode():
    """M2: "算不出来"的三种原因必须分开给, 不能一律报"不是对角的"。"""
    from unfoldfs.cli import _naive_supercell_count
    # 1) 对角但 P_i 不是 M_ii 的整数倍 -> 第三条消息
    n, why = _naive_supercell_count((6, 6, 1), np.diag([4, 3, 1]))
    assert n is None
    assert "整数倍" in why
    # 2) 非对角 -> 第一条消息
    n, why = _naive_supercell_count((6, 6, 1), np.array([[1, 1, 0], [0, 3, 0],
                                                          [0, 0, 1]]))
    assert n is None
    assert "对角" in why
    # 3) 对角元不是整数 -> 第二条消息
    n, why = _naive_supercell_count((6, 6, 1), np.diag([2.5, 3.0, 1.0]))
    assert n is None
    assert "整数" in why
    # 正常情形
    n, why = _naive_supercell_count((6, 6, 1), np.diag([3, 3, 1]))
    assert n == 4 and why == ""


# ------------------------------------------------------------------ 原胞完整点群
# 修复轮 6: CLI 的默认对称化必须用**原胞的完整点群**。
#
# 迷你体系(正交 Si)对 M=diag(3,1,1) 是**保持对称**的(公共操作 = 完整点群)
# ⇒ 测不出"只用公共子群"的差别。要测出差别必须用**破坏对称性**的超胞:
# 这里用 fixture 的 1H-MoS2 原胞(点群 D3h = 12 个操作)+ M=diag(3,1,1)
# ⇒ 公共操作只剩 4 个, 完整点群仍是 12 个。

_MOS2_PRIMCELL = str(REPO_ROOT / "tests" / "data" / "PRIMCELL.vasp")
_MOS2_M = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 1]])
_MOS2_ENERGIES = (-10.0, -6.0, -2.0, 1.0)
_MOS2_DOS_EF = -5.0
# 副本相位(沿 a 方向差 2pi/3): 让 4 种相位约定有区分度(实测相对差 ~9.6%)
_MOS2_U = np.array([0.1027125684732821 - 0.21014635803702303j,
                    0.23240310131893024 - 0.7116491117596169j,
                    0.5501292501189036 + 1.001533391119041j])
_MOS2_THETA = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def _write_mos2_supercell(directory):
    """写 MoS2 原胞 + 3x1x1 超胞的结构文件(破坏面内 C3)。"""
    os.makedirs(directory, exist_ok=True)
    prim = read_poscar(_MOS2_PRIMCELL)
    a_sc = _MOS2_M @ prim.lattice
    inv_m = np.linalg.inv(_MOS2_M.astype(float))
    frac, species = [], []
    for p, sp in zip(prim.positions, prim.species):
        for j in range(3):
            frac.append((p + j * np.array([1.0, 0.0, 0.0])) @ inv_m)
            species.append(sp)
    uniq = list(dict.fromkeys(species))
    _write_poscar(os.path.join(directory, "POSCAR"), "3x1x1 supercell", a_sc,
                  np.array(frac), uniq, [species.count(s) for s in uniq])
    _write_poscar(os.path.join(directory, "PRIMCELL.vasp"), "MoS2 primitive",
                  prim.lattice, prim.positions, uniq,
                  [prim.species.count(s) for s in uniq])
    with open(os.path.join(directory, "TRANSMAT.in"), "w",
              encoding="utf-8") as fh:
        fh.write("Read transformation matrix from the TRANSMAT.in file\n")
        for row in _MOS2_M:
            fh.write("  %d  %d  %d   # must be three integers\n" % tuple(row))
    return prim


def _write_mos2_vasp_output(directory, k_points, n_ions):
    """写合成的 EIGENVAL / PROCAR / DOSCAR。

    PROCAR 的复系数 = **原子相位**(随带与原子序号变) × **副本相位**(沿 a 方向
    差 2pi/3)⇒ 展开能把态集中到唯一一个候选 G 上, 且 4 种相位约定有区分度。

    占据数在 k 上跨过 0.5(第 3 条带在偶数/奇数 k 上分别 0 / 1)⇒ 金属 ⇒
    ``--ef auto`` 取 DOSCAR 的 E_F, 不会触发带隙检查的告警。
    """
    n_band = len(_MOS2_ENERGIES)
    with open(os.path.join(directory, "EIGENVAL"), "w", encoding="utf-8") as fh:
        fh.write("   %d   %d    1    1\n" % (n_ions, n_ions))
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic MoS2 supercell\n")
        fh.write("    2      %d    %d\n" % (len(k_points), n_band))
        fh.write("\n")
        for ik, kfrac in enumerate(k_points):
            fh.write("  %14.7E %14.7E %14.7E %14.7E\n"
                     % (kfrac[0], kfrac[1], kfrac[2], 1.0))
            occ = (1.0, 1.0, 1.0 if ik % 2 else 0.0, 0.0)
            for b, e in enumerate(_MOS2_ENERGIES):
                fh.write("  %5d %14.6f %10.6f\n" % (b + 1, e, occ[b]))
            fh.write("\n")
    coeffs = np.zeros((n_band, n_ions, 1), dtype=complex)
    for b, theta in enumerate(_MOS2_THETA):
        for i in range(n_ions):
            coeffs[b, i, 0] = np.exp(1j * theta * (i // 3)) * _MOS2_U[i % 3]
    with open(os.path.join(directory, "PROCAR"), "w", encoding="utf-8") as fh:
        fh.write("PROCAR lm decomposed + phase\n")
        fh.write("# of k-points: %3d         # of bands:  %d"
                 "         # of ions:  %4d\n" % (len(k_points), n_band, n_ions))
        fh.write("\n")
        for ik, kfrac in enumerate(k_points):
            fh.write(" k-point %5d :    %.8f %.8f %.8f"
                     "     weight = 1.00000000\n"
                     % (ik + 1, kfrac[0], kfrac[1], kfrac[2]))
            for b, e in enumerate(_MOS2_ENERGIES):
                fh.write("\n")
                fh.write("band  %5d # energy  %14.8f # occ.  1.00000000\n"
                         % (b + 1, e))
                fh.write("\n")
                fh.write("ion      s     tot\n")
                tot = 0.0
                for i in range(n_ions):
                    m2 = abs(coeffs[b, i, 0]) ** 2
                    tot += m2
                    fh.write("  %3d  %12.8f  %12.8f\n" % (i + 1, m2, m2))
                fh.write("tot   %12.8f  %12.8f\n" % (tot, tot))
                fh.write("ion      s\n")
                for i in range(n_ions):
                    c = coeffs[b, i, 0]
                    fh.write("  %3d %14.8f %14.8f %14.8f\n"
                             % (i + 1, c.real, c.imag, abs(c) ** 2))
    with open(os.path.join(directory, "DOSCAR"), "w", encoding="utf-8") as fh:
        fh.write("   %d   %d   1   0\n" % (n_ions, n_ions))
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic MoS2 supercell\n")
        fh.write("  1.000000000000000E+01 -1.000000000000000E+01  301"
                 " %20.10E  1.000000000000000E+00\n" % _MOS2_DOS_EF)
        for i in range(301):
            fh.write("  %14.6f 0.0000E+00 0.0000E+00\n"
                     % (-10.0 + i * 20.0 / 300.0))


def _mos2_run(tmp_path):
    """一个"破坏对称性"的 MoS2 3x1x1 计算目录(gen 出来的 grid_map + 合成输出)。"""
    src = tmp_path / "src"
    run = tmp_path / "run"
    prim = _write_mos2_supercell(str(src))
    assert cli.main(["gen",
                      "--supercell", str(src / "POSCAR"),
                      "--primcell", str(src / "PRIMCELL.vasp"),
                      "--transmat", str(src / "TRANSMAT.in"),
                      "--grid", "6", "6", "1",
                      "--outdir", str(run)]) == 0
    gm = load_grid_map(str(run / "grid_map.npz"))
    assert gm.ops.rotations.shape[0] == 4, "前提: 公共操作只剩 4 个"
    _write_mos2_vasp_output(str(run), [tuple(k) for k in gm.k_super], 9)
    return run, prim, gm


def _full_group_deviation(prim, grid, a):
    """A(k,E) 在**原胞完整点群**下的最大相对偏差(逐操作)。"""
    from unfoldfs.geometry import periodic_distance, rotation_inverse
    ops = primitive_symmetry_ops(prim)
    worst = 0.0
    for o in range(ops.rotations.shape[0]):
        r = ops.rotations[o]
        t = ops.translations[o]
        img = np.mod(grid @ rotation_inverse(r) + t, 1.0)
        d = periodic_distance(grid, img)
        j = d.argmin(axis=1)
        on = np.nonzero(d[np.arange(grid.shape[0]), j] < 1e-6)[0]
        for i in on:
            denom = max(a[i].max(), a[j[i]].max(), 1e-300)
            worst = max(worst, float(np.abs(a[i] - a[j[i]]).max() / denom))
    return worst


def test_unfold_default_output_is_covariant_under_the_full_point_group(tmp_path):
    """默认(unfold)的输出必须在**原胞完整点群**(12 个操作)下协变。

    破坏对称性的超胞(M=diag(3,1,1))上, 公共操作只有 4 个 ⇒ 只用公共子群
    做对称化时, 另外 8 个操作下**不协变**(实测最大相对偏差 1.000e+00)。

    两个产物都要查: ``FS.cube``(用户真正拿去画费米面的文件)与
    ``A(k,E).npz``(完整谱函数)。
    """
    run, prim, gm = _mos2_run(tmp_path)
    fs = tmp_path / "fs"
    assert cli.main(["unfold", "--dir", str(run), "--outdir", str(fs)]) == 0
    grid = primitive_grid_points(gm.spec)
    a = np.load(str(fs / "A(k,E).npz"))["A"]
    worst = _full_group_deviation(prim, grid, a)
    assert worst <= 1e-8, (
        f"A(k,E).npz 在原胞完整点群下不协变: 最大相对偏差 {worst:.3e}")
    cube = read_cube(str(fs / "FS.cube"))
    field = cube["field"].reshape(-1)
    assert field.size == grid.shape[0]
    worst_cube = _full_group_deviation(prim, grid, field)
    assert worst_cube <= 1e-8, (
        f"FS.cube 在原胞完整点群下不协变: 最大相对偏差 {worst_cube:.3e}")


def test_unfold_no_symmetrize_leaves_the_spectrum_non_covariant(tmp_path):
    """--no-symmetrize 时不协变(证明上面的协变确实是**对称化**带来的)。"""
    run, prim, gm = _mos2_run(tmp_path)
    fs = tmp_path / "fs_raw"
    assert cli.main(["unfold", "--dir", str(run), "--outdir", str(fs),
                     "--no-symmetrize"]) == 0
    grid = primitive_grid_points(gm.spec)
    a = np.load(str(fs / "A(k,E).npz"))["A"]
    worst = _full_group_deviation(prim, grid, a)
    assert worst >= 1e-3, (
        f"--no-symmetrize 的输出竟然协变(最大相对偏差只有 {worst:.3e}) —— "
        "要么展开本身变了, 要么对称化没被关掉")


def test_unfold_reports_the_full_point_group_size(tmp_path, capsys):
    """CLI 必须说清用的是**原胞完整点群**以及补回了几个被超胞破坏的操作。"""
    run, _prim, _gm = _mos2_run(tmp_path)
    cli.main(["unfold", "--dir", str(run), "--outdir", str(tmp_path / "fs")])
    out = capsys.readouterr().out
    assert "原胞完整点群" in out
    assert "12 个操作" in out
    assert "公共对称操作" in out



# ------------------------------------------------------------------ band 颜色标度

def test_band_weight_scale_flag_defaults_to_linear_and_accepts_log():
    parser = cli.build_parser()
    args = parser.parse_args(["band", "--dir", "x"])
    assert args.weight_scale == "linear"
    args = parser.parse_args(["band", "--dir", "x", "--weight-scale", "log"])
    assert args.weight_scale == "log"


def test_band_weight_scale_rejects_an_unknown_value():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["band", "--dir", "x", "--weight-scale", "sqrt"])


def test_band_plot_style_flag_defaults_to_line_and_accepts_heatmap():
    parser = cli.build_parser()
    assert parser.parse_args(["band", "--dir", "x"]).plot_style == "line"
    assert parser.parse_args(
        ["band", "--dir", "x", "--plot-style", "heatmap"]).plot_style == "heatmap"


def test_band_plot_style_rejects_an_unknown_value():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["band", "--dir", "x", "--plot-style", "3d"])


def test_symprec_option_defaults_to_1e_5_and_is_passed_through():
    """结构坐标有 1e-5 级噪声时, spglib 会漏掉大部分对称操作。

    实测 1H-NbSe2 单层(用户组的 POSCAR): symprec=1e-5 -> 只认出 4 个操作(Amm2),
    symprec=1e-4 -> 12 个(P-6m2 = D3h) => 对称化必须能用更松的容差。
    """
    parser = cli.build_parser()
    assert parser.parse_args(["unfold", "--dir", "x"]).symprec == 1e-5
    assert parser.parse_args(
        ["band", "--dir", "x", "--symprec", "1e-4"]).symprec == 1e-4
