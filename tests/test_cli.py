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
from unfoldfs.fsout import read_bxsf, read_cube, read_fermisurfer_band
from unfoldfs.geometry import (common_symmetry_ops, read_poscar,
                               read_transmat, reciprocal, resolve_transform)
from unfoldfs.grid import (GridSpec, build_grid_map, filter_ops_by_grid,
                           load_grid_map, primitive_grid_points, save_grid_map,
                           write_kpoints)
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


def _supercell_frac():
    """超胞 6 个原子的分数坐标(原胞原子 x 沿 a 的 3 个副本)。"""
    a_sc = _M @ _A_PRIM
    out = []
    for p in _PRIM_FRAC @ _A_PRIM:
        for j in range(3):
            out.append((p + j * _A_PRIM[0]) @ np.linalg.inv(a_sc))
    return np.array(out)


def _write_mini_system(directory, occupations=None, band_energies=None):
    """写出一个自洽的迷你 VASP 计算目录。

    occupations: 4 个带的占据数(默认 1,1,0,0 -> 半导体);
        给 (1, 1, 0.4, 0) 则出现**带边分数占据**(展宽的产物)。
    band_energies: 4 个带的能量(默认 _BAND_ENERGIES);
        给 _METAL_BAND_ENERGIES 则得到零带隙的**真金属**。
    """
    os.makedirs(directory, exist_ok=True)
    occ = occupations if occupations is not None else (1.0, 1.0, 0.0, 0.0)
    energies = band_energies if band_energies is not None else _BAND_ENERGIES
    a_sc = _M @ _A_PRIM
    _write_poscar(os.path.join(directory, "POSCAR"), "synthetic supercell",
                  a_sc, _supercell_frac(), ["Si"], [6])
    _write_poscar(os.path.join(directory, "PRIMCELL.vasp"),
                  "synthetic primitive", _A_PRIM, _PRIM_FRAC, ["Si"], [2])
    with open(os.path.join(directory, "TRANSMAT.in"), "w",
              encoding="utf-8") as fh:
        fh.write("Read transformation matrix from the TRANSMAT.in file "
                 "if it exists.\n")
        fh.write("  3  0  0   # must be three integers\n")
        fh.write("  0  1  0   # must be three integers\n")
        fh.write("  0  0  1   # must be three integers\n")

    with open(os.path.join(directory, "EIGENVAL"), "w",
              encoding="utf-8") as fh:
        fh.write("   6   6    1    1\n")
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic mini system\n")
        fh.write("    2      1    %d\n" % len(energies))
        fh.write("\n")
        fh.write("  0.0000000E+00  0.0000000E+00  0.0000000E+00  1.0000000E+00\n")
        for b, e in enumerate(energies):
            fh.write("  %5d %14.6f %10.6f\n" % (b + 1, e, occ[b]))

    with open(os.path.join(directory, "DOSCAR"), "w", encoding="utf-8") as fh:
        fh.write("   6   6   1   0\n")
        fh.write("  1.000000000000000E+02  0.000000000000000E+00"
                 "  0.000000000000000E+00  0.100000000000000E-08"
                 "  0.500000000000000E-15\n")
        fh.write("  1.000000000000000E-004\n")
        fh.write("  CAR \n synthetic mini system\n")
        fh.write("  1.000000000000000E+01 -1.000000000000000E+01  301"
                 " %20.10E  1.000000000000000E+00\n" % _DOS_EF)
        for i in range(301):
            fh.write("  %14.6f 0.0000E+00 0.0000E+00\n"
                     % (-10.0 + i * 20.0 / 300.0))

    coeffs = np.zeros((len(energies), 6, 1), dtype=complex)
    for b, theta in enumerate(_THETA[:len(energies)]):
        at = np.array([1.0 + 0j, np.exp(1j * theta)])
        for i in range(2):
            for j in range(3):
                coeffs[b, i * 3 + j, 0] = at[i] * _U[j]
    with open(os.path.join(directory, "PROCAR"), "w", encoding="utf-8") as fh:
        fh.write("PROCAR lm decomposed + phase\n")
        fh.write("# of k-points:    1         # of bands:  %d"
                 "         # of ions:    6\n" % len(energies))
        fh.write("\n")
        fh.write(" k-point     1 :    0.00000000 0.00000000 0.00000000"
                 "     weight = 1.00000000\n")
        for b, e in enumerate(energies):
            fh.write("\n")
            fh.write("band  %5d # energy  %14.8f # occ.  1.00000000\n"
                     % (b + 1, e))
            fh.write("\n")
            fh.write("ion      s     tot\n")
            tot = 0.0
            for i in range(6):
                m2 = abs(coeffs[b, i, 0]) ** 2
                tot += m2
                fh.write("  %3d  %12.8f  %12.8f\n" % (i + 1, m2, m2))
            fh.write("tot   %12.8f  %12.8f\n" % (tot, tot))
            fh.write("ion      s\n")
            for i in range(6):
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
    ops = filter_ops_by_grid(common_symmetry_ops(prim, tr.m_eff), grid)
    gm = build_grid_map(spec, grid, ops, tr, reciprocal(prim.lattice),
                        reciprocal(sc.lattice))
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


def test_gen_grid_map_round_trips(tmp_path):
    out = tmp_path / "run"
    src = _write_mini_system(str(tmp_path / "src"))
    cli.main(_gen_args(src, out, "--resolution", "1.6"))
    gm = load_grid_map(str(out / "grid_map.npz"))
    assert gm.spec.p == (3, 3, 3)
    assert gm.n_k == 9
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
