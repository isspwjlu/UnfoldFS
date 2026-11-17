"""fsout 模块测试: 费米面文件输出(cube / grd / bxsf / FermiSurfer / npz)。

格式规范见 ``docs/formats.md`` —— 本文件里的断言就是那份规范的**可执行副本**:
``test_formats_doc_spec_matches_written_files`` 会把文档里的 spec 块解析出来,
逐条核对实际写出的文件。

夹具用一个**合成的**均匀网格(4x3x2 = 24 个满网格点), 而不是 VASP fixture:
VASP fixture 的 EIGENVAL 只有 2 个 k 点, 与 6x6x1 网格的超胞 k 集合对不上
(实测会触发 unfold 的 k 匹配塌缩检查), 且本模块只关心"展开后的数据 -> 文件"
这一段, 不需要真实 VASP 数据。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from unfoldfs.expand import ExpandedData, build_spectral_function
from unfoldfs.fsout import (BOHR_IN_ANGSTROM, band_edges, is_metal, make_e_grid,
                             read_bxsf, read_cube, read_fermisurfer_band,
                             read_grd, select_fermi_energy, write_bxsf,
                             write_cube, write_fermisurfer_bands, write_grd,
                             write_spectral_npz)
from unfoldfs.grid import GridSpec
from unfoldfs.vaspio import BandData

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "formats.md"

SPEC = GridSpec(p=(4, 3, 2), grid_type="gamma", shift=(0.0, 0.0, 0.0))

# 一个**故意不正交**的倒格矢: 行列写错或转置错会立刻被测出来
B_PRIM = np.array([[2.0, 0.0, 0.0],
                   [1.0, 1.5, 0.0],
                   [0.3, 0.2, 1.8]])

SIGMA = 0.08
E_F = -1.25


# ------------------------------------------------------------------ 夹具

def _make_expanded(spec: GridSpec = SPEC, b_prim: np.ndarray = B_PRIM,
                   n_band: int = 5) -> ExpandedData:
    """造一个 ExpandedData: 均匀网格 + 逐点带数不同(必然有补位槽)。"""
    axes = [np.arange(n) / n for n in spec.p]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    k_cart = mesh @ b_prim
    n = mesh.shape[0]
    band_e = np.full((n, n_band), np.inf)
    band_w = np.zeros((n, n_band))
    valid = np.zeros((n, n_band), dtype=bool)
    for i in range(n):
        m = 1 + (i % n_band)              # 带数逐点不同 -> 必有补位槽
        band_e[i, :m] = -5.0 + 0.13 * i + np.arange(m)
        band_w[i, :m] = 0.8 - 0.05 * np.arange(m)
        valid[i, :m] = True
    return ExpandedData(k_cart=k_cart, band_e=band_e, band_w=band_w,
                         valid=valid, spec=spec)


def _expected_a(expanded: ExpandedData, e_f: float = E_F,
                sigma: float = SIGMA) -> np.ndarray:
    """A(k, E_F) 的参考值(直接对 E_F 求值, 不经过 E 网格)。"""
    return build_spectral_function(expanded, np.array([e_f]), sigma)[:, 0]


def _doc_spec() -> dict:
    """解析 docs/formats.md 里的机器可读 spec 块。"""
    text = DOC_PATH.read_text(encoding="utf-8")
    m = re.search(r"```spec\n(.*?)```", text, re.S)
    assert m is not None, "docs/formats.md 缺少 ```spec 块"
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


# ------------------------------------------------------------------ E_F 选取

def _band(energies, occupations) -> BandData:
    e = np.atleast_2d(np.asarray(energies, dtype=float))
    o = np.atleast_2d(np.asarray(occupations, dtype=float))
    return BandData(kpoints=np.zeros((e.shape[0], 3)), energies=e,
                    occupations=o, efermi=float("nan"))


def test_band_edges_finds_vbm_and_cbm_of_a_semiconductor():
    band = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    vbm, cbm = band_edges(band)
    assert vbm == pytest.approx(-1.0)
    assert cbm == pytest.approx(1.0)


def test_band_edges_of_a_metal_brackets_the_fermi_level():
    band = _band([[-2.0, -0.05, 0.05, 3.0]], [[1.0, 0.6, 0.4, 0.0]])
    vbm, cbm = band_edges(band)
    assert vbm == pytest.approx(-0.05)
    assert cbm == pytest.approx(0.05)


def test_band_edges_rejects_a_file_without_occupied_states():
    band = _band([[1.0, 2.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="占据"):
        band_edges(band)


def test_is_metal_detects_fractional_occupations():
    metal = _band([[-2.0, -0.05, 0.05, 3.0]], [[1.0, 0.6, 0.4, 0.0]])
    insulator = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    assert is_metal(metal)
    assert not is_metal(insulator)


def test_select_fermi_energy_auto_uses_doscar_ef_for_a_metal():
    band = _band([[-2.0, -0.05, 0.05, 3.0]], [[1.0, 0.6, 0.4, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="auto") \
        == pytest.approx(0.3)


def test_select_fermi_energy_auto_uses_vbm_for_a_semiconductor():
    band = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="auto") \
        == pytest.approx(-1.0)


def test_select_fermi_energy_vbm_mode_honours_the_offset():
    band = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="vbm",
                                offset=0.25) == pytest.approx(-0.75)


def test_select_fermi_energy_accepts_an_explicit_value():
    band = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="-0.4") \
        == pytest.approx(-0.4)
    assert select_fermi_energy(dos_ef=0.3, band=band, mode=0.7) \
        == pytest.approx(0.7)


def test_select_fermi_energy_rejects_an_unknown_mode():
    band = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="ef"):
        select_fermi_energy(dos_ef=0.3, band=band, mode="bogus")


# ------------------------------------------------------------------ FS.cube

def test_cube_round_trip(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    got = read_cube(str(path))
    assert got["dims"] == [4, 3, 2]
    assert got["natoms"] == 0
    # cube 的矢量行是**每格点的增量矢量**(QE 的 cube.f90: INCREMENT VECTOR),
    # 且长度单位是原子单位(Bohr)
    inc = B_PRIM / (np.array([4, 3, 2])[:, None] * BOHR_IN_ANGSTROM)
    assert np.allclose(got["vectors"], inc)
    # 盒子 = 增量 × 格点数 = 倒格矢张成的 BZ
    assert np.allclose(got["cell"], B_PRIM)
    assert got["field"].shape == (4, 3, 2)          # (i1, i2, i3) = 网格下标序
    # 场逐点等于 A(k, E_F)
    assert np.allclose(got["field"].reshape(-1), _expected_a(exp), atol=1e-12)


def test_cube_matches_the_standard_convention_independently(tmp_path):
    """独立按 cube 规范读一遍(不复用 fsout 的 reader), 锁住三条约定。

    cube 规范(出处见 docs/formats.md §2): 矢量行是**增量矢量**、长度单位是
    **Bohr**、数据**第 3 个矢量最快**。
    """
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[2].split()[0])
    assert natoms == 0
    dims, inc = [], []
    for a in range(3):
        f = lines[3 + a].split()
        dims.append(int(f[0]))
        inc.append([float(x) for x in f[1:4]])
    vals = [float(x) for ln in lines[6 + natoms:] for x in ln.split()]
    n1, n2, n3 = dims
    assert len(vals) == n1 * n2 * n3
    # C 序 reshape => 第 3 个轴最快
    data = np.asarray(vals).reshape(n1, n2, n3)
    # 盒子 = 增量 × 格点数 × Bohr
    cell = np.asarray(inc) * np.array(dims, dtype=float)[:, None] * BOHR_IN_ANGSTROM
    assert np.allclose(cell, B_PRIM)
    # 场逐点等于 A(k, E_F)
    assert np.allclose(data.reshape(-1), _expected_a(exp), atol=1e-12)


def test_cube_grid_point_count_is_the_full_grid_not_n_k(tmp_path):
    """自检②: 输出的 k 点数 == 满网格点数(不是超胞 k 数)。"""
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    got = read_cube(str(path))
    assert int(np.prod(got["dims"])) == exp.k_cart.shape[0] == 24


def test_cube_grid_reconstructs_the_k_points(tmp_path):
    """cube 的格点 = 满网格的 k 点(周期)。"""
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    got = read_cube(str(path))
    n1, n2, n3 = got["dims"]
    axes = [np.arange(n) / n for n in (n1, n2, n3)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    k_cart = mesh @ got["cell"]
    # 逐点按周期距离匹配(expanded 的顺序是 (i1 最慢, i3 最快))
    for k in k_cart:
        d = np.abs(np.mod(exp.k_cart - k + 0.5, 1.0) - 0.5).sum(axis=1)
        assert d.min() < 1e-9


def test_cube_header_states_the_units_and_the_fake_lattice(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:6])
    assert "2pi/Angstrom" in head or "2π/Å" in head
    assert "eV" in head
    assert "reciprocal" in head.lower()


# ------------------------------------------------------------------ FS.grd

def test_grd_round_trip(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.grd"
    write_grd(str(path), exp, B_PRIM, _expected_a(exp))
    got = read_grd(str(path))
    assert got["nx"] == 4 and got["ny"] == 3
    assert got["nz"] == 2
    assert np.allclose(got["field"].reshape(-1), _expected_a(exp), atol=1e-12)


def test_grd_header_is_the_gnuplot_ascii_grid_layout(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.grd"
    write_grd(str(path), exp, B_PRIM, _expected_a(exp))
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    assert body[0].split() == ["4", "3"]        # nx ny
    assert len(body[1].split()) == 2            # xmin xmax
    assert len(body[2].split()) == 2            # ymin ymax
    assert len(body[3].split()) == 2            # zmin zmax
    # z 切片之间用空行分隔
    assert any(ln.strip() == "" for ln in body[4:])


# ------------------------------------------------------------------ FS.bxsf

def test_bxsf_round_trip(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    got = read_bxsf(str(path))
    assert got["ef"] == pytest.approx(E_F)
    assert got["nb"] == exp.band_e.shape[1]
    assert got["dims"] == [4 + 1, 3 + 1, 2 + 1]     # 周期端点重复一次
    assert np.allclose(got["bvec"], B_PRIM)
    assert len(got["bands"]) == got["nb"]
    assert all(b.size == 5 * 4 * 3 for b in got["bands"])   # (n1+1)(n2+1)(n3+1)


def test_bxsf_band_values_match_the_expanded_band_energies(tmp_path):
    """逐带的值 == band_e(补位槽按"下一条有效带"填充)。"""
    exp = _make_expanded()
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    got = read_bxsf(str(path))
    for n in range(exp.band_e.shape[1]):
        vals = got["bands"][n]
        grid = vals.reshape(4 + 1, 3 + 1, 2 + 1)
        core = grid[:4, :3, :2]
        # 逐点核对: 有效槽必须逐位相等
        for i in range(exp.k_cart.shape[0]):
            idx = _grid_index_of(exp, i)
            if exp.valid[i, n]:
                assert core[idx] == pytest.approx(exp.band_e[i, n], abs=1e-6)
        # 周期端点必须与起点相等
        assert np.allclose(grid[4, :3, :2], grid[0, :3, :2])
        assert np.allclose(grid[:4, 3, :2], grid[:4, 0, :2])
        assert np.allclose(grid[:4, :3, 2], grid[:4, :3, 0])


def test_bxsf_block_structure_matches_the_spec(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [ln.strip() for ln in lines]
    assert stripped[0] == "BEGIN_INFO"
    i_ef = next(i for i, ln in enumerate(lines) if "Fermi Energy:" in ln)
    assert stripped[i_ef + 1] == "END_INFO"
    assert stripped[i_ef - 1].startswith("#")      # INFO 块里带注释行
    assert "BEGIN_BLOCK_BANDGRID_3D" in stripped
    assert "BEGIN_BANDGRID_3D" in stripped
    assert "END_BANDGRID_3D" in stripped
    assert stripped[-1] == "END_BLOCK_BANDGRID_3D"
    n_band_lines = [ln for ln in lines if ln.strip().startswith("BAND:")]
    assert len(n_band_lines) == exp.band_e.shape[1]


def _grid_index_of(exp: ExpandedData, i: int) -> tuple:
    """满网格点 i 的 (i1, i2, i3) 网格下标。"""
    frac = exp.k_cart @ np.linalg.inv(B_PRIM)
    idx = []
    for a, n in enumerate(exp.spec.p):
        idx.append(int(np.mod(np.rint(frac[i, a] * n), n)))
    return tuple(idx)


# ------------------------------------------------------------------ FermiSurfer

def test_fermisurfer_band_round_trip(tmp_path):
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    assert len(paths) == exp.band_e.shape[1]
    assert [Path(p).name for p in paths] == [f"band.{n + 1}"
                                             for n in range(exp.band_e.shape[1])]
    got = read_fermisurfer_band(paths[0])
    assert got["dims"] == [4, 3, 2]           # frmsf 不重复周期端点
    assert got["lshift"] == 1                 # gamma 居中
    assert got["nb"] == 1                    # 每带一个文件
    assert np.allclose(got["bvec"], B_PRIM)
    assert got["energies"].size == 4 * 3 * 2


def test_fermisurfer_band_values_match_the_expanded_band_energies(tmp_path):
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    for n in range(exp.band_e.shape[1]):
        got = read_fermisurfer_band(paths[n])
        e = got["energies"].reshape(4, 3, 2)      # (i1, i2, i3)
        for i in range(exp.k_cart.shape[0]):
            idx = _grid_index_of(exp, i)
            if exp.valid[i, n]:
                assert e[idx] == pytest.approx(exp.band_e[i, n], abs=1e-6)


def test_fermisurfer_band_is_readable_by_a_fscanf_style_parser(tmp_path):
    """严格按 FermiSurfer `read_file()` 的 C 读取顺序逐 token 解析。

    FermiSurfer 的读法: 第一行整行读出(fgets)再 sscanf 三个整数, 之后
    全部用 fscanf 顺序读 —— 这里照抄那条路径, 确认文件能被它读通。
    """
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    text = Path(paths[0]).read_text(encoding="utf-8")
    # 1) fgets 读第一行, sscanf("%d%d%d")
    first, _, rest = text.partition("\n")
    ng = [int(x) for x in first.split()]
    assert ng == [4, 3, 2]
    # 2) 之后全部按 token 顺序读
    tokens = rest.split()
    pos = 0

    def take(n):
        nonlocal pos
        vals = [float(t) for t in tokens[pos:pos + n]]
        pos += n
        return vals

    lshift = int(tokens[pos]); pos += 1
    nb = int(tokens[pos]); pos += 1
    bvec = [take(3) for _ in range(3)]
    n_grid = 4 * 3 * 2
    energies = take(nb * n_grid)
    matrix = take(3 * nb * n_grid)
    assert lshift == 1 and nb == 1
    assert np.allclose(np.asarray(bvec), B_PRIM)
    assert len(energies) == n_grid and len(matrix) == 3 * n_grid
    assert pos == len(tokens)          # 没有多余/缺失的数据
    # 能量按 (i1, i2, i3) 读回, 与展开结果逐点一致
    e = np.asarray(energies).reshape(4, 3, 2)
    for i in range(exp.k_cart.shape[0]):
        idx = _grid_index_of(exp, i)
        if exp.valid[i, 0]:
            assert e[idx] == pytest.approx(exp.band_e[i, 0], abs=1e-6)


def test_fermisurfer_band_for_an_mp_grid_uses_lshift_two(tmp_path):
    spec = GridSpec(p=(4, 3, 2), grid_type="mp", shift=(0.0, 0.0, 0.0))
    axes = [np.arange(n) / n + 0.5 / n for n in spec.p]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    exp = ExpandedData(k_cart=mesh @ B_PRIM,
                        band_e=np.tile(np.arange(4.0), (24, 1)),
                        band_w=np.ones((24, 4)), valid=np.ones((24, 4), bool),
                        spec=spec)
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    assert read_fermisurfer_band(paths[0])["lshift"] == 2


# ------------------------------------------------------------------ npz

def test_spectral_npz_round_trip(tmp_path):
    exp = _make_expanded()
    e_grid = make_e_grid(E_F, half_width=2.0, n=41)
    path = tmp_path / "A(k,E).npz"
    write_spectral_npz(str(path), exp, B_PRIM, E_F, vbm=-2.0, cbm=1.0,
                        e_grid=e_grid, sigma=SIGMA)
    d = np.load(str(path), allow_pickle=False)
    assert d["A"].shape == (exp.k_cart.shape[0], e_grid.size)
    assert np.allclose(d["e_grid"], e_grid)
    assert float(d["ef"]) == pytest.approx(E_F)
    assert float(d["vbm"]) == pytest.approx(-2.0)
    assert float(d["cbm"]) == pytest.approx(1.0)
    assert float(d["sigma"]) == pytest.approx(SIGMA)
    assert str(d["k_units"]) == "2pi/Angstrom"
    assert str(d["e_units"]) == "eV"
    ref = build_spectral_function(exp, e_grid, SIGMA)
    assert np.allclose(d["A"], ref)


def test_make_e_grid_contains_the_fermi_energy_exactly():
    e = make_e_grid(-1.25, half_width=2.0, n=41)
    assert np.any(np.isclose(e, -1.25, atol=0, rtol=0))
    assert e.size == 41


# ------------------------------------------------------------------ 补位槽隔离

def test_invalid_slots_never_reach_any_output(tmp_path):
    """负向测试: 把补位槽的内容换成垃圾, 所有输出必须逐字节不变。"""
    exp = _make_expanded()
    assert not exp.valid.all()

    dirty = ExpandedData(
        k_cart=exp.k_cart,
        band_e=np.where(exp.valid, exp.band_e, 1e30),
        band_w=np.where(exp.valid, exp.band_w, -7.0),
        valid=exp.valid,
        spec=exp.spec,
    )

    a_ef = _expected_a(exp)
    outs = []
    for mod in (exp, dirty):
        d = tmp_path / ("a" if mod is exp else "b")
        d.mkdir()
        write_cube(str(d / "FS.cube"), mod, B_PRIM, a_ef)
        write_grd(str(d / "FS.grd"), mod, B_PRIM, a_ef)
        write_bxsf(str(d / "FS.bxsf"), mod, B_PRIM, E_F)
        write_fermisurfer_bands(str(d), mod, B_PRIM, E_F)
        outs.append({p.name: p.read_bytes() for p in sorted(d.iterdir())})
    assert outs[0].keys() == outs[1].keys()
    for name in outs[0]:
        assert outs[0][name] == outs[1][name], f"{name} 受补位槽影响"


def test_spectral_function_ignores_invalid_slots():
    """补位槽(能量 +inf)不得把展宽污染成 NaN。"""
    exp = _make_expanded()
    a = build_spectral_function(exp, np.array([E_F]), SIGMA)
    assert np.all(np.isfinite(a))


# ------------------------------------------------------------------ 文档即规范

def test_formats_doc_spec_matches_written_files(tmp_path):
    """docs/formats.md 里记录的规范 == 实际写出的文件。"""
    spec = _doc_spec()
    exp = _make_expanded()

    # --- cube
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert int(spec["cube.header_lines"]) == 2
    assert lines[0].strip() and lines[1].strip()          # 两行注释
    assert lines[2].split()[0] == "0"                    # natoms = 0
    data_lines = [ln for ln in lines[6:] if ln.strip()]
    n_vals = sum(len(ln.split()) for ln in data_lines)
    assert n_vals == 24
    assert int(spec["cube.data_per_line"]) == 6
    assert all(len(ln.split()) <= 6 for ln in data_lines)
    assert spec["cube.axis_order"] == "third-axis-fastest"
    assert spec["cube.vector_kind"] == "increment"
    assert spec["cube.vector_unit"] == "Bohr"

    # --- grd
    path = tmp_path / "FS.grd"
    write_grd(str(path), exp, B_PRIM, _expected_a(exp))
    body = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if not ln.startswith("#") and ln.strip()]
    assert spec["grd.header_order"] == "nx ny|xmin xmax|ymin ymax|zmin zmax"
    assert body[0].split() == ["4", "3"]                 # nx ny
    assert len(body[1].split()) == 2                      # xmin xmax
    assert len(body[2].split()) == 2                      # ymin ymax
    assert len(body[3].split()) == 2                      # zmin zmax
    assert spec["grd.slice_separator"] == "blank line"

    # --- bxsf
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    bx = path.read_text(encoding="utf-8").splitlines()
    stripped = [ln.strip() for ln in bx]
    assert spec["bxsf.block"] in stripped
    assert spec["bxsf.grid_block"] in stripped
    i = stripped.index(spec["bxsf.grid_block"])
    assert int(bx[i + 1].split()[0]) == exp.band_e.shape[1]     # nb
    assert bx[i + 2].split() == ["5", "4", "3"]                   # dims = unique+1
    assert spec["bxsf.grid_dims"] == "unique+1"
    assert int(spec["bxsf.values_per_line"]) == 6
    band_idx = [i for i, ln in enumerate(bx) if ln.strip().startswith("BAND:")]
    vals = []
    for ln in bx[band_idx[0] + 1:band_idx[1]]:
        vals += ln.split()
    assert len(vals) == 5 * 4 * 3        # (n1+1)(n2+1)(n3+1)
    assert all(len(ln.split()) <= 6 for ln in bx[band_idx[0] + 1:band_idx[1]])

    # --- FermiSurfer
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    fs = Path(paths[0]).read_text(encoding="utf-8").splitlines()
    assert fs[0].split() == ["4", "3", "2"]
    assert fs[1].split() == [spec["frmsf.lshift_gamma"]]
    assert fs[2].split() == ["1"]
    n_energy = 4 * 3 * 2
    assert int(spec["frmsf.matrix_axes"]) == 3
    assert spec["frmsf.grid_dims"] == "unique"
    body = [ln for ln in fs[6:] if ln.strip()]     # 跳过 3 行倒格矢
    n_all = sum(len(ln.split()) for ln in body)
    assert n_all == n_energy * (1 + 3)      # 能量 + 3 个矩阵元轴


def test_formats_doc_mentions_every_delivered_format():
    text = DOC_PATH.read_text(encoding="utf-8")
    for token in ("FS.cube", "FS.grd", "FS.bxsf", "band.N", "A(k,E).npz"):
        assert token in text
    for token in ("BEGIN_BLOCK_BANDGRID_3D", "Fermi Energy:", "frmsf"):
        assert token in text
