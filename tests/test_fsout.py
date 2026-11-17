"""fsout 模块测试: 费米面文件输出(cube / bxsf / FermiSurfer / npz)。

格式规范见 ``docs/formats.md`` —— 本文件里的断言就是那份规范的**可执行副本**:
``test_formats_doc_spec_matches_written_files`` 会把文档里的 spec 块解析出来,
逐条核对实际写出的文件。

交付面是 **4 种格式**(FS.cube / FS.bxsf / band.N / A(k,E).npz): 原计划的
``FS.grd`` 已删除 —— 它没有任何标准读取器(XCrySDen 无 .grd 代码路径,
gnuplot 的 splot 直接报错), 见 docs/formats.md §0。

夹具用一个**合成的**均匀网格(4x3x2 = 24 个满网格点), 而不是 VASP fixture:
VASP fixture 的 EIGENVAL 只有 2 个 k 点, 与 6x6x1 网格的超胞 k 集合对不上
(实测会触发 unfold 的 k 匹配塌缩检查), 且本模块只关心"展开后的数据 -> 文件"
这一段, 不需要真实 VASP 数据。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pytest

from unfoldfs.expand import ExpandedData, build_spectral_function
from unfoldfs.fsout import (BOHR_IN_ANGSTROM, band_edges, is_metal, make_e_grid,
                             read_bxsf, read_cube, read_fermisurfer_band,
                             select_fermi_energy, write_bxsf,
                             write_cube, write_fermisurfer_bands,
                             write_spectral_npz)
from unfoldfs.grid import GridSpec
from unfoldfs.vaspio import BandData

REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _make_mp_expanded(n_band: int = 4) -> ExpandedData:
    """MP 网格的 ExpandedData(点在 (i+1/2)/n 处)。"""
    spec = GridSpec(p=(4, 3, 2), grid_type="mp", shift=(0.0, 0.0, 0.0))
    axes = [np.arange(n) / n + 0.5 / n for n in spec.p]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n = mesh.shape[0]
    band_e = np.full((n, n_band), np.inf)
    band_w = np.zeros((n, n_band))
    valid = np.zeros((n, n_band), dtype=bool)
    for i in range(n):
        m = 1 + (i % n_band)
        band_e[i, :m] = -5.0 + 0.13 * i + np.arange(m)
        band_w[i, :m] = 0.8 - 0.05 * np.arange(m)
        valid[i, :m] = True
    return ExpandedData(k_cart=mesh @ B_PRIM, band_e=band_e, band_w=band_w,
                         valid=valid, spec=spec)


def _highest_valid_band(exp: ExpandedData, i: int) -> int:
    """点 i 的最高有效带的下标(补位槽的填充源, 见 docs/formats.md §5)。"""
    idx = [m for m in range(exp.band_e.shape[1]) if exp.valid[i, m]]
    assert idx, f"点 {i} 一条有效带都没有"
    return max(idx)


def _expected_band_value(exp: ExpandedData, i: int, n: int,
                         e_f: float = 0.0) -> float:
    """点 i 的第 n 个槽在逐带文件里应该写下的值。

    有效槽 = 该带自己的能量; 补位槽 = 该点最高有效带的能量(减去 e_f,
    因为 .frmsf 写的是相对 E_F 的能量)。
    """
    m = n if exp.valid[i, n] else _highest_valid_band(exp, i)
    return float(exp.band_e[i, m]) - e_f


_NUM_TOKEN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def _numeric_tokens(text: str) -> List[str]:
    """文本里所有"看起来是数字"的 token。"""
    return [t for t in text.split() if _NUM_TOKEN.match(t)]


def _reshape_by_spec(values: np.ndarray, d0: int, d1: int, d2: int,
                     order: str) -> np.ndarray:
    """**按文档里写的轴序**解释文件里的扁平数据。

    文档的 spec 块若被改成另一种轴序, 这里就会用错误的顺序 reshape,
    后面的逐点核对随之失败 —— 这正是"spec 键必须被断言引用"的意思。
    """
    if order == "third-axis-fastest":
        return values.reshape(d0, d1, d2)
    if order == "first-axis-fastest":
        return values.reshape(d2, d1, d0).transpose(2, 1, 0)
    raise AssertionError(f"未知的轴序约定: {order!r}")


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


def test_is_metal_uses_the_gap_criterion_only():
    """带边的**分数占据不再意味着金属**。

    任何用了展宽的**有带隙**体系, 带边占据数都会变成 0.98 这类小数(高斯展宽
    与四面体法都一样) —— 用它判金属会把半导体误判成金属(实测: MoS2 的
    VBM=-2.077449 / CBM=-0.465849 / gap=1.61 eV 被判成金属)。
    """
    smeared = _band([[-2.0, -0.05, 0.05, 3.0]], [[1.0, 0.6, 0.4, 0.0]])
    assert not is_metal(smeared)          # gap = 0.10 eV > gap_tol
    zero_gap = _band([[-2.0, 1.0, 1.0]], [[1.0, 1.0, 0.0]])
    assert is_metal(zero_gap)              # VBM == CBM
    insulator = _band([[-2.0, -1.0, 1.0, 3.0]], [[1.0, 1.0, 0.0, 0.0]])
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


def test_select_fermi_energy_numeric_and_doscar_modes_do_not_need_eigenval():
    """EIGENVAL 全空(没有占据态)时, 数值/doscar 模式仍必须可用。

    用户明明白白给了 E_F(--ef 0.5), 不该被一个用不上的 EIGENVAL 挡住。
    """
    empty = _band([[1.0, 2.0]], [[0.0, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=empty, mode="0.5") \
        == pytest.approx(0.5)
    assert select_fermi_energy(dos_ef=0.3, band=empty, mode="doscar") \
        == pytest.approx(0.3)


def test_select_fermi_energy_auto_falls_back_to_doscar_with_a_warning():
    """auto 模式在 EIGENVAL 不可用时退化到 DOSCAR 的 E_F, 且必须出声。"""
    empty = _band([[1.0, 2.0]], [[0.0, 0.0]])
    with pytest.warns(UserWarning, match="EIGENVAL"):
        ef = select_fermi_energy(dos_ef=0.3, band=empty, mode="auto")
    assert ef == pytest.approx(0.3)


def test_select_fermi_energy_auto_uses_vbm_when_doscar_ef_sits_in_a_smeared_gap():
    """真实 VASP 数据(带展宽): DOSCAR 的 E_F 落在**带隙中央** ⇒ 不能用它。

    实测(MoS2, 服务器端到端跑): band 27 的占据数是 0.9866..1.0267(唯一的
    分数占据), band 28 空 ⇒ VBM=-2.077449, CBM=-0.465849, gap=1.6116 eV
    (明确的半导体), 而 DOSCAR 的 E_F = -1.920079 落在带隙中央 ⇒ 用它画出来
    的费米面是空的。auto 必须取 VBM。
    """
    band = _band([[-2.1233, -2.0774, -0.4658]], [[0.9866, 1.0267, 0.0]])
    ef = select_fermi_energy(dos_ef=-1.920079, band=band, mode="auto")
    assert ef == pytest.approx(-2.0774)          # VBM, 不是带隙中央的 DOSCAR E_F
    # 带偏移时也要跟着 VBM 走
    ef = select_fermi_energy(dos_ef=-1.920079, band=band, mode="auto",
                              offset=0.25)
    assert ef == pytest.approx(-2.0774 + 0.25)


def test_select_fermi_energy_auto_uses_doscar_ef_for_a_true_metal():
    """真金属: 带穿过 E_F ⇒ auto 取 DOSCAR 的 E_F。"""
    band = _band([[-2.0, -0.05, 0.05, 3.0]], [[1.0, 0.6, 0.3, 0.0]])
    # vbm=-0.05, cbm=0.05 ⇒ gap=0.10 eV; DOSCAR 的 E_F=0.3 在带隙之外 ⇒ 金属
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="auto") \
        == pytest.approx(0.3)


def test_select_fermi_energy_auto_uses_doscar_ef_for_a_zero_gap():
    """零带隙(VBM == CBM)⇒ 金属 ⇒ 取 DOSCAR 的 E_F。"""
    band = _band([[-2.0, 1.0, 1.0]], [[1.0, 1.0, 0.0]])
    assert select_fermi_energy(dos_ef=0.3, band=band, mode="auto") \
        == pytest.approx(0.3)


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


def test_cube_header_states_the_field_max_and_a_suggested_isolevel(tmp_path):
    """VESTA 的默认 isolevel = 0 —— 头部必须告诉用户该选什么层级。

    否则"画不画得出来"取决于高斯尾巴是否恰好下溢到 0.0(实测: 换个 sigma
    或能量窗口就可能什么都没有)。
    """
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:2])
    m = re.search(r"max\s*=\s*([0-9.eE+-]+)", head)
    assert m is not None, f"头部没有体数据最大值: {head}"
    vmax = float(m.group(1))
    assert vmax == pytest.approx(float(np.max(_expected_a(exp))), rel=1e-3)
    m = re.search(r"level\s*~?\s*=?\s*([0-9.eE+-]+)", head)
    assert m is not None, f"头部没有建议的等值面层级: {head}"
    level = float(m.group(1))
    assert 0.0 < level < vmax


def test_cube_header_keeps_only_two_comment_lines(tmp_path):
    """cube 的头部必须还是两行注释(第 3 行就是 natoms)。"""
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[2].split()[0] == "0"
    assert lines[0].strip() and lines[1].strip()


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
    """逐带的值 == band_e(补位槽按"该点最高有效带"填充)。"""
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


def test_bxsf_padding_slots_are_filled_with_the_highest_valid_band(tmp_path):
    """补位槽的值 == 该点**最高有效带**的值(规则见 docs/formats.md §5)。"""
    exp = _make_expanded()
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    got = read_bxsf(str(path))
    for n in range(exp.band_e.shape[1]):
        core = got["bands"][n].reshape(5, 4, 3)[:4, :3, :2]
        for i in range(exp.k_cart.shape[0]):
            if exp.valid[i, n]:
                continue
            idx = _grid_index_of(exp, i)
            m = _highest_valid_band(exp, i)
            assert core[idx] == pytest.approx(exp.band_e[i, m], abs=1e-6)


def test_bxsf_refuses_an_mp_grid(tmp_path):
    """FermiSurfer 的 .bxsf 读取路径一律按 Γ 网格解释(shiftk 写死 0)。

    MP 网格的数据会被整体平移半个倒格矢 —— 静默的错误费米面, 必须报错。
    """
    exp = _make_mp_expanded()
    with pytest.raises(ValueError, match="网格"):
        write_bxsf(str(tmp_path / "FS.bxsf"), exp, B_PRIM, E_F)


def test_bxsf_accepts_a_gamma_grid(tmp_path):
    exp = _make_expanded()
    path = tmp_path / "FS.bxsf"
    write_bxsf(str(path), exp, B_PRIM, E_F)
    assert path.is_file()


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
                assert e[idx] == pytest.approx(exp.band_e[i, n] - E_F, abs=1e-6)


def test_fermisurfer_band_energies_are_relative_to_the_fermi_energy(tmp_path):
    """C1: .frmsf 的能量必须**相对 E_F**(E − E_F)。

    FermiSurfer 2.2.1 从批处理文件读 ``fermienergy``, 默认 0.0
    (``fermisurfer.cpp:219``); 官方三个 .frmsf 样例的能量也全部以 0 为中心。
    写绝对能量 => 用户直接打开得到的是 E = 0 eV 的等值面 —— 画不出费米面
    (静默失败)。
    """
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    for n in range(exp.band_e.shape[1]):
        got = read_fermisurfer_band(paths[n])
        e = got["energies"].reshape(4, 3, 2)
        for i in range(exp.k_cart.shape[0]):
            idx = _grid_index_of(exp, i)
            assert e[idx] == pytest.approx(
                _expected_band_value(exp, i, n, e_f=E_F), abs=1e-6)
    # FermiSurfer 默认 EF = 0 => 相对能量必须以 0 为中心, 且 E=0 等值面非空
    all_e = np.concatenate([read_fermisurfer_band(p)["energies"]
                            for p in paths])
    assert all_e.min() < 0.0 < all_e.max()
    crossing = [p for p in paths
                if read_fermisurfer_band(p)["energies"].min() < 0.0
                < read_fermisurfer_band(p)["energies"].max()]
    assert crossing, "没有任何一条带跨过 E_F —— FermiSurfer 画不出费米面"


def test_fermisurfer_padding_slots_are_filled_with_the_highest_valid_band(tmp_path):
    """补位槽的值 == 该点最高有效带的值(相对 E_F, 见 docs/formats.md §5)。"""
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    for n in range(exp.band_e.shape[1]):
        e = read_fermisurfer_band(paths[n])["energies"].reshape(4, 3, 2)
        for i in range(exp.k_cart.shape[0]):
            if exp.valid[i, n]:
                continue
            idx = _grid_index_of(exp, i)
            m = _highest_valid_band(exp, i)
            assert e[idx] == pytest.approx(exp.band_e[i, m] - E_F, abs=1e-6)


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
    # 能量按 (i1, i2, i3) 读回, 与展开结果逐点一致(相对 E_F)
    e = np.asarray(energies).reshape(4, 3, 2)
    for i in range(exp.k_cart.shape[0]):
        idx = _grid_index_of(exp, i)
        if exp.valid[i, 0]:
            assert e[idx] == pytest.approx(exp.band_e[i, 0] - E_F, abs=1e-6)


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


def test_spectral_npz_contains_fractional_k_coordinates(tmp_path):
    """npz 落盘分数坐标(b_prim 的用处): 二次分析不必再去找倒格矢。"""
    exp = _make_expanded()
    path = tmp_path / "A(k,E).npz"
    write_spectral_npz(str(path), exp, B_PRIM, E_F, vbm=-2.0, cbm=1.0,
                       e_grid=make_e_grid(E_F, half_width=2.0, n=41),
                       sigma=SIGMA)
    d = np.load(str(path), allow_pickle=False)
    assert d["k_frac"].shape == exp.k_cart.shape
    assert np.allclose(d["k_frac"],
                       exp.k_cart @ np.linalg.inv(B_PRIM), atol=1e-12)


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
        write_bxsf(str(d / "FS.bxsf"), mod, B_PRIM, E_F)
        write_fermisurfer_bands(str(d), mod, B_PRIM, E_F)
        outs.append({p.name: p.read_bytes() for p in sorted(d.iterdir())})
    assert outs[0].keys() == outs[1].keys()
    for name in outs[0]:
        assert outs[0][name] == outs[1][name], f"{name} 受补位槽影响"


def test_outputs_contain_no_inf_or_nan_tokens(tmp_path):
    """补位槽(+inf)不能把 inf/nan 泄漏进任何输出文件。"""
    exp = _make_expanded()
    cube = tmp_path / "FS.cube"
    write_cube(str(cube), exp, B_PRIM, _expected_a(exp))
    bxsf = tmp_path / "FS.bxsf"
    write_bxsf(str(bxsf), exp, B_PRIM, E_F)
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    bad = {"inf", "-inf", "+inf", "infinity", "nan", "-nan", "+nan"}
    for path in [cube, bxsf] + [Path(p) for p in paths]:
        text = path.read_text(encoding="utf-8")
        for token in text.split():
            assert token.lower() not in bad, f"{path.name} 里出现了 {token}"
        for token in _numeric_tokens(text):
            assert np.isfinite(float(token)), f"{path.name} 里 {token} 非有限"


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
    # BXSF 只支持 Γ 网格(FermiSurfer 的 read_bxsf 一律按 Γ 网格解释)
    assert spec["bxsf.supported_grid"] == "gamma-only"
    band_idx = [i for i, ln in enumerate(bx) if ln.strip().startswith("BAND:")]
    for n in range(exp.band_e.shape[1]):
        start = band_idx[n] + 1
        stop = band_idx[n + 1] if n + 1 < len(band_idx) else len(bx)
        vals = []
        for ln in bx[start:stop]:
            if ln.strip().startswith("END_"):
                break
            vals.extend(float(x) for x in ln.split())
            assert len(ln.split()) <= 6
        assert len(vals) == 5 * 4 * 3        # (n1+1)(n2+1)(n3+1)
        # **按 spec 里写的轴序**解释扁平数据 -> 文档漂移会被这里抓住
        grid = _reshape_by_spec(np.asarray(vals), 5, 4, 3,
                                spec["bxsf.axis_order"])
        core = grid[:4, :3, :2]
        for i_pt in range(exp.k_cart.shape[0]):
            idx = _grid_index_of(exp, i_pt)
            assert core[idx] == pytest.approx(
                _expected_band_value(exp, i_pt, n), abs=1e-6)

    # --- FermiSurfer
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    fs = Path(paths[0]).read_text(encoding="utf-8").splitlines()
    assert fs[0].split() == ["4", "3", "2"]
    assert fs[1].split() == [spec["frmsf.lshift_gamma"]]
    assert fs[2].split() == ["1"]
    n_energy = 4 * 3 * 2
    assert int(spec["frmsf.matrix_axes"]) == 3
    assert spec["frmsf.grid_dims"] == "unique"
    # .frmsf 的能量口径: 相对 E_F(FermiSurfer 默认 EF = 0)
    assert spec["frmsf.energy_origin"] == "relative-to-ef"
    body = [ln for ln in fs[6:] if ln.strip()]     # 跳过 3 行倒格矢
    n_all = sum(len(ln.split()) for ln in body)
    assert n_all == n_energy * (1 + 3)      # 能量 + 3 个矩阵元轴
    energies = np.asarray([float(x) for ln in body
                            for x in ln.split()][:n_energy])
    grid = _reshape_by_spec(energies, 4, 3, 2, spec["frmsf.axis_order"])
    for i_pt in range(exp.k_cart.shape[0]):
        idx = _grid_index_of(exp, i_pt)
        assert grid[idx] == pytest.approx(
            _expected_band_value(exp, i_pt, 0, e_f=E_F), abs=1e-6)

    # --- 补位槽的填充规则(文档里写的就是代码做的)
    assert spec["padding.fill_rule"] == "highest-valid-band"


def test_formats_doc_mentions_every_delivered_format():
    text = DOC_PATH.read_text(encoding="utf-8")
    for token in ("FS.cube", "FS.bxsf", "band.N", "A(k,E).npz"):
        assert token in text
    # .grd 已删除: 文档里不再有它的格式章节
    assert "## 3. `FS.grd`" not in text
    for token in ("BEGIN_BLOCK_BANDGRID_3D", "Fermi Energy:", "frmsf"):
        assert token in text


def test_formats_doc_states_the_isolevel_hint_rule():
    """cube 头部的"建议层级"规则必须写进文档(spec 块)。"""
    spec = _doc_spec()
    assert spec["cube.isolevel_hint"] == "half-of-field-max"


# ------------------------------------------------------------------ 畸形输入

def test_read_cube_reports_a_truncated_file_clearly(tmp_path):
    """读回解析器对畸形文件必须给可读信息, 而不是 IndexError。"""
    exp = _make_expanded()
    path = tmp_path / "FS.cube"
    write_cube(str(path), exp, B_PRIM, _expected_a(exp))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="截断|行"):
        read_cube(str(path))


def test_read_fermisurfer_band_reports_a_truncated_file_clearly(tmp_path):
    exp = _make_expanded()
    paths = write_fermisurfer_bands(str(tmp_path), exp, B_PRIM, E_F)
    path = Path(paths[0])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="截断|行"):
        read_fermisurfer_band(str(path))


# ------------------------------------------------------------------ Python 3.9

def test_fsout_sources_parse_under_python_39():
    """master 上的解释器是 3.9 —— 本模块的语法必须能在 3.9 下解析。

    (仓库级的扫描在 tests/test_cli.py::test_all_sources_parse_under_python_39)
    """
    for path in (REPO_ROOT / "unfoldfs" / "fsout.py",
                 REPO_ROOT / "tests" / "test_fsout.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
