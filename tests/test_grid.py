"""grid 模块测试: 均匀原胞网格、轨道分解、KPOINTS 写出、grid_map 存取。"""
import numpy as np
import pytest

from unfoldfs.geometry import (SymmetryOps, common_symmetry_ops, read_poscar,
                               read_transmat, reciprocal, resolve_transform)
from unfoldfs.grid import (GridMap, GridSpec, choose_grid_points,
                            decompose_grid, filter_ops_by_grid,
                            grid_resolution, load_grid_map, matrix_denominator_lcm,
                            primitive_grid_points, save_grid_map, write_kpoints)
from tests.conftest import FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT

# spglib 的平移代表元带 ~3e-8 数值残差(硬化只吸附接近 0/1 的分量)
TRANSLATION_EPS = 3e-8


def _fixture_ops():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    return prim, m, common_symmetry_ops(prim, m)


def _fixture_grid_map(spec=None):
    """按 cli.cmd_gen 的方式组装 GridMap(供 KPOINTS / 存取测试复用)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    tr = resolve_transform(sc, prim, m)
    if spec is None:
        spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    ops = common_symmetry_ops(prim, tr.m_eff)
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    gm = GridMap(
        spec=spec,
        ir_k_prim=grid[ir_idx],
        ir_k_super=np.array([tr.super_from_prim(k) for k in grid[ir_idx]]),
        multiplicity=mult,
        full_to_ir=full_to_ir,
        rot_of_full=rot,
        ops=ops,
        transform=tr,
        b_prim=reciprocal(prim.lattice),
        b_super=reciprocal(sc.lattice),
    )
    return gm, grid


# ------------------------------------------------------------------ 网格点生成

def test_grid_points_are_gamma_centered():
    spec = GridSpec(p=(4, 4, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    g = primitive_grid_points(spec)
    assert g.shape == (16, 3)
    assert np.allclose(g.min(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(g.max(axis=0), [0.75, 0.75, 0.0])


def test_grid_points_mp_is_shifted():
    spec = GridSpec(p=(4, 4, 1), grid_type="mp", shift=(0.0, 0.0, 0.0))
    g = primitive_grid_points(spec)
    # 网格是 4x4x1 的张量积, 第 0 列每个值重复 4 次 -> 比较唯一值
    assert np.allclose(np.unique(g[:, 0]), [0.125, 0.375, 0.625, 0.875])


def test_grid_point_count_matches_spec():
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    assert primitive_grid_points(spec).shape[0] == spec.n_points == 36


def test_choose_grid_points_respects_comm():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    spec = choose_grid_points(prim.lattice, m, resolution=0.2, grid_type="gamma")
    assert all(p % 3 == 0 for p in spec.p)
    assert matrix_denominator_lcm(m) == 3


def test_choose_grid_points_scales_with_resolution():
    """分辨率越细 -> 网格点数越多; 且点数永远 >= 1。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    coarse = choose_grid_points(prim.lattice, m, resolution=0.5, grid_type="gamma")
    fine = choose_grid_points(prim.lattice, m, resolution=0.1, grid_type="gamma")
    assert all(f >= c for f, c in zip(fine.p, coarse.p))
    assert all(p >= 1 for p in fine.p)


def test_grid_resolution_matches_point_spacing():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    spec = GridSpec(p=(12, 12, 3), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    b = reciprocal(prim.lattice)
    assert np.isclose(grid_resolution(b, spec),
                      max(np.linalg.norm(b[i]) / spec.p[i] for i in range(3)))


# ------------------------------------------------------------------ 轨道分解

def test_decompose_grid_closed_under_symmetry():
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim, m, ops = _fixture_ops()
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    assert sum(mult) == grid.shape[0]
    assert int(mult.max()) >= 1
    for i in range(grid.shape[0]):
        # i 所属轨道的代表点(ir_idx[full_to_ir[i]])必须与 i 同属一个轨道
        assert full_to_ir[i] == full_to_ir[ir_idx[full_to_ir[i]]]


def test_filter_ops_keeps_only_grid_preserving_ops():
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim, m, ops = _fixture_ops()
    filtered = filter_ops_by_grid(ops, grid)
    for i in range(filtered.rotations.shape[0]):
        img = filtered.apply(grid, i)
        assert np.allclose(np.sort(img, axis=0), np.sort(grid, axis=0), atol=1e-6)
        # 逐点检查(列排序只是边缘分布, 判别力弱): 每个像点都必须落回网格
        for p in img:
            d = np.abs(np.mod(p - grid + 0.5, 1.0) - 0.5).sum(axis=1)
            assert d.min() <= 1e-6


def test_filter_ops_keeps_all_ops_for_regular_hex_grid():
    """6x6x1 规则网格被 D3h 的全部 12 个操作保持(平移残差 ~3e-8 不得误删)。"""
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim, m, ops = _fixture_ops()
    assert ops.rotations.shape[0] == 12
    filtered = filter_ops_by_grid(ops, grid)
    assert filtered.rotations.shape[0] == 12


def test_filter_ops_rejects_translation_off_grid():
    """平移不落在网格上 -> 该操作不保持网格; 全部不保持时报错。"""
    spec = GridSpec(p=(2, 2, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    eye = np.eye(3, dtype=int)
    ops = SymmetryOps(np.array([eye]), np.array([[0.3, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        filter_ops_by_grid(ops, grid)


def test_filter_ops_rejects_half_translation_on_odd_grid():
    """2x2 网格允许 0.5 平移; 3x3 网格不允许(0.5 不是 3-网格向量)。"""
    eye = np.eye(3, dtype=int)
    even = SymmetryOps(np.array([eye]), np.array([[0.5, 0.0, 0.0]]))
    grid2 = primitive_grid_points(GridSpec((2, 2, 1), "gamma", (0.0, 0.0, 0.0)))
    assert filter_ops_by_grid(even, grid2).rotations.shape[0] == 1

    grid3 = primitive_grid_points(GridSpec((3, 3, 1), "gamma", (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        filter_ops_by_grid(even, grid3)


def test_decompose_grid_counts_distinct_points_not_operations():
    """两个操作落在同一格点时, 轨道大小按**去重后的格点数**算。"""
    spec = GridSpec(p=(2, 2, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    eye = np.eye(3, dtype=int)
    ops = SymmetryOps(np.array([eye, eye]), np.array([[0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]))
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    assert sum(mult) == grid.shape[0]        # 按操作数算会得到 8 != 4
    assert mult.max() == 1                   # 重复的恒等操作不增加轨道大小


def test_decompose_grid_handles_translation_residual():
    """平移带 ~3e-8 残差时仍须落到正确格点(截断取整会静默错一格)。"""
    spec = GridSpec(p=(2, 2, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    eye = np.eye(3, dtype=int)
    ops = SymmetryOps(
        np.array([eye, eye]),
        np.array([[0.0, 0.0, 0.0], [0.5 - TRANSLATION_EPS, 0.0, 0.0]]),
    )
    filtered = filter_ops_by_grid(ops, grid)
    assert filtered.rotations.shape[0] == 2
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, filtered)
    assert sum(mult) == grid.shape[0]
    assert mult.max() == 2                    # (0,0) 与 (0.5,0) 属同一轨道
    # grid 行序: (0,0,0) (0,0.5,0) (0.5,0,0) (0.5,0.5,0)
    assert full_to_ir[0] == full_to_ir[2]
    assert full_to_ir[0] != full_to_ir[1]


def test_decompose_grid_partitions_grid_into_orbits():
    """轨道分解是网格的一个划分: 每点恰属于一个轨道, 轨道大小 = 该轨道的点数。"""
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim, m, ops = _fixture_ops()
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    n = grid.shape[0]
    assert len(ir_idx) == len(mult)
    assert set(full_to_ir.tolist()) == set(range(len(ir_idx)))
    for k in range(len(ir_idx)):
        members = np.where(full_to_ir == k)[0]
        assert len(members) == mult[k]        # 轨道大小 = 该轨道的格点数
        assert members[0] == ir_idx[k]        # 代表点是轨道的第一个点
        assert full_to_ir[ir_idx[k]] == k     # 代表点属于自己
        assert int(rot[ir_idx[k]]) == -1      # 代表点不需要旋转
        # 轨道内的点确实由代表点经某个操作得到
        for j in members[1:]:
            o = int(rot[j])
            assert o >= 0
            assert np.allclose(np.mod(ops.apply(grid[ir_idx[k]], o) - grid[j] + 0.5,
                                         1.0) - 0.5, 0.0, atol=1e-6)


def test_decompose_grid_reduction_is_meaningful():
    """12 个操作下 6x6 网格必须被明显约化(每个轨道 >1 个点)。"""
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim, m, ops = _fixture_ops()
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    n_ir = len(ir_idx)
    assert n_ir < grid.shape[0]
    assert grid.shape[0] / n_ir > 2.0
    assert mult.max() >= 3


# ------------------------------------------------------------------ KPOINTS / 存取

def test_write_kpoints_format_and_weights(tmp_path):
    gm, grid = _fixture_grid_map()
    path = tmp_path / "KPOINTS"
    write_kpoints(str(path), gm)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("G")
    assert lines[2] == "Reciprocal lattice"
    assert len(lines) == 3 + int(gm.ir_k_prim.shape[0])
    weights = [float(l.split()[3]) for l in lines[3:]]
    assert abs(sum(weights) - gm.spec.n_points) < 1e-9


def test_write_kpoints_header_and_coordinates(tmp_path):
    """首行 = VASPKIT 风格网格声明; 坐标行 = 超胞分数坐标 + 权重。"""
    gm, grid = _fixture_grid_map()
    path = tmp_path / "KPOINTS"
    write_kpoints(str(path), gm)
    lines = path.read_text().splitlines()
    header = lines[0]
    assert header.split("#")[0].split() == [
        "G", str(gm.spec.p[0]), str(gm.spec.p[1]), str(gm.spec.p[2]),
        f"{grid_resolution(gm.b_prim, gm.spec):.3f}"]
    assert "# Parameters to Generate KPOINTS (Don't Edit This Line)" in header
    assert int(lines[1]) == gm.n_ir
    for line, k, w in zip(lines[3:], gm.ir_k_super, gm.multiplicity):
        parts = line.split()
        assert len(parts) == 4
        assert np.allclose([float(x) for x in parts[:3]], k, atol=1e-9)
        assert np.isclose(float(parts[3]), float(w))


def test_write_kpoints_mp_tag(tmp_path):
    spec = GridSpec(p=(6, 6, 1), grid_type="mp", shift=(0.0, 0.0, 0.0))
    gm, _ = _fixture_grid_map(spec)
    path = tmp_path / "KPOINTS"
    write_kpoints(str(path), gm)
    assert path.read_text().splitlines()[0].startswith("M")


def test_grid_map_roundtrip(tmp_path):
    gm, grid = _fixture_grid_map()
    npz = tmp_path / "grid_map.npz"
    save_grid_map(str(npz), gm)
    back = load_grid_map(str(npz))
    assert np.allclose(back.ir_k_prim, gm.ir_k_prim)
    assert np.allclose(back.ir_k_super, gm.ir_k_super)
    assert np.array_equal(back.full_to_ir, gm.full_to_ir)
    assert np.array_equal(back.rot_of_full, gm.rot_of_full)
    assert np.array_equal(back.multiplicity, gm.multiplicity)
    assert back.spec == gm.spec
    assert np.allclose(back.b_prim, gm.b_prim)
    assert np.allclose(back.b_super, gm.b_super)
    assert np.array_equal(back.ops.rotations, gm.ops.rotations)
    assert np.array_equal(back.transform.m_eff, gm.transform.m_eff)
