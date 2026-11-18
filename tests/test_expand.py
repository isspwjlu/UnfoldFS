"""expand 模块测试: 对称扩展到均匀原胞网格 + 谱函数 A(k,E) 组装。

fixture = 3x3x1 MoS2 超胞(27 原子)+ 原胞(3 原子), M = diag(3,3,1), det(M)=9。
结构性测试用 6x6x1 网格(36 个满网格点 -> 4 个超胞 k, 每 k 9 个候选)。

⚠️ fixture 的 PROCAR 只有 3 位小数精度(且只有 2 个 k 点), 这里只做**结构**
断言, 不把它当物理基线; 物理基线用 master 上 100-k 网格的真实数据。
"""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from unfoldfs.expand import (ExpandedData, build_spectral_function,
                             expand_to_full_grid, full_point_group,
                             point_group_orbits, symmetrize_spectral_function)
from unfoldfs.geometry import (SymmetryOps, Transform, common_symmetry_ops,
                               common_symmetry_ops_detailed,
                               primitive_symmetry_ops, read_poscar,
                               read_transmat, reciprocal, resolve_transform)
from unfoldfs.grid import (GridSpec, build_grid_map, filter_ops_by_grid,
                           filter_ops_by_grid_detailed, primitive_grid_points)
from unfoldfs.unfold import (UnfoldedResult, build_atom_copies,
                             candidate_g_for_k, unfold_supercell)
from unfoldfs.vaspio import read_eigenval, read_procar
from tests.conftest import (FIXTURE_EIGENVAL, FIXTURE_POSCAR,
                            FIXTURE_PRIMCELL, FIXTURE_PROCAR, FIXTURE_TRANSMAT)

SPEC = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))

# spglib 平移代表元的数值残差量级
TRANSLATION_EPS = 3e-8


# ------------------------------------------------------------------ 夹具

def _fixture_geometry():
    """fixture 的原胞 / 超胞 / 变换矩阵。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    tr = resolve_transform(sc, prim, read_transmat(str(FIXTURE_TRANSMAT)))
    return prim, sc, tr


def _grid_map(spec=SPEC, grid=None):
    """按 cli.cmd_gen 的方式组装 GridMap(可指定自定义网格点顺序)。"""
    prim, sc, tr = _fixture_geometry()
    if grid is None:
        grid = primitive_grid_points(spec)
    ops = filter_ops_by_grid(common_symmetry_ops(prim, tr.m_eff), grid)
    gm = build_grid_map(spec, grid, ops, tr, reciprocal(prim.lattice),
                        reciprocal(sc.lattice))
    return gm, grid


def _synthetic_unfolded(gm, n_band=6):
    """造一个 UnfoldedResult: 候选顺序与 unfold 一致, 各候选带数不同(有补位槽)。

    候选的分数值按 unfold 的口径算(``candidate_g_for_k`` + ``(K+g)·M^-T``),
    于是"候选索引 <-> 候选分数值"的对应关系与真实展开完全一致。
    """
    m_eff = gm.transform.m_eff
    m_inv_T = gm.transform.m_inv.T
    n_k, det = gm.n_k, gm.transform.det
    k_cand = np.zeros((n_k, det, 3))
    for ik in range(n_k):
        g = candidate_g_for_k(gm.k_super[ik], m_eff, m_inv_T)
        k_cand[ik] = (gm.k_super[ik] + g) @ m_inv_T
    energies = np.full((n_k, det, n_band), np.inf)
    weights = np.zeros((n_k, det, n_band))
    valid = np.zeros((n_k, det, n_band), dtype=bool)
    for ik in range(n_k):
        for j in range(det):
            n = 1 + (ik + j) % n_band           # 各候选带数不同 -> 必有补位槽
            energies[ik, j, :n] = -10.0 + ik + j + np.arange(n)
            weights[ik, j, :n] = 0.1 * (j + 1) + 0.01 * np.arange(n)
            valid[ik, j, :n] = True
    return UnfoldedResult(k_cand=k_cand, energies=energies, weights=weights,
                           valid=valid)


def _candidate_to_grid(gm, grid=None):
    """**独立路径**: 由 (超胞 k, 候选) 反查它命中的满网格点。

    方向与 expand 相反(expand 是 网格点 -> 候选), 且不共用任何匹配代码,
    故可以当作交叉核对。
    """
    if grid is None:
        grid = primitive_grid_points(gm.spec)
    m_eff = gm.transform.m_eff
    m_inv_T = gm.transform.m_inv.T
    out = {}
    for k in range(gm.n_k):
        g = candidate_g_for_k(gm.k_super[k], m_eff, m_inv_T)
        cand = (gm.k_super[k] + g) @ m_inv_T
        for j in range(cand.shape[0]):
            d = np.abs(np.mod(cand[j] - grid + 0.5, 1.0) - 0.5).sum(axis=1)
            hit = np.nonzero(d <= 1e-6)[0]
            assert hit.size == 1, f"候选 ({k},{j}) 命中 {hit.size} 个网格点"
            out[(k, j)] = int(hit[0])
    return out


def _with_bz_copies(res, n_copies=2, shift=0.0):
    """给展开结果追加 (n_copies-1) 份**超胞 BZ 副本**的行。

    副本 = 同一个超胞 k 在原胞第一布里渊区里的另一个 BZ 副本(K 相差一个
    超胞倒格矢)⇒ 它的候选落在**同一批**原胞网格点上 ⇒ 每个网格点因此收到
    多个源(多源求和的物理来源)。

    ``shift``: 副本的带能量整体平移多少 eV(用来测"按能量对齐"的合并口径)。
    """
    if n_copies == 1:
        return res
    e_copies = []
    for _ in range(n_copies - 1):
        e2 = np.asarray(res.energies, dtype=float).copy()
        v2 = np.asarray(res.valid, dtype=bool).copy()
        finite = np.isfinite(e2) & v2
        e2[finite] += float(shift)
        e_copies.append(e2)
    return replace(
        res,
        k_cand=np.concatenate([res.k_cand] * n_copies, axis=0),
        energies=np.concatenate([res.energies] + e_copies, axis=0),
        weights=np.concatenate([res.weights] * n_copies, axis=0),
        valid=np.concatenate([res.valid] * n_copies, axis=0),
    )


# ------------------------------------------------------------------ 扩展

def test_expand_takes_the_band_block_of_the_covering_candidate():
    """每个满网格点取到的能带块必须来自**唯一覆盖它**的 (超胞 k, 候选)。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)

    mapping = _candidate_to_grid(gm)
    # (k, 候选) 与满网格点一一对应(双射)
    assert len(mapping) == gm.n_k * gm.transform.det == gm.spec.n_points
    assert len(set(mapping.values())) == gm.spec.n_points

    for (k, j), i in mapping.items():
        assert np.allclose(exp.band_e[i], res.energies[k, j])
        assert np.allclose(exp.band_w[i], res.weights[k, j])
        assert np.array_equal(exp.valid[i], res.valid[k, j])


def test_expand_covers_every_grid_point_exactly_once():
    """自检①: 每个满网格点恰好命中一个 (k, 候选); 每个 (k, 候选) 也只用一次。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)
    assert exp.band_e.shape == (gm.spec.n_points, res.energies.shape[2])
    assert exp.band_w.shape == exp.band_e.shape
    assert exp.valid.shape == exp.band_e.shape
    # 反查: 全部 (k, 候选) 槽都被用上, 且各用一次
    mapping = _candidate_to_grid(gm)
    assert sorted(mapping.values()) == list(range(gm.spec.n_points))


def test_expand_k_cart_matches_the_uniform_primitive_grid():
    """自检②: k_cart 与均匀原胞网格逐点对应(周期距离 < 1e-6)。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)

    expect = grid @ gm.b_prim
    # 笛卡尔差: 不允许出现"差一个倒格矢"的错(k 侧取模的边界陷阱)
    assert np.abs(exp.k_cart - expect).max() < 1e-6
    # 分数坐标下的周期距离
    frac = exp.k_cart @ np.linalg.inv(gm.b_prim)
    d = np.abs(np.mod(frac - grid + 0.5, 1.0) - 0.5).sum(axis=1)
    assert d.max() < 1e-6


def test_expand_preserves_padding_slots():
    """补位槽(能量 +inf、权重 0)必须原样带过来, 并被 valid 掩码标出。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    assert not res.valid.all()          # 确实存在补位槽
    exp = expand_to_full_grid(res, gm)
    for (k, j), i in _candidate_to_grid(gm).items():
        assert np.array_equal(exp.valid[i], res.valid[k, j])
    assert np.all(np.isinf(exp.band_e[~exp.valid]))
    assert np.all(exp.band_w[~exp.valid] == 0.0)


def test_expand_preserves_total_weight():
    """权重守恒: 满网格上的权重之和 == 展开结果里有效槽的权重之和。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)
    assert exp.band_w.sum() == pytest.approx(res.weights[res.valid].sum(),
                                             rel=1e-12)


# ------------------------------------------------- 多源求和(修复轮 8)
#
# 原胞的谱函数是**所有**落到同一个原胞 k' 的超胞态的投影之和: 每个超胞 k
# 点 K 各贡献一份(经由自己的 G 分量 G_K = k'·Mᵀ − K)。当超胞 k 列表覆盖了
# 原胞第一布里渊区里的**多个超胞 BZ 副本**时, 同一个原胞 k' 会被 det(M) 个
# 超胞 k 覆盖 —— 只取其中一个就丢掉 (n_k−1)/n_k 的谱权重。

def test_expand_sums_the_sources_that_land_on_the_same_grid_point():
    """多源: 同一网格点被多个源覆盖时, 各源的带按能量对齐后权重相加。"""
    gm, grid = _grid_map()
    single = _synthetic_unfolded(gm)
    multi = _with_bz_copies(single, n_copies=2)
    exp1 = expand_to_full_grid(single, gm)
    exp2 = expand_to_full_grid(multi, gm)

    # 带(能量与条数)不变 —— 多源只是把**同一批带**的权重加起来
    assert exp2.band_e.shape == exp1.band_e.shape
    assert np.array_equal(exp2.band_e, exp1.band_e)
    assert np.array_equal(exp2.valid, exp1.valid)
    assert np.allclose(exp2.band_w, 2.0 * exp1.band_w)
    # 权重和随源数线性增长(而不是被截断在一个源上)
    assert exp2.band_w.sum() == pytest.approx(2.0 * exp1.band_w.sum(), rel=1e-12)
    three = expand_to_full_grid(_with_bz_copies(single, n_copies=3), gm)
    assert three.band_w.sum() == pytest.approx(3.0 * exp1.band_w.sum(),
                                               rel=1e-12)


def test_expand_merges_the_bands_of_two_sources_by_energy():
    """两个源的带按**能量**对齐: 能量差 <= merge_tol 的带合并(权重相加)。"""
    gm, grid = _grid_map()
    single = _synthetic_unfolded(gm)
    exp1 = expand_to_full_grid(single, gm)
    # 副本的带能量整体平移 1e-9 eV(远小于 merge_tol=1e-6)⇒ 仍然同一条带
    near = expand_to_full_grid(_with_bz_copies(single, n_copies=2, shift=1e-9),
                                gm)
    assert near.band_e.shape == exp1.band_e.shape
    assert np.allclose(near.band_e, exp1.band_e)
    assert np.allclose(near.band_w, 2.0 * exp1.band_w)
    # 平移 1e-3 eV(远大于 merge_tol)⇒ 是**不同的**带, 不能合并
    far = expand_to_full_grid(_with_bz_copies(single, n_copies=2, shift=1e-3),
                               gm)
    assert far.band_e.shape[1] == 2 * exp1.band_e.shape[1]
    assert far.band_w.sum() == pytest.approx(2.0 * exp1.band_w.sum(), rel=1e-12)


def test_expand_multi_source_keeps_the_band_energies_of_each_source():
    """多源合并**不改带能量**: 合并后的能量是源 0 的能量(按能量对齐)。"""
    gm, grid = _grid_map()
    single = _synthetic_unfolded(gm)
    multi = _with_bz_copies(single, n_copies=2, shift=1e-9)
    exp = expand_to_full_grid(multi, gm)
    mapping = _candidate_to_grid(gm)
    for (k, j), i in mapping.items():
        v = single.valid[k, j]
        e_src = np.asarray(single.energies[k, j])[v]
        e_out = exp.band_e[i][exp.valid[i]]
        assert np.allclose(e_out, e_src, atol=1e-6)
        # 权重 = 两个源的权重之和
        w_src = np.asarray(single.weights[k, j])[v]
        assert np.allclose(exp.band_w[i][exp.valid[i]], 2.0 * w_src)


def test_expand_multi_source_keeps_the_spectrum_covariant():
    """多源求和与点群对称化是**两道正交的处理**: 对称化后仍须协变。"""
    gm, grid, _ = _fixture_pipeline_spectrum()
    res = _synthetic_unfolded(gm)
    multi = _with_bz_copies(res, n_copies=2)
    exp = expand_to_full_grid(multi, gm)
    a = build_spectral_function(exp, np.linspace(-20.0, 10.0, 61), 0.2)
    devs = _point_group_deviation(gm, grid, symmetrize_spectral_function(a, gm))
    assert devs, "没有可比的对称操作"
    worst = max(devs.values())
    assert worst <= 1e-8, (
        f"多源求和后对称化的 A(k,E) 仍不协变: 最大相对偏差 {worst:.3e}")


def test_expand_rejects_fewer_k_points_than_the_grid_map():
    """展开结果的 k 数少于 grid_map.k_super 时必须报错(会静默漏掉网格点)。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    bad = replace(res, energies=res.energies[:-1], weights=res.weights[:-1],
                  valid=res.valid[:-1], k_cand=res.k_cand[:-1])
    with pytest.raises(ValueError, match="k"):
        expand_to_full_grid(bad, gm)


def test_expand_rejects_a_deliberately_wrong_candidate_index(monkeypatch):
    """负向测试: 故意把候选索引打乱 -> 自检必须抓到(否则会静默取错带)。"""
    import unfoldfs.expand as expand_mod
    real = expand_mod.candidate_g_for_k

    def _reversed(k_super_frac, m_eff, m_inv_T, tol=1e-9):
        # 分数值集合不变, 只是顺序被打乱 -> 索引 j 与展开结果的候选轴错位
        return real(k_super_frac, m_eff, m_inv_T, tol)[::-1].copy()

    monkeypatch.setattr(expand_mod, "candidate_g_for_k", _reversed)
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    with pytest.raises(ValueError, match="候选索引"):
        expand_to_full_grid(res, gm)


def test_expand_rejects_k_count_mismatch():
    """展开结果的 k 数与 grid_map 不一致时必须报错(否则会取错带)。"""
    gm, grid = _grid_map()
    res = _synthetic_unfolded(gm)
    bad = replace(res, energies=res.energies[:-1], weights=res.weights[:-1],
                  valid=res.valid[:-1])
    with pytest.raises(ValueError, match="k"):
        expand_to_full_grid(bad, gm)


def test_expand_uses_per_k_candidates_for_a_non_symmetric_transform():
    """一般 M 下逐 k 的候选集与全局 gset 的**成员**不同 —— 必须逐 k 计算。

    用 M = [[2,1,0],[0,2,0],[0,0,1]](det=4) 的合成超胞: 它的 k=2/3 的候选
    成员与全局 gset 不同。若实现误用 gm.gset, 扩展会直接失败(覆盖检查)或
    取错带(候选索引检查)。
    """
    m_eff = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    tr = Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    spec = GridSpec(p=(4, 4, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    ops = SymmetryOps(np.array([np.eye(3, dtype=int)]),
                      np.array([[0.0, 0.0, 0.0]]))
    b_prim = reciprocal(np.eye(3) * 3.0)
    gm = build_grid_map(spec, grid, ops, tr, b_prim, b_prim)

    # 前提: 逐 k 的候选与全局 gset 的成员确实不同(否则本用例不成立)
    gset = {tuple(int(x) for x in r) for r in gm.gset}
    per_k = [{tuple(int(x) for x in r)
              for r in candidate_g_for_k(gm.k_super[k], gm.transform.m_eff,
                                         gm.transform.m_inv.T)}
             for k in range(gm.n_k)]
    assert any(s != gset for s in per_k)

    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)
    mapping = _candidate_to_grid(gm)
    assert len(set(mapping.values())) == gm.spec.n_points
    for (k, j), i in mapping.items():
        assert np.allclose(exp.band_e[i], res.energies[k, j])


def test_expand_matches_candidate_across_the_periodic_boundary():
    """匹配必须用**周期距离**: 0.99999997 与 0.0 是同一个点。

    构造: M = diag(2,1,1); 网格点顺序把 (0,0.5,0) 放在最前, 对称操作带
    ~3e-8 的平移残差 -> 代表点的像算出来是 0.99999997, 而真实格点是 0.0。
    若用非周期的距离, 网格点 (0,0,0) 会被匹配到另一个候选 -> 取错带。
    """
    m_eff = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    tr = Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    grid = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.0], [0.5, 0.0, 0.0]])
    ops = SymmetryOps(np.array([np.eye(3, dtype=int)]),
                      np.array([[0.0, 0.5 - TRANSLATION_EPS, 0.0]]))
    spec = GridSpec(p=(2, 2, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    b_prim = reciprocal(np.eye(3) * 3.0)
    gm = build_grid_map(spec, grid, ops, tr, b_prim, b_prim)

    res = _synthetic_unfolded(gm)
    exp = expand_to_full_grid(res, gm)

    mapping = _candidate_to_grid(gm, grid)
    (k_hit, j_hit) = [(k, j) for (k, j), i in mapping.items() if i == 1][0]
    g = candidate_g_for_k(gm.k_super[k_hit], gm.transform.m_eff,
                           gm.transform.m_inv.T)
    frac = (gm.k_super[k_hit] + g[j_hit]) @ gm.transform.m_inv.T
    # 网格点 1 的真实坐标是 (0,0,0), 必须取到 frac≈(0,0,0) 的那个候选
    assert np.allclose(frac, [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(exp.band_e[1], res.energies[k_hit, j_hit])
    # 同一 k 上的另一个候选(frac=(0.5,0,0))的能带块不能被取用
    (k_other, j_other) = [(k, j) for (k, j), i in mapping.items() if i == 3][0]
    assert not np.allclose(exp.band_e[1], res.energies[k_other, j_other])


# ------------------------------------------------------------------ 谱函数

def _expanded_with_padding():
    """两个网格点, 每个都带一个 +inf / 0 权重的补位槽。"""
    band_e = np.array([[1.0, np.inf], [2.0, np.inf]])
    band_w = np.array([[0.5, 0.0], [0.25, 0.0]])
    valid = np.array([[True, False], [True, False]])
    return ExpandedData(k_cart=np.zeros((2, 3)), band_e=band_e, band_w=band_w,
                         valid=valid,
                         spec=GridSpec(p=(2, 1, 1), grid_type="gamma",
                                       shift=(0.0, 0.0, 0.0)))


def test_build_spectral_function_matches_the_gaussian_sum():
    """A(k,E) = Σ_n w_n·exp(−(E−E_n)²/2σ²)。"""
    exp = _expanded_with_padding()
    e_grid = np.linspace(-5.0, 5.0, 41)
    sigma = 0.3
    a = build_spectral_function(exp, e_grid, sigma)
    assert a.shape == (2, e_grid.size)
    for i, (e0, w0) in enumerate([(1.0, 0.5), (2.0, 0.25)]):
        expect = w0 * np.exp(-((e_grid - e0) ** 2) / (2.0 * sigma ** 2))
        assert np.allclose(a[i], expect, rtol=1e-12, atol=1e-14)


def test_build_spectral_function_ignores_padding_slots():
    """补位槽(+inf 能量 / 0 权重)不得污染谱函数。"""
    exp = _expanded_with_padding()
    clean = ExpandedData(k_cart=exp.k_cart, band_e=exp.band_e[:, :1],
                         band_w=exp.band_w[:, :1],
                         valid=exp.valid[:, :1], spec=exp.spec)
    e_grid = np.linspace(-5.0, 5.0, 41)
    a = build_spectral_function(exp, e_grid, 0.3)
    a_clean = build_spectral_function(clean, e_grid, 0.3)
    assert np.all(np.isfinite(a))
    assert np.allclose(a, a_clean, rtol=1e-12, atol=1e-14)


def test_build_spectral_function_rejects_non_positive_sigma():
    exp = _expanded_with_padding()
    with pytest.raises(ValueError, match="sigma"):
        build_spectral_function(exp, np.linspace(-1, 1, 5), 0.0)
    with pytest.raises(ValueError, match="sigma"):
        build_spectral_function(exp, np.linspace(-1, 1, 5), -0.1)


# ------------------------------------------------------------------ 端到端

def test_expand_on_the_real_unfold_pipeline():
    """端到端(结构): fixture 的真实 PROCAR -> unfold_supercell -> 扩展到满网格。

    fixture 只有 2 个 k 点, 这里把 Γ 点的系数搬到网格的 4 个超胞 k 上,
    只验证**结构**(映射唯一性、权重守恒、谱函数有限), 不当物理基线。
    """
    prim, sc, tr = _fixture_geometry()
    gm, grid = _grid_map()
    band = read_eigenval(str(FIXTURE_EIGENVAL))
    procar = read_procar(str(FIXTURE_PROCAR))
    copies = build_atom_copies(prim, sc, tr)

    n_k = gm.n_k
    band2 = replace(band, kpoints=gm.k_super.copy(),
                     energies=np.repeat(band.energies[:1], n_k, axis=0))
    procar2 = replace(procar, kpoints=gm.k_super.copy(),
                      coeffs=np.repeat(procar.coeffs[:1], n_k, axis=0),
                      weights=np.repeat(procar.weights[:1], n_k, axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = unfold_supercell(band2, procar2, tr, copies, tr.m_eff,
                                prim.lattice, prim.positions @ prim.lattice,
                                reciprocal(prim.lattice),
                                reciprocal(sc.lattice), gm.k_super)

    exp = expand_to_full_grid(res, gm)
    assert exp.band_e.shape == (gm.spec.n_points, res.energies.shape[2])
    assert exp.band_w.sum() == pytest.approx(res.weights[res.valid].sum(),
                                             rel=1e-12)
    # 每个满网格点都取到了自己的候选块
    mapping = _candidate_to_grid(gm)
    for (k, j), i in mapping.items():
        assert np.allclose(exp.band_e[i], res.energies[k, j])
    # 谱函数有限(补位槽没把 +inf 带进展宽)
    e_grid = np.linspace(-20.0, 10.0, 61)
    a = build_spectral_function(exp, e_grid, 0.2)
    assert a.shape == (gm.spec.n_points, e_grid.size)
    assert np.all(np.isfinite(a))


# ------------------------------------------------------------------ 点群协变性
# spec §8 要求: **扩展后的 A(k',E) 在每个对称操作下数值不变**(阈值 1e-8)。
#
# 修复轮 5: 该性质原本**为假** —— 展开得到的谱权重在点群下不协变(最大相对
# 偏差 ~1.0, 即一个谱峰被映到 0)。修法是**按点群轨道平均 A(k,E)**(把展开谱
# 投影到"对称不变子空间"), 机制与实测见 docs/algorithms.md §15。

def _point_group_deviation(gm, grid, a):
    """每个对称操作下 A(k,E) 与 A(k',E) 的最大相对偏差(用 gm.ops)。"""
    return _deviation_under_ops(gm.ops, grid, a)


def _deviation_under_ops(ops, grid, a):
    """每个对称操作下 A(k,E) 与 A(k',E) 的最大相对偏差(逐操作)。"""
    from unfoldfs.geometry import periodic_distance, rotation_inverse
    devs = {}
    for o in range(ops.rotations.shape[0]):
        r = ops.rotations[o]
        t = ops.translations[o]
        img = np.mod(grid @ rotation_inverse(r) + t, 1.0)
        d = periodic_distance(grid, img)
        j = d.argmin(axis=1)
        on = np.nonzero(d[np.arange(grid.shape[0]), j] < 1e-6)[0]
        rel = [np.abs(a[i] - a[j[i]]).max()
               / max(a[i].max(), a[j[i]].max(), 1e-300) for i in on]
        devs[o] = float(max(rel)) if rel else 0.0
    return devs


def _fixture_pipeline_spectrum():
    """fixture 的真实 PROCAR -> unfold -> 扩展 -> A(k,E)(与端到端测试同口径)。"""
    prim, sc, tr = _fixture_geometry()
    gm, grid = _grid_map()
    band = read_eigenval(str(FIXTURE_EIGENVAL))
    procar = read_procar(str(FIXTURE_PROCAR))
    copies = build_atom_copies(prim, sc, tr)
    n_k = gm.n_k
    band2 = replace(band, kpoints=gm.k_super.copy(),
                     energies=np.repeat(band.energies[:1], n_k, axis=0))
    procar2 = replace(procar, kpoints=gm.k_super.copy(),
                      coeffs=np.repeat(procar.coeffs[:1], n_k, axis=0),
                      weights=np.repeat(procar.weights[:1], n_k, axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = unfold_supercell(band2, procar2, tr, copies, tr.m_eff,
                                prim.lattice, prim.positions @ prim.lattice,
                                reciprocal(prim.lattice),
                                reciprocal(sc.lattice), gm.k_super)
    exp = expand_to_full_grid(res, gm)
    a = build_spectral_function(exp, np.linspace(-20.0, 10.0, 61), 0.2)
    return gm, grid, a


def test_expanded_spectral_function_is_invariant_under_the_point_group():
    """spec §8: 展开 + 对称化后 A(k',E) 在每个对称操作下数值不变(<= 1e-8)。

    修复轮 5 把这条从 xfail 改成普通断言 —— 对称化后偏差降到机器精度。
    """
    gm, grid, a = _fixture_pipeline_spectrum()
    devs = _point_group_deviation(gm, grid, symmetrize_spectral_function(a, gm))
    assert devs, "没有可比的对称操作"
    worst = max(devs.values())
    assert worst <= 1e-8, (
        f"对称化后的 A(k,E) 在点群下仍不协变: 最大相对偏差 {worst:.3e} "
        f"(逐操作: { {k: f'{v:.3e}' for k, v in devs.items()} })")


def test_the_raw_spectrum_is_still_not_covariant():
    """锁住"**原始**(未对称化)的谱仍然不协变"。

    这条锁的是"修好靠的是**对称化**, 不是展开本身变了" —— 若有人改了
    unfold.py 让原始权重自己协变了, 本条会失败 ⇒ 提醒更新
    docs/algorithms.md §15(机制描述会过时)。
    """
    gm, grid, a = _fixture_pipeline_spectrum()
    devs = _point_group_deviation(gm, grid, a)
    worst = max(devs.values())
    assert worst >= 0.5, (
        f"原始谱的协变性偏差只有 {worst:.3e} —— 展开本身似乎被改了, "
        "请更新 docs/algorithms.md §15 与本文件的说明")
    assert worst <= 1.0 + 1e-9


def test_symmetrize_spectral_function_is_bit_identical_for_a_covariant_spectrum():
    """本来就没有对称性破缺时, 对称化必须**逐位**无操作。

    对应"保留全部 12 个操作的超胞"(如 √3×√3 R30° 莫尔超胞): 那里的谱权重
    本来就是协变的 ⇒ 轨道平均后必须**逐位不变**(不能引入 1 ulp 的噪声)。
    """
    gm, grid, a = _fixture_pipeline_spectrum()
    assert gm.ops.rotations.shape[0] == 12, "fixture 应当保留全部 12 个操作"
    a_cov = symmetrize_spectral_function(a, gm)      # 造一个协变的谱
    again = symmetrize_spectral_function(a_cov, gm)
    assert np.array_equal(again, a_cov), (
        "对称化对已经协变的谱不是逐位无操作 —— 会引入 1 ulp 量级的噪声")


def test_symmetrize_spectral_function_preserves_the_total_weight():
    """轨道平均**不改变**总权重: 每个点取轨道平均 ⇒ 逐轨道的权重和不变。"""
    gm, grid, a = _fixture_pipeline_spectrum()
    a_sym = symmetrize_spectral_function(a, gm)
    assert a_sym.shape == a.shape
    assert float(a_sym.sum()) == pytest.approx(float(a.sum()), rel=1e-12)


def test_symmetrize_spectral_function_leaves_a_single_point_orbit_alone():
    """只有一个点的轨道(恒等轨道)必须**原样**保留。"""
    gm, grid, a = _fixture_pipeline_spectrum()
    orbits = point_group_orbits(grid, gm.ops)
    singles = [o for o in orbits if o.size == 1]
    assert singles, "fixture 应当有单点轨道(否则本用例无意义)"
    a_sym = symmetrize_spectral_function(a, gm)
    for o in singles:
        assert np.array_equal(a_sym[o], a[o])


def test_point_group_orbits_match_the_grid_map_decomposition():
    """轨道分解必须与 ``grid_map.full_to_ir`` 一致。

    不一致 ⇒ 对称化的轨道与网格自己的轨道不是一回事 ⇒ 对称化会把不相干的
    点平均进来(docs/algorithms.md §15 的"只对可比的点做"就失效了)。
    """
    gm, grid = _grid_map()
    orbits = point_group_orbits(grid, gm.ops)
    by_orbit = sorted(tuple(sorted(o.tolist())) for o in orbits)
    ir = np.asarray(gm.full_to_ir, dtype=int)
    by_ir = sorted(tuple(sorted(np.nonzero(ir == k)[0].tolist()))
                   for k in np.unique(ir))
    assert by_orbit == by_ir


def test_point_group_orbits_partition_the_grid():
    """轨道必须是网格的一个**划分**(每个点恰好属于一个轨道)。"""
    gm, grid = _grid_map()
    orbits = point_group_orbits(grid, gm.ops)
    all_members = np.concatenate(orbits)
    assert sorted(all_members.tolist()) == list(range(grid.shape[0]))


def test_symmetrize_spectral_function_rejects_a_mismatched_grid():
    """A 的点数与网格点数不一致时必须报错(否则会静默取错带)。"""
    gm, grid, a = _fixture_pipeline_spectrum()
    with pytest.raises(ValueError, match="点数"):
        symmetrize_spectral_function(a[:-1], gm)


def test_the_symmetry_invariance_test_is_still_present():
    """回归锁: spec §8 要求的那条对称不变性测试不能被悄悄删掉。

    删掉它 = 取消了对"费米面存在对称性破缺"的公开披露 ⇒ 必须变红。
    """
    import pathlib
    import re
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    # 必须匹配**行首**的函数定义(本断言自身的字符串不算)
    found = re.search(r"^def test_expanded_spectral_function_"
                      r"is_invariant_under_the_point_group\(", src, re.M)
    assert found, "对称不变性测试被删掉了 —— 必须保留它作为回归锁"


def test_each_grid_point_is_covered_by_exactly_one_supercell_candidate():
    """折叠是 **1:1** 的: 每个原胞网格点只被**唯一**一个 (超胞 k, 候选) 覆盖。

    这条结构性事实决定了"构造 C(把所有落到 k' 的 (K,G) 贡献加起来)"在自洽网格上
    **退化成构造 A** —— 也就是 §15.2a 的判定实验的结论。

    证明(见 docs/algorithms.md §15.2a): 候选集合 {G : (K+G)·M⁻ᵀ ∈ [0,1)³} 与
    原胞 k' 一一对应; 两个不同的 K 若有共同的候选 k', 则
    k'·Mᵀ − K₁ ∈ Z³ 且 k'·Mᵀ − K₂ ∈ Z³ ⇒ K₁ ≡ K₂ (mod 1) ⇒ K₁ = K₂, 矛盾。
    """
    from unfoldfs.geometry import periodic_distance
    gm, grid = _grid_map()
    m_eff = np.asarray(gm.transform.m_eff, dtype=float)
    m_inv_T = np.asarray(gm.transform.m_inv, dtype=float).T
    n_cover = np.zeros(grid.shape[0], dtype=int)
    for k in range(gm.n_k):
        kk = np.mod(gm.k_super[k], 1.0)
        g = candidate_g_for_k(kk, m_eff, m_inv_T)
        cand = (kk + g) @ m_inv_T                  # (det,3) 该超胞 k 的全部候选
        d = periodic_distance(grid, cand)
        hit = d < 1e-6
        # 每个候选至多覆盖一个网格点(否则候选集与网格不是一一对应)
        assert hit.sum(axis=1).max() <= 1
        n_cover += hit.sum(axis=1)
    assert n_cover.min() >= 1, "有网格点没有被任何候选覆盖"
    assert n_cover.max() == 1, (
        f"有网格点被 {int(n_cover.max())} 个 (超胞 k, 候选) 覆盖 —— "
        "折叠不是 1:1 的, §15.2a 的'构造 C ≡ 构造 A'结论就不再成立")


# ------------------------------------------------------------------ 原胞完整点群
# 修复轮 6: 对称化必须用**原胞的完整点群**。
#
# 背景: 修复轮 5 的对称化用的是 ``grid_map.ops`` —— 那是"原胞与超胞的**公共**
# 对称操作"(判据 M^-T·R·M^T ∈ Z³)。对 1T-TaS2 的 3×1×1 超胞, 面内 C3 被破坏
# ⇒ 公共操作只剩 4/12 个 ⇒ 六重对称恢复不了(费米面仍然不对称)。
#
# ⇒ 现在用原胞的完整点群(12 个操作)做轨道平均: 原胞的谱函数按构造就在这个
# 点群下不变 ⇒ 把展开谱投影到"在这个点群下不变"的子空间是有物理依据的,
# 且**任何**超胞 M 都能这样折回原胞的对称性。

def _grid_map_symmetry_breaking(spec=SPEC):
    """M = diag(3,1,1) 的超胞网格图(破坏面内 C3 ⇒ 公共子群只剩 4 个操作)。

    超胞晶格直接由 ``M @ A_prim`` 算出(不需要真的写 POSCAR): 这里只做
    网格/轨道层面的测试。
    """
    prim, _sc, _tr = _fixture_geometry()
    m = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    tr = Transform(m, np.linalg.inv(m.astype(float)))
    grid = primitive_grid_points(spec)
    all_ops, dropped = common_symmetry_ops_detailed(prim, m)
    ops, dropped_grid = filter_ops_by_grid_detailed(all_ops, grid)
    gm = build_grid_map(spec, grid, ops, tr, reciprocal(prim.lattice),
                        reciprocal(m @ prim.lattice),
                        dropped=tuple(dropped) + tuple(dropped_grid))
    return gm, grid


def _synthetic_spectrum(gm, n_band=6, e_grid=None, sigma=0.2):
    """由**合成**展开结果造的 A(k,E)(各候选权重不同 ⇒ 天然不协变)。"""
    res = _synthetic_unfolded(gm, n_band=n_band)
    exp = expand_to_full_grid(res, gm)
    if e_grid is None:
        e_grid = np.linspace(-20.0, 10.0, 61)
    return build_spectral_function(exp, e_grid, sigma)


def test_symmetry_breaking_supercell_keeps_only_four_common_operations():
    """前提: M=diag(3,1,1) 确实把公共对称操作从 12 个砍到 4 个。

    没有这条, 下面的用例就可能因为"公共子群本来就等于完整点群"而空过。
    """
    gm, _grid = _grid_map_symmetry_breaking()
    assert gm.ops.rotations.shape[0] == 4
    assert primitive_symmetry_ops(_fixture_geometry()[0]).rotations.shape[0] == 12


def test_symmetrization_restores_the_full_primitive_point_group():
    """核心回归锁(修复轮 6): 破坏对称性的超胞上, 对称化后的 A(k,E) 在
    **原胞完整点群**(12 个操作)下协变(最大相对偏差 <= 1e-8)。

    修复轮 5 的实现只用 4 个公共操作 ⇒ 这条会红(偏差 ~1.0)。
    """
    gm, grid = _grid_map_symmetry_breaking()
    prim = _fixture_geometry()[0]
    full = filter_ops_by_grid(primitive_symmetry_ops(prim), grid)
    assert full.rotations.shape[0] == 12, "6x6x1 网格应当容得下整个点群"

    a = _synthetic_spectrum(gm)
    raw = _deviation_under_ops(full, grid, a)
    assert max(raw.values()) >= 0.1, (
        "合成的谱本来就不协变(否则本用例无意义): "
        f"{max(raw.values()):.3e}")

    a_sym = symmetrize_spectral_function(a, gm)
    devs = _deviation_under_ops(full, grid, a_sym)
    worst = max(devs.values())
    assert worst <= 1e-8, (
        f"对称化后的 A(k,E) 在原胞完整点群下仍不协变: 最大相对偏差 "
        f"{worst:.3e} (逐操作: "
        f"{ {k: f'{v:.3e}' for k, v in devs.items()} })")


def test_symmetrization_with_the_common_subgroup_alone_is_not_covariant():
    """对照: 显式只用**公共子群**(修复轮 5 的做法)时, 六重对称恢复不了。

    这条同时是"变异自检"的固化版本: 把默认实现退回 ``grid_map.ops``,
    上面的 ``test_symmetrization_restores_the_full_primitive_point_group``
    就会变红。
    """
    gm, grid = _grid_map_symmetry_breaking()
    prim = _fixture_geometry()[0]
    full = filter_ops_by_grid(primitive_symmetry_ops(prim), grid)
    a = _synthetic_spectrum(gm)
    a_common = symmetrize_spectral_function(a, gm, ops=gm.ops)
    devs = _deviation_under_ops(full, grid, a_common)
    assert max(devs.values()) >= 1e-3, (
        "只用公共子群时竟然也协变了 —— 前提(公共子群 < 完整点群)不成立?")


def test_full_group_symmetrization_is_idempotent():
    """完整点群对称化是**幂等**的: 再做一次不改变任何位。"""
    gm, _grid = _grid_map_symmetry_breaking()
    a = _synthetic_spectrum(gm)
    once = symmetrize_spectral_function(a, gm)
    twice = symmetrize_spectral_function(once, gm)
    assert np.array_equal(twice, once), (
        "对称化不是幂等的 —— 第二次又改了结果")


def test_full_group_symmetrization_is_bit_identical_for_a_covariant_spectrum():
    """本来就没有对称性破缺时, 完整点群对称化必须**逐位**无操作。

    对应"保留全部 12 个操作的超胞"(如 √3×√3 R30° 莫尔超胞)与 3×3×1 的
    fixture: 谱权重本来就是协变的 ⇒ 轨道平均不能引入 1 ulp 的噪声。
    """
    gm, grid, a = _fixture_pipeline_spectrum()
    prim = _fixture_geometry()[0]
    full = filter_ops_by_grid(primitive_symmetry_ops(prim), grid)
    assert full.rotations.shape[0] == 12
    a_cov = symmetrize_spectral_function(a, gm, ops=full)
    again = symmetrize_spectral_function(a_cov, gm, ops=full)
    assert np.array_equal(again, a_cov), (
        "完整点群对称化对已经协变的谱不是逐位无操作")


def test_full_point_group_is_recovered_from_the_grid_map():
    """``full_point_group(grid_map)`` 必须复原出原胞的完整点群。

    grid_map.ops 只装了公共子群, 被剔除的那些(带旋转与平移)记在
    ``grid_map.dropped`` 里 ⇒ 两者合起来 = 原胞点群。
    """
    gm, _grid = _grid_map_symmetry_breaking()
    recovered = full_point_group(gm)
    assert recovered.rotations.shape[0] == 12
    prim = _fixture_geometry()[0]
    truth = primitive_symmetry_ops(prim)
    key = lambda r: tuple(np.asarray(r, dtype=int).ravel())
    assert {key(r) for r in recovered.rotations} == \
        {key(r) for r in truth.rotations}

