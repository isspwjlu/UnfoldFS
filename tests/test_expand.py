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
                             expand_to_full_grid)
from unfoldfs.geometry import (SymmetryOps, Transform, common_symmetry_ops,
                               read_poscar, read_transmat, reciprocal,
                               resolve_transform)
from unfoldfs.grid import (GridSpec, build_grid_map, filter_ops_by_grid,
                           primitive_grid_points)
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
# (C1) spec §8 要求: **扩展后的 A(k',E) 在每个对称操作下数值不变**(阈值 1e-8)。
# 实测: 该性质**为假** —— 谱权重在点群下不协变(最大相对偏差 ~1.0)。
# 这里把这条要求**原样写成测试**: 它现在是红的(xfail), 作为回归锁留着。

def _point_group_deviation(gm, grid, a):
    """每个对称操作下 A(k,E) 与 A(k',E) 的最大相对偏差。"""
    from unfoldfs.geometry import periodic_distance, rotation_inverse
    devs = {}
    for o in range(gm.ops.rotations.shape[0]):
        r = gm.ops.rotations[o]
        t = gm.ops.translations[o]
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


@pytest.mark.xfail(
    strict=True,
    reason="C1(已知缺陷, 未修): 展开后的 A(k,E) 在晶体点群下**不**协变 —— "
           "谱权重在对称相关点之间不一致(实测最大相对偏差 ~1.0, 即一个谱峰被"
           "映到 0)。机制与实测见 docs/algorithms.md §14 与 §15。"
           "修好之后请把本条改成普通断言(并删掉下面两条锁)。")
def test_expanded_spectral_function_is_invariant_under_the_point_group():
    """spec §8: 扩展后 A(k',E) 在每个对称操作下数值不变(<= 1e-8)。"""
    gm, grid, a = _fixture_pipeline_spectrum()
    devs = _point_group_deviation(gm, grid, a)
    assert devs, "没有可比的对称操作"
    worst = max(devs.values())
    assert worst <= 1e-8, (
        f"A(k,E) 在点群下不协变: 最大相对偏差 {worst:.3e} "
        f"(逐操作: { {k: f'{v:.3e}' for k, v in devs.items()} })")


def test_point_group_covariance_defect_stays_within_the_documented_band():
    """把已知缺陷的**量级**锁住: fixture 上 12 个操作的偏差都是 ~1.0。

    这条锁的是"缺陷还在、且没有变本加厉" —— 缺陷若被修好, 上面那条 xfail 会
    变成 XPASS(strict) 而报错, 这条也会失败 ⇒ 两者一起提醒人更新文档。
    """
    gm, grid, a = _fixture_pipeline_spectrum()
    devs = _point_group_deviation(gm, grid, a)
    worst = max(devs.values())
    assert worst >= 0.5, (
        f"协变性偏差只有 {worst:.3e} —— 缺陷可能被修好了(或数据变了), "
        "请更新 docs/algorithms.md §14/§15 与本文件的 xfail 标记")
    assert worst <= 1.0 + 1e-9


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
