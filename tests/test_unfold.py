"""unfold 模块测试: Popescu 投影法展开、副本相位还原平均、候选 G 重组。

fixture = 3x3x1 MoS2 超胞 (27 原子) + 原胞 (3 原子), M = diag(3,3,1), det(M)=9。
数值基线均来自 fixture 实测(与仓库根目录已验证的原型 unfold_procar.py 同源):
  * Γ 点最低带: 9 个候选中第 0 个单 G 主导(主导度 1.0), 其余权重 ~1e-33;
  * k=(0.1,0,0) 第 32 带: 4 种相位约定的主导度差异最大(0.52 / 0.96 / 0.98 / 0.70)。
"""
import numpy as np
import pytest

from unfoldfs.geometry import (candidate_g_vectors, read_poscar, read_transmat,
                               reciprocal, resolve_transform)
from unfoldfs.unfold import (build_atom_copies, candidate_g_for_k,
                             select_phase_convention, unfold_at_k,
                             unfold_supercell)
from unfoldfs.vaspio import read_eigenval, read_procar
from tests.conftest import (FIXTURE_EIGENVAL, FIXTURE_POSCAR,
                            FIXTURE_PRIMCELL, FIXTURE_PROCAR, FIXTURE_TRANSMAT)


def _fixture():
    """fixture 的几何 + 数据 + 副本表(测试里到处都要用, 统一在这里组装)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    tr = resolve_transform(sc, prim, read_transmat(str(FIXTURE_TRANSMAT)))
    return {
        "prim": prim,
        "sc": sc,
        "tr": tr,
        "copies": build_atom_copies(prim, sc, tr),
        "band": read_eigenval(str(FIXTURE_EIGENVAL)),
        "procar": read_procar(str(FIXTURE_PROCAR)),
        "m_eff": tr.m_eff,
        "a_prim": prim.lattice,
        "prim_cart": prim.positions @ prim.lattice,
        "b_prim": reciprocal(prim.lattice),
        "b_super": reciprocal(sc.lattice),
    }


def _unfold(fx, k_frac, coeffs, s1, s2):
    """对单个 k 展开, 返回 (k_cand, E, w)。"""
    return unfold_at_k(
        k_frac, coeffs, np.zeros(coeffs.shape[0]), fx["tr"], fx["copies"],
        fx["m_eff"], fx["a_prim"], fx["prim_cart"], fx["b_prim"],
        fx["b_super"], s1, s2)


def _dominance(w):
    """每条带在 det(M) 个候选上的单 G 主导度 max(w)/sum(w)。"""
    tot = w.sum(axis=0)
    out = np.zeros_like(tot)
    ok = tot > 0
    out[ok] = w.max(axis=0)[ok] / tot[ok]
    return out


def _naive_weights(k_frac, coeffs, copies, m_eff, a_prim, prim_cart,
                   b_prim, b_super, s1, s2):
    """原型 unfold_procar.py 的逐元素实现(独立于模块的向量化实现)。

    用来交叉核对模块: 两条代码路径独立, 权重必须逐位一致。
    """
    m_inv_t = np.linalg.inv(np.asarray(m_eff, dtype=float)).T
    g = candidate_g_for_k(k_frac, m_eff, m_inv_t)
    k_super_cart = np.asarray(k_frac, dtype=float) @ b_super
    out = np.zeros((g.shape[0], coeffs.shape[0]))
    for gi, gg in enumerate(g):
        k_cand_cart = ((np.asarray(k_frac, dtype=float) + gg) @ m_inv_t) @ b_prim
        d_g = k_cand_cart - k_super_cart
        ph = np.exp(1j * s2 * (k_cand_cart @ prim_cart.T))
        for bi in range(coeffs.shape[0]):
            cvec = np.zeros((len(copies.copies), coeffs.shape[2]), dtype=complex)
            for i, lst in enumerate(copies.copies):
                acc = np.zeros(coeffs.shape[2], dtype=complex)
                for (j, t) in lst:
                    acc += coeffs[bi, j, :] * np.exp(
                        1j * s1 * float(d_g @ (np.asarray(t, float) @ a_prim)))
                cvec[i] = acc / len(lst)
            out[gi, bi] = float(np.abs(np.sum(cvec * ph[:, None])) ** 2)
    return g, out


# ------------------------------------------------------------------ 副本表

def test_build_atom_copies_gives_nine_copies_per_primitive_atom():
    """每个原胞原子在超胞里有 det(M)=9 个副本。"""
    fx = _fixture()
    assert fx["prim"].n_atoms == 3
    assert fx["sc"].n_atoms == 27
    counts = [len(lst) for lst in fx["copies"].copies]
    assert counts == [9, 9, 9]


def test_build_atom_copies_translations_span_the_3x3_supercell():
    """副本平移 T 以原胞格矢为单位, Mo 的 9 个副本覆盖 3x3 平移。"""
    fx = _fixture()
    mo = fx["copies"].copies[0]
    assert sorted(int(j) for j, _ in mo) == list(range(9))
    trans = sorted(tuple(int(x) for x in t) for _, t in mo)
    assert trans == sorted((i, j, 0) for i in range(3) for j in range(3))


def test_build_atom_copies_matches_cartesian_difference():
    """T 必须由笛卡尔差算出: 副本位置 = 原胞位置 + T·A_prim。"""
    fx = _fixture()
    prim, sc = fx["prim"], fx["sc"]
    for i, lst in enumerate(fx["copies"].copies):
        p_cart = prim.positions[i] @ prim.lattice
        for (j, t) in lst:
            q_cart = sc.positions[j] @ sc.lattice
            assert np.allclose(p_cart + np.asarray(t, float) @ prim.lattice,
                               q_cart, atol=1e-6)


# ------------------------------------------------------------------ 候选 G

def test_candidate_g_for_k_returns_det_candidates_inside_bz():
    fx = _fixture()
    for k in (np.zeros(3), np.array([0.1, 0.0, 0.0]),
              np.array([0.37, 0.62, 0.0])):
        g = candidate_g_for_k(k, fx["m_eff"], fx["tr"].m_inv.T)
        assert g.shape == (9, 3)
        frac = (k + g) @ fx["tr"].m_inv.T
        assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)


def test_candidate_g_for_k_at_gamma_agrees_with_geometry_candidate_g_vectors():
    """K=0 时逐 k 计算的候选与 geometry.candidate_g_vectors 给出同一组分数值。

    只比**分数值集合**(g 的成员可以不同), 这是两个实现口径一致的可观测量。
    """
    fx = _fixture()
    g_geo = candidate_g_vectors(fx["m_eff"])
    frac_geo = g_geo @ fx["tr"].m_inv.T
    g = candidate_g_for_k(np.zeros(3), fx["m_eff"], fx["tr"].m_inv.T)
    frac = g @ fx["tr"].m_inv.T
    assert sorted(map(tuple, np.round(frac, 9))) == \
        sorted(map(tuple, np.round(frac_geo, 9)))


def test_candidate_g_for_k_shifts_with_k_for_non_diagonal_transform():
    """一般 M 下候选集随 K 平移 —— fixture 的对角 M 无法区分, 这里用合成 M。

    用非对角 M 时, 固定 gset 会给出错误的候选, 故必须逐 k 计算。
    """
    m_eff = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    m_inv_t = np.linalg.inv(m_eff.astype(float)).T
    det = int(round(abs(np.linalg.det(m_eff))))
    sets = []
    for k in (np.zeros(3), np.array([0.13, 0.41, 0.0]),
              np.array([0.77, 0.05, 0.0])):
        g = candidate_g_for_k(k, m_eff, m_inv_t)
        assert g.shape == (det, 3)
        frac = (k + g) @ m_inv_t
        assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)
        sets.append({tuple(np.round(x, 9)) for x in frac})
    # K=0 的候选集 == geometry.candidate_g_vectors 的分数值集合
    frac_geo = candidate_g_vectors(m_eff) @ m_inv_t
    assert sets[0] == {tuple(np.round(x, 9)) for x in frac_geo}
    # 非零 K 的候选集与 K=0 不同(否则"逐 k 计算"就没有意义)
    assert sets[1] != sets[0]
    assert sets[2] != sets[0]


# ------------------------------------------------------------------ 展开

def test_candidate_g_for_k_finds_all_candidates_for_large_offdiagonal_transform():
    """搜索范围必须按 |M| 的**行和**给界, 否则会漏掉离对角元素大的候选。

    这个 M 的合法候选里 g 的第三个分量可达 5, 而 max|M|+2 = 5 的立方体搜索
    范围不够 —— 用行和界(6,6,7)才找得全 16 个。
    """
    m_eff = np.array([[2, -3, 0], [1, -3, 1], [3, 2, 1]])
    m_inv_t = np.linalg.inv(m_eff.astype(float)).T
    det = int(round(abs(np.linalg.det(m_eff.astype(float)))))
    assert det == 16
    for k in (np.zeros(3), np.array([0.137, 0.419, 0.815])):
        g = candidate_g_for_k(k, m_eff, m_inv_t)
        assert g.shape == (det, 3)
        frac = (k + g) @ m_inv_t
        assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)
        # 候选互不重复
        assert len({tuple(int(x) for x in row) for row in g}) == det


def test_unfold_at_k_gamma_lowest_band_has_single_dominant_candidate():
    """Γ 点最低带: 9 个候选, 其中恰好 1 个主导(主导度 1.0)。"""
    fx = _fixture()
    k = fx["band"].kpoints[0]
    k_cand, energies, w = _unfold(fx, k, fx["procar"].coeffs[0], 1, 1)
    assert k_cand.shape == (9, 3)
    assert energies.shape == (9, fx["band"].energies.shape[1])
    assert w.shape == energies.shape
    band0 = w[:, 0]
    assert band0.argmax() == 0
    assert band0.max() / band0.sum() == pytest.approx(1.0, abs=1e-12)
    # 其余候选的权重掉到数值零(原型实测 ~1e-33)
    assert band0[1:].max() < 1e-30
    # 主导候选就是 Γ 本身
    assert np.allclose(k_cand[0], np.zeros(3))


def test_unfold_at_k_matches_naive_reference_implementation():
    """与原型逐元素实现的权重逐位一致(独立代码路径交叉核对)。"""
    fx = _fixture()
    for ik in (0, 1):
        k = fx["band"].kpoints[ik]
        _, _, w = _unfold(fx, k, fx["procar"].coeffs[ik], -1, 1)
        _, w_naive = _naive_weights(k, fx["procar"].coeffs[ik], fx["copies"],
                                    fx["m_eff"], fx["a_prim"], fx["prim_cart"],
                                    fx["b_prim"], fx["b_super"], -1, 1)
        assert np.allclose(w, w_naive, rtol=1e-9, atol=1e-20)


def test_unfold_at_k_phase_restoring_average_recovers_coherent_amplitude():
    """相位还原的副本平均: 副本带相位 e^{∓i(k'−K)·T} 时应被还原成同一振幅。

    合成用例: 取一个 dG≠0 的候选 g*, 令副本值 = A_i·exp(-i·dG·T),
    则 s1=+1 的副本平均恰好还原出 A_i -> 权重全部集中到 g* 上;
    反过来的相位符号使副本不再相干 -> 该态被完全投影掉。
    """
    fx = _fixture()
    k = np.array([0.1, 0.0, 0.0])
    g = candidate_g_for_k(k, fx["m_eff"], fx["tr"].m_inv.T)
    k_cand_frac = (k + g) @ fx["tr"].m_inv.T
    k_cand_cart = k_cand_frac @ fx["b_prim"]
    d_g = k_cand_cart - k @ fx["b_super"]
    target = int(np.argmax(np.linalg.norm(d_g, axis=1)))   # dG≠0 的候选
    assert np.linalg.norm(d_g[target]) > 1e-3

    amps = np.array([1.0 + 0.0j, 0.5 + 0.0j, 0.25 + 0.0j])
    coeffs = np.zeros((1, fx["sc"].n_atoms, 1), dtype=complex)
    for i, lst in enumerate(fx["copies"].copies):
        for (j, t) in lst:
            shift = float(d_g[target] @ (np.asarray(t, float) @ fx["a_prim"]))
            coeffs[0, j, 0] = amps[i] * np.exp(-1j * shift)

    _, _, w = _unfold(fx, k, coeffs, 1, 1)
    assert _dominance(w)[0] == pytest.approx(1.0, abs=1e-9)
    assert w[target, 0] > 0
    assert w.sum() == pytest.approx(w[target, 0], rel=1e-9)
    # 反过来的相位符号 -> 副本不再相干 -> 态被搬到"镜像"候选上(而不是消失)
    _, _, w_wrong = _unfold(fx, k, coeffs, -1, 1)
    assert w_wrong[target, 0] < 1e-20 * w.sum()
    assert w_wrong.argmax() != target
    assert _dominance(w_wrong)[0] == pytest.approx(1.0, abs=1e-9)


def test_unfold_at_k_wrong_phase_convention_lowers_dominance():
    """负向测试: 用错的相位约定 -> 同一态的主导度下降(fixture 基线)。"""
    fx = _fixture()
    k = fx["band"].kpoints[1]
    coeffs = fx["procar"].coeffs[1]
    dom = {}
    for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _, _, w = _unfold(fx, k, coeffs, s1, s2)
        dom[(s1, s2)] = _dominance(w)[32]
    assert dom[(-1, 1)] == pytest.approx(0.9832, abs=1e-3)
    assert dom[(1, 1)] == pytest.approx(0.5178, abs=1e-3)
    assert dom[(-1, 1)] > dom[(1, 1)]


# ------------------------------------------------------------------ 相位约定择优

def test_select_phase_convention_returns_the_max_dominance_convention():
    fx = _fixture()
    doms = {}
    for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _, _, w = _unfold(fx, fx["band"].kpoints[1], fx["procar"].coeffs[1],
                          s1, s2)
        doms[(s1, s2)] = float(_dominance(w).mean())
    s1, s2, dominance = select_phase_convention(
        fx["band"].kpoints[1], fx["procar"].coeffs[1],
        fx["band"].energies[1], fx["tr"], fx["copies"], fx["m_eff"],
        fx["a_prim"], fx["prim_cart"], fx["b_prim"], fx["b_super"])
    assert (s1, s2) == max(doms, key=doms.get)
    assert dominance == pytest.approx(doms[(s1, s2)], rel=1e-9)


def test_select_phase_convention_raises_when_dominance_is_too_low():
    """主导度 < 0.5 -> 抛错(展开没有把态集中到唯一一个候选 G 上)。"""
    fx = _fixture()
    rng = np.random.default_rng(20261117)
    coeffs = rng.normal(size=(20, fx["sc"].n_atoms, 9)) \
        + 1j * rng.normal(size=(20, fx["sc"].n_atoms, 9))
    with pytest.raises(ValueError, match="主导度"):
        select_phase_convention(
            fx["band"].kpoints[0], coeffs, fx["band"].energies[0][:20],
            fx["tr"], fx["copies"], fx["m_eff"], fx["a_prim"],
            fx["prim_cart"], fx["b_prim"], fx["b_super"])


def test_select_phase_convention_raises_when_conventions_are_indistinguishable():
    """4 种约定给不出区分 -> 约定本身没有意义 -> 抛错。

    合成用例: det(M)=1 且只有 1 个原子 1 个副本 —— 相位因子全部退化,
    4 种约定给出完全相同的主导度。
    """
    fx = _fixture()
    prim, sc = fx["prim"], fx["sc"]
    m_eff = np.eye(3, dtype=int)
    from unfoldfs.geometry import Transform
    tr = Transform(m_eff, np.eye(3))
    prim1 = type(prim)(prim.lattice, prim.positions[:1], ("Mo",))
    sc1 = type(sc)(sc.lattice, sc.positions[:1], ("Mo",))
    copies = build_atom_copies(prim1, sc1, tr)
    coeffs = np.array([[[0.3 + 0.4j]]])
    with pytest.raises(ValueError, match="区分度"):
        select_phase_convention(
            np.zeros(3), coeffs, np.array([0.0]), tr, copies, m_eff,
            prim1.lattice, prim1.positions @ prim1.lattice,
            reciprocal(prim1.lattice), reciprocal(sc1.lattice))


# ------------------------------------------------------------------ 候选重组

def test_unfold_supercell_shapes_and_candidate_consistency():
    fx = _fixture()
    k_list = fx["band"].kpoints
    res = unfold_supercell(fx["band"], fx["procar"], fx["tr"], fx["copies"],
                            fx["m_eff"], fx["a_prim"], fx["prim_cart"],
                            fx["b_prim"], fx["b_super"], k_list)
    n_k, det_m = k_list.shape[0], 9
    assert res.k_cand.shape == (n_k, det_m, 3)
    assert res.energies.shape[:2] == (n_k, det_m)
    assert res.weights.shape == res.energies.shape
    assert res.energies.shape[2] == res.weights.shape[2] > 0
    # 候选原胞 k 与逐 k 计算的结果一致
    for ik, k in enumerate(k_list):
        g = candidate_g_for_k(k, fx["m_eff"], fx["tr"].m_inv.T)
        assert np.allclose(res.k_cand[ik], (k + g) @ fx["tr"].m_inv.T)


def test_unfold_supercell_recombination_assigns_each_band_to_one_candidate():
    """候选重组: 每条超胞带只归属到权重最大的那一个候选, 候选内按能量排序。"""
    fx = _fixture()
    k_list = fx["band"].kpoints
    res = unfold_supercell(fx["band"], fx["procar"], fx["tr"], fx["copies"],
                            fx["m_eff"], fx["a_prim"], fx["prim_cart"],
                            fx["b_prim"], fx["b_super"], k_list)
    for ik, k in enumerate(k_list):
        assigned = 0
        for j in range(9):
            e = res.energies[ik, j]
            w = res.weights[ik, j]
            valid = w > 0
            assert np.all(np.diff(e[valid]) >= -1e-12)      # 按能量升序
            assert np.all(np.isinf(e[~valid]))              # 补位处能量为 +inf
            assigned += int(valid.sum())
        # 每条(权重非零的)带恰好归属到一个候选
        _, _, w_all = _unfold(fx, k, fx["procar"].coeffs[ik], res.s1, res.s2)
        assert assigned == int((w_all.sum(axis=0) > 0).sum())
        # 重组保留的是每条带在其归属候选上的权重(主权重), 泄漏到别的候选的
        # 那一部分按设计丢弃
        dominant = w_all.max(axis=0)
        assert res.weights[ik].sum() == pytest.approx(
            dominant[dominant > 0].sum(), rel=1e-9)


def test_unfold_supercell_rejects_k_point_absent_from_eigenval():
    fx = _fixture()
    with pytest.raises(ValueError, match="k 点"):
        unfold_supercell(fx["band"], fx["procar"], fx["tr"], fx["copies"],
                          fx["m_eff"], fx["a_prim"], fx["prim_cart"],
                          fx["b_prim"], fx["b_super"],
                          np.array([[0.123, 0.456, 0.0]]))
