"""unfold 模块测试: Popescu 投影法展开、副本相位还原平均、候选 G 重组。

fixture = 3x3x1 MoS2 超胞 (27 原子) + 原胞 (3 原子), M = diag(3,3,1), det(M)=9。
数值基线均来自 fixture 实测(与仓库根目录已验证的原型 unfold_procar.py 同源):
  * Γ 点最低带: 9 个候选中第 0 个单 G 主导(主导度 1.0), 其余权重 ~1e-33;
  * k=(0.1,0,0) 第 32 带: 4 种相位约定的主导度差异最大(0.52 / 0.96 / 0.98 / 0.70)。
"""
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unfoldfs.geometry import (candidate_g_vectors, read_poscar, read_transmat,
                               reciprocal, resolve_transform)
from unfoldfs.unfold import (CONVENTIONS, build_atom_copies, candidate_g_for_k,
                             select_phase_convention, unfold_at_k,
                             unfold_supercell)
from unfoldfs.vaspio import BandData, ProcarData, read_eigenval, read_procar
from tests.conftest import (FIXTURE_EIGENVAL, FIXTURE_POSCAR,
                            FIXTURE_PRIMCELL, FIXTURE_PROCAR, FIXTURE_TRANSMAT)

# 超胞展开路径的**逐位**基线(det(M)=9 的 fixture): 由"修复轮 4"之前的
# 代码生成, 用来锁住"新增 det(M)=1 直通分支后超胞路径逐位不变"。
_GOLDEN_DET9 = Path(__file__).parent / "data" / "unfold_supercell_det9_golden.npz"


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


def _unfold_supercell(fx, k_list=None, band=None, procar=None,
                      suppress_warnings=True, **kwargs):
    """跑 unfold_supercell。

    fixture 上相位约定的择优落在**退化区**(两个"好"约定只差 0.1%), 模块会
    告警 —— 这里默认收声(需要检查告警时传 suppress_warnings=False)。
    """
    args = (band if band is not None else fx["band"],
            procar if procar is not None else fx["procar"],
            fx["tr"], fx["copies"], fx["m_eff"], fx["a_prim"], fx["prim_cart"],
            fx["b_prim"], fx["b_super"],
            fx["band"].kpoints if k_list is None else k_list)
    if not suppress_warnings:
        return unfold_supercell(*args, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return unfold_supercell(*args, **kwargs)


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
    res = _unfold_supercell(fx, k_list)
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
    res = _unfold_supercell(fx, k_list)
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


# ==================================================================
# fix round 1/5 —— 相位约定的**生产路径**必须与已验证原型同口径
#
# 评审 Important 1: 旧的 unfold_supercell 用**全部 k × 全部带**择优,
# 原型 unfold_procar.py 只用 **首 4 个 k × 首 40 条带**; 两者在 fixture 上
# 选出**不同**的约定 -> 164/240 个 (k,带) 的候选归属改变 -> 重组后的
# 能带结构(即费米面)随之改变。
# ==================================================================

def _prototype_convention_scores(fx):
    """照抄原型 unfold_procar.py 的 [conv] 择优循环(独立代码路径)。

    原型口径: k 取前 min(4, nk) 个, 带取前 min(nband, 40) 条;
    每个 (k, 带) 算单 G 主导度 max(w)/sum(w)(总权重 > 1e-12 才计入),
    最后对全部样本取平均。
    """
    nk = fx["band"].kpoints.shape[0]
    nband = fx["band"].energies.shape[1]
    scores = {}
    for (s1, s2) in CONVENTIONS:
        peaks = []
        for ik in range(min(4, nk)):
            k = fx["band"].kpoints[ik]
            for ib in range(min(nband, 40)):
                _, w = _naive_weights(k, fx["procar"].coeffs[ik, ib:ib + 1],
                                      fx["copies"], fx["m_eff"], fx["a_prim"],
                                      fx["prim_cart"], fx["b_prim"], fx["b_super"],
                                      s1, s2)
                ws = w[:, 0]
                sw = ws.sum()
                if sw > 1e-12:
                    peaks.append(ws.max() / sw)
        scores[(s1, s2)] = float(np.mean(peaks))
    return scores


def test_unfold_supercell_selects_the_same_convention_as_the_validated_prototype():
    """约定择优的生产路径必须与原型 unfold_procar.py 同口径(评审 Important 1)。

    原型在 fixture 上选 (s1=-1, s2=+1); 旧的"全部 k × 全部带"口径会选 (1,-1),
    两者在 164/240 个 (k,带) 上候选归属不同。
    """
    fx = _fixture()
    proto = _prototype_convention_scores(fx)
    assert max(proto, key=proto.get) == (-1, 1)
    res = _unfold_supercell(fx)
    assert (res.s1, res.s2) == (-1, 1)
    # 模块的 4 个分数与原型的 [conv] 指标逐位相同
    for i, conv in enumerate(CONVENTIONS):
        assert res.dominance_report.scores[i] == pytest.approx(
            proto[conv], rel=1e-9, abs=1e-12)


def test_unfold_supercell_matches_naive_reference_on_the_full_path():
    """交叉核对改走 unfold_supercell 全路径(评审 Important 1)。

    报告 §4 原来的比对只在 unfold_at_k 层**手工指定**约定, 从没走过生产路径;
    这里用**独立**的朴素实现(照抄原型公式)重算权重, 再走一遍候选重组,
    要求重组后的 (E, w) 与模块逐位一致。
    """
    fx = _fixture()
    res = _unfold_supercell(fx)
    for ik in range(fx["band"].kpoints.shape[0]):
        k = fx["band"].kpoints[ik]
        _, w_naive = _naive_weights(k, fx["procar"].coeffs[ik], fx["copies"],
                                    fx["m_eff"], fx["a_prim"], fx["prim_cart"],
                                    fx["b_prim"], fx["b_super"], res.s1, res.s2)
        assign = w_naive.argmax(axis=0)
        for j in range(9):
            sel = np.nonzero(assign == j)[0]
            sel = sel[np.argsort(fx["band"].energies[ik, sel], kind="stable")]
            n = sel.size
            assert np.allclose(res.energies[ik, j, :n],
                               fx["band"].energies[ik, sel], rtol=0, atol=1e-12)
            assert np.allclose(res.weights[ik, j, :n], w_naive[j, sel],
                               rtol=1e-9, atol=1e-20)
            # 补位槽必须被 valid 掩码标出来
            assert not res.valid[ik, j, n:].any()


def test_unfold_supercell_convention_choice_changes_candidate_assignment():
    """记录约定选择的分量: 两种口径选出的约定在 164/240 个 (k,带) 上归属不同。

    这正是旧实现"改的是费米面本身"的量化证据。
    """
    fx = _fixture()
    assign = {}
    for conv in ((1, -1), (-1, 1)):
        rows = []
        for ik in range(fx["band"].kpoints.shape[0]):
            _, _, w = _unfold(fx, fx["band"].kpoints[ik], fx["procar"].coeffs[ik],
                              conv[0], conv[1])
            rows.append(w.argmax(axis=0))
        assign[conv] = rows
    diff = sum(int((assign[(1, -1)][ik] != assign[(-1, 1)][ik]).sum())
               for ik in range(len(assign[(1, -1)])))
    assert diff == 164


def test_unfold_supercell_reports_convention_margin_and_flags_degenerate_choice():
    """winner 与 runner-up 的 margin 必须随结果输出(评审 Important 2)。

    fixture 上两个"好"约定只差 0.1% —— 落在退化区, 选择是噪声驱动的,
    必须**可见**(告警), 不能静默。
    """
    fx = _fixture()
    with pytest.warns(UserWarning, match="退化"):
        res = _unfold_supercell(fx, suppress_warnings=False)
    rep = res.dominance_report
    assert rep.margin > 0
    assert rep.margin_rel == pytest.approx(rep.margin / rep.selection_score)
    assert rep.runner_up_score < rep.selection_score
    assert rep.degenerate is True
    assert rep.margin_rel < 1e-3


def test_unfold_supercell_reports_per_state_dominance_distribution():
    """每态主导度的分布(min / median / frac>0.85)必须随结果输出(评审 Important 3)。"""
    fx = _fixture()
    res = _unfold_supercell(fx)
    rep = res.dominance_report
    doms = []
    for ik in range(fx["band"].kpoints.shape[0]):
        _, _, w = _unfold(fx, fx["band"].kpoints[ik], fx["procar"].coeffs[ik],
                          res.s1, res.s2)
        tot = w.sum(axis=0)
        doms.extend(_dominance(w)[tot > 0].tolist())
    doms = np.array(doms)
    assert rep.mean == pytest.approx(doms.mean())
    assert rep.min == pytest.approx(doms.min())
    assert rep.median == pytest.approx(np.median(doms))
    assert rep.frac_above_good == pytest.approx(np.mean(doms > 0.85))
    assert res.dominance == pytest.approx(doms.mean())


def test_unfold_supercell_strict_dominance_flag_enforces_per_state_floor():
    """严格开关(简报自检①的严格版): 每个态的主导度都必须 >= 0.5。

    默认关闭 —— fixture 是合成数据, 有 10.8% 的态低于 0.5, 严格版会误报;
    真实 VASP 数据上应打开。
    """
    fx = _fixture()
    res = _unfold_supercell(fx)
    assert res.dominance_report.min < 0.5      # fixture 上确实有态不达标
    with pytest.raises(ValueError, match="每态"):
        _unfold_supercell(fx, strict_dominance=True)


def test_unfold_supercell_returns_valid_mask_for_padding_slots():
    """补位槽(能量 +inf、权重 0)必须被有效位掩码标出来, 否则下游 argmin 会选中它。"""
    fx = _fixture()
    res = _unfold_supercell(fx)
    assert res.valid is not None
    assert res.valid.shape == res.energies.shape
    assert np.array_equal(res.valid, res.weights > 0)
    assert np.all(np.isinf(res.energies[~res.valid]))
    assert np.all(np.isfinite(res.energies[res.valid]))
    # 确实存在补位槽(否则这条测试没有意义)
    assert not res.valid.all()


def test_unfold_supercell_warns_when_eigenval_and_procar_band_counts_differ():
    """EIGENVAL 与 PROCAR 的带数不一致时必须告警, 不能静默截断。"""
    fx = _fixture()
    procar = replace(fx["procar"],
                     coeffs=fx["procar"].coeffs[:, :100],
                     weights=fx["procar"].weights[:, :100])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _unfold_supercell(fx, procar=procar, suppress_warnings=False)
    msgs = [str(w.message) for w in caught]
    assert any("PROCAR" in m and "条带" in m for m in msgs), msgs


def test_unfold_supercell_reports_k_match_collapse_instead_of_silent_reuse():
    """两个不同的目标 k 落到同一个源 k(源 k 点集合缺一个)时必须报错。

    否则会静默复用同一组系数 -> 展开结果悄悄错掉。
    """
    fx = _fixture()
    band = replace(fx["band"], kpoints=fx["band"].kpoints[:1])   # 只留 Γ
    with pytest.raises(ValueError, match="塌缩"):
        _unfold_supercell(fx, band=band, k_list=fx["band"].kpoints)


def test_candidate_g_for_k_is_robust_to_k_outside_the_first_bz():
    """K 落在 [0,1)³ 之外时(如 -0.2 / 10.3)也必须给出同一组候选分数值。

    候选集的搜索范围按 |M| 行和给界, 该界隐含假设 K ∈ [0,1)³。
    """
    fx = _fixture()
    k = np.array([0.37, 0.62, 0.0])
    g = candidate_g_for_k(k, fx["m_eff"], fx["tr"].m_inv.T)
    ref = sorted(map(tuple, np.round((k + g) @ fx["tr"].m_inv.T, 9)))
    for k_shift in (k + 1.0, k - 1.0, k + 10.0, k + np.array([1.0, -1.0, 2.0])):
        g2 = candidate_g_for_k(k_shift, fx["m_eff"], fx["tr"].m_inv.T)
        got = sorted(map(tuple, np.round(
            (np.mod(k_shift, 1.0) + g2) @ fx["tr"].m_inv.T, 9)))
        assert got == ref


# ------------------------------------------------------------------ det(M)=1 直通模式
#
# det(M)=1 ⇒ "超胞"就是原胞, 没有折叠、没有多份副本。但工具原来的 Popescu
# 路径在数学上**退化**: 候选 G 只有一个(g=0) ⇒ 4 种相位约定的主导度必然
# 都是 1.0(完全相同)⇒ 择优守卫抛"4 种相位约定没有区分度", 工具连原胞
# 都算不了。
#
# ⇒ det(M)=1 时**不做展开**, 直接用原胞的带能量与轨道投影权重
#     w_n(k) = Σ_{ion,lm} |c_{ion,lm,n}(k)|²      (非相干和)
# 与超胞展开模式的权重(Popescu 的**相干** G 投影)不是同一个定义。

def _primitive_direct_fixture():
    """det(M)=1 的合成数据: 原胞当超胞(TRANSMAT = 单位阵)。

    PROCAR 的复投影系数是**已知**的 ⇒ 权重可以逐带逐 k 独立核对。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    tr = resolve_transform(prim, prim, np.eye(3, dtype=int))
    copies = build_atom_copies(prim, prim, tr)
    k_list = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0]])
    n_k, n_band, n_ions, norb = 2, 3, 3, 9
    rng = np.random.default_rng(20261118)
    coeffs = (rng.normal(size=(n_k, n_band, n_ions, norb))
              + 1j * rng.normal(size=(n_k, n_band, n_ions, norb)))
    # 第 2 个 k 的带能量**故意乱序** ⇒ 用来验证"按能量升序 + 权重跟着重排"
    energies = np.array([[-10.0, -4.0, 1.0], [1.0, -9.0, -3.0]])
    band = BandData(kpoints=k_list.copy(), energies=energies,
                    occupations=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
                    efermi=float("nan"))
    procar = ProcarData(kpoints=k_list.copy(), weights=np.abs(coeffs) ** 2,
                        coeffs=coeffs, has_phase=True)
    return {
        "prim": prim, "tr": tr, "copies": copies, "band": band,
        "procar": procar, "k_list": k_list, "coeffs": coeffs,
        "energies": energies,
        "b_prim": reciprocal(prim.lattice), "b_super": reciprocal(prim.lattice),
    }


def _unfold_primitive_direct(pd):
    return unfold_supercell(
        pd["band"], pd["procar"], pd["tr"], pd["copies"], pd["tr"].m_eff,
        pd["prim"].lattice, pd["prim"].positions @ pd["prim"].lattice,
        pd["b_prim"], pd["b_super"], pd["k_list"])


def test_primitive_direct_mode_does_not_raise_when_det_is_one():
    """det(M)=1 时**不能**抛"4 种相位约定没有区分度"。

    det(M)=1 只有一个候选 G ⇒ 4 种约定的主导度必然相同 ⇒ 原来的守卫会把
    原胞自己的计算也拒掉(用户要的正是原胞的费米面)。
    """
    pd = _primitive_direct_fixture()
    res = _unfold_primitive_direct(pd)
    assert res.energies.shape == (2, 1, 3)      # (n_k, det=1, n_band)
    assert res.weights.shape == (2, 1, 3)
    assert res.valid is not None and res.valid.all()


def test_primitive_direct_weights_are_the_orbital_projection_sums():
    """权重 = Σ_{ion,lm} |c_{ion,lm,n}(k)|²(非相干和), 逐带逐 k 核对。"""
    pd = _primitive_direct_fixture()
    res = _unfold_primitive_direct(pd)
    expect = np.sum(np.abs(pd["coeffs"]) ** 2, axis=(2, 3))     # (n_k, n_band)
    # 第 2 个 k 的带是乱序的 ⇒ 期望值也要按能量升序重排
    order = np.argsort(pd["energies"], axis=1, kind="stable")
    expect = np.take_along_axis(expect, order, axis=1)
    assert np.allclose(res.weights[:, 0, :], expect, rtol=1e-12, atol=0.0)


def test_primitive_direct_energies_are_sorted_ascending():
    pd = _primitive_direct_fixture()
    res = _unfold_primitive_direct(pd)
    e = res.energies[:, 0, :]
    assert np.all(np.diff(e, axis=1) >= 0.0)
    # 排序后的能量集合与 EIGENVAL 的能量集合相同(只是换了顺序)
    assert np.allclose(np.sort(pd["energies"], axis=1), e)


def test_primitive_direct_skips_the_convention_selection():
    """直通模式**没有**相位约定可择优 ⇒ DominanceReport 必须是"不适用"的填充值。

    不能伪造分数(比如硬写 1.0)—— 那会让读者以为做过择优。
    """
    pd = _primitive_direct_fixture()
    res = _unfold_primitive_direct(pd)
    rep = res.dominance_report
    assert rep is not None
    assert rep.scores == ()
    assert np.isnan(rep.selection_score)
    assert np.isnan(rep.runner_up_score)
    assert np.isnan(res.dominance)


def test_primitive_direct_is_not_the_popescu_coherent_weight():
    """直通模式(Σ|c|²)与展开模式(Popescu 相干 G 投影)不是同一个定义。

    det(M)=1 时 Popescu 权重会退化成 |Σ_i e^{i k'·r_i} Σ_lm c_{i,lm}|² ——
    逐原子的**相干**求和。两者必须不同(否则文档里"两种模式"的说法是假的)。
    """
    pd = _primitive_direct_fixture()
    res = _unfold_primitive_direct(pd)
    c = pd["coeffs"]
    # Popescu(相干): 先在原胞原子上相干求和, 再取模方
    bloch = np.exp(1j * ((pd["k_list"] @ pd["b_prim"])
                         @ (pd["prim"].positions @ pd["prim"].lattice).T))  # (n_k,n_ions)
    coherent = np.abs(np.einsum("ki,kbin->kb", bloch, c)) ** 2
    incoherent = np.sum(np.abs(c) ** 2, axis=(2, 3))
    assert not np.allclose(coherent, incoherent, rtol=1e-6)
    # 而工具给的是**非相干**的那一个
    assert np.allclose(res.weights[:, 0, :],
                       np.take_along_axis(incoherent,
                                          np.argsort(pd["energies"], axis=1,
                                                     kind="stable"), axis=1),
                       rtol=1e-12, atol=0.0)


def test_supercell_path_is_bit_identical_after_the_primitive_direct_branch():
    """回归锁: det(M)=9 的超胞展开路径**逐位不变**。

    基线是"修复轮 4"之前(只有超胞路径)的代码生成的 npz。
    """
    fx = _fixture()
    res = _unfold_supercell(fx)
    g = np.load(_GOLDEN_DET9)
    assert np.array_equal(res.k_cand, g["k_cand"])
    assert np.array_equal(res.energies, g["energies"])
    assert np.array_equal(res.weights, g["weights"])
    assert np.array_equal(res.valid, g["valid"])
    assert (res.s1, res.s2) == (int(g["s1"]), int(g["s2"]))
    assert res.dominance == float(g["dominance"])
    assert res.dominance_report.scores == tuple(g["scores"])
