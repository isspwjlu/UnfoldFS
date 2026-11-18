import numpy as np
import pytest

from unfoldfs.geometry import (
    Structure, read_poscar, reciprocal, read_transmat,
    resolve_transform, matrix_denominator_lcm, candidate_g_vectors,
    common_symmetry_ops, periodic_delta, primitive_symmetry_ops,
)
from tests.conftest import (
    DATA_DIR, FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT,
    FIXTURE_EIGENVAL, FIXTURE_DOSCAR, FIXTURE_PROCAR,
)


@pytest.mark.parametrize("path", [FIXTURE_POSCAR, FIXTURE_PRIMCELL,
                                  FIXTURE_TRANSMAT, FIXTURE_EIGENVAL,
                                  FIXTURE_DOSCAR, FIXTURE_PROCAR,
                                  DATA_DIR / "INCAR"])
def test_fixture_is_present(path):
    assert path.is_file() and path.stat().st_size > 0


def test_read_poscar_sc():
    sc = read_poscar(str(FIXTURE_POSCAR))
    assert sc.n_atoms == 27
    assert sc.species[0] == "Mo"
    assert sc.lattice.shape == (3, 3)
    # 超胞第一条晶格矢量 = 3 × 原胞第一条
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    assert np.allclose(sc.lattice[0], 3.0 * prim.lattice[0])


def test_reciprocal_is_inverse_transpose():
    lat = np.array([[3.0, 0.0, 0.0], [-1.5, 2.763, 0.0], [0.0, 0.0, 17.0]])
    b = reciprocal(lat)
    assert np.allclose(b @ lat.T, 2.0 * np.pi * np.eye(3))


def test_read_transmat():
    m = read_transmat(str(FIXTURE_TRANSMAT))
    assert m.tolist() == [[3, 0, 0], [0, 3, 0], [0, 0, 1]]


def test_resolve_transform_detects_row_convention():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    tr = resolve_transform(sc, prim, m)
    assert np.allclose(tr.m_eff, m)


def test_resolve_transform_detects_column_convention():
    """同样的物理超胞, TRANSMAT 写成列约定 -> 应自动识别并转成行约定。

    注意: fixture 的 M 是对角的(M.T == M), 无法区分两种约定,
    因此这里用**非对角** M 构造合成用例。
    """
    prim = Structure(
        lattice=np.array([[2.0, 0.0, 0.0], [-1.0, 1.7320508, 0.0],
                          [0.0, 0.0, 10.0]]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.5]]),
        species=("A", "B"),
    )
    m = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])   # 非对角, det=4
    sc = Structure(lattice=m @ prim.lattice,
                   positions=np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.5]]),
                   species=("A", "B"))
    tr_row = resolve_transform(sc, prim, m)
    assert np.allclose(tr_row.m_eff, m)
    tr_col = resolve_transform(sc, prim, m.T)
    # 列约定写法必须被还原成同一个行约定矩阵
    assert np.allclose(tr_col.m_eff, m)
    assert tr_col.det == 4


def test_resolve_transform_rejects_inconsistent():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    with pytest.raises(ValueError):
        resolve_transform(sc, prim, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]]))


def test_candidate_g_vectors_count_and_range():
    m = read_transmat(str(FIXTURE_TRANSMAT))
    g = candidate_g_vectors(m)
    assert g.shape == (9, 3)
    inv = np.linalg.inv(m)
    frac = g @ inv.T
    assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)


def test_denominator_lcm():
    assert matrix_denominator_lcm(np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])) == 3
    assert matrix_denominator_lcm(np.array([[2, 1], [0, 2]])) == 4


def test_common_symmetry_ops_rotations_are_integer_in_supercell_basis():
    """判据 = 旋转条件 M^-T R M^T ∈ Z³ (该旋转把超胞晶格映到自身)。

    spglib 的旋转矩阵是**列向量**约定 (x' = R x), 实测 12/12 个原胞操作
    按列向量作用能把原子映回自身、按行向量只有 4/12 —— 故超胞基底下的
    旋转是 M^-T R M^T (行向量约定下才是 M R M^-1, 两者对对称 M 恒等)。

    平移**不参与**筛选: spglib 给的 t 是模原胞格矢的代表元 (C3^2 的代表元是
    (1,0,0)), 换基底后 t·M^-1 未必是整数, 用它筛选会误删操作。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    assert ops.rotations.shape[0] >= 1
    inv = np.linalg.inv(m.astype(float))
    for r in ops.rotations:
        r_sc = inv.T @ r.astype(float) @ m.astype(float).T
        assert np.allclose(r_sc, np.round(r_sc))


def test_common_symmetry_ops_translations_are_hardened():
    """平移必须硬化: spglib 给 C3^2 的代表元是 (0.9999999999999999, 0, 0),
    与 0 等价但数值上 != 0; 原样存下会让下游 int(mod(t*n,1)*n) 静默错一格。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    for t in ops.translations:
        # fixture 的 12 个操作的平移代表元都等价于 0 -> 硬化后必须精确为 0
        assert np.allclose(t, 0.0, atol=1e-6)


def test_common_symmetry_ops_translations_congruent_to_spglib():
    """存下的平移必须与 spglib 原始返回值在 mod 1 下同余(硬化不改变等价类)。"""
    import spglib

    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    ids = np.array([sorted(set(prim.species)).index(s) + 1 for s in prim.species],
                   dtype=int)
    sym = spglib.get_symmetry((prim.lattice, prim.positions, ids), symprec=1e-5)
    raw = sym["translations"]
    assert raw.shape[0] == ops.translations.shape[0]
    for t_raw, t_stored in zip(raw, ops.translations):
        d = np.mod(t_raw - t_stored + 0.5, 1.0) - 0.5
        assert np.allclose(d, 0.0, atol=1e-6)


def test_common_symmetry_ops_count_for_fixture():
    """MoS2 3x3x1 超胞与原胞相容, 公共对称集 = 原胞 D3h 的全部 12 个操作。

    平移不参与筛选: spglib 返回的 t 是模原胞格矢的代表元(spglib 给 C3^2 的
    代表元是 t=(1,0,0)), 换到超胞基底后 t·M^-1 未必是整数。若拿它筛选就会
    误删 C3^2, 得到 8 个**不成群**的操作(C3 在集合内而 C3^2 不在), 泛点轨道
    只有 8 像而不是 D3h 的 12 像, 轨道分解与对称扩展会因此出错。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    assert ops.rotations.shape[0] == 12


def test_common_symmetry_ops_form_a_group():
    """公共对称集必须是群: 复合封闭 + 每个元素可逆(轨道分解依赖这一点)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    n = ops.rotations.shape[0]
    assert n >= 1

    def compose(i, j):
        """{R_i|t_i} o {R_j|t_j} = {R_j R_i | t_j + R_j t_i} (列向量约定)。"""
        rj, tj = ops.rotations[j].astype(float), ops.translations[j]
        ri, ti = ops.rotations[i].astype(float), ops.translations[i]
        return rj @ ri, np.mod(tj + rj @ ti, 1.0)

    def find(r, t):
        for k in range(n):
            if np.allclose(ops.rotations[k], np.round(r), atol=1e-6) and \
                    np.allclose(np.mod(ops.translations[k] - t + 0.5, 1.0) - 0.5,
                                0.0, atol=1e-6):
                return k
        return None

    for i in range(n):
        for j in range(n):
            r, t = compose(i, j)
            assert find(r, t) is not None, f"复合 {i} o {j} 不在集合内"
        r, t = compose(i, i)
        assert find(r, t) is not None


def test_common_symmetry_ops_apply_is_invertible():
    """作用一次再作用逆, 回到原点(操作的阶可以是 1/2/3/6, 不都是对合)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    n = ops.rotations.shape[0]
    k = np.array([0.13, 0.07, 0.21])
    for i in range(n):
        ri = ops.rotations[i].astype(float)
        ti = ops.translations[i]
        # 逆元: R_j = R_i^-1, t_j = -R_j t_i
        rj = np.linalg.inv(ri)
        tj = np.mod(-rj @ ti, 1.0)
        for j in range(n):
            if np.allclose(ops.rotations[j], np.round(rj), atol=1e-6) and \
                    np.allclose(np.mod(ops.translations[j] - tj + 0.5, 1.0) - 0.5,
                                0.0, atol=1e-6):
                k1 = ops.apply(k, i)
                k2 = ops.apply(k1, j)
                # 用环绕距离比较(mod(x,1) 对 -1e-16 会给 1.0 而不是 0);
                # 容差 1e-6: spglib 的平移代表元带 ~1e-8 数值残差
                assert np.allclose(np.mod(k2 - k + 0.5, 1.0) - 0.5, 0.0, atol=1e-6)
                break
        else:
            raise AssertionError(f"操作 {i} 在集合内没有逆元")


def test_common_symmetry_ops_apply_follows_group_composition():
    """apply() 必须与群复合律一致 —— 这才是有判别力的检查。

    列向量约定下 {R_i|t_i} o {R_j|t_j} = {R_j R_i | t_j + R_j t_i},
    故"先作用 i 再作用 j"必须等于"直接作用该复合元"。
    (旧断言只是"像点落在 [0,1)³", 而 apply 内部已 mod -> 恒真, 判别力为零。)

    另验证泛点轨道大小 = 操作数: 每个操作必须给出不同的像。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    n = ops.rotations.shape[0]
    k = np.array([0.13, 0.07, 0.21])

    def find(r, t):
        for p in range(n):
            if np.allclose(ops.rotations[p], np.round(r), atol=1e-6) and \
                    np.allclose(np.mod(ops.translations[p] - t + 0.5, 1.0) - 0.5,
                                0.0, atol=1e-6):
                return p
        return None

    for i in range(n):
        for j in range(n):
            ri = ops.rotations[i].astype(float)
            rj = ops.rotations[j].astype(float)
            r_comp = rj @ ri                                   # {R_j R_i | ...}
            t_comp = np.mod(ops.translations[j] + rj @ ops.translations[i], 1.0)
            p = find(r_comp, t_comp)
            assert p is not None, f"复合 {i} o {j} 不在集合内"
            two_step = ops.apply(ops.apply(k, i), j)
            one_step = ops.apply(k, p)
            assert np.allclose(np.mod(two_step - one_step + 0.5, 1.0) - 0.5,
                               0.0, atol=1e-6)

    imgs = {tuple(np.round(ops.apply(k, i), 8)) for i in range(n)}
    assert len(imgs) == n          # 泛点的轨道大小 = 操作数


# ---------------------------------------------------------------- 非对称 M 的合成用例
# fixture 的 TRANSMAT 是对角的 (M.T == M), 转置错误在它上面检测不出来。
# 以下用例用非对称 M 锁定「直接空间用 M、倒空间用 M^-T」的约定。

def _asymmetric_prim_sc(m):
    """用 fixture 的六方原胞 + 非对称 M 造一个自洽的超胞(只用于几何关系)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = Structure(lattice=m @ prim.lattice, positions=prim.positions,
                    species=prim.species)
    return prim, sc


def test_k_mapping_matches_reciprocal_lattice_for_asymmetric_m():
    """k 侧必须用 (M^-1)^T: 直接空间用 M, 倒空间用 M^-T。

    物理定义: k_cart = k_sc · B_sc = k_prim · B_prim, 且 B_sc = (M^-1)^T B_prim。
    """
    m = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    prim, sc = _asymmetric_prim_sc(m)
    tr = resolve_transform(sc, prim, m)
    b_prim, b_sc = reciprocal(prim.lattice), reciprocal(sc.lattice)

    k_super = np.array([0.17, 0.29, 0.0])
    k_prim_phys = (k_super @ b_sc) @ np.linalg.inv(b_prim)
    assert np.allclose(tr.prim_from_super(k_super, np.zeros(3)), k_prim_phys)

    k_prim = np.array([0.11, 0.23, 0.0])
    k_super_phys = (k_prim @ b_prim) @ np.linalg.inv(b_sc)
    diff = np.mod(tr.super_from_prim(k_prim) - k_super_phys + 0.5, 1.0) - 0.5
    assert np.allclose(diff, 0.0, atol=1e-9)

    # 回环: prim_from_super(super_from_prim(k), G=0) ≡ k (mod 1)
    back = tr.prim_from_super(tr.super_from_prim(k_prim), np.zeros(3))
    assert np.allclose(np.mod(back - k_prim + 0.5, 1.0) - 0.5, 0.0, atol=1e-9)


def test_candidate_g_vectors_fold_back_consistent_for_asymmetric_m():
    """每个候选 k 折叠回超胞 k 必须回到原点(mod 1): k_prim · M^T ≡ K。"""
    m = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    prim, sc = _asymmetric_prim_sc(m)
    tr = resolve_transform(sc, prim, m)
    gset = candidate_g_vectors(m)
    assert gset.shape == (tr.det, 3)

    k_super = np.array([0.17, 0.29, 0.0])
    seen = set()
    for g in gset:
        k_prim = tr.prim_from_super(k_super, g)
        assert np.all(k_prim >= -1e-9) and np.all(k_prim < 1.0 + 1e-9)
        back = np.mod(k_prim @ tr.m_eff.T, 1.0)
        diff = np.mod(back - k_super + 0.5, 1.0) - 0.5
        assert np.allclose(diff, 0.0, atol=1e-9)
        seen.add(tuple(np.round(k_prim, 8)))
    assert len(seen) == gset.shape[0]      # 候选互不重复


def test_common_symmetry_ops_rotation_criterion_for_asymmetric_m():
    """旋转判据 M^-T R M^T ∈ Z³ (spglib 列向量约定) —— 用非对称 M 锁定。

    √3 型超胞基 (C3 相容) 下 D3h 的 12 个操作全部保持超胞晶格;
    若误用行向量约定的判据 M R M^-1, 只会留下 4 个。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]])
    ops = common_symmetry_ops(prim, m)
    assert ops.rotations.shape[0] == 12


def test_common_symmetry_ops_rotation_criterion_matches_cartesian_truth():
    """判据必须与笛卡尔物理基准**逐个**吻合(不只是个数相同)。

    物理基准 = "该旋转把超胞晶格映回自身"(笛卡尔几何, 与分数约定无关):
        A_sc · R_cart^T · A_sc^-1 ∈ Z³,  R_cart = A_col R A_col^-1 (列约定)。
    60° 莫尔型 M=[[1,1],[-1,2]] 下真值只保留 4/12, 而旧判据 M^-1 R M 与
    字面式 M R M^-1 都会留下 12/12(把被剪切破坏的对称也留下) -> 只有
    M^-T R M^T 逐个吻合。
    """
    import spglib

    prim = read_poscar(str(FIXTURE_PRIMCELL))
    a_p = prim.lattice
    a_col = a_p.T
    sym = spglib.get_symmetry((a_p, prim.positions, [1, 2, 2]), symprec=1e-5)

    for m in (np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]]),
              np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 1]])):
        a_sc = m @ a_p
        truth = []
        for r in sym["rotations"]:
            r_cart = a_col @ np.asarray(r, float) @ np.linalg.inv(a_col)
            x = a_sc @ r_cart.T @ np.linalg.inv(a_sc)
            if np.allclose(x, np.round(x), atol=1e-6):
                truth.append(np.asarray(r, int))

        ops = common_symmetry_ops(prim, m)
        assert ops.rotations.shape[0] == len(truth)
        for r in ops.rotations:
            assert any(np.array_equal(r, t) for t in truth), \
                f"判据多留了操作 {r.tolist()}"


# ---------------------------------------------------------------- 候选 G 的搜索范围(行和界)

def test_candidate_g_vectors_row_sum_bound_beats_max_abs_bound():
    """回归: 搜索半径必须按 M 的**行和**给, 不能用 max|M|+2。

    M=[[10,10,0],[0,10,0],[0,0,1]] 的候选满足 g1 ∈ [0,10), g0−g1 ∈ [0,10)
    (整数), 故 g0 最大到 18 —— 已超过旧界 max|M|+2 = 12,
    旧实现只找到 79/100 个候选 -> 直接抛错。
    """
    m = np.array([[10, 10, 0], [0, 10, 0], [0, 0, 1]])
    g = candidate_g_vectors(m)
    assert g.shape == (100, 3)            # det(M) = 100
    assert np.abs(g).max() > 12           # 旧界(max|M|+2)覆盖不到
    frac = g @ np.linalg.inv(m.astype(float)).T
    assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)


def test_candidate_g_vectors_random_matrices_give_exactly_det_candidates():
    """压力测试: 随机整数 M(含大 off-diagonal)必须恰好找到 det(M) 个候选。

    旧实现用 max|M|+2 给界, |M| ~ 10~30 时 300 个随机 M 里 276 个漏候选
    (候选数 != det(M) -> 直接抛错, 莫尔型超胞根本跑不动);
    换成行和界后 0 失败。固定种子 ✓。
    """
    rng = np.random.default_rng(20261117)
    checked = 0
    while checked < 200:
        m = rng.integers(-30, 31, size=(3, 3))
        det = abs(round(np.linalg.det(m.astype(float))))
        if det == 0:
            continue                       # 奇异矩阵没有候选, 不参与
        checked += 1
        g = candidate_g_vectors(m)
        assert g.shape == (det, 3), \
            f"M={m.tolist()} 候选数 {g.shape[0]} != det(M) {det}"
        frac = g @ np.linalg.inv(m.astype(float)).T
        assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)
        assert len({tuple(r) for r in g.tolist()}) == det      # 候选互不重复


def test_candidate_g_vectors_matches_unfold_candidate_g_for_k_for_symmetric_m():
    """对称 M 下, geometry.candidate_g_vectors 与 unfold.candidate_g_for_k
    必须给出一致的**分数值**集合。

    比较分数值而不是 g 的下标/成员: 两者的枚举顺序无关紧要, 但
    (K=0 时) 同一个判据 g·M^-T ∈ [0,1)³ 的解集必须一致。
    """
    from unfoldfs.unfold import candidate_g_for_k

    rng = np.random.default_rng(4242)
    checked = 0
    while checked < 20:
        a = rng.integers(-15, 16, size=(3, 3))
        m = a + a.T                       # 对称整数矩阵, off-diagonal 可达 ±30
        det = abs(round(np.linalg.det(m.astype(float))))
        if det == 0:
            continue
        checked += 1
        inv = np.linalg.inv(m.astype(float))
        g_geom = candidate_g_vectors(m)
        g_unf = candidate_g_for_k(np.zeros(3), m, inv.T)
        assert g_geom.shape == (det, 3)
        assert g_unf.shape == (det, 3)
        f_geom = np.sort(g_geom @ inv.T, axis=0)
        f_unf = np.sort(g_unf @ inv.T, axis=0)
        assert np.allclose(f_geom, f_unf, atol=1e-9)


def test_candidate_g_vectors_error_message_reports_range_and_expected_count():
    """错误信息必须能区分「搜索范围不足」与「计数不对」。

    旧信息只有「候选 G 数 N != det(M)」, 看不出是范围不够还是矩阵有问题;
    新信息同时报出实际数、期望数与逐轴搜索区间, 便于定位。
    """
    m = np.array([[0.5, 0, 0], [0, 2, 0], [0, 0, 2]])   # 非整数矩阵 -> 计数必然不符
    with pytest.raises(ValueError) as excinfo:
        candidate_g_vectors(m)
    msg = str(excinfo.value)
    assert "候选 G 数 4" in msg
    assert "期望 2" in msg
    assert "搜索区间" in msg


# ------------------------------------------------------------------ 动量作用公式
# (C2) SymmetryOps.apply 的动量作用必须是 **R^-1**, 不是 R^T。
# 判据: **笛卡尔几何真值**(与分数基的行/列约定无关):
#   分数基下的旋转 R(spglib 的列向量约定 x' = R x)对应笛卡尔旋转
#   Q = A^-1 · R^T · A(行向量作用 r' = r·Q); 倒空间随之 k' = k·Q。
# 分数基**不是**正交阵 ⇒ "R 的转置 = R 的逆"不成立。

def _fixture_ops():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    tr = resolve_transform(sc, prim, read_transmat(str(FIXTURE_TRANSMAT)))
    return prim, tr, common_symmetry_ops(prim, tr.m_eff)


def test_symmetry_ops_apply_matches_the_cartesian_geometry_truth():
    """apply() 的动量作用 = k·R^-1(笛卡尔几何真值裁决)。"""
    prim, _tr, ops = _fixture_ops()
    a = np.asarray(prim.lattice, dtype=float)
    b = reciprocal(a)
    b_inv = np.linalg.inv(b)
    ks = [np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0]),
          np.array([0.5, 0.25, 0.75]), np.array([1 / 3, 2 / 3, 1 / 3])]
    n_checked = 0
    for i in range(ops.rotations.shape[0]):
        r = np.asarray(ops.rotations[i])
        q = np.linalg.inv(a) @ r.T @ a            # 笛卡尔旋转(行向量作用)
        assert np.allclose(q @ q.T, np.eye(3), atol=1e-8), "Q 必须是正交阵"
        for k in ks:
            truth = np.mod((k @ b) @ q @ b_inv, 1.0)
            got = ops.apply(k, i)
            # 容差 1e-6: spglib 的平移代表元带 ~3e-8 残差(apply 含 +t,
            # 而平移不改变倒空间的 k —— 这一项残差是唯一的偏差来源)。
            # 比较必须用**周期距离**: 0.99999997 与 0.0 是同一个点
            assert periodic_delta(got, truth).max() < 1e-6, (
                f"op{i}: apply 与笛卡尔几何真值不符\n"
                f"  真值 {truth.tolist()}\n  实得 {got.tolist()}")
            n_checked += 1
    assert n_checked >= 12 * 4


def test_symmetry_ops_apply_is_not_the_transpose_formula():
    """反例: 旧公式 k·R^T 在**非正交分数基**下与真值不符(判别力)。

    若哪天有人把 apply 改回 R^T(或 R), 这条测试会立刻抓住 ——
    48 个样本里只有 16 个(R^T)/32 个(R)与真值一致。
    """
    prim, _tr, ops = _fixture_ops()
    a = np.asarray(prim.lattice, dtype=float)
    b = reciprocal(a)
    b_inv = np.linalg.inv(b)
    ks = [np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.25, 0.75]),
          np.array([1 / 3, 2 / 3, 1 / 3]), np.array([0.2, 0.4, 0.6])]
    n_rt, n_r, n_total = 0, 0, 0
    for i in range(ops.rotations.shape[0]):
        r = np.asarray(ops.rotations[i])
        q = np.linalg.inv(a) @ r.T @ a
        for k in ks:
            truth = np.mod((k @ b) @ q @ b_inv, 1.0)
            n_total += 1
            n_rt += int(periodic_delta(np.mod(k @ r.T, 1.0), truth).max() < 1e-6)
            n_r += int(periodic_delta(np.mod(k @ r, 1.0), truth).max() < 1e-6)
    assert n_total >= 12 * 4
    # 转置写法必然与真值大面积不符(否则这条测试没有判别力)
    assert n_rt < n_total
    assert n_r < n_total


def test_rotation_inverse_is_exact_for_unimodular_matrices():
    """整数阵的逆必须**精确**是整数阵(下游全是整数性判据)。"""
    from unfoldfs.geometry import rotation_inverse
    r = np.array([[1, 0, 0], [1, -1, 0], [0, 0, 1]])
    inv = rotation_inverse(r)
    assert inv.dtype.kind == "i"
    assert np.array_equal(inv @ r, np.eye(3, dtype=int))


# ------------------------------------------------------------------ POSCAR 格式

def test_read_poscar_rejects_the_vasp4_format_with_a_clear_message(tmp_path):
    """VASP-4 格式(第 6 行是原子数, 没有元素符号行)必须给出**可操作**的报错。

    旧实现会掉进 ``int("Direct")`` 的 ValueError, 用户完全看不懂。
    """
    text = FIXTURE_PRIMCELL.read_text(encoding="utf-8").splitlines()
    v4 = text[:5] + text[6:]          # 去掉元素符号行 -> VASP-4 格式
    path = tmp_path / "POSCAR_v4"
    path.write_text("\n".join(v4) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_poscar(str(path))
    msg = str(excinfo.value)
    assert "VASP-4" in msg
    assert "VASP-5" in msg
    assert "元素符号" in msg


# ------------------------------------------------------------------ 对称操作剔除项

def test_common_symmetry_ops_detailed_reports_the_dropped_operations():
    """被剔除的对称操作必须**带着原因**返回(spec §7: info 里要列出)。"""
    from unfoldfs.geometry import common_symmetry_ops_detailed
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    # 换一个与 C3 不相容的超胞(几何真值: 只剩 4 个操作)
    m_bad = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    ops, dropped = common_symmetry_ops_detailed(prim, m_bad)
    assert ops.rotations.shape[0] == 4
    assert len(dropped) == 8
    for _r, _t, why in dropped:
        assert why and "M^-T" in why


# ------------------------------------------------------------------ 原胞完整点群
# 修复轮 6: 对称化必须用**原胞的完整点群**, 而不是"原胞与超胞的公共对称操作"
# —— 后者在破坏对称性的超胞上只剩 4/12 个操作, 六重对称恢复不了。

def test_primitive_symmetry_ops_returns_the_whole_point_group():
    """原胞点群 = spglib 给的全部 12 个操作(D3h), **不按超胞变换过滤**。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    ops = primitive_symmetry_ops(prim)
    assert ops.rotations.shape == (12, 3, 3)
    assert ops.translations.shape == (12, 3)


def test_primitive_symmetry_ops_rotations_are_integer_and_unimodular():
    """旋转必须是**整数**矩阵(否则作用不到分数坐标上)且行列式为 ±1。

    ``rotation_inverse`` 要求幺模整数阵(否则抛错)⇒ 这里的过滤是必要的。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    ops = primitive_symmetry_ops(prim)
    for r in ops.rotations:
        assert np.array_equal(r, np.round(r).astype(int))
        assert abs(abs(np.linalg.det(r.astype(float))) - 1.0) < 1e-9


def test_primitive_symmetry_ops_keeps_the_translations():
    """平移分量必须保留(spglib 给的是模原胞格矢的代表元)。"""
    import spglib

    prim = read_poscar(str(FIXTURE_PRIMCELL))
    ops = primitive_symmetry_ops(prim)
    ids = np.array([sorted(set(prim.species)).index(s) + 1 for s in prim.species],
                   dtype=int)
    sym = spglib.get_symmetry((prim.lattice, prim.positions, ids), symprec=1e-5)
    raw = sym["translations"]
    assert raw.shape[0] == ops.translations.shape[0]
    for t_raw, t_stored in zip(raw, ops.translations):
        d = np.mod(t_raw - t_stored + 0.5, 1.0) - 0.5
        assert np.allclose(d, 0.0, atol=1e-6)


def test_primitive_symmetry_ops_is_a_superset_of_the_common_subgroup():
    """破坏对称性的超胞(M=diag(3,1,1)): 公共子群只剩 4 个, 完整点群仍是 12 个。

    这正是修复轮 6 的核心: 只用公共子群做对称化 ⇒ 六重对称恢复不了。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m_bad = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    common = common_symmetry_ops(prim, m_bad)
    full = primitive_symmetry_ops(prim)
    assert common.rotations.shape[0] == 4
    assert full.rotations.shape[0] == 12

    def key(r):
        return tuple(np.round(np.asarray(r, dtype=float)).astype(int).ravel())

    full_keys = {key(r) for r in full.rotations}
    assert all(key(r) in full_keys for r in common.rotations)


def test_primitive_symmetry_ops_forms_a_group():
    """完整点群必须是群(轨道分解依赖封闭性)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    ops = primitive_symmetry_ops(prim)
    n = ops.rotations.shape[0]

    def compose(i, j):
        """{R_i|t_i} o {R_j|t_j} = {R_j R_i | t_j + R_j t_i}(列向量约定)。"""
        rj, tj = ops.rotations[j].astype(float), ops.translations[j]
        ri, ti = ops.rotations[i].astype(float), ops.translations[i]
        return rj @ ri, np.mod(tj + rj @ ti, 1.0)

    def find(r, t):
        for k in range(n):
            if np.allclose(ops.rotations[k], np.round(r), atol=1e-6) and \
                    np.allclose(np.mod(ops.translations[k] - t + 0.5, 1.0) - 0.5,
                                0.0, atol=1e-6):
                return k
        return None

    for i in range(n):
        for j in range(n):
            r, t = compose(i, j)
            assert find(r, t) is not None, f"复合 {i} o {j} 不在集合内"


def test_pyproject_requires_a_spglib_with_the_dict_api():
    """I2: 代码用 spglib **>= 2.0 的 dict API**(sym["rotations"]),
    依赖声明必须一致 —— 否则"装得上、跑起来炸"(TypeError: tuple indices...)。"""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'"spglib\s*([><=!~][^"]*)"', text)
    assert m, "pyproject.toml 里没有 spglib 的依赖声明"
    spec = m.group(1)
    assert "2.0" in spec, (
        f"spglib 的依赖声明是 {spec!r}, 但代码用的是 >= 2.0 的 dict API "
        "(sym['rotations']) —— 声明与实现不一致")
