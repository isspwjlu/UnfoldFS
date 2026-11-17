import numpy as np
import pytest

from unfoldfs.geometry import (
    Structure, read_poscar, reciprocal, read_transmat,
    resolve_transform, matrix_denominator_lcm, candidate_g_vectors,
    common_symmetry_ops,
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


def test_common_symmetry_ops_map_k_point_into_bz():
    """操作作用后仍落在原胞 BZ 内(分数坐标 mod 1)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    k = np.array([0.13, 0.07, 0.21])
    for i in range(ops.rotations.shape[0]):
        img = ops.apply(k, i)
        assert np.all(img >= -1e-9) and np.all(img < 1.0 + 1e-9)


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
