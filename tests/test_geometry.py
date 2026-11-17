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
    frac = g @ inv
    assert np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9)


def test_denominator_lcm():
    assert matrix_denominator_lcm(np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])) == 3
    assert matrix_denominator_lcm(np.array([[2, 1], [0, 2]])) == 4


def test_common_symmetry_ops_are_valid_in_both_bases():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    assert ops.rotations.shape[0] >= 1
    inv = np.linalg.inv(m.astype(float))
    for r in ops.rotations:
        assert np.allclose(inv @ r @ m.astype(float), np.round(inv @ r @ m.astype(float)))
        assert np.allclose(np.round(inv @ r @ m.astype(float)), r)   # 对角阵下二者等价
    for t in ops.translations:
        assert np.allclose(t @ inv, np.round(t @ inv))


def test_common_symmetry_ops_count_for_fixture():
    """MoS2 3x3x1 超胞与原胞相容, 公共对称集 = 原胞 D3h 的全部 12 个操作。

    不能直接拿 spglib 返回的 t 去判 "t·M^-1 ∈ Z³": t 只在原胞格矢意义下
    确定(spglib 给 C3^2 的代表元是 t=(1,0,0)), 用未归一化的代表元去筛
    会误删 C3^2, 得到 8 个**不成群**的操作(C3 在集合内而 C3^2 不在),
    后续轨道分解与对称扩展都会因此出错。
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
                # 用环绕距离比较(mod(x,1) 对 -1e-16 会给 1.0 而不是 0)
                assert np.allclose(np.mod(k2 - k + 0.5, 1.0) - 0.5, 0.0, atol=1e-8)
                break
        else:
            raise AssertionError(f"操作 {i} 在集合内没有逆元")


def test_common_symmetry_ops_map_k_point_into_bz():
    """平移归一化后, 操作把原胞 BZ 内的点仍映射到 BZ 内(平移为格矢 -> 平凡)。"""
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    k = np.array([0.13, 0.07, 0.21])
    for i in range(ops.rotations.shape[0]):
        img = ops.apply(k, i)
        assert np.all(img >= -1e-9) and np.all(img < 1.0 + 1e-9)
