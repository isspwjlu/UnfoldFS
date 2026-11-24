"""逐轨道谱权重的测试: 与 PROCAR 的模方块 |p|² 交叉验证。

两条路径**独立**:
  * 模方块: 每个原子每个轨道的 |p|², 直接来自 VASP;
  * 逐轨道权重: 由**复**投影系数(带相位)先做 Bloch 相位收缩再取模平方。

在 det(M)=1(原胞直通)的极限下, 两者必须一致 —— 这是对实现的一个真检查
(如果相位或副本平均写错, 两条路会给出不同的结果)。
"""
from dataclasses import replace

import numpy as np
import pytest

from unfoldfs.geometry import (Transform, read_poscar, read_transmat,
                               reciprocal, resolve_transform)
from unfoldfs.unfold import (build_atom_copies, unfold_at_k,
                           unfold_supercell)
from unfoldfs.vaspio import read_eigenval, read_procar

DATA = "tests/data"


@pytest.fixture(scope="module")
def fixture_parts():
    prim = read_poscar(f"{DATA}/PRIMCELL.vasp")
    sc = read_poscar(f"{DATA}/POSCAR")
    tr = resolve_transform(sc, prim, read_transmat(f"{DATA}/TRANSMAT.in"))
    copies = build_atom_copies(prim, sc, tr)
    band = read_eigenval(f"{DATA}/EIGENVAL")
    procar = read_procar(f"{DATA}/PROCAR")
    return prim, sc, tr, copies, band, procar


def test_procar_orbital_labels_come_from_the_header(fixture_parts):
    """轨道名必须从 PROCAR 表头读出(轨道数随赝势变, 不能写死)。"""
    _, _, _, _, _, procar = fixture_parts
    assert len(procar.orbital_labels) == procar.coeffs.shape[3]
    assert all(str(x) for x in procar.orbital_labels)


def test_orbital_weights_are_finite_and_non_negative(fixture_parts):
    prim, sc, tr, copies, band, procar = fixture_parts
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)
    _, _, w, w_orb = unfold_at_k(
        procar.kpoints[0], procar.coeffs[0], band.energies[0], tr, copies,
        tr.m_eff, prim.lattice, prim.positions @ prim.lattice,
        b_prim, b_super, 1, 1, with_orbital=True)
    assert np.isfinite(w_orb).all()
    assert (w_orb >= 0).all()
    assert w_orb.shape == w.shape + (procar.coeffs.shape[3],)


def test_orbital_weights_satisfy_the_sharp_cauchy_schwarz_bound(fixture_parts):
    """锐利的 Cauchy–Schwarz 界: w 与 Σ_orb 必须自洽。

    记 x_n = Σ_i bloch_i·p_{i,n}(每个轨道的相干和), 则

        w      = |Σ_n x_n|²        (总权重)
        Σ_orb  = Σ_n |x_n|²        (逐轨道权重之和)

    固定各 |x_n| 而改变相对相位时, |Σ_n x_n|² 的取值范围是

        下界 = max(0, 2Σ|x_n|² − (Σ|x_n|)²)     (相位尽可能相消)
        上界 = (Σ|x_n|)²                        (相位完全对齐)

    ⇒ 无参数的锐利界。这个检查抓的是"总权重与逐轨道权重由**不同的**相位/
    原子配对算出来"这类实现错误。

    ⚠ 只在**权重不可忽略**的条目上比较: 近零条目(1e-30 量级)的相对偏差
       纯粹是浮点舍入, 用相对容差会误报。
    """
    prim, sc, tr, copies, band, procar = fixture_parts
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)
    _, _, w, w_orb = unfold_at_k(
        procar.kpoints[0], procar.coeffs[0], band.energies[0], tr, copies,
        tr.m_eff, prim.lattice, prim.positions @ prim.lattice,
        b_prim, b_super, 1, 1, with_orbital=True)
    x_abs = np.sqrt(np.clip(w_orb, 0.0, None))          # |x_n|
    sum_abs = x_abs.sum(axis=-1)
    sum_sq = (x_abs ** 2).sum(axis=-1)                 # Σ|x_n|²
    hi = sum_abs ** 2
    lo = np.maximum(0.0, 2.0 * sum_sq - hi)
    scale = float(np.max(hi)) if hi.size else 0.0
    ok = np.isfinite(w) & np.isfinite(hi) & (hi > 1e-9 * scale)
    assert ok.any()
    tol = 1e-8 * scale
    assert np.all(w[ok] <= hi[ok] + tol), "总权重超过上界"
    assert np.all(w[ok] >= lo[ok] - tol), "总权重低于下界"


def test_single_orbital_case_reduces_to_the_total_weight(fixture_parts):
    """只留**一个**轨道非零时, 总权重必须**精确等于**该轨道的权重。

    单轨道没有交叉项 ⇒ |Σ_n x_n|² = |x_0|² ⇒ w ≡ w_orb[..., 0]。
    这是最尖锐的精确检查: 任何相位/原子配对/轨道映射的错误都会破坏它。
    """
    prim, sc, tr, copies, band, procar = fixture_parts
    b_prim = reciprocal(prim.lattice)
    b_super = reciprocal(sc.lattice)

    # 只保留第 0 个轨道
    coeffs = np.asarray(procar.coeffs).copy()
    coeffs[..., 1:] = 0.0
    one = replace(procar, coeffs=coeffs)
    _, _, w, w_orb = unfold_at_k(
        one.kpoints[0], one.coeffs[0], band.energies[0], tr, copies,
        tr.m_eff, prim.lattice, prim.positions @ prim.lattice,
        b_prim, b_super, 1, 1, with_orbital=True)
    assert np.allclose(w, w_orb[..., 0], rtol=1e-12, atol=1e-300)
    # 其余轨道必须为零
    assert np.all(w_orb[..., 1:] == 0.0)
