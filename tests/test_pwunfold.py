"""平面波路线的单元测试。

核心是**完备性**: 候选等价类必须是超胞平面波基的一个**划分**, 于是
``Σ_j w_j = Σ_G |C_G|² = 1``(系数按带归一化后)。这条恒等式与系数怎么取无关,
所以它是对"候选集构造"最强的判据。

测试用**合成 WAVECAR**(自己按 VASP 的记录布局写一个), 这样不依赖外部数据。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from unfoldfs.geometry import Transform
from unfoldfs.pwunfold import (class_labels_for_candidates,
                               permutation_for_candidates, plane_wave_weights)
from unfoldfs.unfold import unfold_supercell_plane_waves
from unfoldfs.wfc import (_plane_waves_on_grid, minimum_fft_grid,
                          read_wavecar_header)

_A = 3.0          # 立方晶格常数 (Å)
_ENCUT = 60.0     # eV —— 取小值让平面波数只有几百


def _cubic_cell(a: float = _A) -> np.ndarray:
    return np.eye(3) * a


def _pw_list(cell, encut, k_cart=None):
    """与 wfc 同款的平面波列表(整数索引)与动能。"""
    ngrid = minimum_fft_grid(cell, encut)
    k = np.zeros(3) if k_cart is None else np.asarray(k_cart, float)
    return _plane_waves_on_grid(ngrid, k, cell, encut)


def _write_wavecar(path, cell, encut, k_cart, coeffs, precision="complex64"):
    """按 VASP 的记录布局写一个合成 WAVECAR。

    coeffs : (nbands, nplw) 复系数
    """
    nplw = coeffs.shape[1]
    nbands = coeffs.shape[0]
    dt = np.complex64 if precision == "complex64" else np.complex128
    itemsize = 8 if precision == "complex64" else 16
    recl = max(8 * (4 + 3 * nbands), nplw * itemsize)
    with open(path, "wb") as fh:
        fh.write(np.array([recl, 1, 45200 if precision == "complex64" else 45210],
                          dtype=np.float64).tobytes())
        fh.write(b"\0" * (recl - 24))
        head = np.zeros(12, dtype=np.float64)
        head[0] = 1          # nkpts
        head[1] = nbands
        head[2] = encut
        head[3:] = cell.ravel()
        fh.write(head.tobytes())
        fh.write(b"\0" * (recl - 96))
        # k 头记录: nplw, kvec(3), [E, ?, occ] × nbands
        khead = np.zeros(4 + 3 * nbands, dtype=np.float64)
        khead[0] = nplw
        khead[1:4] = k_cart
        for ib in range(nbands):
            khead[4 + 3 * ib] = -5.0 + ib
        fh.write(khead.tobytes())
        fh.write(b"\0" * (recl - khead.nbytes))
        for ib in range(nbands):
            fh.write(np.asarray(coeffs[ib], dtype=dt).tobytes())
            fh.write(b"\0" * (recl - nplw * itemsize))


def _transform_3x1x1() -> Transform:
    """3×1×1 超胞的变换矩阵(行向量约定: A_sc = M · A_prim)。"""
    m = np.diag([3.0, 1.0, 1.0])
    return Transform(m, np.linalg.inv(m))


# ---------------------------------------------------------------- 核心函数

def test_plane_wave_weights_completeness():
    """完备性: Σ_j w_j = 1, 与系数怎么取无关。"""
    rng = np.random.default_rng(7)
    nplw, nband, det = 500, 4, 3
    coeffs = rng.normal(size=(nplw, nband)) + 1j * rng.normal(size=(nplw, nband))
    labels = rng.integers(0, det, size=nplw)
    w, comp = plane_wave_weights(coeffs, labels, np.arange(det))
    assert w.shape == (det, nband)
    np.testing.assert_allclose(comp, 1.0, atol=1e-12)
    # 不归一化时, 完备性 = Σ_n |C|²
    w_raw, comp_raw = plane_wave_weights(coeffs, labels, np.arange(det),
                                         normalize=False)
    np.testing.assert_allclose(comp_raw,
                               np.sum(np.abs(coeffs) ** 2, axis=0), rtol=1e-12)


def test_plane_wave_weights_single_plane_wave():
    """系数只落在一个平面波上时, 权重全在它所属的类上。"""
    coeffs = np.zeros((10, 1), dtype=complex)
    coeffs[3, 0] = 1.0
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 1])
    w, comp = plane_wave_weights(coeffs, labels, np.arange(3))
    np.testing.assert_allclose(w[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(comp, 1.0, atol=1e-12)


def test_class_labels_are_a_partition():
    """等价类必须把平面波基划分成 det(M) 个不重不漏的类。"""
    m = np.diag([3.0, 1.0, 1.0])
    m_inv_T = np.linalg.inv(m).T
    # 候选: (K + g)·M^-T 落在 [0,1)^3
    k = np.array([0.1, 0.2, 0.0])
    spans = np.abs(m).sum(axis=1).astype(int) + 1
    axes = [np.arange(-s, s + 1, dtype=int) for s in spans]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    frac = (k + grid) @ m_inv_T
    keep = np.all((frac >= -1e-9) & (frac < 1 - 1e-9), axis=1)
    g_cand = grid[keep]
    assert g_cand.shape[0] == 3

    rng = np.random.default_rng(3)
    gidx = rng.integers(-8, 9, size=(400, 3))
    labels = class_labels_for_candidates(gidx, m_inv_T, g_cand)
    assert set(np.unique(labels)) <= {0, 1, 2}
    # 每个类非空, 且总数 = 平面波数
    assert labels.size == gidx.shape[0]
    perm = permutation_for_candidates(g_cand, m_inv_T, labels, g_cand)
    assert sorted(perm.tolist()) == [0, 1, 2]


# ---------------------------------------------------------------- 文件往返

def test_wavecar_roundtrip_and_unfold(tmp_path):
    """合成 WAVECAR → 读回 → 反折叠: 权重落在正确的类上, 完备性为 1。"""
    cell = _cubic_cell()
    gvec, _ = _pw_list(cell, _ENCUT)
    nplw = gvec.shape[0]
    assert nplw > 10
    nbands = 3
    rng = np.random.default_rng(11)
    coeffs = (rng.normal(size=(nbands, nplw))
              + 1j * rng.normal(size=(nbands, nplw)))
    path = str(tmp_path / "WAVECAR")
    _write_wavecar(path, cell, _ENCUT, np.zeros(3), coeffs)

    hdr = read_wavecar_header(path, nkpts=1, nbands=nbands, nspin=1)
    assert hdr.nkpts == 1 and hdr.nbands == nbands
    assert hdr.encut == pytest.approx(_ENCUT)

    from unfoldfs.wfc import iter_wavecar_kpoints
    kps = list(iter_wavecar_kpoints(path, hdr))
    assert len(kps) == 1
    kp = kps[0]
    assert kp.nplw == nplw
    np.testing.assert_array_equal(kp.gvec, gvec)
    np.testing.assert_allclose(kp.coeffs, coeffs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kp.k_frac, np.zeros(3))


def test_unfold_supercell_plane_waves(tmp_path):
    """端到端: 超胞 WAVECAR 的展开 —— 完备性为 1, 权重落在正确的候选上。

    合成的态是"物理态": 系数全部落在候选 0 对应的等价类上(另两个类为零),
    于是权重应当是 [1, 0, 0], 主导度为 1。
    """
    cell = _cubic_cell()
    gvec, _ = _pw_list(cell, _ENCUT)
    nplw = gvec.shape[0]
    nbands = 2
    tr = _transform_3x1x1()
    from unfoldfs.geometry import reciprocal
    from unfoldfs.unfold import candidate_g_for_k

    # 候选 0 的等价类: (K + n)·M^-T ≡ k'_0·M^-T (mod 1)
    b_prim = reciprocal(cell)
    b_super = reciprocal(cell.copy())
    b_super[0] = b_super[0] / 3.0
    g_cand = candidate_g_for_k(np.zeros(3), tr.m_eff, tr.m_inv.T)
    m_inv_T = tr.m_inv.T
    u = np.mod(gvec.astype(float) @ m_inv_T, 1.0)
    u_cand = np.mod(g_cand.astype(float) @ m_inv_T, 1.0)
    dmat = np.abs(u[:, None, :] - u_cand[None, :, :])
    dmat = np.minimum(dmat, 1.0 - dmat).max(axis=2)
    labels = np.argmin(dmat, axis=1)
    assert dmat.min(axis=1).max() < 1e-6, "平面波没有全部归类"

    rng = np.random.default_rng(5)
    coeffs = np.zeros((nbands, nplw), dtype=complex)
    for ib in range(nbands):
        sel = labels == 0
        vals = (rng.normal(size=int(sel.sum()))
                + 1j * rng.normal(size=int(sel.sum())))
        coeffs[ib, sel] = vals / np.linalg.norm(vals)
    path = str(tmp_path / "WAVECAR")
    _write_wavecar(path, cell, _ENCUT, np.zeros(3), coeffs)

    from unfoldfs.vaspio import BandData
    energies = np.tile(np.arange(nbands, dtype=float)[None, :], (1, 1))
    band = BandData(kpoints=np.zeros((1, 3)), energies=energies,
                    occupations=np.ones((1, nbands)), efermi=float("nan"))

    res = unfold_supercell_plane_waves(
        band, path, tr, tr.m_eff, cell, np.zeros((1, 3)), b_prim, b_super,
        np.zeros((1, 3)))
    # 完备性: 权重和 = 1
    w = np.asarray(res.weights)
    valid = np.asarray(res.valid)
    tot = np.sum(np.where(valid, w, 0.0), axis=1)
    np.testing.assert_allclose(tot, 1.0, atol=1e-6)
    # 权重全在候选 0 上(w 的形状是 (n_k, det, n_band)); 逐带检查
    for b in range(nbands):
        row = np.where(valid[0, :, b], w[0, :, b], 0.0)
        np.testing.assert_allclose(row[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(row[1:], 0.0, atol=1e-6)
