"""fs 模块测试: 脊顶提取(三道闸门) + 协变性诊断 + 验证协议 + 费米速度。

核心断言都是**结构性**的(在人工构造的场上有确定答案), 不依赖 fixture 的
物理精度 —— fixture 的 PROCAR 只有 3 位小数, 且只有 2 个 k 点。

⚠ 与论文出图脚本的关系: unfoldfs.fs 是那些脚本里逻辑的**包内版本**,
   语义必须一致(论文里的数字由此产生)。所以这里也断言两条路给出的
   曲线在几何上相同(见 test_package_matches_figure_script_contour)。
"""
import numpy as np
import pytest

from unfoldfs.fs import (crest_curves, crest_field, dispersion,
                         fermi_velocity, fold_to_bz, load_spectral_npz,
                         point_to_segment_distance, spectral_curves,
                         split_at_jumps, verify_fermi_surface,
                         velocity_to_si)
from unfoldfs.geometry import SymmetryOps


# ------------------------------------------------------------------ 合成场

def _grid(n=41):
    """一个 n×n 的分数坐标网格(0..1, gamma 型)。"""
    return np.linspace(0.0, 1.0, n, endpoint=False)


def _gaussian_ridge(n=41, e_f=0.0, sigma=0.06, width=0.15, amp=1.0):
    """一条高斯"脊": 脊顶在 k_x = 0.5 的竖线上, 脊高 amp, 垂直宽度 width。"""
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    crest = 0.5
    d = (kx - crest) * 2.0          # 分数坐标 -> 近似 2π/a
    e_peak = e_f + 0.05 * np.cos(2 * np.pi * ky)   # 脊顶轻微起伏(不是直线)
    de = (e_peak - e_f)
    # 沿"离脊顶的能量距离"做高斯: band_energy(k) = e_peak + slope*(kx-crest)
    slope = 0.6
    band_e = e_peak + slope * (kx - crest)
    return fx, e_peak, band_e, amp


def _spectral_from_bands(band_e, e_grid, sigma, weights=None):
    """由逐带能量构造 A(k,E) = Σ_b w_b exp[-(E-E_b)²/2σ²], 形状 (n1, n2, nE)。

    ``band_e`` 可以是 (n1, n2)(单带)或 (n_band, n1, n2)(多带)。
    """
    be = np.asarray(band_e, dtype=float)
    if be.ndim == 2:
        be = be[None, ...]
    w = np.ones_like(be) if weights is None else np.asarray(weights, dtype=float)
    d = e_grid[None, None, None, :] - be[..., None]     # (nb, n1, n2, nE)
    return np.einsum("bij,bije->ije", w,
                     np.exp(-d * d / (2 * sigma * sigma)))


# ------------------------------------------------------------------ 脊顶提取

def test_crest_field_finds_the_ridge_crest_not_the_flanks():
    """脊顶提取必须给出**过 E_F 的那条带**的位置, 而不是 A 的极大点。

    n 取偶数 ⇒ 0.5 正好落在网格点上(gamma 网格 x_i = i/n)。
    """
    n = 60
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    e_grid = np.linspace(-1.0, 1.0, 401)
    e_f = 0.0
    sigma = 0.06
    # 一条斜带: E(kx) = 2*(kx-0.5); 过 E_F 在 kx=0.5
    band_e = 2.0 * (kx - 0.5)
    a = _spectral_from_bands(band_e, e_grid, sigma)

    crest = crest_field(a, e_grid, e_f, sigma, max_shift=1.0,
                        min_weight_frac=0.0)
    i_crest = int(np.argmin(np.abs(fx - 0.5)))
    # 脊顶所在点上, E* ≈ E_F(窗口内加权平均, 允许小的截断不对称)
    assert abs(crest[i_crest, i_crest] - e_f) < 1e-3
    # 远离脊顶的地方(带能量远离 E_F)应当被闸门②挡掉
    assert not np.isfinite(crest[0, 0])


def test_crest_field_rejects_a_non_local_maximum():
    """闸门①: 窗边缘上强带的尾巴**不是**峰 ⇒ 该点必须给 NaN。

    构造: 在搜索窗边缘放一个强带, 它在窗内单调(不是极大)。
    """
    n = 31
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    e_grid = np.linspace(-1.0, 1.0, 401)
    sigma = 0.06
    # 带能量 = 0.5(|kx-0.5|) + 0.35: 在 kx>0.5 一侧单调下降, 不是极大
    band_e = 0.5 * np.abs(kx - 0.5) + 0.35
    a = _spectral_from_bands(band_e, e_grid, sigma)

    crest = crest_field(a, e_grid, 0.0, sigma, max_shift=1.0,
                        min_weight_frac=0.0)
    # 靠近带最小值(kx=0.5)的点: 窗内的 A 单调 ⇒ 不是真极大 ⇒ NaN
    i_c = int(np.argmin(np.abs(fx - 0.5)))
    assert not np.isfinite(crest[i_c, i_c])


def test_crest_field_gate_three_removes_regions_without_weight_at_ef():
    """闸门③: A(k,E_F) 低于场最大值 2% 的区域必须被清掉(否则会画出幽灵小圈)。

    直接构造场: 一半 k 有强权重, 另一半只有 0.1% —— 后者必须被挡掉。
    ⚠ 判据是**相对**的(A(k,E_F) > 0.02·max A), 与论文 §III.B 的写法一致。
    """
    n = 41
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    e_grid = np.linspace(-1.0, 1.0, 401)
    sigma = 0.06
    amp = np.where(kx < 0.5, 1.0, 1e-3)[:, :, None]
    a = amp * np.exp(-(e_grid[None, None, :] ** 2) / (2 * sigma * sigma))

    crest = crest_field(a, e_grid, 0.0, sigma, max_shift=0.25,
                        min_weight_frac=0.02)
    strong = crest[fx < 0.4]
    weak = crest[fx > 0.6]
    assert np.isfinite(strong).any()
    assert not np.isfinite(weak).any()


# ------------------------------------------------------------------ 协变性

def _ops(rotations):
    n = len(rotations)
    return SymmetryOps(rotations=np.asarray(rotations),
                       translations=np.zeros((n, 3), dtype=float))


def test_dispersion_is_zero_for_a_covariant_field():
    """在每个操作下不变的场 ⇒ δ(R) = 0。

    用 sin²(2πx)+sin²(2πy): 在 90° 旋转 (x,y)→(−y,x) 下不变。
    """
    n = 20
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    field = np.sin(2 * np.pi * kx) ** 2 + np.sin(2 * np.pi * ky) ** 2
    ops = _ops([np.eye(3, dtype=int), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    dmax, dmean = dispersion(field, ops, np.eye(3))
    assert dmax == pytest.approx(0.0, abs=1e-12)
    assert dmean == pytest.approx(0.0, abs=1e-12)


def test_dispersion_is_one_when_a_peak_is_mapped_to_zero():
    """一个谱峰被映到零权重 ⇒ δ = 1(论文里的 1.000e+00)。

    峰放在 (i=0, j=1): 90° 旋转把它映到 (n−1, 0), 那里权重为 0。
    """
    n = 20
    field = np.zeros((n, n))
    field[0, 1] = 1.0
    ops = _ops([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]])   # 90° 旋转
    dmax, _ = dispersion(field, ops, np.eye(3))
    assert dmax == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ 几何

def test_point_to_segment_distance_is_smaller_than_vertex_distance():
    """点到线段必须 ≤ 点到顶点(顶点只是采样点)。"""
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    p = np.array([[0.5, 0.1]])
    d_seg = point_to_segment_distance(p, curve)
    # 最近顶点是 (0,0) 或 (1,0), 距离 0.1; 点到线段是 0.1 —— 相等
    assert d_seg[0] == pytest.approx(0.1, abs=1e-12)

    curve2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    p2 = np.array([[0.5, 0.0]])
    d_seg2 = point_to_segment_distance(p2, curve2)
    # 点到线段 = 到直线 y=x 的距离 = 0.5/sqrt(2) ≈ 0.3536
    assert d_seg2[0] == pytest.approx(0.5 / np.sqrt(2), abs=1e-12)


def test_verify_fermi_surface_reports_full_coincidence_for_identical_curves():
    """同一条曲线与自己 ⇒ 重合率 100%, 距离 0。"""
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    rep = verify_fermi_surface(curve, curve, 1.0)
    assert rep["coincidence"] == pytest.approx(1.0)
    assert rep["median"] == pytest.approx(0.0, abs=1e-12)


def test_verify_fermi_surface_flags_a_displaced_curve():
    """整体平移 3 个网格间距的曲线 ⇒ 重合率 0。

    距离是**点到线段**: (1,1) 到平移后折线的最近点是端点 (3,0),
    距离 √5 ≈ 2.236 —— 比到 (4,1) 的 3 更近 ⇒ 中位数是 (2.236+3)/2。
    """
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    shifted = curve + np.array([3.0, 0.0])
    rep = verify_fermi_surface(curve, shifted, 1.0)
    assert rep["coincidence"] == pytest.approx(0.0)
    assert rep["median"] == pytest.approx((np.sqrt(5.0) + 3.0) / 2.0, abs=1e-12)
    assert rep["max"] == pytest.approx(3.0, abs=1e-12)


def test_fold_to_bz_maps_a_shifted_vertex_back():
    """折回 BZ: 平移一个倒格矢的顶点应回到原点附近。"""
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3) / 2])
    cart = np.array([[1.0, 0.0]])
    folded = fold_to_bz(cart, b1, b2)
    assert np.linalg.norm(folded) < 1e-12


def test_split_at_jumps_cuts_a_segment_crossing_the_boundary():
    """折回后跨越 BZ 边界的段必须被切开。"""
    curve = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
    pieces = split_at_jumps(curve, factor=3.0)
    assert len(pieces) == 2
    assert all(len(p) == 2 for p in pieces)


# ------------------------------------------------------------------ 费米速度

def test_fermi_velocity_is_nonzero_and_scales_with_the_slope():
    """陡带的费米速度应大于缓带(同构造、不同斜率)。"""
    n = 121
    fx = _grid(n)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    e_grid = np.linspace(-1.0, 1.0, 401)
    sigma = 0.06
    b2d = np.eye(2)

    def _crest_and_v(slope):
        band_e = slope * (kx - 0.5)
        a = _spectral_from_bands(band_e, e_grid, sigma)
        crest = crest_field(a, e_grid, 0.0, sigma, max_shift=1.0,
                            min_weight_frac=0.0)
        return crest

    v_steep = fermi_velocity(_crest_and_v(4.0), fx, fx, b2d, e_f=0.0, n_rays=36)
    v_soft = fermi_velocity(_crest_and_v(0.5), fx, fx, b2d, e_f=0.0, n_rays=36)
    assert len(v_steep["v_f"]) > 0
    assert len(v_soft["v_f"]) > 0
    steep = np.nanmedian(v_steep["v_f"])
    soft = np.nanmedian(v_soft["v_f"])
    assert steep > soft


def test_velocity_to_si_converts_fractional_derivative_to_metres_per_second():
    """单位换算: 分数坐标下的梯度 → m/s(∇_k E = ∇_k_frac · B⁻¹)。

    ⚠ Å→m 是 **1e-10**(不是 1e10)。量级自检: dE/dk ~ 1 eV·Å ⇒
    v ~ 1/ħ Å/s ≈ 1.5×10⁶ m/s —— 与金属的真实 v_F 同量级。
    """
    b2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = velocity_to_si(np.array([1.0, 0.0]), b2d)
    hbar = 6.582119569e-16
    expect = 1.0 / hbar * 1.0e-10
    assert v[0] == pytest.approx(expect, rel=1e-9)
    # 量级自检: 1 eV·Å 的斜率 ⇒ 10⁵~10⁷ m/s
    assert 1e5 < v[0] < 1e7


def test_velocity_to_si_uses_the_inverse_b_not_the_b_norm():
    """分数基不正交 ⇒ 必须用 B⁻¹ 而不是 |B|。

    取一个非正交的 b2d: |B| 与 |B⁻¹| 相差很大, 用错会给出明显不同的答案。
    """
    b2d = np.array([[1.0, 0.0], [0.5, 0.5]])       # 斜交的倒格矢
    grad = np.array([[1.0, 0.0]])
    v = velocity_to_si(grad, b2d)
    expect = (np.linalg.norm(grad @ np.linalg.inv(b2d))
              / 6.582119569e-16 * 1.0e-10)
    assert v[0] == pytest.approx(expect, rel=1e-9)


# ------------------------------------------------------------------ npz 载入

def test_load_spectral_npz_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_spectral_npz(str(tmp_path / "nope.npz"))


# ------------------------------------------------------------------ 原始谱重建

def test_spectral_at_ef_from_bands_matches_a_direct_broadening():
    """从逐带数据重建的 A(k,E_F) 要与直接展宽一致。"""
    from unfoldfs.fs import spectral_at_ef_from_bands
    n1, n2, n_band, sigma = 5, 4, 3, 0.06
    e_grid = np.linspace(-1.0, 1.0, 401)
    e_f = 0.1
    rng = np.random.default_rng(7)
    band_e = rng.normal(size=(n_band, n1, n2)) * 0.3
    band_w = rng.uniform(0.5, 1.5, size=(n_band, n1, n2))
    valid = np.ones_like(band_e, dtype=bool)
    got = spectral_at_ef_from_bands(band_e, band_w, valid, e_grid, e_f, sigma)
    # 逐点手算
    i_ef = int(np.argmin(np.abs(e_grid - e_f)))
    for i in range(n1):
        for j in range(n2):
            want = sum(band_w[b, i, j] * np.exp(
                -(e_grid[i_ef] - band_e[b, i, j]) ** 2 / (2 * sigma ** 2))
                for b in range(n_band))
            assert got[i, j] == pytest.approx(want, rel=1e-12)


def test_spectral_at_ef_from_bands_ignores_invalid_slots():
    """补位槽(valid=False)必须被排除, 否则 +inf 能量会把展宽污染成 NaN。"""
    from unfoldfs.fs import spectral_at_ef_from_bands
    e_grid = np.linspace(-1.0, 1.0, 401)
    band_e = np.full((2, 3, 3), np.inf)
    band_w = np.ones((2, 3, 3))
    valid = np.zeros((2, 3, 3), dtype=bool)
    valid[0, :, :] = True
    got = spectral_at_ef_from_bands(band_e, band_w, valid, e_grid, 0.0, 0.06)
    assert np.isfinite(got).all()
    assert (got >= 0).all()


# ------------------------------------------------------------------ 轨道成分

def test_orbital_fraction_at_peak_is_normalized_and_picks_the_crossing_band():
    """轨道成分: 取能量最接近脊顶的那条带, 并归一化到和为 1。"""
    from unfoldfs.fs import orbital_fraction_at_peak
    n_band, n1, n2, n_orb = 3, 2, 2, 4
    band_e = np.zeros((n_band, n1, n2))
    band_e[0] = 0.5          # 远离脊顶
    band_e[1] = 0.0          # 恰在 E*
    band_e[2] = -0.4
    orb = np.zeros((n_band, n1, n2, n_orb))
    orb[1, :, :, 2] = 3.0    # 第 1 带集中在第 3 个轨道
    orb[0] = 1.0
    orb[2] = 1.0
    valid = np.ones((n_band, n1, n2), dtype=bool)
    e_star = np.zeros((n1, n2))
    frac = orbital_fraction_at_peak(orb, band_e, valid, e_star)
    assert frac is not None
    assert frac.shape == (n1, n2, n_orb)
    assert np.allclose(frac.sum(axis=-1), 1.0)
    assert np.allclose(frac[:, :, 2], 1.0)      # 全部落在第 3 个轨道


def test_orbital_fraction_at_peak_returns_none_without_orbital_data():
    from unfoldfs.fs import orbital_fraction_at_peak
    assert orbital_fraction_at_peak(None, np.zeros((1, 2, 2)), np.ones((1, 2, 2)),
                                    np.zeros((2, 2))) is None


# ------------------------------------------------------------------ 模糊

def test_crest_field_blur_smooths_the_jump_between_bands():
    """主峰换带处 E* 会**跳变**; 笛卡尔模糊把阶跃摊成斜坡(逐格梯度变小)。

    ⚠ 模糊治的是"折角", 不是横穿 BZ 的直线 —— 后者由 ``split_at_jumps``
    治(见 ``unfoldfs.fs._split_jumps`` 与论文 §III.A 的坑)。
    """
    from unfoldfs.fs import crest_field
    n = 41
    fx = np.linspace(0.0, 1.0, n, endpoint=False)
    b2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    e_grid = np.linspace(-1.0, 1.0, 401)
    kx = fx[:, None] * np.ones(n)
    # 一条带, 但在 kx>0.5 换成另一条(能量跳变 0.2 eV)
    band_e = np.where(kx > 0.5, 0.2, 0.0)
    d = e_grid[None, None, :] - band_e[:, :, None]
    a = np.exp(-d * d / (2 * 0.06 ** 2))
    raw = crest_field(a, e_grid, 0.0, 0.06, max_shift=1.0,
                      min_weight_frac=0.0, b2d=b2d, blur_cells=0.0)
    blurred = crest_field(a, e_grid, 0.0, 0.06, max_shift=1.0,
                          min_weight_frac=0.0, b2d=b2d, blur_cells=1.0)

    def _max_grad(f):
        return float(np.nanmax(np.abs(np.gradient(np.nan_to_num(f, nan=0.0),
                                                axis=0))))

    # 跳变的高度不变, 但逐格梯度被摊薄
    assert _max_grad(blurred) < _max_grad(raw)
    assert np.nanmax(raw) - np.nanmin(raw) == pytest.approx(
        np.nanmax(blurred) - np.nanmin(blurred), rel=0.25)


def test_split_jumps_cuts_a_segment_that_crosses_the_bz_boundary():
    """折回 BZ 后, 跨越边界的那一跳必须被切开。

    不切开的话, 这一跳会被画成**一条横穿整个布里渊区的直线** —— 那不是
    费米面, 是折回造成的假象(论文图 7 踩过这个坑)。
    """
    from unfoldfs.fs import _split_jumps
    curve = np.array([[0.90, 0.0], [0.95, 0.0], [1.40, 0.0], [1.45, 0.0]])
    pieces = _split_jumps(curve, factor=3.0)
    assert len(pieces) == 2
    assert all(len(p) == 2 for p in pieces)
    # 没有异常跳变时不该切
    smooth = np.array([[0.0, 0.0], [0.05, 0.0], [0.10, 0.0], [0.15, 0.0]])
    assert len(_split_jumps(smooth, factor=3.0)) == 1


# ------------------------------------------------------------------ 场的轨道平均

def test_symmetrize_field_is_nan_safe():
    """全是 NaN 的点必须保持 NaN。

    写成 0 会把"这里没有峰"当成"峰在 E=0", 随后被闸门整片删掉 ⇒
    单层 1T-TaS2 直接变 0 段(见 feedback_fs_curve_pipeline 的第 4 条)。
    """
    from unfoldfs.fs import symmetrize_field
    n = 8
    fx = np.linspace(0.0, 1.0, n, endpoint=False)
    field = np.full((n, n), np.nan)
    field[0, 0] = 1.0
    ops = SymmetryOps(rotations=np.eye(3, dtype=int)[None, :, :],
                      translations=np.zeros((1, 3)))
    out = symmetrize_field(field, fx, fx, ops)
    assert out[0, 0] == pytest.approx(1.0)     # 只有恒等像 ⇒ 不变
    assert np.isnan(out[1, 1])                   # 其余仍是 NaN(不是 0)


def test_symmetrize_field_always_includes_the_point_itself():
    """没有操作把网格映到自身时, 轨道不能是空的(否则全场变 NaN)。"""
    from unfoldfs.fs import symmetrize_field
    n = 6
    fx = np.linspace(0.0, 1.0, n, endpoint=False)
    field = np.ones((n, n))
    # 一个与网格不兼容的旋转
    ops = SymmetryOps(rotations=np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]]),
                      translations=np.zeros((1, 3)))
    out = symmetrize_field(field, fx, fx, ops)
    assert np.isfinite(out).all()


def test_symmetrize_field_averages_over_the_orbit():
    """轨道内的值被平均(对称相关的点给出同一个值)。"""
    from unfoldfs.fs import symmetrize_field
    n = 12
    fx = np.linspace(0.0, 1.0, n, endpoint=False)
    kx, ky = np.meshgrid(fx, fx, indexing="ij")
    # 恒等 + 把 x 轴映到 y 轴的交换
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    ops = SymmetryOps(rotations=np.array([np.eye(3, dtype=int), swap]),
                      translations=np.zeros((2, 3)))
    field = kx ** 2 + 3 * ky ** 2               # 交换 x/y 后不同 ⇒ 不是不变量
    out = symmetrize_field(field, fx, fx, ops)
    # 轨道内(x,y)与(y,x)的值被平均 ⇒ 关于 x=y 对称
    i, j = 1, 4
    assert out[i, j] == pytest.approx(out[j, i])
    assert out[i, j] != pytest.approx(field[i, j])
