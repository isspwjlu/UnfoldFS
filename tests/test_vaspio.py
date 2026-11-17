"""vaspio 模块测试: EIGENVAL / DOSCAR / PROCAR(LORBIT=12 复投影) 解析。"""
import math

import numpy as np
import pytest

from unfoldfs.vaspio import (read_doscar_fermi, read_eigenval, read_procar)
from tests.conftest import FIXTURE_DOSCAR, FIXTURE_EIGENVAL, FIXTURE_PROCAR

# VASP 只把复投影系数打印到 3 位小数, 且模方块与复投影块的归一化/取整口径不同:
# 实测 58320 个逐轨道比较中 0.51% 超过 2e-3(最大 7.5e-3) -> 容差取 1e-2。
AMP_ATOL = 1e-2
# 复投影块末列 (Σ|c|^2) 与由复系数算出的 Σ_m|c_m|^2 一致到 <=7.6e-4(纯打印舍入)。
SUM_ATOL = 1e-3


# ----------------------------------------------------------------- EIGENVAL
def test_read_eigenval_shape():
    b = read_eigenval(str(FIXTURE_EIGENVAL))
    assert b.kpoints.shape == (2, 3)
    assert b.energies.shape == (2, 120)
    assert b.occupations.shape == (2, 120)
    assert np.allclose(b.kpoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(b.kpoints[1], [0.1, 0.0, 0.0])


def test_read_eigenval_skips_blank_line_before_second_kpoint():
    """VASP 在每个 k 点块之间插空行; 不跳过就会把空行当 k 点读崩。"""
    b = read_eigenval(str(FIXTURE_EIGENVAL))
    assert np.allclose(b.kpoints[1], [0.1, 0.0, 0.0])
    # 带号/能量对必须对齐到正确的带(否则第 2 个 k 点的能量会串行)
    assert abs(b.energies[1, 0] - (-16.186923)) < 1e-6
    assert abs(b.energies[1, -1] - 4.246739) < 1e-6


def test_read_eigenval_values_match_raw_file():
    b = read_eigenval(str(FIXTURE_EIGENVAL))
    assert abs(b.energies[0, 0] - (-16.202533)) < 1e-6
    assert abs(b.energies[0, -1] - 4.189011) < 1e-6
    assert b.occupations[0, 0] == 1.0
    assert b.occupations[0, -1] == 0.0
    # 占据数沿带号单调(占据 -> 空), 中间存在分界
    occ = b.occupations[0]
    filled = int((occ > 0.5).sum())
    assert 0 < filled < 120
    assert np.all(occ[:filled] > 0.5) and np.all(occ[filled:] < 0.5)


def test_read_eigenval_efermi_is_nan():
    """EIGENVAL 不含费米能; E_F 由 DOSCAR 提供。"""
    b = read_eigenval(str(FIXTURE_EIGENVAL))
    assert math.isnan(b.efermi)


# ------------------------------------------------------------------- DOSCAR
def test_read_doscar_fermi_is_fourth_field_of_line6():
    ef = read_doscar_fermi(str(FIXTURE_DOSCAR))
    lines = open(str(FIXTURE_DOSCAR)).read().splitlines()
    assert abs(ef - float(lines[5].split()[3])) < 1e-12


def test_read_doscar_fermi_value_and_not_emax_emin():
    ef = read_doscar_fermi(str(FIXTURE_DOSCAR))
    assert abs(ef - (-1.84458790)) < 1e-8
    lines = open(str(FIXTURE_DOSCAR)).read().splitlines()
    fields = [float(x) for x in lines[5].split()]
    # 第 6 行 = EMAX EMIN NEDOS EFERMI ...; E_F 不是 EMAX/EMIN/NEDOS
    assert ef != fields[0] and ef != fields[1] and ef != fields[2]


# ------------------------------------------------------------------- PROCAR
def test_read_procar_shapes():
    p = read_procar(str(FIXTURE_PROCAR))
    assert p.has_phase is True
    assert p.kpoints.shape == (2, 3)
    assert p.weights.shape == (2, 120, 27, 9)
    assert p.coeffs.shape == p.weights.shape
    assert np.iscomplexobj(p.coeffs)


def test_read_procar_kpoints():
    p = read_procar(str(FIXTURE_PROCAR))
    assert np.allclose(p.kpoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(p.kpoints[1], [0.1, 0.0, 0.0])


def test_read_procar_has_phase_block():
    p = read_procar(str(FIXTURE_PROCAR))
    assert p.has_phase is True
    assert p.coeffs.shape[:3] == p.weights.shape[:3]
    # 模方块与复投影块模长自洽
    # 注意: VASP 两块的归一化/取整口径不同 —— 实测 58320 个比较中 0.51% 超 2e-3
    # (最大 7.5e-3), 故容差取 1e-2; 末列(Σ|c|^2)与模方块逐离子合计则一致到 ≤8e-4
    amp = np.abs(p.coeffs)
    assert np.allclose(amp ** 2, p.weights, atol=AMP_ATOL)


def test_read_procar_complex_last_column_is_sum_of_squared_moduli():
    """复投影块末列 = 该离子 Σ_m|c_m|^2, 与解析出的复系数自洽(≤7.6e-4)。

    末列直接从未经解析的原始行里取, 因此这是对 (Re, Im) 交错解释的独立交叉校验:
    若把 18 个数按 "9×(Re,Re,...)" 分组解释, 该恒等式会立刻失效。
    """
    p = read_procar(str(FIXTURE_PROCAR))
    raw_last = _raw_complex_last_column(str(FIXTURE_PROCAR))
    parsed = (np.abs(p.coeffs[0, 0]) ** 2).sum(axis=1)
    assert raw_last.shape == (27,)
    assert np.allclose(raw_last, parsed, atol=SUM_ATOL)


def _raw_complex_last_column(path):
    """从原始 PROCAR 文本里取第 1 个 k 点第 1 个 band 的复投影块末列。"""
    rows = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        toks = s.split()
        if toks and toks[0].isdigit() and len(toks) == 20:
            rows.append(float(toks[19]))
            if len(rows) == 27:
                break
    return np.array(rows)


def test_read_procar_golden_values():
    """直接从原始文件取的值: 解析错列/错行会立刻暴露。"""
    p = read_procar(str(FIXTURE_PROCAR))
    # k=Gamma, band 1, ion 1 (Mo1), s 轨道
    assert abs(p.weights[0, 0, 0, 0] - 0.014) < 1e-12
    assert abs(p.coeffs[0, 0, 0, 0].real - (-0.079)) < 1e-12
    assert abs(p.coeffs[0, 0, 0, 0].imag - (-0.083)) < 1e-12
    # k=Gamma, band 82, ion 1, dz2
    assert abs(p.weights[0, 81, 0, 6] - 0.138) < 1e-12
    assert abs(p.coeffs[0, 81, 0, 6].real - 0.254) < 1e-12
    assert abs(p.coeffs[0, 81, 0, 6].imag - 0.267) < 1e-12
    # k=(0.1,0,0), band 1, ion 1, s 轨道
    assert abs(p.coeffs[1, 0, 0, 0].real - 0.093) < 1e-12
    assert abs(p.coeffs[1, 0, 0, 0].imag - (-0.066)) < 1e-12


def test_read_procar_equivalent_copies_have_same_value():
    """同一原胞原子的副本, 复投影值相同。

    实测: 3x3x1 超胞里同一原胞原子的 9 个副本中, 由**超胞晶格平移**
    (3a1 / 3a2) 联系的那 3 个复值相同; 跨平移的则相差 e^{i K.T}。
    这里只断言"同一组内"相同: 连续 3 个离子为 1 组。
    """
    p = read_procar(str(FIXTURE_PROCAR))
    c = p.coeffs[0, 0, :, 0]          # band 0, k=Gamma, s 轨道
    for start in range(0, 27, 3):
        group = c[start:start + 3]
        assert np.allclose(group, group[0], atol=1e-6)


def test_read_procar_copies_form_three_equivalent_groups():
    """更强的判别力断言: band 82 的 dz2 分量上, 9 个 Mo 副本恰好取 3 个值,
    每个值出现 3 次(超胞内的三重对称等价类)。"""
    p = read_procar(str(FIXTURE_PROCAR))
    c = p.coeffs[0, 81, 0:9, 6]       # band 82, k=Gamma, dz2, 9 个 Mo 副本
    uniq = np.unique(np.round(c, 12))
    assert uniq.size == 3
    for u in uniq:
        assert int(np.sum(np.isclose(c, u, atol=1e-9))) == 3


def _lorbit11_procar(tmp_path):
    """由真实 PROCAR 剥掉复投影块, 造一个 LORBIT=11 风格的文件(只有模方块)。"""
    lines = FIXTURE_PROCAR.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("charge"):
            continue
        if s.startswith("ion"):
            # 复投影块的表头比模方块表头少一列("tot")
            if len(s.split()) != 11:
                continue
        toks = s.split()
        if toks and toks[0].isdigit() and len(toks) == 20:
            continue
        kept.append(ln)
    kept[0] = "PROCAR lm decomposed\n"
    path = tmp_path / "PROCAR_L11"
    path.write_text("".join(kept), encoding="utf-8")
    return path


def test_read_procar_without_phase_block_raises(tmp_path):
    """LORBIT=11 不写复投影块 -> 必须报错并提示改用 LORBIT=12。"""
    path = _lorbit11_procar(tmp_path)
    with pytest.raises(ValueError, match="LORBIT=12"):
        read_procar(str(path))


def test_read_procar_missing_ion_header_raises(tmp_path):
    path = tmp_path / "PROCAR_BROKEN"
    path.write_text("PROCAR lm decomposed + phase\n"
                    "# of k-points: 1 # of bands: 1 # of ions: 1\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="ion"):
        read_procar(str(path))
