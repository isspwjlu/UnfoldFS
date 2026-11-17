"""VASP 输出解析: EIGENVAL / DOSCAR / PROCAR(LORBIT=12 复投影)。

实测要点 (VASP 6.6.0, 3x3x1 MoS2 超胞 fixture):

- EIGENVAL 第 6 行 = ``NIONS NKDIM NBANDS``; 第 7 行起是 k 点行
  (只有 4 个数: kx ky kz weight), 每个 k 点块之前有一个空行 —— 解析必须跳过空行,
  否则第 2 个 k 点起会整体错位。
- DOSCAR 第 6 行 = ``EMAX EMIN NEDOS EFERMI ...``; **E_F 是第 4 个数**
  (不是第 2 个的 EMIN)。
- PROCAR 第 2 行 = ``# of k-points / # of bands / # of ions``; k 点行是
  ``k-point 1 : x y z weight = ...`` (坐标在第 3~5 个 token)。每个 band 含两个块:

  - 模方块: 表头 ``ion s py ... x2-y2 tot`` (norb+2 个字段), 行 = ion + norb 个
    |p|^2 + tot;
  - 复投影块: 表头 ``ion s ... x2-y2`` (norb+1 个字段), 行 = ion + 9 对 (Re, Im)
    **交错**排列 + 末列 |c|^2 (2*norb+2 个字段)。

  复数按 (Re, Im) 交错解释才与模方块逐轨道自洽; 末列 Σ|c|^2 与由复系数算出的
  Σ_m|c_m|^2 一致到 <=7.6e-4 (纯打印舍入)。
  ``LORBIT=11`` 不写复投影块 -> 显式报错提示改用 ``LORBIT=12``。
- PROCAR 的 k 点行与 EIGENVAL 的 k 点行给出**同一组数值**(本 fixture 实测:
  Γ 与 (0.1,0,0) 完全一致), 即两者同约定; 需要笛卡尔坐标时自行用
  ``unfoldfs.geometry.reciprocal`` 变换(倒空间用 M^-T)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = ["BandData", "ProcarData", "read_eigenval", "read_doscar_fermi",
           "read_procar"]


@dataclass(frozen=True)
class BandData:
    """EIGENVAL 的能带数据。"""

    kpoints: np.ndarray      # (nk,3) 分数坐标(超胞倒空间基)
    energies: np.ndarray     # (nk, nb) eV
    occupations: np.ndarray  # (nk, nb)
    efermi: float            # EIGENVAL 不含 E_F -> NaN(用 read_doscar_fermi 取)


@dataclass(frozen=True)
class ProcarData:
    """PROCAR 的轨道投影数据 (LORBIT=12)。"""

    kpoints: np.ndarray      # (nk,3) VASP 原样打印的 k 点
    weights: np.ndarray      # (nk, nb, nions, norb) |p|^2 (模方块)
    coeffs: np.ndarray       # (nk, nb, nions, norb) complex (复投影块)
    has_phase: bool          # 是否存在复投影块(无则 read_procar 直接报错)


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def _skip_blank(lines: list[str], idx: int) -> int:
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _find_prefix(lines: list[str], idx: int, prefix: str) -> int:
    """从 idx 起找第一个以 prefix 开头的行; 找不到返回 -1。"""
    while idx < len(lines) and not lines[idx].lstrip().startswith(prefix):
        idx += 1
    return idx if idx < len(lines) else -1


def _eigenval_kpoint(line: str, path: str) -> list[float]:
    """k 点行: 3 个分数坐标(+ 可选权重)。"""
    try:
        return [float(x) for x in line.split()[:3]]
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"EIGENVAL k 点行无法解析: {line.strip()!r} ({path})") from exc


def read_eigenval(path: str) -> BandData:
    """读 EIGENVAL -> 带数、能量(eV)、占据数与 k 点(分数坐标)。"""
    lines = _read_lines(path)
    if len(lines) < 7:
        raise ValueError(f"EIGENVAL 行数不足, 无法解析: {path}")
    hdr = lines[5].split()
    nkpts, nbands = int(hdr[-2]), int(hdr[-1])
    kpoints = np.zeros((nkpts, 3))
    energies = np.zeros((nkpts, nbands))
    occupations = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        idx = _skip_blank(lines, idx)
        if idx >= len(lines):
            raise ValueError(f"EIGENVAL 在第 {ik + 1} 个 k 点处提前结束: {path}")
        kpoints[ik] = _eigenval_kpoint(lines[idx], path)
        idx += 1
        for ib in range(nbands):
            if idx >= len(lines):
                raise ValueError(
                    f"EIGENVAL 在 k={ik + 1} 的第 {ib + 1} 条带处提前结束: {path}")
            v = lines[idx].split()
            try:
                energies[ik, ib] = float(v[1])
                occupations[ik, ib] = float(v[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"EIGENVAL 带数据行无法解析: {lines[idx].strip()!r}") from exc
            idx += 1
    return BandData(kpoints, energies, occupations, float("nan"))


def read_doscar_fermi(path: str) -> float:
    """DOSCAR 第 6 行: EMAX EMIN NEDOS EFERMI ... -> E_F (第 4 个数, eV)。"""
    lines = _read_lines(path)
    if len(lines) < 6:
        raise ValueError(f"DOSCAR 行数不足, 无法解析 E_F: {path}")
    fields = lines[5].split()
    if len(fields) < 4:
        raise ValueError(f"DOSCAR 第 6 行无法解析 E_F: {lines[5].strip()!r}")
    return float(fields[3])


def _procar_counts(lines: list[str]) -> tuple[int, int, int]:
    if len(lines) < 2:
        raise ValueError("PROCAR 行数不足, 无法解析头部")
    nums = [int(x) for x in re.findall(r"\d+", lines[1])]
    if len(nums) < 3:
        raise ValueError(f"PROCAR 头部无法解析: {lines[1].strip()!r}")
    return nums[0], nums[1], nums[2]


def _procar_norb(lines: list[str]) -> int:
    """轨道数 = 模方块 ion 表头的列数 - 2 (去掉 'ion' 与 'tot')。

    首个 'ion' 行必是模方块表头(复投影块的表头出现在每个 band 的模方块之后)。
    """
    for line in lines[2:]:
        if line.lstrip().startswith("ion"):
            return len(line.split()) - 2
    raise ValueError("PROCAR: 未找到 ion 表头")


def _procar_kpoint(line: str) -> list[float]:
    """k 点行: ' k-point 1 : x y z weight = ...' -> 坐标在第 3~5 个 token。"""
    toks = line.split()
    try:
        return [float(toks[3]), float(toks[4]), float(toks[5])]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"PROCAR k 点行无法解析: {line.strip()!r}") from exc


def read_procar(path: str) -> ProcarData:
    """读 PROCAR -> 模方块 |p|^2 与复投影系数 (需要 LORBIT=12)。"""
    lines = _read_lines(path)
    nk, nb, nions = _procar_counts(lines)
    norb = _procar_norb(lines)
    weights = np.zeros((nk, nb, nions, norb))
    coeffs = np.zeros((nk, nb, nions, norb), dtype=complex)
    kpoints = np.zeros((nk, 3))
    has_phase = False
    n_mod = norb + 2          # 模方块行: ion + norb 个 |p|^2 + tot
    n_cplx = 2 * norb + 2     # 复投影行: ion + norb 对 (Re, Im) + |c|^2
    idx = 0
    for ik in range(nk):
        idx = _find_prefix(lines, idx, "k-point")
        if idx < 0:
            raise ValueError(f"PROCAR 缺少第 {ik + 1} 个 k 点: {path}")
        kpoints[ik] = _procar_kpoint(lines[idx])
        idx += 1
        for ib in range(nb):
            idx = _find_prefix(lines, idx, "band")
            if idx < 0:
                raise ValueError(
                    f"PROCAR 缺少 k={ik + 1} 的第 {ib + 1} 条带: {path}")
            rows = []
            j = idx + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("band") or s.startswith("k-point"):
                    break
                # 只收数据行: 'tot' / 'charge' 汇总行与 ion 表头都以字母开头
                if s and s[0].isdigit():
                    rows.append(s.split())
                j += 1
            idx = j
            for v in rows:
                ia = int(v[0]) - 1
                if not 0 <= ia < nions:
                    continue
                if len(v) == n_mod:
                    weights[ik, ib, ia] = [float(x) for x in v[1:1 + norb]]
                elif len(v) == n_cplx:
                    has_phase = True
                    for m in range(norb):
                        coeffs[ik, ib, ia, m] = (float(v[1 + 2 * m])
                                                  + 1j * float(v[2 + 2 * m]))
    if not has_phase:
        raise ValueError(
            "PROCAR 中没有复投影块 —— 需要 LORBIT=12; LORBIT=11 不写相位")
    return ProcarData(kpoints, weights, coeffs, has_phase)
