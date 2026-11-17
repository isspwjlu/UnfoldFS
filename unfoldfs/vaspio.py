"""VASP 输出解析: EIGENVAL / DOSCAR / PROCAR(LORBIT=12 复投影)。

实测要点 (VASP 6.6.0, 3x3x1 MoS2 超胞 fixture):

- EIGENVAL 第 1 行 = ``NIONS NIONS ISPIN ISYM`` (ISPIN 是第 3 个字段)。
  **仅支持 ISPIN=1**: 自旋极化 (ISPIN=2) 的带行是 5 个字段
  (E↑ occ↑ E↓ occ↓), 按 3 字段解析会把自旋下半**静默丢掉**而数组看起来完整
  -> 显式报错。
- EIGENVAL 第 6 行 = ``NELECT NKPT NBANDS`` (fixture 实测 ``162 2 120``:
  162 = 27 原子 × 6 电子 = 81 条占据带 × 2, 2 = 文件里**实际**的 k 点块数,
  120 = 带数)。头字段一律用**位置索引**, 不用负索引 —— VASP 若加列, 负索引
  会静默取错列; 且 NKPT 必须与文件里实际的 k 点块数一致(见 ``_count_kpoint_rows``)。
- EIGENVAL 第 7 行起是 k 点行 (只有 4 个数: kx ky kz weight), 每个 k 点块之前有一个
  空行 —— 解析必须跳过空行, 否则第 2 个 k 点起会整体错位。
  带行是 3 个字段 (带号 能量 占据), 带号必须从 1 起连续递增。
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
  每个 band 块必须**恰好**给出 nions 条模方行 + nions 条复投影行: 文件被截断
  (作业被 kill / 磁盘满)时 VASP 只写了一半, 静默补 0 会让该带的轨道投影整个消失。
- PROCAR 的 k 点行与 EIGENVAL 的 k 点行给出**同一组数值**(本 fixture 实测:
  Γ 与 (0.1,0,0) 完全一致), 即两者同约定(均为**分数坐标**); 需要笛卡尔坐标时自行用
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
    """EIGENVAL 的能带数据 (仅 ISPIN=1)。"""

    kpoints: np.ndarray      # (nk,3) 分数坐标(超胞倒空间基)
    energies: np.ndarray     # (nk, nb) eV
    occupations: np.ndarray  # (nk, nb)
    efermi: float            # EIGENVAL 不含 E_F -> NaN(用 read_doscar_fermi 取)


@dataclass(frozen=True)
class ProcarData:
    """PROCAR 的轨道投影数据 (LORBIT=12)。"""

    kpoints: np.ndarray      # (nk,3) 分数坐标(VASP 原样打印, 与 EIGENVAL 同约定)
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


def _eigenval_ispin(lines: list[str], path: str) -> int:
    """EIGENVAL 第 1 行 = NIONS NIONS ISPIN ISYM -> ISPIN(第 3 个字段)。"""
    fields = lines[0].split()
    if len(fields) < 3:
        raise ValueError(
            f"EIGENVAL 第 1 行无法解析 ISPIN: {lines[0].strip()!r} ({path})")
    try:
        return int(fields[2])
    except ValueError as exc:
        raise ValueError(
            f"EIGENVAL 第 1 行无法解析 ISPIN: {lines[0].strip()!r} ({path})") from exc


def _eigenval_header(line: str, path: str) -> tuple[int, int, int]:
    """EIGENVAL 第 6 行 = NELECT NKPT NBANDS (3 个整数, 位置索引)。

    用位置索引而不是负索引: VASP 若在头部加列, ``hdr[-2]/hdr[-1]`` 会**静默**
    取错列, 而位置索引会立刻炸掉。
    """
    fields = line.split()
    if len(fields) < 3:
        raise ValueError(f"EIGENVAL 第 6 行无法解析: {line.strip()!r} ({path})")
    try:
        nelect = int(fields[0])   # NELECT: 电子数(仅作记录, 不参与解析)
        nkpt = int(fields[1])      # NKPT: 文件里实际的 k 点块数
        nbands = int(fields[2])    # NBANDS: 带数
    except ValueError as exc:
        raise ValueError(
            f"EIGENVAL 第 6 行无法解析: {line.strip()!r} ({path})") from exc
    return nelect, nkpt, nbands


def _count_kpoint_rows(lines: list[str]) -> int:
    """数出 EIGENVAL 里实际的 k 点块数(用于校验头部声明的 NKPT)。

    k 点行的结构特征: 恰好 4 个浮点字段 (kx ky kz weight); 带行是 3 个
    (带号 能量 占据), 头部行不是 4 个浮点。只从第 7 行(数据区)起数 —— 第 1 行的
    ``NIONS NIONS ISPIN ISYM`` 恰好也是 4 个整数。charge 段之后没有 k 点块。
    """
    n = 0
    for line in lines[6:]:
        s = line.strip()
        if s.startswith("charge"):
            break
        fields = s.split()
        if len(fields) != 4:
            continue
        try:
            [float(x) for x in fields]
        except ValueError:
            continue
        n += 1
    return n


def read_eigenval(path: str) -> BandData:
    """读 EIGENVAL -> 带数、能量(eV)、占据数与 k 点(分数坐标)。

    仅支持 ISPIN=1(自旋非极化); ISPIN=2 的文件显式报错。
    """
    lines = _read_lines(path)
    if len(lines) < 7:
        raise ValueError(f"EIGENVAL 行数不足, 无法解析: {path}")
    ispin = _eigenval_ispin(lines, path)
    if ispin != 1:
        raise ValueError(
            f"EIGENVAL 的 ISPIN={ispin}(自旋极化) —— 本工具的展开按自旋非极化设计, "
            f"仅支持 ISPIN=1; 自旋极化文件的带行是 5 个字段(E↑ occ↑ E↓ occ↓), "
            f"会被静默丢掉自旋下半: {path}")
    _nelect, nkpts, nbands = _eigenval_header(lines[5], path)
    n_kpt_rows = _count_kpoint_rows(lines)
    if n_kpt_rows != nkpts:
        raise ValueError(
            f"EIGENVAL 头部声明 NKPT={nkpts}, 但文件里实际有 {n_kpt_rows} 个 "
            f"k 点块(头部布局与文件不符): {path}")
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
                band_no = int(v[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"EIGENVAL 带数据行无法解析: {lines[idx].strip()!r} "
                    f"({path})") from exc
            if band_no != ib + 1:
                raise ValueError(
                    f"EIGENVAL 带号错位: k={ik + 1} 处读到带 {band_no}, "
                    f"应为 {ib + 1}: {path}")
            try:
                energies[ik, ib] = float(v[1])
                occupations[ik, ib] = float(v[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"EIGENVAL 带数据行无法解析: {lines[idx].strip()!r} "
                    f"({path})") from exc
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

    首个 'ion' 行必是模方块表头(复投影块的表头出现在每个 band 的模方块之后),
    因此它**必须**含 'tot' 列; 否则说明文件被截断/格式不符, 立即报错, 而不是
    静默按错误的列数算轨道数。
    """
    for line in lines[2:]:
        if line.lstrip().startswith("ion"):
            fields = line.split()
            if "tot" not in fields:
                raise ValueError(
                    f"PROCAR: 首个 ion 表头没有 'tot' 列(不是模方块表头) -> "
                    f"文件不完整或格式不符: {line.strip()!r}")
            return len(fields) - 2
    raise ValueError("PROCAR: 未找到 ion 表头")


def _procar_kpoint(line: str) -> list[float]:
    """k 点行: ' k-point 1 : x y z weight = ...' -> 坐标在第 3~5 个 token。"""
    toks = line.split()
    try:
        return [float(toks[3]), float(toks[4]), float(toks[5])]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"PROCAR k 点行无法解析: {line.strip()!r}") from exc


def _procar_band_number(line: str, path: str) -> int:
    """'band 12 # energy ... # occ. ...' -> 带号 12。"""
    toks = line.split()
    try:
        return int(toks[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"PROCAR 带行无法解析: {line.strip()!r} ({path})") from exc


def _has_complex_rows(lines: list[str], n_cplx: int) -> bool:
    """整个文件里是否存在复投影行(2*norb+2 个字段的数据行)。

    用来区分两种"没有复投影数据"的情况:
    - 全文件都没有 -> LORBIT=11(只写模方块);
    - 只有部分 band 没有 -> 文件被截断。
    """
    for line in lines:
        s = line.strip()
        if not s or not s[0].isdigit():
            continue
        if len(s.split()) == n_cplx:
            return True
    return False


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
    file_has_complex = _has_complex_rows(lines, n_cplx)
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
            band_no = _procar_band_number(lines[idx], path)
            if band_no != ib + 1:
                raise ValueError(
                    f"PROCAR 带号错位: k={ik + 1} 处读到 band {band_no}, "
                    f"应为 {ib + 1}: {path}")
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
            n_mod_rows = 0
            n_cplx_rows = 0
            for v in rows:
                if len(v) not in (n_mod, n_cplx):
                    raise ValueError(
                        f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 数据行有 "
                        f"{len(v)} 个字段, 既不是模方块({n_mod})也不是复投影"
                        f"({n_cplx}): {' '.join(v)!r} ({path})")
                try:
                    ia = int(v[0]) - 1
                except ValueError as exc:
                    raise ValueError(
                        f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 离子序号无法"
                        f"解析: {' '.join(v)!r} ({path})") from exc
                if not 0 <= ia < nions:
                    raise ValueError(
                        f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 离子序号 "
                        f"{ia + 1} 越界(共 {nions} 个): ({path})")
                try:
                    vals = [float(x) for x in v[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 数值无法解析: "
                        f"{' '.join(v)!r} ({path})") from exc
                if len(v) == n_mod:
                    n_mod_rows += 1
                    weights[ik, ib, ia] = vals[:norb]
                else:
                    n_cplx_rows += 1
                    has_phase = True
                    for m in range(norb):
                        coeffs[ik, ib, ia, m] = (vals[2 * m] + 1j * vals[2 * m + 1])
            if n_mod_rows != nions:
                raise ValueError(
                    f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 模方块只有 "
                    f"{n_mod_rows} 行, 应为 {nions} 行(文件被截断?): {path}")
            if n_cplx_rows != nions:
                if not file_has_complex:
                    raise ValueError(
                        "PROCAR 中没有复投影块 —— 需要 LORBIT=12; LORBIT=11 "
                        f"不写相位(文件被截断时也会缺复投影块): {path}")
                raise ValueError(
                    f"PROCAR 在 k={ik + 1} 的第 {ib + 1} 条带: 复投影只有 "
                    f"{n_cplx_rows} 行, 应为 {nions} 行(文件被截断?): {path}")
    if not has_phase:
        raise ValueError(
            "PROCAR 中没有复投影块 —— 需要 LORBIT=12; LORBIT=11 不写相位")
    return ProcarData(kpoints, weights, coeffs, has_phase)
