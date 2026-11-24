"""WAVECAR 的平面波数据 (VASP 6, rtag=45200/45210)。

为什么需要它: 反折叠的**严格谱权重**是
    w(k') = |⟨k'|Ψ_K⟩|² = Σ_g |C(g + k' - K)|²,
其中 c 是超胞态在平面波基 |K+G⟩ 上的系数 —— 这正是 Popescu--Zunger 的构造。
用 PROCAR 的分波系数去逼近这个重叠是可行的(本项目的原做法), 但那不是严格的谱权重:
分波基非正交、不完备, 权重只在单次计算内相对可比, 而且引入了一个额外的
"代表原子平均"步骤(见论文 §II.B)。

**文件布局**(实测 VASP 6.6.0, 全部按记录读, 每个记录 recl 字节):
    记录 0      recl, nspin, rtag                (float64)
    记录 1      nkpts, nbands, encut, cell(3×3) (float64)
    之后        每个 (spin, k) 占 (nbands+1) 个记录:
                  记录 +0 : nplw, kvec(3), [E, ?, occ] × nbands   (float64)
                  记录 +1..+nbands : 该带的 nplw 个复系数 (complex64 或 complex128)
    记录号      2 + (ispin-1)*nkpts*(nbands+1) + (ikpt-1)*(nbands+1) + iband

⚠ **VASP 不在 WAVECAR 里存 G 矢量** —— 必须自己重建: 由 ENCUT 定 FFT 网格,
  取 |G+k|²/2 < ENCUT 的球内格点, 按 VASP 的顺序(网格序, x 变化最快)。
  重建的格点数必须与文件里记录的 nplw 一致, 否则说明网格/半格点约定搞错了
  (代码里会显式报错)。

**单位约定(实测 VASP 6.6.0)**:
* 记录 1 里的**晶胞是 Å**(实测 1T-TaS₂ 3×1×1 的晶胞与 POSCAR 逐位相同:
  a1 = 10.0947 Å); 不同版本可能写成 bohr, 所以本模块**两种都试**, 用
  WAVECAR 自己记录的 nplw 判定 —— 判据来自文件本身, 不依赖版本。
* 记录里的 **k 是分数坐标(超胞的晶体坐标)**, 不是笛卡尔坐标(实测与
  EIGENVAL 的 k 逐位相同)。因此动能要写成

      E = HSQDTM·|(n + k_frac)·(2π·B)|²

  —— 整数网格索引 n 与分数 k 相加后再乘倒格矢, 就是 G_cart + k_cart
  (easyunfold 的写法也是这个)。

**FFT 网格的定法**: 平面波索引 n 满足 ``G·a_j = 2π n_j``, 故球内的 n 满足
``|n_j| ≤ k_max·|a_j|/(2π)``; VASP 取 ``ngrid_j = 2·ceil(k_max·|a_j|/2π)+1``
(实测逐点复现文件里的 nplw)。注意网格**大小**不影响平面波列表的**顺序**
(两者都是"非负 G 在前、负 G 在后"的字典序), 所以取最小网格即可。

G 矢量的重建算法与 WAVECAR 的读取约定参照 easyunfold (Zhu et al., JOSS 9, 5974
(2024), MIT 许可) 的实现, 并按本项目的需要改写: 我们不需要把系数写回 .wfc 文件,
只需要 G 矢量与系数, 逐 k 流式读。

物理常数 (VASP 的单位制): 1 bohr = 0.529177249 Å, 1 Ry = 13.605826 eV。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

__all__ = ["WavecarHeader", "WavecarKpoint", "read_wavecar_header",
           "iter_wavecar_kpoints", "gvectors_for_k", "minimum_fft_grid"]

_AUTOA = 0.529177249          # bohr -> Angstrom
_RYTOEV = 13.605826           # Ry -> eV
_PI = np.pi
_TPI = 2.0 * _PI
_HSQDTM = _RYTOEV * _AUTOA * _AUTOA     # = hbar^2 / 2m in eV·Å^2


@dataclass(frozen=True)
class WavecarHeader:
    """WAVECAR 的头部信息。"""
    recl: int
    nspin: int
    rtag: int
    nkpts: int
    nbands: int
    encut: float
    cell: np.ndarray        # 实空间晶格(行向量); 单位由 cell_scale 给出
    ngrid: np.ndarray       # FFT 网格三边
    precision: str          # 'complex64' | 'complex128'
    data_offset: int        # 第一个 (spin,k) 头记录的字节偏移(= 2*recl)
    cell_scale: float = 1.0
    """晶胞长度单位换算到 **Å** 的因子(晶胞 = cell · cell_scale Å)。

    实测 VASP 6.6 写的是 Å(1.0); 老版本写 bohr(0.529177)。由
    :func:`gvectors_for_k` 用 nplw 判定并回填。
    """


@dataclass(frozen=True)
class WavecarKpoint:
    """一个 k 点的平面波数据。"""
    index: int
    k_frac: np.ndarray      # **分数** k(超胞的晶体坐标) —— VASP 就这么写的
    nplw: int
    gvec: np.ndarray        # (nplw,3) 整数 G 索引(超胞倒空间基)
    coeffs: np.ndarray      # (nbands, nplw) 复系数


def _precision_of(rtag: int) -> str:
    if rtag == 45200:
        return "complex64"
    if rtag == 45210:
        return "complex128"
    if rtag in (53300, 53310):
        raise ValueError("VASP5 的 WAVECAR 格式未实现(rtag=%d)" % rtag)
    raise ValueError("未知的 WAVECAR 版本标记 rtag=%d" % rtag)


def read_wavecar_header(path: str, *, nkpts: Optional[int] = None,
                        nbands: Optional[int] = None,
                        nspin: Optional[int] = None) -> WavecarHeader:
    """读 WAVECAR 的两个头记录, 并(可选)与调用方的预期比对。

    ``nkpts/nbands/nspin`` 给了就检查 —— WAVECAR 与 EIGENVAL 必须出自同一次
    计算, 否则带的编号会错位(实测: WAVECAR 记 48 条带而 EIGENVAL 只有 40 条,
    这种不一致必须当场报错, 不能等到展开时才体现为"某几条带没权重")。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        d0 = np.frombuffer(fh.read(24), dtype=np.float64, count=3)
        recl, nspin_f, rtag = int(d0[0]), int(d0[1]), int(d0[2])
        precision = _precision_of(rtag)
        fh.seek(recl)
        d1 = np.frombuffer(fh.read(96), dtype=np.float64, count=12)
    nkpts_f, nbands_f, encut = int(d1[0]), int(d1[1]), float(d1[2])
    cell = d1[3:].reshape(3, 3)
    if nkpts is not None and nkpts_f != nkpts:
        raise ValueError("WAVECAR 的 k 点数 %d != 预期的 %d —— WAVECAR 与 "
                         "EIGENVAL 必须出自同一次计算" % (nkpts_f, nkpts))
    if nbands is not None and nbands_f != nbands:
        raise ValueError("WAVECAR 的带数 %d != 预期的 %d —— WAVECAR 与 EIGENVAL "
                         "必须出自同一次计算" % (nbands_f, nbands))
    if nspin is not None and nspin_f != nspin:
        raise ValueError("WAVECAR 的自旋分量数 %d != 预期的 %d —— 本实现只支持 "
                         "ISPIN=1" % (nspin_f, nspin))
    return WavecarHeader(recl=recl, nspin=nspin_f, rtag=rtag, nkpts=nkpts_f,
                         nbands=nbands_f, encut=encut, cell=cell,
                         ngrid=np.zeros(3, dtype=int), precision=precision,
                         data_offset=2 * recl, cell_scale=1.0)


def _rec_offset(hdr: WavecarHeader, ispin: int, ikpt: int, iband: int) -> int:
    """记录号 (每个记录 recl 字节)。ispin/ikpt/iband 都从 1 起。"""
    return (2 + (ispin - 1) * hdr.nkpts * (hdr.nbands + 1)
            + (ikpt - 1) * (hdr.nbands + 1) + iband)


def minimum_fft_grid(cell_ang: np.ndarray, encut: float) -> np.ndarray:
    """VASP 的最小 FFT 网格 (每轴 2·ceil(k_max·|a_i|/2π)+1)。

    推导: 平面波索引 n 满足 ``G·a_j = 2π n_j``; 球 |G| < k_max 内
    ``|n_j| ≤ k_max·|a_j|/(2π)`` ⇒ 每轴要 2·ceil(...)+1 个格点
    (VASP 用**奇数**网格, 实测与 "FFT dimensions" 一致)。
    """
    a = np.asarray(cell_ang, dtype=float)
    k_max = np.sqrt(2.0 * float(encut) / _HSQDTM)          # Å⁻¹
    abc = np.linalg.norm(a, axis=1)
    return np.array([2 * int(np.ceil(k_max * abc[i] / _TPI)) + 1
                     for i in range(3)], dtype=int)


def _plane_waves_on_grid(ngrid: np.ndarray, k_frac: np.ndarray,
                         cell_ang: np.ndarray, encut: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """FFT 网格序(x 变化最快)的平面波索引与动能。

    ``k_frac`` 是**分数** k(超胞晶体坐标): 动能写���
    ``HSQDTM·|(n + k_frac)·(2π·B)|²`` —— 整数网格索引与分数 k 相加后再乘倒格矢,
    就是 G_cart + k_cart。
    """
    ngrid = np.asarray(ngrid, dtype=int)
    fx, fy, fz = (np.arange(n, dtype=int) for n in ngrid)
    fx[ngrid[0] // 2 + 1:] -= ngrid[0]
    fy[ngrid[1] // 2 + 1:] -= ngrid[1]
    fz[ngrid[2] // 2 + 1:] -= ngrid[2]
    gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape((3, -1))
    kg = np.array([gx, gy, gz], dtype=float).T
    recip = np.linalg.inv(np.asarray(cell_ang, dtype=float)).T * _TPI   # 含 2π
    kinetic = _HSQDTM * np.linalg.norm((kg + k_frac) @ recip, axis=1) ** 2
    keep = kinetic < encut
    return kg[keep].astype(int), kinetic[keep]


def gvectors_for_k(hdr: WavecarHeader, k_frac: np.ndarray, nplw: int,
                   cell_scale: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """重建某个 k 点的 G 矢量(整数索引, 超胞倒空间基)。

    做法: 由 ENCUT 定 FFT 网格, 取 |G+k|²/2 < ENCUT 的格点(网格序, x 变化最快),
    与 WAVECAR 里记录的 nplw 比对 —— 不一致就报错(说明网格/半格点约定不对)。

    **晶胞单位**: 实测 VASP 6.6 写的是 Å, 老版本写 bohr。这里两种都试, 用
    nplw 判定(判据来自文件本身)。判定结果作为 ``cell_scale`` 返回, 供后续
    的 k 点复用, 免得每个 k 都重试一遍。

    返回
    ----
    (gvec, cell_scale)
      gvec       : (nplw, 3) 整数 G 索引
      cell_scale : 晶胞长度换算到 Å 的因子(1.0 = 已经是 Å)
    """
    scales = (cell_scale,) if cell_scale is not None else (1.0, _AUTOA)
    for scale in scales:
        cell_ang = np.asarray(hdr.cell, dtype=float) * scale
        ngrid = minimum_fft_grid(cell_ang, hdr.encut)
        g, _ = _plane_waves_on_grid(ngrid, np.asarray(k_frac, dtype=float),
                                    cell_ang, hdr.encut)
        if g.shape[0] == nplw:
            return g, scale
    if cell_scale is not None:
        raise ValueError(
            "重建的 G 矢量数与 WAVECAR 记录的 nplw %d 不一致(晶胞单位按 %g 试过)"
            % (nplw, cell_scale))
    raise ValueError(
        "重建的 G 矢量数与 WAVECAR 记录的 nplw %d 不一致: 晶胞单位按 Å 与 bohr "
        "两种都试过 —— 检查 ENCUT 与晶格是否与计算时相同" % nplw)


def iter_wavecar_kpoints(path: str, hdr: WavecarHeader
                          ) -> Iterator[WavecarKpoint]:
    """逐 k 点读出 (k, G 索引, 系数)。

    系数按 (band, nplw) 排布; 只取第一个自旋分量(本实现只支持 ISPIN=1)。
    """
    dt = np.complex64 if hdr.precision == "complex64" else np.complex128
    itemsize = 8 if hdr.precision == "complex64" else 16
    cell_scale: Optional[float] = None
    with open(path, "rb") as fh:
        for ikpt in range(1, hdr.nkpts + 1):
            off = _rec_offset(hdr, 1, ikpt, 0) * hdr.recl
            fh.seek(off)
            dump = np.frombuffer(fh.read(8 * (4 + 3 * hdr.nbands)),
                                 dtype=np.float64, count=4 + 3 * hdr.nbands)
            nplw = int(dump[0])
            k_frac = np.asarray(dump[1:4], dtype=float)
            gvec, cell_scale = gvectors_for_k(hdr, k_frac, nplw, cell_scale)
            coeffs = np.empty((hdr.nbands, nplw), dtype=np.complex128)
            for ib in range(hdr.nbands):
                fh.seek(_rec_offset(hdr, 1, ikpt, ib + 1) * hdr.recl)
                raw = fh.read(itemsize * nplw)
                if len(raw) < itemsize * nplw:
                    raise ValueError("WAVECAR 在第 %d 个 k 的第 %d 条带处被截断"
                                     % (ikpt, ib + 1))
                coeffs[ib] = np.frombuffer(raw, dtype=dt, count=nplw)
            yield WavecarKpoint(index=ikpt - 1, k_frac=k_frac, nplw=nplw,
                               gvec=gvec, coeffs=coeffs)
