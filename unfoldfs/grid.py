"""均匀原胞网格、对称性轨道分解、KPOINTS 写出。

约定(与 geometry.py 一致):
  * 网格点用**原胞**分数坐标(行向量写法), 超胞坐标由 Transform.super_from_prim 给出;
  * 对称操作作用于原胞分数坐标 k -> k @ R^T + t (spglib 是列向量约定);
  * 轨道大小按**去重后的格点数**计 —— 不同操作可能落到同一格点;
  * spglib 的平移代表元带 ~3e-8 数值残差(硬化只吸附接近 0/1 的分量),
    故一切"点是否落在网格上"的判断都用**容差比较**, 不使用
    int(mod(t*n, 1)*n) 这类会静默错一格的截断写法。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (SymmetryOps, Transform, matrix_denominator_lcm,
                       reciprocal)

# 分数坐标下的匹配容差: 远大于 spglib 平移残差(~3e-8), 又远小于网格间距
_TOL = 1e-6


@dataclass(frozen=True)
class GridSpec:
    p: Tuple[int, int, int]
    grid_type: str
    shift: Tuple[float, float, float]

    @property
    def n_points(self) -> int:
        return int(self.p[0] * self.p[1] * self.p[2])


def primitive_grid_points(spec: GridSpec) -> np.ndarray:
    """原胞 BZ 内的均匀网格点(分数坐标), 行=点序。

    "gamma": 点在 i/n; "mp": 点在 (i+1/2)/n。
    """
    axes = []
    for i, n in enumerate(spec.p):
        if spec.grid_type == "mp":
            frac = (np.arange(n) + 0.5) / n + spec.shift[i]
        else:
            frac = np.arange(n) / n + spec.shift[i]
        axes.append(np.mod(frac, 1.0))
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return mesh.reshape(-1, 3)


def choose_grid_points(lattice: np.ndarray, m_eff: np.ndarray,
                       resolution: float, grid_type: str = "gamma") -> GridSpec:
    """按分辨率选网格点数, 并强制 P 为 comm(M^-1) 的倍数。

    lattice  : 原胞晶格(行=实矢量)
    m_eff    : 变换矩阵(行约定)
    resolution: 倒空间步长上限, 单位与 |B| 相同(含 2π), 即 2π/Å
    """
    b = reciprocal(lattice)
    comm = matrix_denominator_lcm(m_eff)
    p = []
    for i in range(3):
        n = max(int(np.ceil(np.linalg.norm(b[i]) / resolution)), 1)
        while n % comm != 0:
            n += 1
        p.append(n)
    return GridSpec(tuple(p), grid_type, (0.0, 0.0, 0.0))


def grid_resolution(b_prim: np.ndarray, spec: GridSpec) -> float:
    """网格的实际倒空间步长(各方向取最大)。"""
    return float(max(np.linalg.norm(b_prim[i]) / spec.p[i] for i in range(3)))


class _PeriodicIndex:
    """周期边界下的网格点索引: 把任意点匹配到容差内的网格点。

    用 27 份周期镜像建 KD-tree, 这样 k=0.99999999 与 k=0.0 的距离是 1e-8
    而不是 1 —— 不存在"桶边界"式的静默错配。
    """

    def __init__(self, points: np.ndarray, tol: float = _TOL):
        self.points = np.mod(np.asarray(points, dtype=float), 1.0)
        self.tol = float(tol)
        shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)),
                          dtype=float)
        ext = (self.points[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        self._tree = cKDTree(ext)
        self._owner = np.tile(np.arange(self.points.shape[0]), shifts.shape[0])

    def find(self, queries: np.ndarray) -> np.ndarray:
        """每个查询点对应的网格下标; 容差外返回 -1。"""
        q = np.mod(np.atleast_2d(np.asarray(queries, dtype=float)), 1.0)
        dist, idx = self._tree.query(q, k=1)
        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        out = np.full(dist.shape, -1, dtype=int)
        hit = dist <= self.tol
        out[hit] = self._owner[idx[hit]]
        return out


def filter_ops_by_grid(ops: SymmetryOps, grid: np.ndarray) -> SymmetryOps:
    """只保留把网格映射到自身的操作(平移也必须落在网格上)。"""
    index = _PeriodicIndex(grid)
    keep_r, keep_t = [], []
    for i in range(ops.rotations.shape[0]):
        if np.all(index.find(ops.apply(grid, i)) >= 0):
            keep_r.append(ops.rotations[i])
            keep_t.append(ops.translations[i])
    if not keep_r:
        raise ValueError("没有操作保持该网格不变, 请调整网格点数/类型")
    return SymmetryOps(np.array(keep_r), np.array(keep_t))


def decompose_grid(grid: np.ndarray, ops: SymmetryOps) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
    """把网格点按对称轨道分组。

    返回 (ir_list, multiplicity, full_to_ir, rot_of_full):
      ir_list       -> 代表点在 grid 中的行号
      multiplicity  -> 每个代表点的轨道大小(**去重后的格点数**, 不是操作数)
      full_to_ir[i] -> i 所属代表点在"代表点列表"中的序号
      rot_of_full[i]-> 把代表点映到 i 的操作下标(-1 = i 本身就是代表点)
    """
    n = grid.shape[0]
    index = _PeriodicIndex(grid)
    n_ops = ops.rotations.shape[0]
    ir_list = []
    mult = []
    rot_of_full = np.full(n, -1, dtype=int)
    rep_of = np.full(n, -1, dtype=int)      # 每点所属的代表点行号
    for i in range(n):
        if rep_of[i] >= 0:
            continue
        orbit = []
        seen = set()
        for o in range(n_ops):
            j = int(index.find(ops.apply(grid[i], o))[0])
            if j >= 0 and j not in seen:      # 不同操作可能落到同一格点 -> 必须去重
                seen.add(j)
                orbit.append((j, o))
        if i not in seen:
            orbit.append((i, -1))
        for j, o in orbit:
            rep_of[j] = i
            rot_of_full[j] = o
        rot_of_full[i] = -1                  # 代表点自身不需要旋转
        ir_list.append(i)
        mult.append(len(orbit))
    if np.any(rep_of < 0):
        raise ValueError("轨道分解不完整: 有网格点未被任何轨道覆盖")
    if sum(mult) != n:
        raise ValueError(f"轨道分解不封闭: sum(mult)={sum(mult)} != n_points={n}")
    ir_index = {rep: k for k, rep in enumerate(ir_list)}
    full_to_ir = np.array([ir_index[int(rep_of[i])] for i in range(n)], dtype=int)
    return (np.array(ir_list, dtype=int), np.array(mult, dtype=int),
            full_to_ir, rot_of_full)


@dataclass(frozen=True)
class GridMap:
    spec: GridSpec
    ir_k_prim: np.ndarray
    ir_k_super: np.ndarray
    multiplicity: np.ndarray
    full_to_ir: np.ndarray
    rot_of_full: np.ndarray
    ops: SymmetryOps
    transform: Transform
    b_prim: np.ndarray
    b_super: np.ndarray

    @property
    def n_ir(self) -> int:
        return int(self.ir_k_prim.shape[0])


def write_kpoints(path: str, grid_map: GridMap) -> None:
    """VASPKIT/标准 VASP「特殊点 + 重数」格式。

    实测: VASP 把 "Reciprocal lattice" 之后的行按 k 点读(不是倒格矢)。
    """
    tag = "G" if grid_map.spec.grid_type == "gamma" else "M"
    p = grid_map.spec.p
    res = grid_resolution(grid_map.b_prim, grid_map.spec)
    with open(path, "w") as fh:
        fh.write(f"{tag}    {p[0]}    {p[1]}    {p[2]}    {res:.3f}"
                 f"     # Parameters to Generate KPOINTS (Don't Edit This Line)\n")
        fh.write(f"   {grid_map.n_ir}\n")
        fh.write("Reciprocal lattice\n")
        for k, w in zip(grid_map.ir_k_super, grid_map.multiplicity):
            fh.write(f"    {k[0]:.12f}    {k[1]:.12f}    {k[2]:.12f}    "
                     f"{float(w):.12f}\n")


def save_grid_map(path: str, grid_map: GridMap) -> None:
    np.savez(
        path,
        spec_p=np.array(grid_map.spec.p),
        spec_type=grid_map.spec.grid_type,
        spec_shift=np.array(grid_map.spec.shift),
        ir_k_prim=grid_map.ir_k_prim,
        ir_k_super=grid_map.ir_k_super,
        multiplicity=grid_map.multiplicity,
        full_to_ir=grid_map.full_to_ir,
        rot_of_full=grid_map.rot_of_full,
        rotations=grid_map.ops.rotations,
        translations=grid_map.ops.translations,
        m_eff=grid_map.transform.m_eff,
        b_prim=grid_map.b_prim,
        b_super=grid_map.b_super,
    )


def load_grid_map(path: str) -> GridMap:
    d = np.load(path, allow_pickle=False)
    ops = SymmetryOps(d["rotations"], d["translations"])
    m_eff = d["m_eff"]
    tr = Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    spec = GridSpec(tuple(int(x) for x in d["spec_p"]), str(d["spec_type"]),
                     tuple(float(x) for x in d["spec_shift"]))
    return GridMap(
        spec=spec,
        ir_k_prim=d["ir_k_prim"],
        ir_k_super=d["ir_k_super"],
        multiplicity=d["multiplicity"],
        full_to_ir=d["full_to_ir"],
        rot_of_full=d["rot_of_full"],
        ops=ops,
        transform=tr,
        b_prim=d["b_prim"],
        b_super=d["b_super"],
    )
