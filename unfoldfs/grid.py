"""均匀原胞网格、对称性轨道分解、超胞 k 点去重、KPOINTS 写出。

约定(与 geometry.py 一致):
  * 网格点用**原胞**分数坐标(行向量写法), 超胞坐标由 Transform.super_from_prim 给出;
  * 对称操作作用于原胞分数坐标 k -> k @ R^T + t (spglib 是列向量约定);
  * 轨道大小按**去重后的格点数**计 —— 不同操作可能落到同一格点;
  * spglib 的平移代表元带 ~3e-8 数值残差(硬化只吸附接近 0/1 的分量),
    故一切"点是否落在网格上"的判断都用**容差比较**, 不使用
    int(mod(t*n, 1)*n) 这类会静默错一格的截断写法。

折叠(超胞 k 点):
  满网格点 k' 折到超胞 k:  K = k'·M^T (mod 1);  同一超胞 k 上有 det(M) 个原胞点。
  KPOINTS 里**每个不同的超胞 k 只写一行**, 权重 = 折到该 k 的满网格点数,
  权重之和 == 满网格点数(设计规格 §5.2 的自检)。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (SymmetryOps, Transform, candidate_g_vectors,
                       matrix_denominator_lcm, reciprocal)

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


def per_dimension_multiples(m_eff: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """对角 M 的**逐维**约束: 第 i 维要求 M_ii | P_i; 非对角 M 返回 None。

    对角 M 时折叠是**逐分量**的(K = k'·M^T ⇒ K_i = k'_i·M_ii), 于是第 i 维
    只要 P_i 是 |M_ii| 的倍数, 该维的 P_i 个网格点就**整除**成 P_i/|M_ii|
    个不同的超胞 k ⇒ 逐维约束是**充要**条件(且保证 n_points = n_k·det(M))。

    ⚠ **非对角 M 下逐维条件不成立**: 候选折叠矢量 g 跨维耦合
    (``candidate_g_for_k`` 的候选集随 K 平移), 只有全局 LCM
    (= comm(M^-1)) 才保证折叠一致 ⇒ 那时返回 None, 调用方用全局 LCM。
    """
    m = np.asarray(m_eff, dtype=float)
    # rtol=0: "是不是对角"与"对角元是不是整数"都是**绝对**判据 ——
    # 默认的 rtol=1e-5 会让大对角元(比如 1000)上的 1e-3 级非对角元被放过。
    if not np.allclose(m, np.diag(np.diag(m)), atol=1e-8, rtol=0.0):
        return None
    diag = np.diag(m)
    if not np.allclose(diag, np.rint(diag), atol=1e-8, rtol=0.0):
        return None
    d = np.abs(np.rint(diag)).astype(int)
    if np.any(d < 1):
        return None
    return (int(d[0]), int(d[1]), int(d[2]))


def choose_grid_points(lattice: np.ndarray, m_eff: np.ndarray,
                       resolution: float, grid_type: str = "gamma") -> GridSpec:
    """按分辨率选网格点数, 并保证网格与超胞变换自洽。

    lattice  : 原胞晶格(行=实矢量)
    m_eff    : 变换矩阵(行约定, 用 resolve_transform 探测过的那个)
    resolution: 倒空间步长上限, 单位与 |B| 相同(含 2π), 即 2π/Å

    **约束分两种**:

    * **对角 M**: 逐维约束 M_ii | P_i(见 ``per_dimension_multiples``)。
      第 i 维只有被放大时才需要它的倍数 —— 没被放大的那一维(如 monolayer
      超胞的 c 轴 M_33=1)**不受约束**, 否则会白烧机时(全局 LCM 会把
      P_3 强制成 3 的倍数, 真实 monolayer 只需要 1)。
    * **非对角 M**: 全局 LCM = comm(M^-1)(逐维条件在那里不成立)。
    """
    b = reciprocal(lattice)
    per_dim = per_dimension_multiples(m_eff)
    comm = matrix_denominator_lcm(m_eff) if per_dim is None else None
    p = []
    for i in range(3):
        n = max(int(np.ceil(np.linalg.norm(b[i]) / resolution)), 1)
        mult = per_dim[i] if per_dim is not None else comm
        while n % mult != 0:
            n += 1
        p.append(n)
    return GridSpec(tuple(p), grid_type, (0.0, 0.0, 0.0))


def grid_resolution(b_prim: np.ndarray, spec: GridSpec) -> float:
    """网格的实际倒空间步长(各方向取最大)。"""
    return float(max(np.linalg.norm(b_prim[i]) / spec.p[i] for i in range(3)))


def _infer_tol(grid: np.ndarray) -> float:
    """由网格自身推出一个安全的匹配容差。

    _TOL 是固定绝对容差; 当格距(1/P)比它还小时, 匹配可能命中错误邻居,
    故上限取"最小格距的 10%"(P ≲ 1e5 时仍是 1e-6)。
    """
    spacings = []
    for axis in range(3):
        vals = np.unique(np.mod(grid[:, axis], 1.0))
        if vals.size > 1:
            gaps = np.diff(np.concatenate([vals, vals[:1] + 1.0]))
            spacings.append(float(gaps.min()))
    if not spacings:
        return _TOL
    return float(min(_TOL, 0.1 * min(spacings)))


class _PeriodicIndex:
    """周期边界下的网格点索引: 把任意点匹配到容差内的网格点。

    用 27 份周期镜像建 KD-tree, 这样 k=0.99999999 与 k=0.0 的距离是 1e-8
    而不是 1 —— 不存在"桶边界"式的静默错配。
    """

    def __init__(self, points: np.ndarray, tol: Optional[float] = None):
        self.points = np.mod(np.asarray(points, dtype=float), 1.0)
        self.tol = float(tol) if tol is not None else _infer_tol(self.points)
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


def filter_ops_by_grid_detailed(ops: SymmetryOps, grid: np.ndarray,
                                 tol: Optional[float] = None):
    """只保留把网格映射到自身的操作 —— 并报告被剔除的操作与原因。

    返回 (保留的 ops, [(旋转, 平移, 原因), ...])。
    """
    index = _PeriodicIndex(grid, tol=tol)
    keep_r, keep_t = [], []
    dropped = []
    for i in range(ops.rotations.shape[0]):
        if np.all(index.find(ops.apply(grid, i)) >= 0):
            keep_r.append(ops.rotations[i])
            keep_t.append(ops.translations[i])
        else:
            dropped.append((ops.rotations[i], ops.translations[i],
                            "该操作不把当前网格映射到自身(平移没落在网格上)"))
    if not keep_r:
        raise ValueError("没有操作保持该网格不变, 请调整网格点数/类型")
    return SymmetryOps(np.array(keep_r), np.array(keep_t)), dropped


def filter_ops_by_grid(ops: SymmetryOps, grid: np.ndarray,
                       tol: Optional[float] = None) -> SymmetryOps:
    """只保留把网格映射到自身的操作(平移也必须落在网格上)。"""
    kept, _dropped = filter_ops_by_grid_detailed(ops, grid, tol=tol)
    return kept


def decompose_grid(grid: np.ndarray, ops: SymmetryOps,
                   tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """把网格点按对称轨道分组。

    返回 (ir_list, multiplicity, full_to_ir, rot_of_full):
      ir_list       -> 代表点在 grid 中的行号
      multiplicity  -> 每个代表点的轨道大小(**去重后的格点数**, 不是操作数)
      full_to_ir[i] -> i 所属代表点在"代表点列表"中的序号
      rot_of_full[i]-> 把代表点映到 i 的操作下标(-1 = i 本身就是代表点)
    """
    n = grid.shape[0]
    index = _PeriodicIndex(grid, tol=tol)
    n_ops = ops.rotations.shape[0]
    # 批量匹配: 每个操作一次查完全部点(而不是逐点单查)——
    # 旧实现只对代表点逐点查(n_ir x n_ops 次); 30^3 网格实测 0.93s -> 0.45s。
    match = np.empty((n, n_ops), dtype=int)
    for o in range(n_ops):
        match[:, o] = index.find(ops.apply(grid, o))
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
            j = int(match[i, o])
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


def _k_keys(k_arr: np.ndarray, tol: float) -> List[Tuple[int, int, int]]:
    """分数坐标 -> 周期安全的整数键(供 k 点去重)。

    先把"距 1 不到 tol"的分量归零(周期边界的两侧其实是同一个点),
    再按 tol 取整; tol 不超过最小格距的 10%, 故不同的 k 不会被合并。
    """
    kk = np.mod(np.asarray(k_arr, dtype=float), 1.0)
    kk = np.where(kk > 1.0 - tol, 0.0, kk)
    scaled = np.round(kk / tol).astype(np.int64)
    return [tuple(int(x) for x in scaled[i]) for i in range(scaled.shape[0])]


def _dedup_k(points: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """按整数键去重 -> (去重后的点(首次出现顺序), 每个输入点对应的行号)。"""
    keys = _k_keys(points, tol)
    lookup: dict = {}
    first: List[int] = []
    idx = np.empty(len(keys), dtype=int)
    for i, key in enumerate(keys):
        j = lookup.get(key)
        if j is None:
            j = len(first)
            lookup[key] = j
            first.append(i)
        idx[i] = j
    return points[first], idx


def supercell_kpoints(grid: np.ndarray, transform: Transform,
                      tol: Optional[float] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把满网格点折叠到超胞 k 点并去重。

    折叠: K = k'·M^T (mod 1);  det(M) 个原胞点折到同一个超胞 k。

    返回:
      k_super   (n_k,3) 不同的超胞 k(分数坐标), 行序 = 满网格首次出现顺序
      full_to_k (n_points,) 每个满网格点对应的 k_super 行号
      weight    (n_k,) 折到该 k 的满网格点数(= KPOINTS 里写出的权重)
    """
    if tol is None:
        tol = _infer_tol(grid)
    k_all = np.mod(np.asarray(grid, dtype=float) @ transform.m_eff.T, 1.0)
    k_super, full_to_k = _dedup_k(k_all, tol)
    weight = np.bincount(full_to_k, minlength=k_super.shape[0]).astype(int)
    return k_super, full_to_k, weight


@dataclass(frozen=True)
class GridMap:
    spec: GridSpec
    ir_k_prim: np.ndarray       # (n_ir,3) 原胞 IR 点(分数坐标)
    multiplicity: np.ndarray    # (n_ir,) 每个 IR 点的轨道重数
    k_super: np.ndarray         # (n_k,3) **去重后**要计算的超胞 k(分数坐标)
    ir_to_k: np.ndarray         # (n_ir,) IR 点 -> k_super 索引
    k_to_ir: np.ndarray         # (n_k,)  k_super 索引 -> 首个映射到它的 IR 点
    full_to_ir: np.ndarray      # (n_points,) 满网格点 -> IR 索引
    full_to_k: np.ndarray       # (n_points,) 满网格点 -> k_super 索引
    rot_of_full: np.ndarray     # (n_points,) 满网格点 -> 操作索引(-1 = 不动)
    ops: SymmetryOps
    transform: Transform
    b_prim: np.ndarray
    b_super: np.ndarray
    gset: np.ndarray            # (detM,3) 候选折叠矢量
    # 被剔除的对称操作 [(旋转, 平移, 原因), ...] —— 供 `info` 打印
    dropped: Tuple[Tuple[np.ndarray, np.ndarray, str], ...] = ()

    @property
    def n_ir(self) -> int:
        return int(self.ir_k_prim.shape[0])

    @property
    def n_k(self) -> int:
        return int(self.k_super.shape[0])

    @property
    def k_weight(self) -> np.ndarray:
        """每个超胞 k 折到的满网格点数(= KPOINTS 里写出的权重)。"""
        return np.bincount(self.full_to_k, minlength=self.k_super.shape[0])


def build_grid_map(spec: GridSpec, grid: np.ndarray, ops: SymmetryOps,
                   transform: Transform, b_prim: np.ndarray, b_super: np.ndarray,
                   tol: Optional[float] = None,
                   dropped: Tuple[Tuple[np.ndarray, np.ndarray, str], ...] = ()
                   ) -> GridMap:
    """组装 GridMap: 轨道分解 + 超胞 k 去重 + 索引映射 + 候选折叠矢量。

    grid  : 原胞分数坐标的满网格点(primitive_grid_points 的输出)
    ops   : 已按网格过滤过的对称操作(filter_ops_by_grid 的输出)
    """
    if tol is None:
        tol = _infer_tol(grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops, tol=tol)
    ir_k_prim = grid[ir_idx]
    k_super, full_to_k, _ = supercell_kpoints(grid, transform, tol=tol)
    # IR 点的超胞像 -> k_super 行号; 反向: k_super 行 -> 首个映到它的 IR 点
    key_to_k = {key: i for i, key in enumerate(_k_keys(k_super, tol))}
    ir_k_super = np.mod(np.asarray(ir_k_prim, dtype=float) @ transform.m_eff.T,
                        1.0)
    ir_to_k = np.array([key_to_k[key] for key in _k_keys(ir_k_super, tol)],
                       dtype=int)
    k_to_ir = np.full(k_super.shape[0], -1, dtype=int)
    for a, k in enumerate(ir_to_k):
        if k_to_ir[k] < 0:
            k_to_ir[k] = a
    return GridMap(
        spec=spec,
        ir_k_prim=ir_k_prim,
        multiplicity=mult,
        k_super=k_super,
        ir_to_k=ir_to_k,
        k_to_ir=k_to_ir,
        full_to_ir=full_to_ir,
        full_to_k=full_to_k,
        rot_of_full=rot,
        ops=ops,
        transform=transform,
        b_prim=b_prim,
        b_super=b_super,
        gset=candidate_g_vectors(transform.m_eff),
        dropped=tuple(dropped),
    )


def write_kpoints(path: str, grid_map: GridMap, *, explicit: bool = False) -> None:
    """VASPKIT/标准 VASP「特殊点 + 重数」格式。

    写出的行 = **去重后的超胞 k 点**(每个不同的超胞 k 只算一次),
    权重 = 折到该 k 的满网格点数(权重之和 == 满网格点数)。

    实测: VASP 把 "Reciprocal lattice" 之后的行按 k 点读(不是倒格矢)。

    ``explicit=True``: 首行写成**纯注释**(以 'u' 开头), 不落在 VASP 的任何
    分支关键字('G'/'K'/'S'/'C'/'R')上 —— 保证 VASP 走"显式 k 点列表"分支,
    而不是自己按首行重新生成网格(k 点的个数与顺序都会变)。首行格式的
    唯一事实来源就是这里(不要在调用方再改一遍)。
    """
    tag = "G" if grid_map.spec.grid_type == "gamma" else "M"
    p = grid_map.spec.p
    res = grid_resolution(grid_map.b_prim, grid_map.spec)
    k_super = grid_map.k_super
    weights = grid_map.k_weight
    if explicit:
        header = (f"unfoldfs: explicit k-point list (do not edit) -- "
                  f"{tag} grid {p[0]}x{p[1]}x{p[2]}")
    else:
        header = (f"{tag}    {p[0]}    {p[1]}    {p[2]}    {res:.3f}"
                  f"     # Parameters to Generate KPOINTS (Don't Edit This Line)")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        fh.write(f"   {k_super.shape[0]}\n")
        fh.write("Reciprocal lattice\n")
        for k, w in zip(k_super, weights):
            fh.write(f"    {k[0]:.12f}    {k[1]:.12f}    {k[2]:.12f}    "
                     f"{float(w):.12f}\n")


def save_grid_map(path: str, grid_map: GridMap) -> None:
    np.savez(
        path,
        spec_p=np.array(grid_map.spec.p),
        spec_type=grid_map.spec.grid_type,
        spec_shift=np.array(grid_map.spec.shift),
        ir_k_prim=grid_map.ir_k_prim,
        multiplicity=grid_map.multiplicity,
        k_super=grid_map.k_super,
        ir_to_k=grid_map.ir_to_k,
        k_to_ir=grid_map.k_to_ir,
        full_to_ir=grid_map.full_to_ir,
        full_to_k=grid_map.full_to_k,
        rot_of_full=grid_map.rot_of_full,
        rotations=grid_map.ops.rotations,
        translations=grid_map.ops.translations,
        m_eff=grid_map.transform.m_eff,
        b_prim=grid_map.b_prim,
        b_super=grid_map.b_super,
        gset=grid_map.gset,
        dropped_rotations=np.array([d[0] for d in grid_map.dropped],
                                   dtype=int).reshape(len(grid_map.dropped), 3, 3),
        dropped_translations=np.array([d[1] for d in grid_map.dropped],
                                      dtype=float).reshape(len(grid_map.dropped), 3),
        dropped_reasons=np.array([d[2] for d in grid_map.dropped]),
    )


def load_grid_map(path: str) -> GridMap:
    d = np.load(path, allow_pickle=False)
    ops = SymmetryOps(d["rotations"], d["translations"])
    m_eff = d["m_eff"]
    tr = Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    spec = GridSpec(tuple(int(x) for x in d["spec_p"]), str(d["spec_type"]),
                     tuple(float(x) for x in d["spec_shift"]))
    dropped = []
    if "dropped_rotations" in d.files:
        for r, t, why in zip(d["dropped_rotations"], d["dropped_translations"],
                             d["dropped_reasons"]):
            dropped.append((r, t, str(why)))
    return GridMap(
        spec=spec,
        ir_k_prim=d["ir_k_prim"],
        multiplicity=d["multiplicity"],
        k_super=d["k_super"],
        ir_to_k=d["ir_to_k"],
        k_to_ir=d["k_to_ir"],
        full_to_ir=d["full_to_ir"],
        full_to_k=d["full_to_k"],
        rot_of_full=d["rot_of_full"],
        ops=ops,
        transform=tr,
        b_prim=d["b_prim"],
        b_super=d["b_super"],
        gset=d["gset"],
        dropped=tuple(dropped),
    )
