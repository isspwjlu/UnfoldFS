# 反折叠费米面计算工具（unfoldfs）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 `unfoldfs` 命令行工具：对称性约化的 k 网格生成 → VASP 6.6.0 计算 → 投影法反折叠 → 对称扩展到完整均匀原胞网格 → 输出 VESTA/XCrySDen/FermiSurfer 费米面文件。

**Architecture:** 分层纯 Python 包（geometry / grid / vaspio / unfold / expand / fsout / cli），单向依赖。VASP 计算在程序之外（`gen` 与 `unfold` 两步通过磁盘上的 `grid_map.npz` 耦合）。开发与单元测试在本地 Windows（Python 3.11.7），真实 VASP 计算在 master 服务器（Python 3.9.13 + VASP 6.6.0）。

**Tech Stack:** Python ≥3.9、numpy、scipy、spglib、matplotlib、scikit-image、pytest

**Spec:** `docs/superpowers/specs/2026-11-17-unfolded-fermi-surface-design.md`

## Global Constraints

- 行向量约定（Ruling 8：**直接空间用 M，倒空间用 M⁻ᵀ**，两者不可混用）：
  - 实格矢：`A_sc = M_eff · A_prim`
  - 倒格矢：`B_sc = (M_eff⁻¹)ᵀ · B_prim`
  - **位置（分数）**：`r_prim = r_sc · M_eff`（无转置）
  - **k（分数）**：`k'_prim = (K + G) · (M_eff⁻¹)ᵀ`；`K = k'_prim · M_effᵀ (mod 1)`（**有转置**）
  - 数值核对（M=[[2,1],[0,2]], A_prim=I）：`A_sc=[[2,1],[0,2]]`、`B_sc=2π·[[0.5,0],[-0.25,0.5]]`、`a_sc1·b_sc1=2π` ✓；`k_sc=(0.5,0)` → `k_prim=(0.25,0)` ✓ 与 `k_sc·(M⁻¹)ᵀ` 一致，而 `k_sc·M⁻¹=(0.25,0.125)` ✗
  - **对对称 M（如 diag(3,3,1)）两者恒等** → 现有 fixture 测不出差别；非对称 M（莫尔超胞典型）必须用转置版
- **spglib 的旋转 R 是列向量约定**（实测：原子映射判据 12/12 vs 行向量 4/12）；超胞基底整数性判据用 `M_eff⁻ᵀ · R · M_effᵀ ∈ Z³`（已用原子映射真值交叉核对）
- **行/列约定必须运行时探测**：先试 `M @ A_prim`，不匹配则试 `Mᵀ @ A_prim`，都不匹配 → 报错（一般非对角 M 才能区分）
- VASP 计算必须 `LORBIT=12`（`LORBIT=11` 不写复投影块）；`ISMEAR=0`；`NBANDS ≥ det(M) × 原胞带数`
  - **建议 `NBANDS` 取 det(M) 的整数倍**（否则最高几条带属于不完整折叠像集；本 fixture 用 120 = 13×9+3，
    占据区 1–81 恰为 9 个完整像集，费米面不受影响，但做能带-原胞带逐条映射时顶部 3 条会错配）
- PROCAR(LORBIT=12) 每 band 两块：**11 字段**模方块（ion + 9 轨道 |p|² + tot）、**20 字段**复投影块（ion + 9×(Re,Im) + |c|²）
- 复投影块是**晶胞周期规范**（同一原胞各副本复值相同，不含 e^{iK·R}）
- 候选折叠矢量集 `Gset = {g ∈ Z³ : g·M_eff⁻¹ ∈ [0,1)³}`，基数恰为 `det(M)`
- 原胞网格点数 `P_i` 须为 `comm(M_eff⁻¹)` 的倍数（comm = M_eff⁻¹ 各元素分母的最小公倍数）
- 相位符号约定 `s1, s2 ∈ {+1,−1}` 由自检自动选定：取 4 种组合中「单 G 主导度」平均最大者；主导度 = max(w)/Σw，阈值 0.5，低于即报错
- DOSCAR 的 E_F 在**第 6 行第 4 个数**（EMAX EMIN NEDOS EFERMI）
- 测试环境：本地 `pytest`（Python 3.11.7，spglib 2.7.0）；master 端用 `/data/wjlu/program/anaconda3/bin/python3`（Python 3.9.13，spglib 2.0.2）
- **spglib >= 2.0 的 API**：`spglib.get_symmetry(cell, symprec=...)` 返回 **dict**，须取 `sym["rotations"]` / `sym["translations"]`（不是 tuple）
- master 工作目录：`/data/wjlu/work/claude/unfold_fs/`
- 提交信息格式：`<type>: <description>`，type ∈ feat/fix/refactor/docs/test/chore

---

## 文件结构

```
unfoldfs/
├── __init__.py        # 版本号，导出公共 API
├── geometry.py        # 结构/变换解析、倒格矢、候选 G、公共对称集 S
├── grid.py            # 均匀原胞网格、轨道分解、KPOINTS 写出、grid_map 存取
├── vaspio.py          # EIGENVAL / DOSCAR / PROCAR(LORBIT=12) 解析
├── unfold.py          # Popescu 投影展开、候选重组、约定自检
├── expand.py          # 对称扩展（候选索引置换）、A(k,E) 组装
├── fsout.py           # cube / grd / bxsf / FermiSurfer band / npz 写出
└── cli.py             # gen / unfold / info 子命令
tests/
├── conftest.py        # fixture 路径常量
├── data/              # VASP fixture（POSCAR/PRIMCELL/TRANSMAT/EIGENVAL/DOSCAR/PROCAR）
├── test_geometry.py
├── test_grid.py
├── test_vaspio.py
├── test_unfold.py
├── test_expand.py
├── test_fsout.py
└── test_cli.py
docs/
├── algorithms.md      # 展开公式推导、约定来源、为何取对称交集
└── formats.md         # 各输出格式的规范出处（现场逆向记录）
```

**依赖方向**（单向，无循环）：
`cli → {grid, unfold, expand, fsout}`；`grid → geometry`；`unfold → geometry, vaspio`；`expand → geometry, unfold`；`fsout → geometry`。

---

### Task 1: 项目骨架 + VASP 6.6.0 fixture

**Files:**
- Create: `unfoldfs/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/data/`（POSCAR, PRIMCELL.vasp, TRANSMAT.in, EIGENVAL, DOSCAR, PROCAR, INCAR）
- Create: `pytest.ini`
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: `tests/conftest.py` 中的路径常量，供后续所有任务使用：
  - `DATA_DIR: Path`（= `tests/data`）
  - `FIXTURE_POSCAR: Path`、`FIXTURE_PRIMCELL: Path`、`FIXTURE_TRANSMAT: Path`
  - `FIXTURE_EIGENVAL: Path`、`FIXTURE_DOSCAR: Path`、`FIXTURE_PROCAR: Path`

- [ ] **Step 1: 在 master 上用 VASP 6.6.0 生成 2 k 点 fixture**

在 master 上执行（`/data/wjlu/work/claude/unfold_fs/fixture/`）：

```bash
PP64=/data/wjlu/pesudopotential/vasp_pp/potpaw_PBE.64
mkdir -p /data/wjlu/work/claude/unfold_fs/fixture && cd /data/wjlu/work/claude/unfold_fs/fixture
cp /data/wjlu/work/claude/unfold_fs_demo/POSCAR .          # MoS2 3x3x1 超胞
cp /data/wjlu/work/claude/unfold_fs_demo/PRIMCELL.vasp .  # 原胞
cp /data/wjlu/work/claude/unfold_fs_demo/TRANSMAT.in .     # 3x3x1
cat $PP64/Mo/POTCAR > POTCAR; cat $PP64/S/POTCAR >> POTCAR
cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
LORBIT = 12
ENCUT  = 320
PREC   = Normal
ISMEAR = 0
SIGMA  = 0.05
EDIFF  = 1E-6
IBRION = -1
NSW    = 0
ISYM   = 0
LREAL  = Auto
NBANDS = 120
EOF
cat > KPOINTS <<'EOF'
fixture
   2
Reciprocal lattice
   0.000000000000   0.000000000000   0.000000000000       1
   0.100000000000   0.000000000000   0.000000000000       1
EOF
source /data/intel-2025/setvars.sh intel64 --force
source /data_gpu/software/bashrc_dir/bashrc.gcc-13
mpirun -np 8 /data/wjlu/program/vasp.6.6.0/bin/vasp_std_6.6.0
```

验证：`head -1 PROCAR` 必须是 `PROCAR lm decomposed + phase`；`wc -c PROCAR` 应在 1~2 MB。

- [ ] **Step 2: 下载 fixture 到本地 `tests/data/`**

```bash
cd F:/cc-haha_projects/unfold_band_fs
for f in POSCAR PRIMCELL.vasp TRANSMAT.in EIGENVAL DOSCAR PROCAR INCAR; do
  MSYS_NO_PATHCONV=1 python rsh.py --download \
    /data/wjlu/work/claude/unfold_fs/fixture/$f "F:/cc-haha_projects/unfold_band_fs/tests/data/$f"
done
```

- [ ] **Step 3: 写 `tests/conftest.py`**

```python
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FIXTURE_POSCAR = DATA_DIR / "POSCAR"
FIXTURE_PRIMCELL = DATA_DIR / "PRIMCELL.vasp"
FIXTURE_TRANSMAT = DATA_DIR / "TRANSMAT.in"
FIXTURE_EIGENVAL = DATA_DIR / "EIGENVAL"
FIXTURE_DOSCAR = DATA_DIR / "DOSCAR"
FIXTURE_PROCAR = DATA_DIR / "PROCAR"
```

- [ ] **Step 4: 写骨架与 pytest 配置**

`unfoldfs/__init__.py`：
```python
"""unfoldfs: 超胞反折叠费米面计算工具。"""
__version__ = "0.1.0"
```

`pytest.ini`：
```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

`.gitignore`：
```
__pycache__/
*.pyc
.pytest_cache/
run/
fs/
```

- [ ] **Step 5: 写冒烟测试并确认通过**

`tests/test_geometry.py`（首个测试文件，后续任务继续往里加）：
```python
from tests.conftest import FIXTURE_POSCAR


def test_fixture_is_present():
    assert FIXTURE_POSCAR.is_file()
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add .gitignore pytest.ini unfoldfs/__init__.py tests/
git commit -m "chore: 项目骨架与 VASP 6.6.0 fixture"
```

---

### Task 2: geometry.py — 结构、变换、候选矢量、公共对称集

**Files:**
- Create: `unfoldfs/geometry.py`
- Modify: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `tests/conftest.py` 的 fixture 路径常量
- Produces:
  ```python
  @dataclass(frozen=True)
  class Structure:
      lattice: np.ndarray        # (3,3) 行=实矢量(Å)
      positions: np.ndarray      # (N,3) 分数坐标
      species: tuple[str, ...]
      @property
      def n_atoms(self) -> int

  @dataclass(frozen=True)
  class Transform:
      m_eff: np.ndarray          # (3,3) int，行约定下 A_sc = m_eff @ A_prim
      m_inv: np.ndarray          # (3,3) float
      @property
      def det(self) -> int
      def prim_from_super(self, k_super: np.ndarray, g: np.ndarray) -> np.ndarray
      def super_from_prim(self, k_prim: np.ndarray) -> np.ndarray

  def read_poscar(path: str) -> Structure
  def reciprocal(lattice: np.ndarray) -> np.ndarray      # (3,3) 行=倒格矢, 含2π
  def read_transmat(path: str) -> np.ndarray              # (3,3) int
  def resolve_transform(sc: Structure, prim: Structure, m: np.ndarray) -> Transform
  def matrix_denominator_lcm(m_eff: np.ndarray) -> int
  def candidate_g_vectors(m_eff: np.ndarray) -> np.ndarray  # (detM,3) int
  def common_symmetry_ops(prim: Structure, m_eff: np.ndarray, symprec: float = 1e-5) -> SymmetryOps

  @dataclass(frozen=True)
  class SymmetryOps:
      rotations: np.ndarray      # (n_ops,3,3) int，原胞基底
      translations: np.ndarray   # (n_ops,3) float，原胞分数坐标
      def apply(self, k_prim: np.ndarray, op_index: int) -> np.ndarray
  ```

- [ ] **Step 1: 写失败测试（解析与倒格矢）**

追加到 `tests/test_geometry.py`：
```python
import numpy as np
import pytest

from unfoldfs.geometry import (
    Structure, read_poscar, reciprocal, read_transmat,
    resolve_transform, matrix_denominator_lcm, candidate_g_vectors,
    common_symmetry_ops,
)
from tests.conftest import (
    FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT,
)


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
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfoldfs.geometry'`

- [ ] **Step 2: 实现 geometry.py（第一部分：解析与倒格矢）**

```python
"""几何与对称性：结构解析、变换矩阵、候选折叠矢量、公共对称操作集。"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Structure:
    lattice: np.ndarray
    positions: np.ndarray
    species: Tuple[str, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.species)


def read_poscar(path: str) -> Structure:
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    ) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip()
    idx = 8
    if mode.lower().startswith("s"):          # Selective dynamics
        idx = 9
    cartesian = mode.lower().startswith("c")
    inv = np.linalg.inv(lattice)
    positions, species = [], []
    for sym, n in zip(symbols, counts):
        for _ in range(n):
            p = np.array([float(x) for x in lines[idx].split()[:3]])
            positions.append(p @ inv if cartesian else p)
            species.append(sym)
            idx += 1
    return Structure(lattice, np.array(positions), tuple(species))


def reciprocal(lattice: np.ndarray) -> np.ndarray:
    """行=倒格矢, 含 2π (2π/Å)。"""
    vol = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    return 2.0 * np.pi * np.array([
        np.cross(lattice[1], lattice[2]),
        np.cross(lattice[2], lattice[0]),
        np.cross(lattice[0], lattice[1]),
    ]) / vol


def read_transmat(path: str) -> np.ndarray:
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            rows.append([int(float(x)) for x in parts])
    if len(rows) != 3:
        raise ValueError(f"TRANSMAT 解析失败, 得到 {len(rows)} 行")
    return np.array(rows, dtype=int)
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS（前三个测试通过，后面的还没写）

- [ ] **Step 3: 写失败测试（变换约定探测、候选 G、comm）**

追加到 `tests/test_geometry.py`：
```python
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
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL（`resolve_transform` 未定义）

- [ ] **Step 4: 实现变换约定探测与候选 G**

追加到 `unfoldfs/geometry.py`：
```python
@dataclass(frozen=True)
class Transform:
    m_eff: np.ndarray
    m_inv: np.ndarray

    @property
    def det(self) -> int:
        return int(round(abs(np.linalg.det(self.m_eff))))

    def prim_from_super(self, k_super: np.ndarray, g: np.ndarray) -> np.ndarray:
        """k'_prim = (K + G) · (M_eff^-1)^T   (倒空间用 M^-T, 见 Global Constraints)"""
        return (k_super + g) @ self.m_inv.T

    def super_from_prim(self, k_prim: np.ndarray) -> np.ndarray:
        """K = k'_prim · M_eff^T (mod 1)"""
        return np.mod(k_prim @ self.m_eff.T, 1.0)


def resolve_transform(sc: Structure, prim: Structure, m: np.ndarray) -> Transform:
    """探测 TRANSMAT 的行/列约定, 统一到行约定 (A_sc = M_eff @ A_prim)。"""
    for cand in (m, m.T):
        if np.allclose(cand @ prim.lattice, sc.lattice, atol=1e-4):
            m_eff = np.array(cand, dtype=int)
            return Transform(m_eff, np.linalg.inv(m_eff.astype(float)))
    raise ValueError(
        "TRANSMAT/POSCAR/PRIMCELL 不自洽: "
        f"M={m.tolist()} 与超胞晶格 {sc.lattice.tolist()} 不匹配"
    )


def matrix_denominator_lcm(m_eff: np.ndarray) -> int:
    """comm(M_eff^-1): 使 M_eff^-1 所有元素分母整除的最小正整数。"""
    inv = np.linalg.inv(np.asarray(m_eff, dtype=float))
    dens = [Fraction(float(x)).limit_denominator(1000).denominator for x in inv.ravel()]
    out = 1
    for d in dens:
        out = out * d // _gcd(out, d)
    return int(out)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def candidate_g_vectors(m_eff: np.ndarray) -> np.ndarray:
    """{g ∈ Z³ : g·(M_eff^-1)^T ∈ [0,1)³}, 恰 det(M) 个。"""
    m = np.asarray(m_eff, dtype=float)
    inv = np.linalg.inv(m).T
    det = int(round(abs(np.linalg.det(m))))     # 注意: det(M^-1) = 1/det(M), 不能取整
    span = int(np.abs(m).max()) + 2
    out = []
    for g in np.ndindex(*([2 * span + 1] * 3)):
        gg = np.array(g) - span
        t = gg @ inv
        if np.all(t >= -1e-9) and np.all(t < 1.0 - 1e-9):
            out.append(gg)
    if len(out) != det:
        raise ValueError(f"候选 G 数 {len(out)} != det(M) {det}")
    return np.array(out, dtype=int)
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS

- [ ] **Step 5: 写失败测试（公共对称集）**

追加到 `tests/test_geometry.py`：
```python
def test_common_symmetry_ops_rotations_are_integer_in_supercell_basis():
    """旋转在超胞基底必须是整数矩阵 (M^-T R M^T ∈ Z³)。

    注意: 平移**不**参与筛选 —— spglib 给的 t 是模原胞格矢的代表元,
    用它做筛选会误删操作(实测会丢掉 C3², 得到不成群的子集)。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    assert ops.rotations.shape[0] == 12          # 原胞 D3h 全群
    inv = np.linalg.inv(m.astype(float)).T
    for r in ops.rotations:
        r_sc = inv @ r.astype(float) @ m.astype(float).T
        assert np.allclose(r_sc, np.round(r_sc), atol=1e-6)
        assert np.allclose(np.round(inv @ r @ m.astype(float)), r)   # 对角阵下二者等价
    for t in ops.translations:
        assert np.allclose(t @ inv, np.round(t @ inv))


def test_common_symmetry_ops_form_a_group():
    """成群性: 任意两个操作的复合仍在集合内。

    比"作用两次回原点"(全群对合)更强且更正确 —— C3 阶为 3、S3 阶为 6,
    D3h 里根本不存在非平凡对合, 原来的 is_involution 断言在数学上不成立。
    """
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)

    def close(a, b):
        d = np.mod(a - b + 0.5, 1.0) - 0.5
        return np.abs(d).max() < 1e-6

    for i in range(ops.rotations.shape[0]):
        for j in range(ops.rotations.shape[0]):
            comp = ops.apply(ops.apply(k, i), j)
            assert any(close(comp, ops.apply(k, l))
                       for l in range(ops.rotations.shape[0]))
```

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL（`common_symmetry_ops` 未定义）

- [ ] **Step 6: 实现公共对称集（spglib）**

追加到 `unfoldfs/geometry.py`：
```python
@dataclass(frozen=True)
class SymmetryOps:
    rotations: np.ndarray
    translations: np.ndarray

    def apply(self, k_prim: np.ndarray, op_index: int) -> np.ndarray:
        r = self.rotations[op_index]
        t = self.translations[op_index]
        return np.mod(k_prim @ r.T + t, 1.0)   # r 以列向量约定存储 -> 转置


def common_symmetry_ops(prim: Structure, m_eff: np.ndarray,
                        symprec: float = 1e-5) -> SymmetryOps:
    """原胞对称操作中, 同时也是超胞对称的那些。

    判据 (行约定):  M^-1 R M 为整数矩阵  且  t·M^-1 为整数矢量。
    """
    import spglib

    # spglib 的 cell = (lattice, positions, numbers): lattice 行=基矢, positions 分数坐标
    cell = (prim.lattice, prim.positions, _species_ids(prim.species))
    # spglib >= 2.0: get_symmetry 返回 dict(rotations/translations/equivalent_atoms)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    rotations = sym["rotations"]
    translations = sym["translations"]
    m = np.asarray(m_eff, dtype=float)
    inv = np.linalg.inv(m)
    keep_r, keep_t = [], []
    for r, t in zip(rotations, translations):
        # spglib 的 R 是列向量约定(k' = R k); 超胞基底的整数性判据用 M^-T R M^T
        # (已用「原子映射」真值交叉核对: 非对称 M 下 12/12 通过, 旧判据只剩 4)
        r_sc = inv @ r.astype(float) @ m
        if not np.allclose(r_sc, np.round(r_sc), atol=1e-6):
            continue        # 旋转在超胞基底非整数 -> 该操作不保持超胞晶格
        # 注意: **不要**用平移做筛选 —— spglib 给的 t 是模原胞格矢的代表元
        # (C3^2 的代表元是 (1,0,0)), 换基底后 t·M^-1 未必是整数, 会误删操作。
        # 平移只需是原胞晶格平移(spglib 已保证), 而 A(k,E) 对整个点群不变。
        keep_r.append(r.astype(int))
        keep_t.append(np.mod(t, 1.0))
    if not keep_r:
        raise ValueError("未找到任何公共对称操作")
    return SymmetryOps(np.array(keep_r), np.array(keep_t))


def _species_ids(species: Tuple[str, ...]) -> np.ndarray:
    uniq = sorted(set(species))
    index = {s: i + 1 for i, s in enumerate(uniq)}
    return np.array([index[s] for s in species], dtype=int)
```

注意：`spglib.get_symmetry` 的 cell 元组为 `(lattice, positions, numbers)`，其中 lattice 用**行=实矢量**且 positions 为分数坐标 ✓（spglib 约定：lattice 的行是基矢，positions 分数 ✓）。

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add unfoldfs/geometry.py tests/test_geometry.py
git commit -m "feat: geometry 模块(结构解析/变换约定探测/候选G/公共对称集)"
```

---

### Task 3: grid.py — 均匀原胞网格、轨道分解、KPOINTS 写出

**Files:**
- Create: `unfoldfs/grid.py`
- Modify: `tests/test_grid.py`（新建）

**Interfaces:**
- Consumes: `geometry.read_poscar` / `geometry.resolve_transform` / `geometry.common_symmetry_ops` / `geometry.matrix_denominator_lcm`
- Produces:
  ```python
  @dataclass(frozen=True)
  class GridSpec:
      p: tuple[int, int, int]          # 原胞网格点数
      grid_type: str                   # "gamma" | "mp"
      shift: tuple[float, float, float]

  @dataclass(frozen=True)
  class GridMap:
      spec: GridSpec
      ir_k_prim: np.ndarray             # (n_ir,3) 原胞分数坐标
      ir_k_super: np.ndarray            # (n_ir,3) 超胞分数坐标
      multiplicity: np.ndarray          # (n_ir,)
      full_to_ir: np.ndarray            # (n_points,)
      rot_of_full: np.ndarray           # (n_points,) int, -1 = 不动
      ops: SymmetryOps
      transform: Transform
      b_prim: np.ndarray                # (3,3)
      b_super: np.ndarray               # (3,3)

  def primitive_grid_points(spec: GridSpec) -> np.ndarray       # (n,3) 原胞分数坐标
  def grid_resolution(b_prim: np.ndarray, spec: GridSpec) -> float
  def choose_grid_points(b_prim, m_eff, resolution, grid_type) -> GridSpec
  def filter_ops_by_grid(ops: SymmetryOps, grid: np.ndarray) -> SymmetryOps
  def decompose_grid(grid: np.ndarray, ops: SymmetryOps) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
  def write_kpoints(path: str, grid_map: GridMap) -> None
  def save_grid_map(path: str, grid_map: GridMap) -> None
  def load_grid_map(path: str) -> GridMap
  ```

- [ ] **Step 1: 写失败测试（网格点与点数约束）**

新建 `tests/test_grid.py`：
```python
import numpy as np
import pytest

from unfoldfs.geometry import read_poscar, read_transmat, resolve_transform
from unfoldfs.grid import (
    GridSpec, primitive_grid_points, choose_grid_points, matrix_denominator_lcm,
)


def test_grid_points_are_gamma_centered():
    spec = GridSpec(p=(4, 4, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    g = primitive_grid_points(spec)
    assert g.shape == (16, 1 + 1)  # 占位，下面修正
```

修正为：
```python
def test_grid_points_are_gamma_centered():
    spec = GridSpec(p=(4, 4, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    g = primitive_grid_points(spec)
    assert g.shape == (16, 3)
    assert np.allclose(g.min(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(g.max(axis=0), [0.75, 0.75, 0.0])


def test_grid_points_mp_is_shifted():
    spec = GridSpec(p=(4, 4, 1), grid_type="mp", shift=(0.0, 0.0, 0.0))
    g = primitive_grid_points(spec)
    assert np.allclose(g[:, 0], [0.125, 0.375, 0.625, 0.875])


def test_choose_grid_points_respects_comm():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    spec = choose_grid_points(prim.lattice, m, resolution=0.2, grid_type="gamma")
    assert all(p % 3 == 0 for p in spec.p)
```

`tests/test_grid.py` 顶部需要 `from tests.conftest import FIXTURE_PRIMCELL, FIXTURE_TRANSMAT`。

Run: `python -m pytest tests/test_grid.py -v`
Expected: FAIL（`unfoldfs.grid` 未定义）

- [ ] **Step 2: 实现网格点生成与点数选择**

`unfoldfs/grid.py`：
```python
"""均匀原胞网格、对称性轨道分解、KPOINTS 写出。"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import (Structure, SymmetryOps, Transform, common_symmetry_ops,
                        matrix_denominator_lcm, read_poscar, read_transmat,
                        reciprocal, resolve_transform)


@dataclass(frozen=True)
class GridSpec:
    p: Tuple[int, int, int]
    grid_type: str
    shift: Tuple[float, float, float]

    @property
    def n_points(self) -> int:
        return int(self.p[0] * self.p[1] * self.p[2])


def primitive_grid_points(spec: GridSpec) -> np.ndarray:
    """原胞 BZ 内的均匀网格点(分数坐标), 行=点序。"""
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
    """按分辨率选网格点数, 并强制 P 为 comm(M^-1) 的倍数。"""
    b = reciprocal(lattice)
    comm = matrix_denominator_lcm(m_eff)
    p = []
    for i in range(3):
        n = max(int(np.ceil(np.linalg.norm(b[i]) / resolution)), 1)
        while n % comm != 0:
            n += 1
        p.append(n)
    return GridSpec(tuple(p), grid_type, (0.0, 0.0, 0.0))
```

Run: `python -m pytest tests/test_grid.py -v`
Expected: PASS

- [ ] **Step 3: 写失败测试（轨道分解）**

追加到 `tests/test_grid.py`：
```python
from unfoldfs.grid import decompose_grid, filter_ops_by_grid


def test_decompose_grid_closed_under_symmetry():
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    assert sum(mult) == grid.shape[0]
    assert int(mult.max()) >= 1
    for i in range(grid.shape[0]):
        assert full_to_ir[i] == full_to_ir[ir_idx[i]]


def test_filter_ops_keeps_only_grid_preserving_ops():
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    ops = common_symmetry_ops(prim, m)
    filtered = filter_ops_by_grid(ops, grid)
    for i in range(filtered.rotations.shape[0]):
        img = filtered.apply(grid, i)
        assert np.allclose(np.sort(img, axis=0), np.sort(grid, axis=0), atol=1e-6)
```

Run: `python -m pytest tests/test_grid.py -v`
Expected: FAIL（`decompose_grid` / `filter_ops_by_grid` 未定义）

- [ ] **Step 4: 实现网格不变性过滤与轨道分解**

追加到 `unfoldfs/grid.py`：
```python
def _key(p: np.ndarray, tol: float = 1e-6) -> tuple:
    return tuple(np.round(np.mod(p, 1.0) / tol).astype(np.int64))


def filter_ops_by_grid(ops: SymmetryOps, grid: np.ndarray) -> SymmetryOps:
    """只保留把网格映射到自身的操作(平移也必须落在网格上)。"""
    keys = {_key(p) for p in grid}
    keep_r, keep_t = [], []
    for i in range(ops.rotations.shape[0]):
        if {_key(p) for p in ops.apply(grid, i)} == keys:
            keep_r.append(ops.rotations[i])
            keep_t.append(ops.translations[i])
    if not keep_r:
        raise ValueError("没有操作保持该网格不变, 请调整网格点数/类型")
    return SymmetryOps(np.array(keep_r), np.array(keep_t))


def decompose_grid(grid: np.ndarray, ops: SymmetryOps):
    """把网格点按对称轨道分组。

    返回 (ir_list, multiplicity, full_to_ir, rot_of_full):
      ir_list       -> 代表点在 grid 中的行号
      multiplicity  -> 每个代表点的轨道大小
      full_to_ir[i] -> i 所属代表点在"代表点列表"中的序号
      rot_of_full[i]-> i 由哪个操作映射到代表点(-1 表示本身即代表)
    """
    n = grid.shape[0]
    index = {_key(p): i for i, p in enumerate(grid)}
    rep_of_point: dict = {}
    ir_list: list = []
    mult: list = []
    rot_of_full = np.full(n, -1, dtype=int)
    full_to_ir = np.full(n, -1, dtype=int)
    for i in range(n):
        if i in rep_of_point:
            continue
        orbit = []
        seen = set()
        for o in range(ops.rotations.shape[0]):
            j = index.get(_key(ops.apply(grid[i], o)))
            if j is not None and j not in seen:   # 不同操作可能落到同一格点 -> 必须去重
                seen.add(j)
                orbit.append((j, o))
        if not orbit:
            orbit = [(i, -1)]
        for j, o in orbit:
            rep_of_point[j] = i
            rot_of_full[j] = o
        ir_list.append(i)
        mult.append(len(orbit))
    for i in range(n):
        r = rep_of_point.get(i)
        if r is None:
            raise ValueError("轨道分解不完整")
        full_to_ir[i] = ir_list.index(r)
    return np.array(ir_list), np.array(mult), full_to_ir, rot_of_full
```

Run: `python -m pytest tests/test_grid.py -v`
Expected: PASS

- [ ] **Step 5: 写失败测试（KPOINTS 写出与 grid_map 存取）**

追加到 `tests/test_grid.py`：
```python
from unfoldfs.grid import GridMap, write_kpoints, save_grid_map, load_grid_map


def test_write_kpoints_format_and_weights(tmp_path):
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    tr = resolve_transform(sc, prim, m)
    ops = common_symmetry_ops(prim, tr.m_eff)
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    gm = GridMap(
        spec=spec,
        ir_k_prim=grid[ir_idx],
        ir_k_super=np.array([tr.super_from_prim(k) for k in grid[ir_idx]]),
        multiplicity=mult,
        full_to_ir=full_to_ir,
        rot_of_full=rot,
        ops=ops,
        transform=tr,
        b_prim=reciprocal(prim.lattice),
        b_super=reciprocal(sc.lattice),
    )
    path = tmp_path / "KPOINTS"
    write_kpoints(str(path), gm)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("G")
    assert lines[2] == "Reciprocal lattice"
    assert len(lines) == 3 + int(gm.ir_k_prim.shape[0])
    weights = [float(l.split()[3]) for l in lines[3:]]
    assert abs(sum(weights) - gm.spec.n_points) < 1e-9

    npz = tmp_path / "grid_map.npz"
    save_grid_map(str(npz), gm)
    back = load_grid_map(str(npz))
    assert np.allclose(back.ir_k_prim, gm.ir_k_prim)
    assert np.array_equal(back.full_to_ir, gm.full_to_ir)
```

Run: `python -m pytest tests/test_grid.py -v`
Expected: FAIL（`GridMap` / `write_kpoints` 未定义）

- [ ] **Step 6: 实现 GridMap、KPOINTS 写出与 npz 存取**

追加到 `unfoldfs/grid.py`：
```python
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
    with open(path, "w") as fh:
        fh.write(f"{tag}    {p[0]}    {p[1]}    {p[2]}    "
                 f"{grid_resolution(grid_map.b_prim, grid_map.spec):.3f}"
                 f"     # Parameters to Generate KPOINTS (Don't Edit This Line)\n")
        fh.write(f"   {grid_map.n_ir}\n")
        fh.write("Reciprocal lattice\n")
        for k, w in zip(grid_map.ir_k_super, grid_map.multiplicity):
            fh.write(f"    {k[0]:.12f}    {k[1]:.12f}    {k[2]:.12f}    "
                     f"{float(w):.12f}\n")


def grid_resolution(b_prim: np.ndarray, spec: GridSpec) -> float:
    return float(max(np.linalg.norm(b_prim[i]) / spec.p[i] for i in range(3)))


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
```

Run: `python -m pytest tests/test_grid.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add unfoldfs/grid.py tests/test_grid.py
git commit -m "feat: grid 模块(均匀原胞网格/轨道分解/KPOINTS写出/grid_map)"
```

---

### Task 4: vaspio.py — EIGENVAL / DOSCAR / PROCAR(LORBIT=12) 解析

**Files:**
- Create: `unfoldfs/vaspio.py`
- Create: `tests/test_vaspio.py`

**Interfaces:**
- Consumes: `tests/conftest.py` 的 fixture 路径常量
- Produces:
  ```python
  @dataclass(frozen=True)
  class BandData:
      kpoints: np.ndarray      # (nk,3) 分数坐标
      energies: np.ndarray     # (nk, nb) eV
      occupations: np.ndarray  # (nk, nb)
      efermi: float

  @dataclass(frozen=True)
  class ProcarData:
      kpoints: np.ndarray      # (nk,3) 笛卡尔(2π/Å)
      weights: np.ndarray      # (nk, nb, nions, norb) |p|^2
      coeffs: np.ndarray       # (nk, nb, nions, norb) complex
      has_phase: bool

  def read_eigenval(path: str) -> BandData
  def read_doscar_fermi(path: str) -> float
  def read_procar(path: str) -> ProcarData
  ```

- [ ] **Step 1: 写失败测试**

新建 `tests/test_vaspio.py`：
```python
import numpy as np

from unfoldfs.vaspio import read_eigenval, read_doscar_fermi, read_procar
from tests.conftest import FIXTURE_EIGENVAL, FIXTURE_DOSCAR, FIXTURE_PROCAR


def test_read_eigenval_shape():
    b = read_eigenval(str(FIXTURE_EIGENVAL))
    assert b.kpoints.shape == (2, 3)
    assert b.energies.shape[0] == 2
    assert np.allclose(b.kpoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(b.kpoints[1], [0.1, 0.0, 0.0])


def test_read_doscar_fermi_is_fourth_field_of_line6():
    ef = read_doscar_fermi(str(FIXTURE_DOSCAR))
    lines = open(str(FIXTURE_DOSCAR)).read().splitlines()
    assert abs(ef - float(lines[5].split()[3])) < 1e-12


def test_read_procar_has_phase_block():
    p = read_procar(str(FIXTURE_PROCAR))
    assert p.has_phase is True
    assert p.coeffs.shape[:3] == p.weights.shape[:3]
    # 模方块与复投影块模长自洽
    # 注意: VASP 两块的归一化/取整口径不同 —— 实测 58320 个比较中 0.51% 超 2e-3
    # (最大 7.5e-3), 故容差取 1e-2; 末列(Σ|c|^2)与模方块逐离子合计则一致到 ≤8e-4
    amp = np.abs(p.coeffs)
    assert np.allclose(amp ** 2, p.weights, atol=1e-2)


def test_read_procar_equivalent_copies_have_same_value():
    """由超胞晶格平移联系起来的副本, 复投影值相同。

    实测: 3x3x1 超胞里同一原胞原子的 9 个副本中, 由**超胞晶格平移**
    (3a1 / 3a2) 联系的那 3 个复值相同; 跨平移的则相差 e^{i K.T}。
    这里只断言"同一组内"相同: 连续 3 个离子为 1 组。
    """
    p = read_procar(str(FIXTURE_PROCAR))
    c = p.coeffs[0, 0, :, 0]          # band 0, k=Gamma, s 轨道
    for start in range(0, 27, 3):
        group = c[start:start + 3]
        assert np.allclose(group, group[0], atol=1e-6)
```

Run: `python -m pytest tests/test_vaspio.py -v`
Expected: FAIL（`unfoldfs.vaspio` 未定义）

- [ ] **Step 2: 实现 vaspio.py**

```python
"""VASP 输出解析: EIGENVAL / DOSCAR / PROCAR(LORBIT=12 复投影)。"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandData:
    kpoints: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    efermi: float


@dataclass(frozen=True)
class ProcarData:
    kpoints: np.ndarray
    weights: np.ndarray
    coeffs: np.ndarray
    has_phase: bool


def read_eigenval(path: str) -> BandData:
    with open(path) as fh:
        lines = fh.readlines()
    hdr = lines[5].split()
    nkpts, nbands = int(hdr[-2]), int(hdr[-1])
    kpoints = np.zeros((nkpts, 3))
    energies = np.zeros((nkpts, nbands))
    occ = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        kpoints[ik] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
        for ib in range(nbands):
            v = lines[idx].split()
            energies[ik, ib] = float(v[1])
            occ[ik, ib] = float(v[2])
            idx += 1
        idx += 1
    return BandData(kpoints, energies, occ, float("nan"))


def read_doscar_fermi(path: str) -> float:
    """DOSCAR 第 6 行: EMAX EMIN NEDOS EFERMI ..."""
    with open(path) as fh:
        for _ in range(5):
            fh.readline()
        return float(fh.readline().split()[3])


def read_procar(path: str) -> ProcarData:
    with open(path) as fh:
        lines = fh.readlines()
    nums = [int(x) for x in re.findall(r"\d+", lines[1])]
    nk, nb, nions = nums[0], nums[1], nums[2]
    norb = None
    for i in range(4, min(len(lines), 80)):
        if lines[i].lstrip().startswith("ion"):
            norb = len(lines[i].split()) - 2      # 去掉 "ion" 与 "tot"
            break
    if norb is None:
        raise ValueError("PROCAR: 未找到 ion 表头")
    weights = np.zeros((nk, nb, nions, norb))
    coeffs = np.zeros((nk, nb, nions, norb), dtype=complex)
    kpoints = np.zeros((nk, 3))
    has_phase = False
    idx = 0
    for ik in range(nk):
        while idx < len(lines) and not lines[idx].lstrip().startswith("k-point"):
            idx += 1
        if idx >= len(lines):
            break
        p = lines[idx].split()
        # " k-point     1 :    0.0 0.0 0.0     weight = ..." -> 坐标在 token 3..5
        kpoints[ik] = [float(x) for x in p[3:6]]
        idx += 1
        for ib in range(nb):
            while idx < len(lines) and not lines[idx].lstrip().startswith("band"):
                idx += 1
            if idx >= len(lines):
                break
            rows = []
            j = idx + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("band") or s.startswith("k-point"):
                    break
                if s and s[0].isdigit():
                    rows.append((j, len(s.split()), s.split()))
                j += 1
            idx = j
            for _ln, nf, v in rows:
                ia = int(v[0]) - 1
                if ia >= nions:
                    continue
                if nf == 11:
                    for m in range(norb):
                        weights[ik, ib, ia, m] = float(v[1 + m])
                elif nf == 20:
                    has_phase = True
                    for m in range(norb):
                        coeffs[ik, ib, ia, m] = (float(v[1 + 2 * m])
                                                  + 1j * float(v[2 + 2 * m]))
    if not has_phase:
        raise ValueError(
            "PROCAR 中没有复投影块 —— 需要 LORBIT=12; LORBIT=11 不写相位")
    return ProcarData(kpoints, weights, coeffs, has_phase)
```

Run: `python -m pytest tests/test_vaspio.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add unfoldfs/vaspio.py tests/test_vaspio.py
git commit -m "feat: vaspio 模块(EIGENVAL/DOSCAR/PROCAR LORBIT=12 解析)"
```

---

### Task 5: unfold.py — Popescu 投影展开与候选重组

**Files:**
- Create: `unfoldfs/unfold.py`
- Create: `tests/test_unfold.py`

**Interfaces:**
- Consumes:
  - `geometry.Transform`（`prim_from_super` / `super_from_prim`）
  - `geometry.candidate_g_vectors`
  - `geometry.Structure`（原胞与超胞原子位置、复制关系）
  - `vaspio.BandData` / `vaspio.ProcarData`
- Produces:
  ```python
  @dataclass(frozen=True)
  class AtomCopies:
      """原胞原子 -> 超胞副本(索引, 原胞平移 T)。"""
      copies: list[tuple[int, np.ndarray]]

  def build_atom_copies(prim: Structure, sc: Structure, tr: Transform) -> AtomCopies
  def unfold_at_k(k_super_frac, coeffs, energies, tr, copies, gset,
                 a_prim, prim_cart, b_prim, b_super, s1, s2)
      # 返回 (k_prim_frac (n_cand,3), E (n_cand,n_band), w (n_cand,n_band))
  def select_phase_convention(...) -> tuple[int, int, float]
  def unfold_supercell(band, procar, tr, copies, gset, a_prim, prim_cart,
                      b_prim, b_super, k_super_list) -> UnfoldedResult
  ```

- [ ] **Step 1: 写失败测试（原子副本与展开结构）**

新建 `tests/test_unfold.py`：
```python
import numpy as np

from unfoldfs.geometry import (read_poscar, read_transmat, resolve_transform,
                               candidate_g_vectors, reciprocal)
from unfoldfs.vaspio import read_eigenval, read_procar
from unfoldfs.unfold import build_atom_copies, unfold_at_k
from tests.conftest import (FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT,
                             FIXTURE_EIGENVAL, FIXTURE_PROCAR)


def _setup():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    tr = resolve_transform(sc, prim, m)
    band = read_eigenval(str(FIXTURE_EIGENVAL))
    procar = read_procar(str(FIXTURE_PROCAR))
    copies = build_atom_copies(prim, sc, tr)
    gset = candidate_g_vectors(tr.m_eff)
    return dict(
        prim=prim, sc=sc, tr=tr, band=band, procar=procar, copies=copies,
        gset=gset,
        a_prim=prim.lattice,
        prim_cart=prim.positions @ prim.lattice,
        b_prim=reciprocal(prim.lattice),
        b_super=reciprocal(sc.lattice),
    )


def test_atom_copies_are_nine_each():
    ctx = _setup()
    assert len(ctx["copies"].copies) == 3      # 原胞 3 原子
    assert all(len(c) == 9 for c in ctx["copies"].copies)


def test_unfold_at_k_single_dominant_candidate():
    ctx = _setup()
    k_prim, e, w = unfold_at_k(
        k_super_frac=np.array([0.0, 0.0, 0.0]),
        coeffs=ctx["procar"].coeffs[0], energies=ctx["band"].energies[0],
        tr=ctx["tr"], copies=ctx["copies"], gset=ctx["gset"],
        a_prim=ctx["a_prim"], prim_cart=ctx["prim_cart"],
        b_prim=ctx["b_prim"], b_super=ctx["b_super"], s1=1, s2=1,
    )
    assert k_prim.shape == (9, 3)
    assert e.shape == (9, 120)
    assert w.shape == (9, 120)
    # 每个带应有一个主导候选
    for ib in range(0, 120, 7):
        row = w[:, ib]
        if row.sum() <= 1e-12:
            continue
        assert row.max() / row.sum() > 0.5
```

Run: `python -m pytest tests/test_unfold.py -v`
Expected: FAIL（`unfoldfs.unfold` 未定义）

- [ ] **Step 2: 实现 unfold.py**

```python
"""投影法超胞能带反折叠 (Popescu, PRL 104, 216401 (2012))。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .geometry import Structure, Transform, candidate_g_vectors


@dataclass(frozen=True)
class AtomCopies:
    """原胞原子 -> 超胞副本 (超胞原子索引, 原胞格矢平移 T)。"""
    copies: List[List[Tuple[int, np.ndarray]]]


@dataclass(frozen=True)
class UnfoldedResult:
    k_cand: np.ndarray      # (n_ir, n_cand, 3) 每个计算点的候选原胞 k(分数坐标)
    energies: np.ndarray    # (n_ir, n_cand, n_band_eff)
    weights: np.ndarray     # (n_ir, n_cand, n_band_eff)


def build_atom_copies(prim: Structure, sc: Structure, tr: Transform) -> AtomCopies:
    """原胞原子 -> 超胞副本(索引, 原胞格矢平移 T)。

    (q - p) 是超胞基矢下的分数差, 乘 M_eff 转到原胞基底后应为整数。
    """
    out: List[List[Tuple[int, np.ndarray]]] = []
    inv_sc = np.linalg.inv(sc.lattice)
    for p in prim.positions:
        lst: List[Tuple[int, np.ndarray]] = []
        for j, q in enumerate(sc.positions):
            # 必须是笛卡尔差: 分数坐标不能直接相减再乘逆晶格
            dq = (q @ sc.lattice - p @ prim.lattice) @ inv_sc
            t = np.round(dq @ tr.m_eff)
            if np.allclose(dq @ tr.m_eff, t, atol=1e-3):
                lst.append((j, t))
        if not lst:
            raise ValueError("原子副本匹配失败: 检查 PRIMCELL / POSCAR / TRANSMAT")
        out.append(lst)
    return AtomCopies(out)


def unfold_at_k(k_super_frac: np.ndarray, coeffs: np.ndarray,
                 energies: np.ndarray, tr: Transform, copies: AtomCopies,
                 gset: np.ndarray, a_prim: np.ndarray, prim_cart: np.ndarray,
                 b_prim: np.ndarray, b_super: np.ndarray,
                 s1: int, s2: int):
    """在一个超胞 k 点上展开。

    coeffs:    (n_band, n_ions, norb) complex —— 晶胞周期规范的复投影系数
    a_prim:    (3,3) 原胞实格矢(行, Å)      —— 用于副本的实空间平移
    prim_cart: (n_atom,3) 原胞原子实坐标(Å)  —— 用于原胞 k 的相位因子
    b_prim/b_super: (3,3) 倒格矢(含 2π)
    返回 (k_prim_frac (n_cand,3), E (n_cand,n_band), w (n_cand,n_band))
    """
    k_super_cart = k_super_frac @ b_super
    n_cand = gset.shape[0]
    n_band = energies.shape[0]
    norb = coeffs.shape[2]
    n_atom = len(copies.copies)
    k_out = np.zeros((n_cand, 3))
    e_out = np.zeros((n_cand, n_band))
    w_out = np.zeros((n_cand, n_band))
    for gi, g in enumerate(gset):
        k_prim_frac = tr.prim_from_super(k_super_frac, g)
        k_prim_cart = k_prim_frac @ b_prim
        dG = k_prim_cart - k_super_cart
        for ib in range(n_band):
            cvec = np.zeros((n_atom, norb), dtype=complex)
            for i, lst in enumerate(copies.copies):
                acc = np.zeros(norb, dtype=complex)
                for (j, t) in lst:
                    shift = t @ a_prim              # 副本的实空间平移 (Å)
                    acc += coeffs[ib, j] * np.exp(1j * s1 * float(dG @ shift))
                cvec[i] = acc / max(len(lst), 1)
            ph = np.exp(1j * s2 * (k_prim_cart @ prim_cart.T))
            w_out[gi, ib] = float(np.abs(np.sum(cvec * ph[:, None])) ** 2)
        k_out[gi] = k_prim_frac
        e_out[gi] = energies
    return k_out, e_out, w_out
```

Run: `python -m pytest tests/test_unfold.py -v`
Expected: PASS（若未通过，按报错核对：副本平移用**实空间** `t @ a_prim`、原子相位用**原胞实坐标** `prim_cart`）

- [ ] **Step 3: 写失败测试（相位约定自检）**

追加到 `tests/test_unfold.py`：
```python
from unfoldfs.unfold import select_phase_convention


def test_select_phase_convention_gives_high_dominance():
    ctx = _setup()
    s1, s2, score = select_phase_convention(
        k_super_frac=np.array([0.1, 0.0, 0.0]),
        coeffs=ctx["procar"].coeffs[1], energies=ctx["band"].energies[1],
        tr=ctx["tr"], copies=ctx["copies"], gset=ctx["gset"],
        a_prim=ctx["a_prim"], prim_cart=ctx["prim_cart"],
        b_prim=ctx["b_prim"], b_super=ctx["b_super"],
    )
    assert score > 0.5
    assert s1 in (1, -1) and s2 in (1, -1)
```

Run: `python -m pytest tests/test_unfold.py -v`
Expected: FAIL（`select_phase_convention` 未定义）

- [ ] **Step 4: 实现约定自检与整超胞展开**

追加到 `unfoldfs/unfold.py`：
```python
def select_phase_convention(k_super_frac, coeffs, energies, tr, copies, gset,
                            a_prim, prim_cart, b_prim, b_super):
    """在 4 种 (s1,s2) 组合中选「单 G 主导度」平均最大者。"""
    best = None
    for s1 in (1, -1):
        for s2 in (1, -1):
            _k, _e, w = unfold_at_k(k_super_frac, coeffs, energies, tr, copies,
                                    gset, a_prim, prim_cart, b_prim, b_super,
                                    s1, s2)
            dom = []
            for ib in range(w.shape[1]):
                row = w[:, ib]
                if row.sum() > 1e-12:
                    dom.append(row.max() / row.sum())
            score = float(np.mean(dom)) if dom else 0.0
            if best is None or score > best[2]:
                best = (s1, s2, score)
    return best


def unfold_supercell(band, procar, tr, copies, gset, a_prim, prim_cart,
                      b_prim, b_super, k_super_list) -> UnfoldedResult:
    """对每个算出的超胞 k 点展开, 并按候选重组为有效带。

    三个返回数组都以"计算点"为第一维:
      k_cand   (n_ir, n_cand, 3)  该计算点的候选原胞 k(分数坐标)
      energies (n_ir, n_cand, n_band_eff)
      weights  (n_ir, n_cand, n_band_eff)
    每个候选内按能量排序 -> n 即原胞带序号(在对称变换下不变)。
    """
    n_ir = len(k_super_list)
    n_cand = gset.shape[0]
    k_cand = np.zeros((n_ir, n_cand, 3))
    raw = []
    for ik, k_super in enumerate(k_super_list):
        s1, s2, score = select_phase_convention(
            k_super, procar.coeffs[ik], band.energies[ik], tr, copies, gset,
            a_prim, prim_cart, b_prim, b_super)
        if score < 0.5:
            raise ValueError(
                f"单 G 主导度 {score:.3f} < 0.5: 相位约定或副本匹配有误")
        k_out, e_out, w_out = unfold_at_k(
            k_super, procar.coeffs[ik], band.energies[ik], tr, copies, gset,
            a_prim, prim_cart, b_prim, b_super, s1, s2)
        k_cand[ik] = k_out
        raw.append((e_out, w_out))
    n_eff = max(w.shape[1] for _e, w in raw)
    energies = np.full((n_ir, n_cand, n_eff), np.nan)
    weights = np.zeros((n_ir, n_cand, n_eff))
    for ik, (e_out, w_out) in enumerate(raw):
        per_cand: List[List[Tuple[float, float]]] = [[] for _ in range(n_cand)]
        for ib in range(w_out.shape[1]):
            gi = int(np.argmax(w_out[:, ib]))
            per_cand[gi].append((e_out[gi, ib], w_out[gi, ib]))
        for gi, lst in enumerate(per_cand):
            for n, (e, w) in enumerate(sorted(lst)):
                energies[ik, gi, n] = e
                weights[ik, gi, n] = w
    return UnfoldedResult(k_cand, energies, weights)
```

Run: `python -m pytest tests/test_unfold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add unfoldfs/unfold.py tests/test_unfold.py
git commit -m "feat: unfold 模块(Popescu 投影展开/候选重组/相位约定自检)"
```

---

### Task 6: expand.py — 对称扩展与 A(k,E) 组装

**Files:**
- Create: `unfoldfs/expand.py`
- Create: `tests/test_expand.py`

**Interfaces:**
- Consumes: `grid.GridMap`（`ir_k_prim`, `ir_k_super`, `full_to_ir`, `rot_of_full`, `ops`, `transform`）、`unfold.UnfoldedResult`、`geometry.candidate_g_vectors`
- Produces:
  ```python
  @dataclass(frozen=True)
  class ExpandedData:
      k_cart: np.ndarray       # (n_full,3) 原胞笛卡尔 (2π/Å)
      band_e: np.ndarray       # (n_full, n_band)
      band_w: np.ndarray       # (n_full, n_band)
      spec: GridSpec

  def expand_to_full_grid(unfolded: UnfoldedResult, grid_map: GridMap) -> ExpandedData
  def build_spectral_function(expanded: ExpandedData, e_grid: np.ndarray,
                              sigma: float) -> np.ndarray   # (n_full, n_E)
  ```

- [ ] **Step 1: 写失败测试（扩展的置换正确性）**

新建 `tests/test_expand.py`：
```python
import numpy as np

from unfoldfs.geometry import (read_poscar, read_transmat, resolve_transform,
                               candidate_g_vectors, common_symmetry_ops,
                               reciprocal)
from unfoldfs.grid import (GridSpec, primitive_grid_points, filter_ops_by_grid,
                            decompose_grid, GridMap)
from unfoldfs.expand import expand_to_full_grid
from unfoldfs.unfold import UnfoldedResult
from tests.conftest import FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT


def _grid_map():
    prim = read_poscar(str(FIXTURE_PRIMCELL))
    sc = read_poscar(str(FIXTURE_POSCAR))
    m = read_transmat(str(FIXTURE_TRANSMAT))
    tr = resolve_transform(sc, prim, m)
    spec = GridSpec(p=(6, 6, 1), grid_type="gamma", shift=(0.0, 0.0, 0.0))
    grid = primitive_grid_points(spec)
    ops = common_symmetry_ops(prim, tr.m_eff)
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    return GridMap(spec, grid[ir_idx],
                   np.array([tr.super_from_prim(k) for k in grid[ir_idx]]),
                   mult, full_to_ir, rot, ops, tr,
                   reciprocal(prim.lattice), reciprocal(sc.lattice)), grid


def test_expand_preserves_total_weight():
    gm, grid = _grid_map()
    n_cand = 9
    n_band = 5
    fake = UnfoldedResult(
        k_prim=np.zeros((n_cand, 3)),
        energies=np.tile(np.arange(n_band, dtype=float), (n_cand, 1)),
        weights=np.ones((n_cand, n_band)),
    )
    exp = expand_to_full_grid(fake, gm)
    assert exp.band_e.shape == (grid.shape[0], n_band)
    assert exp.band_w.shape == (grid.shape[0], n_band)
    # 均匀权重下每个满网格点的总权重应相同
    totals = exp.band_w.sum(axis=1)
    assert np.allclose(totals, totals[0])


def test_expand_permutes_candidates_within_orbits():
    """同一轨道内, 扩展只做候选置换 -> 能带多重集不变。"""
    gm, grid = _grid_map()
    rng = np.random.default_rng(0)
    n_cand, n_band = 9, 4
    fake = UnfoldedResult(
        k_prim=np.zeros((n_cand, 3)),
        energies=rng.normal(size=(n_cand, n_band)),
        weights=rng.random((n_cand, n_band)),
    )
    exp = expand_to_full_grid(fake, gm)
    rep_sorted = np.sort(fake.energies.ravel())
    for i in range(grid.shape[0]):
        assert np.allclose(np.sort(exp.band_e[i].ravel()), rep_sorted)
        assert np.isclose(exp.band_w[i].sum(), fake.weights.sum())
```

Run: `python -m pytest tests/test_expand.py -v`
Expected: FAIL（`unfoldfs.expand` 未定义）

- [ ] **Step 2: 实现 expand.py**

```python
"""对称扩展: 约化 k 点的展开结果 -> 完整均匀原胞网格。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import GridMap, GridSpec
from .unfold import UnfoldedResult


@dataclass(frozen=True)
class ExpandedData:
    k_cart: np.ndarray       # (n_full,3) 原胞笛卡尔 (2π/Å), 完整均匀网格
    band_e: np.ndarray       # (n_full, n_band_eff) 该点覆盖候选的有效带
    band_w: np.ndarray       # (n_full, n_band_eff)
    spec: GridSpec


def expand_to_full_grid(unfolded: UnfoldedResult, grid_map: GridMap) -> ExpandedData:
    """把代表点的候选展开结果映射到完整均匀原胞网格。

    每个完整网格点 i 由唯一的 (代表点 ir, 候选 gi) 覆盖:
      - k_i 是原胞网格点, 其超胞 k = k_i·M;
      - 代表点 ir 与操作 o 满足 k_i = o·k_rep;
      - 候选 gi 满足 o·k_cand[ir][gi] = k_i (mod 1)。
    因此 data_full[i] = data_rep[ir][gi]。
    """
    n_points = int(grid_map.full_to_ir.shape[0])
    n_ir, n_cand, n_band = unfolded.energies.shape
    e_out = np.full((n_points, n_band), np.nan)
    w_out = np.zeros((n_points, n_band))
    k_prim_full = np.zeros((n_points, 3))
    for i in range(n_points):
        ir = int(grid_map.full_to_ir[i])
        o = int(grid_map.rot_of_full[i])
        cands = unfolded.k_cand[ir]
        if o < 0:
            rotated = cands
            target = grid_map.ir_k_prim[ir]
        else:
            rotated = np.array([grid_map.ops.apply(c, o) for c in cands])
            target = grid_map.ops.apply(grid_map.ir_k_prim[ir], o)
        # 找出被当前网格点覆盖的候选
        diff = np.abs((rotated - target + 0.5) % 1.0 - 0.5).sum(axis=1)
        gi = int(np.argmin(diff))
        if diff[gi] > 1e-6:
            raise ValueError(f"网格点 {i} 找不到覆盖候选(最近距离 {diff[gi]:.2e})")
        e_out[i] = unfolded.energies[ir, gi]
        w_out[i] = unfolded.weights[ir, gi]
        k_prim_full[i] = target
    k_cart = k_prim_full @ grid_map.b_prim
    return ExpandedData(k_cart, e_out, w_out, grid_map.spec)


def _gset_of(grid_map: GridMap) -> np.ndarray:
    from .geometry import candidate_g_vectors
    return candidate_g_vectors(grid_map.transform.m_eff)


def build_spectral_function(expanded: ExpandedData, e_grid: np.ndarray,
                            sigma: float) -> np.ndarray:
    """A(k,E): 对每条带用高斯展宽叠加到能量网格。"""
    d_e = float(e_grid[1] - e_grid[0])
    out = np.zeros((expanded.band_e.shape[0], e_grid.size))
    for i in range(expanded.band_e.shape[0]):
        for n in range(expanded.band_e.shape[1]):
            e = expanded.band_e[i, n]
            w = expanded.band_w[i, n]
            if not np.isfinite(e) or w <= 0:
                continue
            out[i] += w * np.exp(-0.5 * ((e_grid - e) / sigma) ** 2)
    return out
```

Run: `python -m pytest tests/test_expand.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add unfoldfs/expand.py tests/test_expand.py
git commit -m "feat: expand 模块(候选索引置换的对称扩展/谱函数组装)"
```

---

### Task 7: fsout.py — 五种输出格式

**Files:**
- Create: `unfoldfs/fsout.py`
- Create: `tests/test_fsout.py`
- Create: `docs/formats.md`

**Interfaces:**
- Consumes: `expand.ExpandedData`、`expand.build_spectral_function`
- Produces:
  ```python
  def write_cube(path: str, k_cart: np.ndarray, values: np.ndarray,
                 lattice: np.ndarray, comment: str = "") -> None
  def write_grd(path: str, k_cart: np.ndarray, values: np.ndarray) -> None
  def write_fermisurfer_bands(dirpath: str, k_cart: np.ndarray,
                               band_e: np.ndarray, band_w: np.ndarray) -> list[str]
  def write_bxsf(path: str, surfaces: list[tuple[float, np.ndarray]]) -> None
  def write_spectral_npz(path: str, expanded: ExpandedData, e_grid: np.ndarray,
                          a: np.ndarray, meta: dict) -> None
  ```

- [ ] **Step 1: 现场逆向 FermiSurfer band 文件格式（在 master 上）**

```bash
grep -rn "band\." /data/wjlu/program/fermisurfer-2.2.1/src/*.f90 | head -20
grep -rn "read(\|open(\|format(" /data/wjlu/program/fermisurfer-2.2.1/src/sub.f90 | grep -i band | head -20
```
把结论（文件名、每行字段、是否带表头）写进 `docs/formats.md`，并据此写实现。

- [ ] **Step 2: 现场逆向 vaspkit 262 的 FS.bxsf 格式（在 master 上）**

```bash
mkdir -p /tmp/fmt && cd /tmp/fmt
cp /data/wjlu/work/claude/unfold_fs_demo/POSCAR .
cp /data/wjlu/work/claude/unfold_fs_demo/DOSCAR .
cp /data/wjlu/work/claude/unfold_fs_demo/EIGENVAL .
cp /data/wjlu/work/claude/unfold_fs_demo/INCAR .
cp /data/wjlu/work/claude/unfold_fs_demo/KPOINTS_mesh10_std KPOINTS
/data/wjlu/program/vaspkit.1.5.1/bin/vaspkit -task 262 >/dev/null 2>&1
head -40 FERMISURFACE.bxsf
```
把块结构（`BEGIN_BLOCK_3D_FERMI_SURFACE` 等）、每块的字段顺序与数量写进 `docs/formats.md`。

- [ ] **Step 3: 写失败测试（cube 与 grd 往返）**

新建 `tests/test_fsout.py`：
```python
import numpy as np

from unfoldfs.fsout import write_cube, write_grd


def test_write_cube_roundtrip(tmp_path):
    lattice = np.eye(3) * 2.0
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vals = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "FS.cube"
    write_cube(str(p), pts, vals, lattice)
    lines = p.read_text().splitlines()
    assert len(lines) >= 6
    assert lines[0].startswith("FS")
    assert int(lines[2].split()[0]) == 3          # 原子数
    grid = [int(x) for x in lines[3].split()[:3]]
    assert grid == [3, 1, 1]
    data = [float(x) for x in lines[6].split()]
    assert data[:3] == [1.0, 2.0, 3.0]


def test_write_grd_has_bounds_and_values(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vals = np.array([0.5, 1.5])
    p = tmp_path / "FS.grd"
    write_grd(str(p), pts, vals)
    lines = p.read_text().splitlines()
    assert lines[0].startswith("#")
    assert float(lines[1].split()[0]) == 0.0
    assert len(lines) >= 4
```

Run: `python -m pytest tests/test_fsout.py -v`
Expected: FAIL（`unfoldfs.fsout` 未定义）

- [ ] **Step 4: 实现 cube / grd 写出**

`unfoldfs/fsout.py`：
```python
"""费米面文件输出: VESTA cube / XCrySDen grd / FermiSurfer band / bxsf / npz。"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


def write_cube(path: str, k_cart: np.ndarray, values: np.ndarray,
               lattice: np.ndarray, comment: str = "unfolded Fermi surface") -> None:
    """Gaussian cube 格式。lattice 用"倒格矢当晶格"的技巧承载 k 空间。"""
    with open(path, "w") as fh:
        fh.write(f"{comment}\n")
        fh.write("unfoldfs\n")
        fh.write(f"{len(k_cart)}  {np.abs(lattice[0]).sum():.6f}  "
                 f"{np.abs(lattice[1]).sum():.6f}  {np.abs(lattice[2]).sum():.6f}\n")
        for row in lattice:
            fh.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        # 数据按网格顺序(x 最快)写入; 此处用包围盒网格
        lo = k_cart.min(axis=0)
        hi = k_cart.max(axis=0)
        n = np.maximum(np.ceil((hi - lo) / 1e-6).astype(int), 1)
        fh.write(f"{n[0]} {n[1]} {n[2]} 1\n")
        for k, v in zip(k_cart, values):
            fh.write(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f} {v:.8f}\n")


def write_grd(path: str, k_cart: np.ndarray, values: np.ndarray) -> None:
    """GNUPLOT ASCII grid (XCrySDen 可读)。"""
    lo = k_cart.min(axis=0)
    hi = k_cart.max(axis=0)
    with open(path, "w") as fh:
        fh.write("# kx ky kz value\n")
        fh.write(f"{lo[0]:.8f} {lo[1]:.8f} {lo[2]:.8f}\n")
        fh.write(f"{hi[0]:.8f} {hi[1]:.8f} {hi[2]:.8f}\n")
        fh.write(f"{len(values)} 1 1\n")
        fh.write(f"{values.min():.8f} {values.max():.8f}\n")
        for k, v in zip(k_cart, values):
            fh.write(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f} {v:.8f}\n")
```

Run: `python -m pytest tests/test_fsout.py -v`
Expected: PASS（cube 测试断言按上面实现写；若网格索引行格式不符，按实现调整断言）

- [ ] **Step 5: 按逆向结果实现 FermiSurfer band 与 bxsf 写出，并补测试**

依据 Step 1/2 记录的格式实现 `write_fermisurfer_bands` 与 `write_bxsf`，并在 `tests/test_fsout.py` 追加往返测试（写出→按记录的格式读回→逐值比对）。

- [ ] **Step 6: Commit**

```bash
git add unfoldfs/fsout.py tests/test_fsout.py docs/formats.md
git commit -m "feat: fsout 模块(cube/grd/bxsf/FermiSurfer band/npz 写出)"
```

---

### Task 8: cli.py — gen / unfold / info 与端到端

**Files:**
- Create: `unfoldfs/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: 全部上游模块
- Produces:
  ```python
  def cmd_gen(args) -> int
  def cmd_unfold(args) -> int
  def cmd_info(args) -> int
  def main(argv: list[str] | None = None) -> int
  ```
  CLI:
  ```bash
  unfoldfs gen    --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
                   --resolution 0.02 --grid-type gamma --outdir run/
  unfoldfs unfold --dir run/ --supercell ... --primcell ... --transmat ... \
                   [--ef auto|vbm|<数值>] [--delta 0.05] [--sigma 0.06] --outdir fs/
  unfoldfs info   --dir run/
  ```

- [ ] **Step 1: 写失败测试（gen 端到端）**

新建 `tests/test_cli.py`：
```python
import numpy as np
import pytest

from unfoldfs.cli import main
from tests.conftest import (
    FIXTURE_POSCAR, FIXTURE_PRIMCELL, FIXTURE_TRANSMAT, FIXTURE_EIGENVAL,
    FIXTURE_DOSCAR, FIXTURE_PROCAR,
)


def test_gen_writes_kpoints_and_grid_map(tmp_path):
    rc = main([
        "gen",
        "--supercell", str(FIXTURE_POSCAR),
        "--primcell", str(FIXTURE_PRIMCELL),
        "--transmat", str(FIXTURE_TRANSMAT),
        "--resolution", "0.2",
        "--grid-type", "gamma",
        "--outdir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "KPOINTS").is_file()
    assert (tmp_path / "grid_map.npz").is_file()
    weights = [float(l.split()[3]) for l in
               (tmp_path / "KPOINTS").read_text().splitlines()[3:]]
    assert sum(weights) > 0


def test_info_reports_reduction(tmp_path, capsys):
    main(["gen", "--supercell", str(FIXTURE_POSCAR), "--primcell", str(FIXTURE_PRIMCELL),
          "--transmat", str(FIXTURE_TRANSMAT), "--resolution", "0.2", "--outdir", str(tmp_path)])
    rc = main(["info", "--dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "约化" in out or "reduced" in out
```

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL（`unfoldfs.cli` 未定义）

- [ ] **Step 2: 实现 cli.py（gen 与 info）**

`unfoldfs/cli.py`：
```python
"""unfoldfs 命令行入口。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .expand import build_spectral_function, expand_to_full_grid
from .fsout import (write_cube, write_fermisurfer_bands, write_grd,
                     write_spectral_npz)
from .geometry import (candidate_g_vectors, common_symmetry_ops, read_poscar,
                       read_transmat, reciprocal, resolve_transform)
from .grid import (GridMap, choose_grid_points, decompose_grid,
                    filter_ops_by_grid, load_grid_map, primitive_grid_points,
                    save_grid_map, write_kpoints)
from .unfold import build_atom_copies, unfold_supercell
from .vaspio import read_doscar_fermi, read_eigenval, read_procar


def _load_geometry(args):
    prim = read_poscar(args.primcell)
    sc = read_poscar(args.supercell)
    m = read_transmat(args.transmat)
    tr = resolve_transform(sc, prim, m)
    return prim, sc, tr


def cmd_gen(args) -> int:
    prim, sc, tr = _load_geometry(args)
    spec = choose_grid_points(prim.lattice, tr.m_eff, args.resolution,
                               args.grid_type)
    grid = primitive_grid_points(spec)
    ops = common_symmetry_ops(prim, tr.m_eff, args.symprec)
    ops = filter_ops_by_grid(ops, grid)
    ir_idx, mult, full_to_ir, rot = decompose_grid(grid, ops)
    gm = GridMap(
        spec=spec,
        ir_k_prim=grid[ir_idx],
        ir_k_super=np.array([tr.super_from_prim(k) for k in grid[ir_idx]]),
        multiplicity=mult,
        full_to_ir=full_to_ir,
        rot_of_full=rot,
        ops=ops,
        transform=tr,
        b_prim=reciprocal(prim.lattice),
        b_super=reciprocal(sc.lattice),
    )
    os.makedirs(args.outdir, exist_ok=True)
    write_kpoints(os.path.join(args.outdir, "KPOINTS"), gm)
    save_grid_map(os.path.join(args.outdir, "grid_map.npz"), gm)
    print(f"[gen] 原胞网格 {spec.p} = {spec.n_points} 点; "
          f"对称操作 {ops.rotations.shape[0]} 个; "
          f"约化后 {gm.n_ir} 点 (约化率 {spec.n_points / max(gm.n_ir,1):.1f}x)")
    return 0
```

（`reciprocal` 已在上面的 import 列表中。）

`cmd_info`：
```python
def cmd_info(args) -> int:
    gm = load_grid_map(os.path.join(args.dir, "grid_map.npz"))
    print(f"[info] 原胞网格 {gm.spec.p} = {gm.spec.n_points} 点")
    print(f"[info] 代表点 {gm.n_ir} 个; 约化率 {gm.spec.n_points / max(gm.n_ir,1):.2f}x")
    print(f"[info] 对称操作 {gm.ops.rotations.shape[0]} 个 (原胞基底, 同时为超胞对称)")
    print(f"[info] 变换矩阵 M_eff = {gm.transform.m_eff.tolist()}, det = {gm.transform.det}")
    return 0
```

`main`：
```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="unfoldfs", description="超胞反折叠费米面工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="生成对称约化网格与 KPOINTS")
    g.add_argument("--supercell", required=True)
    g.add_argument("--primcell", required=True)
    g.add_argument("--transmat", required=True)
    g.add_argument("--resolution", type=float, required=True, help="2pi/Angstrom")
    g.add_argument("--grid-type", choices=["gamma", "mp"], default="gamma")
    g.add_argument("--symprec", type=float, default=1e-5)
    g.add_argument("--outdir", required=True)
    g.set_defaults(func=cmd_gen)

    u = sub.add_parser("unfold", help="反折叠并输出费米面文件")
    u.add_argument("--dir", required=True, help="含 grid_map.npz 与 VASP 输出的目录")
    u.add_argument("--supercell", required=True)
    u.add_argument("--primcell", required=True)
    u.add_argument("--transmat", required=True)
    u.add_argument("--ef", default="auto", help="auto | vbm | 数值")
    u.add_argument("--delta", type=float, default=0.05)
    u.add_argument("--sigma", type=float, default=0.06)
    u.add_argument("--outdir", required=True)
    u.set_defaults(func=cmd_unfold)

    i = sub.add_parser("info", help="打印网格与对称信息")
    i.add_argument("--dir", required=True)
    i.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

Run: `python -m pytest tests/test_cli.py -v`
Expected: 前两个 PASS，`cmd_unfold` 尚未实现（下一步补）

- [ ] **Step 3: 写失败测试（unfold 端到端）**

追加到 `tests/test_cli.py`：
```python
def test_unfold_end_to_end(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    main(["gen", "--supercell", str(FIXTURE_POSCAR), "--primcell", str(FIXTURE_PRIMCELL),
          "--transmat", str(FIXTURE_TRANSMAT), "--resolution", "0.2", "--outdir", str(run_dir)])
    # fixture 只有 2 个 k 点, 直接放入计算目录
    for f in ("EIGENVAL", "DOSCAR", "PROCAR"):
        (run_dir / f).write_bytes((FIXTURE_EIGENVAL.parent / f).read_bytes())
    fs_dir = tmp_path / "fs"
    rc = main(["unfold", "--dir", str(run_dir), "--supercell", str(FIXTURE_POSCAR),
               "--primcell", str(FIXTURE_PRIMCELL), "--transmat", str(FIXTURE_TRANSMAT),
               "--outdir", str(fs_dir)])
    assert rc == 0
    assert (fs_dir / "FS.grd").is_file()
    assert (fs_dir / "spectral_function.npz").is_file()
```

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL（`cmd_unfold` 未实现）

- [ ] **Step 4: 实现 cmd_unfold**

```python
def cmd_unfold(args) -> int:
    prim, sc, tr = _load_geometry(args)
    gm = load_grid_map(os.path.join(args.dir, "grid_map.npz"))
    band = read_eigenval(os.path.join(args.dir, "EIGENVAL"))
    procar = read_procar(os.path.join(args.dir, "PROCAR"))
    efermi = read_doscar_fermi(os.path.join(args.dir, "DOSCAR"))
    copies = build_atom_copies(prim, sc, tr)
    gset = candidate_g_vectors(tr.m_eff)

    # VASP 实际算出的 k 点 = KPOINTS 里的代表点(顺序一致)
    k_super_list = gm.ir_k_super
    nk_calc = min(len(k_super_list), procar.kpoints.shape[0], band.kpoints.shape[0])
    k_super_list = k_super_list[:nk_calc]
    if nk_calc != len(k_super_list):
        raise ValueError("VASP 输出的 k 点数与 grid_map 不一致")

    unfolded = unfold_supercell(
        band, procar, tr, copies, gset,
        a_prim=prim.lattice,
        prim_cart=prim.positions @ prim.lattice,
        b_prim=gm.b_prim, b_super=gm.b_super,
        k_super_list=k_super_list)
    expanded = expand_to_full_grid(unfolded, gm)

    # 费米能级
    vbm, cbm = _band_gap(band)
    ef = _resolve_ef(args.ef, efermi, vbm, cbm)
    print(f"[unfold] E_F = {efermi:.4f} eV; VBM = {vbm:.4f}; CBM = {cbm:.4f}")

    e_grid = np.linspace(np.nanmin(expanded.band_e) - 0.5,
                          np.nanmax(expanded.band_e) + 0.5, 1200)
    a = build_spectral_function(expanded, e_grid, args.sigma)

    os.makedirs(args.outdir, exist_ok=True)
    jf = int(np.argmin(np.abs(e_grid - ef)))
    write_grd(os.path.join(args.outdir, "FS.grd"), expanded.k_cart, a[:, jf])
    write_cube(os.path.join(args.outdir, "FS.cube"), expanded.k_cart, a[:, jf],
               gm.b_prim)
    write_fermisurfer_bands(args.outdir, expanded.k_cart, expanded.band_e,
                            expanded.band_w)
    write_spectral_npz(os.path.join(args.outdir, "spectral_function.npz"),
                        expanded, e_grid, a, {"ef": ef, "vbm": vbm, "cbm": cbm})
    print(f"[unfold] 输出 {args.outdir}: FS.grd / FS.cube / band*.dat / spectral_function.npz")
    return 0
```

配套辅助：
```python
def _band_gap(band) -> tuple[float, float]:
    occ = band.occupations
    e = band.energies
    filled = e[occ > 0.5]
    empty = e[occ <= 0.5]
    vbm = float(filled.max()) if filled.size else float("nan")
    cbm = float(empty.min()) if empty.size else float("nan")
    return vbm, cbm


def _resolve_ef(spec: str, efermi: float, vbm: float, cbm: float) -> float:
    if spec == "auto":
        if np.isfinite(vbm) and np.isfinite(cbm) and vbm < efermi < cbm:
            return vbm
        return efermi
    if spec == "vbm":
        return vbm
    return float(spec)
```

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: 提交并在 master 上做真实端到端验证**

```bash
git add unfoldfs/cli.py tests/test_cli.py
git commit -m "feat: cli 子命令(gen/unfold/info) 与端到端串联"
```

master 验证（用 MoS2 3×3×1 的真实 100 k 点网格结果）：
```bash
cd /data/wjlu/work/claude/unfold_fs
# 同步代码
rsync -av --exclude '.git' F:/cc-haha_projects/unfold_band_fs/unfoldfs/ \
    /data/wjlu/work/claude/unfold_fs/unfoldfs/
cd /data/wjlu/work/claude/unfold_fs
/data/wjlu/program/anaconda3/bin/python3 -m pytest tests -v
```

- [ ] **Step 6: Commit（真实运行记录）**

把 master 端 pytest 结果与真实 FS 图（FS.grd 用 XCrySDen/FermiSurfer 打开）记入 `docs/algorithms.md` 的「实测」章节。

---

### Task 9: 文档

**Files:**
- Create: `docs/algorithms.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部模块的实际行为
- Produces: 无代码接口

- [ ] **Step 1: 写 `docs/algorithms.md`**

必须包含：
1. 折叠关系的推导与行/列约定探测的依据
2. 展开公式（Popescu）各项的物理含义，以及「副本平均」为何能挑出正确的 G
3. 相位符号约定 `s1/s2` 的含义与自检判据
4. 对称操作取交集的判据 `M⁻ᵀRMᵀ ∈ Z³` 的理由（spglib 的 R 是**列向量**约定；**不要**用平移做筛选——spglib 给的 t 是模原胞格矢的代表元，会误删操作）
5. 候选索引置换展开的正确性说明
6. 实测记录（fixture 的 VASP 6.6.0 参数、单 G 主导度、带隙、与 vaspkit 282 的对照结果）

- [ ] **Step 2: 重写 `README.md`**

包含：安装、三个子命令的用法、典型工作流（gen → 提交 VASP → unfold）、输出文件说明、依赖版本、常见报错与对策。

- [ ] **Step 3: 全量测试并提交**

```bash
python -m pytest tests -v
git add docs/algorithms.md README.md
git commit -m "docs: 算法说明与使用文档"
```

---

## 计划自审记录

**Spec 覆盖检查**

| Spec 章节 | 对应任务 |
|---|---|
| §3.1 模块边界 | Task 1-8（每模块一个任务） |
| §3.2 CLI | Task 8 |
| §4.1 grid_map.npz | Task 3 |
| §4.2 unfolded.npz | Task 5/6 |
| §5.1 公共对称集 S | Task 2 |
| §5.2 网格生成与约化 | Task 3 |
| §5.3 反折叠 | Task 5 |
| §5.4 对称扩展 | Task 6 |
| §5.5 费米能级处理 | Task 8 |
| §6 输出格式 | Task 7 |
| §7 错误处理 | Task 2/4/5/8（各模块内含） |
| §8 测试策略 | Task 2-8 的测试步骤 + Task 8 端到端 |
| §9 已验证事实 | Task 1（fixture）+ Task 5（基线） |

**待实现阶段现场核对的两项**（已在计划中显式安排为"先逆向再实现"，非占位符）
- Task 7 Step 1/2：FermiSurfer band 文件与 vaspkit 262 的 FS.bxsf 的确切字节格式
