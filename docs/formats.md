# 输出格式规范（unfoldfs 费米面文件）

> 本文件是 `unfoldfs/fsout.py` 的**权威格式规范**。每条规范都注明了出处
> （实测样例 / 工具源码），代码按本文件实现；`tests/test_fsout.py` 里的
> `test_formats_doc_spec_matches_written_files` 会把下面 `spec` 块解析出来，
> 逐条核对实际写出的文件 —— 文档与实现不一致时测试会失败。
>
> 核对环境：服务器 `<server>`（master），2026-11-17。
> 参考物：XCrySDen 1.5.60（`$TOOLSDIR/xcrysden-1.5.60-bin-semishared`）、
> FermiSurfer 2.2.1（`$TOOLSDIR/fermisurfer-2.2.1`）、
> VASPKIT 1.5.1（`$TOOLSDIR/vaspkit.1.5.1`）。

## 0. 全局约定

| 项 | 约定 | 说明 |
|---|---|---|
| k 单位 | **2π/Å**（含 2π） | 与 `unfoldfs.geometry.reciprocal` 一致；`b_prim` 的行 = 倒矢量 |
| E 单位 | **eV** | 绝对能量（相对真空零点），E_F 另列 |
| 网格 | 原胞的**均匀网格**，点数 = `spec.n_points`（满网格点数） | **不是**超胞 k 点数 `n_k` |
| 补位槽 | `valid == False` 的槽（能量 `+inf`、权重 0）**绝不进入任何输出** | 见 §6 |
| 网格下标 | `i1/i2/i3` 对应倒格矢 `b1/b2/b3` 方向 | 与 `expand.ExpandedData.k_cart` 的分数坐标一致 |

## 1. 机器可读规范

```spec
cube.header_lines = 2
cube.data_per_line = 6
cube.axis_order = third-axis-fastest
cube.vector_kind = increment
cube.vector_unit = Bohr
grd.header_order = nx ny|xmin xmax|ymin ymax|zmin zmax
grd.slice_separator = blank line
bxsf.block = BEGIN_BLOCK_BANDGRID_3D
bxsf.grid_block = BEGIN_BANDGRID_3D
bxsf.values_per_line = 6
bxsf.grid_dims = unique+1
bxsf.axis_order = third-axis-fastest
frmsf.lshift_gamma = 1
frmsf.lshift_mp = 2
frmsf.matrix_axes = 3
frmsf.grid_dims = unique
frmsf.axis_order = third-axis-fastest
```

## 2. `FS.cube` —— Gaussian cube（消费者：VESTA / XCrySDen / VMD）

**出处**：Quantum ESPRESSO 7.0 的 cube 写出器
`$TOOLSDIR/q-e-qe-7.0/PP/src/cube.f90`（文件头注释即格式规范，
并注明 "LAST CHECKED AGAINST GAUSSIAN 98"），以及第三方读取器
ASE 3.26 的 `ase/io/cube.py::read_cube`。

**布局**：

```
 <注释行 1>                       # 单位/约定写在这里
 <注释行 2>
    0    0.000000    0.000000    0.000000      # natoms = 0, 原点
 <n1> <v1x> <v1y> <v1z>                          # 增量矢量(见下)
 <n2> <v2x> <v2y> <v2z>
 <n3> <v3x> <v3y> <v3z>
 <体数据: n1*n2*n3 个, 每行 6 个>
```

三条**实测确定**的约定（每条都有源码/实测出处）：

1. **矢量行写的是"每格点的增量矢量"，不是整条晶格矢量**
   （QE `cube.f90` 的格式注释：`4-6  I5,3F12.6  #GRIDPOINTS, INCREMENT VECTOR`）。
   读取器据此把盒子还原成 `n_i × v_i`（ASE：`cell[i] = n * Bohr * vec[i]`）。
   ⇒ 本项目写 `v_i = b_i / n_i`，于是盒子正好是 b1,b2,b3 张成的 **BZ**，
   画出来的等值面就是 k 空间里的费米面。
2. **长度单位是原子单位（Bohr）**（QE `cube.f90`：
   "ALL COORDINATES ARE GIVEN IN ATOMIC UNITS"；ASE 也按 Bohr 解析）。
   ⇒ 本项目把 `b_i/n_i` 除以 `0.529177210903` 再写。体数据 `A(k,E_F)`
   无量纲，不受影响。
3. **存储顺序 = 第 3 个矢量最快**（数据线性下标
   `i1·n2·n3 + i2·n3 + i3`）。QE 的写循环是
   `DO i1 / DO i2 / WRITE (rho(i1,i2,i3), i3=1,nr3)`，ASE 的读取器是
   `data.reshape((n1,n2,n3))`（C 序 ⇒ i3 最快）✓ 两者一致。

其余细节：数值格式 `6E13.5`（每值 `%13.5E`，每行 6 个）——与 QE 完全一致；
格点位置 `origin + (i1/n1)·b1 + (i2/n2)·b2 + (i3/n3)·b3`，与展开结果的
均匀网格逐点对应（`i` 取 0…n-1，不含重复的周期端点）。

> ⚠ **VASP 的 `CHGCAR` 不是这个约定**（它写整条晶格矢量，且是 VASP 专有
> 格式），别拿 CHGCAR 当 cube 的参考。EPW 写出的 cube 也有自己的扩展头
> （多一行网格信息），同样不能当参考。

**实测核对（2026-11-17）**：用本模块写一个"值 = 第一条网格轴下标"的探针
cube（4×3×2，故意用非正交倒格矢），用服务器上的 **ASE 3.26
`ase.io.cube.read_cube`** 读回：

```
data[i1,i2,i3] = [[0 0],[0 0],[0 0]] , [[1 1],...] , [[2 2],...] , [[3 3],...]
ASE 还原的盒子(Å⁻¹) = [[2,0,0],[1,1.5,0],[0.3,0.2,1.8]]   ← 与写入的 b_prim 逐位相同
```

即：轴序、增量约定、Bohr 单位三条同时被第三方读取器验证 ✓。

**倒易矢量约定核对**：用 demo 的 `POSCAR` 算出的 `2π·A⁻ᵀ` 与 VASPKIT 262
生成的 `FERMISURFACE.bxsf` 里的三行矢量**逐位相同**
（`0.656466561 0.379011146 0` / `0 0.758022291 0` / `0 0 0.355815763`），
即本项目用 `b_prim`（含 2π）作为"假盒子"是与 VASPKIT 一致的。

## 3. `FS.grd` —— GNUPLOT ASCII grid（消费者：XCrySDen 的 grid 导入）

**出处**：简报指定的格式族（GNUPLOT ASCII grid）；XCrySDen 1.5.60 的
`bin/xcrys` 内含 `cryGeneralGridRegPeriodInterpolator` 等 grid 读取例程，
`Tcl/parseDataGrid.tcl` 是其 “DataGrid” 展示脚本。

**布局**：

```
 # 注释行（单位/约定）
 <nx> <ny>
 <xmin> <xmax>
 <ymin> <ymax>
 <zmin> <zmax>
 <z 切片 1: ny 行, 每行 nx 个值, x 最快>
 <空行>
 <z 切片 2: ...>
```

* `nx`/`ny` 是前两个网格方向的点数；**nz 不写在头部**，由空行分隔的
  切片数给出。
* 三个坐标轴 = 倒格矢 `b1/b2/b3` 方向上的**笛卡尔长度**（2π/Å）。
  gnuplot 的 grid 格式只支持正交轴，非正交 BZ 的形变无法避免 ——
  需要保持 BZ 形状请用 `FS.cube` 或 `FS.bxsf`。
* 坐标区间取**实际采样范围**（第 n 个点在 `(n-1)/n` 处），与 gnuplot
  按“区间等分”插值的口径一致。
* 体数据 = `A(k, E_F)`。

> 注：VASPKIT 也会写 `.grd`，但那是 **k 路径剖面**（两列：路径距离 + 值），
> 与本文件的三维场格式**不是一回事**，不能混用。

## 4. `FS.bxsf` —— Band-XCRYSDEN-Structure-File（消费者：XCrySDen / FermiSurfer）

**出处**（三份互相独立的一致证据）：

1. XCrySDen 自带样例
   `$TOOLSDIR/xcrysden-1.5.60-bin-semishared/examples/FermiSurface/MgB2.bxsf.gz`；
2. FermiSurfer 自带样例
   `$TOOLSDIR/fermisurfer-2.2.1/examples/pb.bxsf`；
3. VASPKIT 262 的输出 `/tmp/fmt/FERMISURFACE.bxsf`；
   以及 FermiSurfer 的读取源码 `src/read_file.cpp::read_bxsf`。

**布局**：

```
 BEGIN_INFO
   # 注释行（单位/约定）
 Fermi Energy:     <E_F>
 END_INFO
 BEGIN_BLOCK_BANDGRID_3D
 band_energies
 BEGIN_BANDGRID_3D
 <带数>
 <n1> <n2> <n3>
 <原点 x y z>
 <b1x b1y b1z>
 <b2x b2y b2z>
 <b3x b3y b3z>
 BAND: <带号, 从 1 起>
 <该带的能量值 ...>
 ...
 END_BANDGRID_3D
 END_BLOCK_BANDGRID_3D
```

**逐条核对到的细节**：

* `Fermi Energy:` 那一行必须恰好是 **两个词 + 一个数**
  （FermiSurfer 用 `sscanf("%s %s %lf")` 解析）。
* 三个矢量行是**倒格矢**（FermiSurfer 读成 `bvec`，XCrySDen 的
  `fsReadBXSF` 把它们输出为 `RECIP-PRIMVEC`，并由其求逆得到
  `PRIMVEC`）✓ 与 §2 的约定一致。
* **网格维度 = 唯一点数 + 1**：文件里存 `(n1+1)(n2+1)(n3+1)` 个值，
  最后一个平面/行/列是第一个的重复（周期端点）。
  FermiSurfer 的读取器 `ng0[ii] -= 1` 后按 `(ng0+1)³` 读，XCrySDen 的
  MgB2 样例（`13 13 10` × 1690 个值）也满足这个关系 ✓。
* **存储顺序 = 第 3 个轴最快**（实测：对 XCrySDen 的 MgB2 样例做
  周期端点相等性检验，只有 `(n1, n2, n3)` 的 C 序展开在三个轴上
  全部满足“首末相等”）。
* 能量是**绝对值**（eV）：VASPKIT 写出的第一条带能量 `-16.178176`
  与 fixture `EIGENVAL` 的最低带能量 `-16.202533` 同量级同符号，
  而非平移后的值；FermiSurfer 的 `read_bxsf` 自己会 `eig -= EF`。
* `BEGIN_BANDGRID_3D` 这个标记是 **XCrySDen 与 FermiSurfer 都接受**的
  写法（VASPKIT 262 的输出用它，被 XCrySDen 自带的 `fsReadBXSF`
  成功读出等值面顶点 ✓）。XCrySDen 自己写文件时用的是
  `BANDGRID_3D_BANDS`，本项目统一用 `BEGIN_BANDGRID_3D`。
* 每个带一段，数值每行 6 个（实测样例即如此）。

**本项目写入的内容**：逐带 —— 第 n 段是**展开后第 n 条带**在均匀网格上的
能量 `E_n(k)`（绝对 eV）。补位槽的处理见 §6。

## 5. `band.N` —— FermiSurfer 逐带文件（消费者：FermiSurfer）

**出处**：FermiSurfer 2.2.1 的源码
`src/read_file.cpp::read_file()`（读 `.frmsf`）、`src/read_bxsf()`（读 `.bxsf`）
与 `src/fermisurfer.cpp`（按扩展名分派）。

**重要修正**：简报设想的“逐带 (kx, ky, kz, E) 文本文件”**在 FermiSurfer
2.2.1 里没有读取器** —— 该版本只支持 `.frmsf` 与 `.bxsf` 两种输入。
因此 `band.N` 按源码写成 `.frmsf` 格式（每文件一条带），FermiSurfer 可直接
打开。

**布局**（`read_file()` 的读取顺序）：

```
 <n1> <n2> <n3>      网格点数(不含重复的周期端点)
 <lshift>            0 = MP 网格; 1 = Gamma 居中; 2 = Gamma + 半格
 <nb>                带数(逐带文件恒为 1)
 <b1x b1y b1z>
 <b2x b2y b2z>
 <b3x b3y b3z>
 <能量: n1*n2*n3 个, 第 3 个轴最快>
 <矩阵元: 3 个轴 × n1*n2*n3 个>
```

* `lshift`：本项目 Gamma 网格写 `1`，MP 网格写 `2`（两种情况下
  FermiSurfer 都按 `ii = i` 直读，索引不变换）。
* 存储顺序 = 第 3 个轴最快（`for i0 → for i1 → for i2`）✓ 与 §4 一致。
* 能量是**绝对值**；FermiSurfer 从批处理文件（`frmsf.in`）读
  `fermienergy` 来定等值面 —— 打开本文件时需把 `fermienergy` 设成本次
  计算的 E_F（或用 `FS.bxsf`，它把 E_F 写在文件里）。
* **矩阵元**：FermiSurfer 在能量之后还会读 3 个轴的矩阵元（用于费米速度
  着色）。本模块没有速度数据，写 **0**（确定值；不留未初始化内存给
  FermiSurfer）。文件里共有 `n1·n2·n3·(1+3)` 个数。
* 矢量行：FermiSurfer 只用它们画 BZ 形状（整体比例任意）。本模块与
  §2/§4 保持一致，写 `b_prim`（2π/Å）。

## 6. 补位槽（`valid == False`）在逐带输出里的填充规则

补位槽表示“该候选在这个 k 点上没有第 n 条带”（能量 `+inf`、权重 0）。
BXSF / FermiSurfer 的网格是**规则**的，无法留空，故按下面规则填：

* 填**同一 k 点上下一条有效带的能量**；若没有下一条，则填该点**最高
  有效带**的能量。
* 这样跨过有效/无效边界时场是连续的，**不会凭空多出一张等值面**。
* 权重视为 0（该态不存在），故不进入谱函数。

## 7. `A(k,E).npz` —— 完整谱函数（自用 / 二次分析）

`numpy.savez_compressed`，键：

| 键 | 内容 |
|---|---|
| `A` | (n_points, n_E) 谱函数 `A(k,E) = Σ_n w_n(k)·exp(−(E−E_n)²/2σ²)` |
| `e_grid` | (n_E,) 能量网格 (eV)，用 `make_e_grid` 构造时**恰好含 E_F** |
| `k_cart` | (n_points, 3) 笛卡尔 k (2π/Å) |
| `band_e` / `band_w` / `valid` | 逐带能量 / 权重 / 有效位掩码（原始展开数据） |
| `ef` / `vbm` / `cbm` | 费米能 / 价带顶 / 导带底 (eV) |
| `sigma` | 展宽 (eV) |
| `k_units` | `"2pi/Angstrom"` |
| `e_units` | `"eV"` |

## 8. 费米能（E_F）选取

| 模式 | 语义 |
|---|---|
| `auto`（默认） | 金属 → DOSCAR 的 `E_F`；半导体/绝缘体 → VBM（可加偏移） |
| `vbm` | 总是 VBM + 偏移 |
| `doscar` | 总是 DOSCAR 的 `E_F` |
| 数值 | 直接当作 E_F（绝对值） |

* **VBM/CBM** 由 EIGENVAL 的占据数给出：占据判据为 `occ > 0.5`，
  VBM = 占据态最高能量，CBM = 未占据态最低能量。
* **金属判据**：带隙 ≤ `gap_tol`（默认 0.05 eV）**或**存在分数占据态
  （占据数严格落在 0 与 1 之间）。后者在 smearing 较大、表观带隙不可信时
  仍能判对。

## 9. 复核记录（2026-11-17）

* FermiSurfer 2.2.1 的 `src/read_file.cpp`、`src/fermisurfer.cpp` 通读；
  `examples/mgb2_vfz.frmsf`、`examples/pb.bxsf` 逐字节查看。
* XCrySDen 1.5.60 的 `examples/FermiSurface/MgB2.bxsf.gz` 解压后逐行查看，
  并用 numpy 做了周期端点/存储顺序的判别性检验。
* VASPKIT 1.5.1 的 `-task 262` 在 demo 目录实跑，产出 `FERMISURFACE.bxsf`
  与 `FERMI_ENERGY`，与 `POSCAR` 的倒格矢逐位比对。
* XCrySDen 自带的 `bin/fsReadBXSF` 实跑：能读出 VASPKIT 262 的
  `FERMISURFACE.bxsf`，也能读出**本模块写出的 `FS.bxsf`**，并给出
  正确的 BZ 几何（六方 BZ 的 8 个面 ✓）。
* **BXSF 逐位往返**：把 XCrySDen 自带的 `MgB2.bxsf` 读进来、用本模块的
  `write_bxsf` 写回去，三个带的全部数值**偏差 0.000e+00 eV**，网格维度
  (13,13,10)、E_F、带数全部一致 ✓ —— 说明本模块的存储顺序、维度约定
  （唯一点+1 + 周期端点重复）、矢量行顺序与 XCrySDen 自己的写法完全一致。
* **cube 实测**：探针 cube 用 **ASE 3.26 `ase.io.cube.read_cube`** 读回，
  轴序/增量约定/Bohr 单位三条同时验证 ✓（见 §2）。
* **grd 实测**：本模块写出的 `FS.grd` 用服务器上的 **gnuplot 5.2** 解析
  无错；按 gnuplot grid 格式手工解析也得到 (nz, ny, nx) 的体数据 ✓。
* `bin/cube2xsf` 在服务器上**对参考 cube 也报同样的错**（工具自身损坏），
  故未用作 cube 的验证器。
