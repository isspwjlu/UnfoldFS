# 输出格式规范（unfoldfs 费米面文件）

> 本文件是 `unfoldfs/fsout.py` 的**权威格式规范**。每条规范都注明了出处
> （实测样例 / 工具源码），代码按本文件实现；`tests/test_fsout.py` 里的
> `test_formats_doc_spec_matches_written_files` 会把下面 `spec` 块解析出来，
> 逐条核对实际写出的文件 —— 文档与实现不一致时测试会失败。
>
> 核对环境：服务器 `210.73.16.15`（master），2026-11-17。
> 参考物：XCrySDen 1.5.60（`/data/wjlu/program/xcrysden-1.5.60-bin-semishared`）、
> FermiSurfer 2.2.1（`/data/wjlu/program/fermisurfer-2.2.1`）、
> VASPKIT 1.5.1（`/data/wjlu/program/vaspkit.1.5.1`）。

## 0. 全局约定

**交付面 = 4 种格式**：

| 文件 | 消费者 | 内容 |
|---|---|---|
| `FS.cube` | VESTA / XCrySDen / VMD | Gaussian cube：假晶格 = 倒格矢，体数据 = A(k, E_F) |
| `FS.bxsf` | XCrySDen / FermiSurfer | 逐带能量网格（**只支持 Γ 网格**，见 §3） |
| `band.N` | FermiSurfer | FermiSurfer 的 `.frmsf` 逐带文件（能量**相对 E_F**，见 §4） |
| `A(k,E).npz` | 自用/二次分析 | 完整谱函数体数据 + 元信息 |

> **原计划的 `FS.grd` 已删除**（2026-11-18）。它**没有任何标准读取器**：
> XCrySDen 1.5.60 全树没有 `.grd` 代码路径（`grep -rIl grd` 只命中一句英文
> 注释与无关的 Tcl 变量 `newgrd`；`bin/*` 的 strings 命中数全为 0；`usage`
> 里没有 `--grd`）；gnuplot 5.2 的 `splot` **直接报错**
> （`2 columns only possible with explicit pm3d style (line 3)`）—— 它自己的
> 手册里根本没有"带头部的三维网格"这种格式。简报 v2 要求"不要凭空设计
> 格式"，故这一格交付物被**删掉**，而不是保留一个谁也不读的文件。
> 二次分析的需求由 `A(k,E).npz` 覆盖。

| 项 | 约定 | 说明 |
|---|---|---|
| k 单位 | **2π/Å**（含 2π） | 与 `unfoldfs.geometry.reciprocal` 一致；`b_prim` 的行 = 倒矢量 |
| E 单位 | **eV** | `FS.cube` / `FS.bxsf` 写**绝对**能量；`band.N` 写**相对 E_F** 的能量（见 §4） |
| 网格 | 原胞的**均匀网格**，点数 = `spec.n_points`（满网格点数） | **不是**超胞 k 点数 `n_k` |
| 补位槽 | `valid == False` 的槽（能量 `+inf`、权重 0）**绝不进入任何输出** | 见 §5 |
| 网格下标 | `i1/i2/i3` 对应倒格矢 `b1/b2/b3` 方向 | 与 `expand.ExpandedData.k_cart` 的分数坐标一致 |

## 1. 机器可读规范

```spec
cube.header_lines = 2
cube.data_per_line = 6
cube.axis_order = third-axis-fastest
cube.vector_kind = increment
cube.vector_unit = Bohr
cube.isolevel_hint = half-of-field-max
bxsf.block = BEGIN_BLOCK_BANDGRID_3D
bxsf.grid_block = BEGIN_BANDGRID_3D
bxsf.values_per_line = 6
bxsf.grid_dims = unique+1
bxsf.axis_order = third-axis-fastest
bxsf.supported_grid = gamma-only
frmsf.lshift_gamma = 1
frmsf.lshift_mp = 2
frmsf.matrix_axes = 3
frmsf.grid_dims = unique
frmsf.axis_order = third-axis-fastest
frmsf.energy_origin = relative-to-ef
padding.fill_rule = highest-valid-band
```

## 2. `FS.cube` —— Gaussian cube（消费者：VESTA / XCrySDen / VMD）

**出处**：Quantum ESPRESSO 7.0 的 cube 写出器
`/data/wjlu/program/q-e-qe-7.0/PP/src/cube.f90`（文件头注释即格式规范，
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

**头部给出的等值面层级建议**（`cube.isolevel_hint = half-of-field-max`）：
第 2 行注释里带体数据的最大值与建议层级，例如

```
 field max = 3.700e-01; use isosurface level ~ 1.850e-01
```

VESTA 的默认 **isolevel = 0** —— "画不画得出来"只取决于高斯尾巴是否
**恰好**下溢到 `0.0`（实测：换个 sigma 或能量窗口就可能什么都没有）。
故头部必须写明该选什么层级；建议值取最大值的一半，落在 `(0, max)` 内。
头部只有两行（cube 格式如此），单位说明与这一条挤在第 2 行。

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

## 3. `FS.bxsf` —— Band-XCRYSDEN-Structure-File（消费者：XCrySDen / FermiSurfer）

**出处**（三份互相独立的一致证据）：

1. XCrySDen 自带样例
   `/data/wjlu/program/xcrysden-1.5.60-bin-semishared/examples/FermiSurface/MgB2.bxsf.gz`；
2. FermiSurfer 自带样例
   `/data/wjlu/program/fermisurfer-2.2.1/examples/pb.bxsf`；
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
  周期端点相等性检验，**在排除前两轴互换之后**，只有"第 3 轴最快"的 C 序
  展开在三个轴上全部满足"首末相等"。注意 MgB2 样例的 `n1 == n2`，前两轴
  本来就不可区分 —— 故可区分的结论只有"第 3 轴最快"这一条）。
* 能量是**绝对值**（eV）：VASPKIT 写出的第一条带能量 `-16.178176`
  与 fixture `EIGENVAL` 的最低带能量 `-16.202533` 同量级同符号，
  而非平移后的值；FermiSurfer 的 `read_bxsf` 自己会 `eig -= EF`。
* `BEGIN_BANDGRID_3D` 这个标记是 **XCrySDen 与 FermiSurfer 都接受**的
  写法（VASPKIT 262 的输出用它，被 XCrySDen 自带的 `fsReadBXSF`
  成功读出等值面顶点 ✓）。XCrySDen 自己写文件时用的是
  `BANDGRID_3D_BANDS`，本项目统一用 `BEGIN_BANDGRID_3D`。
* 每个带一段，数值每行 6 个（实测样例即如此）。

**只支持 Γ 网格**（`bxsf.supported_grid = gamma-only`）：FermiSurfer 的
`read_bxsf()` 结尾写死 `for (ii = 0; ii < 3; ii++) shiftk[ii] = 0;` ——
**一律按 Γ 网格解释**（而 `.frmsf` 路径有 MP 处理：`lshift == 2` ⇒ 半格
偏移）。MP 网格的数据（第 i 个点在 `(i+0.5)/n`）会被放到 `i/n`，整张费米面
相对 BZ **平移半个倒格矢** ⇒ 静默的错误费米面。
⇒ `write_bxsf` 遇到 MP 网格**直接报错**（消息里给出可操作的替代方案）。
XCrySDen 侧的 BXSF 读取器同样假定 Γ（其源码不在本机，无法核对）。

**本项目写入的内容**：逐带 —— 第 n 段是**展开后第 n 条带**在均匀网格上的
能量 `E_n(k)`（绝对 eV）。补位槽的处理见 §5。

## 4. `band.N` —— FermiSurfer 逐带文件（消费者：FermiSurfer）

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
* 存储顺序 = 第 3 个轴最快（`for i0 → for i1 → for i2`）✓ 与 §3 一致。
* **能量是相对 E_F 的（E − E_F）**（`frmsf.energy_origin = relative-to-ef`）：
  FermiSurfer 从批处理文件（`frmsf.in`）读 `fermienergy` 来定等值面，
  缺省是 `0.0`（`fermisurfer.cpp:219`）；它自带的三个官方样例
  `mgb2_vfz`（`[-0.428, +0.498]`）、`pb_vf3D`（`[-4.28, +7.50]`）、
  `srvo3_orb`（`[-0.0396, +0.0522]`）的能量**全部以 0 为中心**，配套的
  `frmsf.in` 也写 `fermienergy 0.0`。
  ⇒ 写绝对能量 + FermiSurfer 默认 `EF = 0` ⇒ 打开画不出费米面（**静默
  失败**：文件能打开、不报错，画出来的东西不是费米面）。
  ⇒ 本模块写 `E − E_F`，用户**直接打开即可**，不需要另配 `fermienergy`。
  （与 §3 的 `FS.bxsf` 不同：BXSF 的 INFO 块里带着 E_F，FermiSurfer 的
  `read_bxsf` 自己会 `eig -= EF`，所以 BXSF 写绝对值。）
* **矩阵元**：FermiSurfer 在能量之后还会读 3 个轴的矩阵元（用于费米速度
  着色）。本模块没有速度数据，写 **0**（确定值；不留未初始化内存给
  FermiSurfer）。文件里共有 `n1·n2·n3·(1+3)` 个数。
* 矢量行：FermiSurfer 只用它们画 BZ 形状（整体比例任意）。本模块与
  §2/§3 保持一致，写 `b_prim`（2π/Å）。
* ⚠ **`.frmsf` 放不下注释行**（第一行就被 `sscanf("%d%d%d")` 读走，加注释
  会直接读崩）⇒ `band.N` **没有头部注释**，与简报"每个文件头部写明单位"
  的要求不符。单位与能量口径以本节与 `A(k,E).npz` 为准。

## 5. 补位槽（`valid == False`）在逐带输出里的填充规则

补位槽表示“该候选在这个 k 点上没有第 n 条带”（能量 `+inf`、权重 0）。
BXSF / FermiSurfer 的网格是**规则**的，无法留空，故按下面规则填
（`padding.fill_rule = highest-valid-band`）：

* 填**该点最高有效带的能量**；该点一条有效带都没有时填 **0**。
  （实现上取"下标更大且有效的带"优先，但 `unfold` 产生的补位槽恒为**连续
  尾部** ⇒ 不存在这样的带 ⇒ 实际恒等于"最高有效带"。）
* 权重视为 0（该态不存在），故不进入谱函数。
* **实话**：补位带与最高有效带**重合** ⇒ 若最高带在某些 k 上跨过 E_F，
  那里会出现若干张**重合的**等值面（不是"多出一张"，也不是"不多出"）。

## 6. `A(k,E).npz` —— 完整谱函数（自用 / 二次分析）

`numpy.savez_compressed`，键：

| 键 | 内容 |
|---|---|
| `A` | (n_points, n_E) 谱函数 `A(k,E) = Σ_n w_n(k)·exp(−(E−E_n)²/2σ²)`；**默认已按点群轨道对称化**（见 §6c） |
| `e_grid` | (n_E,) 能量网格 (eV)，用 `make_e_grid` 构造时**恰好含 E_F** |
| `k_cart` | (n_points, 3) 笛卡尔 k (2π/Å) |
| `k_frac` | (n_points, 3) 分数坐标 k（`k_cart · b_prim⁻¹`），便于二次分析 |
| `band_e` / `band_w` / `valid` | 逐带能量 / 权重 / 有效位掩码（**原始**展开数据，**不做**对称化，见 §6c） |
| `ef` / `vbm` / `cbm` | 费米能 / 价带顶 / 导带底 (eV) |
| `sigma` | 展宽 (eV) |
| `k_units` | `"2pi/Angstrom"` |
| `e_units` | `"eV"` |

⚠ **`band_w` 是"多源求和"之后的结果**（修复轮 8）：同一个原胞网格点若被多个
超胞 k 覆盖（超胞 k 列表覆盖整个原胞第一布里渊区时才会发生），它们的带按
**能量对齐**后合并、权重相加。当前 `unfoldfs gen` 写出的 KPOINTS 是**折叠网格**
⇒ 每个网格点恰好一个源 ⇒ `band_w` 与单源展开逐位相同
（机制见 `docs/algorithms.md` §8.7）。

## 6b. 两种模式：**超胞展开** vs **原胞直通** —— 谱权重不是同一个定义

`unfoldfs unfold` 有两条互斥的路径，产出的 `A(k,E)`（以及由它导出的 `FS.cube` /
`FS.bxsf` / `band.N`）**权重定义不同**，形状**不可直接比较**。

| | **超胞展开模式**（`det(M) > 1`） | **原胞直通模式**（`det(M) == 1`） |
|---|---|---|
| 什么时候走这条 | "超胞"的 VASP 计算要反折叠回原胞 | **原胞自己**的 VASP 计算（TRANSMAT = 单位阵） |
| 谱权重 `band_w` | Popescu 的 **相干** G 投影 `\|Σ_i e^{i k'·r_i} Σ_lm p_i(lm)\|²` | **轨道投影** `w_n(k) = Σ_{ion,lm} \|c_{ion,lm,n}(k)\|²`（**非相干**和） |
| 相位约定 s1/s2 | `s1` 由物理唯一确定（−1）+ `s2` 靠单 G 集中度择优（见 `docs/algorithms.md` §7.2） | **不存在**（`DominanceReport` 是 NaN 填充的"不适用"） |
| `band_e` | 超胞带能量，按候选归属重组 | EIGENVAL 的带能量，逐 k 按能量升序 |
| 与 vaspkit 282 | **不同**（见 §13.5） | **同一个量**（都是原胞带的 \|c(k)\|²） |

`det(M) == 1` 时候选 G 只有一个 ⇒ 4 种相位约定的主导度必然相同 ⇒ 择优守卫会拒掉
原胞自己的计算 ⇒ 工具**改走直通分支**（`unfold._unfold_primitive_direct`）。

⚠ **两种模式的"费米面"形状不可直接比较**（相干 vs 非相干）；可直接比较的是
E_F、带能量、能带走向与拓扑。

⚠ **两种模式的谱权重都受 C1 缺陷影响**（原始权重在晶体点群下**不协变**，
畸变可达 100 %）—— 详见 `docs/algorithms.md §15`。默认输出已做**轨道对称化**
（§6c）⇒ 等值面在超胞保留的对称操作下对称；但**绝对**形状（面积、半径）仍是
投影后的结果，不可靠。可靠的是能带能量、E_F 的选取与费米面的拓扑。

## 6c. ★ 默认输出是**对称化后**的（点群轨道平均），`--no-symmetrize` 关掉

`unfoldfs unfold` 与 `unfoldfs band` **默认**把展开谱按**原胞完整点群**的轨道对称化
（`unfoldfs/expand.py::symmetrize_spectral_function`、
`unfoldfs/band.py::symmetrize_band_weights`）：

```
轨道(i) = { j : j = R⁻¹·i + t 落在网格上, R ∈ 原胞完整点群 }   (去重)
A_sym(i, E) = mean_{j ∈ 轨道(i)} A(j, E)
```

**为什么**：展开得到的谱权重在晶体点群下**天然不协变**（对称相关的两个 k 点
给出的谱函数可以差到 100 %）⇒ 费米面的等值面相对晶体对称性是畸变的。
对称化把展开谱**投影到对称不变子空间**（原胞的谱函数本来就该是对称的 ⇒
这个投影是无偏的）。机制与实测见 `docs/algorithms.md §15`。

★ **用的是原胞的完整点群**（不是"原胞与超胞的公共对称操作"）——
 修复轮 6 的改动（`docs/algorithms.md §15.8`）：原胞的谱函数按构造就在原胞
 点群下不变 ⇒ 投影有物理依据；且**任何**超胞（含破坏面内 C3 的 3×1×1）都能
 折回原胞的对称性（实测 3×1×1 的 1T-TaS₂：`A(k,E_F)` 的轨道离散度
 1.000e+00 → **0.000e+00**）。

### 各输出在默认（对称化）与 `--no-symmetrize` 下分别是什么

| 输出 | 默认（对称化） | `--no-symmetrize` |
|---|---|---|
| `FS.cube` 的体数据 = `A(k,E_F)` | **对称化后** ⇒ 等值面在**原胞完整点群**下严格对称 | 原始逐 (K,G) 权重 ⇒ 等值面带对称性破缺 |
| `A(k,E).npz` 的 `A` | **对称化后** | 原始 |
| `A(k,E).npz` 的 `band_e` / `band_w` / `valid` | **原始**（逐带数据不做对称化，见下） | 原始 |
| `FS.bxsf` / `band.N`（逐带能量） | **原始** | 原始 |
| `unfoldfs band` 的 `WEIGHT.grd` | 路径上**对称相关**的点的权重按轨道平均 | 原始 |

⚠ **两个必须知道的限制**（`docs/algorithms.md §15.7.3` / `§15.8.4`）：

1. **逐带数据不做对称化**：对称相关点的"带集合"**本身就不同**（候选归属不一致）。
   实测 3×1×1 的 1T-TaS₂：19 条（12×12×1）/ 61 条（24×24×1）轨道里只有
   1 条 / 2 条的**有效带数**一致；在这几条"一致"的轨道里，同一带序号的能量
   最大差 **8.51 eV** ⇒ 同一带序号**并不是同一条带** ⇒ 按带序号平均会把物理上
   不同的带混在一起。
   ⇒ **默认输出里 `A`（`FS.cube` 与 `A(k,E).npz` 的 `A`）是完全对称的，而
   `band_e`/`band_w` 与 `FS.bxsf`/`band.N` 仍是原始的**；`A` 不再等于用
   `band_e`/`band_w` 重算出来的谱函数。
2. **对称化用的是原胞的完整点群**（修复轮 6）⇒ **任意**超胞都能折回原胞的
   对称性（3×1×1 这种破坏面内 C3 的超胞，实测 `A(k,E_F)` 的轨道离散度
   1.000e+00 → **0.000e+00**，见 `§15.8.3`）；但**逐带**文件里的带能量仍是
   **超胞**的能带（破坏对称性的超胞在对称相关的 k 上本来就给出不同的超胞带
   能量）⇒ 逐带文件的对称性破缺是**超胞的性质**，不是展开的缺陷。

⇒ **什么时候该关掉**：要研究"原始逐 (K,G) 权重"本身（例如与 vaspkit 282 对照、
或量化 C1 缺陷本身）时用 `--no-symmetrize`；做费米面图时用默认（对称化）。


## 7. 费米能（E_F）选取

| 模式 | 语义 | E_F 的选取需要 EIGENVAL 的**占据数**？ |
|---|---|---|
| `auto`（默认） | **先判费米穿越**（金属），再看带隙（见下） | 是（不可用时退化到 DOSCAR 的 E_F 并 warning） |
| `vbm` | 总是 VBM + 偏移 | 是 |
| `doscar` | 总是 DOSCAR 的 `E_F` | **否** |
| 数值 | 直接当作 E_F（绝对值） | **否** |

> ⚠ **但 `unfold` 命令整体仍然要求 EIGENVAL 存在** —— 反折叠要用它的**带能量**
> （PROCAR / DOSCAR 里都没有能量）。准确的说法是"**E_F 的选取**只有 `auto` 与
> `vbm` 需要占据数"，不是"数值/`doscar` 模式完全不碰 EIGENVAL"。
> EIGENVAL 缺失时 CLI 给**按模式区分**的可操作报错（数值/`doscar` 说缺的是
> 带能量，`auto`/`vbm` 说缺的是占据数），不是裸的 `[Errno 2]`。

* **VBM/CBM** 由 EIGENVAL 的占据数给出：占据判据为 `occ > 0.5`，
  VBM = 占据态最高能量，CBM = 未占据态最低能量。
* **`auto` 的判据**（按**这个顺序**，不需要展宽宽度）：

  ```
  ① has_fermi_crossing(band)  -> DOSCAR 的 E_F   (金属: 有带在 k 上穿过费米能)
  ② gap = CBM - VBM <= gap_tol -> DOSCAR 的 E_F   (零带隙 / 半金属)
  ③ VBM <= dos_ef <= CBM      -> VBM + offset      (E_F 落在带隙里 ⇒ 半导体)
  ④ 否则                       -> DOSCAR 的 E_F     (E_F 在带隙外)
  ```

* **① 的判据**（`fsout.has_fermi_crossing`）：**同一条带**的占据数在 k 上
  **跨过 0.5**（`min(occ) < 0.5 < max(occ)`）。
  实测：MoS2（半导体，带展宽）`band 27` 的占据是 `0.9866..1.0267` ⇒ **不跨**
  ⇒ 半导体 ✓；1T-TaS2（金属）带 25/26/27 的占据是 `0.1885..1.0198` ⇒ **跨**
  ⇒ 金属 ✓。
* ⚠ **金属的"表观带隙"是有限 k 网格的采样假象**，不能用来判断金属性：
  实测（1T-TaS2）6×6×1 网格的表观 gap = `0.209156 eV`、12×12×1 = `0.072827 eV`
  （网格加密 4 倍 gap 缩到 1/3），而 DOSCAR 的 E_F 两次都落在里面 ⇒ 必须先判 ①。
* ⚠ **"带边有分数占据"不是金属判据**：任何用了展宽的**有带隙**体系，带边的
  占据数都会变成 0.98 这类小数（高斯展宽与四面体法都一样）。
  实测（MoS2，服务器端到端跑）：`band 27` 的占据数是 `0.9866..1.0267`
  （唯一的分数占据），`band 28` 空 ⇒ VBM = `-2.077449`、CBM = `-0.465849`、
  gap = `1.6116 eV`（明确的半导体），但若按"有分数占据 ⇒ 金属"判，
  `auto` 会取 DOSCAR 的 E_F = `-1.920079 eV` —— 它落在**带隙中央**
  ⇒ 画出来的费米面是空的。
* **E_F 的选取**只有 `auto` 与 `vbm` 需要 EIGENVAL 的**占据数**：数值与
  `doscar` 模式**不使用**占据数 —— 用户明明白白给了 E_F，不该被一个用不上的
  占据数挡住。`auto` 模式若 EIGENVAL 里没有占据态（VBM 不可得）⇒ 退化到
  DOSCAR 的 `E_F` 并给出明确 warning（不静默）。
  （**带能量**任何模式都要 —— 见本节开头的方框。）
* **`band.N` 的编号是能量升序**：`band.1` 是**最低的深带**（**不是**"第一条
  穿过 E_F 的带"）。实测（1T-TaS2，12×12×1）跨 E_F 的是 **band.5 – band.14**。

## 8. 复核记录（2026-11-17）

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
* **`.frmsf` 能量口径**：FermiSurfer 2.2.1 自带的三个官方样例
  `mgb2_vfz.frmsf` / `pb_vf3D.frmsf` / `srvo3_orb.frmsf` 的特征值区间分别是
  `[-0.428, +0.498]` / `[-4.280, +7.505]` / `[-0.0396, +0.0522]` ——
  **全部以 0 为中心**；配套 `examples/frmsf.in` 写 `fermienergy 0.0`，
  且 `fermisurfer.cpp:219` 的缺省 `EF = 0.0` ⇒ `.frmsf` 的约定是
  **相对 E_F**（见 §4）。
* **`.grd` 的实测结论（2026-11-18 更正）**：原 FS.grd 已按上文的理由删除。
  更正前本文件曾写"gnuplot 5.2 解析**无错**" —— **不成立**：gnuplot 的
  `splot` 直接拒绝该文件（`2 columns only possible with explicit pm3d
  style (line 3)`，因为 gnuplot 根本没有"带头部的三维网格"格式；
  加 `set pm3d style line 3` 则是 `invalid pm3d option`），
  `plot` 只能把它当普通列数据。
* `bin/cube2xsf` 在服务器上**对参考 cube 也报同样的错**（工具自身损坏），
  故未用作 cube 的验证器。

## 9. KPOINTS 首行与 VASP 的解析分支（2026-11-18 端到端实跑）

`unfoldfs gen` 写出的 `KPOINTS` 的**首行**决定 VASP 走哪个解析分支：

| 首行形式 | VASP 6.6.0 的行为（OUTCAR 实测） |
|---|---|
| `Parameters to Generate KPOINTS (Don't Edit This Line) -- explicit k-point list, G grid 3x3x3` | `Read explicit list of k-points.` / `Found 9 k-points` ✓ |
| `G    3    3    3    0.758     # Parameters to Generate KPOINTS (Don't Edit This Line)`（`grid.write_kpoints` 的原始输出） | `Read explicit list of k-points.` / `Found 9 k-points` ✓ |

**结论**：VASP 6.6.0 对这两种首行**都**走"显式 k 点列表"分支 —— OUTCAR
明确打印 `Read explicit list of k-points.`，且算出的 k 点集合与顺序与
`KPOINTS` 里写的完全一致。`KPOINTS` 的第二行是 k 点个数、第三行是
`Reciprocal lattice`（坐标类型），这个"体"才是决定性的。

`unfoldfs gen` 仍把首行写成**纯注释**（把 `G/M` 标签挪进注释里）：这样不
依赖 VASP 的分支判定细节，语义也自解释（"这是一份显式 k 点列表"），
与 VASP 官方文档里的 `Explicit k-point list` 写法一致。

**实测事实（同一目录、同一份 INCAR，只换 KPOINTS 首行）**：

```
 KPOINTS: G    3    3    3    0.758     # Paramete
Read explicit list of k-points.
 Found      9 k-points:
   k-points           NKPTS =      9   k-points in BZ     NKDIM =      9   number of bands    NBANDS=     39
```

⇒ **VASP 不对显式 k 点表去重**：算出的 k 点数（9）== `grid_map.n_k`（9）。

## 10. 端到端实跑（2026-11-18，master / VASP 6.6.0）

服务器 `210.73.16.15`，Python 3.9.13（numpy 1.23.5 / scipy 1.13.1 /
spglib 2.0.2），VASP `vasp.6.6.0 06Mar2026`。3x1x1 MoS2 超胞（9 原子，
`M = diag(3,1,1)`），原胞网格 3x3x3（满网格 27 点 → 9 个超胞 k 点）：

* `unfoldfs gen` 写出的 `KPOINTS` + `INCAR.suggested` 原样跑 VASP，
  `EIGENVAL` 头 `54  9  39`（NELECT / **NKPT=9** / NBANDS），与
  `grid_map.n_k = 9` 一致；
* `unfoldfs unfold --strict-dominance` 通过（单 G 主导度 0.8747），
  产出 `FS.cube` / `FS.bxsf` / `band.1..band.18` / `A(k,E).npz`
  （**不含 `FS.grd`** —— 该格式已删除，见 §0）；
* `FS.bxsf` 用 XCrySDen 自带的 `bin/fsReadBXSF` 读出（exit=0，
  `ALL VERICES OF VORONOI: NVER=96`）✓；
* `FS.cube` 用 **ASE 3.26 `ase.io.cube.read_cube`** 读回：盒子 =
  `b_prim`（`[[1.96939968, 1.13703344, 0], [0, 2.27406687, 0],
  [0, 0, 0.35581576]]`），增量矢量 = `b_prim/n`，体数据 `(3,3,3)` ✓
  —— §2 的三条约定再次被第三方读取器验证；
* `band.N` 是 GUI-only 消费方（FermiSurfer），按 §4 的规范实现
  （能量相对 E_F）。

## 12. 端到端实跑披露（2026-11-18 修复轮）：ZVAL / E_F 自检 / `--grid`

> 章节号 12 是为了给并行的 `fsout` 修复轮留出 §11（避免合并冲突）。

### 12.1 ⚠️ 上一轮（§10）的实跑用的是**半核势** —— 物理基线未验证

§10 那轮实跑用的 POTCAR 是 `potpaw_PBE.64/Mo/POTCAR`，**`ZVAL = 6.000`**
（`VRHFIN =Mo: 4p5s4d` —— Mo 的 4d 态在核内），**不是**标准 Mo 势：

```
OUTCAR: POTCAR:    PAW_PBE Mo 08Apr2002
        VRHFIN =Mo: 4p5s4d
        POMASS =   95.940; ZVAL   =    6.000
        ZVAL   =   6.00  6.00
        NELECT =      54.0000    total number of electrons
```

⇒ 那个 1.61 eV 的带隙与由此得到的"费米面"**都不是 MoS₂ 的物理**；
§10 验证的是**流水线与文件格式**，不是电子结构。

同一台服务器上两个 Mo 势的 ZVAL：

| POTCAR | ZVAL | 含义 |
|---|---|---|
| `potpaw_PBE.64/Mo/POTCAR` | **6.000** | 半核势，4d 在核内 |
| `potpaw_PBE.64/Mo_sv/POTCAR` | **14.000** | 标准 Mo，4d 在核外 |

### 12.2 用标准势（`Mo_sv`，ZVAL = 14）重跑 —— 物理基线已验证

流程与 §10 完全一致（同一份结构文件、同一份 `INCAR.suggested`、同一个
`KPOINTS`），**只改 POTCAR 的组装**：`Mo_sv/POTCAR + S/POTCAR`。

**3x1x1 超胞**（9 原子，`M = diag(3,1,1)`，原胞网格 3x3x3）：

```
OUTCAR: ZVAL   =  14.00  6.00
        NELECT =      78.0000    total number of electrons
        NBANDS = 49
        Found      9 k-points:        NKPTS = 9   NBANDS = 49
EIGENVAL: (nk, nbands) = (9, 49)   分数占据 6 / 441
  band 39  E= -1.5171.. -1.4905  occ= 0.9913..1.0174   <-- 唯一的分数占据(带边)
VBM = -1.490480   CBM = 0.131685   gap = 1.622165 eV
DOSCAR E_F = -1.32951378                 <-- 落在带隙里
select_fermi_energy(auto) = -1.440480    <-- = VBM + delta(0.05)
```

> ⚠ 上面这段是 **Python 层的检查**(直接调 `fsout.band_edges` /
> `fsout.select_fermi_energy` 打的), **不是 CLI 的输出** ——
> `unfoldfs unfold` 打印的是 `相位约定 / VBM / CBM / E_F / 已写出` 那几行。
> `is_metal(...)` 是 `fsout` 的**公开 API**(保留给二次分析), 生产路径走的是
> `has_fermi_crossing`（Ruling 20/27）, CLI 从不调用 `is_metal`。
> `docs/algorithms.md §12.4` 里那一行同理。

| 量 | 半核势（ZVAL=6，§10） | 标准势（ZVAL=14，本节） |
|---|---|---|
| NELECT | 54 | **78** |
| 占据带数 | 27 | **39** |
| CLI 估计占据带数 | 39（偏大 12 条，碰巧安全） | **39（正好相等）** |
| 建议 NBANDS（旧=零余量 / 新=带余量） | 39 | 39 → **49** |
| VBM / CBM (eV) | −2.077449 / −0.465849 | **−1.490480 / 0.131685** |
| 带隙 (eV) | 1.611600 | **1.622165** |
| `--ef auto` 选出的 E_F (eV) | −1.920079（**带隙中央** ⇒ 空的费米面） | **−1.440480（VBM+delta）** |
| A(k,E_F) 峰值 / A 全局峰值 | 0.032 | **0.706** |

⇒ **`A(k, E_F)` 在 E_F 处有真峰**（0.2599 vs 全局峰值 0.3679，比值 0.706），
不再是旧跑里"只有峰值 3%"的 Lorentzian 尾巴；产出的费米面**非空** ✓。
（谱峰位置在 E = −1.4905 eV = VBM，E_F = VBM + 0.05 eV 处仍保有 70% 的权重。）

**3x3x1 超胞**（27 原子，项目标尺体系，`M = diag(3,3,1)`，`--grid 6 6 1`）：
见 §12.5。

### 12.3 E_F 落在带隙里 ⇒ `unfoldfs unfold` **拒绝执行**

CLI 在写任何文件之前自检：若选出的 E_F 落在带隙里（`gap > 0.05 eV` 且
`VBM < E_F < CBM`），且 E_F 离**最近的一个态**（任意 k、任意带的能量）
超过 `2×sigma`（谱函数在该处已衰减到峰值的 13% 以下），则：

* `--ef auto` / `--ef vbm` / `--ef doscar` ⇒ **拒绝**：stderr 打出可操作
  的错误信息，**退出码非零，不写出任何费米面文件**；
* `--ef <数值>` ⇒ **只告警、继续**（用户明给了 E_F，是他自己的判断）。

实测（旧跑的坏数据就是这个形态 —— 同一个目录 `/data/wjlu/work/claude/unfoldfs_e2e/run`）：

```
$ unfoldfs unfold --dir <旧跑> --ef doscar --outdir fs_doscar
错误: 选出的 E_F = -1.920079 eV 落在带隙内 (VBM=-2.077449, CBM=-0.465849, gap=1.611600 eV), 离最近的态 0.157370 eV (> 2×sigma) —— 该能量上没有任何态, 产出的费米面是空的。
      半导体请用 --ef vbm (VBM+delta); 金属请检查 ISMEAR/展宽设置。已中止, 未写出任何费米面文件。
RC=1
输出目录是否被创建: ls: cannot access fs_doscar: No such file or directory   <-- 零输出

$ unfoldfs unfold --dir <旧跑> --ef auto --outdir fs_auto
VBM / CBM     : -2.077449 / -0.465849 eV
E_F           : -2.027449 eV  (--ef auto, --delta 0.05)      <-- VBM+delta, 放行
提示: E_F = -2.027449 eV 落在带隙内 ..., 但离最近的态只有 0.050000 eV (<= 2×sigma) —— 画的是价带顶附近的等值面。
RC=0

$ unfoldfs unfold --dir <旧跑> --ef -1.920079 --outdir fs_num
提示: 选出的 E_F = -1.920079 eV 落在带隙内 ... —— 该能量上没有任何态, 产出的费米面是空的。
RC=0                                                        <-- 数值模式: 只告警, 继续
```

（0.157370 eV > 2×0.06 eV ⇒ 谱函数在 E_F 处只剩 3% ⇒ 费米面确实是空的。）

**注意**：`ISMEAR = 0` **修不好**这个问题 —— 高斯展宽一样会让带边占据数变成
小数。带隙体系的正确画法是 `--ef vbm`（VBM + delta）。

⚠ **不要**改成四面体法（`ISMEAR <= -4`）来"修"占据数：四面体法要求 VASP 从
KPOINTS 的**首行**自己解析 MP 网格（首行须为 `M P G X Y Z SHIFT`），与本工具的
**显式 k 点列表**互斥（首行是纯注释）⇒ VASP 报
`Error reading KPOINTS file, at line: N+1`。四面体法只与 `KSPACING`（VASP 自建
网格）兼容。

⚠ **金属必须跳过这个自检**：金属的"表观带隙"是**有限 k 网格的采样假象**
（1T-TaS2 实测：6×6×1 的表观 gap = `0.209156 eV`，12×12×1 = `0.072827 eV`，
而 DOSCAR 的 E_F 两次都落在里面）⇒ 拿它拒绝执行会把一次成功的金属计算扔掉。
CLI 用与 E_F 选取**同一个**判据（`fsout.has_fermi_crossing`：有带的占据数在 k
上跨过 0.5）识别金属，识别到就跳过自检并打印
`提示: 检测到费米穿越(金属) …… 跳过带隙检查。`

### 12.4 `unfoldfs gen --grid P1 P2 P3`（网格逃生口）

`--resolution` 走 `grid.choose_grid_points`，它强制每个维度都是
`comm(M⁻¹)` 的**全局**最小公倍数的倍数 ⇒ 对角 M 时 P3 也会被强制成 3 的
倍数（3x3x1 超胞 ⇒ P3 ∈ {3,6,9,…}，**6x6x1 永不可达**，机时白白多 3 倍）。
`--grid` 让用户直接点名网格，绕过这条规则：

```bash
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
             --grid 6 6 1 --outdir run/
```

* `--grid` 与 `--resolution` **互斥**（同时给 ⇒ 报错）；
* 网格必须自洽（`n_points == n_k · det(M)`，否则报错并说明原因）；
* 网格与体系对称性不兼容时（例如没有操作保持该网格不变）也会报错。

**3x3x1 超胞上的实测**（`--grid 6 6 1` ⇒ 满网格 36 点 → 4 个超胞 k 点，
权重各 9，和 = 36 ✓）：

```
unfoldfs: explicit k-point list (do not edit) -- G grid 6x6x1
   4
Reciprocal lattice
    0.000000000000    0.000000000000    0.000000000000    9.000000000000
    0.000000000000    0.500000000000    0.000000000000    9.000000000000
    0.500000000000    0.000000000000    0.000000000000    9.000000000000
    0.500000000000    0.500000000000    0.000000000000    9.000000000000
```

### 12.5 3x3x1 超胞（27 原子，标尺体系）的实跑结果

`Mo_sv` 势 + `--grid 6 6 1`（简报要求的网格）：

```
（**Python 层检查**的转录, **不是 CLI 的输出** —— CLI 打印的是
"相位约定 / VBM / CBM / E_F / 已写出" 那几行）
EIGENVAL 头 : 234      4    129      <-- NELECT=234 (9x14+18x6), NKPT=4, NBANDS=129
占据带数 = 117            CLI 的估计也是 117 => 正好相等
VBM = -1.510947  CBM = 0.102175  gap = 1.613122 eV
DOSCAR E_F = -1.37508366   (落在带隙里)
select_fermi_energy(auto) = -1.460947    <-- = VBM + delta
```

⇒ NELECT / 占据带数 / 带隙 / E_F 选取全部符合预期 ✓。

**但 6x6x1 的网格对 3x3x1 超胞太粗**：`unfold` 被主导度门槛挡住 ——

```
错误: 单 G 主导度 0.4992 < 0.5 —— 展开没有把态集中到唯一一个候选 G 上, 相位约定/副本平均不可信
```

原因：6x6x1 的原胞网格折到 3x3x1 超胞只剩 `∏(P_i/M_i) = 2×2×1 = 4` 个
超胞 k 点 ⇒ **超胞 BZ 只被 2×2 个点采样** ⇒ 态在候选 G 之间几乎等权
（主导度 ≈ 0.5）⇒ 展开不可信。

⇒ **网格细不细取决于超胞 BZ 被采样了几个点**（`n_k = ∏(P_i/M_i)`），不是
原胞网格点数本身。标尺体系（3x3x1）至少要 `12x12x1`（⇒ 4×4 = 16 个超胞
k 点）才谈得上展开；3x1x1 那次是 3x3x3（⇒ 9 个超胞 k 点，主导度 0.8936 ✓）。

**`--grid 12 12 1` 的结果**：能展开，但**相位约定落在退化区** ⇒ 结果不可复现
（⚠ 下面是**旧判据**下的输出：修复轮 9 换了判据 —— `s1` 现在由物理唯一确定，
`margin` 比较的是**两个 `s1=−1` 的候选**，告警文案也改了 ⇒ 这段输出需要
重跑才能复现）：

```
UserWarning: 相位约定择优落在退化区: winner=(+1,-1) 主导度 0.664424,
             runner-up 主导度 0.664405, 相对差仅 2.74e-05 < 1e-03 ——
             选出的约定由噪声主导(换一批采样就可能换一个约定), 结果不可复现
A(k,E_F) 峰值 / 全局峰值 = 0.134
```

| 3x3x1 网格 | 超胞 k 点数 | 主导度 | 结果 |
|---|---|---|---|
| 6x6x1 | 4（2×2 采样超胞 BZ） | 0.4992 | `unfold` **拒绝**（< 0.5） |
| 12x12x1 | 16（4×4） | 0.6644 | 能展开，但相位约定由噪声决定 ⇒ 不可复现 |

⇒ **标尺体系（3x3x1）在当前精度下没有可信的费米面**：网格粗了展不开，
网格细了相位约定不可复现 —— 这是 `unfold` 模块在该尺度上的能力边界，
不是 CLI 的问题（CLI 拒绝给误导性结果、并把不可复现性警告出来）。

### 12.6 已知限制（上报项，未在本轮修改）

`unfoldfs/grid.py` 的 `choose_grid_points` 用**全局** `comm(M⁻¹)` 约束所有
维度，而对角 M 的正确条件是**逐维** `M_ii | P_i`。这是 Task 3 模块的
**过约束**（不是数学必然）：`expand_to_full_grid` 真正需要的只是
`n_points == n_k · det(M)`，逐维条件即可满足。本轮用 `--grid` 逃生口
绕过（§12.4），**未修改 `grid.py` 的网格逻辑** —— 留给 grid 模块的 owner。

## 13. 展开能带的四个文件（vaspkit 格式，消费者：用户组现有的 MATLAB 脚本）

> 本节是 `unfoldfs/band.py` 的**权威格式规范**。格式出处是服务器上的实测
> （2026-11-18，`/data/wjlu/work/vasp_work/TaS2_new/AH_sliding_1v6/unfolded_band/`
> 与 `/data/wjlu/work/vasp_work/TaS2/A/band_unfold/`），互操作性的逐条核对见
> §13.6。

### 13.1 交付面

`unfoldfs band --dir <run>/ --outdir <bandfs>/` 写出 6 个文件：

| 文件 | 内容 | 消费者 |
|---|---|---|
| `ENERGY.grd` | (n_band, n_k) 带能量（eV，**绝对值**） | MATLAB `unfold_band_plot_mirror.m` |
| `MOMENTUM.grd` | (n_band, n_k) 沿路径的累计距离（每行重复同一条序列） | 同上 |
| `WEIGHT.grd` | (n_band, n_k) 权重 | 同上 |
| `KLABELS` | 高对称点的标签 + 距离 | 同上 |
| `unfolded_band.dat` | 长表：路径距离 / E(eV) / 权重 / 带号 | 自用（二次分析） |
| `unfolded_band.png` | matplotlib 画的加权能带图（x 轴 = `k-path (2π/Å)`；**顶部一条横向 colorbar：左 Low 右 High、jet 配色、无数值刻度，颜色 = 归一化后的谱权重**；高对称点标签在色条**上方**；图上标出 E_F 的**数值**；图注给出**画出来的**能量窗口 `E_plot`） | 自用（没有 MATLAB 时） |

**两种画法**（`--plot-style`）：

| `--plot-style` | 画法 | 布局 |
|---|---|---|
| `line`（默认） | 加权能带**线**图 | 顶部横向 colorbar（Low/High），高对称点标签在色条**上方** |
| `heatmap` | **谱函数热图** A(k,E) | x = k-path（无数字刻度）、y = 能量；**E_F 处白色虚线**；高对称点标签在图**下方**；**底部**横向 colorbar（Low/High，无刻度）；与用户参考截图同款 |

热图的颜色 = 每条带的权重按 `--weight-scale` 归一化后的值，**再用 σ 展宽**成 A(k,E)
⇒ 动态范围有界（0~1）。⚠ **不能**直接对物理 A(k,E) 做 log 归一化：高斯尾巴
可到 1e-200，会把色标拉爆（背景全红）✓（已在代码里避开）。


**前四个的文件名与布局与 vaspkit 完全一致** ⇒ 用户组现有的 MATLAB 脚本
（`unfold_band_plot_mirror.m`）**不改一行**就能画我们的数据。

### 13.2 三个 `.grd` 的布局

```
ENERGY.grd / MOMENTUM.grd / WEIGHT.grd  ——  都是 (n_band, n_k) 的矩阵
  行 = 带（band）
  列 = 路径上的 k 点（k-point）
  三个文件必须**同样大小**（MATLAB 的 pcolor(momentum, energy, weight) 要求）
```

* `ENERGY.grd[i][j]` = 第 i 条带在第 j 个路径点上的能量（eV，**绝对值**）
* `MOMENTUM.grd[i][j]` = 第 j 个路径点的累计距离（**每一行都是同一条序列**）
* `WEIGHT.grd[i][j]` = 第 i 条带在第 j 个路径点上的权重

数值格式：**每个数值占 14 列**（`"%14.6f"`），一行 = 14 × n_k 个字符，
字段之间没有额外的分隔符（靠定宽对齐）。这与 vaspkit 实测的 `.grd`
逐字节同格式（实测行长 1848 = 14 × 132 列）。

#### 13.2.1 ★ 列数 = **路径点**数，不是 KPOINTS 的行数（修复轮 7 / B1）

`.grd` 的**列**、`KLABELS` 的标签、`unfolded_band.dat` 的行，用的都是**原始
路径点**（**含重复端点**）；而 `KPOINTS` 的**行**是**去重后**的待算超胞 k。
两个数**一般不相等**：

| | 数量 | 出处 |
|---|---|---|
| 路径点（含重复端点） | 每段 n 个 × 段数 | `.grd` 的列 / `KLABELS` / `unfolded_band.dat` |
| 去重后的计算 k 点 | **≤** 路径点数 | `KPOINTS` 的行数与表头计数 |

重复的两个来源：① 路径的**相邻段共享端点**（M 既是第 1 段的终点又是第 2 段
的起点）；② **折叠**到超胞后重合（不同的原胞 k 折到同一个超胞 k —— 同一个
超胞 k 只算一次，展开时再各自取自己的候选）。

⇒ 阶段 2 把每个路径点映射回**已计算的** k：重复端点映射到**同一个** k
（两列的数据逐位相同），不同的原胞 k 折到同一个超胞 k 时各取自己的候选。

⚠ **为什么必须去重**：**VASP 6.6.0 读到含重复 k 点的显式列表会崩**（服务器
实测 2026-11-18：每次都在第一个 k 点算完后死掉；手工去重并把计数改成 33 后，
同一次计算 194 s 跑完全部 33 个 k 点）。⇒ `band.write_kpoints_explicit`
按周期距离去重（容差 1e-6）且**表头计数 = 实际行数**；`unfoldfs gen` 写出的
网格 KPOINTS 同样有"无重复"的断言。

### 13.3 `KLABELS`

```
K-Label    Coordinate of high-symmetry k-point in band-structure plots ⏎
           M       0.000⏎
       GAMMA       0.299⏎
     M|GAMMA       0.599⏎
           A       0.730⏎
 ⏎
 ⏎
* Tip: Please label each high symmetry point in KPOINTS (KPATH.in) file. Otherwise, they will be identified as 'XX' in KLABELS file⏎
```

* 第 1 行是表头（**末尾有一个空格**）；
* 随后是 **N 行**"标签 + 距离"：**标签右对齐在 12 列**、距离右对齐在 12 列
  （3 位小数）⇒ 每行恰好 24 个字符；
* 再是 **2 行只含一个空格**的行；
* 最后是 vaspkit 的提示行（131 字符）。

**标签必须右对齐**：MATLAB 脚本用 `regexp(str,'\s+','split')` 切分，
前导空白会切出一个空字段，只有右对齐时 `S(2)` 才是标签、`S(3)` 才是距离
（标签左对齐的话 `S(2)` 会变成距离 ⇒ 脚本画错）。

**行数约定**：MATLAB 脚本读 `Nline-4` 行标签 ⇒ 文件里必须正好有 `Nline-4`
行标签（1 表头 + N 标签 + 2 空白 + 1 提示 = N+4 行）。

**标签的来历**：段内两端同名时只写一次；不同时写 `前一段的终点|后一段的起点`。
实测（3 段 M→GAMMA, GAMMA→M, GAMMA→A）⇒ `M` / `GAMMA` / `M|GAMMA` / `A`
—— 与 vaspkit 实测的 KLABELS 逐字节一致。

### 13.4 距离的单位

`MOMENTUM.grd` 与 `KLABELS` 里的距离是**沿路径的累计长度**，用**原胞**倒格矢
计算，单位 **2π/Å**（与本工具其它地方一致）。⚠ vaspkit 的绝对标度可能不同
（其输出目录里的 `.grd` 与 `KPATH.in`/`KPOINTS` 不是同一次运行的产物，
无法反推其标度）；这不影响互操作 —— MATLAB 脚本只把距离当 x 轴坐标用，
只要**我们自己的四个文件自洽**即可（`KLABELS` 的距离与 `MOMENTUM.grd` 的
距离序列一致，容差 1e-3，因为 `KLABELS` 只保留 3 位小数）。

**修复轮 7 / B2 —— PNG 的 x 轴标签已改正**：`unfolded_band.png` 的 x 轴现在
标成 **`k-path (2π/Å)`**（旧实现写的是 `k-path (1/Angstrom)`，与
`MOMENTUM.grd` / `KLABELS` / `unfolded_band.dat` 的单位矛盾 ⇒ 同一个横坐标在
图与数据里单位不同，是个会误导人的不一致）。

**为什么 `.grd` 里没有"单位"表头**：三个 `.grd` 必须是**纯数字矩阵** ——
MATLAB 脚本用 `load('MOMENTUM.grd')` 直接读，加任何表头行都会让它读错矩阵。
⇒ 单位**不在文件里**、写在本节；`unfolded_band.dat`（我们自己的长表，不是
vaspkit 格式）才有表头 `# 路径距离(2π/Å)  E(eV, 绝对值)  权重  带号`。

### 13.5 ⚠ 权重的含义与 vaspkit 282 **不同**

| | 本工具的 `WEIGHT.grd` | vaspkit 282 的 `WEIGHT.grd` |
|---|---|---|
| 做法 | Popescu 投影法：把超胞态投影到原胞的各个候选 k，权重 = 该候选上的谱权重 | WAVECAR 平面波展开：把超胞波函数展开成原胞带，权重 = 原胞带系数 \|c(k)\|² |
| 含义 | **G 分辨**的谱权重（"这个态属于哪个 G"） | 原胞带的**平面波系数**的模方 |
| 归一 | **不归一**（数值量级 ~ 原胞的电子数，典型 0.1 ~ 几十） | 归一（0 ~ 1） |

两者都叫"权重"但**物理含义不同** ⇒ 数值不能互相比较。我们的权重在**同一次
计算内**可比；跨体系比较没有意义。MATLAB 脚本用 `caxis` 自动归一，
所以配色不受影响。

### 13.5b ⚠ 展开能带图的两种模式（`--unfold-mode`）

| | `assign`（默认） | `full`（经典 Popescu 展开能带图） |
|---|---|---|
| 每个路径点画哪些带 | 只画**归属到该 k′** 的带（`unfold.unfold_supercell` 的候选重组：`assign = w.argmax(axis=1)`） | 画**全部**超胞带（3×1×1 时 96 条） |
| 颜色 | 该带在 k′ 通道的权重 | 同左 |
| 画面 | **就是原胞的能带 + 权重** —— 干净，但**没有折叠副本** | 强带 = 原胞在 k′ 的带；弱带 = 别的 k′ 的带（**折叠副本**）⇒ 有"低权重色散"的**弥散背景** |
| 典型带数（3×1×1） | 35 | 96 |

⚠ **为什么默认模式"看起来像原胞"**：候选重组那一步把"归属到别的候选"的带
**整个丢掉**了 ⇒ 每个路径点只剩"原胞在 k′ 的那几条带" ⇒ 图 = 原胞能带 + 权重。
这不是 bug（那个语义对"看原胞能带"很干净），但它**不是**论文里那种展开能带图。

⇒ 画论文式的展开能带请加 `--unfold-mode full`；默认保持 `assign` 是为了
**不改变** `.grd` / `.dat` 四个文件的语义（用户组现有 MATLAB 脚本读它们）✓。

### 13.6 互操作性的逐条核对（对着 MATLAB 脚本的源码）

用户组脚本 `/data/wjlu/work/vasp_work/TaS2_new/AH_sliding_1v6/unfolded_band/
unfold_band_plot_mirror.m`（2026-11-18 读原文）：

| 脚本里的动作 | 我们的文件 | 结论 |
|---|---|---|
| `load('MOMENTUM.grd')` / `load('ENERGY.grd')` / `load('WEIGHT.grd')` | 三个同样大小的纯数字矩阵 | ✓ |
| `pcolor(momentum, energy, weight)` | 三个文件同样大小 | ✓ |
| `energy(n)`、`weight(n,:)` 取行 | 行 = 带、列 = k 点 | ✓（数值沿 k 变化） |
| `Nline-4` 行标签，`S(2)`=标签、`S(3)`=距离 | `KLABELS` 的布局（§13.3） | ✓ |
| `K_label` 用 `S(2)`，`GAMMA` 特判、`\|` 分支取 `label(1:2)` | 标签右对齐、`GAMMA`/`M\|GAMMA` 原样给出 | ✓ |
| `xlim([K_length(Nkpt_start) K_length(Nkpt_end)])` | 距离单调递增 | ✓ |
| `line([...],[0 0],'--')` 画费米能 | 我们的 `.grd` 存**绝对**能量，脚本用 `Efermi_adjust` 平移 | ✓（与 vaspkit 一致） |
| `IsPlotMirror` 分支：`[-fliplr(momentum) momentum]` | `--mirror` 只影响我们的 PNG（`.grd` 不镜像） | ✓ |

⚠ **一处残留风险（脚本自身的，不是文件格式的）**：脚本用
`energy(n)`（线性下标）与 `weight(n,:)`（行下标）挑行算 `caxis`；
若**第一条路径点上没有任何带的能量落在 `E_plot_min..E_plot_max` 窗口内**，
`weight_plot` 为空 ⇒ 脚本在 `min(min(weight_plot))` 处报错。这是脚本的
健壮性问题，我们无法通过改文件格式规避（除非把路径点顺序换掉）。

⚠⚠ **`unfoldfs band` 的 `--ef auto` 在**路径**上不可靠 ⇒ 用 `--ef <数值>`**
（实测，1T-TaS₂ 的 Γ-M-K-Γ，60 点/段 = 177 个 k 点）：

| E_F 取法 | 值 (eV) | 评价 |
|---|---|---|
| 96×96×1 **密网格**的 `--ef auto` | **−1.501119** | ✓ 基准 |
| 同一次 band 运行的 `--ef auto`（原胞，177 点路径） | −1.325055 | ✗ 偏高 0.176 eV |
| 同一次 band 运行的 `--ef auto`（3×1×1 超胞，177 点路径） | −1.543908 | ✗ 偏低 0.043 eV |

⇒ `--ef auto` 走的是 **DOSCAR 的 E_F**，而路径上的 DOS 采样很稀 ⇒ 金属体系的
E_F 估计会飘。⇒ **画能带时请显式给 `--ef`**（用密网格算出的那个值），
否则图上那条 E_F 虚线会落在错的位置。

⚠ **PNG 的颜色标度默认是**全局**的**（`--weight-scale linear` = `w/w_max`，
w_max 取**所有带所有 k**的最大值）⇒ 强的局域带会把 E_F 附近的带压成一片深蓝
（实测 1T-TaS₂ 的 3×1×1 展开能带：+10.8 eV 那条强局域平带 `w = 9.23`，
而 E_F 附近的带只有 **0.26~1.12** ⇒ 只占色标的 3%~12% ⇒ 图上"看不到权重"）。
⇒ **画反折叠能带请加 `--weight-scale log`**（`log10(w)` 在 `[w_min⁺, w_max]`
上线性映射到 0~1；色条标签会写 "spectral weight (log scale)"）：
实测色标偏斜度 `max(w)/median(w>0)` = **175.6**（3×1×1）/ **76.2**（原胞直通）
⇒ 工具在用线性色标且偏斜度 ≥ 30 时会**打印提醒** ✓。
（`docs/algorithms.md` §14 第 16 条另给了"按能量窗归一化"的画法。）


