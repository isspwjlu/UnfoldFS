# unfoldfs —— 超胞能带反折叠 → 原胞 BZ 内的费米面

`unfoldfs` 把**超胞 VASP 计算**的结果反折叠回**原胞第一布里渊区**，给出带谱权重的
费米面（`A(k, E_F)` 的等值面），并写成能被 **VESTA / XCrySDen / FermiSurfer**
直接打开的文件。

* 算法、约定与每一项实测数字 → [`docs/algorithms.md`](docs/algorithms.md)
* 输出文件的逐字段格式规范 → [`docs/formats.md`](docs/formats.md)

---

## 1. 安装

```bash
pip install -e .
```

装完就有 `unfoldfs` 命令。**不想装包**也可以直接在仓库根目录用模块入口
（所有命令等价）：

```bash
python -m unfoldfs.cli --help
python -m unfoldfs.cli gen --help
python -m unfoldfs.cli unfold --help
python -m unfoldfs.cli band --help
python -m unfoldfs.cli info --help
```

### 依赖

| 项 | 版本要求 | 说明 |
|---|---|---|
| Python | **≥ 3.9** | 开发机实测 3.11.7；服务器 master 实跑用的是 3.9.13 |
| numpy | ≥ 1.20 | |
| scipy | ≥ 1.6 | 周期 KD-tree（网格点匹配） |
| spglib | **≥ 2.0** | 原胞点群（代码用 **spglib ≥ 2.0 的 dict API**：`sym["rotations"]`；1.x 的 tuple API 会直接 `TypeError`） |
| VASP | **6.6.0**（实测） | 只用来产生输入（`EIGENVAL` / `PROCAR` / `DOSCAR`） |
| matplotlib | **可选**（`pip install 'unfoldfs[plot]'`） | 只有 `unfoldfs band` 的 PNG 需要；`gen` / `unfold` / `info` 与四个 vaspkit 文本文件都**不需要** |
| VESTA 3.5.8 / XCrySDen 1.5.60 / FermiSurfer 2.2.1 | 可选 | **消费方**，用来打开输出的费米面文件 |

费米面那部分**不画图** ⇒ **不需要** matplotlib / scikit-image；
只有 `unfoldfs band` 的 `unfolded_band.png` 需要 matplotlib（缺了会给出
可操作的报错，并提示 `pip install 'unfoldfs[plot]'`；也可以用 `--no-plot`
只出数据文件）。

> 服务器 master 的实测环境：Python 3.9.13（numpy 1.23.5 / scipy 1.13.1 /
> spglib 2.0.2）、VASP `vasp.6.6.0 06Mar2026`。
>
> ⚠ **服务器上装本包要注意 setuptools**：本项目的构建要求 `setuptools>=77`，
> 而 master 的 anaconda3 是 **67.6.1** ⇒ 要么先 `pip install -U setuptools`，
> 要么允许 pip 联网（**默认的 build isolation 会自动拉一个新 setuptools**，
> 可行）；`pip install -e . --no-build-isolation` 在没升级 setuptools 时会失败。

---

## 2. 四个子命令

### 2.1 `unfoldfs gen` —— 生成网格与 KPOINTS

```
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in
              [--resolution RESOLUTION] [--grid P1 P2 P3]
              [--grid-type {gamma,mp}] [--outdir OUTDIR]
              [--nbands NBANDS] [--encut ENCUT] [--kpar KPAR]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--supercell` | ✔ | — | 超胞 POSCAR 路径 |
| `--primcell` | ✔ | — | 原胞 `PRIMCELL.vasp` 路径 |
| `--transmat` | ✔ | — | `TRANSMAT.in` 路径（行/列约定由工具自动探测） |
| `--resolution` | | `0.02` | k 网格分辨率，单位 **2π/Å**，按 `\|b_i\|` 定义 |
| `--grid P1 P2 P3` | | — | 显式点名网格，**绕过**分辨率规则；与 `--resolution` **互斥** |
| `--grid-type` | | `gamma` | `gamma`（Γ 居中）或 `mp`（Monkhorst-Pack） |
| `--outdir` | | `.` | 输出目录 |
| `--nbands` | | 按价电子数估计 | 建议 INCAR 里的 `NBANDS` |
| `--encut` | | `320` | 建议 INCAR 里的 `ENCUT`（eV） |
| `--kpar` | | `1` | 建议 INCAR 里的 `KPAR` |

**产出**（全部写进 `--outdir`）：

| 文件 | 内容 |
|---|---|
| `KPOINTS` | **去重后的**超胞 k 点 + 权重（权重之和 == 满网格点数） |
| `grid_map.npz` | 网格与对称性的全部映射（`unfold` 阶段要用） |
| `INCAR.suggested` | 建议的 INCAR（含**必须保留**的 `ISYM = 0` 与 `LORBIT = 12`） |
| `POSCAR` / `PRIMCELL.vasp` / `TRANSMAT.in` | 从输入路径复制过来（`unfold` 默认从 `--dir` 里找它们） |

`gen` 会打印网格规模、不可约点数、实际计算的 k 点数、权重和与 `det(M)`。

**`--resolution` 的网格点数规则**（`--grid` 显式指定时**不走**这条规则）：

* **对角 `M`**（面内放大、c 轴不放大这类最常见的情况）：第 i 维要求
  **`M_ii | P_i`** —— **没被放大的那一维不受约束**。
  例：`M = diag(3,3,1)` ⇒ `P_1`、`P_2` 必须是 3 的倍数，`P_3` **任意**
  （真实 monolayer 的 c 轴有 ~17.7 Å 真空，`P_3 = 1` 就够）。
* **非对角 `M`**：用**全局** `comm(M⁻¹)`（逐维条件在那里不成立）。

⇒ 对 `M = diag(3,3,1)`，同样的 `--resolution` 现在选出 **6×6×1**（满网格 36 点）
而不是 6×6×3（108 点）—— **3 倍**的机时差别。
详见 [`docs/algorithms.md`](docs/algorithms.md) §3.5。

### 2.2 `unfoldfs unfold` —— 反折叠并写费米面文件

```
unfoldfs unfold --dir DIR [--supercell SUPERCELL] [--primcell PRIMCELL] [--transmat TRANSMAT]
                 [--grid-map GRID_MAP] [--ef EF] [--delta DELTA] [--sigma SIGMA]
                 [--strict-dominance] [--no-symmetrize] [--outdir OUTDIR]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--dir` | ✔ | — | VASP 计算目录（须含 `EIGENVAL` / `PROCAR` / `grid_map.npz`；**`DOSCAR` 仅 `--ef auto\|doscar` 需要**） |
| `--supercell` | | `<dir>/POSCAR` | |
| `--primcell` | | `<dir>/PRIMCELL.vasp` | |
| `--transmat` | | `<dir>/TRANSMAT.in` | |
| `--grid-map` | | `<dir>/grid_map.npz` | |
| `--ef` | | `auto` | `auto` \| `vbm` \| `doscar` \| **数值(eV)** |
| `--delta` | | `0.05` | 半导体时 `VBM + delta` 的偏移（eV） |
| `--sigma` | | `0.06` | 谱函数展宽（eV） |
| `--strict-dominance` | | 关 | 要求**每个态**的单 G 主导度 ≥ 0.5（严格模式） |
| `--no-symmetrize` | | 关 | **不做**点群轨道对称化（默认**做**，用**原胞完整点群**）⇒ `A(k,E)` 保持逐 (K,G) 的原始谱权重，费米面带对称性破缺 |
| `--outdir` | | `fs` | 费米面文件输出目录 |

`--ef` 的语义：

| 值 | 含义 |
|---|---|
| `auto` | **先判费米穿越**：有带在 k 上穿过费米能（金属）⇒ 取 DOSCAR 的 E_F；否则有带隙且 DOSCAR 的 E_F 落在带隙里 ⇒ 取 `VBM + delta` |
| `vbm` | 总是 `VBM + delta`（半导体的正确画法） |
| `doscar` | 总是 DOSCAR 里的 E_F |
| 数值 | 直接当作 E_F（绝对值，eV）；**只告警、不阻断** |

> **金属的"表观带隙"是有限 k 网格的采样假象**，不能拿它判断金属性：
> 实测 1T-TaS₂（金属）在 6×6×1 网格上切出 `0.209156 eV` 的假带隙、
> 12×12×1 上是 `0.072827 eV`（网格加密 4 倍 gap 缩到 1/3），而 DOSCAR 的 E_F
> 两次都落在里面。⇒ 判据是**有没有带真的穿过费米能**（同一带占据数在 k 上
> **跨过 0.5**）。金属时 CLI 也**跳过**"E_F 落在带隙内"的拒绝检查。
> 详见 [`docs/algorithms.md`](docs/algorithms.md) §9。

`unfold` 会打印选出的相位约定 `(s1, s2)` 与单 G 主导度、VBM/CBM、选出的 E_F、
**点群对称化是开是关**，并把它们写进输出文件的头部与 `A(k,E).npz` 的元信息。

#### ★ 默认输出是**对称化后**的（点群轨道平均）

展开得到的谱权重 `A(k,E)` 在晶体点群下**天然不协变**（对称相关的两个 k 点，
谱函数可以差到 **100 %**）⇒ 费米面的等值面相对晶体对称性是畸变的。
⇒ **默认把 `A(k,E)` 按原胞点群轨道平均**（把展开谱**投影到对称不变子空间**），
实测最大相对偏差从 **1.000e+00 降到 0.000e+00**（三个真实数据集）。
`--no-symmetrize` 关掉 ⇒ 回到"逐 (K,G) 的原始权重"。

★ **用的是原胞的完整点群**（12 个操作），**不是**"原胞与超胞的公共对称操作"
⇒ **任何**超胞都能折回原胞的对称性：实测 3×1×1（破坏面内 C3）的 1T-TaS₂，
`A(k,E_F)` 在 12 个操作下的相对离散度 **1.000e+00 → 0.000e+00** ⇒
**六重对称恢复**（修复轮 6）。

⚠ 一条限制：**逐带数据**（`band.N` / `FS.bxsf` / npz 的 `band_e`/`band_w`）
**不**对称化 —— 对称相关点的"带集合"本身就不同（实测 3×1×1：19 / 61 条轨道里
只有 1 / 2 条的有效带数一致，且带能量最大差 8.51 eV ⇒ 同带序号不是同一条带）。
机制、实测与对照表 → [`docs/algorithms.md`](docs/algorithms.md) §15.7 / §15.8；
每个输出在两种模式下的含义 → [`docs/formats.md`](docs/formats.md) §6c。

#### 2.2b ★ 原胞自己的费米面（`det(M) = 1`，"直通模式"）

**你有一个原胞的 VASP 计算，想要原胞的费米面**（不涉及任何超胞）：

```bash
# TRANSMAT.in 写单位阵（"超胞" = 原胞），POSCAR 与 PRIMCELL.vasp 相同
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
              --grid 24 24 1 --outdir run/
# ... 跑 VASP ...
unfoldfs unfold --dir run/ --outdir fs/        # 照常工作
```

* `unfold` 会打印 `模式 : 原胞直通 (det(M)=1 —— 不做展开; 权重 = Σ_{ion,lm}|c|² …)`。
* ⚠ **权重定义与超胞展开模式不同**：

  | | 超胞展开（`det(M) > 1`） | **原胞直通（`det(M) = 1`）** |
  |---|---|---|
  | 权重 | Popescu 的**相干** G 投影 | **轨道投影** `Σ_{ion,lm}\|c\|²`（**非相干**） |
  | 相位约定 | 要择优 s1/s2 | 不存在 |

  ⇒ **两种模式的费米面形状不可直接比较**；可直接比较的是 E_F、带能量、能带走向
  与拓扑。详见 [`docs/algorithms.md`](docs/algorithms.md) §6.4。
* 以前这里会直接报错"4 种相位约定没有区分度"（候选 G 只有一个 ⇒ 4 种约定必然
  同分 ⇒ 择优守卫把原胞自己的计算也拒了）。现在走独立的直通分支。

### 2.3 `unfoldfs band` —— 沿 k 路径做展开能带（unfolded band）

和 `gen`→VASP→`unfold` 同一套路，分**两个阶段**：

```bash
# 阶段 1：读 vaspkit 的 KPATH.in，写 KPOINTS + INCAR.suggested + KLABELS.in
unfoldfs band --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
              --kpath KPATH.in --outdir run/

# 阶段 2：读 VASP 输出，展开并写 vaspkit 的四个文件（+ 长表 + PNG）
unfoldfs band --dir run/ --outdir bandfs/ [--ef auto|vbm|doscar|数值]
              [--sigma 0.06] [--e-min E] [--e-max E] [--mirror] [--no-plot]
              [--no-symmetrize]
```

> ⚠ 阶段 1 写出的 `KPOINTS` 里**没有重复 k 点**（按周期距离去重，表头计数 =
> 实际行数）—— **VASP 6.6.0 读到含重复 k 点的显式列表会崩**（实测）。路径的
> 相邻段共享端点、且不同的原胞 k 可能折到同一个超胞 k ⇒ 去重后的 k 点数
> **少于**路径点数；两者都打在 stdout 里（"路径点总数" / "KPOINTS k 点"）。

| 参数 | 阶段 | 默认 | 说明 |
|---|---|---|---|
| `--supercell` / `--primcell` / `--transmat` | 1 | — | 结构文件（阶段 2 默认从 `--dir` 里找） |
| `--kpath` | 1 | — | **vaspkit 格式**的 `KPATH.in`（原胞倒数分数坐标 + 标签） |
| `--dir` | 2 | — | VASP 计算目录（须含 `EIGENVAL` / `PROCAR` / `KPATH.in`） |
| `--outdir` | 1 / 2 | `run` / `bandfs` | 输出目录 |
| `--ef` | 2 | `auto` | 与 `unfold` **完全一致**的语义；只用于画图上的费米能虚线 |
| `--e-min` / `--e-max` | 2 | 自动 | 图的能量范围（eV，**绝对值**） |
| `--mirror` | 2 | 关 | 把镜像路径（距离取负、顺序反过来）也画上（对应 MATLAB 脚本的 `IsPlotMirror`） |
| `--no-plot` | 2 | 关 | 不画 PNG（不需要 matplotlib） |
| `--no-symmetrize` | 2 | 关 | **不做**路径上的点群轨道对称化（默认**做**；泛路径通常无操作，见 §16.7） |
| `--nbands` / `--encut` / `--kpar` | 1 | 同 `gen` | 建议 INCAR 里的 `NBANDS` / `ENCUT` / `KPAR` |

**产出**（阶段 2 写进 `--outdir`）：

| 文件 | 内容 |
|---|---|
| `ENERGY.grd` | (n_band, n_k) 带能量（eV，**绝对值**） |
| `MOMENTUM.grd` | (n_band, n_k) 沿路径的累计距离 |
| `WEIGHT.grd` | (n_band, n_k) 权重 |
| `KLABELS` | 高对称点的标签 + 距离（**与 vaspkit 逐字节同格式**） |
| `unfolded_band.dat` | 长表：路径距离 / E(eV) / 权重 / 带号（自用） |
| `unfolded_band.png` | 加权能带图（x 轴 = `k-path (2π/Å)`；**顶部横向 colorbar：左 Low 右 High、jet 配色、无数值刻度，颜色 = 归一化谱权重**；图上标 E_F 的数值；图注给 `E_plot` 窗口）（自用，需 matplotlib） |

前四个**就是 vaspkit 的文件名与布局** ⇒ 用户组现有的 MATLAB 脚本
（`unfold_band_plot_mirror.m`）**不改一行**直接能画我们的数据。
格式规范见 [`docs/formats.md`](docs/formats.md) §13。

> ⚠ **路径点 ≠ 计算的 k 点**（修复轮 7）：上面四个文件里的**列**是**原始路径点**
> （含重复端点），而阶段 1 写出的 `KPOINTS` 的**行**是**去重后**的待算超胞 k
> —— 两个数一般不相等。**VASP 6.6.0 读到含重复 k 点的显式列表会崩**
> （服务器实测：每次都在第一个 k 点算完后死掉）⇒ 写 KPOINTS 时按周期距离去重，
> 表头计数 = 实际行数。见 [`docs/algorithms.md`](docs/algorithms.md) §16.1.1。
>
> ⚠ **`WEIGHT.grd` 的权重与 vaspkit 282 的不是同一件事**：本工具给的是 **Popescu
> 投影法的 G 分辨权重**，vaspkit 282 给的是 **WAVECAR 平面波展开得到的原胞带
> |c(k)|²**。两者都叫"权重"但物理含义不同、数值不可比（我们的不归一）。
>
> ⚠ 展开谱权重在晶体点群下**天然不协变**（带能量不受影响）⇒ **能带曲线可信，
> 能带形状不可信**。默认对路径上**对称相关**的点做轨道平均（`--no-symmetrize`
> 可关），但泛路径点的轨道离开路径 ⇒ 通常是无操作。
> 详见 [`docs/algorithms.md`](docs/algorithms.md) §15 / §16.6 / §16.7。

### 2.4 `unfoldfs info` —— 查看网格与对称信息

```
unfoldfs info --dir DIR [--grid-map GRID_MAP]
```

打印网格规格、满网格点数、不可约点数、实际计算的 k 点数、**对称操作的
保留/剔除数与原因**、朴素超胞网格点数（∏(P_i/M_i)；算不出来时给**具体原因**）、
相对原胞满网格的比例、`det(M)`、KPOINTS 权重和。

---

## 3. 典型工作流

### ① 生成网格与 KPOINTS

```bash
unfoldfs gen \
    --supercell  POSCAR \
    --primcell   PRIMCELL.vasp \
    --transmat   TRANSMAT.in \
    --grid       6 6 1 \
    --outdir     run/
```

```
网格规格      : 6 x 6 x 1 (gamma), 满网格 36 点
对称不可约点  : 10
实际计算 k 点 : 4 (超胞分数坐标), 权重和 = 36 == 满网格点数
超胞变换      : det(M) = 9
```

### ② 跑 VASP 6.6.0

```bash
cp run/INCAR.suggested run/INCAR      # 建议值已含 ISYM = 0 / LORBIT = 12
cp POTCAR run/POTCAR                   # 或 cat Mo_sv/POTCAR S/POTCAR > run/POTCAR
cd run && mpirun -np 8 vasp_std
```

> ⚠ **`ISYM = 0` 必须保留**（`INCAR.suggested` 已写好，**不要改**）。
> 否则 VASP 会把显式 k 点表按对称性归约到不可约部分（IBZ），
> **k 点的个数与顺序都会变** ⇒ 反折叠的「k 点 → 候选 G」映射随之失效，
> `unfold` 会直接报错 `VASP 算出的 k 点数 N != grid_map 记录 M`。
>
> ⚠ **`LORBIT = 12` 必须保留**：`LORBIT = 11` **不写相位块**，
> PROCAR 里没有复投影系数 ⇒ 解析器直接报错（消息里带 `LORBIT=12` 提示）。
>
> ⚠ **`NBANDS` 请按实际 POTCAR 的 `ZVAL` 核对**。CLI 只能从元素符号**估计**
> 价电子数（内置表，未知元素按 8 计），不知道你用的势是半核势还是标准势。
> 表里 Mo = 14 只对 `potpaw_PBE.64/Mo_sv` 成立（半核 `Mo` 是 6）。
> 看 `OUTCAR` 的 `NELECT` 最保险；不对就用 `unfoldfs gen --nbands <N>` 覆盖。
>
> ⚠ **不要**把 `ISMEAR` 改成四面体法（`ISMEAR <= -4`）。四面体法要求 VASP 从
> KPOINTS 的**首行**自己解析 MP 网格（首行须为 `M P G X Y Z SHIFT`），与本工具
> 写出的**显式 k 点列表**互斥（首行是纯注释）⇒ VASP 报
> `Error reading KPOINTS file, at line: N+1`。四面体法只与 `KSPACING`
> （VASP 自建网格）兼容。推荐仍是 `ISMEAR = 1` + `SIGMA = 0.2`
>（想更小的展宽用 `ISMEAR = 0` + 更小的 `SIGMA`）。展宽只影响带边占据数，
> **不能让费米能更准**。

### ③ 反折叠

```bash
unfoldfs unfold --dir run/ --outdir fs/ --ef vbm
```

```
（节选 —— 实际输出还有"提示:"开头的 stderr 行与"估计电子数"等行）
相位约定      : s1 = -1, s2 = +1 (单 G 主导度 0.8936)
VBM / CBM     : -1.490480 / 0.131685 eV
E_F           : -1.440480 eV  (--ef vbm, --delta 0.05)
已写出        : fs/FS.cube
已写出        : fs/band.1 ... band.19
已写出        : fs/A(k,E).npz
已写出        : fs/FS.bxsf
```

### ④ 打开费米面

| 工具 | 打开哪个文件 | 看什么 |
|---|---|---|
| **VESTA** | `FS.cube` | `A(k, E_F)` 的**体数据等值面**（等值面层级见文件头第 2 行给出的建议值） |
| **XCrySDen** | `FS.bxsf` | 逐带能量网格的费米面（E_F 写在文件的 INFO 块里） |
| **XCrySDen** | `FS.cube` | 体数据等值面（菜单过滤器 `{.cube}`） |
| **FermiSurfer** | `band.N` | 逐带费米面（能量**相对 E_F**，直接打开即可） |
| Python | `A(k,E).npz` | 完整谱函数 + 元信息，二次分析用 |

### ⑤ 展开能带（沿 k 路径 —— 另一条流水线）

```bash
# 1) 准备 VASP 输入：读 vaspkit 的 KPATH.in -> KPOINTS + INCAR.suggested + KLABELS.in
unfoldfs band --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
              --kpath KPATH.in --outdir run/

# 2) 跑 VASP（同 ②；INCAR 里的 ISYM = 0 / LORBIT = 12 同样必须保留）

# 3) 展开并出图
unfoldfs band --dir run/ --outdir bandfs/
```

```
相位约定      : s1 = -1, s2 = +1 (单 G 主导度 0.7980)
E_F           : -2.026108 eV  (--ef auto, --delta 0.05)
路径点        : 2(高对称点 GAMMA, X)
展开带数      : 16
已写出        : bandfs/ENERGY.grd
已写出        : bandfs/MOMENTUM.grd
已写出        : bandfs/WEIGHT.grd
已写出        : bandfs/KLABELS
已写出        : bandfs/unfolded_band.dat
已写出        : bandfs/unfolded_band.png
```

（上例是 3×3×1 MoS₂ 超胞 fixture 的实跑：2 个路径点、16 条展开带。）

`bandfs/` 里的 `ENERGY.grd` / `MOMENTUM.grd` / `WEIGHT.grd` / `KLABELS`
**就是 vaspkit 的文件名与布局** ⇒ 用户组现有的 MATLAB 脚本
（`unfold_band_plot_mirror.m`）不改一行就能画我们的数据。

---

## 4. 输出文件（4 种格式）

| 文件 | 消费者 | 内容 |
|---|---|---|
| `FS.cube` | VESTA / XCrySDen / VMD | Gaussian cube：假晶格 = 倒格矢，体数据 = `A(k, E_F)`。第 2 行注释写明 E_F、VBM/CBM、单位与**建议等值面层级** |
| `FS.bxsf` | XCrySDen / FermiSurfer | 逐带能量网格（**只支持 Γ 网格**；MP 网格下 CLI 会跳过它并说明原因） |
| `band.N` | FermiSurfer | FermiSurfer 的 `.frmsf` 逐带文件（每文件一条带，能量**相对 E_F**） |
| `A(k,E).npz` | 自用 / 二次分析 | `A`（谱函数体数据）、`e_grid`、`k_cart`、`k_frac`、`band_e`/`band_w`/`valid`、`ef`/`vbm`/`cbm`/`sigma`、单位说明 |

每个文件的**逐字段规范**（字段顺序、单位、约定、第三方读取器证据）在
[`docs/formats.md`](docs/formats.md)。

> ⚠ **`band.N` 的编号是能量升序**：`band.1` 是**最低的深带**，**不是**"第一条
> 穿过 E_F 的带"。（实测：1T-TaS₂ 的 12×12×1 与 6×6×1 两轮里，跨 E_F 的都是
> **band.5 – band.14**。）

> ### ⚠ 关于 `FS.grd`：原计划有这一格，**已删除**
>
> 它**没有任何标准读取器**：
>
> * XCrySDen 1.5.60 全树**没有** `.grd` 代码路径（`grep -rIl grd` 只命中一句英文
>   注释与无关的 Tcl 变量）；
> * gnuplot 5.2 的 `splot` **直接报错**（`2 columns only possible with explicit pm3d
>   style`）—— gnuplot 根本没有"带头部的三维网格"这种格式。
>
> 保留一个谁也不读的文件没有意义，故**删掉**，而不是改名或保留。
> 二次分析的需求由 `A(k,E).npz` 覆盖。
> **想在 XCrySDen 里画体数据等值面请用 `FS.cube`**（XCrySDen 的两个体数据入口是
> "Open BXSF" 与 "Gaussian98 Cube File"，过滤器 `{.cube}`）。

### 4c. 展开能带的输出文件（`unfoldfs band`）

| 文件 | 消费者 | 内容 |
|---|---|---|
| `ENERGY.grd` | 用户组的 MATLAB 脚本 | (n_band, n_k) 带能量（eV，**绝对值**） |
| `MOMENTUM.grd` | 同上 | (n_band, n_k) 沿路径的累计距离（每行重复同一条序列） |
| `WEIGHT.grd` | 同上 | (n_band, n_k) 权重（**Popescu G 分辨权重，不归一**） |
| `KLABELS` | 同上 | 高对称点的标签 + 距离（**与 vaspkit 逐字节同格式**） |
| `unfolded_band.dat` | 自用 / 二次分析 | 长表：路径距离 / E(eV) / 权重 / 带号 |
| `unfolded_band.png` | 自用 | 加权能带图（x 轴 = `k-path (2π/Å)`；**顶部横向 colorbar：左 Low 右 High、jet 配色、无数值刻度**；标 E_F 数值；图注给 `E_plot` 窗口）（需 matplotlib） |

前四个与 vaspkit **同名同布局** ⇒ 那个 MATLAB 脚本不改一行就能画。
逐字段规范见 [`docs/formats.md`](docs/formats.md) §13。

---

## 4b. 两个端到端实例：MoS₂（数学回归基线）与 1T-TaS₂（费米面演示）

> **MoS₂ 是数学回归基线，1T-TaS₂ 是费米面演示。**
>
> * **MoS₂（半导体）**：用来把**展开公式**钉在已验证原型上 —— 展开后的带隙
>   **1.622165 eV** 与已知的 MoS₂ 带隙一致 ⇒ 公式与相位约定是对的。但 MoS₂ 是
>   **半导体，本身没有费米面**：`--ef auto` 给出的是**价带顶的 Hofmann 面**
>   （A(E_F) 的峰值比 0.706），不是费米面。
> * **1T-TaS₂（金属）**：才有真正的费米面 —— 带 25/26/27 穿过费米能
>   （占据 `0.1885 .. 1.0198`）⇒ 费米面**非空**。

### 1T-TaS₂ 实测（服务器 / VASP 6.6.0 / 单核；12 k 点 116 s、48 k 点 416 s）

| 项 | 6×6×1 | 12×12×1 |
|---|---|---|
| NELECT / NKPT / NBANDS | 51 / 12 / 40 | 51 / 48 / 40 |
| DOSCAR 的 E_F | −1.377658 | −1.460379 |
| 表观 VBM / CBM（**假带隙**） | −1.568592 / −1.359436（gap 0.209156） | −1.471216 / −1.398389（gap 0.072827） |
| `--ef auto` 选出 | **−1.377658**（= DOSCAR 的 E_F ✓） | **−1.460379**（= DOSCAR 的 E_F ✓） |
| A(E_F)/A_max | **0.8770** | **0.9839** |
| A > 50 %·A_max 的网格点 | 1/36 | 4/144 |
| A > 1 %·A_max 的网格点 | 8/36 | 26/144 |
| 单 G 主导度 | 0.9035（未退化） | 0.966864 vs runner-up 0.966714（相对差 1.55e-4 ⇒ 退化区告警） |

```bash
unfoldfs unfold --dir run/ --ef auto --outdir fs/
```

```
（节选 —— 实际输出还有"估计电子数 / 已写出 / 已复制"几行）
相位约定      : s1 = +1, s2 = -1 (单 G 主导度 0.9035)
VBM / CBM     : -1.568592 / -1.359436 eV (金属: 表观值, 有限 k 网格的采样假象)
E_F           : -1.377658 eV  (--ef auto, --delta 0.05)
提示: 检测到费米穿越(金属) —— 有带的占据数在 k 上跨过 0.5; 金属的表观带隙是有限 k 网格的采样假象, 跳过带隙检查。
已写出        : fs/FS.cube
已写出        : fs/band.1 ... band.16
已写出        : fs/A(k,E).npz
已写出        : fs/FS.bxsf
```

> ⚠ **"费米面非空"只说明"这个能量上有态"，不等于"等值面的形状可信"** ——
> 展开谱权重在点群下天然不协变（对称相关点的谱函数最大差 **100 %**）；
> 默认输出已按**原胞完整点群**的轨道**对称化**（偏差降到 **0**），但对称化是**投影**：
> 绝对形状（面积/半径）仍不可当定量结果用。
> 见 §7 第 8 条与 [`docs/algorithms.md`](docs/algorithms.md) §15.7 / §15.8。

> ⚠ **`band.N` 的编号是能量升序**：`band.1` 是最低的深带（**不跨** E_F：
> 绝对能量 −16.2163 … −14.1675 eV，相对 E_F 是 −14.7559 … −12.7071）。
> 跨 E_F 的是 **band.5 – band.14**。
>
> ⚠ **NBANDS 的估值按 `_sv` 约定的 ZVAL 表**：1T-TaS₂ 的 `NELECT = 51`
> （3×Ta(5) + 6×S(6)），而内置表里**没有 Ta**（按缺省值计）⇒ `INCAR.suggested`
> 会提示"元素 Ta 不在价电子数表里, 按缺省值计"。用非 `_sv` 势时**必须**用
> `unfoldfs gen --nbands <N>` 覆盖，并按 `OUTCAR` 的 `NELECT` 核对。

（完整的实测数字与算法约定 → [`docs/algorithms.md`](docs/algorithms.md) §9.8。）

---

## 5. 为什么不用 vaspkit（实测的负结果）

VASPKIT 的能带反折叠与费米面两个模块**没有交点**：

| 模块 | 任务 | 必需输入 | 输出 | 含谱权重 |
|---|---|---|---|---|
| 能带反折叠 | 281/282/283-285 | **WAVECAR** + `TRANSMAT.in` + `KPOINTS_MAPPING_TABLE.in` + `KPATH.in` | `EBS.dat` (E, w) | 是 |
| 费米面 | 261/262/263-267 | **EIGENVAL + DOSCAR** + POSCAR | `FERMISURFACE.bxsf` | **否** |

实测证据：

1. `vaspkit -task 282` 在只有 INCAR/KPOINTS 的目录报
   `Error: The WAVECAR File NOT Found` ⇒ 282 走**平面波 (WAVECAR) 展开**，不是
   PROCAR 投影展开；
2. `vaspkit -task 262` 的日志只读 `EIGENVAL`（能量）+ `DOSCAR`（费米能级），
   纯能量插值出等值面，**没有任何展开权重的概念**；
3. 把手写网格 KPOINTS + 网格映射表喂给 282：流程能跑完（甚至输出 100 k × 120 带），
   但 `WEIGHT.grd` / `MOMENTUM.grd` **全为 0** ⇒ 网格模式不可用。

⇒ 不是操作问题，是两个模块**没有接口**。

---

## 6. 常见报错与对策

| 现象 / 报错 | 原因 | 对策 |
|---|---|---|
| `错误: VASP 算出的 k 点数 9 != grid_map 记录 27` | KPOINTS 不是 `gen` 写出的那份，或 INCAR 里 `ISYM ≠ 0`（VASP 把显式 k 表归约到 IBZ，k 点数与顺序都变） | 用 `gen` 写出的 `KPOINTS`；确认 INCAR 里 **`ISYM = 0`** |
| `UserWarning: 相位约定择优落在退化区: … 相对差仅 2.74e-05 < 1e-03` | winner 与 runner-up 的**单 G 集中度**几乎相同 ⇒ **选出的约定由噪声主导**。⚠ 后果**不是**"不可复现"那么轻：选错相位约定会让 **55%~65% 的 (k,带) 换候选归属** ⇒ **整张费米面错形**（实测 Pearson 0.16 vs 0.99、带能量最近距离 0.154 eV vs 2.05e-05 eV） | 换更密的网格（超胞 BZ 至少被 4×4 个点采样）；或人工核对候选归属；**不要直接采信该次结果的能带归属与费米面形状** |
| `错误: 单 G 主导度 0.4992 < 0.5 —— 展开没有把态集中到唯一一个候选 G 上` | 网格太粗：超胞 BZ 只被 2×2 个点采样，态在候选 G 之间几乎等权 | 加密网格。3×3×1 超胞用 `gen --grid 12 12 1`；注意 `--resolution` 选不出 6×6×1（见下） |
| `错误: 选出的 E_F = … 落在带隙内 … 离最近的态 … (> 2×sigma) —— 该能量上没有任何态, 产出的费米面是空的` | 半导体上 E_F 选到了带隙深处 | 半导体用 **`--ef vbm`**（VBM + delta）或 `--ef <数值>`（只告警、继续） |
| `提示: E_F = … 落在带隙内 … 但离最近的态只有 0.050000 eV (<= 2×sigma)` | 这是**正常**的：画的是价带顶附近的等值面 | 不用管；想换能量就显式 `--ef <数值>` |
| `提示: 检测到费米穿越(金属) …… 跳过带隙检查` | 有带的占据数在 k 上**跨过 0.5** ⇒ 这是**金属**（不是带隙） | **正常**：金属的"表观带隙"是有限 k 网格的采样假象，`--ef auto` 会取 DOSCAR 的 E_F |
| `警告: 最高带(第 N 带)的占据数最大为 … > 0 —— NBANDS 可能偏小, 占据态被截断` | 建议的 NBANDS 偏小 | 加大 NBANDS（`gen --nbands <N>`），并按 OUTCAR 的 `NELECT` 核对 |
| `已跳过: fs/FS.bxsf (… 只支持 Gamma 网格 …)` | 用 `--grid-type mp` 生成的 MP 网格，而 FermiSurfer 的 BXSF 读取器一律按 Γ 网格解释 | MP 网格改看 `FS.cube` / `band.N` / `A(k,E).npz`；要 BXSF 就用 Γ 网格 |
| `错误: 候选 G 数 N != 期望 D 个 … 请检查 m_eff 是否为整数矩阵、是否奇异` | `TRANSMAT.in` 不是整数矩阵或矩阵奇异 | 修 `TRANSMAT.in`；`TRANSMAT` 的行/列约定不用自己管（工具自动探测） |
| `错误: 满网格点数 36 != n_k·det(M) = 4×9` | 网格与超胞变换不自洽（点数不是 `comm(M⁻¹)` 的倍数） | 换一个 `--grid`，或用 `--resolution` 让工具自己选 |
| `错误: 画展开能带需要 matplotlib, 但当前环境没有装` | 只有 `unfoldfs band` 的 PNG 需要 matplotlib（**可选依赖**） | `pip install 'unfoldfs[plot]'`；或加 `--no-plot` 只出四个 vaspkit 文本文件 |
| `错误: 阶段 2 需要 …/KPATH.in` | `--dir` 里没有路径文件 | 用阶段 1（`unfoldfs band --kpath … --outdir <dir>`）生成，它会把 `KPATH.in` 复制过去 |
| `错误: 路径点 … 在展开结果里找不到对应的候选` | KPATH.in 与实际计算的 k 点不自洽（多半是 KPOINTS 不是本工具写的，或 INCAR 里 `ISYM ≠ 0`） | 用阶段 1 写出的 `KPOINTS`；确认 `ISYM = 0` |

> **关于 `valid == False` 的补位槽**：`A(k,E).npz` 里的 `band_e` / `band_w` 有补位槽
> （能量 `+inf`、权重 0），表示"该候选在这个 k 点上没有第 n 条带"。
> 它们**不会**进入任何输出文件（谱函数内部已排除），但你在 `npz` 上自己做
> `argmin` / 求最小值时**必须**用 `valid` 掩码排除 —— 否则会选中 `+inf` 槽。
> **不要手工改这些槽**。

---

## 7. 已知限制（务必读）

1. ~~**`--resolution` 选不出 6×6×1 这类网格**~~ —— **已修（修复轮 4）**：
   `grid.choose_grid_points` 现在对**对角 M** 用**逐维**约束 `M_ii | P_i`
   （非对角 M 仍是全局 `comm(M⁻¹)` —— 那里的逐维条件不成立）。
   ⇒ `M = diag(3,3,1)` 的 `--resolution` 自动选点从 6×6×3 变成 **6×6×1**
   （满网格点 108 → 36，**3 倍**机时）；`--grid 6 6 1` 这条显式路径一直可用。
2. **27 原子（3×3×1）超胞的展开目前不可靠**：6×6×1 网格的主导度 0.4992（被拒），
   12×12×1 网格能展开但**旧判据**下相位约定落在退化区 ⇒ 不可复现
   （⚠ 修复轮 9 换了判据：`margin` 现在比较的是**两个 `s1=−1` 的候选**，
   这两个数字是**旧判据**下测的，要重新评估需重跑）。
   **已验证的基线是 9 原子的 3×1×1 超胞**（主导度 0.8936）。
3. **没有开 SOC**（`ISPIN = 1`、无 `LSORBIT`）：带隙数值只能当**基线**，
   不能与实验直接比。实跑用的网格够做基线，**不够**做收敛的费米面。
4. **`FS.bxsf` 只支持 Γ 网格**（见 §6）。
5. **`FS.cube` 不携带等值面层级**：文件头第 2 行给出体数据最大值与建议层级
   （≈ 最大值的一半）。VESTA 的默认层级是 0 ⇒ 不看提示可能什么都画不出来。
6. **FermiSurfer 的 `.frmsf` 携带相对 E_F 的能量**（这样直接打开即可）；
   **费米速度着色是常量** —— 本工具不计算矩阵元。
7. **金属路径的 k 采样仍然稀**：1T-TaS₂ 的 12×12×1 实跑里相位约定落在**退化区**
   （**旧判据**下相对差 1.55e-4 < 1e-3；修复轮 9 换判据后这个 margin 的定义
   变了，要重新评估需重跑），6×6×1 只有 **1/36** 个网格点超过
   50 %·A_max ⇒ **E_F 的选取已修对**（§4b），但**费米面的形状**仍受 k 采样
   限制 —— 要定量形状必须把超胞 BZ 采得更密。
8. ⚠ **费米面的形状有对称性破缺（实测最大相对偏差 ~1.0）**：展开后的谱权重
   `A(k,E)` 在晶体点群下**天然不协变** —— 对称相关的两个 k 点，谱函数可以差到
   **100 %**（一个谱峰被映到 0）。**能带能量不受影响**（对称相关点的带能量
   吻合到 8e-6 eV），受影响的只是**权重在 k 上的分配**。
   ⇒ **修复轮 5**: 默认把 `A(k,E)` 按**点群轨道平均**（投影到对称不变子空间），
   实测偏差 **1.000e+00 → 0.000e+00**（MoS₂ 9.186e-01 → 0.000e+00，三个真实
   数据集），费米面等值面恢复对称。`--no-symmetrize` 可关掉（回到原始逐
   (K,G) 权重）。
   ⇒ **修复轮 6**: 对称化改用**原胞的完整点群** ⇒ **任意**超胞（包括破坏面内
   C3 的 3×1×1）都能折回原胞的对称性（实测 3×1×1 的 1T-TaS₂：
   `A(k,E_F)` 在 12 个操作下的相对离散度 **1.000e+00 → 0.000e+00**）。
   ⇒ **仍然要当心**: **逐带数据**（`band.N` / `FS.bxsf` / npz 的
   `band_e`/`band_w`）**不**对称化（对称相关点的带集合不同，见 §15.8.4）；
   对称化是**投影** ⇒ **绝对**形状（口袋面积、各向异性）仍不可当定量结果用。
   ⇒ **能量、带的连接、费的拓扑（有几个口袋）、E_F 的选取是可靠的**；
   等值面的**对称性**（在原胞完整点群下）现在也可靠。
   实测数字、机制与修法 → [`docs/algorithms.md`](docs/algorithms.md) §15 / §15.8。

9. ⚠ **相位约定的选择对采样敏感**（**修复轮 9 已改判据**）：真实路径
   （1T-TaS₂，60 个路径点）上，4 种约定的主导度只差 **0.1 %～0.3 %**，可换成
   另一条约定会让 **66 % 的 (k,带) 换候选归属**（总权重差 56 %）⇒ **旧判据
   （"单 G 主导度取最大"）分辨不出对错** ⇒ 这正是费米面错形的根因。
   **修复轮 9**: 判据换成"**有判别力**"的那个（`docs/algorithms.md` §7.2）——
   `s1`（副本相位符号）由**物理唯一确定**（副本相位必须抵消超胞 Bloch 因子
   `e^{+iK·R_T}` ⇒ `s1 = −1`），`s2`（Bloch character 符号）才是真正的约定，
   在剩下两个候选里按单 G 集中度择优（相对差 2.0 %～2.1 %，比噪声量级
   1e-3 大 20 倍）⇒ 三组真实数据（MoS₂ fixture / 1T-TaS₂ 3×1×1 / √3×√3）
   都选出 `(s1=−1, s2=+1)`。
   ⇒ 退化区告警仍然存在（比较的是两个 `s1=−1` 的候选），但它的后果要说清：
   选错约定 = **整张费米面错形**，不是"不可复现"（见 §7 第 12 条）。
   → [`docs/algorithms.md`](docs/algorithms.md) §7.2 / §15.6 / §16.4。

10. ⚠ **展开能带（`unfoldfs band`）同样受上面的协变性缺陷影响**：**能带能量可信**
   （对称相关点吻合到 8e-6 eV），**权重的 k 依赖原本不可信** ⇒ 能带的"形状/对称性"
   不能当定量结果用。默认对路径上**对称相关**的点做轨道平均（`--no-symmetrize`
   可关），但泛路径点的轨道离开路径 ⇒ 通常是**无操作**。
   另有一条**格式层面的**区别：`WEIGHT.grd` 里的权重是
   **Popescu G 分辨权重**，不是 vaspkit 282 的 WAVECAR 平面波展开 |c(k)|²
   ⇒ 两者的数值不可比（见 §2.3 与 `docs/formats.md` §13.5）。

11. ⚠ ★ **对称化修好了"对称性"，但没修好"绝对谱形状"**（修复轮 7 的实测）：
   1T-TaS₂ 的 3×1×1 超胞、完整点群对称化之后 —— 协变性偏差
   **1.000e+00 → 0.000e+00**、`A(k,E_F)` 的轨道离散度
   **1.000e+00 → 0.000e+00**（**对称性已恢复** ✓），但与原胞直接计算的
   Pearson 相关只有 **0.313 → 0.720**（**绝对谱形状仍不同** ✗）。
   ⇒ **面内整数倍超胞（3×1×1）的绝对形状不能当定量费米面用**。
   ⇒ ⚠ ⚠ **更正（修复轮 8/9 的实测推翻上一句）**：上面那个 0.72 是在
   **修复轮 7 当时错选的相位约定**下测的；修好相位约定之后，3×1×1 的
   `Pearson(A(k,E_F))` 是 **0.9890** ✅ ⇒ **3×1×1 现在可以当定量费米面用**。
   ⇒ **这一条必须连同下一条（第 12 条）一起读** ✓。
   → [`docs/algorithms.md`](docs/algorithms.md) §14 第 11 条。

12. ⚠ ⚠ **相位约定选错 = 整张费米面错形**（修复轮 8 的定位 + 修复轮 9 的修法）：
   上一条说"要做定量的费米面请用 √3×√3 R30°"—— **与实测相反**，而且真正
   的原因不是超胞而是**相位约定选错**。1T-TaS₂ 的 48×48×1 三组实跑
   （对称化之后，与原胞直接计算比；修复轮 9 的判据）：

   | | 原胞直通 | 3×1×1 超胞 | √3×√3 R30° 超胞 |
   |---|---|---|---|
   | 选出的相位约定 | —（直通模式无约定） | `(s1=−1, s2=+1)` | `(s1=−1, s2=+1)` ✅ |
   | `Pearson(A(k,E_F))` | 1.0000 | **0.9890** ✅ | **0.9941** ✅（修复前 0.1556 ❌） |
   | 带能量与原胞同一点的最近距离（中位） | 0 | **1.10e-05 eV** ✅ | **2.10e-05 eV** ✅（修复前 8.18e-02 eV ❌） |
   | 单 G 集中度 | —（直通模式无此量） | 0.9915 | 0.7699 |
   | 完整点群下的协变性偏差 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

   ⇒ **旧判据**（"单 G 主导度取最大"）在 √3 上选到了错的约定
   （四种约定的主导度只差 0.0028，纯噪声）⇒ **55 %～65 % 的 (k,带) 换候选
   归属** ⇒ 整张费米面错形（Pearson 0.16 vs 0.99）。
   ⇒ **修复轮 9** 的判据（`s1` 物理唯一 + `s2` 靠集中度）在三组真实数据上
   都选出 `(−1,+1)` ⇒ **3×1×1 与 √3×√3 都能给出定量费米面**。
   ⇒ **给用户的建议**：定量费米面用 3×1×1 这类面内整数倍超胞即可
   （对称化会用原胞完整点群把 C3/C6 补回来）；√3×√3 现在同样可用。
   → [`docs/algorithms.md`](docs/algorithms.md) §7.2 / §14 第 11 条。

13. **多源求和（修复轮 8）**：原胞的谱函数是**所有**落到同一个原胞 k' 的超胞态
   的投影之和 ⇒ 同一个 k' 被几个超胞 k 覆盖，权重就该是几份之和。`unfoldfs`
   现在按**多源**累加（多个源落在同一个网格点时，带按能量对齐合并、权重相加）。
   ⚠ 但 `unfoldfs gen` 写出的 KPOINTS 是**折叠网格** —— 实测每个原胞网格点
   恰好**一个**源（48×48×1 的两个超胞都是 min = max = 1）⇒ 在当前流程里
   多源求和是**恒等变换**（权重和、带能量、Pearson、协变性全部逐位不变）。
   ⇒ 要让多源真正生效，超胞计算必须覆盖**整个原胞第一布里渊区**
   （k 列表含相差一个超胞倒格矢的副本）—— 那是**计算流程**的改动。
   ⇒ 多源求和与**对称化**是两道**正交**的处理：前者补"被丢掉的权重"，
   后者补"点群下的不协变"。
   → [`docs/algorithms.md`](docs/algorithms.md) §8.7。

14. ⚠ **VASP 6.6.0 读到含重复 k 点的显式列表会崩**（修复轮 7 的实测）：本工具
   写出的 KPOINTS 已按周期距离去重（表头计数 = 实际行数）；但**用户自己拼的
   KPOINTS 不受此保护** —— 手工写重复 k 点仍会让 VASP 崩。

15. ⚠ **画费米面请用 `FS.bxsf` / `FS.cube`（FermiSurfer / XCrySDen）**，
    不要直接画 `A(k,E_F)` 的热图（实测，1T-TaS₂ 原胞 96×96×1）：
   * **谱权重**在费米面上是**均匀**的：实测 9216 个 k 点里过 E_F 的带数分布是
     **{0: 7925, 1: 1290, 2: 1}** ⇒ 费米面上每点只有 1 条带穿过 ⇒ 谱权重恒为 1；
     那条带的**局域投影权重** `Σ_{ion,lm}|c_proj|²` 也几乎是常数
     (**0.732–0.867**，中位 0.776) ⇒ **费米面本该是一个颜色** ✓；
   * 热图里"亮度不匀"是**渲染**效应：`A(k,E_F) = w·exp(−ΔE²/2σ²)`，ΔE 是该 k 点
     上带离 E_F 的距离（σ=0.06 eV）⇒ 擦着 E_F 过去的地方就暗；再加上 12 个点群像的
     **轨道平均**与分箱平均 ⇒ 网格加密（48×48 → 96×96）几乎不改变这个分布
     （展宽因子中位 0.485 → 0.509）；
   * ⇒ `FS.bxsf` / `FS.cube` 走的是 **E_F 处的等值面** ⇒ 出来是**一条干净的曲线、
     一个颜色** ✓（FermiSurfer / XCrySDen 就是这么画的）。

完整的限制清单与全部实测数字 → [`docs/algorithms.md`](docs/algorithms.md) §14。

---

## 8. 仓库结构

```
unfoldfs/            本工具（可安装的包）
  geometry.py        结构解析 / 变换约定探测 / 候选 G / 公共对称集
  grid.py            均匀网格 / 对称轨道分解 / k 点去重 / KPOINTS 写出
  vaspio.py          EIGENVAL / DOSCAR / PROCAR(LORBIT=12) 解析
  unfold.py          Popescu 投影展开 / 副本相位还原平均 / 候选重组
  expand.py          对称扩展到均匀网格 / 谱函数 A(k,E)
  fsout.py           FS.cube / FS.bxsf / band.N / A(k,E).npz 写出
  band.py            KPATH.in 解析 / 展开能带组装 / vaspkit 四件套写出 / 画图
  cli.py             gen / unfold / band / info
docs/
  algorithms.md      算法、约定与实测数字
  formats.md         输出格式的逐字段规范
tests/               258 个用例通过 + 1 个 xfail（含 tests/data/ 下的合成 fixture）
unfold_procar.py     仓库根的**已验证原型**（与本工具的全路径交叉核对基准，
                     max|Δw| = 4.99e-09、max|ΔE| = 0）
```

`unfold_procar.py` 与 `make_figures.py` 等根目录脚本是**探索阶段的原型**，
不是 `unfoldfs` 包的一部分（不参与测试、不参与安装）。
