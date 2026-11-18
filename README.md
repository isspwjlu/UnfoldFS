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
python -m unfoldfs.cli info --help
```

### 依赖

| 项 | 版本要求 | 说明 |
|---|---|---|
| Python | **≥ 3.9** | 开发机实测 3.11.7；服务器 master 实跑用的是 3.9.13 |
| numpy | ≥ 1.20 | |
| scipy | ≥ 1.6 | 周期 KD-tree（网格点匹配） |
| spglib | ≥ 1.15 | 原胞点群（spglib ≥ 2.0 的 dict API） |
| VASP | **6.6.0**（实测） | 只用来产生输入（`EIGENVAL` / `PROCAR` / `DOSCAR`） |
| VESTA 3.5.8 / XCrySDen 1.5.60 / FermiSurfer 2.2.1 | 可选 | **消费方**，用来打开输出的费米面文件 |

`unfoldfs` 本身**不画图** ⇒ **不需要** matplotlib / scikit-image。

> 服务器 master 的实测环境：Python 3.9.13（numpy 1.23.5 / scipy 1.13.1 /
> spglib 2.0.2）、VASP `vasp.6.6.0 06Mar2026`。

---

## 2. 三个子命令

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

### 2.2 `unfoldfs unfold` —— 反折叠并写费米面文件

```
unfoldfs unfold --dir DIR [--supercell SUPERCELL] [--primcell PRIMCELL] [--transmat TRANSMAT]
                 [--grid-map GRID_MAP] [--ef EF] [--delta DELTA] [--sigma SIGMA]
                 [--strict-dominance] [--outdir OUTDIR]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--dir` | ✔ | — | VASP 计算目录（须含 `EIGENVAL` / `PROCAR` / `DOSCAR` / `grid_map.npz`） |
| `--supercell` | | `<dir>/POSCAR` | |
| `--primcell` | | `<dir>/PRIMCELL.vasp` | |
| `--transmat` | | `<dir>/TRANSMAT.in` | |
| `--grid-map` | | `<dir>/grid_map.npz` | |
| `--ef` | | `auto` | `auto` \| `vbm` \| `doscar` \| **数值(eV)** |
| `--delta` | | `0.05` | 半导体时 `VBM + delta` 的偏移（eV） |
| `--sigma` | | `0.06` | 谱函数展宽（eV） |
| `--strict-dominance` | | 关 | 要求**每个态**的单 G 主导度 ≥ 0.5（严格模式） |
| `--outdir` | | `fs` | 费米面文件输出目录 |

`--ef` 的语义：

| 值 | 含义 |
|---|---|
| `auto` | 金属取 DOSCAR 的 E_F；**有带隙**且 DOSCAR 的 E_F 落在带隙里 ⇒ 取 `VBM + delta` |
| `vbm` | 总是 `VBM + delta`（半导体的正确画法） |
| `doscar` | 总是 DOSCAR 里的 E_F |
| 数值 | 直接当作 E_F（绝对值，eV）；**只告警、不阻断** |

`unfold` 会打印选出的相位约定 `(s1, s2)` 与单 G 主导度、VBM/CBM、选出的 E_F，
并把它们写进输出文件的头部与 `A(k,E).npz` 的元信息。

### 2.3 `unfoldfs info` —— 查看网格与对称信息

```
unfoldfs info --dir DIR [--grid-map GRID_MAP]
```

打印网格规格、满网格点数、不可约点数、实际计算的 k 点数、朴素超胞网格点数
（∏(P_i/M_i)）、相对原胞满网格的比例、`det(M)`、KPOINTS 权重和。

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

### ③ 反折叠

```bash
unfoldfs unfold --dir run/ --outdir fs/ --ef vbm
```

```
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
| `UserWarning: 相位约定择优落在退化区: … 相对差仅 2.74e-05 < 1e-03` | winner 与 runner-up 的主导度几乎相同 ⇒ **选出的约定由噪声驱动，结果不可复现** | 换更密的网格（超胞 BZ 至少被 4×4 个点采样）；或人工核对候选归属；不要直接采信该次结果 |
| `错误: 单 G 主导度 0.4992 < 0.5 —— 展开没有把态集中到唯一一个候选 G 上` | 网格太粗：超胞 BZ 只被 2×2 个点采样，态在候选 G 之间几乎等权 | 加密网格。3×3×1 超胞用 `gen --grid 12 12 1`；注意 `--resolution` 选不出 6×6×1（见下） |
| `错误: 选出的 E_F = … 落在带隙内 … 离最近的态 … (> 2×sigma) —— 该能量上没有任何态, 产出的费米面是空的` | 半导体上 E_F 选到了带隙深处 | 半导体用 **`--ef vbm`**（VBM + delta）或 `--ef <数值>`（只告警、继续） |
| `提示: E_F = … 落在带隙内 … 但离最近的态只有 0.050000 eV (<= 2×sigma)` | 这是**正常**的：画的是价带顶附近的等值面 | 不用管；想换能量就显式 `--ef <数值>` |
| `警告: 最高带(第 N 带)的占据数最大为 … > 0 —— NBANDS 可能偏小, 占据态被截断` | 建议的 NBANDS 偏小 | 加大 NBANDS（`gen --nbands <N>`），并按 OUTCAR 的 `NELECT` 核对 |
| `已跳过: fs/FS.bxsf (… 只支持 Gamma 网格 …)` | 用 `--grid-type mp` 生成的 MP 网格，而 FermiSurfer 的 BXSF 读取器一律按 Γ 网格解释 | MP 网格改看 `FS.cube` / `band.N` / `A(k,E).npz`；要 BXSF 就用 Γ 网格 |
| `错误: 候选 G 数 N != 期望 D 个 … 请检查 m_eff 是否为整数矩阵、是否奇异` | `TRANSMAT.in` 不是整数矩阵或矩阵奇异 | 修 `TRANSMAT.in`；`TRANSMAT` 的行/列约定不用自己管（工具自动探测） |
| `错误: 满网格点数 36 != n_k·det(M) = 4×9` | 网格与超胞变换不自洽（点数不是 `comm(M⁻¹)` 的倍数） | 换一个 `--grid`，或用 `--resolution` 让工具自己选 |

> **关于 `valid == False` 的补位槽**：`A(k,E).npz` 里的 `band_e` / `band_w` 有补位槽
> （能量 `+inf`、权重 0），表示"该候选在这个 k 点上没有第 n 条带"。
> 它们**不会**进入任何输出文件（谱函数内部已排除），但你在 `npz` 上自己做
> `argmin` / 求最小值时**必须**用 `valid` 掩码排除 —— 否则会选中 `+inf` 槽。
> **不要手工改这些槽**。

---

## 7. 已知限制（务必读）

1. **`--resolution` 选不出 6×6×1 这类网格**：`grid.choose_grid_points` 用**全局**
   `comm(M⁻¹)` 约束三个维度，而对角 M 的正确条件是逐维 `M_ii | P_i` ⇒ 对
   `M = diag(3,3,1)`，P3 被强制成 3 的倍数。**用 `gen --grid 6 6 1` 绕过**
   （但 3×3×1 超胞因此要多付约 3 倍机时）。
2. **27 原子（3×3×1）超胞的展开目前不可靠**：6×6×1 网格的主导度 0.4992（被拒），
   12×12×1 网格能展开但相位约定落在退化区 ⇒ 不可复现。
   **已验证的基线是 9 原子的 3×1×1 超胞**（主导度 0.8936）。
3. **没有开 SOC**（`ISPIN = 1`、无 `LSORBIT`）：带隙数值只能当**基线**，
   不能与实验直接比。实跑用的网格够做基线，**不够**做收敛的费米面。
4. **`FS.bxsf` 只支持 Γ 网格**（见 §6）。
5. **`FS.cube` 不携带等值面层级**：文件头第 2 行给出体数据最大值与建议层级
   （≈ 最大值的一半）。VESTA 的默认层级是 0 ⇒ 不看提示可能什么都画不出来。
6. **FermiSurfer 的 `.frmsf` 携带相对 E_F 的能量**（这样直接打开即可）；
   **费米速度着色是常量** —— 本工具不计算矩阵元。

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
  cli.py             gen / unfold / info
docs/
  algorithms.md      算法、约定与实测数字
  formats.md         输出格式的逐字段规范
tests/               196 个用例（含 tests/data/ 下的合成 fixture）
unfold_procar.py     仓库根的**已验证原型**（与本工具的全路径交叉核对基准，
                     max|Δw| = 4.99e-09、max|ΔE| = 0）
```

`unfold_procar.py` 与 `make_figures.py` 等根目录脚本是**探索阶段的原型**，
不是 `unfoldfs` 包的一部分（不参与测试、不参与安装）。
