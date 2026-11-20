# unfoldfs —— 超胞能带反折叠 → 原胞 BZ 内的费米面

`unfoldfs` 把**超胞 VASP 计算**的结果反折叠回**原胞第一布里渊区**，给出带谱权重的
费米面（`A(k, E_F)` 的等值面），并写成能被 **VESTA / XCrySDen / FermiSurfer**
直接打开的文件。

* 算法、约定与全部实测数字 → [`docs/algorithms.md`](docs/algorithms.md)
* 输出文件的逐字段规范 → [`docs/formats.md`](docs/formats.md)

---

## 0. 仓库与论文

* 代码: **https://github.com/isspwjlu/unfold_fs**
* 论文: *Unfolded Fermi surfaces from supercell band calculations: extracting,
  validating, and rendering the Fermi surface of a supercell spectral function*
  (arXiv: 待发布)
* 许可: MIT (`LICENSE`)

开源版本不含测试数据、计算结果与论文稿件；依赖它们的测试会自动跳过。
复现论文算例需自己跑 VASP（脚本见 `scripts/cdw/`、`scripts/figures/`）。

## 1. 安装

```bash
pip install -e .
```

装完就有 `unfoldfs` 命令。不想装包也可以直接用模块入口
（`python -m unfoldfs.cli --help`，所有命令等价）。

| 依赖 | 要求 | 说明 |
|---|---|---|
| Python | ≥ 3.9 | |
| numpy / scipy | ≥ 1.20 / ≥ 1.6 | 周期 KD-tree（网格点匹配） |
| spglib | **≥ 2.0** | 原胞点群（代码用 2.x 的 dict API） |
| matplotlib | **可选** | 只有 `unfoldfs band` 的 PNG 需要（`pip install 'unfoldfs[plot]'`） |
| VASP | 6.6.0（实测） | 只用来产生输入（`EIGENVAL` / `PROCAR` / `DOSCAR`） |

## 2. 四个子命令

```bash
unfoldfs gen    --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in
                [--resolution R] [--grid P1 P2 P3] [--grid-type {gamma,mp}]
                [--outdir DIR] [--nbands N] [--encut E] [--kpar K]
unfoldfs unfold --dir DIR [--ef {auto,vbm,doscar,数值}] [--delta D] [--sigma S]
                [--strict-dominance] [--no-symmetrize] [--outdir DIR]
unfoldfs band   --supercell … --primcell … --transmat … --kpath KPATH.in --outdir DIR
                # 阶段 2: unfoldfs band --dir DIR --outdir DIR2 [--e-min E] [--e-max E]
                #          [--mirror] [--no-plot] [--no-symmetrize]
unfoldfs info   --dir DIR [--grid-map GRID_MAP]
```

* **`gen`** —— 生成折叠网格与 `KPOINTS`（已去重，**VASP 6.6.0 读到重复 k 点会崩**），
  外加 `grid_map.npz`（`unfold` 要用）、`INCAR.suggested`（含 `ISYM = 0`、
  `LORBIT = 12`）。
* **`unfold`** —— 反折叠并写费米面文件。`--ef auto` 先判金属性：金属取 DOSCAR 的
  E_F，半导体取 `VBM + delta`。
* **`band`** —— 沿 k 路径的展开能带，分「读 KPATH.in 写输入」与「读 VASP 输出展开」
  两阶段。
* **`info`** —— 打印网格规格、不可约点数、对称操作的保留/剔除原因、`det(M)` 等。

每个命令的完整参数见 `--help`。

### `unfold` 的一个默认行为

展开得到的谱权重 `A(k,E)` 在晶体点群下**天然不协变**（对称相关的两个 k 点，谱函数
可以差到 100 %）⇒ 费米面的等值面相对晶体对称性是畸变的。**默认把 `A(k,E)` 按原胞
完整点群做轨道平均**，实测相对偏差 1.000e+00 → 0.000e+00。
`--no-symmetrize` 关闭。逐带数据（`band.N` / `FS.bxsf` / npz 的 `band_e`/`band_w`）
**不**对称化 —— 对称相关点的"带集合"本就不同。

## 3. 典型工作流

```bash
# ① 生成网格与 KPOINTS
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp \
              --transmat TRANSMAT.in --grid 6 6 1 --outdir run/

# ② 跑 VASP（INCAR.suggested 里的 ISYM = 0 / LORBIT = 12 必须保留）
cp run/INCAR.suggested run/INCAR
cp POTCAR run/POTCAR
cd run && mpirun -np 8 vasp_std

# ③ 反折叠
unfoldfs unfold --dir run/ --outdir fs/ --ef vbm

# ④ 打开费米面
```

| 工具 | 打开哪个文件 | 看什么 |
|---|---|---|
| **VESTA** | `FS.cube` | `A(k, E_F)` 的体数据等值面（建议层级写在文件头第 2 行） |
| **XCrySDen** | `FS.bxsf` | 逐带能量网格的费米面（**只支持 Γ 网格**） |
| **FermiSurfer** | `band.N` | 逐带费米面（能量相对 E_F，直接打开即可） |
| Python | `A(k,E).npz` | 完整谱函数 + 元信息，二次分析用 |

展开能带（另一条流水线）：

```bash
unfoldfs band --supercell POSCAR --primcell PRIMCELL.vasp \
              --transmat TRANSMAT.in --kpath KPATH.in --outdir run/
# ... 跑 VASP（同上）...
unfoldfs band --dir run/ --outdir bandfs/
```

`bandfs/` 里的 `ENERGY.grd` / `MOMENTUM.grd` / `WEIGHT.grd` / `KLABELS`
**就是 vaspkit 的文件名与布局** ⇒ 现有的 MATLAB 绘图脚本不改一行就能用。

## 4. 输出文件

| 文件 | 消费者 | 内容 |
|---|---|---|
| `FS.cube` | VESTA / XCrySDen / VMD | Gaussian cube：体数据 = `A(k, E_F)`，文件头给出 E_F 与建议等值面层级 |
| `FS.bxsf` | XCrySDen / FermiSurfer | 逐带能量网格（**只支持 Γ 网格**） |
| `band.N` | FermiSurfer | `.frmsf` 逐带文件（每文件一条带，能量相对 E_F） |
| `A(k,E).npz` | 自用 | 谱函数体数据 + 网格 + 逐带能量/权重 + 元信息 |
| `ENERGY/MOMENTUM/WEIGHT.grd`, `KLABELS` | vaspkit / MATLAB | 展开能带的四个同名同布局文件 |

> `band.N` 的编号按**能量升序**：`band.1` 是最低的深带，**不是**第一条穿过 E_F 的带。
> npz 里的 `band_e`/`band_w` 有 `+inf` 补位槽，二次分析时必须用 `valid` 掩码排除。

## 5. 几个必须知道的坑

| 现象 | 对策 |
|---|---|
| `错误: VASP 算出的 k 点数 N != grid_map 记录 M` | INCAR 里 `ISYM ≠ 0`（VASP 把显式 k 表归约到 IBZ）⇒ 必须 `ISYM = 0` |
| PROCAR 里没有相位块 | `LORBIT = 11` 不写相位 ⇒ 必须 `LORBIT = 12` |
| `警告: NBANDS 可能偏小` | 按 OUTCAR 的 `NELECT` 核对，用 `gen --nbands N` 覆盖 |
| `错误: Error reading KPOINTS file` | 用了 `ISMEAR ≤ -4`（四面体法要求 VASP 自建网格）⇒ 用 `ISMEAR = 1` + `SIGMA = 0.2` |
| `错误: 单 G 主导度 0.49 < 0.5` | 网格太粗，超胞 BZ 采样不足 ⇒ 加密网格 |
| `错误: 选出的 E_F 落在带隙内 …` | 半导体上 E_F 选错 ⇒ 用 `--ef vbm` 或 `--ef <数值>` |
| ⚠ **选错相位约定 = 整张费米面错形**（不是"不可复现"） | 相位约定由 `s1`（物理唯一）+ `s2`（按单 G 集中度择优）定；退化区告警时**不要**采信该次结果的费米面形状 ⇒ 加密网格 |

## 6. 已知限制（要点）

1. **没有开 SOC**（`ISPIN = 1`）—— 带隙数值只能当基线。
2. 金属路径的 k 采样仍稀 ⇒ 要定量的**费米面形状**必须把超胞 BZ 采得更密；
   能量、带的连接、口袋拓扑与 E_F 的选取是可靠的。
3. 逐带数据不做对称化 ⇒ 逐带权重不可直接当定量结果。
4. **画费米面请用 `FS.bxsf` / `FS.cube`**，不要直接画 `A(k,E_F)` 的热图
   （后者给的是高斯脊的两壁，不是脊顶）。
5. `unfoldfs band` 的 `WEIGHT.grd` 是 **Popescu G 分辨权重**，与 vaspkit 282 的
   WAVECAR 平面波展开 |c(k)|² 物理含义不同、数值不可比。

完整清单与全部实测数字 → [`docs/algorithms.md`](docs/algorithms.md) §14。

## 7. 仓库结构

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
docs/                algorithms.md（算法与实测数字）/ formats.md（格式规范）
tests/                单元与集成测试
scripts/figures/      出图脚本（画法约定见其 README）
scripts/cdw/          CDW 态算例（建超胞 / 查结构 / 对比展开结果）
examples/             归档区：探索期原型 + 服务器算例脚本（不属于包）
```

`scripts/` 与 `examples/` 下的脚本不属于 `unfoldfs` 包（不参与安装）。
