# 反折叠费米面 (unfolded Fermi surface) 计算方案

> 目标：超胞计算结果 → **原胞 BZ 内**的费米面（带谱权重）。
> 本目录是调研 + 实测记录；脚本已在 master (210.73.16.15) 上跑通端到端。

---

## 1. 为什么 VASPkit 做不到（已在 master 实测）

VASPKIT 的两个模块机制完全不同，**没有交点**：

| 模块 | 任务 | 必需输入 | 输出 | 含谱权重 |
|---|---|---|---|---|
| 能带反折叠 | 281/282/283-285 | **WAVECAR** + `TRANSMAT.in` + `KPOINTS_MAPPING_TABLE.in` + `KPATH.in` | `EBS.dat` (E, w) | 是 |
| 费米面 | 261/262/263-267 | **EIGENVAL + DOSCAR** + POSCAR | `FERMISURFACE.bxsf` | **否** |

实测证据：

1. `vaspkit -task 282` 在只有 INCAR/KPOINTS 的目录报 `Error: The WAVECAR File NOT Found`
   → 282 走**平面波 (WAVECAR) 展开**，不是 PROCAR 投影展开。
2. `vaspkit -task 262` 日志只读 `EIGENVAL`（能量）+ `DOSCAR`（费米能级），纯能量插值出等值面，
   **没有任何展开权重的概念**。
3. 把手写网格 KPOINTS + 网格映射表喂给 282：流程能跑完（甚至输出 100 k × 120 带），
   但 `WEIGHT.grd` / `MOMENTUM.grd` 全为 0，权重基本为 0 → **网格模式不可用**。
   （附注：NKPTS 来自 `KPATH.in` 的路径段数；把 KPATH.in 改成 N 段可让 282 输出 N 个 k 点，
   但权重仍是 0。）

结论：不是操作问题，是 VASPkit 两个模块没有接口。

---

## 2. 反折叠费米面的定义

超胞 BZ 是原胞 BZ 的子区（体积 1/|det M|）。反折叠费米面即原胞 BZ 内的谱函数

```
A(k, E) = Σ_G Σ_n  w(k+G, n) · δ(E − E_n(k+G))
```

在 `E = E_F` 处的等值面。`G` 取遍 |det M| 个候选（把原胞 BZ 切成 |det M| 块）。

---

## 3. 路线 A（推荐，已跑通）：稠密网格超胞 + 投影法展开

### 3.1 关键 INCAR

```bash
LWAVE  = .TRUE.   # 若用平面波法(vaspkit 282 / bands4vasp)必须
LORBIT = 12       # ★ 投影法必须 LORBIT=12（见 3.3）
ENCUT  = 320
ISMEAR = 0
SIGMA  = 0.05
NBANDS = 120      # >= det(M) × 原胞带数（3×3×1 MoS2: 9×17=153）
KPAR   = 8        # 大超胞务必开 k 点并行
LREAL  = Auto
```

k 点必须是**稠密网格**（不是高对称路径）。网格密度按原胞 BZ 的费米面分辨率要求定；
`det(M)` 越大，超胞网格需要越密。

### 3.2 ★ LORBIT=12 而不是 11（VASP 5.4.4 / 6.x 实测源码确认）

`vasp.5.4.4/src/sphpro.F` 第 1096-1104 行：

```fortran
IF (LORBIT==11) THEN
   WRITE(IUP,'(A)')'PROCAR lm decomposed'            ! 无相位！
ELSE IF (LORBIT==12) THEN
   WRITE(IUP,'(A)')'PROCAR lm decomposed + phase'    ! ★ 有相位
ELSE IF (LORBIT==13) THEN ...
ELSE IF (LORBIT==14) THEN ...
```

**LORBIT=11 不写相位，LORBIT=12 才写。**（很多网上资料写反了）

LORBIT=12 的 PROCAR 每个 band 的结构（实测）：

```
band     1 # energy  -16.17817627 # occ.  2.00000000
ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot
    1  0.015  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.015   ← 模方块(11 字段)
   ...
tot    0.683  0.000  0.010  0.000  0.000  0.000  0.000  0.000  0.000  0.693
ion          s             py             pz             px             dxy            dyz            dz2            dxz          dx2-y2
    1  0.079  0.083  -0.000 -0.000   0.000  0.000  -0.000 -0.000   0.000  0.000  -0.000 -0.000   0.004  0.004   0.000  0.000   0.000  0.000   0.013   ← ★复投影块(20 字段)
```

复投影块 = 每个离子的**复投影系数**（Re, Im 成对，末位是 |c|²）。
注意它是**晶胞周期规范**（同一原胞的 9 个副本复值完全相同，不含 e^{iK·R}）。

### 3.3 展开公式与自检

```
p_i,lm(k') = (1/N_copies) Σ_T c_{i+T,lm} · exp(−i (k'−K)·T)     # 副本平均 + 相位还原
w(k')      = | Σ_{i,lm} p_{i,lm}(k') · exp(i k'·r_i) |²          # Bloch character
k'         = K + G,   G ∈ {0,…,M_d−1} 的各种组合（|det M| 个）
```

自检（脚本自动做）：每个超胞态在 9 个 G 候选上**只有一个主导**（max/Σ ≈ 0.85~1.0，
其余降到 1e-33），否则相位约定或副本匹配有问题。

### 3.4 输出

- `unfolded_bands.dat` —— (kx ky kz, E, w)，原胞 BZ 笛卡尔坐标 (2π/Å)
- `spectral_function.dat` —— (kx ky kz, E, A)
- `unfolded_specfunc_slices.pdf` —— A(k,E) 的能量切片（费米面演化）
- `unfolded_fermi_surface.pdf` —— E_F 处 A(k,E_F) 二维图
- `bands_unfolded/band*.dat` —— 逐带 (k, E, w)，可直接喂 FermiSurfer

---

## 4. VASPKIT 映射表格式（官方文档未写，实测逆向得到）

`KPOINTS_MAPPING_TABLE.in`：

```
   Kmesh for primitive cell                             Kmesh for super cell                                 G vectors for super cell
   0.000000000000   0.500000000000   0.000000000000     0.000000000000  -0.500000000000   0.000000000000     0   2   0
```

- **定宽格式**，列宽必须与官方一致，否则 Fortran `input conversion error`：
  `"   %.12f   %.12f   %.12f     %.12f  %.12f   %.12f     %d   %d   %d"`
- **G 必须是非负整数**（官方只用 0/1/2）。用负数 G 会破坏列对齐 → 解析失败。
- **折叠关系**：`k_prim = (k_super + G) · M⁻¹`（用官方 MoS2 3×3 算例 159 行逐行核对过）。
- **行数 = n_kpoints × det(M)**，少一行 → `end-of-file during read`。
- KPOINTS 首行是 VASPKIT 自定义格式（`Parameters to Generate KPOINTS...`），
  与 262 需要的标准 VASP 格式**不通用**。

`make_unfold_mesh.py` 负责生成网格 KPOINTS + 映射表 + 几何 JSON。

---

## 5. 路线 B：官方 bands4vasp

`/data/wjlu/program/bands4vasp-main`（含官方费米面范例、Fortran 源码、install.sh）。
功能最全（2D/3D 费米面、Fermi 矢量、谱函数），但：

- **必须用打补丁的 VASP**：`QuantumMaterialsModelling/UnfoldingPatch4vasp`
  仓库目前只有 `vasp.6.2.1_U-patch.21.08`（2021），master 上是 5.4.4/6.4.2/6.5.1/6.6.0
  → 需试 patch 能否套上。
- FS 模式工作流：k 点拆成上百条"径向线"，每条一个目录跑 VASP，最后 `b4vasp --fermi %calc`。
- 产物：`FERMISURFACE.dat`、`FERMISURFACE.specfun`、`FERMIROOTS.dat`。

## 6. 路线 C：WannierTools

master 有 `wannier_tools-2.7.2` 源码（`unfolding.f90` + `fermisurface.f90` 已编译成 .o，
未链接出可执行文件）+ `wannier90-3.1.0`。

源码核对（`src/main.f90`）：unfold 走 `unfolding_kpath` / `unfolding_kplane`，
费米面走独立的 `BulkFS` → **两条代码路径不能直接组合**，无现成的反折叠费米面任务。
折中：`unfold` 的 kplane 模式出一系列 2D 谱函数切片再堆叠。

## 7. 完美超胞的"几何等价"捷径

超胞 FS 按原胞倒格矢复制到原胞 BZ，就得到原胞 FS 的全部面片；无缺陷/无掺杂时几何上精确。
反折叠权重的额外价值在于区分"干净原胞态"与"超胞引入的额外态（缺陷/掺杂态）"。

```bash
vaspkit 261      # 稠密网格 KPOINTS（vaspkit 自己的格式）
# VASP 静态 -> EIGENVAL + DOSCAR
vaspkit 262      # -> FERMISURFACE.bxsf，XCrySDen / FermiSurfer 渲染
```

---

## 8. 本机实测（master，MoS2 3×3×1 超胞作示例）

| 步骤 | 命令 | 结果 |
|---|---|---|
| 造超胞 | vaspkit 401 + `TRANSMAT.in` (3×3×1) | 27 原子超胞 / 3 原子原胞 |
| 网格 | `make_unfold_mesh.py --n1 10 --n2 10 --n3 1` | 100 k + 900 行映射表 |
| 计算 | VASP 5.4.4, LORBIT=12, KPAR=8, 32 核 | 11 min，PROCAR 78 MB |
| 展开 | `unfold_procar.py` | 108000 条 (k,E,w)；单 G 主导度 0.85~1.0 |
| 出图 | `make_figures.py` | A(k,E) 切片 + E_F 费米面图 |

物理校验：VBM = −2.117 eV，CBM = −0.495 eV，带隙 **1.622 eV**（MoS2/PBE 合理）；
E_F = −1.878 eV 落在带隙内（半导体 → 真费米面为空），
价带顶处展开费米面给出 **Γ 点与 K 点的空穴口袋** ✓。

---

## 9. 文件清单

| 文件 | 说明 |
|---|---|
| `rsh.py` | master 远程执行/上传助手 |
| `make_unfold_mesh.py` | 生成网格 KPOINTS + 映射表 + 几何 JSON |
| `unfold_procar.py` | 投影法反折叠 + 谱函数 + 费米面图 |
| `make_figures.py` | 由 unfolded_bands.dat 出图件（切片/费米面/逐带） |
| `unfold_fs.py` | 从 vaspkit EBS.dat 组装谱函数（备用路线） |
| `validate_unfold.py` | 自实现展开 vs vaspkit 282 对比 |
| `ref/VASPKIT_manual.md` | VASPKIT 中文手册（本地留档） |
| `out/*.pdf` | 演示图件 |

服务器工作目录：`/data/wjlu/work/claude/unfold_fs_demo/`
（`mesh12/` = 网格计算，`mesh12/fs_out/` = 展开结果与图件）
