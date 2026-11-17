# 设计规格：超胞反折叠费米面计算工具（unfoldfs）

- 日期：2026-11-17
- 状态：待评审
- 目标平台：master 服务器 (210.73.16.15)，Python 3.9 (anaconda3)，VASP 6.6.0
- 现状：已有原型脚本在 `F:\cc-haha_projects\unfold_band_fs\`，本 spec 描述将其重构为可复用工具

---

## 1. 背景与目标

### 1.1 问题

超胞计算把原胞能带折叠进超胞 Brillouin 区（体积 1/|det M|）。VASPKIT 能做**能带反折叠**（任务 281/282），但**算不了反折叠费米面**。实测根因（master 上 vaspkit 1.5.1）：

| 模块 | 必需输入 | 输出 | 含谱权重 |
|---|---|---|---|
| 能带反折叠 281/282 | **WAVECAR** + `TRANSMAT.in` + `KPOINTS_MAPPING_TABLE.in` + `KPATH.in` | `EBS.dat` (E, w) | 有 |
| 费米面 261/262 | **EIGENVAL + DOSCAR** + POSCAR | `FERMISURFACE.bxsf` | **无** |

- `vaspkit -task 282` 报 `Error: The WAVECAR File NOT Found` → 282 走平面波展开，非 PROCAR
- `vaspkit -task 262` 只读 EIGENVAL + DOSCAR → 纯能量插值，无权重概念
- 手写网格 KPOINTS + 网格映射表喂给 282 → 流程跑完但 `WEIGHT.grd`/`MOMENTUM.grd` 全为 0，网格模式不可用

结论：VASPKIT 两模块无接口，非操作问题。

### 1.2 目标

自研 Python 工具，实现 **超胞 → 反折叠 → 原胞 BZ 内均匀网格上的谱函数 → 费米面文件** 的完整链路：

1. 对称性约化的 k 点网格生成（避免全网格计算量爆炸），参考 VASPKIT 281/282 的思路
2. 反折叠（投影法，Popescu PRL 104, 216401 (2012)），权重来自 VASP 6.6.0 的 `LORBIT=12` PROCAR
3. 对称性展开：把约化 k 点的结果扩展回**完整均匀原胞网格**
4. 输出费米面文件（VESTA cube / XCrySDen grd+bxsf / FermiSurfer band / 原始 npz）

### 1.3 非目标（YAGNI）

- 不负责构建超胞（用 VASPKIT 401 生成超胞 + `TRANSMAT.in`）
- 不负责提交/管理 VASP 作业（两步走，用户在集群上自行提交）
- 不做轨道分辨（orbital-projected）费米面 —— 数据结构预留，但不在首版
- 不做磁性/自旋极化的对称性分析（时间反演对称按 spglib 默认处理）
- 不做掺杂/缺陷体系的电荷自洽（VASP 侧的事）

---

## 2. 术语与符号

| 符号 | 含义 |
|---|---|
| `M` | 超胞↔原胞整数变换矩阵（`TRANSMAT.in`），**行向量约定**：`A_sc[i] = Σ_j M[i][j] · a_prim[j]` |
| `K` | 超胞分数倒空间坐标（3 分量） |
| `k'` | 原胞分数倒空间坐标（3 分量） |
| `G` | 整数折叠矢量（展开候选的标签） |
| `S` | 公共对称操作集（见 §5.1） |
| `P_i` | 原胞均匀网格第 i 方向的点数 |
| `N_prim` | 原胞原子数；`N_sc` 超胞原子数；`detM = |det M|` |

**已核对的核心关系**（用 VASPKIT 官方 MoS2 3×3×1 算例的 159 行映射表逐行验证）：

```
k'_prim = (K + G) · M⁻¹          展开候选（原胞分数坐标）
K       = k'_prim · M  (mod 1)    原胞网格 → 超胞 k 列表
B_sc    = (M⁻¹)ᵀ · B_prim        倒格矢关系（行向量约定）
```

---

## 3. 总体架构与数据流

```
POSCAR(超胞) + PRIMCELL.vasp + TRANSMAT.in
  │
  ├─ geometry ─→ 超胞/原胞倒格矢、M、候选折叠矢量集 Gset、公共对称集 S
  ├─ grid    ─→ 均匀原胞网格 {k'} (P1×P2×P3)
  │              → 超胞 k 列表 K = k'·M (mod 1)
  │              → 在 S 下轨道分解约化 → 代表点 + 重数
  │              → 写 KPOINTS(特殊点+重数) + grid_map.npz
  ▼
★ VASP 6.6.0（用户自行提交）
   INCAR: LORBIT=12, LWAVE=.TRUE., ISMEAR=0, NBANDS ≥ detM × 原胞带数
  ▼
  ├─ vaspio  ─→ EIGENVAL(能量/占据) / DOSCAR(E_F) / PROCAR(LORBIT=12 复投影块)
  ├─ unfold  ─→ 每个代表点：detM 个原胞候选 → 每候选一组有效带 (E_n, w_n)
  ├─ symmetry─→ 扩展到完整均匀原胞网格：A(k',E) 标量场 + 逐带 (E_n, w_n)
  └─ fsout   ─→ FS.cube / FS.grd / FS.bxsf / band.N / A(k,E).npz
```

### 3.1 模块边界

| 模块 | 单一职责 | 只依赖 |
|---|---|---|
| `geometry.py` | 解析结构与变换矩阵；倒格矢；Gset；公共对称集 S | numpy, spglib |
| `grid.py` | 网格规格→超胞 k 列表→轨道分解→KPOINTS+grid_map | geometry, spglib |
| `vaspio.py` | 解析 VASP 输出（EIGENVAL/DOSCAR/PROCAR） | numpy |
| `unfold.py` | 反折叠：G 候选、投影展开、有效带重组、自检 | geometry, vaspio |
| `symmetry.py` | 约化结果→完整均匀网格；A(k,E) 组装 | geometry, unfold |
| `fsout.py` | 各格式写出 | numpy |
| `cli.py` | 子命令与参数解析 | argparse |

依赖方向单向，无循环。`unfold.py` 不依赖 `grid.py`（只吃几何信息），便于独立测试。

### 3.2 CLI

```bash
unfoldfs gen    --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
                 --resolution 0.02 --grid-type gamma [--shift 0 0 0] --outdir run/
unfoldfs unfold --dir run/ --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in \
                 [--ef auto|vbm|<数值>] [--delta 0.05] [--sigma 0.06] --outdir fs/
unfoldfs info   --dir run/
```

`gen` 与 `unfold` 通过磁盘上的 `grid_map.npz` + VASP 输出文件耦合，两步可独立重跑。

---

## 4. 数据文件

### 4.1 `grid_map.npz`（`gen` 写出）

| 键 | 形状 | 含义 |
|---|---|---|
| `ir_K` | (n_red, 3) | 代表点的**超胞**分数坐标 |
| `ir_k` | (n_red, 3) | 代表点的**原胞**分数坐标 |
| `multiplicity` | (n_red,) | 每个代表点的轨道重数（= 满网格中同轨道的点数） |
| `full_to_ir` | (P1·P2·P3,) | 满网格点 → 代表点索引 |
| `rot_of_full` | (P1·P2·P3,) | 满网格点 → S 中的操作索引（-1 = 不动） |
| `P` | (3,) | 原胞网格点数 |
| `grid_type` / `shift` | str / (3,) | 网格类型与偏移 |
| `M` / `M_inv` | (3,3) | 变换矩阵与逆 |
| `Gset` | (detM, 3) | 候选折叠矢量集 |
| `b_prim` / `b_sc` | (3,3) | 原胞/超胞倒格矢（2π/Å，行=倒格矢） |
| `rotations` | (n_ops, 3, 3) | S 中各操作的整数旋转矩阵（原胞基底） |

### 4.2 `unfolded.npz`（`unfold` 写出，中间产物）

`k_cart` (n_full, 3)、`band_E` (n_full, n_bands)、`band_w` (n_full, n_bands)、`ef`、`vbm`、`cbm`、`is_metal`

---

## 5. 算法规格

### 5.1 公共对称集 S（geometry.py）

1. spglib 求原胞操作 `(R_pr, t)`（原胞基底下整数矩阵）与超胞操作 `(R_sc, t')`（超胞基底下整数矩阵）
2. 对每个 `R_sc`：换算到原胞基底 `R_pr = M · R_sc · M⁻¹`
   - 若 `R_pr` 非整数 → 剔除（该超胞对称在原胞基底不是晶格对称）
   - 若 `R_pr` 不在原胞操作集 → 剔除（超胞有原胞没有的额外对称，如莫尔 3 重旋转）
   - 否则保留
3. 结果按去重后的 `R_pr` 排序存入 `rotations`；平移分量 `t` 在相对偏移中相消，不参与后续计算

**依据**：用户确认取「两者交集」；保证展开后的原胞谱函数按原胞对称性自洽。

### 5.2 网格生成与约化（grid.py）

1. **网格点数**：`P_i = ceil(|b_prim[i]| / resolution)`，再向上取整到 **comm(M⁻¹) 的倍数**
   - `comm(M⁻¹)` = 使 `M⁻¹ · (整数矢量)` 全为有理数且分母整除的最小正整数，即 M⁻¹ 各元素分母的最小公倍数
   - 例：M=diag(3,3,1) → comm=3 → P 须为 3 的倍数
   - 例：M=[[2,1],[0,2]] → M⁻¹=[[0.5,-0.25],[0,0.5]] → comm=4 → P 须为 4 的倍数
   - **必要性**：否则展开候选 `(K+G)·M⁻¹` 不落在均匀网格上
2. **网格类型**：默认 Γ-centered（与 VASPKIT 261/231 生成的 KPOINTS 一致，其首字母 `G`）；可选 MP
3. **超胞 k 列表**：`K = k'·M (mod 1)`（行向量）
4. **轨道分解**（自实现，不用 `get_ir_reciprocal_mesh`——因为扩展阶段还需要旋转信息）：
   - 对每个满网格点 p，求其轨道 `{ g·p : g ∈ S }`（在原胞分数坐标下作用）
   - 轨道首个未归类者 → 代表点 ir；记录 `full_to_ir[p]=ir`、`rot_of_full[p]=g`
   - 代表点重数 = |轨道|
5. **写 KPOINTS**：沿用 VASPKIT 261/231 生成的「特殊点 + 重数」格式（**已实测 VASP 读法：`Reciprocal lattice` 之后的行按 k 点读，不是倒格矢**）
   ```
   G    <n1> <n2> <n3> <resolution>     # Parameters to Generate KPOINTS (Don't Edit This Line)
          <n_ir>
   Reciprocal lattice
       <代表点超胞分数坐标 x y z>  <重数>
       ...
   ```
   首行首字母为网格类型标记（`G` = Γ-centered，`M` = MP）；权重之和须等于满网格点数 P1·P2·P3。
6. **自检**：Σ重数 = P1·P2·P3；每个轨道闭合；代表点坐标落在超胞 BZ 内

### 5.3 反折叠（unfold.py）

输入：`EIGENVAL`（能量/占据）、`PROCAR`（LORBIT=12 复投影块）、几何信息。

**PROCAR 解析要点（实测）**：
- 头部第 2 行含 NKPTS/NBANDS/NIONS
- 每个 band 两块：11 字段的模方块（ion + 9 轨道 |p|² + tot）、20 字段的复投影块（ion + 9×(Re,Im) + |c|²）
- 复投影块是**晶胞周期规范**（同一原胞的各副本复值相同，不含 e^{iK·R}）

**展开**（对每个代表点 K、每条带 m）。记 `J` 为虚数单位、`i` 为原胞原子序号、`lm` 为轨道：
```
p_{i,lm}(k') = (1/N_copies) · Σ_T c_{j,lm} · exp(s1·J·(k'−K)·T_j)
w(k')        = | Σ_{i,lm} p_{i,lm}(k') · exp(s2·J·k'·r_i) |²
```
- `s1, s2 ∈ {+1,−1}`：相位符号约定，由自检自动选定（见下）
- 候选：`k' = (K + G)·M⁻¹`，`G ∈ Gset = { g ∈ Z³ : g·M⁻¹ ∈ [0,1)³ }`（detM 个）

**候选重组**：每条超胞带归属到权重最大的候选；每个候选得到一组有效带 `(E_n, w_n)`，**按能量排序 → n = 原胞带序号**（原胞带序号在对称变换下不变 → 逐带输出自洽）。权重不可忽略的多候选归属记为「谱泄漏」，主归属仍取最大权重。

**自检（强制）**：每个态在 detM 个候选中应**单 G 主导**——定义主导度 = max(w)/Σw，阈值 0.5。低于阈值 → 报错退出（说明相位约定或副本匹配有误）。
`s1,s2` 的选择：以「主导度的平均值最大」为准则自动挑选 4 种组合中的最优。

**实测基线**（MoS2 3×3×1，VASP 5.4.4 与 6.6.0 一致）：主导度 0.85~1.0，非主导候选权重降到 1e-33。

### 5.4 对称扩展（symmetry.py）

**关键洞察**：候选集在对称操作下是一个置换 —— `G ↦ R_pr·G (mod 晶格)`。

```
对每个满网格点 i：
    rep = full_to_ir[i]          # 代表点
    r   = rotations[rot_of_full[i]]
    data[i][G''] = data[rep][ π_r⁻¹(G'') ]     # π_r: G ↦ R_pr·G mod 晶格
```

即扩展 = 取代表点的展开结果 + 按该点旋转置换候选索引。无需重算，无需处理平移（平移在相对偏移中相消）。

扩展后得到完整均匀原胞网格上的：
- `A(k', E)` 标量场：`A(k',E) = Σ_n w_n(k')·δ(E − E_n(k'))`
- 逐带 `E_n(k'), w_n(k')`

### 5.5 费米能级处理（fsout.py 前置）

- `E_F`：DOSCAR 第 6 行**第 4 个数**（前 5 行是头，第 6 行 = EMAX EMIN NEDOS EFERMI …）
- `VBM/CBM`：EIGENVAL 占据数（occ>0.5 → 占据）
- 自动判别：`E_F` 落在带内 → 金属；落在 VBM..CBM → 半导体
- 输出截面：`E_F`（金属）或 `VBM+Δ`（半导体，默认 Δ=0.05 eV）；**始终输出全量 A(k,E)**

---

## 6. 输出文件格式

| 文件 | 消费者 | 格式规范来源（不凭记忆写） |
|---|---|---|
| `FS.cube` | VESTA | Gaussian cube；**把倒格矢作为「假晶格」实矢量** → VESTA 渲出的等值面即 k 空间费米面 |
| `FS.grd` | XCrySDen | GNUPLOT ASCII grid（k 空间笛卡尔，2π/Å） |
| `FS.bxsf` | XCrySDen | 逐带等值面顶点表；**先用 vaspkit 262 生成样例，逐字节对齐** |
| `band.N` | FermiSurfer | 逐带 (k, E)；**从 `/data/wjlu/program/fermisurfer-2.2.1` 源码读真实期望格式** |
| `A(k,E).npz` | 自用/二次分析 | k_cart (n_full,3)、E (n_E,)、A (n_full, n_E)、ef/vbm/cbm |

`FS.bxsf` 与 `band.N` 的确切字节格式在实现第一步从上述工具现场逆向核对并记入 `docs/formats.md`。

---

## 7. 错误处理

| 情况 | 行为 |
|---|---|
| PRIMCELL / POSCAR / TRANSMAT 不自洽 | 立即报错，给出 M 与两套晶格的实际对应关系 |
| PROCAR 缺 `+ phase` 块 | 报错并提示「需要 `LORBIT=12`；`LORBIT=11` 不写相位」 |
| 单 G 主导度 < 0.5 | 报错（相位约定或副本匹配有误），打印 4 种约定的对比 |
| VASP 实际 k 点数 ≠ grid_map 记录 | 报错，给出两边对照 |
| 对称操作在超胞基底非整数 / 不属原胞操作集 | 剔除并在 `info` 中列出被剔除项及原因 |
| 分母约束导致 P 远大于 resolution 需求 | 警告并给出实际采用的 P |

---

## 8. 测试策略

**单元测试（pytest）**
- `geometry`：POSCAR/TRANSMAT 解析、倒格矢与 M⁻¹ 分母、Gset 基数 = detM
- `grid`：轨道封闭性（Σ重数 = 满网格点数）、代表点坐标落在 BZ 内、网格点在 S 下闭合、P 为分母倍数
- `vaspio`：PROCAR 双块解析（模方块 + 复投影块）、EIGENVAL 占据、DOSCAR 的 E_F 位置
- `unfold`：单 G 主导自检、4 种相位约定的比较、候选重组后每候选带数一致
- `symmetry`：扩展后 A(k',E) 在每个对称操作下数值不变（≤1e-8 相对误差）、网格点集守恒
- `fsout`：输出文件能被对应工具读回（回环测试：写出→读入→逐值比对）

**回归测试**：固定 fixture（MoS2 3×3×1，2 k 点的 PROCAR/EIGENVAL）→ 展开结果数值基线，防回归

**验收**：与 VASPKIT 282 路径模式对照（同一超胞、同一 k 点）：**有效能带 E_n(k') 逐条一致**（硬指标）；权重只比相关性与排序（投影法 vs 平面波法，量级本就不该相同）

---

## 9. 已验证的事实（实现前已实测，非假设）

1. VASPKIT 282 走 WAVECAR（平面波）展开，不读 PROCAR；网格模式权重全 0，不可用
2. VASPKIT 262 只读 EIGENVAL + DOSCAR，无权重概念
3. `LORBIT=12` 写复投影块，`LORBIT=11` 不写（VASP 5.4.4 源码 `sphpro.F` 第 1096 行确认；VASP 6.6.0 实测同样）
4. VASP 5.4.4 与 6.6.0 的 LORBIT=12 PROCAR 结构完全一致（11 字段模方块 + 20 字段复投影块）
5. 折叠关系 `k'_prim = (K+G)·M⁻¹`（官方 MoS2 3×3 算例 159 行映射表逐行核对）
6. VASPKIT 映射表为定宽格式且 G 必须非负（用负数 G → Fortran `input conversion error`）；行数须 = n_kpoints × detM
7. Popescu 投影法展开在 MoS2 3×3×1 上给出单 G 主导（0.85~1.0）的正确结构，物理结果合理：带隙 1.622 eV，价带顶在 Γ 与 K 有空穴口袋

---

## 10. 后续（不在本 spec 范围）

- 轨道分辨费米面（数据已含轨道投影，扩展 `unfold.py` 输出维度即可）
- 3D 等值面直接出图（pyvista/VTK）而不依赖外部工具
- 与 WannierTools / bands4vasp 的定量交叉验证
