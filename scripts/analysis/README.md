# scripts/analysis/ —— 数据分析脚本（不出图）

与 `scripts/figures/`（出图）分开：这里只做**数值比较与判据**，产物是表格/JSON。

## `supercell_convergence.py`

**用途**：回答"展开谱与原胞直算的差异，有多少来自展开方法，有多少来自超胞本身？"

做法：把原胞网格加密，比较 `Pearson(A(k,E_F), A_prim(k,E_F))`。

- Pearson **上升** ⇒ 差异主要来自**离散化**（粗网格上高斯脊采样不足），不是投影失真；
- Pearson **饱和** ⇒ 饱和值就是**超胞的有限尺寸效应**（与投影方式无关）。

**主指标**（可信）：

| 指标 | 含义 |
|---|---|
| `Pearson(归一化)` | 谱在 k 空间的**强度分布形状**与原胞的一致程度（对仿射不变，不含整体强度差） |
| `谱重心差 max` | 每个 k 的 A(k,·) 的能量重心之差 ⇒ 分辨"谱结构错位"与"强度分布差异" |
| `相对L2(归一化)` | 上面两者的补充 |

**辅助指标**（assign 模式下不可靠，仅供参考）：`带能量最近距` —— 展开的 `band_e`
只保留**归属**到该 k 的带，带数与原胞不等（3×1×1 展开 22 条 vs 原胞 32 条），
所以这个量在 96 网格上偏大。**主结论只用 Pearson 与谱重心差。**

### 用法

```bash
# 不对称化(直接比原始输出)
python scripts/analysis/supercell_convergence.py \
    --prim .fsdata/tas2_fs48_prim.npz \
    --sc   .fsdata/tas2_fs48_sc3x1x1.npz .fsdata/tas2_fs48_scsqrt3.npz \
    --label 3x1x1 sqrt3xsqrt3

# 按原胞完整点群做轨道平均后比较(= `unfoldfs unfold` 的默认行为)
#   --gm 每个 --sc 对应的 grid_map.npz(unfoldfs gen 的产物), --primcell 原胞 POSCAR
python scripts/analysis/supercell_convergence.py \
    --prim .fsdata/tas2_fs24_prim.npz \
    --sc   .fsdata/tas2_fs24_sc3x1x1.npz \
    --label 3x1x1 \
    --gm   .fsdata/tas2_gm_sc24.npz \
    --primcell .fsdata/sc3x1x1_wave/PRIMCELL.vasp --symmetrize \
    --out .fsdata/conv/grid24.json
```

对称化的调用与 `unfoldfs unfold` 完全一致：`primitive_symmetry_ops(prim)` 给出
**原胞完整点群**，再 `symmetrize_spectral_function(A, grid_map, ops=full_ops)`。

## 实测结果（1T-TaS₂，2026-11-23）

见 `paper/fact_sheet.md` §2.3b。要点：**24×24 → 48×48 时 Pearson 0.720 → 0.989，
96×96 饱和** ⇒ 展开本身没有引入可测量的误差；饱和值 0.989 才是 3×1×1 超胞的
有限尺寸效应。同面积但各向同性的 √3×√3 超胞只有 0.156 ⇒ 有限尺寸效应的大小
强烈依赖超胞形状。

### `covariance_attribution.py`

**用途**：回答"展开谱的协变性缺陷来自哪里？"

对原胞点群的**每一个**操作 $R$，判定它是否**保持超胞晶格**
（判据 $R_{
m sc}=M^{-T}RM^T \in \mathbb{Z}^3$，与 `unfoldfs.geometry.common_symmetry_ops_detailed`
同口径），再对照该操作的 $\delta(R)$：

- 不保持超胞晶格 ⇒ 对称相关的原胞动量被折到**不同的**超胞动量集合上
  ⇒ 展开谱必然不协变（**任何**展开方法都如此）；
- **保持**超胞晶格却仍不协变 ⇒ 缺陷来自**代表原子平均**这一步，是**本实现特有的**。

**判决性算例**：1T-TaS₂ 的 √3×√3 R30 超胞，空间群与原胞相同（$Par{3}m1$）⇒
**12 个操作全部保持超胞晶格**（折叠完全协变），而 $\delta(R)$ 仍达 **0.99**
⇒ 该算例的缺陷 100% 来自权重构造。

**用法**：

```bash
python scripts/analysis/covariance_attribution.py     --prim .fsdata/TaS2_PRIMCELL.vasp     --npz  .fsdata/tas2_fs96_scsqrt3.npz     --matrix "2 1 0;-1 1 0;0 0 1"     --supercell-poscar .fsdata/TaS2_scsqrt3.vasp     --label "1T-TaS2 sqrt3xsqrt3 R30"
```

`--supercell-poscar` 还会打印**原胞与超胞的空间群**，用于 CDW 情形的判据口径：
CDW 超胞是 $Par{3}$、原胞是 $Par{3}m1$ ⇒ **CDW 态自己就破坏了原胞 12 个
点群操作中的 6 个** ⇒ CDW 情形下 $\delta(R)$ 必须对着**超胞的点群**读
（被 CDW 破坏的那 6 个操作上 $\deltapprox1$ 是**态的性质**，另外 6 个才反映
展开方法自身的缺陷）。

**实测结果**：见 `paper/fact_sheet.md` §1.4b / §1.4c；论文 §II.B 与 §VI.C 引用这些数。

### CDW 对照（pristine vs CDW，同超胞同赝势同网格）

`covariance_attribution.py` 的 `--compare-npz` 给出**逐操作** δ(R) 的并排对照，
用来回答"CDW 破坏了多少对称性、δ(R) 里有多少是态的"：

```bash
python scripts/analysis/covariance_attribution.py     --prim  .fsdata/TaS2_PRIMCELL.vasp     --npz   .fsdata/tas2_fs_scsqrt3.npz        # pristine √3×√3     --compare-npz .fsdata/tas2_cdw_scsqrt3.npz # CDW √3×√3(同一超胞, 只差 POSCAR)     --matrix "2 1 0;-1 1 0;0 0 1"     --supercell-poscar <CDW 超胞 POSCAR>
```

**注意**：CDW 态的反折叠必须用**未畸变**的超胞几何建立原子对应
（畸变后原子移动了，按整数平移找副本会失败），权重取自畸变计算：

```bash
python -m unfoldfs.cli unfold --dir <CDW 计算目录>     --supercell <未畸变超胞 POSCAR> --outdir <输出目录>
```

CDW 的 √3×√3 模型用 `scripts/cdw/build_trimer_cdw.py`（径向三聚化，空间群 P31m/C3v，
恰好破坏原胞 D3d 的 12 个操作中的 6 个）。
