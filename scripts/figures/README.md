# 出图脚本 —— 展开能带图与费米面图

这两个脚本把 `unfoldfs` 产出的数据画成论文级的图：

| 脚本 | 画什么 | 输入 |
|---|---|---|
| `plot_band.py` | 展开能带对比图（原胞蓝线 + 两种超胞反折叠热图） | `unfoldfs band` 的 `.dat` 长表 |
| `plot_fs.py` | 费米面图（真费米面 + 谱权重配色） | `unfoldfs unfold` 产的 `A(k,E).npz` |
| `plot_fs_3d.py` | **三维**反折叠费米面（PyVista/VTK 渲染） | 同上 |
| `plot_fs_kz.py` | **任意 kz 切片**上的 kx-ky 费米面（多切片并排） | 同上 |
| `setup_fs_kz.sh` | 服务器端：**带 kz 采样**的费米面算例 | — |
| `check_fs_symmetry.py` | 核对费米面图的对称性（几何 + 颜色） | 同上 |
| `make_final_figures.sh` | 一次出齐 4 幅大图 + 12 幅小图（+ 4 幅三维图） | 上面三者的组合 |

> 这些是**画图**脚本，不属于 `unfoldfs` 包（不参与安装与测试）；
> 数据的生成仍然走 `unfoldfs gen / unfold / band`（见仓库根 README）。

## 一键出图

```bash
DATA=/path/to/data OUT=/path/to/out bash scripts/figures/make_final_figures.sh
```

`DATA` 目录里需要（1T-TaS₂ 与 1H-NbSe₂ 各一套）：

```
PRIMCELL.vasp  NbSe2_prim.vasp
tas2_prim.dat  tas2_full_sc.dat  tas2_full_scsqrt3.dat
tas2_fs_prim.npz  tas2_fs_sc3x1x1.npz  tas2_fs_scsqrt3.npz
nbse2/prim_assign.dat  nbse2/full_sc3x1x1.dat  nbse2/full_scsqrt3.dat
nbse2/prim.npz  nbse2/sc3x1x1.npz  nbse2/scsqrt3.npz
```

产出：`OUT/<材料>_{band,fs}.png`（大图，3 栏）+ `OUT/small/<材料>_{band,fs}_{a,b,c}.png`（12 幅小图）
+ `OUT/<材料>_fs3d_{tube,sheet}.png`（三维费米面）。

## 画法约定（不要改）

### 能带图

1. **E_F 置 0**：y 轴画 `E − E_F`。
2. 超胞面板 = `--unfold-mode full` + **按画窗归一化的线性色标**（`band._band_figure_colors`，
   弱副本暗但可见、强带亮）。不要用 `log`（会把弱副本抬成高权重），
   也不要退回 `assign`（只画归属到 k′ 的带 ⇒ 图就是原胞能带、没有折叠副本）。
3. **原胞面板画原 band 本身、纯蓝线**，不画 unfolded。

### 费米面图

1. **费米面 = `E_b(k)` 恰过 `E_F` 的那条线**（等势面）。
   ⚠ **绝对不要画 `A(k,E_F)` 的等值线**：A 在费米面处是一条高斯**脊**，
   任何低于脊高的等值线都会给出脊**两侧各一条**（"两壁"），
   真费米面在**两壁中间**（脊顶）。实测 1H-NbSe₂：沿 Γ–M 切线，A 的 0.5max
   等值线在 k1_frac = 0.214 / 0.245，真费米面在 0.229。
   附带症状：离 E_F 还有 ~0.09 eV 的带也会在 A 等值线里留下"幽灵"小圈。
2. `--method peak`（默认）：A(k,·) 在 E_F 附近**主峰的能量** E\*(k) 的零等值线
   —— 光滑、对称（A 已按原胞点群对称化）、原胞与反折叠**同一口径**。
   `--method band`：逐带 (E_b − E_F) 的零等值线 —— 精确，但只适用原胞
   （反折叠超胞里同一带号在不同 k 上是不同副本，曲线会碎成竖条）。
3. **颜色 = 过 E_F 那条带的谱权重 w(k)**（不是展宽后的 A(k,E_F)）。
   原胞面板权重处处相同 ⇒ 画**蓝线**（`--blue-panels 0`）；超胞用 jet 配色。
4. **反折叠面板的曲线必须与原胞费米面 100% 重合** —— 反折叠只改权重、不改位置。
   实测 1T-TaS₂ / 1H-NbSe₂ 的 3×1×1 与 √3×√3 都是 100.00%。
   做不到 100% 就是判据或数据出了问题。

## kz 切片（`plot_fs_kz.py` / `plot_fs.py --kz`）

**工具本身已经支持三维 k 网格**：

```bash
unfoldfs gen --grid 48 48 3 --grid-type gamma      # ⇒ kz = 0, 1/3, 2/3(实测 KPOINTS)
unfoldfs unfold --dir <算例目录> --ef auto        # ⇒ A(k,E).npz 带 kz 维; FS.cube 是三维体
```

`plot_fs.py --kz 0.3333` 与 `plot_fs_kz.py --kz-list 0,0.3333,0.6667`
从 npz 里**取指定 kz 层**画 kx-ky 费米面（kz 用分数坐标，1.0 = 2π/c）。

⚠ **物理前提**：单层体系（c = 17.66 Å）带在 kz 上**几乎是平的**
（层间耦合 ≈ e^{-c/3} ≈ 0.003）⇒ **各 kz 切片的费米面本来就该相同**。
所以：

* 现在的算例是 2D k 网格（只有 kz = 0 一层）⇒ 任意 kz 都只能返回那一层，
  三个切片画出来一样（`plot_fs_kz.py` 会在标题里报出**实际用的那一层**）；
* 带 kz 采样的算例（`setup_fs_kz.sh`）才能给出真正不同的切片，
  同时让 `A(k,E)` 变成**三维体数据** ⇒ `FS.cube` 可以用 FermiSurfer/VESTA
  画经典的三维等值面费米面。

## 三维费米面（`plot_fs_3d.py`）

单层体系（1T-TaS₂ / 1H-NbSe₂）**kz 方向无色散**（c = 17.66 Å，带在 kz 上是平的），
所以三维布里渊区里的费米面 = 二维费米面轨迹沿 kz 拉伸成的**带**（kz = 0 与 ±π/c 各一份，
周期性等价）。算例只在 kz = 0 取样 ⇒ 三维 FS 的"面"没有额外的色散信息。

因此有两种诚实的渲染：

| `--mode` | 画法 | 含义 |
|---|---|---|
| `sheet`（**默认，也是定稿风格**） | FS 轨迹沿 kz 扫成**带** | **物理版**：三维 FS 就是这个（kz 方向拉伸 3 倍显示） |
| `tube`（备选） | FS 轨迹吹成**管**，**管半径 ∝ 谱权重** | 权重版：强副本粗、弱副本细；⚠ 管在 kz 方向的粗细**不代表色散** |

> 用户 2026-11-19 明确选定 **sheet** 这一种作为三维费米面的呈现风格。

**实现要点（都是踩过的坑）**：

1. **VTK 的内置字体没有中文字形** ⇒ 3D 面板里只放 `G/M/K` 这类拉丁标签，
   中文标题/图注/颜色条一律用 matplotlib 拼版（`plot_fs_3d.py` 内部就是这么分工的）。
2. **VTK 的 cell 数组是扁平格式** `[n0, i0, i1, …]` ⇒ 自己拼，别直接塞
   `np.array([[a,b,c,d], …])`（会报 "cell array size is invalid"）。
3. **相机要手动给，而且距离要按视场角算**：`reset_camera()` 会被点标注的包围盒带偏
   （缩到贴脸）；`view_isometric()` 也不重置距离 ⇒ 用
   `dist = 1.3·半尺寸 / tan(视场角/2)`，再 `focal_point=(0,0,0)` + `up=(0,0,1)`。
4. **kz 半高 = |b3|/2**（`b_prim[2]` 就是沿 c 的倒格矢，长度 = 2π/c）——
   写成 `0.5·(2π/|b3|)` 会把 kz 放大 17 倍（当成了晶格常数 c）⇒ 画出来是一根细柱。
5. **各超胞面板要共享色标**（先扫一遍所有面板的权重上限再渲染），否则每栏各自
   归一化，颜色不可比。
6. **管/带沿 kz 拉伸倍数**用 `--kz-scale`（默认 3）：单层 BZ 只有面内的 1/5 高，
   等比例画几乎看不出厚度。

## 论文图(PRB 版式)

论文用的 6+1 张图走**另一套脚本**, 口径与上面的中文图完全一致(同一份数据、
同一套画法), 只是版式按期刊规范重排:

| 脚本 | 画什么 |
|---|---|
| `prb_style.py` | **统一版式**: 字号层级、线宽、去顶右脊、无框图例、矢量导出 |
| `paper_fig_bands.py` | 图 1: 展开能带(原胞蓝线 + 两种超胞) |
| `paper_fig_ridge.py` | 图 2: 费米面 = 高斯脊的**脊顶**(不是等值线的两壁) |
| `paper_fig_symmetry.py` | 图 3: 展开谱的**点群协变性**破缺与修法 |
| `paper_fig_fs.py` | 图 4: 反折叠费米面(原胞蓝线 + 两种超胞) |
| `paper_fig_verify.py` | 图 5: 验证协议(重合 + 距离 + 权重分布) |
| `paper_fig_cdw.py` | 图 6: 1T-TaS2 **CDW 态**的反折叠能带(有隙) |
| `make_paper_figures.sh` | 一次出齐上面全部图 -> `paper/figs/*.pdf` |

```bash
DATA=/path/to/data OUT=/path/to/paper/figs bash scripts/figures/make_paper_figures.sh
```

### 论文版式的三条硬规矩(`prb_style.py` 顶部有详细说明)

1. **画布宽度 = 最终印刷宽度**(PRB 全栏 7.0 in)。画布比最终尺寸大 ⇒ LaTeX 缩放 ⇒
   图里写的字号跟着变小(曾经把 9.5 pt 的标题画在 11.4 in 宽的画布上, 缩到 3.5 in
   就只剩 2.9 pt, 印出来根本读不清)。所以字号是**印刷时的真实磅值**。
2. **图里不写总标题、不写长注释** —— 那是 caption 的活。图内只保留: 面板标号
   (a)(b)(c)、轴标题与单位、刻度、图例、色标。
   ⚠ 文字里**不要出现 markdown 星号**(`*强调*` / `**强调**`): matplotlib 不解析
   markdown, 星号会**原样印在图上**(已犯过)。
3. **任何写在坐标区外的文字都不能和轴标签/刻度重叠**。办法: 文字要么放在坐标区
   **内**并配半透明白底框, 要么用 `fig.text` 放在画布留白里, 绝不用
   `ax.text(transform=ax.transAxes, y=-0.16)` 这种"贴在轴下面"的写法 ——
   它必然和 x 轴标签打架(已犯过: 图 3 的 dispersion 数字压住了 $k_1$)。

### CDW 例子(体相 39 原子, 论文图 7)的完整流程

```bash
# 1. 建 √3×√3 R30° 的 CDW 超胞(本地)
python scripts/cdw/build_cdw_tas2.py --poscar sc_sqrt3.vasp     --out POSCAR_CDW --delta-ta 0.035 --delta-s 0.06 --delta-z 0.04
#    (脚本会打印 Ta-Ta 键长的变化和 S 的位移幅度 —— 先看这两行再决定振幅)

# 2. 上传 POSCAR_CDW, 跑 VASP(脚本见 scripts/cdw/setup_cdw_tas2.sh)
bash scripts/cdw/setup_cdw_tas2.sh

# 3. 算完后反折叠(服务器):  bash <run_dir>/unfold.sh
#    => unfolded_band.dat
#    ⚠ 两个坑: (a) 原子对应要用**未畸变**的 POSCAR(--supercell .../sc_sqrt3.vasp),
#      畸变 POSCAR 匹配不上; (b) 要 --unfold-mode full --weight-scale linear,
#      且未畸变的那份也要用**同一个包**重跑(pkg_r10 没有 --unfold-mode, 用 pkg_r18)

# 4. 出图(本地)
python scripts/figures/paper_fig_cdw.py --ef <E_F>     --pristine tas2_full_scsqrt3.dat --cdw <CDW 的 unfolded_band.dat>     --labels '$\Gamma$:0,M:1.078,K:1.700,$\Gamma$:2.945'     --out paper/figs/fig6_cdw.pdf --png paper/figs/fig6_cdw.png
```

⚠ **集群是共用的**: 节点经常占满, 提交前先 `squeue`; `isspfml_gpu*` 的 QOS 只给
6 个 CPU, 跑不动 VASP, 别往那儿提交。


**体相 CDW(39 原子)的流程**(与上面的单层冻结畸变不同):

```bash
# 1. 建反折叠要用的三个文件(原胞 / TRANSMAT / 未畸变超胞)
python scripts/cdw/build_bulk_cdw_prim.py --cdw-poscar <39 原子 POSCAR>     --out-dir <dir>
# 2. 服务器跑 VASP: 要 **LORBIT=12**(11 不写相位 ⇒ 展开时没有复投影块),
#    ISYM=0, 180 带, GGA+rev-vdW-DF2, 原胞基的 Γ-M-K-Γ | Γ-A-L-H-A 路径
# 3. 反折叠: --supercell <未畸变的超胞> --unfold-mode full --weight-scale linear
#    ⚠ CDW 态的单-G 主导度只有 0.486 < 0.5 ⇒ 工具的"展开可信度"自检会报错;
#      那是物理(CDW 态不是原胞平移的本征态), 要放宽 MIN_DOMINANCE 才能出图。
# 4. 出图: **单栏**! --width 3.4
python scripts/figures/paper_fig_cdw.py --ef <E_F> --dat <unfolded_band.dat>     --width 3.4 --height 5.6 --labels "$\Gamma$:0,M:...,K:..."     --out paper/figs/fig6_cdw.pdf
```
⚠ **CDW 图必须单栏**: 全栏 `figure*` 会被 LaTeX 推到讨论之后(就是"最后一个图跑很靠后"那个毛病)。

### 色标约定

* **画布底部的横向色标不要**(用户 2026-11-20 明确): 颜色含义写进图注
  ("超胞面板的颜色是展开态的谱权重, 按画窗归一化")。
  受影响: `paper_fig_bands.py`(图 1)、`paper_fig_cdw.py`(图 7)—— 底部留白收紧。
* **k 路径能带图的横坐标只要高对称点标记**: 不要数字刻度, 也不要 "k-path (2π/Å)"
  轴标题(单位写进图注)。
* **画线型的能带面板(原胞)边框四周都要**(house style 默认只留左/下, 这类面板要
  单独打开 top/right 边框)。
* **高对称点标记贴着下边框**: 画在 `ax.get_xaxis_transform()` 的 `y=-0.022`
  (va="top"), 底部留白随之收紧。
* **面板旁的竖直色标保留**: 图 2 的 $A(k,E_F)$、图 3 的 $w/w_{\max}$、图 4 的
  $A(k,E)$ —— 那是那个量唯一的标度, 去掉图就读不懂了。

### 其它约定

* 色标: 展开能带/费米面用 `turbo`(感知均匀, 比 `jet` 灰度打印不塌); 谱函数用
  `magma`。
* 费米面/谱图面板用**深底**(权重色标低端是深蓝, 白底上会看不见)。
* 存 PDF 时**网格数据栅格化**(`rasterized=True`): 一个 pcolormesh 就是几 MB,
  栅格化后 PDF 从 6.6 MB 降到 60 kB, 坐标轴与文字仍是矢量。

## 踩过的坑（每个都花了一轮才找到）

1. **A(k,E_F) 的等值线 = 脊的两壁**，不是费米面（见上）。
2. **`npz` 的 `band_e`/`band_w` 布局是 (点, 带) 点优先** ⇒ 要 (带, n1, n2) 必须
   **先转置**：`d["band_e"].T.reshape(-1, n1, n2)`。直接 reshape 会把点与带搅乱。
3. **轨道平均要把「原点」的值加到「像」上**：`np.add.at(out, (i2, j2), field.ravel())`。
   写成 `field[i2, j2]` 就是"把每点加给自己" ⇒ 平均退化成恒等 ⇒ 图案仍然不对称。
4. **取色插值必须"点群协变"**：倒格矢基不正交，点群里的 C3 在分数基里是**剪切型**
   变换 ⇒ 平方插值格子被映射成平行四边形 ⇒ 双线性插值在对称相关点上给出不同的值
   （实测残差达场值跨度的 10%）。改用**笛卡尔等距的高斯核**（支撑集按笛卡尔距离
   定义），且要同时避开三个坑：核支撑集按笛卡尔距离选、"最近格点"按**周期**笛卡尔
   距离找、**行列约定** `cart→frac` 是 `p @ inv(b2d)`（行向量）且查询点先折回胞内。
5. **`LineCollection` 的颜色数组必须"一段一个"**（取段中点的值）。
   曾传"每顶点一个颜色"，段数比颜色数少 ⇒ 颜色**沿曲线整体错位** ⇒ 画出来看着不对称。
   1T-TaS₂ 的权重沿 FS 变化快 ⇒ 错位很明显；1H-NbSe₂ 变化慢 ⇒ 几乎看不出来 ——
   **同一个 bug 在两个材料上表现不同**。
6. **追"峰"要有三道闸门**，否则会画出孤零零的小圈（那些圈上 A(k,E_F)=0）：
   ① 峰必须是**真的局域极大**（`A[j-1] < A[j] > A[j+1]`），否则会把"搜索窗口边缘上
   强带的尾巴"当成峰；② 模糊后只保留 `|E*−E_F| < 0.25 eV`；
   ③ **E_F 处确实有谱权重**（`A(k,E_F) > 0.02·max`）。
7. **核对对称性要用"点到线段"距离**，别用"光栅化比像素"（亚像素位移就整片翻红，
   会把 0.001 格的偏差报成 10%），也别用"顶点配对"距离（量的是顶点采样）。
   出图前再核一遍**渲染出来的像素**：读 PNG，找面板中轴（用六边形的像素范围定），
   比较曲线内部像素与其镜像的颜色差。

## 核对

```bash
python scripts/figures/check_fs_symmetry.py --prim PRIMCELL.vasp tas2_fs_scsqrt3.npz
```

实测（1T-TaS₂ 96×96 / 1H-NbSe₂ 48×96）：

| | 颜色对称残差 | 几何偏差（曲线到其像） |
|---|---|---|
| 1T-TaS₂ | 2.2e-16 | 中位 0.0001 格，最大 0.18 格 |
| 1H-NbSe₂ | 2.3e-12 | 中位 0.0001 格，最大 0.91 格 |

渲染像素级：中位 9~10 / 765（剩下的就是抗锯齿与亚像素）。
