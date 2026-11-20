# examples/ —— 探索期原型与算例脚本（归档区）

这里放的是**工具定型之前**写的脚本。它们记录了"这个工具是怎么从零搭起来的"，
但**不属于 `unfoldfs` 包**：不参与安装（`pyproject.toml` 只打包 `unfoldfs/`）、
不参与测试（`tests/` 不引用它们）、也不需要维护。

要用这个仓库，请看根目录 `README.md` 与 `docs/`；用 `unfoldfs` 命令行。

## prototypes/ —— 探索期的独立实现（Python）

| 文件 | 内容 |
|---|---|
| `unfold_procar.py` | **投影法超胞能带反折叠**（Popescu, PRL 104, 216401 (2012)）的独立实现。`unfoldfs` 包的展开算法就是从这里移植的，`docs/algorithms.md` §11.1 用它做**全路径交叉核对**（`max|Δw| = 4.99e-09`、`max|ΔE| = 0`），`tests/test_unfold.py` 的数值基线也同源于它 |
| `unfold_fs.py` | 由 vaspkit 282 的 `EBS.dat` 构建 $E_F$ 处的谱函数并渲染反折叠费米面（早期的"用别人的展开结果"路线） |
| `validate_unfold.py` | 在 $\Gamma$ 点对比"自实现投影法展开"与"VASPKIT 282（平面波法）展开"的能量与权重 |
| `diag_unfold.py` | 诊断展开公式的变体：单 $G$ 主导度、归一化 $\sum_G w / \sum|p|^2$ |
| `make_unfold_mesh.py` | 为超胞稠密网格生成 `KPOINTS` + `KPOINTS_MAPPING_TABLE.in` |
| `make_figures.py` | 由 `unfolded_bands.dat` 画 $A(k_x,k_y)$ 的二维能量切片（费米面演化） |

## vasp/ —— 服务器上的 VASP 算例脚本（Shell）

`run_*.sh` 提交 VASP 作业、`diag_*.sh` / `test_*.sh` / `inspect_*.sh` / `analyze_ebs.sh`
是排查问题用的一次性诊断脚本。

> ⚠️ **这些脚本里的路径是作者服务器上的历史路径**（如 `/data/wjlu/work/claude/…`、
> `/data/wjlu/program/vaspkit.1.5.1/…`），照搬前需要改成你自己的工作目录。
> 相对路径调用已按本目录结构调整（例如 `run_unfold.sh` 调
> `../prototypes/unfold_procar.py`）。

论文算例本身（1T-TaS$_2$ 的 CDW 态）用的是整理过的版本，见 `scripts/cdw/`。
