#!/bin/bash
# 论文图(PRB 版式)一键生成: 6 张图 -> paper/figs/*.pdf
#
# 口径与版式:
#   * 画布 7.0 in 宽 = PRB 全栏 \linewidth ⇒ 图里的字号就是印刷磅值;
#   * 图内**不写总标题/长注释**(caption 承担), 面板标号 (a)(b)(c) 保留;
#   * 任何写在坐标区外的文字都不与轴标签重叠;
#   * 全部导出**矢量 PDF**(投稿用), 另出一张 PNG 供目视核查。
#
# 数据目录 DATA 里要有:
#   1T-TaS2 :  PRIMCELL.vasp  tas2_prim.dat  tas2_full_sc.dat
#              tas2_full_scsqrt3.dat  tas2_fs_prim.npz  tas2_fs_sc3x1x1.npz
#              tas2_fs_scsqrt3.npz
#   1H-NbSe2:  NbSe2_prim.vasp  nbse2/prim_assign.dat  nbse2/full_sc3x1x1.dat
#              nbse2/full_scsqrt3.dat  nbse2/prim.npz  nbse2/sc3x1x1.npz
#              nbse2/scsqrt3.npz
#
# 用法:  DATA=/path/to/data OUT=/path/to/paper/figs bash make_paper_figures.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-F:/tmp/tas2}"
OUT="${OUT:-$HERE/../../paper/figs}"
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"
cd "$DATA"

TAS_TITLE='primitive cell (direct calculation),$3\times1\times1$ supercell,$\sqrt{3}\times\sqrt{3}$ supercell'
NB_TITLE='primitive cell (direct calculation),$3\times1\times1$ supercell,$\sqrt{3}\times\sqrt{3}$ supercell'

# ---------------------------------------------------------------- 图 1 能带
python "$HERE/paper_fig_bands.py" --ef -1.501119 \
  --prim-dat tas2_prim.dat --unfold tas2_full_sc.dat,tas2_full_scsqrt3.dat \
  --titles "$TAS_TITLE" --labels '$\Gamma$:0,M:1.078,K:1.700,$\Gamma$:2.945' \
  --out "$OUT/fig1_bands.pdf" --png "$OUT/fig1_bands.png" \
  --e-min -4.2 --e-max 1.0 --sigma 0.05

# ---------------------------------------------------------------- 图 2 脊顶
python "$HERE/paper_fig_ridge.py" --npz nbse2/prim.npz --prim NbSe2_prim.vasp \
  --out "$OUT/fig2_ridge.pdf" --png "$OUT/fig2_ridge.png"

# ---------------------------------------------------------------- 图 3 协变性
python "$HERE/paper_fig_symmetry.py" --sc-npz tas2_fs_sc3x1x1.npz \
  --prim-npz tas2_fs_prim.npz --prim PRIMCELL.vasp \
  --out "$OUT/fig3_symmetry.pdf" --png "$OUT/fig3_symmetry.png"

# ---------------------------------------------------------------- 图 4 费米面
python "$HERE/paper_fig_fs.py" --prim PRIMCELL.vasp \
  --npz tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz,tas2_fs_scsqrt3.npz \
  --titles "$TAS_TITLE" --blue-panels 0 \
  --out "$OUT/fig4a_TaS2_fs.pdf" --png "$OUT/fig4a_TaS2_fs.png"

python "$HERE/paper_fig_fs.py" --prim NbSe2_prim.vasp \
  --npz nbse2/prim.npz,nbse2/sc3x1x1.npz,nbse2/scsqrt3.npz \
  --titles "$NB_TITLE" --blue-panels 0 \
  --out "$OUT/fig4b_NbSe2_fs.pdf" --png "$OUT/fig4b_NbSe2_fs.png"

# ---------------------------------------------------------------- 图 5 验证协议
python "$HERE/paper_fig_verify.py" --prim PRIMCELL.vasp \
  --prim-npz tas2_fs_prim.npz --sc3-npz tas2_fs_sc3x1x1.npz \
  --scs-npz tas2_fs_scsqrt3.npz \
  --out "$OUT/fig5_verify.pdf" --png "$OUT/fig5_verify.png"

# ---------------------------------------------------------------- 图 7 费米速度
# 等值面(两壁) vs 脊顶: 能量偏移、面积抵消、v_F 散布。
# 逐点数值与正文 §III.A 同源(见 paper/fact_sheet.md §2.13)。
# ⚠ --npz-unf 要的是**同一材料**的超胞展开数据(与 --npz-prim 同体系,
#   否则 (a)(b) 的倒格矢对不上)。这里用 1H-NbSe2 的原胞与 3x1x1 展开。
python "$HERE/paper_fig_vf.py" --npz-prim nbse2/prim.npz \
  --npz-unf nbse2/sc3x1x1.npz \
  --out "$OUT/fig7_velocity.pdf" --png "$OUT/fig7_velocity.png" --height 2.6

# ---------------------------------------------------------------- 图 6 CDW 态
# 需要 CDW 算例的反折叠结果(见 scripts/cdw/); 没有就跳过。
# 单栏图: 画布 3.4 x 2.4 in => 论文里印刷高 2.19 in(全栏宽会把它压扁)。
CDW_DAT="${CDW_DAT:-A_unfolded_full_G.dat}"
CDW_ASSIGN="${CDW_ASSIGN:-A_unfolded_assign_G.dat}"
if [ -f "$CDW_DAT" ]; then
  python "$HERE/paper_fig_cdw.py" --ef "${CDW_EF:-5.467486}"     --assign "$CDW_ASSIGN" --full "$CDW_DAT"     --width 3.4 --height 2.4 --sigma 0.05 --e-min -1.0 --e-max 1.0     --labels '$\Gamma$:0,M:1.080,K:1.703,$\Gamma$:2.950,A:3.475,L:4.555,H:5.178,A:6.424'     --out "$OUT/fig6_cdw.pdf" --png "$OUT/fig6_cdw.png"
else
  echo "(跳过图 6: 没找到 $CDW_DAT)"
fi

echo "=== 论文图 ==="; ls -la "$OUT"/*.pdf
