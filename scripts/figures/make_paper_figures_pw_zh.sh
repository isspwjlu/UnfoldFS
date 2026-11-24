#!/bin/bash
# 平面波路线的论文图(**中文版**)一键生成 -> paper_pw/figs_zh/*.pdf
#
# = make_paper_figures_pw.sh 的中文版: 数据、版式、图号完全一样, 只是给
#   六个图脚本加上 --lang zh(图内文字换成中文; 数学符号与高对称点不变)。
# 英文版(paper_pw/figs/)**不受影响**, 仍然由 make_paper_figures_pw.sh 生成。
#
# 两个数据源:
#   DATA   = 平面波路线的 TaS2/NbSe2 算例(默认 F:/tmp/tas2_pw)
#   DEFECT = 缺陷超胞的本地数据(仓库 .fsdata/); 服务器端见 make_defect_pw.sh
#
# 图的清单(与 paper_pw/main.tex 的图号对应):
#   fig1_bands      图 3  展开能带(原胞 + 两个超胞, 叠原胞的带)
#   fig2_covariance 图 1  协变性: A(k,E_F) + 逐操作 δ(R)(含地板线)
#   fig4_verify     图 2  验证: TaS2/NbSe2 各三个面板(背景场 + 费米面)
#   fig5_ridge      图 4  脊顶 vs 等值线
#   fig6_velocity   图 7  费米速度
#   fig7_cdw39      图 5  CDW 态的展开谱
#                     该图仍为英文。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-F:/tmp/tas2_pw}"
DEFECT="${DEFECT:-$HERE/../../.fsdata}"
OUT="${OUT:-$HERE/../../paper_pw/figs_zh}"
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"
cd "$DATA"

# 面板标题: 数学部分仍用 mathtext($...$), 中文部分直接写汉字。
# 中文与拉丁/公式之间**不留空格**(中文排版惯例)。
TAS_TITLE='原胞（直接计算）,$3\times1\times1$超胞,$\sqrt{3}\times\sqrt{3}$超胞'

# --------------------------------------------------------------- 图 3 展开能带
# ⚠ 平面波路线的权重**近乎二值**(93% 的点 = 1) ⇒ 必须用 symlog 色标,
#   否则副本全被压成同一个颜色。而且必须**叠上原胞的带**(白线), 否则
#   读者只看到一堆饱和的亮带, 看不出"原胞的带被重建出来了"。
python "$HERE/paper_fig_bands.py" --ef -1.501119 \
  --prim-dat tas2_prim.dat --unfold tas2_full_sc_pw.dat,tas2_full_scsqrt3_pw.dat \
  --titles "$TAS_TITLE" --labels '$\Gamma$:0,M:1.078,K:1.700,$\Gamma$:2.945' \
  --out "$OUT/fig1_bands.pdf" --png "$OUT/fig1_bands.png" \
  --e-min -4.2 --e-max 1.0 --sigma 0.06 --color-scale symlog --overlay-prim \
  --lang zh

# --------------------------------------------------------------- 图 1 协变性
python "$HERE/paper_fig_pw_covariance.py" \
  --npz tas2_fs96_sc3x1x1_pw.npz --prim PRIMCELL.vasp \
  --supercell "$HERE/../../.fsdata/TaS2_sc3x1x1_POSCAR" \
  --direct-npz tas2_fs96_prim_pwref.npz --panels a --height 2.6 \
  --out "$OUT/fig2_covariance.pdf" --png "$OUT/fig2_covariance.png" \
  --lang zh

# --------------------------------------------------------------- 图 2 验证协议
python "$HERE/paper_fig_fs_field.py" \
  --tas2-prim PRIMCELL.vasp \
  --tas2-npz tas2_fs96_prim_pwref.npz,tas2_fs96_sc3x1x1_pw.npz,tas2_fs96_scsqrt3_pw.npz \
  --nbse2-prim NbSe2_prim.vasp \
  --nbse2-npz nbse2/nbse2_fs48_prim_pwref.npz,nbse2/nbse2_fs48_sc3x1x1_pw.npz,nbse2/nbse2_fs48_scsqrt3_pw.npz \
  --out "$OUT/fig4_verify.pdf" --png "$OUT/fig4_verify.png" \
  --lang zh

# --------------------------------------------------------------- 图 4 脊顶
python "$HERE/paper_fig_ridge.py" --npz nbse2/nbse2_fs48_prim_pwref.npz \
  --prim NbSe2_prim.vasp \
  --out "$OUT/fig5_ridge.pdf" --png "$OUT/fig5_ridge.png" \
  --lang zh

# --------------------------------------------------------------- 图 7 费米速度
python "$HERE/paper_fig_vf.py" --npz-prim nbse2/nbse2_fs48_prim_pwref.npz \
  --npz-unf nbse2/nbse2_fs48_sc3x1x1_pw.npz \
  --out "$OUT/fig6_velocity.pdf" --png "$OUT/fig6_velocity.png" --height 2.6 \
  --lang zh

# --------------------------------------------------------------- 图 5 CDW 展开谱
python "$HERE/paper_fig_cdw.py" --ef 5.389338 \
  --assign tas2_cdw39_pw.dat --full tas2_cdw39_pw_full.dat \
  --width 3.4 --height 2.4 --sigma 0.06 --e-min -3.0 --e-max 1.5 \
  --labels '$\Gamma$:0,M:1.080,K:1.703,$\Gamma$:2.950,A:3.475,L:4.555,H:5.178,A:6.424' \
  --out "$OUT/fig7_cdw39.pdf" --png "$OUT/fig7_cdw39.png" \
  --lang zh

# --------------------------------------------------------------- 图 6 缺陷超胞
# 缺陷超胞的图已随缺陷一节从论文中移除(24x24 网格给出的费米面不够
# 光滑),故不再出图; 脚本仍在 scripts/figures/paper_fig_defect.py。


echo "=== 论文图(平面波路线, 中文版) ==="; ls -la "$OUT"/*.pdf
