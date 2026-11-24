#!/bin/bash
# 缺陷超胞的**48×48** 平面波路线 —— 服务器端: 展开 → 原胞参考。
#
# 为什么重算: 24×24 的网格(间距 0.090 2π/Å)与费米面的特征尺度(~0.3~0.8)
# 相当 ⇒ 直接提等值线得到的是折线碎片, 插值重建也抹不掉位置的误差。
# 48×48 把间距减半, 曲线才是光滑的。
#
# 前置(已提交): static_defect_bulk_48 / static_perfect_bulk_48 的 VASP 算例
# 原胞参考**不需要**重算: prim96_pw 的 EIGENVAL 含 96×96 的全部 k 点,
# 48×48 是它的子集 ⇒ primitive_reference.py 会按超胞的 k 网格取带。
set -u
TA=$WORKDIR
D=$TA/defect
PY=$TOOLSDIR/anaconda3_2025/envs/phonopy4/bin/python3
export PYTHONPATH=$TA/pkg_pw2
export PYTHONIOENCODING=utf-8
mkdir -p /tmp/pw_analysis

echo "=== 1. 展开(缺陷态不是原胞平移的本征态 ⇒ 主导度闸门放宽) ==="
for src in static_defect_bulk_48 static_perfect_bulk_48; do
  cd "$D/$src"
  if [ ! -f "out_pw/A(k,E).npz" ]; then
    $PY -m unfoldfs.cli unfold --dir . --wavecar WAVECAR --ef auto \
        --outdir out_pw --min-dominance 0.20 2>&1 \
        | grep -E "主导度|E_F |已写出|错误|Traceback" | head -8
  else
    echo "  ($src 已有 out_pw, 跳过)"
  fi
done

echo "=== 2. 原胞参考(权重=1, 与超胞同网格) ==="
cp -f $WORKDIR/../../claude 2>/dev/null || true
EF=$(python3 -c "import sys
with open(sys.argv[1], errors='ignore') as fh:
    for i, line in enumerate(fh):
        if i == 5:
            print(float(line.split()[3])); break" "$D/static_defect_bulk_48/DOSCAR")
echo "  E_F(原胞参考) = $EF"
if [ -f /tmp/pw_analysis/defect_prim48_pwref.npz ]; then
  echo "  (原胞参考已存在, 跳过)"
else
  $PY /tmp/pw_analysis/primitive_reference.py \
    --eigenval $TA/prim96_pw/EIGENVAL \
    --ref-npz "$D/static_defect_bulk_48/out_pw/A(k,E).npz" \
    --prim $TA/prim96_pw/PRIMCELL.vasp \
    --ef $EF --out /tmp/pw_analysis/defect_prim48_pwref.npz
fi
echo "=== 服务器端完成(拉回与作图在本地跑 pull_defect_pw.py) ==="
