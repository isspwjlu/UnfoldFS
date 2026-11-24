#!/bin/bash
# ⚠ 下面的 $WORKDIR / $TOOLSDIR / $PPDIR 是**占位符**(原为某台机器上的
#   具体路径, 已从公开仓库中移除)。使用前请自行设置:
#   export WORKDIR=/path/to/work  TOOLSDIR=/path/to/tools  PPDIR=/path/to/pp
# 缺陷超胞的平面波路线 —— **服务器端**部分: 检查 → 展开 → 原胞参考 → 数字。
#
# 幂等: 已经做过的步骤会跳过 ⇒ 可以反复跑(监控用)。
# 退出码: 0 = 全部做完; 3 = 计算还没完(稍后再跑)
#
# 前置:
#   $TA/defect/static_defect_bulk/   缺陷超胞的 VASP 算例(WAVECAR 写完)
#   $TA/prim96_pw/                 体相原胞 96x96x1 的 VASP 算例(给参考)
set -u
TA=$WORKDIR
D=$TA/defect
LOCAL="${LOCAL:-$HERE/../..}"
export PYTHONPATH=$TA/pkg_pw2
export PYTHONIOENCODING=utf-8

echo "=== 1. 检查两个 VASP 算例 ==="
# 作业名(slurm 会把名字截断成 12 字符, 故用前缀匹配)
for pair in "101739:$D/static_defect_bulk" "101740:$TA/prim96_pw"; do
  jid=${pair%%:*}; d=${pair#*:}
  w=$(stat -c %s "$d/WAVECAR" 2>/dev/null || echo 0)
  k=$(grep -c 'DAV:' "$d/OSZICAR" 2>/dev/null || echo 0)
  # 完成判据: ① 作业已不在跑(sacct 按**作业 ID** 查, 名字会被截断) 
  #          ② WAVECAR 已写出(VASP 边算边写, 大小不能作判据)
  # ⚠ slurm 会在状态后加 "+"(如 "RUNNING+"), 先去掉再匹配
  state=$(sacct -j "$jid" --format=State --noheader 2>/dev/null | tr -d ' ' | sed 's/+//' | grep -E '^COMPLETED' | head -1)
  echo "  $(basename $d): WAVECAR=$w  作业 $jid 已完成=${state:-否}"
  # 保守判据: 必须**明确**看到 COMPLETED 才算完; 状态读不出来也当没算完
  if [ "$w" -lt 1000000 ] || [ "$state" != "COMPLETED" ]; then
    echo ">>> 还没算完, 退出(稍后再跑)"; exit 3
  fi
done

echo "=== 2. 缺陷超胞的展开(主导度闸门放宽: 缺陷态不是原胞平移的本征态) ==="
cd $D/static_defect_bulk
if [ ! -f "out_pw/A(k,E).npz" ]; then
  python3 -m unfoldfs.cli unfold --dir . --wavecar WAVECAR --ef auto \
      --outdir out_pw --min-dominance 0.20 2>&1 \
      | grep -E "权重定义|主导度|E_F |已写出|错误|Traceback" | head -10
else
  echo "  (已有 out_pw, 跳过)"
fi

echo "=== 2b. 体相完美 3x3x1 超胞的展开(同网格对照) ==="
cd $D/static_perfect_bulk
if [ ! -f "out_pw/A(k,E).npz" ]; then
  python3 -m unfoldfs.cli unfold --dir . --wavecar WAVECAR --ef auto       --outdir out_pw --min-dominance 0.20 2>&1       | grep -E "权重定义|主导度|E_F |已写出|错误|Traceback" | head -8
else
  echo "  (已有 out_pw, 跳过)"
fi

echo "=== 3. 原胞参考(权重=1, 与超胞同网格) ==="
mkdir -p /tmp/pw_analysis
cp -f $LOCAL/scripts/analysis/primitive_reference.py /tmp/pw_analysis/ 2>/dev/null
# ⚠ 直接原胞参考必须用**原胞自己**的 E_F(从它的 DOSCAR 读),
# 不能用超胞的 E_F —— 否则画出来的不是体相的费米面(大小差很多)。
EF=$(python3 -c "import sys
with open(sys.argv[1], errors='ignore') as fh:
    for i, line in enumerate(fh):
        if i == 5:
            print(float(line.split()[3])); break" "$TA/prim96_pw/DOSCAR")
echo "  E_F(原胞参考) = $EF"
cd $D
if [ -f /tmp/pw_analysis/defect_prim96_pwref.npz ]; then
  echo "  (原胞参考已存在, 跳过)"
else
python3 /tmp/pw_analysis/primitive_reference.py \
  --eigenval $TA/prim96_pw/EIGENVAL \
  --ref-npz "$D/static_defect_bulk/out_pw/A(k,E).npz" \
  --prim $TA/prim96_pw/PRIMCELL.vasp \
  --ef $EF --out /tmp/pw_analysis/defect_prim96_pwref.npz
fi

echo "=== 3. 服务器端完成(数字与图在本地跑 finish_defect_pw.py) ==="
