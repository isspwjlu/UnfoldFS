#!/bin/bash
# vaspkit 282 网格展开：先试"多 G"映射表(900 行)，失败则降级为 9 次单 G 运行
W=$WORKDIR/claude/unfold_fs_demo
cd "$W"
VK=$TOOLSDIR/vaspkit.1.5.1/bin/vaspkit

echo "############ 1) 多 G 映射表 (900 行) ############"
cp -f KPOINTS_mesh10 KPOINTS
cp -f KPOINTS_MAPPING_TABLE_mesh10.in KPOINTS_MAPPING_TABLE.in
rm -f EBS.dat
$VK -task 282 2>&1 | tail -25
NB=$(grep -c 'Kpoint-index' EBS.dat 2>/dev/null || echo 0)
echo ">>> 块数 = $NB (期望 900)"

if [ "$NB" -lt 800 ]; then
  echo
  echo "############ 降级：拆成 9 组单 G 映射表 ############"
  rm -f EBS_all.dat
  for g in 0 1 2 3 4 5 6 7 8; do
    awk -v g=$g 'NR>1 && (NR-2)%9==g {print}' KPOINTS_MAPPING_TABLE_mesh10.in > table_g${g}.in
    n=$(wc -l < table_g${g}.in)
    cp -f table_g${g}.in KPOINTS_MAPPING_TABLE.in
    rm -f EBS.dat
    $VK -task 282 > vk_g${g}.log 2>&1
    b=$(grep -c 'Kpoint-index' EBS.dat 2>/dev/null || echo 0)
    echo "  G组 $g: 映射 $n 行 -> EBS 块数 $b"
    if [ -f EBS.dat ]; then cat EBS.dat >> EBS_all.dat; fi
  done
  cp -f EBS_all.dat EBS.dat
  echo ">>> 合并后块数 = $(grep -c 'Kpoint-index' EBS.dat)"
fi

echo
echo "############ 产物检查 ############"
ls -la EBS.dat
echo "--- 前 8 行 ---"
head -8 EBS.dat
echo "--- 权重范围 ---"
awk '!/Kpoint-index/{print $3}' EBS.dat | sort -g | sed -n '1p;$p'
