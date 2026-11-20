#!/bin/bash
# 测试 vaspkit 282 能否处理多 G 映射表（网格展开）
W=$WORKDIR/claude/unfold_fs_demo
cd "$W"

echo "=== 参考算例费米能级 ==="
cat FERMI_ENERGY

echo
echo "=== 备份路径模式文件 ==="
mkdir -p path_mode
cp -f KPOINTS_MAPPING_TABLE.in path_mode/ 2>/dev/null || true
cp -f KPOINTS path_mode/ 2>/dev/null || true
cp -f EBS.dat path_mode/EBS_ref.dat 2>/dev/null || true
ls path_mode/

echo
echo "=== 切换到网格模式 ==="
cp -f KPOINTS_mesh10 KPOINTS
cp -f KPOINTS_MAPPING_TABLE_mesh10.in KPOINTS_MAPPING_TABLE.in
wc -l KPOINTS_MAPPING_TABLE.in

echo
echo "=== vaspkit 282 (网格) ==="
$TOOLSDIR/vaspkit.1.5.1/bin/vaspkit -task 282 2>&1 | tail -40

echo
echo "=== 产物 ==="
ls -la EBS.dat 2>/dev/null
if [ -f EBS.dat ]; then
  echo "--- 块数 ---"
  grep -c 'Kpoint-index' EBS.dat
  echo "--- 前 12 行 ---"
  head -12 EBS.dat
fi
