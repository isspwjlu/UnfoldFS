cd /data/wjlu/work/claude/unfold_fs_demo
VK=/data/wjlu/program/vaspkit.1.5.1/bin/vaspkit
mkdir -p fs261
cd fs261 || exit 1
cp -f ../POSCAR ../INCAR .
echo "=== vaspkit 261 (生成费米面用 KPOINTS) ==="
printf '0.06\n' | $VK -task 261 2>&1 | grep -E "Error|Written|K-Mesh|resolution|Input" | head -12
echo "--- 生成的 KPOINTS 前 10 行 ---"
head -10 KPOINTS 2>/dev/null
echo "--- 行数 ---"
wc -l KPOINTS 2>/dev/null
