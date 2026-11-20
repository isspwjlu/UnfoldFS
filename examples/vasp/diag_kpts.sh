cd /data/wjlu/work/claude/unfold_fs_demo
echo "=== ENERGY.grd 头部 (网格维度) ==="
head -3 ENERGY.grd
echo "=== WEIGHT.grd 头部 ==="
head -3 WEIGHT.grd
echo "=== MOMENTUM.grd 头部 ==="
head -3 MOMENTUM.grd
echo
echo "=== KPATH.in (当前目录) ==="
cat KPATH.in
echo
echo "=== 官方示例 KPATH.in ==="
cat /data/wjlu/program/vaspkit.1.5.1/examples/band_unfolding/MoS2/sc331_perfect/KPATH.in
echo
echo "=== ENERGY.grd 数据段前 5 行 ==="
sed -n '3,8p' ENERGY.grd
