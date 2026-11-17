cd /data/wjlu/work/claude/unfold_fs_demo
echo "=== vk_all.log 完整 (30-90 行) ==="
sed -n '1,60p' vk_all.log
echo "..."
echo "=== 单组 G 表是否存在 (fallback 产物) ==="
ls -la table_g*.in 2>/dev/null | head -3
echo
echo "=== 官方路径模式 EBS.dat 对照 ==="
head -5 /data/wjlu/program/vaspkit.1.5.1/examples/band_unfolding/MoS2/sc331_perfect/EBS.dat
echo
echo "=== 官方路径模式 KPOINTS 首行 vs 我的网格首行 ==="
head -1 /data/wjlu/program/vaspkit.1.5.1/examples/band_unfolding/MoS2/sc331_perfect/KPOINTS
head -1 KPOINTS_mesh10
