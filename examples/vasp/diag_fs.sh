cd $WORKDIR/claude/unfold_fs_demo/mesh12
echo "=== DOSCAR 前 6 行 ==="
head -6 DOSCAR
echo
echo "=== 展开能量范围 ==="
awk '!/^#/{print $4}' fs_out/unfolded_bands.dat | sort -g | sed -n '1p;$p'
echo
echo "=== 各能量的权重总和 (前 20 个 E 附近的最大值) ==="
awk '!/^#/ {if($5>m[$4]) m[$4]=$5; s[$4]+=$5} END{for(e in s) print e, s[e]}' fs_out/unfolded_bands.dat | sort -g | tail -8
echo
echo "=== 权重最大的一些条目 ==="
sort -k5 -g -r fs_out/unfolded_bands.dat | head -8
echo
echo "=== fermi_surface_map.dat 前 5 行 & 最大值 ==="
head -3 fs_out/fermi_surface_map.dat
sort -k4 -g -r fs_out/fermi_surface_map.dat | head -3
