cd /data/wjlu/work/claude/unfold_fs_demo/mesh_run2
mkdir -p fs_out
/data/wjlu/program/anaconda3/bin/python3 ../unfold_procar.py --dir . --outdir fs_out 2>&1 | tail -30
echo
echo "=== 产物 ==="
ls -la fs_out/
echo "--- unfolded_bands.dat 前 8 行 ---"
head -8 fs_out/unfolded_bands.dat
