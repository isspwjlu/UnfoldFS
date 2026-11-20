cd /data/wjlu/work/claude/unfold_fs_demo
echo "=== VASP 如何解读该 KPOINTS (OUTCAR 记录) ==="
grep -A 12 -i "reciprocal lattice" OUTCAR | head -20
echo
echo "=== OUTCAR 里的 k-point 列表(前 6 个) ==="
grep -E "k-point +[0-9]+ :" OUTCAR | head -6
echo
echo "=== 点数与权重 ==="
grep -E "number of k-points|irreducible" OUTCAR | head -4
echo
echo "=== 进度 ==="
grep -cE 'k-point [0-9]+/' vasp_path.out || true
grep -E 'k-point [0-9]+/' vasp_path.out | tail -2
