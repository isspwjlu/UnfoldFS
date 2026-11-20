W=$WORKDIR/claude/unfold_fs_demo
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH
cd "$W/lo12" || exit 1
cat > KPOINTS <<'EOF'
LORBIT12 test
   2
Reciprocal lattice
   0.000000000000   0.000000000000   0.000000000000       1
   0.100000000000   0.000000000000   0.000000000000       1
EOF
rm -f PROCAR
mpirun -np 8 $TOOLSDIR/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1
grep -E "Error reading|error occured" vasp.out | head -3
ls -la PROCAR 2>/dev/null && {
  echo "=== PROCAR 前 16 行 ==="
  head -16 PROCAR | cut -c1-170
  echo "=== 各行列数 ==="
  awk 'NR>=5 && NR<=300 {print NF}' PROCAR | sort | uniq -c
}
