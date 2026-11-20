set -e
W=$WORKDIR/claude/unfold_fs_demo
PP64=$PPDIR/vasp_pp/potpaw_PBE.64

echo "=== potpaw_PBE.64/Mo/POTCAR 的 TITEL (确认 VASP6 格式) ==="
grep TITEL "$PP64/Mo/POTCAR" | head -2
grep TITEL "$PP64/S/POTCAR" | head -2

source /data/intel-2025/setvars.sh intel64 --force >/dev/null 2>&1 || true
source /data_gpu/software/bashrc_dir/bashrc.gcc-13 >/dev/null 2>&1 || true
export PATH=$TOOLSDIR/vasp.6.6.0/bin:$PATH
export PATH=/data/software/intel/2025/oneapi/bin:$PATH

mkdir -p "$W/v660"
cd "$W/v660"
cp -f "$W"/POSCAR .
cat "$PP64"/Mo/POTCAR > POTCAR
cat "$PP64"/S/POTCAR  >> POTCAR
cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
LORBIT = 12
ENCUT  = 320
PREC   = Normal
ISMEAR = 0
SIGMA  = 0.05
EDIFF  = 1E-6
IBRION = -1
NSW    = 0
ISYM   = 0
LREAL  = Auto
NBANDS = 120
EOF
cat > KPOINTS <<'EOF'
VASP 6.6.0 LORBIT12 test
   2
Reciprocal lattice
   0.000000000000   0.000000000000   0.000000000000       1
   0.100000000000   0.000000000000   0.000000000000       1
EOF
echo "=== 启动 VASP 6.6.0 ==="
mpirun -np 8 $TOOLSDIR/vasp.6.6.0/bin/vasp_std_6.6.0 > vasp.out 2>&1 || true
grep -E "Error reading|error occured|licence|license|LICENSE" vasp.out | head -5
tail -5 vasp.out
ls -la PROCAR 2>/dev/null && {
  echo "=== PROCAR 头部 ==="
  head -12 PROCAR | cut -c1-140
  echo "=== 行字段数分布 ==="
  awk 'NR>=5 && NR<=400 {print NF}' PROCAR | sort | uniq -c
}
