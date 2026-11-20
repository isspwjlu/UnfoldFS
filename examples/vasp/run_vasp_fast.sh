#!/bin/bash
# 优化后的 VASP 运行：NBANDS 固定 + LREAL=Auto + KPAR 并行
set -e
W=$WORKDIR/claude/unfold_fs_demo
cd "$W"
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH

cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
ENCUT  = 350
PREC   = Normal
ISMEAR = 0
SIGMA  = 0.05
EDIFF  = 1E-6
IBRION = -1
NSW    = 0
ISYM   = 0
LREAL  = Auto
NBANDS = 170
KPAR   = 4
EOF

echo "=== 启动路径计算 (159 k, 16 核) ==="
rm -f WAVECAR EIGENVAL CHGCAR OUTCAR
nohup mpirun -np 16 $TOOLSDIR/vasp.5.4.4/bin/vasp_std > vasp_path.out 2>&1 &
echo "PID=$!"
sleep 90
echo "=== 90 秒后进度 ==="
grep -E 'k-point [0-9]+/' vasp_path.out | tail -3
grep -cE 'k-point [0-9]+/' vasp_path.out || true
tail -3 vasp_path.out
