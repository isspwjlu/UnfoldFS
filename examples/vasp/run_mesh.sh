#!/bin/bash
# 超胞稠密网格计算 -> 用于反折叠费米面
set -e
W=/data/wjlu/work/claude/unfold_fs_demo
cd "$W"
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH

# 停掉之前的路径计算
pkill -f 'vasp_std' 2>/dev/null || true
sleep 3

# 10x10x1 超胞网格 -> 30x30 原胞网格
/data/wjlu/program/anaconda3/bin/python3 ../prototypes/make_unfold_mesh.py --n1 10 --n2 10 --n3 1 --tag mesh10

cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
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
KPAR   = 8
EOF
cp -f KPOINTS_mesh10 KPOINTS
echo "=== 启动网格计算 (100 k, 32 核) ==="
rm -f WAVECAR EIGENVAL CHGCAR OUTCAR DOSCAR
nohup mpirun -np 32 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp_mesh.out 2>&1 &
echo "PID=$!"
sleep 120
echo "=== 120 秒后 ==="
grep -cE 'k-point [0-9]+/' vasp_mesh.out || true
grep -E 'k-point [0-9]+/' vasp_mesh.out | tail -3
tail -2 vasp_mesh.out
