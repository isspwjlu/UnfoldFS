#!/bin/bash
# 网格计算: LORBIT=12 (含复投影相位) + KPAR=8 + 32 核
W=/data/wjlu/work/claude/unfold_fs_demo
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH

mkdir -p "$W/mesh12"
cd "$W/mesh12" || exit 1
cp -f "$W"/POSCAR "$W"/POTCAR .
cp -f "$W"/KPOINTS_mesh10 KPOINTS
cp -f "$W"/KPOINTS_MAPPING_TABLE_mesh10.in .
cp -f "$W"/TRANSMAT.in "$W"/PRIMCELL.vasp .
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
KPAR   = 8
EOF
echo "启动 LORBIT=12 网格计算: 100 k, 32 核"
nohup mpirun -np 32 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1 &
echo "PID=$!"
