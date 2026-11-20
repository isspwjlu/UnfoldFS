#!/bin/bash
# 并行跑: 路径模式(验证用) + 网格模式(费米面用), 两者都要 PROCAR(LORBIT=11)
W=/data/wjlu/work/claude/unfold_fs_demo
EX=/data/wjlu/program/vaspkit.1.5.1/examples/band_unfolding/MoS2/sc331_perfect
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH
cd "$W" || exit 1

cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
LORBIT = 11
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

# --- 路径模式 ---
mkdir -p path_run
cd "$W/path_run"
cp -f ../POSCAR ../POTCAR ../INCAR .
cp -f $EX/KPOINTS .
cp -f $EX/KPOINTS_MAPPING_TABLE.in .
cp -f $EX/KPATH.in $EX/KLABELS $EX/TRANSMAT.in $EX/PRIMCELL.vasp .
nohup mpirun -np 16 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1 &
echo "路径计算 PID=$!"

# --- 网格模式 ---
mkdir -p mesh_run
cd "$W/mesh_run"
cp -f ../POSCAR ../POTCAR ../INCAR .
cp -f ../KPOINTS_mesh10 KPOINTS
cp -f ../KPOINTS_MAPPING_TABLE_mesh10.in .
cp -f ../TRANSMAT.in ../PRIMCELL.vasp .
nohup mpirun -np 16 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1 &
echo "网格计算 PID=$!"

sleep 60
echo "=== 60 秒后 ==="
echo "--- path_run ---"; grep -cE 'k-point [0-9]+/' "$W/path_run/vasp.out" || true; tail -1 "$W/path_run/vasp.out"
echo "--- mesh_run ---"; grep -cE 'k-point [0-9]+/' "$W/mesh_run/vasp.out" || true; tail -1 "$W/mesh_run/vasp.out"
