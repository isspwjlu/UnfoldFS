#!/bin/bash
W=/data/wjlu/work/claude/unfold_fs_demo
source /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh intel64 >/dev/null 2>&1 || true
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH
mkdir -p "$W/mesh_run"
cd "$W/mesh_run" || exit 1
cp -f "$W"/POSCAR "$W"/POTCAR "$W"/INCAR .
cp -f "$W"/KPOINTS_mesh10 KPOINTS
cp -f "$W"/KPOINTS_MAPPING_TABLE_mesh10.in .
cp -f "$W"/TRANSMAT.in "$W"/PRIMCELL.vasp .
ls
nohup mpirun -np 16 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1 &
echo "网格计算 PID=$!"
sleep 45
tail -2 vasp.out
