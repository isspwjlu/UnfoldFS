set -e
W=/data/wjlu/work/claude/unfold_fs_demo
cd "$W"
# Intel MPI 环境（vasp 5.4.4 用的 2020.0.139）
for v in /data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin/mpivars.sh \
         /data/intel-2020/setvars.sh; do
  [ -f "$v" ] && { echo "sourcing $v"; source "$v" intel64 >/dev/null 2>&1 || source "$v" >/dev/null 2>&1 || true; }
done
export PATH=/data/software/intel/2020/compilers_and_libraries_2020.0.139/linux/mpi/intel64/bin:$PATH
command -v mpirun
echo "=== VASP 链接的 MPI ==="
ldd /data/wjlu/program/vasp.5.4.4/bin/vasp_std 2>/dev/null | grep -i -E 'mpi|omp' | head -4
echo "=== 启动 VASP (4 核, 后台) ==="
nohup mpirun -np 4 /data/wjlu/program/vasp.5.4.4/bin/vasp_std > vasp.out 2>&1 &
echo "PID=$!"
sleep 25
echo "=== vasp.out 头部 ==="
head -30 vasp.out
echo "..."
tail -5 vasp.out
