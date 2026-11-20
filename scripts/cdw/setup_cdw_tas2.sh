#!/bin/bash
# 1T-TaS2 的 √3×√3 R30° **CDW 态**超胞: 服务器端计算 + 反折叠。
#
# 流程:
#   1. 本地建结构:  python scripts/cdw/build_cdw_tas2.py \
#                       --poscar <未畸变的 √3×√3 POSCAR> --out POSCAR_CDW
#      (CDW 位移场见该脚本顶部的说明)
#   2. 上传 POSCAR_CDW 到服务器, 拷一份未畸变的超胞目录当模板
#      (KPOINTS/PRIMCELL.vasp/TRANSMAT.in/INCAR/POTCAR 全部一样 ⇒
#       展开后的谱可以直接和未畸变的对比)
#   3. sbatch 跑 VASP (176 个 k 点, 9 个原子, 96 条带)
#   4. 反折叠:  $PY -m unfoldfs.cli band-unfold --dir <run_dir>
#
# 注意: 集群是共用的, 节点经常占满 ⇒ 提交前先看 squeue, 必要时换分区
# (isspfml01/02/03 是 CPU 分区; isspfml_gpu* 的 QOS 只给 6 个 CPU, 跑不动 VASP)。
set -e
B=/data/wjlu/work/claude/unfoldfs_e2e_ta
PY=/data/wjlu/program/anaconda3/bin/python3
P=/data/wjlu/pesudopotential/vasp_pp/potpaw_PBE.64
TPL=$B/run_band60_scsqrt3          # 未畸变的 √3×√3 能带算例(当模板)
DST=$B/run_band60_scsqrt3_cdw      # CDW 算例
export PYTHONPATH=$B/pkg_r10
export PYTHONIOENCODING=utf-8

if [ ! -f "$DST/POSCAR" ]; then
    echo "先把 build_cdw_tas2.py 生成的 CDW POSCAR 上传到 $DST/POSCAR"
    exit 1
fi

# ---- INCAR / POTCAR / KPOINTS 与模板完全一致(只换 POSCAR) ----
cp "$TPL/INCAR" "$TPL/KPOINTS" "$TPL/PRIMCELL.vasp" "$TPL/TRANSMAT.in" "$DST/"
cat "$P/S/POTCAR" "$P/Ta/POTCAR" > "$DST/POTCAR"

cat > "$DST/job.slurm" <<'EOF'
#!/bin/bash
#SBATCH -J ufs_ta_cdw
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.error
#SBATCH -p isspfml01
#SBATCH -N 1
#SBATCH --ntasks-per-node=32

source /data/wjlu/program/bashrc
source /data/intel-2025/setvars.sh intel64 --force
source /data_gpu/software/bashrc_dir/bashrc.gcc-13

echo "=== node: $(hostname)  start $(date) ==="
echo "=== k points: $(sed -n 2p KPOINTS) ==="

mpirun -np 32 /data/wjlu/program/vasp.6.6.0/bin/vasp_std_6.6.0 > log

echo "=== end $(date) rc=$? ==="
grep -c "reached required accuracy" OUTCAR
EOF

echo "=== 提交 ==="
cd "$DST" && sbatch job.slurm

cat > "$DST/unfold.sh" <<'EOF'
#!/bin/bash
# 算完之后: 反折叠 -> vaspkit 的长表(unfolded_band.dat)
set -e
B=/data/wjlu/work/claude/unfoldfs_e2e_ta
PY=/data/wjlu/program/anaconda3/bin/python3
export PYTHONPATH=$B/pkg_r10
export PYTHONIOENCODING=utf-8
cd $B/run_band60_scsqrt3_cdw
$PY -m unfoldfs.cli band-unfold --dir . 2>&1 | tail -20
ls -la unfolded_band.dat
EOF
echo "=== 算完后跑: bash $DST/unfold.sh ==="
