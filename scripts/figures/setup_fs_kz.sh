#!/bin/bash
# 带 kz 采样的费米面算例(服务器端)
#
# 为什么需要: 现在的费米面算例是 **2D k 网格**(只有 kz = 0 一层)⇒ 画不出
#   "任意 kz 的 kx-ky 费米面"。本脚本用 `--grid 48 48 3 --grid-type gamma`
#   ⇒ kz = 0, 1/3, 2/3 三层(正好是想要的切片), 6912 个 k 点
#   (≈ 48×48×1 的 3 倍机时, 比 96×96×1 略少)。
#
# ⚠ 物理提醒: 单层体系(1T-TaS2 / 1H-NbSe2, c = 17.66 Å)带在 kz 上**几乎是平的**
#   (层间耦合 e^{-c/3} ≈ 0.003)⇒ 各 kz 切片的费米面**本来就该几乎相同**。
#   这个算例的价值是:
#     (a) 定量确认 kz 无色散(误差几个 meV);
#     (b) 让 A(k,E) 变成**真正的三维体数据** ⇒ FS.cube 是三维体,
#         可以用 FermiSurfer / VESTA 画经典的三维等值面费米面。
#
# 用法: 在服务器上 bash setup_fs_kz.sh
B=/data/wjlu/work/claude/unfoldfs_e2e_ta
PY=/data/wjlu/program/anaconda3/bin/python3
P=/data/wjlu/pesudopotential/vasp_pp/potpaw_PBE.64
W=/tmp/b60
mkdir -p $W && cd $W
export PYTHONPATH=$B/pkg_r10
export PYTHONIOENCODING=utf-8

write_job () {
cat > $1/job.slurm <<'EOF'
#!/bin/bash
#SBATCH -J ufs_ta_fskz
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.error
#SBATCH -p isspfml01
#SBATCH -N 1
#SBATCH --ntasks-per-node=64

source /data/wjlu/program/bashrc
source /data/wjlu/intel-2025/setvars.sh intel64 --force
source /data_gpu/software/bashrc_dir/bashrc.gcc-13

echo "=== node: $(hostname)  start $(date) ==="
echo "=== k points: $(sed -n 2p KPOINTS) ==="

mpirun -np 64 /data/wjlu/program/vasp.6.6.0/bin/vasp_std_6.6.0 > log

echo "=== end $(date) rc=$? ==="
EOF
}

# ---------------- 原胞 48x48x3 (kz = 0, 1/3, 2/3) ----------------
rm -rf $B/runkz_prim
$PY -m unfoldfs.cli gen \
    --supercell $B/run48_prim/POSCAR \
    --primcell $B/run48_prim/PRIMCELL.vasp \
    --transmat $B/run48_prim/TRANSMAT.in \
    --grid 48 48 3 --grid-type gamma --nbands 20 --encut 400 --kpar 4 \
    --outdir $B/runkz_prim 2>&1 | tail -4
cat $P/Ta/POTCAR $P/S/POTCAR > $B/runkz_prim/POSCAR
cat > $B/runkz_prim/INCAR <<'EOF'
ISTART = 0
ICHARG = 2
ISYM = 0
ISMEAR = 1
SIGMA = 0.2
EDIFF = 1E-6
LORBIT = 12
LWAVE = .FALSE.
LCHARG = .FALSE.
IBRION = -1
NSW = 0
PREC = Normal
ENCUT = 400
KPAR = 4
NBANDS = 20
ISPIN = 1
EOF
write_job $B/runkz_prim

# ---------------- 3x1x1 超胞 48x48x3 ----------------
rm -rf $B/runkz_sc3x1x1
$PY -m unfoldfs.cli gen \
    --supercell $B/run48_sc3x1x1/POSCAR \
    --primcell $B/run48_sc3x1x1/PRIMCELL.vasp \
    --transmat $B/run48_sc3x1x1/TRANSMAT.in \
    --grid 48 48 3 --grid-type gamma --nbands 40 --encut 400 --kpar 4 \
    --outdir $B/runkz_sc3x1x1 2>&1 | tail -4
cat $P/S/POTCAR $P/Ta/POTCAR > $B/runkz_sc3x1x1/POSCAR
cat > $B/runkz_sc3x1x1/INCAR <<'EOF'
ISTART = 0
ICHARG = 2
ISYM = 0
ISMEAR = 1
SIGMA = 0.2
EDIFF = 1E-6
LORBIT = 12
LWAVE = .FALSE.
LCHARG = .FALSE.
IBRION = -1
NSW = 0
PREC = Normal
ENCUT = 400
KPAR = 4
NBANDS = 40
ISPIN = 1
EOF
write_job $B/runkz_sc3x1x1

echo "=== kz 取值检查 ==="
$PY - <<'EOF'
import numpy as np
for tag in ("runkz_prim", "runkz_sc3x1x1"):
    p = f"/data/wjlu/work/claude/unfoldfs_e2e_ta/{tag}/KPOINTS"
    lines = [l for l in open(p) if l.strip()]
    zs = sorted({round(float(l.split()[2]), 6) for l in lines[3:] if len(l.split()) >= 3})
    print(f"{tag}: kz = {zs}")
EOF

cd $B/runkz_prim && sbatch job.slurm
cd $B/runkz_sc3x1x1 && sbatch job.slurm
echo "=== DONE (作业已提交) ==="
