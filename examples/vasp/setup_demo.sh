set -e
PP=/data/wjlu/pesudopotential/vasp_pp/potpaw_PBE.54
EX=/data/wjlu/program/vaspkit.1.5.1/examples/band_unfolding/MoS2/sc331_perfect
W=/data/wjlu/work/claude/unfold_fs_demo
mkdir -p "$W"
cd "$W"
cp -f "$EX"/POSCAR "$EX"/KPOINTS "$EX"/TRANSMAT.in "$EX"/KPOINTS_MAPPING_TABLE.in \
   "$EX"/KPATH.in "$EX"/PRIMCELL.vasp "$EX"/SUPERCELL.vasp "$EX"/KLABELS \
   "$EX"/FERMI_ENERGY "$EX"/POSCAR.0 .
# VASP 5.4 格式 POTCAR: potpaw_PBE.54 每元素目录下已是合并好的 POTCAR
cat "$PP"/Mo/POTCAR > POTCAR
cat "$PP"/S/POTCAR  >> POTCAR
grep -m2 TITEL POTCAR
cat > INCAR <<'EOF'
ISTART = 0
ICHARG = 2
LWAVE  = .TRUE.
LCHARG = .FALSE.
LORBIT = 11
ENCUT  = 350
PREC   = Normal
ISMEAR = 0
SIGMA  = 0.05
EDIFF  = 1E-6
IBRION = -1
NSW    = 0
ISYM   = 0
EOF
echo "=== 超胞 POSCAR ==="
head -10 POSCAR
echo "=== 原胞 PRIMCELL.vasp ==="
head -10 PRIMCELL.vasp
echo "=== TRANSMAT.in ==="
cat TRANSMAT.in
echo "=== KPOINTS 首行 (k点数) ==="
head -2 KPOINTS
echo "=== vasp 5.4.4 构建信息 ==="
grep -i -E 'MPIROOT|OMPI_|I_MPI' /data/wjlu/program/vasp.5.4.4/makefile.include | head -6
ls /data/wjlu/program/vasp.5.4.4/bin/
