#!/bin/bash
# ⚠ 下面的 $WORKDIR / $TOOLSDIR / $PPDIR 是**占位符**(原为某台机器上的
#   具体路径, 已从公开仓库中移除)。使用前请自行设置:
#   export WORKDIR=/path/to/work  TOOLSDIR=/path/to/tools  PPDIR=/path/to/pp
# 体相 1T-TaS2 的 3x3x1 超胞 + S 空位缺陷 (Ta9S16): 服务器端计算 + 反折叠。
#
# 阶段 (按顺序执行):
#   relax_prim    体相原胞(3 原子)全弛豫 ISIF=3 -> 拿到干净的 a / c / z_S
#   relax_defect  缺陷超胞 Ta9S16(25 原子)离子弛豫 ISIF=2 (**固定晶格**)
#   static        三个静态算例: 原胞 / 完美 3x3x1 Ta9S18 / 缺陷 Ta9S16
#   unfold        对三个静态算例跑 unfoldfs unfold
#
# 为什么弛豫要固定晶格: 超胞与原胞的平移关系 (TRANSMAT) 要求
# `M @ PRIMCELL.lattice == SUPERCELL.lattice`; 晶格一变这个关系就断了,
# 反折叠的原子副本匹配 (整数平移判据) 随之失效。
# 弛豫后用 fix_lattice.py 把晶格强制改回参考值。
#
# 赝势: 用**用户自己的** Ta_pv (PAW_PBE Ta_pv 07Sep2000, 含 5p 半芯)。
#   potpaw_PBE.64 的 Ta 是 6s5d 半核实 (ZVAL=5) => Ta 的 5p 半芯被当成芯,
#   带能整体偏高 25 eV 量级, 与文献不可比。
#
# 网格: unfoldfs gen --grid 48 48 6
#   原胞网格 48x48x6 = 13824 点; 超胞 k 点 = (48/3)x(48/3)x6 = 16x16x6 = 1536。
#   P1 必须是 3 的倍数 (超胞 BZ 只被 P1/3 个点采样, 否则网格不自洽)。
#
# 注意: 集群是共用的, 节点经常占满 => 提交前先看 squeue。
#       isspfml_gpu* 的 QOS 只给 6 个 CPU, 跑不动 VASP。
set -e

B=$WORKDIR/defect
PY=$TOOLSDIR/anaconda3/bin/python3
VASP=$TOOLSDIR/vasp.6.6.0/bin/vasp_std_6.6.0
# 赝势: 与**单层 1T-TaS2 算例一致**(potpaw_PBE.64 的 Ta, 6s5d 半核实 + S),
# 使体相与单层的电子结构可比(FS 半径一致)。用户 2026-11-23 决定。
# ⚠ 半核实 Ta 把 5p 半芯当芯(物理上 Ta_pv 才对); 已在 fact_sheet 标注。
POTCAR_USER=$PPDIR/vasp_pp/potpaw_PBE.64
export PYTHONPATH=$WORKDIR/pkg_r19
export PYTHONIOENCODING=utf-8

GRID="96 96 1"   # monolayer: same 96x96x1 mesh as the paper Fig.4 data
NBANDS_SUPER=120
NBANDS_PRIM=30

slurm_job () {   # $1 = 目录, $2 = 任务名, $3 = ntasks
  local DIR=$1 NAME=$2 NT=$3
  cat > "$DIR/job.slurm" <<EOF
#!/bin/bash
#SBATCH -J $NAME
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.error
#SBATCH -p isspfml01
#SBATCH -N 1
#SBATCH --ntasks-per-node=$NT

source $TOOLSDIR/bashrc
source /data/intel-2025/setvars.sh intel64 --force
source /data_gpu/software/bashrc_dir/bashrc.gcc-13

echo "=== node: \$(hostname)  start \$(date) ==="
echo "=== k points: \$(sed -n 2p KPOINTS) ==="

mpirun -np $NT $VASP > log

echo "=== end \$(date) rc=\$? ==="
grep -c "reached required accuracy" OUTCAR || true
EOF
}

# ---------------------------------------------------------------- relax_prim
if [ "$1" = "relax_prim" ]; then
  DIR=$B/relax_prim
  rm -rf "$DIR"; mkdir -p "$DIR"
  cp "$B/struct/PRIMCELL.vasp" "$DIR/POSCAR"
  cat "$POTCAR_USER/Ta/POTCAR" "$POTCAR_USER/S/POTCAR" > "$DIR/POTCAR"
  cat > "$DIR/INCAR" <<'EOF'
SYSTEM = TaS2 bulk primitive
ISTART = 0
ICHARG = 2
ISYM = 0
ISPIN = 1
ISMEAR = 1
SIGMA = 0.2
EDIFF = 1E-6
LORBIT = 12
LWAVE = .FALSE.
LCHARG = .FALSE.
IBRION = 2
NSW = 100
EDIFFG = -0.02
ISIF = 3
PREC = Normal
ENCUT = 400
KPAR = 4
NBANDS = 30
EOF
  cat > "$DIR/KPOINTS" <<'EOF'
Automatic mesh
0
Gamma
8 8 6
EOF
  slurm_job "$DIR" ufs_ta_relax_prim 64
  cd "$DIR" && sbatch job.slurm
  echo "=== 已提交原胞弛豫: $DIR ==="
fi

# ------------------------------------------------------------- relax_defect
if [ "$1" = "relax_defect" ]; then
  # 需要先把原胞弛豫后的参数写回结构 (见 rebuild_prim 步骤)
  [ -f "$B/struct/PRIMCELL_relaxed.vasp" ] || {
    echo "缺少 $B/struct/PRIMCELL_relaxed.vasp —— 先用原胞弛豫结果重建超胞"; exit 1; }
  DIR=$B/relax_defect
  rm -rf "$DIR"; mkdir -p "$DIR"
  cp "$B/struct/sc3x3x1_defect.vasp" "$DIR/POSCAR"
  cat "$POTCAR_USER/Ta/POTCAR" "$POTCAR_USER/S/POTCAR" > "$DIR/POTCAR"
  cat > "$DIR/INCAR" <<'EOF'
SYSTEM = TaS2 bulk 3x3x1 supercell with 2 S vacancies (Ta9S16)
ISTART = 0
ICHARG = 2
ISYM = 0
ISPIN = 1
ISMEAR = 1
SIGMA = 0.2
EDIFF = 1E-6
LORBIT = 12
LWAVE = .FALSE.
LCHARG = .FALSE.
IBRION = 2
NSW = 100
EDIFFG = -0.02
ISIF = 2
PREC = Normal
ENCUT = 400
KPAR = 4
NBANDS = 120
EOF
  cat > "$DIR/KPOINTS" <<'EOF'
u explicit k-point list (do not edit) -- G grid 4x4x2
4
Reciprocal lattice
0.000000000000 0.000000000000 0.000000000000 1.000000000000
0.000000000000 0.500000000000 0.000000000000 1.000000000000
0.500000000000 0.000000000000 0.000000000000 1.000000000000
0.500000000000 0.500000000000 0.000000000000 1.000000000000
EOF
  slurm_job "$DIR" ufs_ta_relax_def 64
  cd "$DIR" && sbatch job.slurm
  echo "=== 已提交缺陷超胞弛豫: $DIR ==="
fi

# ------------------------------------------------------------------- static
# 三个算例各自独立提交(static_prim / static_perfect / static_defect),
# 这样缺陷超胞的静态计算可以在它自己的弛豫完成后再提交, 不必等其它两个。
if [ "$1" = "static_prim" ] || [ "$1" = "static_perfect" ] || [ "$1" = "static_defect" ]; then
  case "$1" in
    static_prim)
      DIR=$B/static_prim
      POSCAR=$B/struct/PRIMCELL_relaxed.vasp
      NB=$NBANDS_PRIM
      ;;
    static_perfect)
      DIR=$B/static_perfect
      POSCAR=$B/struct/sc3x3x1_perfect_relaxed.vasp
      NB=$NBANDS_SUPER
      ;;
    static_defect)
      DIR=$B/static_defect
      POSCAR=$B/struct/sc3x3x1_defect_relaxed.vasp
      NB=$NBANDS_SUPER
      ;;
  esac
  rm -rf "$DIR"; mkdir -p "$DIR"
  cp "$POSCAR" "$DIR/POSCAR"
  cp "$B/struct/PRIMCELL_relaxed.vasp" "$DIR/PRIMCELL.vasp"
  if [ "$1" = "static_prim" ]; then
    # 原胞是"直通模式"(det(M)=1)
    cat > "$DIR/TRANSMAT.in" <<'EOF'
Read transformation matrix from the TRANSMAT.in file if it exists.
  1  0  0   # must be three integers
  0  1  0   # must be three integers
  0  0  1   # must be three integers
EOF
  else
    cp "$B/struct/TRANSMAT.in" "$DIR/TRANSMAT.in"
    if [ "$1" = "static_defect" ]; then
      # 原子对应用**理想位置**的缺陷超胞(弛豫后离子偏离格点, 整数平移判据会失败)
      cp "$B/struct/sc3x3x1_defect.vasp" "$DIR/SUPERCELL_IDEAL.vasp"
    fi
  fi
  $PY -m unfoldfs.cli gen --supercell "$DIR/POSCAR" --primcell "$DIR/PRIMCELL.vasp" \
      --transmat "$DIR/TRANSMAT.in" --grid $GRID --nbands $NB --outdir "$DIR"
  grep -v '^#' "$DIR/INCAR.suggested" | grep -v '^$' > "$DIR/INCAR"
  grep -q '^ISPIN' "$DIR/INCAR" || echo 'ISPIN = 1' >> "$DIR/INCAR"
  cat "$POTCAR_USER/Ta/POTCAR" "$POTCAR_USER/S/POTCAR" > "$DIR/POTCAR"
  slurm_job "$DIR" ufs_ta_$1 64
  (cd "$DIR" && sbatch job.slurm)
  echo "=== 已提交 $1 ($DIR) ==="
fi

# ------------------------------------------------------------------- unfold
if [ "$1" = "unfold" ]; then
  # ⚠ 用**理想化晶格**的结构文件: VASP 弛豫后的晶格带 ~2.6e-5 A 的数值倾斜,
  #   它让 spglib 在 symprec=1e-5 下只认出 4 个点群操作(而不是 12 个)
  #   => 轨道平均只恢复部分对称性。原胞与超胞必须**一起**理想化:
  #   原子副本匹配的整数平移容差只有 1e-6, 两边晶格不自洽就直接失败。
  # ⚠ 超胞的原子对应用**理想位置**的版本: 弛豫后离子偏离格点, 严格匹配会失败。
  for D in static_prim static_perfect static_defect; do
    DIR=$B/$D
    echo "================ $D ================"
    cp "$B/struct/PRIMCELL_ideal.vasp" "$DIR/PRIMCELL.vasp"
    # gen will overwrite POSCAR/PRIMCELL/TRANSMAT => keep a copy of the
    # structure VASP actually used
    [ -f "$DIR/POSCAR" ] && cp -n "$DIR/POSCAR" "$DIR/POSCAR.dft"
    case "$D" in
      static_prim)
        # 直通模式: supercell 就是原胞本身
        cp "$B/struct/PRIMCELL_ideal.vasp" "$DIR/POSCAR"
        ( cd "$DIR" && $PY -m unfoldfs.cli gen --supercell POSCAR --primcell PRIMCELL.vasp \
            --transmat TRANSMAT.in --grid $GRID --nbands $NBANDS_PRIM --outdir . 2>&1 | tail -3
          $PY -m unfoldfs.cli unfold --dir . --ef auto --sigma 0.06 \
            --outdir fs 2>&1 | tail -30 ) ;;
      static_perfect)
        cp "$B/struct/sc3x3x1_perfect_ideal.vasp" "$DIR/SUPERCELL_IDEAL.vasp"
        ( cd "$DIR" && $PY -m unfoldfs.cli gen --supercell SUPERCELL_IDEAL.vasp --primcell PRIMCELL.vasp \
            --transmat TRANSMAT.in --grid $GRID --nbands $NBANDS_SUPER --outdir . 2>&1 | tail -3
          $PY -m unfoldfs.cli unfold --dir . --ef auto --sigma 0.06 \
            --supercell SUPERCELL_IDEAL.vasp --outdir fs 2>&1 | tail -30 ) ;;
      static_defect)
        cp "$B/struct/sc3x3x1_defect_ideal.vasp" "$DIR/SUPERCELL_IDEAL.vasp"
        ( cd "$DIR" && $PY -m unfoldfs.cli gen --supercell SUPERCELL_IDEAL.vasp --primcell PRIMCELL.vasp \
            --transmat TRANSMAT.in --grid $GRID --nbands $NBANDS_SUPER --outdir . 2>&1 | tail -3
          $PY -m unfoldfs.cli unfold --dir . --ef auto --sigma 0.06 \
            --supercell SUPERCELL_IDEAL.vasp --outdir fs 2>&1 | tail -30 ) ;;
    esac
  done
fi
