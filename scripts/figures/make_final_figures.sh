#!/bin/bash
# 最终成图: 2 材料 × (能带 + 费米面) = 4 幅大图; 每个面板另出 1 幅小图 = 12 幅
#
#   DATA 目录里要有:
#     1T-TaS2 :  PRIMCELL.vasp  tas2_prim.dat  tas2_full_sc.dat
#                tas2_full_scsqrt3.dat  tas2_fs_prim.npz  tas2_fs_sc3x1x1.npz
#                tas2_fs_scsqrt3.npz
#     1H-NbSe2:  NbSe2_prim.vasp  nbse2/prim_assign.dat  nbse2/full_sc3x1x1.dat
#                nbse2/full_scsqrt3.dat  nbse2/prim.npz  nbse2/sc3x1x1.npz
#                nbse2/scsqrt3.npz
#
# 用法:  DATA=/path/to/data bash make_final_figures.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-F:/tmp/tas2}"
OUT="${OUT:-$HERE/../../out/figures}"
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT/small"
cd "$DATA"

# ---------------------------------------------------------------- 1T-TaS2
python "$HERE/plot_band.py" --ef -1.501119 \
  --prim-dat tas2_prim.dat --unfold tas2_full_sc.dat,tas2_full_scsqrt3.dat \
  --titles "3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --labels "GAMMA:0,M:1.078,K:1.700,GAMMA:2.945" \
  --out "$OUT/TaS2_band.png" --split-dir "$OUT/small" --split-prefix "TaS2_band" \
  --e-min -4.2 --e-max 1.0 --sigma 0.05 \
  --suptitle "1T-TaS2 能带: 原胞 band vs 两种超胞反折叠 (Γ-M-K-Γ, 60 点/段, 177 k 点; E_F = 0)" \
  --split-suptitle "1T-TaS2 能带" \
  --note "超胞: full 模式 + 按画窗归一化的线性色标; 原胞: 原 band 直接画 (纯蓝线)"

python "$HERE/plot_fs.py" --prim PRIMCELL.vasp --method peak \
  --npz "tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz,tas2_fs_scsqrt3.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --blue-panels 0 --out "$OUT/TaS2_fs.png" --split-dir "$OUT/small" --split-prefix "TaS2_fs" \
  --suptitle "1T-TaS2 真费米面: E_b 恰过 E_F 处(不是 A(k,E_F) 的等值线)" \
  --split-suptitle "1T-TaS2 费米面" \
  --note "A(k,E_F) 的等值线是费米面高斯脊的「两壁」, 真费米面在两壁中间; 权重按原胞完整点群轨道平均, 取色用笛卡尔等距核(严格协变)"

# ---------------------------------------------------------------- 1H-NbSe2
python "$HERE/plot_band.py" --ef -2.703913 \
  --prim-dat nbse2/prim_assign.dat --unfold nbse2/full_sc3x1x1.dat,nbse2/full_scsqrt3.dat \
  --titles "3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --labels "GAMMA:0,M:1.054,K:1.663,GAMMA:2.880" \
  --out "$OUT/NbSe2_band.png" --split-dir "$OUT/small" --split-prefix "NbSe2_band" \
  --e-min -4.2 --e-max 0.8 --sigma 0.05 \
  --suptitle "1H-NbSe2 单层能带: 原胞 band vs 两种超胞反折叠 (Γ-M-K-Γ, 60 点/段, 177 k 点; E_F = 0)" \
  --split-suptitle "1H-NbSe2 能带" \
  --note "超胞: full 模式 + 按画窗归一化的线性色标; 原胞: 原 band 直接画 (纯蓝线)"

python "$HERE/plot_fs.py" --prim NbSe2_prim.vasp --method peak \
  --npz "nbse2/prim.npz,nbse2/sc3x1x1.npz,nbse2/scsqrt3.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --blue-panels 0 --out "$OUT/NbSe2_fs.png" --split-dir "$OUT/small" --split-prefix "NbSe2_fs" \
  --suptitle "1H-NbSe2 真费米面: E_b 恰过 E_F 处(不是 A(k,E_F) 的等值线)" \
  --split-suptitle "1H-NbSe2 费米面" \
  --note "A(k,E_F) 的等值线是费米面高斯脊的「两壁」, 真费米面在两壁中间; 权重按原胞完整点群轨道平均, 取色用笛卡尔等距核(严格协变)"


# ---------------------------------------------------------------- 三维费米面
# 单层体系 kz 无色散 ⇒ 三维 FS = 二维 FS 轨迹沿 kz 扫成带(sheet = 物理版)
mkdir -p "$OUT/3d"
python "$HERE/plot_fs_3d.py" --prim PRIMCELL.vasp --mode sheet \
  --npz "tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz,tas2_fs_scsqrt3.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --blue-panels 0 --out "$OUT/3d/TaS2_fs3d.png" \
  --suptitle "1T-TaS2 三维反折叠费米面" \
  --note "单层体系 kz 无色散, 三维 FS = 二维 FS 沿 kz 拉伸; kz 方向拉伸 3 倍显示; 颜色 = 谱权重(各栏共享色标)"

python "$HERE/plot_fs_3d.py" --prim NbSe2_prim.vasp --mode sheet \
  --npz "nbse2/prim.npz,nbse2/sc3x1x1.npz,nbse2/scsqrt3.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠,√3×√3 R30° 超胞反折叠" \
  --blue-panels 0 --out "$OUT/3d/NbSe2_fs3d.png" \
  --suptitle "1H-NbSe2 三维反折叠费米面" \
  --note "单层体系 kz 无色散, 三维 FS = 二维 FS 沿 kz 拉伸; kz 方向拉伸 3 倍显示; 颜色 = 谱权重(各栏共享色标)"


# ---------------------------------------------------------------- kz 切片
# 画任意 kz 的 kx-ky 费米面(当前数据只有 kz=0 一层 ⇒ 各切片相同;
# 要真切片需带 kz 采样的算例: bash setup_fs_kz.sh)
python "$HERE/plot_fs_kz.py" --prim PRIMCELL.vasp --kz-list 0,0.3333,0.6667 \
  --npz "tas2_fs_prim.npz,tas2_fs_sc3x1x1.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠" --blue-panels 0 \
  --out "$OUT/TaS2_fs_kz_slices.png" \
  --suptitle "1T-TaS2 费米面的 kz 切片(kx-ky 平面)" \
  --note "单层体系 kz 无色散 ⇒ 各切片本应相同; 现有算例只有 kz=0 一层, 要真切片需 --grid n1 n2 n3 的算例"

python "$HERE/plot_fs_kz.py" --prim NbSe2_prim.vasp --kz-list 0,0.3333,0.6667 \
  --npz "nbse2/prim.npz,nbse2/sc3x1x1.npz" \
  --titles "原胞直通 (direct),3×1×1 超胞反折叠" --blue-panels 0 \
  --out "$OUT/NbSe2_fs_kz_slices.png" \
  --suptitle "1H-NbSe2 费米面的 kz 切片(kx-ky 平面)" \
  --note "单层体系 kz 无色散 ⇒ 各切片本应相同; 现有算例只有 kz=0 一层, 要真切片需 --grid n1 n2 n3 的算例"

echo "=== 大图 ==="; ls -la "$OUT"/*.png
echo "=== 小图 ==="; ls -la "$OUT"/small/*.png
echo "=== 三维 ==="; ls -la "$OUT"/3d/*.png
