"""从原胞 POSCAR 生成超胞 POSCAR + TRANSMAT.in（unfoldfs gen 需要 --supercell）。

约定（与服务器上现有 3x1x1 算例一致）：
  * 晶格行 = M @ prim_lattice（行向量约定）
  * 超胞分数坐标 s = (f_prim + T) @ M^{-1}，T 为整数平移（**行向量**约定；
    写成 M^{-1} @ (f_prim + T) 会把行向量当列向量，原子位置会错）
  * 副本 T 必须是**余类代表元**：(T @ M^{-1}) 逐分量落在 [0,1) 内（不取模）；
    对 M=[[2,1],[-1,1]] 即 (0,0),(1,0),(1,1)。直接取 T=(i,0,0) 会得到错的
    结构（√3×√3 的 Ta-Ta 变成 1.94 Å，应为 3.36 Å，空间群也退化）。
    下面的循环就是按这个判据扫出来的。
  * 元素顺序与 POTCAR 一致（S 在前）
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PRIM = ROOT / ".fsdata" / "TaS2_PRIMCELL.vasp"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".fsdata"

SPEC = sys.argv[1] if len(sys.argv) > 1 else "6x6x1"
if ";" in SPEC:                      # 显式矩阵, 分号分行
    M = np.array([[int(x) for x in row.split()] for row in SPEC.split(";")], dtype=int)
elif "x" in SPEC:                   # 对角形式 3x1x1
    d = [int(x) for x in SPEC.split("x")]
    assert len(d) == 3, d
    M = np.diag(d).astype(int)
else:
    M = np.diag([N, N, 1]).astype(int)
NSUP = int(round(abs(np.linalg.det(M))))


def read_prim(path):
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)])
    lat *= scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    assert lines[7].strip().lower().startswith(("d", "c", "k")), lines[7]
    coords = np.array([[float(x) for x in lines[8 + j].split()[:3]] for j in range(sum(counts))])
    return species, counts, lat, coords


species, counts, lat_prim, frac_prim = read_prim(PRIM)
lat_super = M @ lat_prim

# 收集超胞原子: s = (f + T) @ M^{-1}, T 为整数向量(行向量约定)
Minv = np.linalg.inv(M.astype(float))

# 按元素分组, 顺序与 POTCAR 一致(S 在前)
order = ["S", "Ta"]
species_super = []
frac_sorted = []
for sp in order:
    start = sum(counts[: species.index(sp)])
    for k in range(counts[species.index(sp)]):
        f = frac_prim[start + k]
        for t in range(-8, 9):
            for u in range(-8, 9):
                for v in range(-8, 9):
                    T = np.array([t, u, v])
                    # 行向量约定: s = (f + T) @ M^{-1}。写成 Minv @ (f + T) 会把
                    # 行向量当列向量, 原子位置会错(√3×√3 时 Ta-Ta 变成 1.94 而非 3.36)
                    s = (f + T) @ Minv
                    if np.all(s >= -1e-9) and np.all(s < 1 - 1e-9):
                        species_super.append(sp)
                        frac_sorted.append(s % 1.0)

assert len(species_super) == NSUP * 3, (len(species_super), NSUP * 3)

n_s = sum(1 for s in species_super if s == "S")
n_ta = len(species_super) - n_s

lines = [f"TaS2-1T supercell M={M.tolist()}", "   1.00000000000000000"]
for row in lat_super:
    lines.append("  " + " ".join(f"{x:.16f}" for x in row))
lines.append(" " + " ".join(order))
lines.append(f"   {n_s} {n_ta}")
lines.append("Direct")
for s, sp in zip(frac_sorted, species_super):
    lines.append("  " + " ".join(f"{x:.16f}" for x in s))

TAG = SPEC.replace(";", "_").replace(",", "_").replace(" ", "")
pos_path = OUT / f"TaS2_sc{TAG}_POSCAR"
pos_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

tm = OUT / f"TaS2_sc{TAG}_TRANSMAT.in"
tm.write_text(
    "\n".join(
        [
            "Read transformation matrix from the TRANSMAT.in file if it exists.",
            "   " + "   ".join(str(x) for x in M[0]) + "   # must be three integers",
            "   " + "   ".join(str(x) for x in M[1]) + "   # must be three integers",
            "   " + "   ".join(str(x) for x in M[2]) + "   # must be three integers",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(f"超胞 {SPEC}: 原子 {len(species_super)} (S {n_s}, Ta {n_ta})")
print("POSCAR   ->", pos_path)
print("TRANSMAT ->", tm)
print("晶格:\n", lat_super)
