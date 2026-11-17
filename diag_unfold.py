"""诊断: 正确的展开公式变体 —— 直接对超胞原子求和
w(k') = (1/N_atoms) |sum_j sum_lm c_{j,lm} exp(i (k' + s0 K).R_j)|^2
检验: (a) 单 G 主导度 max/sum  (b) 归一化 sum_G w / sum|p|^2  (应 ~1)
"""
import sys

import numpy as np

sys.path.insert(0, '/data/wjlu/work/claude/unfold_fs_demo')
import unfold_procar as U

d = '/data/wjlu/work/claude/unfold_fs_demo/lo12'
lat_sc, _, pos_sc = U.read_poscar(f'{d}/POSCAR')
lat_pr, _, pos_pr = U.read_poscar(f'{d}/PRIMCELL.vasp')
M = U.read_transmat(f'{d}/TRANSMAT.in')
Minv = np.linalg.inv(M)
b_pr, b_sc = U.reciprocal(lat_pr), U.reciprocal(lat_sc)
cart_sc = pos_sc @ lat_sc
N = len(pos_sc)

kpts, ener = U.read_eigenval(f'{d}/EIGENVAL')
cplx, w2 = U.read_procar(f'{d}/PROCAR')
span = 4
Gset = [np.array(g) - span for g in np.ndindex(*([3 * (span + 1) - 1] * 3))
        if np.all((Minv @ (np.array(g) - span)) >= -1e-9)
        and np.all((Minv @ (np.array(g) - span)) < 1 - 1e-9)]

for ik in [0, 1]:
    kc = kpts[ik] @ b_sc
    print(f"\n######## k 点 {ik} 分数 {kpts[ik]} ########")
    for s0 in (0.0, +1.0, -1.0):
        for s2 in (+1.0, -1.0):
            ratios, peaks = [], []
            for ib in [0, 5, 20, 40, 60, 80]:
                sw = 0.0
                ws = []
                for G in Gset:
                    kp = (Minv @ (kpts[ik] + G)) @ b_pr
                    ph = np.exp(1j * s2 * (kp @ cart_sc.T) + 1j * s0 * (kc @ cart_sc.T))
                    w = float(np.abs(np.sum(cplx[ik, ib] * ph[:, None])) ** 2) / N
                    ws.append(w)
                ws = np.array(ws)
                tw = float(w2[ik, ib].sum())
                sw = ws.sum()
                if tw > 1e-6:
                    ratios.append(sw / tw)
                peaks.append(ws.max() / max(sw, 1e-30))
            print(f"  s0={s0:+.0f} s2={s2:+.0f}: 归一化 mean={np.mean(ratios):.3f} "
                  f"min={np.min(ratios):.3f} max={np.max(ratios):.3f} | "
                  f"峰锐度 mean={np.mean(peaks):.3f} min={np.min(peaks):.3f}")
