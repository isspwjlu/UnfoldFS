import numpy as np
import spglib
from fractions import Fraction

D = "/data/wjlu/work/claude/unfold_fs_demo"


def read_poscar(path):
    lines = [l.rstrip() for l in open(path)]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip()
    idx = 8
    if mode.lower().startswith("s"):
        idx = 9
    cart = mode.lower().startswith("c")
    inv = np.linalg.inv(lat)
    pos, sp = [], []
    for s, n in zip(symbols, counts):
        for _ in range(n):
            p = np.array([float(x) for x in lines[idx].split()[:3]])
            pos.append(p @ inv if cart else p)
            sp.append(s)
            idx += 1
    return lat, np.array(pos), sp


def read_transmat(path):
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        p = line.split()
        if len(p) == 3:
            rows.append([int(float(x)) for x in p])
    return np.array(rows, dtype=int)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


prim_lat, prim_pos, prim_sp = read_poscar(D + "/PRIMCELL.vasp")
sc_lat, sc_pos, sc_sp = read_poscar(D + "/POSCAR")
M = read_transmat(D + "/TRANSMAT.in")
print("原胞原子数", len(prim_sp), "超胞原子数", len(sc_sp), "M", M.tolist())

uniq = sorted(set(prim_sp))
sid = np.array([uniq.index(s) + 1 for s in prim_sp])
cell = (prim_lat, prim_pos, sid)
out = spglib.get_symmetry(cell, symprec=1e-5)
print("get_symmetry 返回项数:", len(out))
rots, trans = out["rotations"], out["translations"]
print("原胞对称操作数:", len(rots))

inv = np.linalg.inv(M.astype(float))
n_r_ok = n_t_ok = n_both = 0
for r, t in zip(rots, trans):
    r_sc = inv @ r.astype(float) @ M.astype(float)
    t_sc = t @ inv
    ok_r = bool(np.allclose(r_sc, np.round(r_sc), atol=1e-6))
    ok_t = bool(np.allclose(t_sc, np.round(t_sc), atol=1e-6))
    n_r_ok += ok_r
    n_t_ok += ok_t
    n_both += (ok_r and ok_t)
print("整数 M^-1 R M 的操作:", n_r_ok, "/", len(rots))
print("整数 t·M^-1 的操作:", n_t_ok, "/", len(rots))
print("同时满足(公共对称集):", n_both, "/", len(rots))

comm = 1
for x in inv.ravel():
    d = Fraction(float(x)).limit_denominator(1000).denominator
    comm = comm * d // gcd(comm, d)
det = int(round(abs(np.linalg.det(M))))
cands = []
span = int(np.abs(M).max()) + 2
for g in np.ndindex(*([2 * span + 1] * 3)):
    gg = np.array(g) - span
    t = gg @ inv
    if np.all(t >= -1e-9) and np.all(t < 1.0 - 1e-9):
        cands.append(gg)
print("comm(M^-1) =", comm, " det(M) =", det, " 候选G数 =", len(cands))
print("候选G:", cands)

for label, cand in (("行约定 M", M), ("列约定 M.T", M.T)):
    ok = np.allclose(cand @ prim_lat, sc_lat, atol=1e-4)
    print(label, "-> A_sc == M_eff @ A_prim ?", ok)

inv_sc = np.linalg.inv(sc_lat)
n_copy = []
for p in prim_pos:
    c = 0
    for q in sc_pos:
        dq = (q - p) @ inv_sc
        t = np.round(dq @ M)
        if np.allclose(dq @ M, t, atol=1e-3):
            c += 1
    n_copy.append(c)
print("每个原胞原子的超胞副本数:", n_copy)
