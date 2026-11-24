"""残差点在哪里: 找出 |A(k)-A(Rk)| 最大的那些 k, 看它们的位置与场值。"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from unfoldfs.fs import (load_spectral_npz, rotation_inverse,  # noqa: E402
                         spectral_at_ef_from_bands)
from unfoldfs.geometry import primitive_symmetry_ops, read_poscar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--op", type=int, default=2)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    ops = primitive_symmetry_ops(prim, 1e-4)
    s = load_spectral_npz(args.npz)
    f = spectral_at_ef_from_bands(s.band_e, s.band_w, s.valid,
                                   s.e_grid, s.e_f, s.sigma)
    n1, n2 = f.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2,
                     np.zeros_like(idx, float)], 1)
    vals = f.ravel()
    mx = float(np.nanmax(f))
    r = np.asarray(ops.rotations)[args.op]
    f2 = frac @ rotation_inverse(r)
    i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
    j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
    img = vals[i2 * n2 + j2]
    diff = np.abs(vals - img) / mx
    order = np.argsort(diff)[::-1]
    print("op=%d  A_max=%.4e  残差最大的 %d 个点:" % (args.op, mx, args.top))
    print("   i1   i2    k_frac            A(k)        A(Rk)       相对差")
    for i in order[:args.top]:
        i1, i2_ = int(i // n2), int(i % n2)
        print("  %4d %4d  (%.4f, %.4f)  %.5e  %.5e  %.3e"
              % (i1, i2_, frac[i, 0], frac[i, 1], vals[i], img[i], diff[i]))
    # 这些点的场值分布: 是脊顶(高)还是低?
    sel = order[:200]
    print("\n残差最大 200 个点的 A(k) 中位=%.4e ; 全体中位=%.4e"
          % (np.median(vals[sel]), np.median(vals)))
    print("残差 > 1%% 的点数: %d / %d (%.3f%%)"
          % (int((diff > 0.01).sum()), diff.size, 100 * (diff > 0.01).mean()))


if __name__ == "__main__":
    main()
