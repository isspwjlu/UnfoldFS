"""√3×√3 CDW 的三聚化模型(径向收缩): 3 个金属原子沿径向向三角形质心收缩。

为什么不用 scripts/cdw/build_cdw_tas2.py 的 F3m 位移场:
那个场用 Re[û exp(i M_j·r)], 而 √3 的 M 点星 {M1,M2,M3} **不对 -M 闭合**
(M1+M2+M3=0 但 -M1 不在星里) ⇒ 取实部得到的场**不是 C3 协变的**
(实测生成的结构空间群 = P1, 连 C3 都没有) —— 那个脚本有 bug。

本脚本的构造是显式 C3 协变的: 三个金属原子沿 (质心 - 原子) 方向平移同样的距离
⇒ 三角收缩/扩张 ⇒ 空间群 C3v(6 个操作) ⇒ 破坏原胞 D3d 的 12 个操作中的 6 个
⇒ 与论文 §VI.C 的口径一致(CDW 态自己就破坏 12 个中的 6 个)。
"""
import argparse
import io

import numpy as np

from unfoldfs.geometry import read_poscar


def build(src: str, dst: str, metal: str, shrink: float) -> None:
    st = read_poscar(src)
    A = np.asarray(st.lattice)
    cart = np.asarray(st.positions) @ A
    sp = np.asarray(st.species)
    idx = np.where(sp == metal)[0]
    if len(idx) != 3:
        raise SystemExit("需要恰好 3 个 %s 原子, 实际 %d" % (metal, len(idx)))
    c = cart[idx].mean(axis=0)
    # 每个金属原子沿径向移动: 位移量 = shrink × (到质心的距离)
    vec = c - cart[idx]
    disp = vec * shrink
    newcart = cart.copy()
    newcart[idx] += disp
    frac = newcart @ np.linalg.inv(A)
    groups, order = [], []
    for s in st.species:
        if s not in order:
            order.append(s)
    for s in order:
        groups.append([s, int((sp == s).sum())])
    lines = ["%s trimerized CDW (radial shrink %.3f)" % (metal, shrink), "1.0"]
    for r in A:
        lines.append("  " + "  ".join("%20.12f" % x for x in r))
    lines.append("   " + " ".join(g[0] for g in groups))
    lines.append("    " + "  ".join(str(g[1]) for g in groups))
    lines.append("Direct")
    for p in frac:
        lines.append("  " + "  ".join("%20.12f" % x for x in p))
    io.open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    d = [np.linalg.norm(cart[idx[i]] - cart[idx[j]])
         for i in range(3) for j in range(i + 1, 3)]
    d2 = [np.linalg.norm(newcart[idx[i]] - newcart[idx[j]])
          for i in range(3) for j in range(i + 1, 3)]
    print("wrote", dst)
    print("  %s-%s 键长: %s -> %s"
          % (metal, metal, np.round(d, 4), np.round(d2, 4)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metal", default="Ta")
    ap.add_argument("--shrink", type=float, default=0.05,
                    help="径向收缩比例(0.05 = 向质心移动 5%% 的距离)")
    a = ap.parse_args()
    build(a.poscar, a.out, a.metal, a.shrink)
