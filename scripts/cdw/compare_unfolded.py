"""把两份反折叠能带结果放在一起比: 本仓库的 unfoldfs vs 先前的展开结果。

用法:
  python compare_unfolded.py --mine unfolded_band.dat --ref A_EBS.dat \
      --ef 5.465809 --window 0.2 --tol-dist 0.03

两份文件都是 vaspkit 风格的长表, 每行 = 一个 (k, 带) 条目:
  距离(沿路径)  能量(eV)  权重  [带号]
EBS.dat 的带号写在 "# Band-Index   N" 注释里, 数据行只有 3 列。

比较方式:
  * 在公共的 k 点上, 把每个 k 的所有带按能量排序, 逐条配对 => 比带能;
  * 能量零可能整体平移(不同 POTCAR / 不同 E_F 约定)=> 先取平移的中位数;
  * 另比 E_F 附近一个能量窗内的**谱权重**(比逐带权重稳健: 简并带的
    排序配对本来就不可靠)。
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np


def read_long_table(path: str):
    """-> (dist[N], E[N], w[N])  全部 (k, 带) 条目一行一条"""
    D, E, W = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                d, e, w = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue
            D.append(d); E.append(e); W.append(w)
    return np.array(D), np.array(E), np.array(W)


def spectrum_at(D, E, W, d0, tol=0.02):
    """取距离最接近 d0 的一组条目(按能量升序)。

    两条路径的采样间距不一定相同 ⇒ 用"最近点 + 容差"匹配, 不按距离精确相等。
    """
    j = int(np.argmin(np.abs(D - d0)))
    if abs(D[j] - d0) > tol:
        return None
    m = np.abs(D - D[j]) < 1e-6
    e, w = E[m], W[m]
    o = np.argsort(e)
    return e[o], w[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ef", type=float, default=None,
                    help="费米能级(绝对能量, eV); 给了就比较 E_F 附近的谱权重")
    ap.add_argument("--window", type=float, default=0.2,
                    help="E_F 附近谱权重的能量窗半宽 (eV)")
    ap.add_argument("--tol-dist", type=float, default=0.03,
                    help="两条路径上算作同一个 k 的距离容差(2π/Å)")
    args = ap.parse_args()

    Dm, Em, Wm = read_long_table(args.mine)
    Dr, Er, Wr = read_long_table(args.ref)
    print(f"条目数: 我 {len(Dm)}, 参考 {len(Dr)}")
    print(f"路径长度: 我 {Dm.max():.4f}, 参考 {Dr.max():.4f}")

    d_ref = np.unique(np.round(Dr, 5))
    de, dw, am, ar = [], [], [], []
    n = 0
    n_bad = 0
    for d0 in d_ref:
        sp_r = spectrum_at(Dr, Er, Wr, d0)
        sp_m = spectrum_at(Dm, Em, Wm, d0)
        if sp_r is None or sp_m is None:
            continue
        er, wr = sp_r
        em, wm = sp_m
        m = min(len(er), len(em))
        de.append(em[:m] - er[:m])
        dw.append(wm[:m] - wr[:m])
        n += 1
        if args.ef is not None:
            mm = min(len(em), len(er))
            off = float(np.median(em[:mm] - er[:mm]))
            am.append(float(wm[np.abs(em - (args.ef + off)) < args.window].sum()))
            ar.append(float(wr[np.abs(er - args.ef) < args.window].sum()))
            if abs(am[-1] - ar[-1]) > 1e-3:
                n_bad += 1

    de = np.concatenate(de) if de else np.array([])
    dw = np.concatenate(dw) if dw else np.array([])
    print(f"配对的 k 点: {n}")
    if len(de):
        off = float(np.median(de))
        print(f"能量零平移(常数): {off:.5f} eV")
        print(f"能量差(去掉平移): RMS {np.sqrt(((de-off)**2).mean()):.5f} eV, "
              f"中位 {np.median(np.abs(de-off)):.5f} eV, "
              f"最大 {np.abs(de-off).max():.5f} eV")
        print(f"权重差(逐带配对): RMS {np.sqrt((dw**2).mean()):.5f}")
    if am:
        am = np.array(am); ar = np.array(ar)
        am = np.array(am); ar = np.array(ar)
        print(f"E_F={args.ef} ±{args.window} eV 窗内的谱权重: "
              f"中位 我 {np.median(am):.4f} / 参考 {np.median(ar):.4f}; "
              f"两者都>1e-3 的 k 点: {n_bad} / {len(am)}")


if __name__ == "__main__":
    main()
