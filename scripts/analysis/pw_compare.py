"""平面波路线 vs 轨道投影路线: 同一超胞、同一代码, 只换投影基的对照。

**核心实验**: 反折叠的协变性缺陷到底属于"反折叠"本身, 还是属于
"轨道投影"这个近似? 做法是**受控对照** —— 同一个超胞计算、同一套网格、
同一段下游代码(铺网格 / 展宽 / 取脊顶 / 对称化), 只有**权重**的来源不同:

* 轨道投影: PROCAR 的分波系数(Bloch-character 投影);
* 平面波  : WAVECAR 的平面波系数(Popescu--Zunger 的严格谱权重)。

输出的量:

1. 逐操作的 δ(R) = max_k |A(k,E_F) - A(Rk,E_F)| / max_k A(k,E_F)
   —— 并按"该操作是否保持超胞晶格"(R_sc = M^-T R M^T ∈ Z³)分类:
   不保持 ⇒ 折叠段缺陷(任何展开方法都有); 保持 ⇒ 只可能是权重构造段的问题。
2. 与**直接原胞计算**的谱函数对照: Pearson 相关与相对 L2 差。
   平面波路线的直接参考用**权重 = 1** 的纯谱函数(原胞自己的带);
   轨道投影路线的直接参考用它自己的定义(逐轨道投影模方之和)。
3. 单 G 主导度(态被摊到几个候选上)。

用法:
    python scripts/analysis/pw_compare.py \
        --prim  .fsdata/TaS2_PRIMCELL.vasp \
        --pw-npz   .fsdata/tas2_fs96_sc3x1x1_pw.npz \
        --orb-npz  .fsdata/tas2_fs96_sc3x1x1.npz \
        --direct-npz .fsdata/tas2_fs96_prim.npz \
        --matrix "3 0 0;0 1 0;0 0 1" --label "1T-TaS2 3x1x1" --out-json out.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from unfoldfs.fs import (load_spectral_npz, spectral_at_ef_from_bands,  # noqa: E402
                         rotation_inverse)
from unfoldfs.geometry import primitive_symmetry_ops, read_poscar  # noqa: E402


def parse_matrix(text: str) -> np.ndarray:
    rows = []
    for chunk in text.replace("\n", ";").split(";"):
        vals = chunk.split()
        if vals:
            rows.append([float(x) for x in vals])
    return np.array(rows, dtype=float)


def per_op_delta(field: np.ndarray, ops) -> np.ndarray:
    """每个操作的 δ(R) = max_k |A(k) - A(Rk)| / max_k A(k)。"""
    n1, n2 = field.shape
    idx = np.arange(n1 * n2)
    frac = np.stack([(idx // n2) / n1, (idx % n2) / n2, np.zeros_like(idx, float)], 1)
    vals = field.ravel()
    mx = float(np.nanmax(field))
    out = []
    for r in np.asarray(ops.rotations):
        f2 = frac @ rotation_inverse(r)
        i2 = np.rint(f2[:, 0] * n1).astype(int) % n1
        j2 = np.rint(f2[:, 1] * n2).astype(int) % n2
        out.append(float(np.abs(vals - vals[i2 * n2 + j2]).max() / mx))
    return np.array(out)


def preserves_supercell_lattice(ops, M: np.ndarray) -> np.ndarray:
    """R_sc = M^-T R M^T 为整数矩阵 <=> R 保持超胞晶格。"""
    minv_t = np.linalg.inv(M).T
    flags = []
    for r in np.asarray(ops.rotations):
        r_sc = minv_t @ np.asarray(r, float) @ M.T
        flags.append(bool(np.allclose(r_sc, np.round(r_sc), atol=1e-6)))
    return np.array(flags)


def field_from_npz(path: str) -> np.ndarray:
    """npz -> A(k,E_F)(n1, n2); 用**逐带**数据重算 ⇒ 得到未对称化的场。"""
    s = load_spectral_npz(path)
    return spectral_at_ef_from_bands(s.band_e, s.band_w, s.valid,
                                     s.e_grid, s.e_f, s.sigma)


def compare_to_direct(field: np.ndarray, direct: np.ndarray) -> dict:
    """与直接原胞计算的谱函数对照: Pearson 相关与相对 L2 差。"""
    a = field.ravel()
    b = direct.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.std() < 1e-30 or b.std() < 1e-30:
        return {"pearson": float("nan"), "rel_l2": float("nan")}
    pear = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {"pearson": pear, "rel_l2": rel}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prim", required=True)
    ap.add_argument("--pw-npz", default=None, help="平面波路线的 A(k,E).npz")
    ap.add_argument("--orb-npz", default=None, help="轨道投影路线的 A(k,E).npz")
    ap.add_argument("--direct-npz", default=None,
                    help="直接原胞计算的 A(k,E).npz(平面波路线的参考: 权重=1)")
    ap.add_argument("--direct-npz-orb", default=None,
                    help="轨道投影路线的直接参考(权重定义不同, 要各自配)")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    prim = read_poscar(args.prim)
    ops = primitive_symmetry_ops(prim, 1e-4)
    M = parse_matrix(args.matrix)
    keep = preserves_supercell_lattice(ops, M)

    res = {"label": args.label, "n_ops": int(len(keep)),
           "n_keep": int(keep.sum()), "det": int(round(abs(np.linalg.det(M))))}

    fields = {}
    for tag, path in (("pw", args.pw_npz), ("orb", args.orb_npz),
                      ("direct", args.direct_npz),
                      ("direct_orb", args.direct_npz_orb)):
        if path is None:
            continue
        fields[tag] = field_from_npz(path)

    for tag in ("pw", "orb", "direct"):
        if tag not in fields:
            continue
        f = fields[tag]
        d = per_op_delta(f, ops)
        res[tag] = {"delta": d.tolist(),
                    "delta_keep_max": float(d[keep].max()) if keep.any() else None,
                    "delta_keep_mean": float(d[keep].mean()) if keep.any() else None,
                    "delta_other_max": float(d[~keep].max()) if (~keep).any() else None,
                    "delta_max": float(d.max()),
                    "field_max": float(np.nanmax(f))}
        if tag == "pw" and "direct" in fields:
            res[tag].update(compare_to_direct(f, fields["direct"]))
        if tag == "orb" and "direct_orb" in fields:
            res[tag].update(compare_to_direct(f, fields["direct_orb"]))
        if tag == "direct" and "direct_orb" in fields:
            res[tag].update(compare_to_direct(f, fields["direct_orb"]))

    print("### %s" % (args.label or ""))
    print("    原胞点群 %d 个操作, 其中保持超胞晶格的 %d 个; det(M)=%d"
          % (res["n_ops"], res["n_keep"], res["det"]))
    hdr = " op  " + "".join("%-22s" % t for t in ("平面波 delta", "轨道投影 delta"))
    print(hdr)
    for i in range(res["n_ops"]):
        row = "  %-3d" % i
        for tag in ("pw", "orb"):
            row += "%-22.6e" % res[tag]["delta"][i] if tag in res else "%-22s" % "-"
        row += "   " + ("保持" if keep[i] else "不保持")
        print(row)
    for tag in ("pw", "orb", "direct"):
        if tag not in res:
            continue
        r = res[tag]
        print("  %-6s: 保持超胞晶格的操作 delta max=%.4e ; 其余 max=%.4e ; 全局 max=%.4e"
              % (tag, r["delta_keep_max"] or 0.0, r["delta_other_max"] or 0.0,
                 r["delta_max"]))
        if "pearson" in r:
            print("           与直接原胞计算: Pearson=%.6f  相对 L2 差=%.6f"
                  % (r["pearson"], r["rel_l2"]))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("    已写出 %s" % args.out_json)


if __name__ == "__main__":
    main()
