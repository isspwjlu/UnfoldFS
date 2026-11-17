"""验证: 自实现投影法展开 vs VASPKIT 282 (平面波法) 的展开结果

在 Γ 点(超胞 k=0)对比两者的"有效能带"能量与权重。
"""
import re
import sys

import numpy as np


def read_ebs(path):
    """兼容两种 EBS.dat 格式:
       A) 旧格式: '#Kpoint-index: N' + N 行 (E, w)  [每个 k 点一块]
       B) 新格式(vaspkit 1.5.x): '# Band-Index: N' + NKPTS 行
    """
    blocks = []
    cur = None
    nkpts = None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = re.search(r"NKPTS\s*&\s*NBANDS:\s*(\d+)\s+(\d+)", s)
            if m:
                nkpts = int(m.group(1))
            if s.startswith("#Kpoint-index"):
                if cur:
                    blocks.append(cur)
                cur = []
            elif s.startswith("# Band-Index"):
                if cur is not None:
                    blocks.append(cur)
                    cur = None
                cur = {"band": int(s.split(":")[1]), "rows": []}
            continue
        p = s.split()
        if cur is None:
            continue
        if isinstance(cur, list):
            cur.append((float(p[1]), float(p[2])))
        else:
            cur["rows"].append((float(p[1]), float(p[2])))
    if isinstance(cur, list) and cur:
        blocks.append(cur)
    return blocks, nkpts


def read_mine(path):
    """-> {kcart: [(E, w), ...]}"""
    d = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        k = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
        d.setdefault(k, []).append((float(p[3]), float(p[4])))
    return d


def main():
    ebs_path, mine_path = sys.argv[1], sys.argv[2]
    blocks, nkpts = read_ebs(ebs_path)
    mine = read_mine(mine_path)
    print(f"[EBS] {len(blocks)} 块 (NKPTS={nkpts})")
    print(f"[mine] {len(mine)} 个 k 点")

    # Γ: 原胞 k = (0,0,0)
    gam = None
    for k, lst in mine.items():
        if all(abs(x) < 1e-6 for x in k):
            gam = lst
            break
    if gam is None:
        print("[!] 未在 my 输出里找到 Γ")
        return
    gam_sorted = sorted(gam, key=lambda t: -t[1])[:20]
    print(f"\n[我的展开 @ Γ] 前 20 个高权重带:")
    for e, w in gam_sorted:
        print(f"   E={e:12.5f}  w={w:.4f}")

    if blocks and isinstance(blocks[0], list):
        print(f"\n[vaspkit 282 @ 第1个k点] {len(blocks[0])} 条:")
        for e, w in blocks[0][:20]:
            print(f"   E={e:12.5f}  w={w:.4f}")
        # 对比能量集合
        e_mine = sorted(e for e, _ in gam_sorted)
        e_ref = sorted(e for e, _ in blocks[0])
        print(f"\n能量差 (我的 - vaspkit, 前10):")
        for a, b in list(zip(e_mine, e_ref))[:10]:
            print(f"   {a:12.5f}  vs  {b:12.5f}   diff={a-b:+.5f}")


if __name__ == "__main__":
    main()
