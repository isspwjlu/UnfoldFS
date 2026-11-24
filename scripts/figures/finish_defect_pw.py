"""缺陷超胞的平面波路线 —— **本地**部分: 拉回数据 → 作图 → 数字。

服务器端的部分见 ``scripts/figures/make_defect_pw.sh``(服务器上跑)。

用法:  python scripts/figures/finish_defect_pw.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

LOCAL = str(ROOT)


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=LOCAL)


def main() -> int:
    # 1. 拉回
    rc = run([sys.executable, "scripts/figures/pull_defect_pw.py"])
    if rc != 0:
        print("拉取失败")
        return rc
    npz = os.path.join(LOCAL, ".fsdata", "defect_vacancy_pw.npz")
    if not os.path.exists(npz):
        print("缺陷的 npz 还没到(计算可能没完)")
        return 3

    # 2. 作图
    rc = run([sys.executable, "scripts/figures/paper_fig_defect.py",
              "--prim", ".fsdata/defect_PRIMCELL_prim96.vasp",
              "--prim-npz", ".fsdata/defect_prim96_pwref.npz",
              "--perfect-npz", ".fsdata/defect_perfect_bulk_pw.npz",
              "--defect-npz", ".fsdata/defect_vacancy_pw.npz",
              "--kz", "0",
              "--out", "paper_pw/figs/fig8_defect.pdf",
              "--png", "paper_pw/figs/fig8_defect.png"])
    if rc != 0:
        return rc

    # 3. 数字(写正文用)
    out = []
    for tag, npz in (("缺陷 Ta9S16", ".fsdata/defect_vacancy_pw.npz"),
                     ("完美 3x3x1", ".fsdata/defect_perfect_bulk_pw.npz")):
        rc = run([sys.executable, "scripts/analysis/defect_compare.py",
                  "--prim-npz", ".fsdata/defect_prim96_pwref.npz",
                  "--sc-npz", npz,
                  "--prim", ".fsdata/defect_PRIMCELL_prim96.vasp",
                  "--kz", "0", "--label", f"{tag}(平面波) vs 原胞直算"])
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
