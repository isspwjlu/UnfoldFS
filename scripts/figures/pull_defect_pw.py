"""把缺陷超胞的平面波结果从服务器拉回本地 .fsdata/。

拉 24×24 与 48×48 两套(48×48 是新的, 文件不存在就跳过)。
"""
from __future__ import annotations

import os
import sys

# ssh_login 是**本机私有**的远程 helper(不在本仓库里, 见 README §1)。
# 用环境变量 UNFOLDFS_SSH_HELPER 指向它所在的目录。
_helper = os.environ.get("UNFOLDFS_SSH_HELPER", "")
if not _helper:
    raise SystemExit("请把 UNFOLDFS_SSH_HELPER 指向包含 ssh_login.py 的目录")
sys.path.insert(0, _helper)
from ssh_login import connect  # noqa: E402

TA = "$WORKDIR"
D = f"{TA}/defect"
LOCAL = r"<repo>\.fsdata"

FILES = [
    # ---- 48x48(新) ----
    (f"{D}/static_defect_bulk_48/out_pw/A(k,E).npz", "defect_vacancy48_pw.npz"),
    (f"{D}/static_perfect_bulk_48/out_pw/A(k,E).npz", "defect_perfect48_pw.npz"),
    ("/tmp/pw_analysis/defect_prim48_pwref.npz", "defect_prim48_pwref.npz"),
    # ---- 24x24(旧, 留作对照) ----
    (f"{D}/static_defect_bulk/out_pw/A(k,E).npz", "defect_vacancy_pw.npz"),
    (f"{D}/static_perfect_bulk/out_pw/A(k,E).npz", "defect_perfect_bulk_pw.npz"),
    ("/tmp/pw_analysis/defect_prim96_pwref.npz", "defect_prim96_pwref.npz"),
    (f"{TA}/prim96_pw/PRIMCELL.vasp", "defect_PRIMCELL_prim96.vasp"),
]


def main() -> None:
    ssh = connect()
    sftp = ssh.open_sftp()
    os.makedirs(LOCAL, exist_ok=True)
    for remote, name in FILES:
        dest = os.path.join(LOCAL, name)
        try:
            sftp.get(remote, dest)
            print("OK   %-34s %9d bytes" % (name, os.path.getsize(dest)))
        except Exception as exc:
            print("FAIL %-34s %s" % (name, exc))
    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
