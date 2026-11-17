"""Remote shell helper for the master server (210.73.16.15).

Avoids the quoting/mangling problems of passing long shell snippets through
`python ssh_login.py "..."` (Git Bash expands $VAR before Python sees it).

Usage:
    python rsh.py "command"              # run one command
    python rsh.py < script.sh            # read command from stdin
    python rsh.py --upload LOCAL REMOTE  # sftp upload
    python rsh.py --download REMOTE LOCAL
    python rsh.py --script LOCAL.sh      # upload to /tmp then `bash` it
"""
from __future__ import annotations

import os
import posixpath
import sys

sys.path.insert(0, r"C:\Users\42033\.claude")
from ssh_login import connect  # noqa: E402

REMOTE_TMP = "/tmp/_rsh"


def _sftp(ssh):
    return ssh.open_sftp()


def upload(ssh, local: str, remote: str) -> None:
    parent = posixpath.dirname(remote)
    if parent:
        run(ssh, f"mkdir -p '{parent}'")
    with _sftp(ssh) as sftp:
        sftp.put(local, remote)
    print(f"[upload] {local} -> {remote}")


def download(ssh, remote: str, local: str) -> None:
    with _sftp(ssh) as sftp:
        sftp.get(remote, local)
    print(f"[download] {remote} -> {local}")


def run(ssh, cmd: str, timeout: int = 600) -> str:
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if err.strip():
        out += "\n[stderr]\n" + err
    return out


def main(argv: list[str]) -> int:
    ssh = connect()
    try:
        if not argv:
            data = sys.stdin.read()
            print(run(ssh, data).rstrip())
            return 0
        if argv[0] == "--upload":
            upload(ssh, argv[1], argv[2])
            return 0
        if argv[0] == "--download":
            download(ssh, argv[1], argv[2])
            return 0
        if argv[0] == "--script":
            local = argv[1]
            remote = posixpath.join(REMOTE_TMP, os.path.basename(local))
            upload(ssh, local, remote)
            print(run(ssh, f"bash {remote}").rstrip())
            return 0
        print(run(ssh, " ".join(argv)).rstrip())
        return 0
    finally:
        ssh.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
