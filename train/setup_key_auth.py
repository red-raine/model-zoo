"""setup_key_auth.py — install opencode's public key on remote hosts (password once).

Uses paramiko to log in with the given password and append our public key to
~/.ssh/authorized_keys, so subsequent access is passwordless (key auth).

Usage:
    python setup_key_auth.py --host 192.168.8.152 --user zcoder --password ch4ng3m3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import paramiko

PUBKEY = Path.home() / ".ssh" / "id_ed25519.pub"


def install(host: str, user: str, password: str, port: int = 22) -> int:
    if not PUBKEY.exists():
        print(f"no public key at {PUBKEY}", file=sys.stderr)
        return 1
    key = PUBKEY.read_text(encoding="utf-8").strip()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, port=port, username=user, password=password,
                    timeout=15, allow_agent=False, look_for_keys=False)
    except Exception as exc:
        print(f"[{host}] connect failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # idempotent install of the key
    cmd = (
        'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && '
        'chmod 600 ~/.ssh/authorized_keys && '
        f'grep -qxF {shq(key)} ~/.ssh/authorized_keys || echo {shq(key)} >> ~/.ssh/authorized_keys; '
        'echo INSTALLED; wc -l < ~/.ssh/authorized_keys'
    )
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=20)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    ssh.close()
    print(f"[{host}] {out or err}")
    return 0 if "INSTALLED" in out else 1


def shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="zcoder")
    ap.add_argument("--password", required=True)
    ap.add_argument("--port", type=int, default=22)
    args = ap.parse_args()
    return install(args.host, args.user, args.password, args.port)


if __name__ == "__main__":
    raise SystemExit(main())