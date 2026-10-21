"""serve_eval_matrix.py â€” serve each GGUF on nyx and score it on the frozen eval set.

Sequential on purpose: one llama.cpp container on :8083, model swapped between
cells. Six 350M models at once would fit, but a swapped container makes it
impossible to score against the wrong model.

USAGE
-----
    python tools/tm/serve_eval_matrix.py                 # all GGUFs in the manifest
    python tools/tm/serve_eval_matrix.py --only v6-sft
    python tools/tm/serve_eval_matrix.py --compare       # re-print the table
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

GGUF_DIR = r"B:\datasets\forge\gguf"
OUT_DIR = r"B:\datasets\forge\eval"
NYX = "zcoder@nyx"
NYX_MODELS = "/home/red/Docker/ai-stack/models/llama-cpp/models"
CONTAINER = "llama-eval"
PORT = 8090
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-intel-b9144"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(cmd: str) -> subprocess.CompletedProcess:
    return sh(["ssh", "-o", "BatchMode=yes", NYX, cmd])


def push_model(local: str) -> str:
    """Copy a GGUF to nyx's model dir (needs root, so stage via /tmp + alpine)."""
    name = os.path.basename(local)
    chk = ssh(f"test -f {NYX_MODELS}/{name} && echo yes")
    if "yes" in chk.stdout:
        return name
    print(f"    copying {name} ...", flush=True)
    r = sh(["scp", "-o", "BatchMode=yes", local, f"{NYX}:/tmp/{name}"])
    if r.returncode != 0:
        print(f"    [FAIL] scp: {r.stderr[:160]}")
        return ""
    r = ssh(f"docker run --rm -v {NYX_MODELS}:/m -v /tmp:/s alpine "
            f"sh -c 'cp /s/{name} /m/ && chmod a+r /m/{name}'")
    if r.returncode != 0:
        print(f"    [FAIL] move: {r.stderr[:160]}")
        return ""
    ssh(f"rm -f /tmp/{name}")
    return name


def start(name: str) -> bool:
    ssh(f"docker rm -f {CONTAINER} >/dev/null 2>&1")
    # give the GPU a moment after container teardown
    time.sleep(2)
    r = ssh(f"docker run -d --name {CONTAINER} --device /dev/dri --device /dev/accel "
            f"-v {NYX_MODELS}:/models:ro -p {PORT}:8080 {IMAGE} "
            f"--model /models/{name} --host 0.0.0.0 --port 8080 -ngl 99 --jinja")
    if r.returncode != 0:
        print(f"    [FAIL] run: {r.stderr[:200]}")
        return False
    # wait for /v1/models
    for _ in range(90):
        try:
            with urllib.request.urlopen(
                    f"http://192.168.8.237:{PORT}/v1/models", timeout=5) as resp:
                d = json.loads(resp.read().decode())
            ids = [m.get("id") for m in d.get("data", [])]
            if ids:
                print(f"    served: {ids[0]}", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print("    [FAIL] server did not come up")
    ssh(f"docker logs --tail 15 {CONTAINER} 2>&1")
    return False


def stop():
    ssh(f"docker rm -f {CONTAINER} >/dev/null 2>&1")


def eval_one(name: str, arm: str, seed: str) -> str:
    out = os.path.join(OUT_DIR, f"{arm}-s{seed}.json")
    r = sh([sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_frozen.py"),
            "--endpoint", f"http://192.168.8.237:{PORT}/v1",
            "--model", name, "--arm", arm, "--seed", seed,
            "--out", out, "--max-tokens", "256"])
    tail = (r.stdout or r.stderr).strip().splitlines()
    for line in tail[-3:]:
        print("    " + line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        r = sh([sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_frozen.py"),
                "--compare", os.path.join(OUT_DIR, "*.json")])
        print(r.stdout or r.stderr)
        return 0

    manifest = json.load(open(os.path.join(GGUF_DIR, "manifest.json")))
    cells = sorted(manifest)
    if args.only:
        cells = [c for c in cells if any(o in c for o in args.only)]
    if not cells:
        print("no cells")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    done = []
    for i, cell in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell}")
        info = manifest[cell]
        local = info["gguf"]
        if not os.path.exists(local):
            print("    missing gguf, skip")
            continue
        name = push_model(local)
        if not name:
            continue
        if not start(name):
            continue
        # v6-sft-s42-L512 -> arm=v6-sft, seed=42
        parts = cell.split("-")
        seed = ""
        arm = cell
        for p in parts:
            if p.startswith("s") and p[1:].isdigit():
                seed = p[1:]
                arm = cell.split("-s" + seed)[0]
                break
        try:
            done.append(eval_one(name, arm, seed))
        finally:
            stop()

    print(f"\n[done] {len(done)} evaluations")
    if done:
        r = sh([sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_frozen.py"),
                "--compare", os.path.join(OUT_DIR, "*.json")])
        print(r.stdout or r.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
