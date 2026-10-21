"""build_gguf.py — merge LoRA adapters and export GGUFs for the eval matrix.

For every run directory produced by run_ablation.sh:
    adapter -> merged HF model -> f16 GGUF

Then the eval harness serves each GGUF on nyx and scores it against the frozen
48-probe set.

USAGE
-----
    python tools/tm/build_gguf.py --pull            # scp adapters from sol
    python tools/tm/build_gguf.py --build           # merge + convert everything
    python tools/tm/build_gguf.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

BASE_MODEL = "LiquidAI/LFM2.5-350M"
CE_LOCAL = r"B:\datasets\forge\adapters"
CE_GGUF = r"B:\datasets\forge\gguf"
CONVERT = r"E:\vibe_coding\dev\bitnet runner\llama.cpp-hf\convert_hf_to_gguf.py"
SOL_RUNS = "/tmp/runs"
SSH = ["ssh", "-o", "BatchMode=yes", "zcoder@sol"]
SCP = ["scp", "-o", "BatchMode=yes", "-r"]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def list_remote() -> list[str]:
    r = sh(SSH + [f"ls -d {SOL_RUNS}/*/ 2>/dev/null"])
    cells = [os.path.basename(p.rstrip("/")) for p in r.stdout.split() if p.strip()]
    keep = []
    for c in cells:
        chk = sh(SSH + [f"test -f {SOL_RUNS}/{c}/adapter/adapter_model.safetensors && echo yes"])
        if "yes" in chk.stdout:
            keep.append(c)
    return sorted(keep)


def pull(cell: str) -> str:
    dest = os.path.join(CE_LOCAL, cell)
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"  [skip] {cell} already pulled")
        return dest
    os.makedirs(dest, exist_ok=True)

    # containers run as root -> fix perms first so scp can read
    sh(SSH + [f"sudo -n chmod -R a+rX {SOL_RUNS}/{cell} 2>/dev/null || "
              f"docker run --rm -v {SOL_RUNS}/{cell}:/o alpine chmod -R a+rX /o 2>/dev/null"])
    r = sh(SCP + [f"zcoder@sol:{SOL_RUNS}/{cell}/adapter/.",
                  dest])
    if r.returncode != 0:
        print(f"  [FAIL] scp {cell}: {r.stderr[:200]}")
    else:
        print(f"  [ok] pulled {cell} ({len(os.listdir(dest))} files)")
    return dest


def merge(cell: str, adapter_dir: str) -> str:
    merged = os.path.join(CE_LOCAL, cell + "-merged")
    if os.path.isdir(merged) and any(f.endswith(".safetensors") for f in os.listdir(merged)):
        print(f"  [skip] {cell} already merged")
        return merged

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  merging {cell} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                             dtype=torch.float32)
    m = PeftModel.from_pretrained(m, adapter_dir)
    m = m.merge_and_unload()
    m.save_pretrained(merged)
    tok.save_pretrained(merged)
    return merged


def to_gguf(cell: str, merged: str) -> str:
    out = os.path.join(CE_GGUF, f"{cell}.f16.gguf")
    if os.path.exists(out):
        print(f"  [skip] {cell} already converted")
        return out
    os.makedirs(CE_GGUF, exist_ok=True)
    print(f"  converting {cell} -> gguf ...", flush=True)
    r = sh([sys.executable, CONVERT, merged, "--outtype", "f16", "--outfile", out])
    if r.returncode != 0:
        print(f"  [FAIL] convert {cell}: {(r.stderr or r.stdout)[-300:]}")
        return ""
    print(f"  [ok] {os.path.basename(out)} "
          f"{os.path.getsize(out)/1e6:.0f} MB")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)

    cells = list_remote()
    if args.only:
        cells = [c for c in cells if any(o in c for o in args.only)]

    if args.list or not (args.pull or args.build):
        print(f"cells with adapters on sol ({len(cells)}):")
        for c in cells:
            print("  " + c)
        return 0

    os.makedirs(CE_LOCAL, exist_ok=True)
    os.makedirs(CE_GGUF, exist_ok=True)
    manifest = {}

    for c in cells:
        print(f"\n=== {c} ===")
        adir = pull(c) if args.pull or args.build else os.path.join(CE_LOCAL, c)
        if not os.path.isdir(adir) or not os.listdir(adir):
            print("  no adapter, skip")
            continue
        if not args.build:
            continue
        mdir = merge(c, adir)
        g = to_gguf(c, mdir)
        if g:
            manifest[c] = {"gguf": g, "bytes": os.path.getsize(g)}
            rj = os.path.join(adir, "result.json")
            if os.path.exists(rj):
                manifest[c]["train"] = json.load(open(rj))

    if manifest:
        mp = os.path.join(CE_GGUF, "manifest.json")
        with open(mp, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n[build] {len(manifest)} GGUFs -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
