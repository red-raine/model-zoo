"""Run every logos self-test. One command, no arguments, no network required.

    python logos/run_all.py            # all modules
    python logos/run_all.py --fast     # skip the modules that touch the network

Each module is an independent program: `python logos/<mod>.py --self-test`. This runner exists
because "27 modules pass" is a claim, and a claim should be reproducible by one command rather than
by a loop someone pasted into a shell.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (module path, touches network)
MODULES: tuple[tuple[str, bool], ...] = (
    ("logic/syllogism.py", False),
    ("logic/propositional.py", False),
    ("logic/badframes.py", False),
    ("logic/lean_bridge.py", True),        # probes for the lean binary
    ("logic/lean_verify_all.py", True),
    ("logic/sorites.py", False),
    ("logic/channel.py", False),
    ("logic/argumentation.py", False),
    ("logic/quantity.py", False),
    ("ontology.py", False),
    ("schema.py", False),
    ("pgm.py", False),
    ("decision.py", False),
    ("aggregate.py", False),
    ("fusion.py", False),
    ("abstain.py", False),
    ("locator.py", False),
    ("lexical.py", False),
    ("datasets.py", False),
    ("harness.py", False),
    ("per_evidence_synth.py", False),
    ("verify/per_evidence.py", False),
    ("honesty.py", False),
    ("facts.py", True),                    # wikidata live checks in some tests
    ("sources.py", False),
    ("redundancy.py", True),               # live archive quorum
    ("router.py", False),
    ("dating.py", False),
    ("arbiter.py", False),
    ("calibrate.py", False),
    ("egress.py", True),                   # live dual-path probe
)


def run(rel: str, timeout: int = 900) -> tuple[str, bool, str, float]:
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        return rel, False, "MISSING", 0.0
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, path, "--self-test"],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.dirname(HERE) or ".")
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return rel, False, f"TIMEOUT after {timeout}s", time.perf_counter() - t0
    dt = time.perf_counter() - t0
    ok = "ALL PASS" in out
    fails = [ln.strip() for ln in out.splitlines() if "[FAIL]" in ln]
    detail = "; ".join(fails[:3]) if fails else ("" if ok else out.strip().splitlines()[-1][:120]
                                                 if out.strip() else "no output")
    return rel, ok, detail, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run all logos self-tests.")
    ap.add_argument("--fast", action="store_true", help="skip network-touching modules")
    ap.add_argument("--only", default="", help="substring filter on module path")
    args = ap.parse_args(argv)

    todo = [(m, net) for m, net in MODULES if not (args.fast and net)]
    if args.only:
        todo = [(m, n) for m, n in todo if args.only.lower() in m.lower()]

    print(f"logos - running {len(todo)} module self-tests"
          f"{' (fast: network modules skipped)' if args.fast else ''}\n")
    failures = []
    t0 = time.perf_counter()
    for rel, net in todo:
        rel, ok, detail, dt = run(rel)
        mark = "PASS" if ok else "FAIL"
        netflag = " [net]" if net else ""
        print(f"  [{mark}] {rel:32s} {dt:6.1f}s{netflag}"
              + (f"  <- {detail}" if not ok else ""))
        if not ok:
            failures.append((rel, detail))

    total = time.perf_counter() - t0
    print(f"\n{len(todo) - len(failures)}/{len(todo)} modules pass in {total:.1f}s")
    if failures:
        print("\nfailures:")
        for rel, detail in failures:
            print(f"  {rel}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
