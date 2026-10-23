"""logos.logic.lean_verify_all — Lean machine-checks the whole rule catalogue.

WHY THIS IS SEPARATE FROM THE SELF-TEST
---------------------------------------
`propositional.py --self-test` proves every rule valid with a **Python** truth table. That is
one implementation checking itself against its own notion of entailment. If the truth-table
evaluator has a bug, every rule "passes" and the catalogue is confidently wrong.

This module is the second opinion: every rule and every fallacy is handed to **Lean 4**, whose
kernel shares no code with ours. Two outcomes are possible and both are informative:

  * the rule is PROVED by a tactic or by `rfl` on a literal-substituted Bool encoding
    -> validity is now machine-checked, not merely computed by us
  * the rule is NOT proved
    -> either the rule is invalid (a real defect in the catalogue) or the encoding is wrong
       (a real defect in this file). Either way it must not be silently skipped.

The fallacy direction is the sharper test. A fallacy must be **unprovable**. If Lean ever
proves one, the catalogue contains something that is not a fallacy.

COST AND CACHING
----------------
Each Lean invocation costs 1.5-2.5 s (process start + elaboration). The full catalogue is
~51 checks, so a cold run is roughly 2 minutes. Results are cached in a JSON keyed by
(rule name, formula text, Lean version) so a repeat run is instant and a toolchain upgrade
correctly invalidates the cache.

USAGE
-----
    python tools/logos/logic/lean_verify_all.py             # full run, cached
    python tools/logos/logic/lean_verify_all.py --no-cache
    python tools/logos/logic/lean_verify_all.py --rules-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import lean_bridge, propositional as prop  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "logos_lean_verify.json")


def _key(kind: str, name: str, premises: tuple[str, ...], conclusion: str,
         lv: str) -> str:
    # FORMAT version is in the key so a change to the emitted Lean source invalidates old
    # results. Without it, fixing the encoder still served cached failures.
    raw = f"v2|{kind}|{name}|{'|'.join(premises)}|{conclusion}|{lv}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _load_cache() -> dict:
    try:
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                     # noqa: BLE001
        return {}


def _save_cache(c: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)
    except Exception:                     # noqa: BLE001
        pass


def verify_catalogue(use_cache: bool = True, rules_only: bool = False,
                     verbose: bool = True) -> dict:
    lp = lean_bridge.find_lean()
    lv = lean_bridge.lean_version(lp) if lp else ""
    report: dict = {
        "lean_available": bool(lp),
        "lean_version": lv,
        "lean_path": lp or "",
        "rules": [],
        "replacement": [],
        "fallacies": [],
        "failures": [],
        "t0": time.time(),
    }
    if not lp:
        report["failures"].append("Lean not found: nothing is machine-checked")
        return report

    cache = _load_cache() if use_cache else {}
    hits = misses = 0

    def run(kind: str, name: str, premises: tuple[str, ...], conclusion: str,
            expect_proved: bool) -> dict:
        nonlocal hits, misses
        k = _key(kind, name, premises, conclusion, lv)
        if use_cache and k in cache:
            rec = cache[k]
            hits += 1
        else:
            t0 = time.perf_counter()
            r = lean_bridge.check(list(premises), conclusion)
            rec = {"proved": bool(r.proved), "method": r.method,
                   "ms": round(r.elapsed_ms if r.elapsed_ms else
                               (time.perf_counter() - t0) * 1000, 1),
                   "trustworthy": bool(r.trustworthy),
                   "lean": lv}
            cache[k] = rec
            misses += 1
        ok = (rec["proved"] == expect_proved) and rec["trustworthy"]
        row = {**rec, "name": name, "premises": list(premises),
               "conclusion": conclusion, "expect_proved": expect_proved, "ok": bool(ok)}
        if verbose:
            mark = "PASS" if ok else "FAIL"
            want = "proved" if expect_proved else "unprovable"
            got = "proved" if rec["proved"] else "not proved"
            print(f"  [{mark}] {name:34s} want {want:10s} got {got:10s} "
                  f"via {rec['method']:22s} {rec['ms']:7.0f} ms")
        if not ok:
            report["failures"].append(
                f"{kind}:{name} expected {'proved' if expect_proved else 'unprovable'}, "
                f"Lean says {'proved' if rec['proved'] else 'not proved'} "
                f"(method {rec['method']}, trustworthy {rec['trustworthy']})")
        return row

    for r in prop.RULES_OF_INFERENCE:
        report["rules"].append(run("rule", r.name, r.premises, r.conclusion, True))
    if not rules_only:
        # replacement rules are equivalences: both directions must be provable
        for r in prop.RULES_OF_REPLACEMENT:
            row = run("replacement", r.name + "__fwd", r.premises, r.conclusion, True)
            report["replacement"].append(row)
            row2 = run("replacement", r.name + "__rev", (r.conclusion,), r.premises[0], True)
            report["replacement"].append(row2)

        for f in prop.FALLACIES:
            report["fallacies"].append(
                run("fallacy", f.name, f.premises, f.conclusion, False))

    if use_cache and misses:
        _save_cache(cache)

    report["cache_hits"] = hits
    report["cache_misses"] = misses
    report["elapsed_s"] = round(time.time() - report["t0"], 1)
    return report


def run_self_test() -> bool:
    """A tiny sanity run: one rule Lean must prove, one fallacy Lean must not."""
    print("logos.logic.lean_verify_all - SELF TEST (2 checks)")
    ok = True
    lp = lean_bridge.find_lean()
    good = lp is not None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Lean found at {lp}")

    r = lean_bridge.check(["P -> Q", "P"], "Q")
    good = r.trustworthy and r.proved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] modus ponens machine-proved ({r.method})")

    r2 = lean_bridge.check(["P -> Q", "Q"], "P")
    good = (not r2.proved)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] affirming the consequent machine-unprovable")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lean-verify the whole rule catalogue.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    rep = verify_catalogue(use_cache=not args.no_cache, rules_only=args.rules_only)
    n_lemmas = len(rep["rules"]) + len(rep["replacement"])
    n_fal = len(rep["fallacies"])
    proved = sum(1 for r in rep["rules"] + rep["replacement"] if r["proved"])
    unprovable = sum(1 for r in rep["fallacies"] if not r["proved"])

    print()
    print(f"Lean           : {rep['lean_version'] or 'UNAVAILABLE'}")
    print(f"lemmas         : {proved}/{n_lemmas} machine-proved")
    print(f"fallacies      : {unprovable}/{n_fal} machine-unprovable")
    print(f"cache          : {rep.get('cache_hits', 0)} hits / "
          f"{rep.get('cache_misses', 0)} misses")
    print(f"elapsed        : {rep['elapsed_s']}s")
    if rep["failures"]:
        print(f"FAILURES ({len(rep['failures'])}):")
        for f in rep["failures"]:
            print(f"  - {f}")
    else:
        print("RESULT         : every lemma proved, every fallacy unprovable -- "
              "the catalogue is machine-checked")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"report         : {args.out}")
    return 0 if not rep["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
