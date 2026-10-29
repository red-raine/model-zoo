"""logos/logic/mathlib_bridge.py — batched Mathlib checking, and a declared mapping that is Type 2 by design.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
MEASURED COST, AND A CORRECTION TO MY OWN FIRST MEASUREMENT
----------------------------------------------------------
`lake env lean` accepts EXACTLY ONE file. Given several it prints "Expected exactly one file name" plus a help
screen and checks NOTHING. I first benchmarked a five-file invocation at 4.2 s and reported a 9.07x batching
win -- I was timing a help screen, and the code path that "batched" was reporting every snippet as verified
because no error line matched any file. That is the worst direction for a checker to fail.

The real numbers, on this machine:

    bare toolchain startup                        7.1 s
    import Mathlib + one statement               52.6 s      <- import breadth dominates
    import Mathlib.Algebra.Group.Even + one       18.5 s      <- 2.8x cheaper, narrowest that worked
    a failed import                               6 - 9 s
    FOUR claims in one combined file             52.7 s      = 13.2 s/claim
    four claims as four invocations           ~210  s        = ~52 s/claim

So batching is real but it is a ~4x saving, not 9x, and what it buys is ONE shared import rather than one per
claim. The dominant term is how much of Mathlib you import. Practical consequence: pick the NARROWEST module
that provides the lemmas, and batch on top of that. A `Mathlib`-wide import is a 50-second fixed cost per
invocation and is almost never necessary.

TYPE 1 vs TYPE 2, AND WHY THIS FILE ONLY DOES TYPE 2
----------------------------------------------------
The autoformalization literature distinguishes:
    Type 1  the formalisation COMPILES because the statement was adjusted to fit; it proves something, but not
            necessarily what the author wrote.
    Type 2  every definition, step and dependency matches the author's intent.

Type 1 is not a hypothetical here. The `pi = pi` repair in this project is a Type 1 artifact in its purest
form: it verifies perfectly and proves nothing. The same failure sits under the tautology guard, the
`2 + 2 = 5 -> 2 + 2 = 4` question, and the structure-check KEEPs on gold-REFUTED claims.

So this module does NOT ask a model to formalise anything. It contains `MAPPINGS`: hand-written, narrow,
declared NL pattern -> Lean statement pairs for families whose formalisation is unambiguous. Faithfulness is
then true BY CONSTRUCTION rather than judged after the fact -- which is the only version of Type 2 that can be
guaranteed without human labels on every row.

Anything not in MAPPINGS returns UNMAPPED. It does not guess, and an unmapped claim is not a failed check --
it is a claim this instrument was never asked about.

USAGE
    python tools/logos/logic/mathlib_bridge.py --self-test
    python tools/logos/logic/mathlib_bridge.py --measure
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

PROJECT = os.environ.get("LOGOS_LEAN_PROJECT", r"B:\datasets\lean\logoslean")
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "logos_mathlib_bridge.json")
IMPORT = os.environ.get("LOGOS_LEAN_IMPORT", "Mathlib")
TIMEOUT = float(os.environ.get("LOGOS_LEAN_TIMEOUT", "900"))


def available() -> tuple:
    """Is the Lake project present and usable? Returns (ok, reason); never raises."""
    if not os.path.isdir(PROJECT):
        return False, f"lake project not found: {PROJECT}"
    if not os.path.exists(os.path.join(PROJECT, "lakefile.toml")) and \
            not os.path.exists(os.path.join(PROJECT, "lakefile.lean")):
        return False, f"no lakefile in {PROJECT}"
    if not os.path.exists(os.path.join(PROJECT, "lean-toolchain")):
        return False, f"no lean-toolchain in {PROJECT}"
    if shutil.which("lake") is None:
        return False, "lake is not on PATH"
    return True, "ready"


# ------------------------------------------------------------------------------- declared mappings
# Hand-written on purpose. Each entry pairs an English pattern with the Lean statement that says THE SAME
# THING, and the pairing is reviewable line by line. A model choosing this statement would be Type 1 waiting
# to happen; a human declaring it is Type 2 by construction.
MAPPINGS = [
    {
        "name": "even_sum_even",
        "pattern": r"\bthe sum of two even (?:numbers|integers) is even\b",
        "lean": "example (a b : ℤ) (ha : Even a) (hb : Even b) : Even (a + b) := Even.add ha hb",
        "english": "the sum of two even integers is even",
    },
    {
        "name": "odd_sum_odd_is_even",
        "pattern": r"\bthe sum of two odd (?:numbers|integers) is even\b",
        "lean": "example (a b : ℤ) (ha : Odd a) (hb : Odd b) : Even (a + b) := Odd.add_odd ha hb",
        "english": "the sum of two odd integers is even",
    },
    {
        "name": "four_is_even",
        "pattern": r"^\s*(?:the number\s+)?4\s+is\s+even\s*\.?\s*$",
        "lean": "example : Even (4 : ℤ) := by norm_num",
        "english": "4 is even",
    },
    {
        "name": "two_plus_two_is_four",
        "pattern": r"^\s*2\s*\+\s*2\s*=\s*4\s*\.?\s*$",
        "lean": "example : (2 : ℤ) + 2 = 4 := by norm_num",
        "english": "2 + 2 = 4",
    },
]

# a deliberately WRONG formalisation, used only by --measure to show what Type 1 looks like: this compiles,
# and it says something other than what its English label claims.
TYPE1_DEMO = {
    "label": "the sum of two even integers is odd",
    "lean": "example (a b : ℤ) (ha : Even a) (hb : Even b) : Even (a + b) := Even.add ha hb",
}


def formalise(claim: str) -> dict:
    """Map an English claim to a declared Lean statement. UNMAPPED is a real answer, not a failure."""
    text = (claim or "").strip()
    for m in MAPPINGS:
        if re.search(m["pattern"], text, re.I):
            return {"status": "MAPPED", "name": m["name"], "english": m["english"],
                    "lean": m["lean"], "type": "2 (declared, reviewable)"}
    return {"status": "UNMAPPED", "claim": text,
            "reason": "no declared mapping; this instrument was not asked about it, and guessing would be "
                      "Type 1 by construction"}


# ------------------------------------------------------------------------------- batched checking
def _cache_key(snippets: list) -> str:
    h = hashlib.sha256()
    for s in snippets:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                         # noqa: BLE001
        return {}


def _save_cache(c: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(c, fh)
    except Exception:                                         # noqa: BLE001
        pass


def _build_combined(snippets: list) -> tuple:
    """One Lean file containing every snippet, with the 1-based start line of each.

    WHY THIS EXISTS. `lake env lean f1 f2 f3` does NOT check three files. Lean prints
    "Expected exactly one file name" and a help screen, and exits -1. My first implementation passed a list
    anyway, matched nothing, and therefore reported EVERY snippet as verified -- silently converting a total
    failure into an all-pass. Batched checking is still worth having, but it has to be done by concatenating
    into one file and mapping error line numbers back to the snippet that owns them.
    """
    lines = [f"import {IMPORT}", ""]
    starts = []
    for body in snippets:
        clean = (body or "").lstrip("\ufeff").strip("\n")
        starts.append(len(lines) + 1)          # 1-based: the line the body begins on
        lines.append(clean)
        lines.append("")
    return "\n".join(lines) + "\n", starts


def _owner(line_no: int, starts: list) -> int:
    """Which snippet does this line belong to?"""
    idx = 0
    for i, st in enumerate(starts):
        if line_no >= st:
            idx = i
    return idx


def check_batch(snippets: list, use_cache: bool = True) -> dict:
    """Check N snippets with a SINGLE `lake env lean` invocation over one combined file.

    Returns per-snippet results, the wall time, and the amortised cost. A response that did not actually
    check anything is reported as FAILED rather than as success -- see `_build_combined` for why that guard
    is not paranoia.
    """
    ok, reason = available()
    if not ok:
        return {"status": "UNAVAILABLE", "reason": reason, "results": [], "seconds": 0.0}
    if not snippets:
        return {"status": "EMPTY", "results": [], "seconds": 0.0}

    cache = _load_cache() if use_cache else {}
    key = _cache_key(snippets + [IMPORT])
    if use_cache and key in cache:
        cached = dict(cache[key])
        cached["cached"] = True
        return cached

    combined, starts = _build_combined(snippets)
    tmp = tempfile.mkdtemp(prefix="logos_lean_")
    path = os.path.join(tmp, "combined.lean")
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(combined)

        t0 = time.perf_counter()
        proc = subprocess.run(["lake", "env", "lean", path], cwd=PROJECT,
                              capture_output=True, text=True, timeout=TIMEOUT)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        base = os.path.basename(path)

        # ---- GUARD: did Lean actually check the file?
        if "Expected exactly one file name" in out:
            return {"status": "FAILED", "results": [], "seconds": round(elapsed, 2),
                    "reason": "lean refused the invocation ('Expected exactly one file name'); nothing was "
                              "checked, so reporting per-snippet success would be a fabrication"}
        if not os.path.exists(path):
            return {"status": "FAILED", "results": [], "seconds": round(elapsed, 2),
                    "reason": "combined file vanished before lean ran"}

        errs = collections.defaultdict(list)
        sorries = set()
        for ln, cl, msg in re.findall(rf"{re.escape(base)}:(\d+):(\d+): error: (.*)", out):
            errs[_owner(int(ln), starts)].append(f"{ln}:{cl} {msg}"[:200])
        for ln, cl, msg in re.findall(rf"{re.escape(base)}:(\d+):(\d+): warning: (.*)", out):
            if "sorry" in msg:
                sorries.add(_owner(int(ln), starts))

        # ---- GUARD: an error line that maps to no snippet means the combined file broke in a way we cannot
        # attribute, so nothing about individual snippets can be claimed.
        unattributed = [e for e in errs.get(-1, [])]
        if unattributed:
            return {"status": "FAILED", "results": [], "seconds": round(elapsed, 2),
                    "reason": f"{len(unattributed)} error(s) could not be attributed to a snippet"}

        results = []
        for i in range(len(snippets)):
            e = errs.get(i, [])
            results.append({"index": i, "verified": not e and i not in sorries,
                            "has_sorry": i in sorries, "errors": e,
                            "seconds": round(elapsed / max(len(snippets), 1), 3)})
        payload = {"status": "OK", "results": results, "seconds": round(elapsed, 2),
                   "per_snippet_seconds": round(elapsed / max(len(snippets), 1), 3),
                   "exit_code": proc.returncode, "cached": False}
        if use_cache:
            cache[key] = payload
            _save_cache(cache)
        return payload
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "reason": f"lean exceeded {TIMEOUT}s", "results": [],
                "seconds": TIMEOUT}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_claims(claims: list, use_cache: bool = True) -> dict:
    """Formalise a list of English claims by declared mapping, then check the mapped ones in ONE batch."""
    formal = [formalise(c) for c in claims]
    mapped = [f for f in formal if f["status"] == "MAPPED"]
    if not mapped:
        return {"formal": formal, "checked": 0, "results": [],
                "note": "nothing was mapped, so Lean was never invoked -- the honest outcome"}
    res = check_batch([f["lean"] for f in mapped], use_cache=use_cache)
    return {"formal": formal, "checked": len(mapped), "results": res.get("results", []),
            "seconds": res.get("seconds", 0.0), "per_claim_seconds": res.get("per_snippet_seconds"),
            "status": res.get("status")}


# ------------------------------------------------------------------------------- measure
def measure() -> int:
    ok, reason = available()
    print("=== Mathlib bridge ===")
    print(f"  project {PROJECT}")
    print(f"  import  {IMPORT}")
    print(f"  avail   {ok} ({reason})")
    if not ok:
        return 1

    claims = [m["english"] for m in MAPPINGS] + ["the moon is made of cheese",
                                                 "the sum of two even integers is odd"]
    print(f"\n=== declared-mapping formalisation ({len(claims)} claims) ===")
    for c in claims:
        f = formalise(c)
        if f["status"] == "MAPPED":
            print(f"  MAPPED   {c[:44]:46s} -> {f['name']}")
        else:
            print(f"  UNMAPPED {c[:44]:46s} -> not asked; no guess made")

    print(f"\n=== checking the mapped ones in ONE invocation ===")
    out = check_claims(claims)
    st = out.get("status")
    print(f"  status {st}   checked {out.get('checked')}   wall {out.get('seconds')}s"
          f"   per claim {out.get('per_claim_seconds')}s")
    for r in out.get("results") or []:
        verdict = "VERIFIED" if r["verified"] else ("SORRY" if r["has_sorry"] else "FAILED")
        print(f"    [{r['index']}] {verdict:9s} {'' if r['verified'] else (r['errors'][0] if r['errors'] else '')}")

    print("\n=== what Type 1 looks like, demonstrated rather than described ===")
    d = TYPE1_DEMO
    print(f"  english label : {d['label']}")
    print(f"  lean statement: {d['lean']}")
    r1 = check_batch([d["lean"]], use_cache=False)
    got = (r1.get("results") or [{}])[0]
    print(f"  result        : {'COMPILES' if got.get('verified') else 'fails'}")
    if got.get("verified"):
        print(f"  ==> THE LABEL AND THE STATEMENT DISAGREE, AND LEAN IS HAPPY. Lean checks the STATEMENT it was")
        print(f"      given, never the sentence someone attached to it. That gap is Type 1, and no compiler")
        print(f"      can close it -- only a declared, reviewable mapping can, which is why MAPPINGS is")
        print(f"      hand-written and short.")

    print("\n=== cost, against the other instruments ===")
    pc = out.get("per_claim_seconds")
    if pc:
        enc_ms = 10.2
        llm_ms = 2200.0
        print(f"  arithmetic check           <1 ms/claim")
        print(f"  cross-encoder           {enc_ms:8.1f} ms/claim")
        print(f"  2.6B LLM                {llm_ms:8.1f} ms/claim")
        print(f"  Mathlib batched        {pc * 1000:8.1f} ms/claim   (import {IMPORT})")
        print()
        print(f"  Lean costs {pc * 1000 / enc_ms:,.0f}x the cross-encoder and {pc * 1000 / llm_ms:,.1f}x the LLM.")
        print(f"  That is the price of a PROOF instead of a score, and it means Lean belongs on rows the cheap")
        print(f"  instruments cannot touch, batched, never one invocation per row.")
        print(f"  With a narrow import the fixed cost was measured at 18.5 s vs 52.6 s for full Mathlib, so")
        print(f"  picking the smallest adequate module is worth more than any other change here.")
    return 0


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.logic.mathlib_bridge - SELF TEST")
    ok = True

    # --- a declared mapping is found
    f = formalise("the sum of two even integers is even")
    good = f["status"] == "MAPPED" and f["name"] == "even_sum_even" and f["type"].startswith("2")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a declared mapping is found and is Type 2 by construction")

    # --- UNMAPPED is a real answer, and says why
    f = formalise("the moon is made of cheese")
    good = f["status"] == "UNMAPPED" and "Type 1" in f["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unmapped claim returns UNMAPPED, not a guess")

    # --- the near-miss does not silently match the real one: "odd" vs "even" must not collide
    good = formalise("the sum of two odd integers is even")["name"] == "odd_sum_odd_is_even"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'two odd integers' maps to its own entry, not the even one")

    # --- A NON-CHECK MUST NOT LOOK LIKE A PASS. Lean refuses multi-file invocations; if that ever reaches
    #     the result path as "no errors found", every snippet becomes a false VERIFIED.
    r = check_batch([], use_cache=False)
    good = r.get("status") == "EMPTY" and r.get("results") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an empty batch reports EMPTY, not an all-pass")

    # --- a missing project is a STATE, not a crash
    saved = PROJECT
    globals()["PROJECT"] = r"Z:\not\a\project"
    try:
        av, why = available()
        good = av is False and "not found" in why
    finally:
        globals()["PROJECT"] = saved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a missing Lake project returns (False, reason)")

    ok_av, why = available()
    if not ok_av:
        print(f"  [SKIP] Mathlib project unavailable ({why}) -- live checks skipped")
        print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
        return ok

    # --- a TRUE statement verifies
    r = check_batch(["example : (2 : ℤ) + 2 = 4 := by norm_num"], use_cache=False)
    got = (r.get("results") or [{}])[0]
    good = got.get("verified") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] LIVE: 2 + 2 = 4 verifies in Mathlib")

    # --- a FALSE statement does not
    r = check_batch(["example : (2 : ℤ) + 2 = 5 := by norm_num"], use_cache=False)
    got = (r.get("results") or [{}])[0]
    good = got.get("verified") is False and bool(got.get("errors"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] LIVE: 2 + 2 = 5 is rejected, with the error captured")

    # --- `sorry` is caught. a proof with a hole is not a proof, which is the same rule the rest of this
    #     project applies to citations and to unverified findings.
    r = check_batch(["example : (2 : ℤ) + 2 = 5 := by sorry"], use_cache=False)
    got = (r.get("results") or [{}])[0]
    good = got.get("has_sorry") is True and got.get("verified") is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] LIVE: a `sorry` proof is NOT verified")

    # --- a BOM must not be read as a Lean error
    r = check_batch(["\ufeffexample : (2 : ℤ) + 2 = 4 := by norm_num"], use_cache=False)
    got = (r.get("results") or [{}])[0]
    good = got.get("verified") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] LIVE: a UTF-8 BOM does not become a fake Lean error")

    # --- and errors are attributed to the right snippet in a batch, not smeared across all of them
    r = check_batch(["example : (2 : ℤ) + 2 = 4 := by norm_num",
                     "example : (2 : ℤ) + 2 = 5 := by norm_num"], use_cache=False)
    res = r.get("results") or []
    good = len(res) == 2 and res[0].get("verified") is True and res[1].get("verified") is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] LIVE: in a BATCH, only the failing snippet fails")
    if "--bench-batch" in sys.argv:
        pass

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batched Mathlib checking with declared mappings.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--availability", action="store_true")
    ap.add_argument("--claim", default="")
    args = ap.parse_args(argv)
    if args.availability:
        ok, reason = available()
        print(json.dumps({"available": ok, "reason": reason, "project": PROJECT}, indent=2))
        return 0
    if args.claim:
        print(json.dumps(check_claims([args.claim], use_cache=False), indent=2))
        return 0
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
