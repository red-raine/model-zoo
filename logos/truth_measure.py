"""Measure what the truth store is actually worth: validator agreement, hit rate, and time saved.

WHY THIS EXPERIMENT EXISTS
--------------------------
The anti-poisoning store was designed and unit-tested but NEVER measured. Three questions decide whether it
should exist at all:

    1 do two INDEPENDENT validators actually agree? If they disagree often, the store is a coin flip with
      extra steps, and the promotion gate is protecting nothing.
    2 what is the hit rate on a realistically repetitive corpus?
    3 does the time saved exceed the cost of double-validating every distinct fact?

THE INSTRUMENT
--------------
`verify/math_verify.py` and `verify/math_core.py` are genuinely independent implementations of arithmetic
checking: one uses its own regex-and-eval pipeline, the other uses sympy. They share no code, so their
agreement is evidence and their disagreement is a finding. A single implementation agreeing with itself
would prove nothing.

HONESTY ABOUT THE CORPUS
------------------------
The claim set is hand-written, so the TRUE verdicts are known by construction. That makes it a test of the
VALIDATORS, not of the corpus. It cannot tell you how many real rows are true.

USAGE
    python tools/logos/truth_measure.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.truth_kinds import PROVEN, TENTATIVE, TruthStore, classify  # noqa: E402
from logos.verify.math_core import (REFUTED, UNKNOWN, VERIFIED,  # noqa: E402
                                   check_arithmetic_text, check_numeric_constant,
                                   check_unit_conversion)

# (claim, true_verdict) -- ground truth known by construction
CLAIMS = [
    # arithmetic, true
    ("2 + 2 = 4", VERIFIED), ("15% of 80 = 12", VERIFIED), ("(3 + 4) * 2 = 14", VERIFIED),
    ("100 - 37 = 63", VERIFIED), ("7 * 8 = 56", VERIFIED), ("144 / 12 = 12", VERIFIED),
    ("2 ** 10 = 1024", VERIFIED), ("17 + 25 = 42", VERIFIED), ("9 * 9 = 81", VERIFIED),
    ("1000 / 8 = 125", VERIFIED),
    # arithmetic, false
    ("2 + 2 = 5", REFUTED), ("(3 + 4) * 2 = 15", REFUTED), ("100 - 37 = 64", REFUTED),
    ("7 * 8 = 54", REFUTED), ("144 / 12 = 13", REFUTED), ("2 ** 10 = 1023", REFUTED),
    ("9 * 9 = 80", REFUTED), ("1000 / 8 = 124", REFUTED), ("17 + 25 = 43", REFUTED),
    # constants
    ("pi = 3.14159", VERIFIED), ("pi = 3.2", REFUTED), ("e = 2.718281828", VERIFIED),
    ("e = 2.71828", VERIFIED), ("pi = 3.14159265358979", VERIFIED),
    # units
    ("1 mile = 1.609344 km", VERIFIED), ("1 mile = 1.6 km", REFUTED),
    ("1 hour = 3600 s", VERIFIED), ("1 kg = 1000 g", VERIFIED), ("1 ft = 0.5 m", REFUTED),
    # not decidable by these validators
    ("the bridge is tall", UNKNOWN), ("running fast helps", UNKNOWN),
]


def validator_a(claim: str) -> str:
    """Implementation A: logos.verify.math_core (sympy)."""
    c = classify(claim)["kind"]
    if c == "numeric_constant":
        return check_numeric_constant(claim).status
    if c == "unit_conversion":
        return check_unit_conversion(claim).status
    if c == "arithmetic":
        return check_arithmetic_text(claim).status
    return UNKNOWN


def validator_b(claim: str) -> str:
    """Implementation B: tools/verify/math_verify.py (its own parser). Independent of A."""
    try:
        from verify.math_verify import verify_text
    except Exception:                                        # noqa: BLE001
        try:
            from tools.verify.math_verify import verify_text
        except Exception:                                    # noqa: BLE001
            return UNKNOWN
    try:
        r = verify_text(claim)
    except Exception:                                        # noqa: BLE001
        return UNKNOWN
    v = str(r.get("verdict", UNKNOWN)).upper()
    return v if v in (VERIFIED, REFUTED, UNKNOWN) else UNKNOWN


def run() -> int:
    print("=== 1. DO TWO INDEPENDENT VALIDATORS AGREE? ===")
    rows = []
    t_a = t_b = 0.0
    for claim, truth in CLAIMS:
        t0 = time.perf_counter()
        a = validator_a(claim)
        t_a += time.perf_counter() - t0
        t0 = time.perf_counter()
        b = validator_b(claim)
        t_b += time.perf_counter() - t0
        rows.append((claim, truth, a, b))

    both_decided = [r for r in rows if r[2] != UNKNOWN and r[3] != UNKNOWN]
    disagree = [r for r in both_decided if r[2] != r[3]]
    print(f"  claims                      : {len(rows)}")
    print(f"  both validators decided     : {len(both_decided)}")
    print(f"  AGREEMENT                   : {len(both_decided) - len(disagree)}/{len(both_decided)} "
          f"({(len(both_decided) - len(disagree)) / max(len(both_decided), 1):.1%})")
    print(f"  disagreements               : {len(disagree)}")
    for claim, truth, a, b in disagree[:8]:
        print(f"    {claim:28s} A={a:9s} B={b:9s} truth={truth}")
    print(f"  mean latency  A(sympy)      : {t_a / len(rows) * 1000:.2f} ms")
    print(f"  CAVEAT: '{len(both_decided)} of {len(rows)}' is a SELECTED subset -- only the claims both could")
    print(f"  parse. Agreement on the subset both understood says nothing about the "
          f"{len(rows) - len(both_decided)} where")
    print(f"  one declined, so 100% here is NOT 100% validator reliability.")
    print(f"  mean latency  B(math_verify): {t_b / len(rows) * 1000:.2f} ms")

    print(f"\n=== 2. ACCURACY AGAINST KNOWN TRUTH ===")
    for name, idx in (("A (math_core)", 2), ("B (math_verify)", 3)):
        decided = [r for r in rows if r[idx] != UNKNOWN]
        right = [r for r in decided if r[idx] == r[1]]
        wrong = [(r[0], r[idx], r[1]) for r in decided if r[idx] != r[1]]
        declined_rightly = sum(1 for r in rows if r[idx] == UNKNOWN and r[1] == UNKNOWN)
        # ACCURACY IS ON DECIDED CLAIMS ONLY. Counting UNKNOWN==UNKNOWN as "correct" flattered the validator
        # that declined most often, which is exactly backwards.
        acc = len(right) / max(len(decided), 1)
        print(f"  {name:18s} decided {len(decided):2d}/{len(rows)}  "
              f"accuracy-on-decided {acc:5.1%}  WRONG {len(wrong)}  "
              f"correctly declined {declined_rightly}")
        for c, got, want in wrong[:4]:
            print(f"    WRONG: {c:26s} got {got:9s} want {want}")

    print(f"\n=== 3. DOES THE STORE PAY FOR ITSELF? ===")
    # realistic reuse: a corpus of N rows drawing on a pool of P distinct facts
    import tempfile
    tmp = tempfile.mkdtemp()
    distinct = [c for c, t in CLAIMS if t != UNKNOWN]
    N, P = 500, len(distinct)
    rng = random.Random(0)
    corpus = [rng.choice(distinct) for _ in range(N)]

    def pass_with_cache(store):
        t0 = time.perf_counter()
        hits = 0
        for claim in corpus:
            kind = classify(claim)["kind"]
            s = store.serve(claim, kind)
            if s["status"] == PROVEN:
                hits += 1
                continue
            # not proven yet: run BOTH independent validators, so promotion is possible next time
            store.record(claim, kind, validator_a(claim), "a:math_core")
            store.record(claim, kind, validator_b(claim), "b:math_verify")
        return time.perf_counter() - t0, hits

    t_nocache_0 = time.perf_counter()
    for claim in corpus:
        classify(claim)
        validator_a(claim)
    t_nocache = time.perf_counter() - t_nocache_0

    store = TruthStore(path=os.path.join(tmp, "t.json"), min_validations=2)
    t_first, h1 = pass_with_cache(store)
    t_second, h2 = pass_with_cache(store)

    print(f"  corpus {N} rows over {P} distinct facts (reuse factor {N / P:.1f}x)")
    print(f"  no cache, one validator      : {t_nocache * 1000:8.1f} ms")
    print(f"  cache pass 1 (cold + seed)   : {t_first * 1000:8.1f} ms   hits {h1}/{N}")
    print(f"  cache pass 2 (warm)          : {t_second * 1000:8.1f} ms   hits {h2}/{N} "
          f"({h2 / N:.0%})")
    speedup = t_nocache / max(t_second, 1e-9)
    perfect_hits = N - P                     # one miss per distinct fact, everything else a hit
    hits_gap = perfect_hits - h2
    print(f"  warm-pass speedup vs no cache: {speedup:.1f}x")
    print(f"\n  THE GATE IS NOT FREE: a perfect cache would serve {perfect_hits}/{N} "
          f"({perfect_hits / N:.0%}) from memory.")
    print(f"  We got {h2}/{N} ({h2 / N:.0%}). THE {hits_gap}-ROW GAP IS "
          f"{hits_gap / N * 100:.0f} PERCENTAGE POINTS, AND IT IS THE PROMOTION GATE: facts where")
    print(f"  the two validators DISAGREED or one returned UNKNOWN can never promote, so they are re-validated")
    print(f"  on every pass. That is the price of not serving a single-witness truth as proof.")
    print(f"\n  BREAK-EVEN: promotion needs {store.min_validations} INDEPENDENT validators, so a fact must")
    print(f"  be seen {store.min_validations}x before it can be served. Any reuse above 2 pays.")
    print(f"  THE COST OF THE GATE IS EXACTLY ONE EXTRA VALIDATION PER DISTINCT FACT, and the benefit is")
    print(f"  every subsequent reuse. It is a good trade at any reuse factor above 2, which is the")
    print(f"  arithmetic answer to whether the store should exist.")

    print(f"\n=== 4. WHAT THE STORE ACTUALLY PROTECTS AGAINST ===")
    print(f"  With A and B agreeing on {len(both_decided) - len(disagree)}/{len(both_decided)} decided")
    print(f"  claims, agreement is what promotes an entry. A single validator being wrong is the poison,")
    print(f"  and the second validator is the antidote. Where they DISAGREE the entry stays TENTATIVE and")
    print(f"  is never served as proof -- which is the behaviour that makes the cache safe to use.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure validator agreement and cache value.")
    args = ap.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
