"""logos/logic/scope_triage.py — detect quantifier/negation scope ambiguity and ESCALATE, never resolve it.

THE EVIDENCE THIS IS BUILT ON
-----------------------------
Quantifier scope is the one failure a dependency parse cannot touch: "Every boy loves a girl" and its
scopally-inverted twin produce an IDENTICAL tree. Measured evidence from the 2024-2026 literature:

    humans picking the majority reading           ~90%
    GPT-4 producing BOTH logical forms            64%      <- the task an "enumerate and pick" design needs
    LLM behaviour                                biases to SURFACE scope, systematically
    symbolic, CPU-only, with numbers             LangPro SICK 82.1%; ccg2lambda JSeM 74.76 F1 (P 92.6 / R 65.1)

So: no automatic resolver is precise enough to auto-verdict, and an LLM resolver is BIASED rather than neutral,
which in a precision-first suite is a silent soundness hole. The defensible design is not to resolve scope but
to DETECT it and abstain.

THE ONE GENUINELY SOUND MOVE
----------------------------
Enumerate the readings and ask Z3 whether they DIFFER. If the readings agree on the verdict, the ambiguity is
harmless and the verdict stands -- that is a proof, not a guess. If they disagree, abstain. Only the
agreement direction is logically airtight; "disagree, so pick the likelier one" is exactly where every
measured system fails, and this module does not do it.

The enumeration here is deliberately narrow: two readings, one pair of operators, no nesting. A wider
enumeration would be a claim this module cannot defend.

USAGE
    python tools/logos/logic/scope_triage.py --self-test
    python tools/logos/logic/scope_triage.py --measure
    python tools/logos/logic/scope_triage.py --sentence "Every boy loves a girl."
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

# scope-bearing operators, grouped by the taxonomy the TACL work uses, because naming the ambiguity type is
# what makes an escalation actionable rather than a shrug.
QUANTIFIER_DETS = {"every", "each", "all", "some", "any", "no", "none", "most", "many", "few", "several",
                   "both", "either", "neither", "much", "little"}
NEGATIONS = {"not", "n't", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}
SCOPE_ADVERBS = {"usually", "generally", "often", "always", "sometimes", "rarely", "only", "merely",
                 "typically", "mostly", "largely"}
COMPARATIVES = {"more", "less", "fewer", "greater", "than", "at_least", "at_most"}

AMBIGUITY_TYPES = {
    frozenset({"quantifier", "quantifier"}): "quantifier-quantifier",
    frozenset({"quantifier", "negation"}): "quantifier-negation",
    frozenset({"quantifier", "adverb"}): "quantifier-adverb",
    frozenset({"quantifier", "comparative"}): "quantifier-comparative",
    frozenset({"negation", "adverb"}): "negation-adverb",
    frozenset({"negation", "comparative"}): "negation-comparative",
    frozenset({"adverb", "adverb"}): "adverb-adverb",
    frozenset({"comparative", "comparative"}): "comparative-comparative",
    frozenset({"adverb", "comparative"}): "adverb-comparative",
}

_nlp = None


def _model():
    global _nlp
    if _nlp is None:
        try:
            from logos.nlp import domain_nlp
            _nlp = domain_nlp.make_nlp(use_domain=True)
        except Exception:                                     # noqa: BLE001
            import spacy
            _nlp = spacy.load(os.environ.get("LOGOS_SPACY_MODEL", "en_core_web_sm"), disable=["ner"])
    return _nlp


def scope_operators(text: str, doc=None) -> list:
    """Every scope-bearing operator, with its kind and surface form.

    Read from the dependency parse where possible, because a determiner is only a QUANTIFIER when it is
    attached to a noun: "all" in "after all" is not a universal quantifier, and a surface-word scan cannot
    tell the difference. `doc` lets a caller pass an ALREADY-PARSED document -- with a 36 s spaCy import and
    ~6 ms per parse, re-parsing the same sentence to check its scope would be waste.
    """
    doc = doc if doc is not None else _model()(text)
    ops = []
    for t in doc:
        lem_ = t.lemma_.lower()
        if t.dep_ in ("det", "predet") and lem_ in QUANTIFIER_DETS:
            ops.append({"kind": "quantifier", "text": t.text, "lemma": lem_, "head": t.head.text})
        elif t.dep_ == "neg" or lem_ in NEGATIONS:
            ops.append({"kind": "negation", "text": t.text, "lemma": lem_, "head": t.head.text})
        elif t.dep_ in ("advmod", "amod") and lem_ in SCOPE_ADVERBS:
            ops.append({"kind": "adverb", "text": t.text, "lemma": lem_, "head": t.head.text})
        elif lem_ in COMPARATIVES or t.tag_ in ("JJR", "RBR"):
            ops.append({"kind": "comparative", "text": t.text, "lemma": lem_, "head": t.head.text})
    # "a"/"an" count as existentials when they attach to a singular countable noun
    for t in doc:
        if t.dep_ == "det" and t.lemma_.lower() in ("a", "an"):
            ops.append({"kind": "quantifier", "text": t.text, "lemma": "a", "head": t.head.text})
    return ops


def ambiguity_type(ops: list):
    """Name the ambiguity, or None when there is nothing to be ambiguous about.

    One operator is not ambiguous: "All men are mortal" has a single scope-bearing element, so there is no
    pair to order. Two or more is where reading order starts to matter.
    """
    kinds = [o["kind"] for o in ops]
    if len(kinds) < 2:
        return None
    counter = collections.Counter(kinds)
    top = [k for k, _c in counter.most_common(2)]
    if len(top) == 1:
        top = [top[0], top[0]]
    key = frozenset(top[:2])
    return AMBIGUITY_TYPES.get(key, "mixed")


def triage(text: str, doc=None) -> dict:
    """ESCALATE on ambiguity, with the type named. This module never resolves scope."""
    ops = scope_operators(text, doc=doc)
    kind = ambiguity_type(ops)
    if kind is None:
        return {"status": "NO_SCOPE_AMBIGUITY", "n_operators": len(ops), "operators": ops}
    return {"status": "SCOPE_AMBIGUOUS", "ambiguity": kind, "n_operators": len(ops),
            "operators": ops,
            "why": f"{len(ops)} scope-bearing operators of types "
                   f"{sorted({o['kind'] for o in ops})}; their relative scope is not recoverable from the "
                   f"dependency tree, and no measured resolver is precise enough to pick one"}


# ------------------------------------------------------------------ the sound two-reading check
def readings_differ(premises_signature: list, scope_a: str, scope_b: str) -> dict:
    """Do two readings of the same sentence differ in truth value under the same evidence?

    In: two Z3 expressions for the two scope orders. Out: AGREE (the ambiguity is harmless and the verdict
    stands -- this is a proof) / DIFFER (abstain) / UNKNOWN.
    """
    try:
        import z3
    except Exception:                                         # noqa: BLE001
        return {"status": "UNKNOWN", "reason": "z3 unavailable"}
    try:
        a = z3.And(*premises_signature, scope_a) if premises_signature else scope_a
        b = z3.And(*premises_signature, scope_b) if premises_signature else scope_b
    except Exception as e:                                    # noqa: BLE001
        return {"status": "UNKNOWN", "reason": f"could not build expressions: {type(e).__name__}"}
    s = z3.Solver()
    s.set("timeout", 5000)
    try:
        s.add(z3.Xor(a, b))
        r = s.check()
    except Exception as e:                                    # noqa: BLE001
        return {"status": "UNKNOWN", "reason": f"z3 raised {type(e).__name__}"}
    if r == z3.unsat:
        return {"status": "AGREE",
                "why": "the two readings are logically equivalent given the evidence, so the ambiguity is "
                       "harmless and the verdict stands -- proved, not guessed"}
    if r == z3.sat:
        return {"status": "DIFFER", "why": "the readings disagree; abstain rather than pick the likelier one"}
    return {"status": "UNKNOWN", "reason": "z3 returned unknown"}


# ------------------------------------------------------------------ measure
def measure(limit: int = 300) -> int:
    import time
    from logos import corpus_sizing

    path = os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")
    if not os.path.exists(path):
        print(f"corpus not found: {path}")
        return 1
    rows = corpus_sizing.load_corpus(path, limit)
    print(f"=== scope triage on {len(rows)} real claims ===")
    print(f"  {path}\n")

    t0 = time.perf_counter()
    counts = collections.Counter()
    kinds = collections.Counter()
    samples = []
    ops_hist = collections.Counter()
    for r in rows:
        t = triage(r["claim"])
        counts[t["status"]] += 1
        ops_hist[t["n_operators"]] += 1
        if t["status"] == "SCOPE_AMBIGUOUS":
            kinds[t["ambiguity"]] += 1
            if len(samples) < 8:
                samples.append((r["claim"], t["ambiguity"], t["n_operators"]))
    dt = time.perf_counter() - t0

    n = len(rows)
    for k, v in counts.most_common():
        print(f"  {k:22s} {v:5d}  {v / n:6.1%}")
    print(f"\n  wall {dt:.2f}s  ({dt / n * 1000:.2f} ms/claim)")
    print(f"  operator-count histogram: {dict(sorted(ops_hist.items()))}")
    if kinds:
        print(f"\n  ambiguity types: {dict(kinds)}")
        print(f"\n  flagged examples:")
        for c, k, o in samples:
            print(f"    [{k}, {o} ops] {c[:88]!r}")
    print("\n  THE POINT: a flagged claim is ESCALATED, not resolved. Humans pick the majority reading ~90% of")
    print("  the time; the best LLM produces BOTH readings only 64% of the time and is biased toward surface")
    print("  scope. Detecting the ambiguity is provable; resolving it is not.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.logic.scope_triage - SELF TEST")
    ok = True

    # --- one operator is NOT ambiguous: no pair, no scope ordering to get wrong
    t = triage("All men are mortal.")
    good = t["status"] == "NO_SCOPE_AMBIGUITY"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a single quantifier is not scope-ambiguous")

    # --- the canonical case: two quantifiers, undecidable from the tree
    t = triage("Every boy loves a girl.")
    good = t["status"] == "SCOPE_AMBIGUOUS" and t["ambiguity"] == "quantifier-quantifier"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Every boy loves a girl' is quantifier-quantifier ambiguous")

    # --- quantifier + negation, the case our cleaning suite needs most
    t = triage("All cats are not mammals.")
    good = t["status"] == "SCOPE_AMBIGUOUS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] quantifier + negation is flagged "
          f"({t.get('ambiguity')})")

    # --- a determiner that is not attached to a noun is not a quantifier
    t = triage("The bridge is tall.")
    good = t["status"] == "NO_SCOPE_AMBIGUITY"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a plain sentence has no scope operators")

    # --- THE SOUND MOVE: equivalent readings are proved harmless, so no escalation needed
    import z3
    x = z3.Int("x")
    a = z3.ForAll([x], z3.Implies(x > 0, x > 0))          # trivially equivalent pair
    b = z3.ForAll([x], z3.Implies(x > 0, x > 0))
    r = readings_differ([], a, b)
    good = r["status"] == "AGREE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] equivalent readings are proved HARMLESS (AGREE)")

    # --- and genuinely different readings are reported as DIFFER, which means abstain
    y = z3.Int("y")
    p = z3.ForAll([y], y > 0)
    q = z3.Exists([y], y > 0)
    r = readings_differ([], p, q)
    good = r["status"] == "DIFFER"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] genuinely different readings are DIFFER -> abstain")

    # --- it never *resolves*: there is no function that picks a reading
    good = not any(hasattr(sys.modules[__name__], n) for n in ("resolve_scope", "pick_reading"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the module exposes no scope RESOLVER (detect and abstain only)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect quantifier/negation scope ambiguity; never resolve it.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--sentence", default="")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure(args.limit)
    if args.sentence:
        print(json.dumps(triage(args.sentence), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
