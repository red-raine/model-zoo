"""logos/truth_kinds.py — classify the KIND of truth in a row, then reuse validated truths SAFELY.

WHY A TAXONOMY COMES BEFORE A VALIDATOR
---------------------------------------
A row can assert several independent kinds of thing:

    arithmetic        4 + 4 = 8                       decidable, exact
    numeric_constant  pi = 3.1415                     decidable to a tolerance
    algebraic         (x+1)^2 = x^2+2x+1              decidable, symbolic
    calculus          d/dx x^2 = 2x                   decidable, symbolic
    linear_algebra    det([[1,2],[3,4]]) = -2         decidable, exact
    unit_conversion   1 mile = 1.609344 km            decidable, table arithmetic
    code_behaviour    "this function returns 6 for 3" decidable ONLY by running it
    entity_attribute  "the bridge is 120 m tall"      NOT decidable locally, needs evidence
    temporal          "it opened in 1987"             needs a dated source
    causal            "X caused Y"                    not decidable, only supported
    definitional      "a prime has two divisors"      true by convention

**The point of classifying first is that it tells you which validator to call and, more importantly, which
claims are NOT worth a model call at all.** Half of these are decidable by computation, and the other half
must never be given a numeric confidence that looks like the first half's.

THE ANTI-POISONING STORE
------------------------
Reusing validated truths is the obvious speed win and the obvious way to poison a corpus: one wrong entry,
reused a thousand times, is worse than a thousand independent errors because it looks consistent. So the
store has a promotion gate with properties that can be tested:

    INDEPENDENT   two validations by the SAME validator are NOT corroboration. Corroboration requires
                  distinct validators or distinct evidence sources. Repeated runs are one witness.
    AGREEMENT     corroborating validators must agree on the verdict, not merely both be non-empty.
    REVERSIBLE    negative feedback demotes immediately, and a truth past its re-validation horizon is
                  served as TENTATIVE rather than PROVEN.
    HONEST        serve() returns a STATUS, so a caller cannot mistake a tentative entry for a proof.

USAGE
    python tools/logos/truth_kinds.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.verify.math_core import (CAPABILITIES, check_derivative, check_determinant,  # noqa: E402
                                    check_identity, check_integral, check_matrix_product,
                                    check_numeric_constant, check_solution,
                                    check_unit_conversion)

# ------------------------------------------------------------------ the taxonomy
KINDS = {
    "arithmetic": {"decidable": True, "needs": "exact arithmetic",
                   "validator": "verify/math_verify.py"},
    "numeric_constant": {"decidable": True, "needs": "exact constant value",
                         "validator": "verify/math_core.check_numeric_constant"},
    "algebraic": {"decidable": True, "needs": "symbolic simplification",
                  "validator": "verify/math_core.check_identity"},
    "calculus": {"decidable": True, "needs": "symbolic differentiation/integration",
                 "validator": "verify/math_core.check_derivative|check_integral"},
    "linear_algebra": {"decidable": True, "needs": "exact matrix arithmetic",
                       "validator": "verify/math_core.check_determinant|check_matrix_product"},
    "unit_conversion": {"decidable": True, "needs": "exact unit factors",
                        "validator": "verify/math_core.check_unit_conversion"},
    "code_behaviour": {"decidable": True, "needs": "EXECUTION with a checked output",
                       "validator": "code_exec"},
    "logical": {"decidable": True, "needs": "a proof calculus",
                "validator": "logic/syllogism|logic/smt"},
    "entity_attribute": {"decidable": False, "needs": "evidence from a source",
                         "validator": "facts/arbiter"},
    "temporal": {"decidable": False, "needs": "a dated source",
                 "validator": "redundancy/archive"},
    "causal": {"decidable": False, "needs": "study design, not computation",
               "validator": "none -- do not score"},
    "definitional": {"decidable": False, "needs": "a convention",
                     "validator": "ontology"},
}

# Patterns are deliberately SHALLOW. Classification decides which validator to CALL; it does not decide the
# verdict, so a wrong classification costs a wasted call, not a wrong answer -- and every decidable kind
# returns UNKNOWN when the string does not actually parse.
_PATTERNS = (
    ("unit_conversion", re.compile(r"[\d.]+\s*[A-Za-z]+\s*(?:=|is)\s*[\d.]+\s*[A-Za-z]+\s*$")),
    ("calculus", re.compile(r"\b(d/d[a-z]|derivative|integral|antiderivative|∫)\b", re.I)),
    ("linear_algebra", re.compile(r"\b(det|determinant|inverse|matrix)\b|\[\[", re.I)),
    ("numeric_constant", re.compile(r"^\s*(pi|e|tau|phi|golden ratio|sqrt\(?2\)?)\s*=", re.I)),
    ("algebraic", re.compile(r"[a-z]\s*(?:\^|\*\*)\s*\d")),
    ("arithmetic", re.compile(r"\d\s*[-+*/]\s*\d|\d+\s*%")),
    ("logical", re.compile(r"\b(therefore|implies|all\s+\w+\s+are|no\s+\w+\s+are)\b", re.I)),
    ("temporal", re.compile(r"\b(19|20)\d{2}\b")),
    ("code_behaviour", re.compile(r"\b(returns?|raises?|outputs?)\b", re.I)),
    ("causal", re.compile(r"\b(caused|because of|leads to|results in)\b", re.I)),
)


def classify(text: str) -> dict:
    """Which KIND of truth does this text assert? Returns the kind plus every pattern that fired."""
    hits = [k for k, pat in _PATTERNS if pat.search(text)]
    if not hits:
        return {"kind": "unknown", "all_hits": [], "decidable": None,
                "note": "no pattern matched; do not guess a kind"}
    # precedence: the most specific decidable kind wins
    for k in ("unit_conversion", "calculus", "linear_algebra", "numeric_constant", "algebraic",
              "arithmetic", "logical", "code_behaviour", "temporal", "causal"):
        if k in hits:
            return {"kind": k, "all_hits": hits, "decidable": KINDS[k]["decidable"],
                    "validator": KINDS[k]["validator"]}
    return {"kind": hits[0], "all_hits": hits, "decidable": KINDS[hits[0]]["decidable"]}


# ------------------------------------------------------------------ the anti-poisoning store
PROVEN, TENTATIVE, MISS = "PROVEN", "TENTATIVE", "MISS"


@dataclass
class Entry:
    key: str
    claim: str
    kind: str
    verdict: str
    validators: dict = field(default_factory=dict)   # validator name -> verdict, distinct only
    first_seen: float = 0.0
    last_validated: float = 0.0
    uses: int = 0
    demerits: int = 0

    @property
    def corroborations(self) -> int:
        return len(self.validators)

    def promoted(self, min_validations: int) -> bool:
        if self.demerits > 0:
            return False
        if self.corroborations < min_validations:
            return False
        return len(set(self.validators.values())) == 1        # all distinct validators AGREED

    def as_dict(self) -> dict:
        return {"claim": self.claim, "kind": self.kind, "verdict": self.verdict,
                "validators": self.validators, "uses": self.uses, "demerits": self.demerits,
                "corroborations": self.corroborations}


class TruthStore:
    """A cache of validated truths with a promotion gate, because a poisoned cache is worse than none."""

    def __init__(self, path: str = "", min_validations: int = 2, ttl_s: float = 60 * 60 * 24 * 30):
        self.path = path
        self.min_validations = max(2, min_validations)       # never allow a single-witness promotion
        self.ttl_s = ttl_s
        self.entries: dict[str, Entry] = {}
        self.served = 0
        self.penalised = 0
        if path and os.path.exists(path):
            self.load()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def key(claim: str, kind: str) -> str:
        norm = " ".join(claim.lower().split())
        return hashlib.sha1(f"{kind}|{norm}".encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ write
    def record(self, claim: str, kind: str, verdict: str, validator: str) -> Entry:
        """Record ONE validation. The validator NAME is what determines independence."""
        k = self.key(claim, kind)
        e = self.entries.get(k)
        now = time.time()
        if e is None:
            e = Entry(key=k, claim=claim, kind=kind, verdict=verdict, first_seen=now)
            self.entries[k] = e
        # a differing verdict from a distinct validator is a CONFLICT, and conflicts are not corroboration
        e.validators[validator] = verdict
        e.last_validated = now
        if len(set(e.validators.values())) > 1:
            e.demerits += 1
        return e

    def penalise(self, claim: str, kind: str, reason: str = "") -> bool:
        """Negative feedback from downstream use. Demotes immediately -- the whole point of an anti-poisoning
        gate is that bad evidence is cheap to remove and expensive to acquire."""
        e = self.entries.get(self.key(claim, kind))
        if e is None:
            return False
        e.demerits += 1
        self.penalised += 1
        return True

    # ------------------------------------------------------------------ read
    def serve(self, claim: str, kind: str) -> dict:
        """Return the stored truth WITH its status. A tentative entry must never be mistaken for a proof."""
        e = self.entries.get(self.key(claim, kind))
        if e is None:
            return {"status": MISS, "verdict": None, "reason": "not in store"}
        if e.demerits > 0:
            return {"status": TENTATIVE, "verdict": e.verdict,
                    "reason": f"demoted: {e.demerits} demerit(s) from conflicting validators or "
                              f"downstream failure", "entry": e.as_dict()}
        if not e.promoted(self.min_validations):
            return {"status": TENTATIVE, "verdict": e.verdict,
                    "reason": f"only {e.corroborations} independent validator(s); "
                              f"{self.min_validations} required, and they must agree",
                    "entry": e.as_dict()}
        if time.time() - e.last_validated > self.ttl_s:
            return {"status": TENTATIVE, "verdict": e.verdict,
                    "reason": "past the re-validation horizon", "entry": e.as_dict()}
        e.uses += 1
        self.served += 1
        return {"status": PROVEN, "verdict": e.verdict, "reason": "corroborated",
                "entry": e.as_dict()}

    # ------------------------------------------------------------------ io
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({k: dict(e.as_dict(), key=e.key, first_seen=e.first_seen,
                               last_validated=e.last_validated)
                       for k, e in self.entries.items()}, fh, indent=1)

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:                                    # noqa: BLE001
            return
        for k, d in raw.items():
            self.entries[k] = Entry(key=k, claim=d["claim"], kind=d["kind"],
                                    verdict=d["verdict"], validators=d.get("validators", {}),
                                    first_seen=d.get("first_seen", 0.0),
                                    last_validated=d.get("last_validated", 0.0),
                                    uses=d.get("uses", 0), demerits=d.get("demerits", 0))


# ------------------------------------------------------------------ dispatch
def validate(claim: str) -> dict:
    """Classify, then call the validator for that kind. Decidable kinds only; the rest are routed."""
    c = classify(claim)
    kind = c["kind"]
    if kind == "numeric_constant":
        v = check_numeric_constant(claim)
    elif kind == "unit_conversion":
        v = check_unit_conversion(claim)
    elif kind == "logical":
        v = None
        return {"kind": kind, "decidable": True, "verdict": "ROUTED",
                "note": "hand to logic/syllogism or logic/smt", "classification": c}
    elif kind in ("entity_attribute", "temporal", "causal", "definitional", "unknown"):
        return {"kind": kind, "decidable": c.get("decidable"), "verdict": "NOT_COMPUTABLE",
                "note": KINDS.get(kind, {}).get("needs", "no validator"),
                "classification": c}
    else:
        return {"kind": kind, "decidable": True, "verdict": "ROUTED",
                "note": f"hand to {KINDS[kind]['validator']}", "classification": c}
    return {"kind": kind, "decidable": True, "verdict": v.status, "detail": v.detail,
            "classification": c}


# ------------------------------------------------------------------ tests
def run_self_test() -> bool:
    print("logos.truth_kinds - SELF TEST")
    ok = True

    # --- classification picks the right kind for representative rows
    cases = (
        ("pi = 3.14159", "numeric_constant"),
        ("1 mile = 1.609344 km", "unit_conversion"),
        ("4 + 4 = 8", "arithmetic"),
        ("(x+1)^2 = x^2 + 2x + 1", "algebraic"),
        ("All men are mortal, therefore Socrates is mortal", "logical"),
        ("the bridge opened in 1987", "temporal"),
        ("this function returns 6", "code_behaviour"),
        ("the storm caused the flood", "causal"),
    )
    bad = [(t, classify(t)["kind"], want) for t, want in cases if classify(t)["kind"] != want]
    good = not bad
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] classifies {len(cases)} representative rows correctly"
          + (f"  MISSES: {bad}" if bad else ""))

    # --- an unmatched row is 'unknown', never a guess
    good = classify("qwerty zxcvb")["kind"] == "unknown"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unrecognised row is 'unknown', not a guessed kind")

    # --- decidable vs not, which is what routes away from a model
    good = (KINDS["arithmetic"]["decidable"] and not KINDS["causal"]["decidable"]
            and not KINDS["entity_attribute"]["decidable"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the taxonomy separates computable kinds from evidence kinds")

    # --- THE ANTI-POISONING GATE. A single witness must never be promoted.
    S = TruthStore(min_validations=2)
    S.record("pi = 3.14159", "numeric_constant", "VERIFIED", "math_core")
    s1 = S.serve("pi = 3.14159", "numeric_constant")
    good = s1["status"] == TENTATIVE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one validator -> TENTATIVE, never PROVEN")

    # --- and the SAME validator twice is still one witness. This is the property that matters: repeated
    #     runs are not corroboration, and a cache that counts them is how poison becomes consensus.
    S.record("pi = 3.14159", "numeric_constant", "VERIFIED", "math_core")
    S.record("pi = 3.14159", "numeric_constant", "VERIFIED", "math_core")
    s2 = S.serve("pi = 3.14159", "numeric_constant")
    good = s2["status"] == TENTATIVE and s2["entry"]["corroborations"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the SAME validator three times is still ONE witness")

    # --- a genuinely independent second validator promotes it
    S.record("pi = 3.14159", "numeric_constant", "VERIFIED", "independent_check")
    s3 = S.serve("pi = 3.14159", "numeric_constant")
    good = s3["status"] == PROVEN
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a distinct validator that AGREES promotes to PROVEN")

    # --- disagreement is not corroboration: it demotes
    S.record("1 mile = 1.6 km", "unit_conversion", "REFUTED", "math_core")
    S.record("1 mile = 1.6 km", "unit_conversion", "VERIFIED", "other_source")
    sd = S.serve("1 mile = 1.6 km", "unit_conversion")
    good = sd["status"] == TENTATIVE and "demoted" in sd["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] conflicting validators DEMOTE rather than average")

    # --- downstream failure demotes an already-proven truth
    S.penalise("pi = 3.14159", "numeric_constant", "downstream wrong")
    s4 = S.serve("pi = 3.14159", "numeric_constant")
    good = s4["status"] == TENTATIVE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] negative feedback demotes a previously PROVEN truth")

    # --- a miss is explicit
    good = S.serve("2 + 2 = 5", "arithmetic")["status"] == MISS
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown claim is MISS, not a silent default")

    # --- min_validations cannot be set below 2 by a caller who wants a shortcut
    good = TruthStore(min_validations=1).min_validations == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] min_validations is floored at 2 -- single-witness promotion "
          f"is not configurable")

    # --- stale entries are served as TENTATIVE, not PROVEN
    S2 = TruthStore(min_validations=2, ttl_s=-1.0)
    S2.record("e = 2.718281828", "numeric_constant", "VERIFIED", "a")
    S2.record("e = 2.718281828", "numeric_constant", "VERIFIED", "b")
    good = S2.serve("e = 2.718281828", "numeric_constant")["status"] == TENTATIVE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an entry past its re-validation horizon is TENTATIVE")

    # --- persistence round-trips
    import tempfile
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "truth.json")
    S3 = TruthStore(path=p, min_validations=2)
    S3.record("4 + 4 = 8", "arithmetic", "VERIFIED", "math_verify")
    S3.record("4 + 4 = 8", "arithmetic", "VERIFIED", "math_core")
    S3.save()
    S4 = TruthStore(path=p, min_validations=2)
    good = S4.serve("4 + 4 = 8", "arithmetic")["status"] == PROVEN
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the store persists and reloads with status intact")

    # --- dispatch actually calls a validator for computable kinds
    r = validate("pi = 3.14159")
    good = r["verdict"] == "VERIFIED" and r["kind"] == "numeric_constant"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dispatch runs the math validator and returns VERIFIED")

    r = validate("the storm caused the flood")
    good = r["verdict"] == "NOT_COMPUTABLE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a causal claim is NOT_COMPUTABLE, not scored")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Truth-kind taxonomy + anti-poisoning truth store.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--classify", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.classify:
        print(json.dumps(validate(args.classify), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
