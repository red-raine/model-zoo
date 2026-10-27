"""logos/logic/smt.py — deterministic consistency checking with Z3. No LLM, no truth required.

WHERE THIS SITS
---------------
Categorical syllogism validity is already solved in `logic/syllogism.py` (24/24 forms over 256) and does
not need a solver. What that module cannot do is answer: **given a SET of extracted claims, do any of them
contradict each other?** That is the Semantic router's question in `triage.py`, and it needs no access to
the world — which is exactly why it is cheap enough to run on everything.

    claims  ->  encoded fragment  ->  SAT solver  ->  CONSISTENT / INCONSISTENT / UNKNOWN
                                                       + WHICH claims conflict

The last part is the real value. Z3's `unsat_core` names the contradictory subset, so a cleaner gets
"these two sentences cannot both be true" rather than a score.

THE SOUNDNESS ASYMMETRY, AND IT IS THE WHOLE DESIGN
---------------------------------------------------
We encode a FRAGMENT of what the claims mean. That gives an asymmetric guarantee:

    INCONSISTENT is SOUND. If a subset of the encoded constraints is unsatisfiable, adding more
    constraints cannot make it satisfiable -- so the full set is inconsistent whatever we failed to
    encode. A contradiction found is a real contradiction.

    CONSISTENT is NOT a proof. It means no contradiction was found WITHIN THE FRAGMENT. A claim whose
    meaning we did not encode may still contradict something. Reporting this as "consistent" would be
    the same class of error as a 429 read as a dead host.

    UNKNOWN is a first-class answer, returned whenever a predicate or attribute has no declared
    semantics. Failing open here (treating unknown as fine) is how garbage passes a cleaner.

USAGE
    python tools/logos/logic/smt.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

try:
    import z3
except Exception:                                          # noqa: BLE001
    z3 = None

CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"
UNKNOWN = "UNKNOWN"


# ------------------------------------------------------------------------------- claim types
@dataclass(frozen=True)
class Triple:
    """subject --predicate--> object. Meaning comes from the predicate's DECLARED properties."""

    subject: str
    predicate: str
    object: str

    def label(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass(frozen=True)
class NumericClaim:
    """entity.attribute = value +/- tolerance, in a stated unit."""

    entity: str
    attribute: str
    value: float
    tolerance: float = 0.0
    unit: str = ""

    def label(self) -> str:
        return f"{self.entity}.{self.attribute} = {self.value}±{self.tolerance}{self.unit}"


# Predicate semantics, DECLARED rather than guessed. A predicate absent from here yields UNKNOWN --
# because guessing that an unknown predicate is a harmless label is how a contradiction goes unnoticed.
PREDICATE_SEMANTICS: dict[str, dict] = {
    "before": {"transitive": True, "irreflexive": True, "asymmetric": True},
    "after": {"transitive": True, "irreflexive": True, "asymmetric": True},
    "is_a": {"transitive": True, "irreflexive": True, "asymmetric": True},
    "part_of": {"transitive": True, "irreflexive": True, "asymmetric": True},
    "implies": {"transitive": True, "irreflexive": True, "asymmetric": True},
    "same_as": {"transitive": True, "symmetric": True, "reflexive": True},
    "different_from": {"symmetric": True, "irreflexive": True},
}


@dataclass
class Report:
    verdict: str = UNKNOWN
    core: list[str] = field(default_factory=list)
    reason: str = ""
    encoded: int = 0
    unencoded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "conflicting": self.core, "reason": self.reason,
                "encoded_claims": self.encoded, "unencoded": self.unencoded}


# ------------------------------------------------------------------------------- the checker
def check(triples: list[Triple] = (), numerics: list[NumericClaim] = (),
          semantics: dict | None = None) -> Report:
    """Is this set of claims simultaneously satisfiable within the encoded fragment?"""
    if z3 is None:
        return Report(UNKNOWN, reason="z3 is not installed; refusing to guess")

    sem = semantics if semantics is not None else PREDICATE_SEMANTICS
    rep = Report()

    # --- anything we cannot encode makes the answer UNKNOWN, not CONSISTENT. Collect first, decide after.
    for t in triples:
        if t.predicate not in sem:
            rep.unencoded.append(f"predicate {t.predicate!r} has no declared semantics ({t.label()})")
    for n in numerics:
        if n.tolerance < 0:
            rep.unencoded.append(f"negative tolerance in {n.label()}")

    usable = [t for t in triples if t.predicate in sem]
    entities = sorted({e for t in usable for e in (t.subject, t.object)})
    if not entities and not numerics:
        # CAREFUL: an input consisting ENTIRELY of unencodable claims reaches here with nothing to
        # check. Returning CONSISTENT would be the precise fail-open this module exists to prevent, so
        # the unencoded list is consulted first.
        if rep.unencoded:
            rep.verdict = UNKNOWN
            rep.reason = ("nothing encodable was supplied, so consistency is NOT established: "
                          + "; ".join(rep.unencoded[:3]))
        else:
            rep.verdict = CONSISTENT
            rep.reason = "nothing to check"
        return rep

    s = z3.Solver()
    ent_i = {e: i for i, e in enumerate(entities)}

    # --- relational part: one Boolean per (entity, entity) per predicate, with declared properties
    rel: dict[str, dict[tuple[int, int], z3.BoolRef]] = {}
    for pred in {t.predicate for t in usable}:
        props = sem[pred]
        mat = {}
        for a, b in itertools.product(range(len(entities)), repeat=2):
            mat[(a, b)] = z3.Bool(f"{pred}_{a}_{b}")
        rel[pred] = mat
        if props.get("transitive"):
            for a, b, c in itertools.product(range(len(entities)), repeat=3):
                s.add(z3.Implies(z3.And(mat[(a, b)], mat[(b, c)]), mat[(a, c)]))
        if props.get("irreflexive"):
            for a in range(len(entities)):
                s.add(z3.Not(mat[(a, a)]))
        if props.get("reflexive"):
            for a in range(len(entities)):
                s.add(mat[(a, a)])
        if props.get("asymmetric"):
            for a, b in itertools.combinations(range(len(entities)), 2):
                s.add(z3.Not(z3.And(mat[(a, b)], mat[(b, a)])))
        if props.get("symmetric"):
            for a, b in itertools.permutations(range(len(entities)), 2):
                s.add(z3.Implies(mat[(a, b)], mat[(b, a)]))

    # --- assertions are tracked so unsat_core can name the offenders
    for t in usable:
        a, b = ent_i[t.subject], ent_i[t.object]
        s.assert_and_track(rel[t.predicate][(a, b)], f"triple:{t.label()}")
        rep.encoded += 1

    # --- numeric part: one Real per (entity, attribute), interval-constrained
    grouped: dict[tuple[str, str], list[NumericClaim]] = {}
    for n in numerics:
        grouped.setdefault((n.entity, n.attribute), []).append(n)
    for (entity, attribute), claims in grouped.items():
        units = {c.unit for c in claims}
        if len(units) > 1:
            rep.unencoded.append(
                f"{entity}.{attribute} asserted in mixed units {sorted(units)}; not converted")
            continue
        v = z3.Real(f"v_{entity}_{attribute}")
        for c in claims:
            s.assert_and_track(v >= z3.RealVal(c.value - c.tolerance), f"num:{c.label()}:lo")
            s.assert_and_track(v <= z3.RealVal(c.value + c.tolerance), f"num:{c.label()}:hi")
            rep.encoded += 1

    outcome = s.check()
    if outcome == z3.unsat:
        core = s.unsat_core()
        names = []
        for c in core:
            nm = str(c)
            if nm.startswith("triple:"):
                names.append(nm[len("triple:"):])
            elif nm.startswith("num:"):
                base = nm[len("num:"):]
                base = base.rsplit(":", 1)[0]
                if base not in names:
                    names.append(base)
        rep.verdict = INCONSISTENT
        rep.core = names
        rep.reason = ("a SUBSET of the encoded claims is unsatisfiable, and adding constraints cannot "
                      "make an unsatisfiable set satisfiable -- so this conclusion is SOUND regardless "
                      "of what we failed to encode")
    else:
        rep.verdict = CONSISTENT if not rep.unencoded else UNKNOWN
        if rep.unencoded:
            rep.reason = ("no contradiction within the encoded fragment, but some claims could not be "
                          "encoded -- so consistency is NOT established")
        else:
            rep.reason = ("no contradiction found within the encoded fragment; this is not a proof of "
                          "consistency for anything not encoded")
    return rep


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.smt - SELF TEST")
    ok = True
    if z3 is None:
        print("  [SKIP] z3 not installed -- install z3-solver to exercise this module")
        return True

    # --- 1. SOUND: a cycle in a transitive, irreflexive relation is a real contradiction
    r = check([Triple("a", "before", "b"), Triple("b", "before", "c"), Triple("c", "before", "a")])
    good = r.verdict == INCONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cycle a<b<c<a on a transitive irreflexive relation is "
          f"INCONSISTENT")

    # --- 2. and unsat_core NAMES the three claims rather than just failing
    good = len(r.core) == 3 and all("before" in c for c in r.core)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unsat_core names the conflicting claims: {r.core}")

    # --- 3. a chain with no cycle is consistent
    r = check([Triple("a", "before", "b"), Triple("b", "before", "c")])
    good = r.verdict == CONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a<b, b<c with c unconstrained is CONSISTENT")

    # --- 4. THE ASYMMETRY: inconsistency is sound, consistency is fragment-relative. An unencodable
    #        predicate must yield UNKNOWN, not CONSISTENT.
    r = check([Triple("a", "before", "b"), Triple("a", "smells_like", "b")])
    good = r.verdict == UNKNOWN and any("smells_like" in u for u in r.unencoded)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an undeclared predicate makes the answer UNKNOWN, not "
          f"CONSISTENT (fails closed)")

    # --- 4b. and an input consisting ENTIRELY of unencodable claims must not report CONSISTENT.
    #         This was a real bug: the "nothing to check" early return bypassed the unencoded check,
    #         so a wholly-unknown input came back clean. That is the exact fail-open this module is
    #         supposed to prevent, found in its own code.
    r = check([Triple("a", "smells_like", "b")])
    good = r.verdict == UNKNOWN and "nothing encodable" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a wholly-unencodable input is UNKNOWN, never CONSISTENT "
          f"(the 'nothing to check' trap)")

    # --- 5. but an undeclared predicate alongside a real contradiction is still INCONSISTENT --
    #        because unsat of a subset is sound for the superset
    r = check([Triple("a", "before", "b"), Triple("b", "before", "a"),
               Triple("a", "smells_like", "b")])
    good = r.verdict == INCONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unknown claims do NOT rescue a real contradiction "
          f"(soundness holds)")

    # --- 6. asymmetric is_a: X is_a Y and Y is_a X is a contradiction
    r = check([Triple("whale", "is_a", "mammal"), Triple("mammal", "is_a", "whale")])
    good = r.verdict == INCONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] mutual is_a is INCONSISTENT")

    # --- 7. the reviewer's syllogism, as a consistency check (not a validity check)
    r = check([Triple("whale", "is_a", "mammal"), Triple("mammal", "is_a", "warm_blooded")])
    good = r.verdict == CONSISTENT and r.encoded == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] whale is_a mammal is_a warm_blooded is CONSISTENT "
          f"({r.encoded} claims encoded)")

    # --- 8. numeric contradiction with disjoint intervals, and the core names BOTH claims
    r = check([], [NumericClaim("bridge", "height", 120.0, 1.0, "m"),
                   NumericClaim("bridge", "height", 95.0, 1.0, "m")])
    good = r.verdict == INCONSISTENT and len(r.core) == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 120±1m and 95±1m are INCONSISTENT, core names both "
          f"{r.core}")

    # --- 9. overlapping intervals are fine, which is why tolerance matters
    r = check([], [NumericClaim("bridge", "height", 120.0, 1.0, "m"),
                   NumericClaim("bridge", "height", 120.5, 1.0, "m")])
    good = r.verdict == CONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 120±1m and 120.5±1m overlap and are CONSISTENT")

    # --- 10. mixed units are refused rather than silently compared
    r = check([], [NumericClaim("bridge", "height", 120.0, 1.0, "m"),
                   NumericClaim("bridge", "height", 394.0, 1.0, "ft")])
    good = r.verdict == UNKNOWN and any("mixed units" in u for u in r.unencoded)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] mixed units are NOT compared -- UNKNOWN, no silent "
          f"conversion")

    # --- 11. logical constraint: implies is transitive, so A->B->C with C->A is a cycle
    r = check([Triple("A", "implies", "B"), Triple("B", "implies", "C"), Triple("C", "implies", "A")])
    good = r.verdict == INCONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an implication cycle A->B->C->A is INCONSISTENT")

    # --- 12. empty input is trivially consistent, and says so
    r = check()
    good = r.verdict == CONSISTENT and "nothing to check" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty input is CONSISTENT with an explicit reason")

    # --- 13. same_as is symmetric: asserting one direction entails the other, so the pair is fine
    r = check([Triple("x", "same_as", "y"), Triple("y", "same_as", "x")])
    good = r.verdict == CONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] same_as is symmetric, so both directions are CONSISTENT")

    # --- 14. different_from is symmetric and irreflexive: x != x is a contradiction
    r = check([Triple("x", "different_from", "x")])
    good = r.verdict == INCONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] x different_from x is INCONSISTENT (irreflexive)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Z3-backed consistency checking for extracted claims.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
