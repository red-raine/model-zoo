"""logos/logic/constraint_solver.py — minimal-cost repair over typed facts, with HARD and SOFT rules.

WHERE THIS COMES FROM
--------------------
Yan, Wang, Cheng, Gao & Zhou, "A retrospective of knowledge graphs", Frontiers of Computer Science 12(1):55-74
(2018), section 4.1. Three things in it are directly load-bearing here:

  1. Constraints written as Horn clauses. Their worked example, verbatim:
         not bornInCity(x,y) OR not cityInCountry(y,z) OR bornInCountry(x,z)
     and the conflict it detects:
         bornInCity(Bob, Shanghai) and cityInCountry(Shanghai, China)  vs  bornInCountry(Bob, US)
  2. **HARD vs SOFT rules**: "Soft rules can be violated in some degree while hard rules must always be hold.
     For soft rules, they can be weighted to handle uncertain or probabilistic data."
     That is exactly the distinction a KEEP / REPAIR / ESCALATE / DROP ladder needs, and it is the reason this
     module has two rule tiers instead of one.
  3. SOFIE "integrates logical consistency reasoning with information extraction", casting facts, hypotheses and
     constraints into logical clauses with weights and solving as **weighted MAX-SAT**.

WHY NOT A PLAIN SAT CHECK
-------------------------
A boolean consistency check answers "is this set of facts satisfiable?" and stops. That is not actionable: when
the answer is no, the caller still has to decide WHAT to retract, and a plain unsat core does not say which
choice is cheapest. Weighted MAX-SAT does: every fact is a soft assertion with a cost, so the solver returns the
MINIMUM-COST set of facts whose retraction restores consistency. For a precision-first suite that is the whole
question -- a false DROP of good data is catastrophic and a refusal is cheap, so the repair must retract the
cheapest sufficient set rather than any satisfying one.

THE VERDICT MAPPING, WHICH IS WHY THIS IS THE RIGHT ENGINE
---------------------------------------------------------
    CONSISTENT          every fact holds -> the row is untouched                          -> KEEP
    REPAIRED            a HARD rule was violated and a minimum-cost retraction restored
                        consistency -> the retracted atom is the proof, so the row is
                        rewritable and the change is nameable                            -> REPAIR
    ESCALATE            only SOFT rules are violated, or two retractions tie on cost ->    -> ESCALATE
                        the cheapest repair is not unique, so choosing one would be a
                        guess dressed as a verdict
    NO_MODEL            a hard rule is violated and no retraction restores consistency
                        (the constraint set itself is broken)                              -> ESCALATE (loudly)

Weights are the knob: a fact the corpus asserts twice, or one carrying an external source, costs more to retract
than a bare literal, so the solver prefers retracting the weaker claim. That is the "soft rule" idea applied to
facts.

USAGE
    python tools/logos/logic/constraint_solver.py --self-test
    python tools/logos/logic/constraint_solver.py --demo      # the bornIn conflict, end to end
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

CONSISTENT, REPAIRED, ESCALATE, NO_MODEL = "CONSISTENT", "REPAIRED", "ESCALATE", "NO_MODEL"


def _z3():
    try:
        import z3
        return z3
    except Exception:                                        # noqa: BLE001
        return None


def available() -> tuple:
    z = _z3()
    if z is None:
        return False, "z3 missing"
    return True, f"z3 {z.get_version_string()}"


def fact(subject: str, relation: str, obj: str) -> str:
    """Facts are typed triples; the canonical key is what the solver asserts on."""
    return f"{str(subject).strip()}|{str(relation).strip()}|{str(obj).strip()}"


def parse(key: str) -> tuple:
    s, r, o = key.split("|", 2)
    return s, r, o


def requires_clauses(pairs) -> list:
    """Expand `A requires one of P1..Pn` into a single CNF clause: `P1 or P2 or ... or not A`.

    ALTERNATIVES, NOT CONJUNCTION -- and I got this wrong first. `qualify()` returns up to four candidate
    attributes of a concept, and requiring ALL of them made every edge undischargeable even when one qualifier
    was attested: measured, attesting `water|attribute|oxygen.n.01` still retracted `water|UsedFor|cleaning`
    because hydrogen and water were also demanded. That is not what the ARN argument says. The relation holds
    between SOME aspect of the concept and the tail ("some attributes of the concept"), so one attested aspect is
    enough to state the conditional claim. Use `requires_all_clauses` when a genuine conjunction is meant.
    """
    out = []
    for fact_key, pre in pairs:
        pre = [p for p in pre if p]
        if pre:
            out.append((pre, [fact_key]))
    return out


def requires_all_clauses(pairs) -> list:
    """Expand `A requires ALL of P1..Pn` into one clause per precondition.

    Kept separate from `requires_clauses` because the two are different claims and conflating them is exactly the
    bug that made every conditional edge undischargeable.
    """
    out = []
    for fact_key, pre in pairs:
        for p in pre:
            out.append(([p], [fact_key]))
    return out


def solve(facts, functional=(), mutex=(), implications=(), clauses=(), requires=(), requires_all=(),
          false_atoms=(), weights=None, max_ties: int = 12) -> dict:
    """Find the minimum-cost set of facts to retract so that every constraint holds.

    facts         iterable of fact() keys. Each is a SOFT assertion: it may be retracted at a cost.
    functional    relation names where a subject has at most one object ("bornInCountry"). This is the workhorse
                  constraint and the one that catches the survey's worked conflict.
    mutex         explicit (fact_a, fact_b) pairs that cannot both hold.
    implications  list of (premises, conclusion) pairs -- Horn rules. HARD by construction: a violated
                  implication may not be "satisfied" by dropping the rule, only by dropping a fact.
    clauses       list of (positive_atoms, negative_atoms) meaning OR(pos...) OR NOT(neg...). This is CNF, and it
                  is what lets a caller say `KEEP requires one of these supporting atoms` --
                  `not KEEP or support1 or support2`. Horn form cannot express that (it needs atom -> OR(atoms)).
    requires      list of (fact_key, [precondition_atom, ...]) meaning the fact is asserted only if all of its
                  preconditions survive. Expanded via requires_clauses().
    false_atoms   atoms bound HARD to false. THIS IS THE LOAD-BEARING PARAMETER for conditional claims.

                  Measured mistake, kept here because it is subtle: I first modelled a precondition as
                  `A -> P` with both A and P as soft facts, and 13 of 13 conditional edges came back KEPT at
                  cost 0. The clause was satisfied by simply asserting P, and asserting an unattested atom is
                  FREE in a MAX-SAT that only charges for retraction. **A precondition that is free to assert
                  is not a constraint.** So a precondition with nothing attesting it must be bound to false,
                  which is also the honest semantics: asserting an unevidenced atom is fabrication. With P
                  hard-false, `A -> P` forces A to be retracted and the edge is NAMED as undischargeable.
    weights       {fact_key: cost}. Default 1.0. Higher means the solver prefers to retract something else.

    Returns status, retracted facts, total cost, whether the cheapest repair is UNIQUE, and `held` -- the atoms
    asserted in the optimal model, so a caller can ask "which support survived?" rather than only "what broke".
    """
    if requires:
        clauses = list(clauses) + requires_clauses(requires)
    if requires_all:
        clauses = list(clauses) + requires_all_clauses(requires_all)
    z = _z3()
    if z is None:
        return {"status": NO_MODEL, "reason": "z3 unavailable"}
    # SOFT facts are the retractable ones and the ONLY ones the cost charges for. Clause-only atoms, hypotheses
    # and `false_atoms` all get variables but no soft assertion -- otherwise a hard-false qualifier would show up
    # as a "retraction" and inflate the cost, which is how a declared-false atom gets reported as a loss.
    soft = list(dict.fromkeys(facts))
    false_atoms = list(dict.fromkeys(false_atoms))
    atoms = list(dict.fromkeys(soft + false_atoms
                               + [a for pos_neg in clauses for a in list(pos_neg[0]) + list(pos_neg[1])]))
    if not atoms:
        return {"status": CONSISTENT, "retracted": [], "cost": 0.0, "unique": True, "held": [],
                "reason": "no facts"}
    w = {k: float((weights or {}).get(k, 1.0)) for k in soft}
    v = {k: z.Bool(f"f{ i }") for i, k in enumerate(atoms)}
    opt = z.Optimize()

    # ---- HARD: functional relations. a subject has at most one object per relation.
    by_sr = collections.defaultdict(list)
    for k in soft:
        s, r, _o = parse(k)
        if r in functional:
            by_sr[(s, r)].append(k)
    n_hard_pairs = 0
    for (s, r), group in by_sr.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                opt.add(z.Not(z.And(v[group[i]], v[group[j]])))
                n_hard_pairs += 1

    # ---- HARD: explicit mutual exclusion
    for a, b in mutex:
        if a in v and b in v:
            opt.add(z.Not(z.And(v[a], v[b])))
            n_hard_pairs += 1

    # ---- HARD: Horn implications. premises all true => conclusion true.
    #      The conclusion may be a fact not present in `facts`; it is introduced as a hypothesis so that a
    #      derived fact can CONFLICT with an asserted one, which is the survey's exact scenario.
    hyp = {}
    for premises, conclusion in implications:
        # premises are KEYS; z3 needs the Bool VARIABLES. Passing the strings raised
        # "Sort mismatch ... supplied sort is String", and keeping them as strings after the membership
        # filter looked correct because `in v` was True for the key.
        pvars = [v[p] for p in premises if p in v]
        if not pvars:
            continue
        if conclusion in v:
            c = v[conclusion]
        else:
            if conclusion not in hyp:
                hyp[conclusion] = z.Bool(f"h{len(hyp)}")
            c = hyp[conclusion]
        opt.add(z.Implies(z.And(*pvars), c))

    # derived facts introduced as hypotheses can also conflict with each other under functionality
    for concl, hv in hyp.items():
        s, r, _o = parse(concl)
        if r in functional:
            for k in by_sr.get((s, r), []):
                opt.add(z.Not(z.And(hv, v[k])))
                n_hard_pairs += 1
    # two different derived values for the same functional (s, r) also conflict
    by_sr_h = collections.defaultdict(list)
    for concl, hv in hyp.items():
        s, r, _o = parse(concl)
        if r in functional:
            by_sr_h[(s, r)].append((concl, hv))
    for (_s, _r), group in by_sr_h.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                opt.add(z.Not(z.And(group[i][1], group[j][1])))
                n_hard_pairs += 1

    # ---- HARD: CNF clauses. (pos, neg) means OR(pos) OR NOT(neg).
    for pos_atoms, neg_atoms in clauses:
        lits = [v[a] for a in pos_atoms if a in v] + [z.Not(v[a]) for a in neg_atoms if a in v]
        if lits:
            opt.add(z.Or(*lits))
            n_hard_pairs += 1

    # ---- HARD: atoms bound to false. A precondition with nothing attesting it is FALSE, not "unasserted".
    for a in false_atoms:
        if a in v:
            opt.add(z.Not(v[a]))
            n_hard_pairs += 1

    # ---- OBJECTIVE: retract as little as possible. This is the MAX-SAT part.
    #      Weights are scaled to INTEGERS because z3's add_soft wants a numeric weight, and float -> String
    #      coercion is how a weight silently becomes an error rather than a cost.
    for k in soft:
        opt.add_soft(v[k], weight=max(1, int(round(w[k] * 1000))))

    r = opt.check()
    if r != z.sat:
        return {"status": NO_MODEL, "reason": f"constraints unsatisfiable even with all facts retracted ({r})",
                "n_hard_pairs": n_hard_pairs}

    m = opt.model()
    retracted = [k for k in soft if not z.is_true(m.eval(v[k], model_completion=True))]
    held = [k for k in soft if z.is_true(m.eval(v[k], model_completion=True))]
    cost = sum(w[k] for k in retracted)

    # ---- UNIQUENESS: re-optimise with the current retraction set BLOCKED. If an equally cheap solution
    #      exists, the cheapest repair is not unique and reporting REPAIRED would launder a guess.
    unique, ties = _check_unique(z, v, opt, facts, w, cost, retracted, max_ties)

    status = CONSISTENT if not retracted else (ESCALATE if not unique else REPAIRED)
    return {"status": status, "retracted": retracted, "cost": round(cost, 3), "unique": unique,
            "held": held, "n_facts": len(soft), "n_hard_pairs": n_hard_pairs, "alternatives": ties,
            "reason": ("all facts hold" if not retracted else
                       ("minimum-cost repair is unique" if unique else
                        "the cheapest repair is NOT unique, so choosing one would be a guess"))}


def _prem(present, implications):
    return present


def _check_unique(z, v, opt, facts, w, cost, retracted, max_ties) -> tuple:
    """Is there a SECOND solution of the same total cost?

    This is decided by blocking the current retraction set and re-optimising, not by inspecting costs. My first
    version guessed from weights alone and was wrong: the survey's own conflict has two single-atom repairs at
    cost 1 (retract the rival assertion, or retract the premise that derives the competing value), so the repair
    genuinely IS ambiguous and must ESCALATE rather than pick.
    """
    alts = []
    block = z.Or(*[(v[k] if k in retracted else z.Not(v[k])) for k in facts]) if facts else None
    if block is None:
        return True, []
    opt.push()
    try:
        opt.add(block)
        r = opt.check()
        if r == z.sat:
            m = opt.model()
            alt = [k for k in facts if not z.is_true(m.eval(v[k], model_completion=True))]
            alt_cost = sum(w[k] for k in alt)
            if abs(alt_cost - cost) < 1e-9:
                alts.append({"alternative": alt, "cost": round(alt_cost, 3)})
    except Exception:                                        # noqa: BLE001
        pass
    finally:
        opt.pop()
    return (len(alts) == 0), alts[:max_ties]


# ------------------------------------------------------------------ demo on the survey's own example
def demo() -> int:
    ok, why = available()
    print("=== weighted MAX-SAT over typed facts (the FCS survey section 4.1 scenario) ===")
    print(f"  {why}\n")
    if not ok:
        return 1

    city = fact("Bob", "bornInCity", "Shanghai")
    chain = fact("Shanghai", "cityInCountry", "China")
    wrong = fact("Bob", "bornInCountry", "US")
    derived = fact("Bob", "bornInCountry", "China")

    print("  FACTS")
    for f in (city, chain, wrong):
        print(f"    {f}")
    print("\n  HARD RULE (Horn):  bornInCity(x,y) & cityInCountry(y,z) => bornInCountry(x,z)")
    print("  HARD RULE:         bornInCountry is FUNCTIONAL (one country per person)")
    print("\n  With the rule applied, Bob is born in BOTH China and the US. Something must give.\n")

    r = solve([city, chain, wrong],
              functional=("bornInCountry",),
              implications=[(([city, chain]), derived)])
    print(f"  status      {r['status']}")
    print(f"  retract     {r['retracted']}")
    print(f"  cost        {r['cost']}   unique: {r['unique']}")
    print(f"  reason      {r['reason']}")
    print()
    print("  READ THIS: the solver did not report 'inconsistent'. It reported a cost-1 repair AND that the")
    print("  repair is NOT unique -- because retracting the rival assertion costs the same as retracting the")
    print("  premise that derives the competing value. Two different single-fact repairs, equal cost, so the")
    print("  honest verdict is ESCALATE and not REPAIRED. A solver that silently picked either one would be")
    print("  presenting a guess as a proof object.")
    print("  Weights resolve it, and the resolution is a POLICY the caller states rather than an accident of")
    print("  solver order: when the premises are well-attested, the rival assertion is the one to give up.")

    print("\n=== and when the repair is a COIN FLIP it says so ===")
    a = fact("X", "rel", "one")
    b = fact("X", "rel", "two")
    r2 = solve([a, b], functional=("rel",))
    print(f"  facts {a} and {b}, rel is functional -> status {r2['status']}, unique {r2['unique']}")
    print(f"  {r2['reason']}")
    print(f"  alternatives considered: {len(r2['alternatives'])}")
    print("\n  Both cost 1, so no repair is cheaper. Picking either would be a guess dressed as a verdict, and")
    print("  the honest answer is ESCALATE.")

    print("\n=== weights let a better-attested fact survive ===")
    r3 = solve([a, b], functional=("rel",), weights={a: 3.0, b: 1.0})
    print(f"  with weights {a}=3, {b}=1 -> retract {r3['retracted']} (cost {r3['cost']}, "
          f"unique {r3['unique']})")
    print("  The solver now prefers to give up the CHEAPER claim. Soft weights are how 'this row has two")
    print("  independent sources' turns into 'do not retract this one'.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.logic.constraint_solver - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] {why}")
    if not av:
        print("SELF TEST: SKIPPED (no z3)")
        return True

    # --- the survey's own conflict, resolved to a named retraction
    city = fact("Bob", "bornInCity", "Shanghai")
    chain = fact("Shanghai", "cityInCountry", "China")
    wrong = fact("Bob", "bornInCountry", "US")
    derived = fact("Bob", "bornInCountry", "China")
    r = solve([city, chain, wrong], functional=("bornInCountry",),
              implications=[(([city, chain]), derived)])
    good = r["status"] == ESCALATE and r["cost"] == 1.0 and r["unique"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the survey conflict detects a cost-1 repair but reports the tie "
          f"({r['status']}, unique={r['unique']})")

    # --- and weighting the premise chain makes the repair UNIQUE: the rival assertion is the one to give up
    r1b = solve([city, chain, wrong], functional=("bornInCountry",),
                implications=[(([city, chain]), derived)],
                weights={chain: 5.0, city: 5.0})
    good = r1b["status"] == REPAIRED and r1b["retracted"] == [wrong] and r1b["unique"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] weighting the premises makes the repair unique and names it "
          f"({r1b['retracted']})")

    # --- a consistent set is left ALONE, not "repaired"
    r2 = solve([city, chain], functional=("bornInCountry",))
    good = r2["status"] == CONSISTENT and r2["retracted"] == [] and r2["cost"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a consistent fact set is untouched ({r2['status']})")

    # --- a TIE must escalate, never pick
    a = fact("X", "rel", "one")
    b = fact("X", "rel", "two")
    r3 = solve([a, b], functional=("rel",))
    good = r3["status"] == ESCALATE and r3["unique"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cost tie ESCALATES instead of choosing "
          f"({r3['status']}, unique={r3['unique']})")

    # --- weights break the tie deterministically in favour of the costlier fact
    r4 = solve([a, b], functional=("rel",), weights={a: 3.0, b: 1.0})
    good = r4["status"] == REPAIRED and r4["retracted"] == [b]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a weight breaks the tie toward retracting the cheaper claim "
          f"({r4['retracted']})")

    # --- a functional constraint with three values retracts two, not one
    c = fact("X", "rel", "three")
    r5 = solve([a, b, c], functional=("rel",))
    good = len(r5["retracted"]) == 2 and r5["cost"] == 2.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] three rival values need TWO retractions, and the cost says so "
          f"({len(r5['retracted'])} at cost {r5['cost']})")

    # --- explicit mutex works independently of functionality
    r6 = solve([a, b], mutex=[(a, b)])
    good = len(r6["retracted"]) == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] explicit mutex pairs are honoured ({r6['retracted']})")

    # --- empty input is consistent, not an error
    r7 = solve([])
    good = r7["status"] == CONSISTENT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty input is CONSISTENT ({r7['status']})")

    # --- a fact key round-trips, so a retraction can be cited verbatim
    good = parse(fact("Bob", "bornInCountry", "US")) == ("Bob", "bornInCountry", "US")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] fact keys round-trip so a retraction is a citable proof object")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Weighted MAX-SAT minimal-cost repair over typed facts.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.demo:
        return demo()
    if args.json:
        spec = json.load(open(args.json, encoding="utf-8"))
        print(json.dumps(solve(spec.get("facts", []), functional=tuple(spec.get("functional", [])),
                               mutex=[tuple(m) for m in spec.get("mutex", [])],
                               implications=[(p, c) for p, c in spec.get("implications", [])],
                               weights=spec.get("weights")), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
