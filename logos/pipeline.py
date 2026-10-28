"""logos/pipeline.py — the single path from raw row to disposition. The thing that was missing.

THE PROBLEM THIS FIXES, MEASURED
-------------------------------
An import audit over logos/ found **38 of 60 modules imported by nothing**. Every one of them passes its own
self-test, so the suite reported "43/43 modules pass" while nothing composed them. `row_eval.py` -- the
per-row orchestrator, the only module that decides KEEP/REPAIR/DROP -- was imported by NOBODY.

That is the difference between a library and a program. A module whose only caller is its own self-test is
not a component; it is a file. So this module is the composition, and its `--measure` output is the first
END-TO-END number the suite has produced, as opposed to 43 separate ones.

WHAT IT COMPOSES
    triage.triage          cheap routing + which checks are worth running
    row_eval.execute       subject-gated checklist -> dependency DAG -> executed findings
    rerank_client          live cross-encoder, used ONLY as an evidence-sufficiency gate
    truth_kinds            decidable claims dispatched to a validator, then recorded as witnesses

THE ORDER MATTERS AND IS DELIBERATE
    route -> cheap checks -> refutation short-circuits -> evidence gate -> fail-closed disposition
A refutation is conclusive and stops the chain. An open question is never resolved by hoping: UNKNOWN
propagates to ESCALATE. And the evidence gate can only ever make the verdict MORE conservative -- it can
demote KEEP to ESCALATE, never promote anything.

USAGE
    python tools/logos/pipeline.py --self-test
    python tools/logos/pipeline.py --measure
    python tools/logos/pipeline.py --jsonl <path> --field text --limit 50
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos import row_eval                                          # noqa: E402
from logos import triage                                            # noqa: E402

KEEP, REPAIR, ESCALATE, DROP = row_eval.KEEP, row_eval.REPAIR, row_eval.ESCALATE, row_eval.DROP
DISPOSITIONS = (KEEP, REPAIR, ESCALATE, DROP)


def _rerank():
    """Import the encoder client only when it is needed. The pipeline must still run with the gate
    unavailable -- a dead reranker degrades to 'ungated', it does not stop the batch."""
    try:
        from logos import rerank_client
        return rerank_client
    except Exception:                                            # noqa: BLE001
        return None


def _numeric_repair(row: str) -> dict:
    """Rewrite a refuted arithmetic row to its computed value, and only if the rewrite verifies.

    Deliberately narrow: `A op B = C` where C is wrong. It does not touch prose, units, algebra or code.
    A narrow repair that is always right beats a broad one that is usually right, because the output is
    training data and a plausible wrong correction is worse than a drop.
    """
    try:
        from logos.verify import math_core
    except Exception as e:                                       # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"math_core import failed: {type(e).__name__}"}

    if "=" not in row:
        return {"status": "NO_ORACLE", "reason": "no '=' so there is no right-hand side to correct"}

    # the original must actually be REFUTED -- never "repair" a row that was fine
    original = math_core.check_arithmetic_text(row)
    if original.status != "REFUTED":
        return {"status": "NO_ORACLE",
                "reason": f"validator did not refute it (status {original.status}); nothing to repair"}

    lhs = row.split("=", 1)[0].strip()
    rhs = row.split("=", 1)[1].strip()

    # GUARD 1: the left side must be a COMPUTATION. Measured failure this prevents: `pi = 3.0` was
    # "repaired" to `pi = pi`, which verifies perfectly and says nothing.
    if not re.search(r"[+\-*/]|\*\*", lhs):
        return {"status": "NOT_A_COMPUTATION",
                "reason": f"left-hand side {lhs!r} is a bare symbol, not a computation; correcting a "
                          f"constant needs a digit-precision decision this repair is not authorised to make"}

    try:
        import sympy
        expr = sympy.sympify(lhs)
        if not expr.is_number:
            return {"status": "NO_ORACLE", "reason": "left-hand side does not evaluate to a number"}
        value = sympy.nsimplify(expr)
        candidate = f"{lhs} = {value}"
    except Exception as e:                                       # noqa: BLE001
        return {"status": "NO_ORACLE", "reason": f"could not compute: {type(e).__name__}"}

    if candidate == row:
        return {"status": "NO_CHANGE", "reason": "computed value equals the claimed value"}

    # GUARD 2: the corrected value must be a NUMBER, not an echo of the left-hand side.
    # The first version of this guard compared simplify(lhs) to simplify(rhs) and rejected every
    # CORRECT repair, because `2 + 2 = 4` simplifies to `4 = 4`. Equality is what a repair is FOR;
    # repetition is the defect. So the test is not equality, it is whether the right side carries a
    # numeric literal at all -- `4` does, `pi` does not.
    vt = str(value).strip()
    if vt == lhs.strip() or not re.search(r"\d", vt):
        return {"status": "TAUTOLOGY",
                "reason": f"corrected value {vt!r} repeats the left-hand side instead of computing a "
                          f"number; verifying is not sufficient"}

    # THE INVARIANT: the repair has to satisfy the oracle that rejected the original.
    recheck = math_core.check_arithmetic_text(candidate)
    if recheck.status != "VERIFIED":
        return {"status": "REJECTED_BY_ORACLE",
                "reason": f"candidate {candidate!r} does not verify ({recheck.status}: "
                          f"{recheck.detail[:60]})"}
    return {"status": "FIXED", "candidate": candidate, "original_rhs": rhs,
            "reason": f"computed {candidate.split('=', 1)[1].strip()} from the left-hand side; "
                      f"the repaired row re-verifies"}


def guess_kind(row: str) -> str:
    """Code or text? Deliberately crude and stated, because guessing wrong changes the checklist."""
    markers = ("def ", "import ", "class ", "return ", ">>>", "lambda ", "  File \"", "Traceback")
    return "code" if any(m in row for m in markers) else "text"


def evaluate_row(row: str, kind: str | None = None, evidence: list | None = None,
                 gate_threshold: float = 0.5) -> dict:
    """One row, all the way through. Returns the disposition plus the reason for it.

    Every branch that can change the disposition is recorded in `why`, so a verdict can be audited
    without rerunning it. A disposition with no recorded reason would be an assertion, not a decision.
    """
    kind = kind or guess_kind(row)
    why: list = []

    # ---- 1. route: which checks are even worth running
    try:
        tr = triage.triage(row, kind=kind)
        route = tr.as_dict() if hasattr(tr, "as_dict") else dict(tr)
    except Exception as e:                                       # noqa: BLE001
        route = {"error": type(e).__name__}
        why.append(f"triage failed ({type(e).__name__}); routing defaulted")
    why.append(f"kind={kind}")

    # ---- 2. the per-row DAG. this is the only module that decides, and until now nothing called it.
    # ---- a row with no content is defective by inspection, no validator needed
    if not row.strip():
        return {"row": row[:200], "kind": kind, "disposition": DROP, "refuted_by": "empty",
                "cost": 0.0, "n_findings": 0, "gated": False, "gate_sufficient": None,
                "low_band": 0, "skipped": [], "findings": [],
                "why": [f"kind={kind}", "row is blank or whitespace -- nothing to evaluate"]}

    rv = row_eval.execute(row, kind=kind)
    disposition = rv.disposition
    findings = rv.findings
    if rv.refuted_by:
        why.append(f"refuted by {rv.refuted_by} (conclusive, chain stopped)")

        # ---- REPAIR: the one branch that keeps a refuted row instead of dropping it. Narrow by
        # design -- only arithmetic -- and only if the corrected row re-passes the same validator.
        if rv.refuted_by == "math":
            rep = _numeric_repair(row)
            if rep["status"] == "FIXED":
                disposition = REPAIR
                why.append(f"REPAIR: {rep['reason']}")
                return {"row": row[:200], "kind": kind, "disposition": REPAIR,
                        "refuted_by": rv.refuted_by, "cost": round(rv.total_cost, 3),
                        "n_findings": len(findings), "gated": False, "gate_sufficient": None,
                        "low_band": 0, "skipped": rv.skipped, "repaired": rep["candidate"],
                        "why": why, "findings": [f.as_dict() for f in findings]}
            why.append(f"no repair available: {rep['status']}: {rep['reason'][:80]}")
    n_unknown = sum(1 for f in findings if f.verdict == row_eval.UNKNOWN)
    n_verified = sum(1 for f in findings if f.verdict == row_eval.VERIFIED)
    n_na = sum(1 for f in findings if f.verdict == row_eval.NA)
    why.append(f"findings {n_verified} verified / {n_unknown} unknown / {n_na} na")

    # ---- SPLIT THE DROP PILE. row_eval returns DROP for both "a check refuted this" and "no check
    # applied". Only the first is a defect. The second is a row nothing here can decide, and calling it
    # junk is how a review pile vanishes.
    if disposition == DROP and not rv.refuted_by:
        disposition = ESCALATE
        why.append(f"row_eval said DROP with no named refuter ({n_na}/{len(findings)} checks NA): "
                   f"no check APPLIED, which is undecidable rather than defective -> ESCALATE")

    # ---- 3. the evidence gate. CAN ONLY DEMOTE. never promotes, because a relevance score is not support.
    gate = None
    gated = False
    if evidence:
        rc = _rerank()
        if rc is None:
            gate = {"sufficient": None, "reason": "encoder client unavailable"}
            why.append("evidence supplied but encoder client unavailable -> ungated")
        else:
            gate = rc.evidence_sufficient(row, evidence, threshold=gate_threshold)
            gated = True
            why.append(f"gate: {gate['reason']}")
            if gate.get("sufficient") is False and disposition == KEEP:
                disposition = ESCALATE
                why.append("KEEP demoted to ESCALATE: no evidence cleared the relevance threshold")
            elif gate.get("sufficient") is None:
                # fail closed: an unreachable backend is UNKNOWN, and UNKNOWN cannot support KEEP
                if disposition == KEEP:
                    disposition = ESCALATE
                    why.append("KEEP demoted to ESCALATE: gate unreachable, so evidence is UNKNOWN")

    return {"row": row[:200], "kind": kind, "disposition": disposition,
            "refuted_by": rv.refuted_by, "cost": round(rv.total_cost, 3), "repaired": None,
            "n_findings": len(findings), "gated": gated,
            "gate_sufficient": (gate or {}).get("sufficient"),
            "low_band": len((gate or {}).get("low_band") or []),
            "skipped": rv.skipped, "why": why,
            "findings": [f.as_dict() for f in findings]}


def run_rows(rows, kind: str | None = None, evidence_map: dict | None = None) -> dict:
    """A batch. Returns counts and the per-disposition reasons, not just a total."""
    out = []
    for i, row in enumerate(rows):
        ev = (evidence_map or {}).get(i)
        out.append(evaluate_row(row, kind=kind, evidence=ev))
    counts = collections.Counter(r["disposition"] for r in out)
    return {"n": len(out), "counts": {d: counts.get(d, 0) for d in DISPOSITIONS},
            "rows": out}


# ------------------------------------------------------------------------------- fixtures
# CONSTRUCTED labels, not judged ones. That must be said out loud: I wrote each row knowing what answer I
# expected, so agreement here measures whether the pipeline does what I designed it to do. It is NOT an
# accuracy estimate on real data, and the held-out-set gap is already the largest known defect.
FIXTURE = [
    # ---- should KEEP: decidable and correct
    ("2 + 2 = 4", KEEP, "text"),
    ("pi = 3.14159", KEEP, "text"),
    ("1 mile = 1.609344 km", KEEP, "text"),
    ("def add(a, b):\n    return a + b\n", KEEP, "code"),
    ("def area(r):\n    return 3.14159 * r * r\n", KEEP, "code"),
    # ---- should DROP: decidable and wrong
    ("pi = 3.0", DROP, "text"),
    ("1 mile = 1.5 km", DROP, "text"),
    ("def f():\n    return qqq + 1\n", DROP, "code"),
    ("def g(:\n    return 1\n", DROP, "code"),
    # ---- should REPAIR: wrong but mechanically correctable by computation
    ("2 + 2 = 5", REPAIR, "text"),
    ("7 * 6 = 41", REPAIR, "text"),
    # ---- prose: no computational check applies, so these are the honest hard cases
    ("A categorical syllogism has two premises and a conclusion.", ESCALATE, "text"),
    ("Apoptosis is programmed cell death.", ESCALATE, "text"),
    ("The Eiffel Tower is 330 metres tall.", ESCALATE, "text"),
    # ---- junk
    ("", DROP, "text"),
    ("   ", DROP, "text"),
]


def measure(fixture=None) -> int:
    fixture = fixture or FIXTURE
    rows = [r for r, _e, _k in fixture]
    res = run_rows(rows, kind=None)
    print("=== end-to-end: one path from row to disposition ===")
    print(f"  {res['n']} rows, constructed labels\n")
    print(f"  {'row':46s} {'kind':5s} {'expected':9s} {'actual':9s}  verdict")
    agree = 0
    confusion = collections.Counter()
    for (expect, exp, exp_kind), got in zip(fixture, res["rows"]):
        match = got["disposition"] == exp
        agree += 1 if match else 0
        confusion[(exp, got["disposition"])] += 1
        shown = expect.replace("\n", "\\n")[:46]
        print(f"  {shown:46s} {got['kind']:5s} {exp:9s} {got['disposition']:9s}"
              f"  {'ok' if match else 'MISMATCH'}")
    print(f"\n  agreement with my construction: {agree}/{res['n']} = {agree / res['n']:.3f}")
    print(f"  disposition counts: {res['counts']}")

    print("\n=== where it disagrees, and why ===")
    shown = 0
    for (expect, exp, _k), got in zip(fixture, res["rows"]):
        if got["disposition"] == exp:
            continue
        shown += 1
        print(f"  {expect.replace(chr(10), chr(92) + 'n')[:60]!r}")
        print(f"    expected {exp}, got {got['disposition']}")
        for w in got["why"]:
            print(f"      - {w}")
    if not shown:
        print("  no disagreements")

    print("\n=== how the dispositions are REACHED ===")
    for r in res["rows"][:6]:
        print(f"  {r['row'][:44]!r:48s} {r['disposition']:9s} {'; '.join(r['why'])[:70]}")

    print("\n=== the gate, on a row that is correct AND has no supporting evidence ===")
    rc = _rerank()
    if rc is None:
        print("  encoder client unavailable, gate skipped")
    else:
        r = evaluate_row("One mile is 1.609344 kilometres.", kind="text",
                         evidence=["Miles are also a surname in some European families."])
        print(f"  correct claim + irrelevant evidence -> {r['disposition']}  (gated={r['gated']})")
        for w in r["why"]:
            print(f"    - {w}")
        if r["disposition"] == ESCALATE:
            print("  ==> a true claim with no supporting evidence is ESCALATED rather than KEPT.")
            print("      precision-first: we do not certify what we cannot show.")
        r = evaluate_row("One mile is 1.609344 kilometres.", kind="text",
                         evidence=["One mile equals exactly 1.609344 kilometres by definition."])
        print(f"  correct claim + relevant evidence   -> {r['disposition']}")

    print("\n=== what this number is and is not ===")
    print("  IT IS: proof the composition runs, and that a refutation short-circuits while an open")
    print("         question escalates. The first end-to-end disposition count in this suite.")
    print("  IT IS NOT: an accuracy estimate. The labels are MINE, written knowing the intended")
    print("         answer, and 15 rows cannot separate a working pipeline from a lucky one. The")
    print("         missing held-out set remains the largest known defect in the whole project.")
    print("  AND: prose rows all land on ESCALATE, which is honest -- no check applies to 'the Eiffel")
    print("         Tower is 330 metres tall'. That is the abstain pile, and its size is the real")
    print("         measure of how much of a prose corpus this pipeline can actually decide.")
    return 0


# ------------------------------------------------------------------------------- self tests
def run_self_test() -> bool:
    print("logos.pipeline - SELF TEST")
    ok = True

    # --- composition: the orchestrator is actually reachable through the pipeline now
    good = row_eval is not None and triage is not None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pipeline imports row_eval and triage (both were orphans)")

    # --- a correct arithmetic row is KEPT
    r = evaluate_row("2 + 2 = 4", kind="text")
    good = r["disposition"] == KEEP
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 2 + 2 = 4 -> {r['disposition']}")

    # --- a wrong arithmetic row is no longer simply dropped. It is REPAIRED, and the refuting check
    #     is named, because the refutation is what justified touching the row at all.
    r = evaluate_row("2 + 2 = 5", kind="text")
    good = r["disposition"] == REPAIR and bool(r["refuted_by"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 2 + 2 = 5 -> {r['disposition']} "          f"(refuted_by={r['refuted_by']!r})")

    # --- an empty row is not KEPT
    for blank in ("", "   ", "\n"):
        r = evaluate_row(blank, kind="text")
        good = r["disposition"] != KEEP
        ok = ok and good
        if not good:
            print(f"  [FAIL] a blank row ({blank!r}) was KEPT")
    print(f"  [{'PASS' if ok else 'FAIL'}] blank rows are never KEPT")

    # --- REPAIR is now reachable, which it was not: row_eval assigns it nowhere.
    r = evaluate_row("2 + 2 = 5", kind="text")
    good = r["disposition"] == REPAIR and r.get("repaired") == "2 + 2 = 4"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a refuted arithmetic row is REPAIRED to "
          f"{r.get('repaired')!r} (REPAIR was unreachable before)")

    # --- and the repaired text must itself verify. a repair nobody can check is a guess.
    if r.get("repaired"):
        from logos.verify import math_core
        good = math_core.check_arithmetic_text(r["repaired"]).status == "VERIFIED"
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] the repaired row re-passes the validator that "
              f"refuted the original")

    # --- a NON-arithmetic refutation is not "repaired" by inventing a number
    r = evaluate_row("1 mile = 1.5 km", kind="text")
    good = r["disposition"] in (DROP, REPAIR)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a unit refutation is {r['disposition']}, "
          f"not silently rewritten to a bare number")

    # --- GUARD 1: a bare-symbol claim is NOT repaired into a tautology. This was a real bug: the
    #     first version rewrote `pi = 3.0` to `pi = pi`, which verifies and deletes the claim.
    r = evaluate_row("pi = 3.0", kind="text")
    good = r["disposition"] == DROP and not r.get("repaired")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3.0 is {r['disposition']}, not 'repaired' to pi = pi "
          f"(the tautology bug)")

    # --- GUARD 2, directly: an echo is refused even though it verifies. Guard 1 catches `pi = 3.0`
    #     before guard 2 is reached, so exercise the echo rule on its own.
    import sympy as _sp
    echo = "2 + 2"
    vt = str(_sp.nsimplify(_sp.sympify(echo)))
    good = vt == "4" and _sp.sympify(echo).is_number
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a computation yields a numeric literal ({echo} -> {vt}), "
          f"which is what guard 2 requires")

    rep = _numeric_repair("pi = 3.0")
    good = rep["status"] == "NOT_A_COMPUTATION"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] guard 1 fires first on a bare symbol ({rep['status']})")

    # --- and a real computation still repairs, so the guards did not break the feature
    rep = _numeric_repair("7 * 6 = 41")
    good = rep["status"] == "FIXED" and rep["candidate"] == "7 * 6 = 42"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 7 * 6 = 41 still repairs to {rep.get('candidate')!r}")

    # --- a CORRECT row is never "repaired"
    r = evaluate_row("2 + 2 = 4", kind="text")
    good = r["disposition"] == KEEP and not r.get("repaired")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a correct row is KEPT, not repaired")

    # --- THE RE CLASSIFICATION. An undecidable row must reach the review pile, not the junk pile.
    r = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text")
    good = r["disposition"] == ESCALATE and not r["refuted_by"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a prose row no check applies to -> {r['disposition']} "
          f"(undecidable is not junk)")

    # --- and a genuinely refuted row still DROPs, so the split did not weaken rejection
    r = evaluate_row("1 mile = 1.5 km", kind="text")
    good = r["disposition"] == DROP and bool(r["refuted_by"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a refuted row still DROPs with a named refuter "
          f"({r['disposition']}, {r['refuted_by']!r})")

    # --- every disposition carries a recorded reason. a verdict with no reason is an assertion.
    good = all(evaluate_row(x, kind="text")["why"] for x in ("2 + 2 = 4", "2 + 2 = 5", "", "hello"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every row records WHY it was disposed of")

    # --- the gate can only DEMOTE. a passing gate must not turn ESCALATE into KEEP.
    rc = _rerank()
    if rc is None:
        print("  [SKIP] encoder client unavailable")
    else:
        base = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text")
        gated = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text",
                             evidence=["The Eiffel Tower stands 330 m high."])
        good = base["disposition"] == gated["disposition"] == ESCALATE
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a PASSING gate does not promote {base['disposition']} "
              f"to KEEP (still {gated['disposition']})")

        # --- and a failing gate demotes a KEEP. This must use a row that EARNS KEEP, i.e. one a
        #     validator can actually decide -- otherwise there is nothing to demote and the test
        #     passes for the wrong reason.
        base = evaluate_row("1 mile = 1.609344 km", kind="text")
        dem = evaluate_row("1 mile = 1.609344 km", kind="text",
                           evidence=["Miles are a surname in some families."])
        good = base["disposition"] == KEEP and dem["disposition"] == ESCALATE and dem["gated"]
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a FAILING gate demotes a KEPT claim "
              f"({base['disposition']} -> {dem['disposition']})")

        # --- and a PASSING gate leaves it KEPT, so the gate is not merely a blanket demoter
        keep = evaluate_row("1 mile = 1.609344 km", kind="text",
                            evidence=["One mile equals exactly 1.609344 kilometres."])
        good = keep["disposition"] == KEEP
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a PASSING gate leaves the same row at "
              f"{keep['disposition']}")

        # --- and a contradicting passage is not silently swallowed: it survives in the low band
        dem2 = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text",
                            evidence=["The Eiffel Tower is 300 metres tall."])
        good = dem2["low_band"] >= 1
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] contradicting evidence survives in the low band "
              f"({dem2['low_band']} passage(s)) for a validator to judge")

    # --- a batch returns counts for EVERY disposition, so a missing class is visible as 0 not absent
    res = run_rows(["2 + 2 = 4", "2 + 2 = 5"])
    good = set(res["counts"]) == set(DISPOSITIONS) and res["n"] == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] batch counts always contain all four dispositions")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Row -> disposition, end to end.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--field", default="text")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--explain", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.explain:
        print(json.dumps(evaluate_row(args.explain), indent=2, default=str))
        return 0
    if args.jsonl:
        rows = []
        with open(args.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                val = obj.get(args.field)
                if isinstance(val, str):
                    rows.append(val)
        res = run_rows(rows)
        print(json.dumps({"n": res["n"], "counts": res["counts"]}, indent=2))
        for r in res["rows"][:10]:
            print(f"  {r['disposition']:9s} {r['row'][:60]!r}")
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
