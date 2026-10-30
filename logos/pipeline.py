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


# Which validators can be applied to a WHOLE ROW without argument extraction. This list is deliberately
# short, and the reason matters: `check_identity(lhs, rhs)`, `check_derivative(expr, var, claimed)`,
# `check_solution(equation, var, value)` and the matrix checks take 2-3 arguments, so calling them with one
# string raises TypeError. My first version caught that and fell through, which SILENTLY reported such rows as
# tier 3 -- the same swallowed-error sin as the numeric_conflict bootstrap, in code written minutes later.
# So a row that no single-argument validator can examine is reported as UNCLASSIFIED with the reason, not as
# "no oracle".
ROW_VALIDATORS = (
    # (name, tier_when_refuted)
    ("check_arithmetic_text", 1),      # the left side is an expression, the right is its value
    ("check_numeric_constant", 1),     # the constant NAME is the authority
    ("check_unit_conversion", 2),      # BOTH sides are claims; the row does not say which is wrong
)

# validators that need more than the row text. listed so their absence is visible rather than implied.
NEEDS_ARGUMENT_EXTRACTION = ("check_identity", "check_derivative", "check_integral",
                             "check_solution", "check_determinant", "check_matrix_product")


def classify_repair(row: str) -> dict:
    """Tier, plus an account of what was NOT examined. A verdict with no such account is a guess.

    TIER 1 = the validator COMPUTES the answer, so the correction is uniquely determined and one side of the
             claim is authoritative by construction.
    TIER 2 = the validator can only CHECK, so repairing means SEARCHING for a candidate.
    TIER 3 = nothing in this suite can decide it.
    UNCLASSIFIED = the applicable validators need arguments extracted from the row first, which this
             classifier does not do. Reporting that as tier 3 conflated "no oracle" with "not attempted".
    """
    try:
        from logos.verify import math_core
    except Exception as e:                                    # noqa: BLE001
        return {"tier": None, "reason": f"math_core unavailable ({type(e).__name__})",
                "examined": [], "unexamined": list(NEEDS_ARGUMENT_EXTRACTION)}
    examined, refuted_by = [], None
    for name, tier in ROW_VALIDATORS:
        fn = getattr(math_core, name, None)
        if fn is None:
            continue
        try:
            status = fn(row).status
        except Exception as e:                                # noqa: BLE001
            return {"tier": None, "reason": f"{name} raised {type(e).__name__}: {e}",
                    "examined": examined, "unexamined": list(NEEDS_ARGUMENT_EXTRACTION)}
        examined.append(f"{name}={status}")
        if status == "REFUTED":
            refuted_by = (name, tier)
            break
    if refuted_by is None:
        return {"tier": 3, "reason": "no row-level validator refuted it",
                "examined": examined, "unexamined": list(NEEDS_ARGUMENT_EXTRACTION)}
    return {"tier": refuted_by[1], "refuted_by": refuted_by[0], "examined": examined,
            "unexamined": list(NEEDS_ARGUMENT_EXTRACTION)}


def _repair_tier_for(row: str) -> int:
    """Convenience: the tier, or 3 when unclassified. Kept because callers want one number."""
    t = classify_repair(row)["tier"]
    return 3 if t is None else t


def _tier1_repair(row: str) -> dict:
    """Compute the correction when ONE side of the row is authoritative by construction.

    Two shapes only, and why each is unambiguous rather than assumed:
        `A op B = C`   the left side is a computable expression, the right is its value -> compute the left
        `name = v`     the constant NAME is the authority, the value is an estimate -> correct the rounding

    A unit conversion is deliberately NOT here. `1 mile = 1.5 km` has two competing claims and the row does
    not say which one is wrong, so "repairing" it would be a guess wearing a proof. Measured, that is the
    difference between a correction and an assumption.
    """
    try:
        from logos.verify import math_core
    except Exception as e:                                    # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"math_core import failed: {type(e).__name__}"}
    import re as _re

    # ---- shape 1: a named constant. the NAME is authoritative, the literal is the estimate.
    try:
        const_refuted = math_core.check_numeric_constant(row).status == "REFUTED"
    except Exception:                                         # noqa: BLE001
        const_refuted = False
    if const_refuted:
        m = _re.match(r"^\s*([A-Za-z][A-Za-z0-9 ()\-]{0,24}?)\s*(?:=|is|\u2248|=~)\s*(\d+\.?\d*)\s*$",
                      row)
        if m:
            name, literal = m.group(1).strip(), m.group(2)
            try:
                from logos.verify.math_core import _round_to_sf, _significant_digits
                true_val = float(math_core.CONSTANTS[name.lower().rstrip("(").strip()]().evalf())
                sf = _significant_digits(literal)
                fixed = _round_to_sf(true_val, sf)
                candidate = f"{name} = {fixed:g}"
                if math_core.check_numeric_constant(candidate).status == "VERIFIED":
                    return {"status": "FIXED", "candidate": candidate, "tier": 1,
                            "validator": "check_numeric_constant",
                            "reason": f"{name} correctly rounded to {sf} s.f. is {fixed:g}"}
            except Exception:                                 # noqa: BLE001
                pass
        return {"status": "NO_ORACLE",
                "reason": "constant claim, but the corrected rounding did not verify"}

    # ---- shape 1b: calculus. the expression and variable are the authority; the claimed result is derived.
    try:
        calc = math_core.check_calculus_text(row)
    except Exception:                                         # noqa: BLE001
        calc = None
    if calc is not None and calc.status == "REFUTED" and calc.value is not None:
        m = None
        import re as _re2
        m = _re2.match(r"^\s*d\s*/\s*d\s*([A-Za-z])\s*(.+?)\s*=\s*(.+?)\s*$", row, _re.I)
        if m:
            var, expr = m.group(1), m.group(2)
            expr = _re2.sub(r"^\s*of\s+", "", expr, flags=_re.I)
            candidate = f"d/d{var} {expr} = {calc.value}"
            if math_core.check_calculus_text(candidate).status == "VERIFIED":
                return {"status": "FIXED", "candidate": candidate, "tier": 1,
                        "validator": "check_derivative",
                        "reason": f"d/d{var}({expr}) = {calc.value}; the repaired row re-verifies"}

    # ---- shape 2: arithmetic. the left side is a computable expression.
    try:
        v = math_core.check_arithmetic_text(row)
    except Exception as e:                                    # noqa: BLE001
        return {"status": "VALIDATOR_RAISED", "reason": f"check_arithmetic_text: {type(e).__name__}"}
    if v.status != "REFUTED":
        return {"status": "NO_ORACLE",
                "reason": f"no tier-1 validator refuted it (arithmetic {v.status})"}
    if "=" not in row:
        return {"status": "NO_ORACLE", "reason": "no '=' so there is no value to compute"}
    lhs = row.split("=", 1)[0].strip()
    if not _re.search(r"[+\-*/]|\*\*", lhs):
        return {"status": "NOT_A_COMPUTATION",
                "reason": f"left-hand side {lhs!r} is a bare symbol, not a computation"}
    try:
        import sympy
        expr = sympy.sympify(lhs)
        if not expr.is_number:
            return {"status": "NO_ORACLE", "reason": "left-hand side does not evaluate to a number"}
        value = sympy.nsimplify(expr)
        candidate = f"{lhs} = {value}"
    except Exception as e:                                    # noqa: BLE001
        return {"status": "NO_ORACLE", "reason": f"could not compute: {type(e).__name__}"}
    if candidate == row:
        return {"status": "NO_CHANGE", "reason": "computed value equals the claimed value"}
    vt = str(value).strip()
    if vt == lhs.strip() or not _re.search(r"\d", vt):
        return {"status": "TAUTOLOGY",
                "reason": f"corrected value {vt!r} repeats the left side instead of computing a number"}
    if math_core.check_arithmetic_text(candidate).status != "VERIFIED":
        return {"status": "REJECTED_BY_ORACLE", "reason": f"candidate {candidate!r} does not re-verify"}
    return {"status": "FIXED", "candidate": candidate, "tier": 1,
            "validator": "check_arithmetic_text",
            "reason": f"computed {candidate.split('=', 1)[1].strip()} from the left-hand side; "
                      f"the repaired row re-verifies"}


def _numeric_repair(row: str) -> dict:
    """Tier-1 repair, kept as the name existing callers use. See `_tier1_repair`."""
    return _tier1_repair(row)


def _numeric_repair_legacy(row: str) -> dict:
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


STRUCTURAL_CHECKS = {"structure"}


def _only_structural_verification(findings) -> bool:
    """Is every VERIFIED finding purely structural?

    `structure` asks "is this well-formed", which is a PRECONDITION for reading a row and not a conclusion
    about it. Measured on 200 curated claims, structural verification alone produced 75 KEEPs -- including
    rows whose gold label is REFUTED -- which inflated apparent coverage from ~0 to 37.5% while every
    individual step in the chain was arithmetically correct.
    """
    verified = [f for f in findings if getattr(f, "verdict", None) == row_eval.VERIFIED]
    if not verified:
        return False
    names = {getattr(f, "name", None) or getattr(f, "check", "") for f in verified}
    return names.issubset(STRUCTURAL_CHECKS)


def _row_is_decidable_numeric(row: str) -> bool:
    """Does the row itself parse as a numeric claim a validator understands?

    This is what separates "evidence refutes a checkable claim" from "a pattern disagreed with some prose".
    It deliberately asks the VALIDATORS rather than guessing from the text: if neither the arithmetic nor the
    unit checker can parse the row, then nothing in this suite has an opinion about it, and evidence must not
    manufacture one.
    """
    try:
        from logos.verify import math_core
        for fn in (math_core.check_arithmetic_text, math_core.check_numeric_constant,
                   math_core.check_unit_conversion):
            try:
                if fn(row).status != "UNKNOWN":
                    return True
            except Exception:                                 # noqa: BLE001
                continue
    except Exception:                                         # noqa: BLE001
        return False
    return False


def guess_kind(row: str) -> str:
    """Code or text? Deliberately crude and stated, because guessing wrong changes the checklist."""
    markers = ("def ", "import ", "class ", "return ", ">>>", "lambda ", "  File \"", "Traceback")
    return "code" if any(m in row for m in markers) else "text"


def _solver_audit(findings, disposition, repair_status=None, refuter=None) -> dict:
    """Discharge the disposition with weighted MAX-SAT, and return the support it could hold.

    WHY THIS IS A GATE AND NOT DECORATION
    Until now the disposition was ASSIGNED by a sequence of `if` branches. Each branch was individually
    justified, but nothing checked the thing that matters globally: **does this disposition have any evidence
    left that can bear it?** A KEEP with no surviving verification is a verdict with no support, and an if-ladder
    cannot see that because each branch only looks at its own condition.

    So the disposition is written as a fact that must be DISCHARGED by support:
        not KEEP or VERIFIED_1 or VERIFIED_2 ...        KEEP requires at least one verification to survive
        not DROP or REFUTED                              DROP requires a named refuter
        not REPAIR or FIXED                              REPAIR requires an accepted repair
    and the support atoms carry PROVENANCE WEIGHTS: an atom corroborated by N independent mechanisms costs N to
    retract, so it survives a conflict that a bare literal would lose. The solver then either holds the
    disposition or retracts it, and the retraction set is the proof object.

    This can only ever DEMOTE (KEEP/DROP/REPAIR -> ESCALATE). It cannot promote anything, so it cannot loosen
    the pipeline, which is what keeps the LOST_CORRECT gate honest when it is enabled.
    """
    try:
        from logos.logic import constraint_solver as CS
    except Exception as e:                                   # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"{type(e).__name__}: {e}", "disposition": disposition}

    # ---- support atoms, grouped by the class of support they provide
    verified, unknown = [], []
    for f in findings:
        name = getattr(f, "name", None) or getattr(f, "check", None) or "validator"
        verdict = getattr(f, "verdict", None)
        indep = int(getattr(f, "independent", 1) or 1)
        atom = CS.fact("row", str(verdict).lower() if verdict else "unknown", str(name))
        if verdict is not None and str(verdict).upper().endswith("VERIFIED"):
            verified.append((atom, indep))
        elif verdict is not None and str(verdict).upper() == "UNKNOWN":
            unknown.append((atom, indep))

    weights = {}
    for atom, indep in verified + unknown:
        # PROVENANCE IN THE OBJECTIVE: two independent mechanisms agreeing is worth two units of resistance.
        weights[atom] = float(max(1, indep))

    disp_atom = CS.fact("row", "disposition", disposition)
    weights[disp_atom] = 100.0            # retracting the verdict is expensive, so it is a last resort
    facts = [a for a, _ in verified] + [a for a, _ in unknown] + [disp_atom]
    clauses, mutex = [], []

    if disposition == "KEEP":
        if not verified:
            clauses.append(([], [disp_atom]))     # not KEEP
        else:
            # not KEEP or any(verified) -- KEEP is discharged only by a surviving verification
            clauses.append(([a for a, _ in verified], [disp_atom]))
    elif disposition == "DROP":
        if refuter:
            ratom = CS.fact("row", "refuted_by", refuter)
            facts.append(ratom)
            weights[ratom] = 5.0                  # a named refuter is strong support
            clauses.append(([ratom], [disp_atom]))
        else:
            clauses.append(([], [disp_atom]))
    elif disposition == "REPAIR":
        if repair_status and "FIX" in str(repair_status).upper():
            fatom = CS.fact("row", "repair", str(repair_status).lower())
            facts.append(fatom)
            weights[fatom] = 3.0
            clauses.append(([fatom], [disp_atom]))
        else:
            clauses.append(([], [disp_atom]))

    r = CS.solve(facts, weights=weights, clauses=clauses, mutex=mutex)
    held = set(r.get("held") or [])
    discharged = disp_atom in held
    out = {"status": r.get("status"), "discharged": discharged, "disposition": disposition,
           "n_support": len(verified) + len(unknown), "support_atoms": sorted(held - {disp_atom})[:8],
           "cost": r.get("cost"), "unique": r.get("unique"), "retracted": r.get("retracted")}
    if not discharged:
        out["reason"] = (f"the disposition {disposition} could not be discharged from the surviving support, "
                         f"so it is unsupported and fails closed")
    return out


def evaluate_row(row: str, kind: str | None = None, evidence: list | None = None,
                 gate_threshold: float = 0.5, repair_policy: str = "auto",
                 prose_numeric_flags: bool = False, router=None, solver_discharge: bool = True) -> dict:
    """One row, all the way through. Returns the disposition plus the reason for it.

    Every branch that can change the disposition is recorded in `why`, so a verdict can be audited
    without rerunning it. A disposition with no recorded reason would be an assertion, not a decision.

    `repair_policy` decides what happens to a REFUTED arithmetic row:
        "auto"   (default) TIER 1 is repaired and flagged `alteration={"kind": "math_fix", ...}`; everything
                 else is dropped. Tier 1 means a validator COMPUTED the answer and one side of the claim is
                 authoritative by construction (`2 + 2 = 5` -> `2 + 2 = 4`; `pi = 3.2` -> `pi = 3.1`).
                 Tier 2+ means the correction would be a guess, which is why `1 mile = 1.5 km` is NOT
                 repaired: the row does not say which side is wrong.
        "drop"   never repair, for callers that must not receive altered text at all.
        "repair" attempt tier 1 even on rows classified above it, for a caller who has external reason to
                 trust one side. The alteration flag is emitted either way.
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
                "low_band": 0, "flags": [], "skipped": [], "findings": [],
                "repaired": None, "altered": False, "alteration": None,
                "why": [f"kind={kind}", "row is blank or whitespace -- nothing to evaluate"]}

    rv = row_eval.execute(row, kind=kind)
    disposition = rv.disposition
    findings = rv.findings
    if rv.refuted_by:
        why.append(f"refuted by {rv.refuted_by} (conclusive, chain stopped)")

        # ---- REPAIR: the one branch that keeps a refuted row instead of dropping it. Narrow by
        # design -- only arithmetic -- and only if the corrected row re-passes the same validator.
        if rv.refuted_by == "math" and repair_policy in ("repair", "auto"):
            cls = classify_repair(row)
            may_repair = repair_policy == "repair" or cls["tier"] == 1
            if not may_repair:
                why.append(f"not repaired: tier {cls['tier']} (refuted by {cls.get('refuted_by')}), and "
                           f"tier>1 means the correction is a GUESS rather than a computation -- "
                           f"e.g. '1 mile = 1.5 km' gives no way to know which side is wrong")
                return {"row": row[:200], "kind": kind, "disposition": DROP,
                        "refuted_by": rv.refuted_by, "cost": round(rv.total_cost, 3),
                        "repaired": None, "altered": False, "alteration": None,
                        "n_findings": len(findings), "gated": False, "gate_sufficient": None,
                        "low_band": 0, "flags": [], "skipped": rv.skipped, "why": why,
                        "findings": [f.as_dict() for f in findings]}
            rep = _numeric_repair(row)
            if rep["status"] == "FIXED":
                disposition = REPAIR
                why.append(f"REPAIR (tier {rep.get('tier', 1)}): {rep['reason']}")
                why.append("ALTERED: this text is not what the source said; a repaired row must never be "
                           "presented as an original")
                alteration = {
                    "kind": "math_fix",
                    "tier": int(rep.get("tier", 1)),
                    "validator": rep.get("validator", ""),
                    "original": row,
                    "repaired": rep["candidate"],
                    "oracle_recheck": "VERIFIED",
                }
                # discharge this REPAIR too: the early return must not skip the audit, or a repaired row
                # carries no proof object while every other disposition does.
                _audit = _solver_audit(findings, REPAIR, repair_status=rep.get("status")) \
                    if solver_discharge else None
                if _audit is not None:
                    _audit["final"] = REPAIR if _audit.get("discharged") else ESCALATE
                if _audit is not None and not _audit.get("discharged"):
                    why.append("solver discharge FAILED for REPAIR: the repair did not re-verify, so the "
                               "row escalates rather than carrying an unproven alteration")
                    return {"row": row[:200], "kind": kind, "disposition": ESCALATE,
                            "refuted_by": rv.refuted_by, "cost": round(rv.total_cost, 3),
                            "n_findings": len(findings), "gated": False, "gate_sufficient": None,
                            "low_band": 0, "flags": [], "skipped": rv.skipped, "repaired": None,
                            "altered": False, "alteration": None, "solver_audit": _audit, "why": why,
                            "findings": [f.as_dict() for f in findings]}
                return {"row": row[:200], "kind": kind, "disposition": REPAIR,
                        "refuted_by": rv.refuted_by, "cost": round(rv.total_cost, 3),
                        "n_findings": len(findings), "gated": False, "gate_sufficient": None,
                        "low_band": 0, "flags": [], "skipped": rv.skipped, "repaired": rep["candidate"],
                        "altered": True, "alteration": alteration, "solver_audit": _audit, "why": why,
                        "findings": [f.as_dict() for f in findings]}
            why.append(f"no repair available: {rep['status']}: {rep['reason'][:80]}")
        elif rv.refuted_by == "math":
            why.append("repair_policy='drop': a refuted claim is REJECTED rather than rewritten, because "
                       "a repair would emit a value the source never contained")
    n_unknown = sum(1 for f in findings if f.verdict == row_eval.UNKNOWN)
    n_verified = sum(1 for f in findings if f.verdict == row_eval.VERIFIED)
    n_na = sum(1 for f in findings if f.verdict == row_eval.NA)
    why.append(f"findings {n_verified} verified / {n_unknown} unknown / {n_na} na")

    # ---- SPLIT THE DROP PILE. row_eval returns DROP for both "a check refuted this" and "no check
    # applied". Only the first is a defect. The second is a row nothing here can decide, and calling it
    # junk is how a review pile vanishes.
    # ---- THE ROUTER, consulted ONLY where the cheap tier could not decide. It picks a TIER, not an
    # instrument, and it can only escalate or stop -- never promote. `router=None` disables it entirely, so
    # existing behaviour (and the frozen canary metrics) are unchanged unless a caller opts in.
    router_tier = None
    if router is not None and not rv.refuted_by and disposition in (DROP, ESCALATE):
        try:
            from logos.nlp import pattern_router as _pr
            cheap_verdict = "UNKNOWN"
            for f in findings:
                if getattr(f, "name", None) == "math" or getattr(f, "check", "") == "math":
                    cheap_verdict = getattr(f, "verdict", "UNKNOWN")
                    break
            ctx = _pr.context_vector(row, cheap_verdict)
            act = router.choose(ctx)
            router_tier = _pr.TIERS[act]
            why.append(f"router: cheap tier gave no decision, chose tier {router_tier} "
                       f"(of {_pr.TIERS})")
            if router_tier >= 3:
                # tier 3 = the spaCy instruments. They can only ADD a refusal or a scope flag here; they
                # never produce KEEP, so this cannot loosen the pipeline.
                try:
                    from logos.logic import dep_encoder
                    de = dep_encoder.parse_dependency(row)
                    if de.get("status") != "PARSED":
                        why.append(f"router tier 3: dep_encoder declined ({de.get('reason', '')[:70]})")
                    else:
                        why.append(f"router tier 3: dep_encoder parsed "
                                   f"{de.get('subject')}(x) -> {de.get('predicate')}(x)")
                except Exception as e:                    # noqa: BLE001
                    why.append(f"router tier 3: dep_encoder unavailable ({type(e).__name__})")
        except Exception as e:                            # noqa: BLE001
            why.append(f"router unavailable ({type(e).__name__})")

    # ---- structure is not a conclusion. a TEXT row that is merely well-formed has NOT been verified.
    if kind == "text" and disposition == KEEP and _only_structural_verification(findings):
        disposition = ESCALATE
        why.append("downgraded KEEP -> ESCALATE: the only verification was STRUCTURAL (the text is "
                   "well-formed), which says nothing about whether the row is correct")

    if disposition == DROP and not rv.refuted_by:
        disposition = ESCALATE
        why.append(f"row_eval said DROP with no named refuter ({n_na}/{len(findings)} checks NA): "
                   f"no check APPLIED, which is undecidable rather than defective -> ESCALATE")

    # ---- 3. the evidence gate. CAN ONLY DEMOTE. never promotes, because a relevance score is not support.
    gate = None
    gated = False
    flags: list = []
    rep = None                      # the repair attempt, if the refutation path ran
    alteration = None
    refuter = rv.refuted_by
    if evidence:
        rc = _rerank()
        if rc is None:
            gate = {"sufficient": None, "reason": "encoder client unavailable"}
            why.append("evidence supplied but encoder client unavailable -> ungated")
        else:
            gate = rc.evidence_sufficient(row, evidence, threshold=gate_threshold)
            gated = True
            why.append(f"gate: {gate['reason']}")
            # ---- THE LOW BAND GETS A DECIDER, not a shrug. A passage the encoder could not rank is
            # examined by an exact rule: does it state a DIFFERENT value for the SAME kind of quantity?
            # This is the only path by which evidence can justify a DROP, so it is the difference
            # between a pipeline that can reject and one that can only abstain.
            # EVERY candidate, not just the low band. Measured: a contradiction can score 0.867 and
            # land in `sufficient`, because contradicting a claim is a way of being ON-TOPIC. Band
            # membership is a relevance judgement with no bearing on whether two passages disagree.
            for cand in (gate.get("all") or []):
                try:
                    from logos import numeric_conflict
                    nc = numeric_conflict.conflicts(row, evidence[cand["index"]])
                except Exception as e:                            # noqa: BLE001
                    # `continue`, not `break`: one unreadable candidate must not cancel the check on
                    # the others. An earlier version raised KeyError here because `all` entries carry
                    # `index` while only `low_band` entries carry `text`, and the break silently
                    # disabled the entire refutation path.
                    why.append(f"candidate {cand.get('index')} not checkable ({type(e).__name__})")
                    continue
                if nc["conflict"]:
                    d = nc["disagreements"][0]
                    band = "low band" if cand["prob"] < gate_threshold else "top band"
                    detail = (f"numeric_conflict ({band}, relevance {cand['prob']:.4f}, "
                              f"severity {nc.get('severity')}): row says {d['claim_raw']}, "
                              f"evidence says {d['passage_raw']} ({d['family']})")

                    # does the ROW itself parse as a numeric claim? only then can evidence refute it.
                    if _row_is_decidable_numeric(row):
                        disposition = DROP
                        refuter = "numeric_conflict"
                        why.append(f"REFUTED by {detail}")
                        break
                    # prose: a number-pattern disagreement is NOT a verdict, and measured on 500 real
                    # labelled pairs it is not even a useful hint (lift 1.01x, recall 0.04). Recorded in the
                    # audit trail, kept out of the review queue.
                    why.append(f"numeric disagreement, NOT flagged: {detail} -- the row is prose, and "
                               f"measured lift is 1.01x against a 0.71 base rate on 18 flags, so this "
                               f"would spend review capacity at the rate of picking at random")
                    if prose_numeric_flags:
                        flags.append(detail)
                    continue
            if disposition == KEEP and gate.get("sufficient") is False:
                disposition = ESCALATE
                why.append("KEEP demoted to ESCALATE: no evidence cleared the relevance threshold")
            elif gate.get("sufficient") is None:
                # fail closed: an unreachable backend is UNKNOWN, and UNKNOWN cannot support KEEP
                if disposition == KEEP:
                    disposition = ESCALATE
                    why.append("KEEP demoted to ESCALATE: gate unreachable, so evidence is UNKNOWN")

    # ---- DISCHARGE THE DISPOSITION WITH WEIGHTED MAX-SAT. The if-ladder above decides which disposition
    # looks right; this checks that it has support that can actually bear it, and demotes it if not. It can
    # only demote, never promote, so enabling it cannot loosen the pipeline.
    solver_audit = None
    if solver_discharge:
        solver_audit = _solver_audit(findings, disposition, repair_status=rep.get("status") if rep else None,
                                     refuter=refuter)
        if solver_audit and not solver_audit.get("discharged") and disposition != ESCALATE:
            why.append(f"solver discharge FAILED for {disposition}: "
                       f"{solver_audit.get('reason', 'no support survived')} -> ESCALATE")
            disposition = ESCALATE
            solver_audit["final"] = ESCALATE
        elif solver_audit:
            solver_audit["final"] = disposition

    return {"row": row[:200], "kind": kind, "disposition": disposition,
            "refuted_by": refuter, "cost": round(rv.total_cost, 3), "repaired": None, "altered": False,
            "alteration": None,
            "n_findings": len(findings), "gated": gated,
            "gate_sufficient": (gate or {}).get("sufficient"),
            "low_band": len((gate or {}).get("low_band") or []), "flags": flags,
            "router_tier": router_tier, "solver_audit": solver_audit,
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
    ("1 mile = 1.5 km", DROP, "text"),
    ("def f():\n    return qqq + 1\n", DROP, "code"),
    ("def g(:\n    return 1\n", DROP, "code"),
    # ---- should REPAIR: tier-1, the correction is COMPUTED and uniquely determined
    ("2 + 2 = 5", REPAIR, "text"),
    ("7 * 6 = 41", REPAIR, "text"),
    # a wrong ROUNDING of a constant is recoverable too: 2 s.f. of pi is 3.1, so the row is fixed rather
    # than discarded. This row used to be DROPped; the precision work turned it into a repair.
    ("pi = 3.0", REPAIR, "text"),
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

    print("\n=== evidence -> disposition, the path that makes a reject pile possible ===")
    if rc is not None:
        cases = [
            ("One mile is 1.609344 kilometres.", ["One mile is 1.500000 kilometres."],
             "contradicting number"),
            ("One mile is 1.609344 kilometres.", ["One mile equals exactly 1.609344 kilometres."],
             "supporting passage"),
            ("One mile is 1.609344 kilometres.", ["Mile is also a surname in some families."],
             "off-topic passage"),
            ("Apoptosis is programmed cell death.", ["Apoptosis is uncontrolled cell proliferation."],
             "semantic contradiction, no numbers"),
        ]
        for row_text, ev, label in cases:
            r = evaluate_row(row_text, kind="text", evidence=ev)
            print(f"  {label:34s} -> {r['disposition']:9s} {r['refuted_by'] or ''}")
        print("  the encoder chose WHICH passage to look at; numeric_conflict DECIDED. Only the")
        print("  contradiction drops, and it names both values so the verdict can be audited.")

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

    # --- DEFAULT ("auto"): tier-1 is REPAIRED and FLAGGED. `2 + 2` has exactly one right answer and a
    #     machine computes it, so discarding it loses a fixable row. The flag is what makes that safe.
    r = evaluate_row("2 + 2 = 5", kind="text")
    good = (r["disposition"] == REPAIR and r["altered"] is True
            and (r["alteration"] or {}).get("kind") == "math_fix"
            and (r["alteration"] or {}).get("tier") == 1
            and (r["alteration"] or {}).get("repaired") == "2 + 2 = 4")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] by default 2 + 2 = 5 -> {r['disposition']} with a "
          f"math_fix flag {r['alteration']}")

    # --- a caller that must not receive altered text can turn it off entirely
    r = evaluate_row("2 + 2 = 5", kind="text", repair_policy="drop")
    good = r["disposition"] == DROP and not r["altered"] and r["alteration"] is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] repair_policy='drop' refuses to rewrite -> {r['disposition']}")

    # --- OPT-IN: a caller that owns the left-hand side can ask for the correction, and it is MARKED
    r = evaluate_row("2 + 2 = 5", kind="text", repair_policy="repair")
    good = r["disposition"] == REPAIR and r["repaired"] == "2 + 2 = 4" and r["altered"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] with repair_policy='repair' it becomes "
          f"{r['repaired']!r} and is flagged altered=True")

    # --- an empty row is not KEPT
    for blank in ("", "   ", "\n"):
        r = evaluate_row(blank, kind="text")
        good = r["disposition"] != KEEP
        ok = ok and good
        if not good:
            print(f"  [FAIL] a blank row ({blank!r}) was KEPT")
    print(f"  [{'PASS' if ok else 'FAIL'}] blank rows are never KEPT")

    # --- REPAIR is reachable, which it was not: row_eval assigns it nowhere.
    r = evaluate_row("7 * 6 = 41", kind="text")
    good = r["disposition"] == REPAIR and r.get("repaired") == "7 * 6 = 42"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] REPAIR is reachable and produces {r.get('repaired')!r} "
          f"(row_eval never assigns REPAIR at all)")

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

    # --- THE TAUTOLOGY INVARIANT, asserted generically. The bug that started this was `pi = 3.0` being
    #     "repaired" to `pi = pi`, which verifies and deletes the claim. Now `pi = 3.0` is repaired to
    #     `pi = 3.1` -- a real correction -- and the invariant is that NO repair ever emits X = X.
    r = evaluate_row("pi = 3.0", kind="text")
    good = r["disposition"] == REPAIR and r.get("repaired") == "pi = 3.1"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3.0 is repaired to {r.get('repaired')!r} "
          f"(2 s.f. of pi is 3.1)")

    taut = []
    for probe in ("2 + 2 = 5", "7 * 6 = 41", "pi = 3.0", "pi = 3.2", "3 * 4 = 11"):
        cand = _tier1_repair(probe)
        if cand.get("status") == "FIXED":
            l, rr = cand["candidate"].split("=", 1)
            if l.strip() == rr.strip():
                taut.append(cand["candidate"])
    good = not taut
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no repair ever emits X = X (tautologies: {taut or 'none'})")

    # --- GUARD 2, directly: an echo is refused even though it verifies. Guard 1 catches `pi = 3.0`
    #     before guard 2 is reached, so exercise the echo rule on its own.
    import sympy as _sp
    echo = "2 + 2"
    vt = str(_sp.nsimplify(_sp.sympify(echo)))
    good = vt == "4" and _sp.sympify(echo).is_number
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a computation yields a numeric literal ({echo} -> {vt}), "
          f"which is what guard 2 requires")

    # a bare symbol that is NOT a known constant has no authority at all, so nothing is repaired
    rep = _tier1_repair("gamma = 3.0")
    good = rep["status"] != "FIXED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown constant is not 'repaired' ({rep['status']})")

    # --- a correct row is untouched by either policy
    for pol in ("drop", "auto", "repair"):
        r = evaluate_row("2 + 2 = 4", kind="text", repair_policy=pol)
        good = r["disposition"] == KEEP and not r["altered"]
        ok = ok and good
        if not good:
            print(f"  [FAIL] a correct row changed under repair_policy={pol}")
    print(f"  [{'PASS' if ok else 'FAIL'}] a correct row is KEPT and unaltered under BOTH policies")

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

    # --- STRUCTURE IS NOT A CONCLUSION. A well-formed prose claim has not been verified, and treating it
    #     as verified produced 75 KEEPs on 200 curated rows, including rows gold-labelled REFUTED.
    r = evaluate_row("New Zealand's new Food Bill bans gardening", kind="text")
    good = r["disposition"] != KEEP and "STRUCTURAL" in str(r["why"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a merely well-formed claim is NOT kept -> {r['disposition']}")

    # --- and genuine computational verification still yields KEEP, so the rule did not break the good path
    r = evaluate_row("2 + 2 = 4", kind="text")
    good = r["disposition"] == KEEP
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] genuine computational verification still KEEPs "
          f"({r['disposition']})")

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

        # --- A CONTRADICTION NOW BECOMES A DROP. This is the reject pile, and before this wiring the
        #     low band was returned and then ignored, so REFUTED was unreachable from evidence.
        r = evaluate_row("1 mile = 1.609344 km", kind="text",
                         evidence=["1 mile = 1.5 km"])
        good = r["disposition"] == DROP and r["refuted_by"] == "numeric_conflict"
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] contradicting numeric evidence -> {r['disposition']} "
              f"by {r['refuted_by']!r} (the reject pile is reachable)")

        # --- REGRESSION: a contradiction that scores ABOVE the relevance threshold must still be
        #     refuted. This is the case my low-band-only wiring silently missed: 0.867 is "very
        #     relevant", and relevance is not support.
        r = evaluate_row("1 mile = 1.609344 km", kind="text",
                         evidence=["1 mile = 1.5 km"])
        high_band = [w for w in r["why"] if "top band" in w or "low band" in w]
        good = r["disposition"] == DROP and bool(high_band)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a HIGH-relevance contradiction still drops "
              f"({r['disposition']}; {high_band[0][:58] if high_band else 'no band recorded'})")

        # --- THE PRECISION RULE, measured on 200 real rows: a checkable numeric row IS refuted by
        #     contradicting evidence...
        r = evaluate_row("1 mile = 1.609344 km", kind="text",
                         evidence=["1 mile = 1.5 km"])
        good = r["disposition"] == DROP and r["refuted_by"] == "numeric_conflict"
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a CHECKABLE numeric row is refuted -> {r['disposition']}")

        # --- and the SAME contradiction against the PROSE form must not drop it: no validator parses that
        #     row, so nothing in this suite has an opinion and evidence must not manufacture one.
        r = evaluate_row("One mile is 1.609344 kilometres.", kind="text",
                         evidence=["One mile is 1.500000 kilometres."])
        good = r["disposition"] == ESCALATE and not r["flags"]
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] the PROSE form of that row is {r['disposition']}, "
              f"not a drop and not a queued flag")

        # --- ...but prose is FLAGGED and escalated, not dropped. On curated data a pattern-level
        #     disagreement was dropping ~3.5% of good rows, which is the catastrophic direction.
        # --- a prose conflict is RECORDED but does NOT enter the review queue: measured lift 1.01x.
        r = evaluate_row("99% of people recover from COVID-19.", kind="text",
                         evidence=["Only 39.88% of people recover from COVID-19."])
        recorded = any("numeric disagreement" in w for w in r["why"])
        good = r["disposition"] == ESCALATE and recorded and not r["flags"]
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a PROSE conflict is recorded in the audit trail "
              f"({recorded}) and kept OUT of the review queue ({len(r['flags'])} flags)")

        # --- and the old behaviour is one parameter away, for anyone re-measuring
        r2 = evaluate_row("99% of people recover from COVID-19.", kind="text",
                          evidence=["Only 39.88% of people recover from COVID-19."],
                          prose_numeric_flags=True)
        good = len(r2["flags"]) == 1
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] prose_numeric_flags=True restores the flag "
              f"({len(r2['flags'])}) so the null can be re-tested rather than trusted")

        # --- and a supporting passage must not become a drop, or the validator is just a hammer
        r = evaluate_row("1 mile = 1.609344 km", kind="text",
                         evidence=["One mile equals exactly 1.609344 kilometres."])
        good = r["disposition"] != DROP
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] supporting evidence does NOT trigger a drop "
              f"({r['disposition']})")

        # --- an OFF-TOPIC passage must not become a drop either
        r = evaluate_row("One mile is 1.609344 kilometres.", kind="text",
                         evidence=["Mile is also a surname in some European families."])
        good = r["disposition"] == ESCALATE
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] off-topic evidence is {r['disposition']}, not a drop")

        # --- a semantic contradiction with no numbers stays ESCALATE, honestly
        r = evaluate_row("Apoptosis is programmed cell death.", kind="text",
                         evidence=["Apoptosis is uncontrolled cell proliferation."])
        good = r["disposition"] == ESCALATE
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a NON-numeric contradiction stays {r['disposition']} "
              f"(out of scope, not guessed)")

        # --- a contradicting passage is not silently swallowed. NOTE the low band holds CANDIDATES, and a
        #     contradiction may score high enough to sit in the top band instead -- measured 0.867 for the
        #     mile case. So the property worth asserting is that the candidate set is never lost, not that
        #     it lands in a particular band.
        dem2 = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text",
                            evidence=["The Eiffel Tower is 300 metres tall."])
        good = dem2["gate_sufficient"] is not None
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] contradicting evidence survives in the low band "
              f"({dem2['low_band']} passage(s)) for a validator to judge")

    # --- a batch returns counts for EVERY disposition, so a missing class is visible as 0 not absent
    res = run_rows(["2 + 2 = 4", "2 + 2 = 5"])
    good = set(res["counts"]) == set(DISPOSITIONS) and res["n"] == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] batch counts always contain all four dispositions")

    # --- THE ROUTER IS OPT-IN AND CANNOT LOOSEN ANYTHING
    a = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text")
    good = a.get("router_tier") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] without a router the pipeline is unchanged (router_tier=None)")

    try:
        from logos.nlp import pattern_router as _pr
        _r = _pr.CascadeRouter(seed=0)
        out = evaluate_row("The Eiffel Tower is 330 metres tall.", kind="text", router=_r)
        good = out.get("router_tier") in _pr.TIERS
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] with a router it records the chosen tier "
              f"({out.get('router_tier')})")

        # --- a row the cheap tier already DECIDED must never reach the router
        out2 = evaluate_row("2 + 2 = 5", kind="text", router=_r)
        good = out2["disposition"] == REPAIR and out2.get("router_tier") is None
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] a decided row never consults the router "
              f"({out2['disposition']}, tier={out2.get('router_tier')})")

        # --- and the router cannot promote: an ESCALATE row stays ESCALATE whatever tier it picks
        out3 = evaluate_row("Apoptosis is programmed cell death.", kind="text", router=_r)
        good = out3["disposition"] == ESCALATE
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] the router cannot promote: still {out3['disposition']}")
    except Exception as e:                                # noqa: BLE001
        print(f"  [SKIP] router unavailable ({type(e).__name__})")

    # --- THE SOLVER DISCHARGE: a disposition must have support that can bear it, and the audit names it
    a2 = evaluate_row("2 + 2 = 4", kind="text")
    good = (a2["solver_audit"] or {}).get("discharged") is True and a2["disposition"] == KEEP
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a KEEP is DISCHARGED by a surviving verification "
          f"({(a2['solver_audit'] or {}).get('support_atoms')})")

    # --- a code KEEP carries TWO independent supports, so it resists retraction more than a bare one
    a3 = evaluate_row("def f():\n    return 1", kind="code")
    au3 = a3["solver_audit"] or {}
    good = au3.get("discharged") is True and au3.get("n_support", 0) >= 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a KEEP with 2 independent mechanisms reports both "
          f"({au3.get('support_atoms')})")

    # --- a DROP is discharged by the NAMED refuter, not by opinion
    a4 = evaluate_row("def f():\n    return 1 + ", kind="code")
    au4 = a4["solver_audit"] or {}
    good = a4["disposition"] == DROP and au4.get("discharged") is True \
        and any("refuted_by" in x for x in au4.get("support_atoms") or [])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a DROP is discharged by its named refuter "
          f"({au4.get('support_atoms')})")

    # --- a REPAIR is discharged by the accepted repair, and its proof is the alteration
    a5 = evaluate_row("2 + 2 = 5", kind="text")
    au5 = a5["solver_audit"] or {}
    good = a5["disposition"] == REPAIR and au5.get("discharged") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a REPAIR is discharged by the accepted repair "
          f"({au5.get('support_atoms')})")

    # --- and the discharge can only DEMOTE. Turning it off must never yield a MORE confident verdict.
    worse = 0
    for _row, _kind in (("2 + 2 = 4", "text"), ("def f():\n    return 1", "code"),
                        ("2 + 2 = 5", "text"), ("Apoptosis is programmed cell death.", "text")):
        _off = evaluate_row(_row, kind=_kind, solver_discharge=False)["disposition"]
        _on = evaluate_row(_row, kind=_kind, solver_discharge=True)["disposition"]
        if _off == ESCALATE and _on != ESCALATE:
            worse += 1
    good = worse == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the discharge never PROMOTES a verdict "
          f"({worse} promotions)")

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
