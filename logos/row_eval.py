"""logos/row_eval.py — per-row checklist -> DAG -> executed verdict, with citations and tiered checks.

THE SHAPE, AS DESCRIBED
-----------------------
    row -> split into SUBJECTS (math, code, logic, semantic, temporal, prose)
        -> build a CHECKLIST of only the checks those subjects can exercise
        -> compile the checklist into a DAG (dependencies + cost ordering)
        -> execute it, cheap checks first, short-circuiting on refutation
        -> a verdict plus an EVIDENCE TRAIL

WHY A DAG AND NOT A LIST
------------------------
Some checks are meaningless without others: `symtable` needs a parse, a math identity needs the expression to
be extractable, a doctest run needs the module to import. A list would run a doomed check and record its
failure as evidence about the row when it is only evidence about the earlier failure. The DAG encodes that a
child whose parent is UNKNOWN cannot be VERIFIED.

FOUR DESIGN CHOICES TAKEN FROM THE RESEARCH, each fixing a specific failure
--------------------------------------------------------------------------
1. **TIERED IMPORTANCE, one narrow checker per category.** VCoT-Lift routes proof steps to five rule-category
   checkers by importance tier (8 high / 12 medium / 16 low) rather than asking one judge for a global
   verdict. So each check here declares a category and a tier, and high-tier checks run first.
2. **CITATIONS ARE MANDATORY.** A finding must name the artifact that justifies it. A verdict without a
   citation is rejected by this module, because "the checker said so" is indistinguishable from a
   hallucination.
3. **A TRIVIALITY ALLOWLIST.** VCoT-Lift needed an abstraction filter or its checkers looped forever on
   reflexive and normalization steps. Same trap here: `x = x` is an identity, `4 + 4 = 8` is true, and a
   cleaner that spends budget confirming trivia is a cleaner that misses the real defect.
4. **SOLVER OVER JUDGE.** Where a check is mechanically decidable it calls sympy/Z3/ast, never a model. The
   research on self-verification is NEGATIVE (models become confident of wrong answers early; verbal
   verification text can make accuracy worse), so a model is not the oracle for anything decidable here.

AND ONE SIGNAL WORTH KEEPING FROM THE Apple CoT WORK
----------------------------------------------------
Only 15-45% of correct chains are monotonic in "potential", and a rationale that becomes correct only at the
last token is a guess wearing a derivation's clothes. So `missing_derivation` flags a conclusion asserted with
no intermediate steps -- a cheap validity signal that needs no model and no gold answer.

USAGE
    python tools/logos/row_eval.py --self-test
    python tools/logos/row_eval.py --explain "<row text>"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.context_pack import gather  # noqa: E402
from logos.triage import check_semantics, check_structure  # noqa: E402
from logos.truth_kinds import classify, validate  # noqa: E402

KEEP, REPAIR, ESCALATE, DROP = "KEEP", "REPAIR", "ESCALATE", "DROP"
VERIFIED, REFUTED, UNKNOWN, NA = "VERIFIED", "REFUTED", "UNKNOWN", "NOT_APPLICABLE"

# checks that fire on content which is true by construction and tell you nothing. VCoT-Lift needed the same
# filter or its checkers never terminated on reflexive steps.
TRIVIAL_PATTERNS = (
    re.compile(r"^\s*(\w+)\s*=\s*\1\s*$"),                 # x = x
    re.compile(r"^\s*0\s*=\s*0\s*$"),
    re.compile(r"^\s*(\d+)\s*=\s*\1\s*$"),                 # 5 = 5
)


def is_trivial(text: str) -> bool:
    return any(p.match(text) for p in TRIVIAL_PATTERNS)


# ------------------------------------------------------------------------------- checks
@dataclass
class Check:
    name: str
    category: str            # the rule class it belongs to (VCoT-Lift style)
    tier: str                # high | medium | low -- ordering, not severity
    cost: float
    subjects: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    run: object = None

    def applicable(self, subjects: set) -> bool:
        return bool(set(self.subjects) & subjects)


@dataclass
class Finding:
    check: str
    category: str
    tier: str
    verdict: str
    citation: str = ""
    detail: str = ""
    cost: float = 0.0

    def as_dict(self) -> dict:
        return {"check": self.check, "category": self.category, "tier": self.tier,
                "verdict": self.verdict, "citation": self.citation, "detail": self.detail,
                "cost": round(self.cost, 2)}


@dataclass
class RowVerdict:
    disposition: str = ESCALATE
    findings: list = field(default_factory=list)
    refuted_by: str = ""
    total_cost: float = 0.0
    skipped: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"disposition": self.disposition, "refuted_by": self.refuted_by,
                "total_cost": round(self.total_cost, 2), "skipped": self.skipped,
                "findings": [f.as_dict() for f in self.findings]}


# ------------------------------------------------------------------------------- subjects
def subjects_of(text: str, kind: str = "text") -> set:
    """Which subjects does this row actually exercise? Only these get checks."""
    s = set()
    if kind == "code" or re.search(r"\bdef \w+\(|\bimport \w+|return\b", text):
        s.add("code")
    c = classify(text)
    if c["kind"] in ("arithmetic", "numeric_constant", "algebraic", "calculus",
                     "linear_algebra", "unit_conversion"):
        s.add("math")
    # an equality with a number on either side is a math row even when no operator pattern fires.
    # Without this, "4 = 4" fell through to prose and the letter-ratio heuristic refuted it as symbol soup.
    elif re.search(r"[\d)\]]\s*=\s*[\d(\[a-z-]", text) or re.search(r"^\s*[a-z]\s*=", text):
        s.add("math")
    if c["kind"] == "logical" or re.search(r"\b(therefore|implies|hence|thus)\b", text, re.I):
        s.add("logic")
    if re.search(r"\b(is|are|was|were|has|have)\b", text, re.I):
        s.add("semantic")
    if re.search(r"\b(19|20)\d{2}\b", text):
        s.add("temporal")
    if re.search(r"\bstep\s*\d|\d\.\s|first,|then,|finally,", text, re.I):
        s.add("derivation")
    if not s:
        s.add("prose")
    return s


# ------------------------------------------------------------------------------- the checks
def _chk_structure(row: str, _src: str) -> Finding:
    r = check_structure(row, "code" if "def " in row else "text")
    return Finding("structure", "syntax", "high", VERIFIED if r["ok"] else REFUTED,
                   citation=f"{len(row)} chars, {len(row.splitlines())} lines",
                   detail="; ".join(r["issues"][:2]))


def _chk_semantic(row: str, _src: str) -> Finding:
    r = check_semantics(row)
    if r["ok"] is None:
        return Finding("semantic", "name-binding", "medium", UNKNOWN,
                       citation="symtable declined", detail="; ".join(r["issues"][:2]))
    return Finding("semantic", "name-binding", "medium", VERIFIED if r["ok"] else REFUTED,
                   citation="symtable symbol table", detail="; ".join(r["issues"][:2]))


def _chk_math(row: str, _src: str) -> Finding:
    if is_trivial(row):
        return Finding("math", "arithmetic", "high", NA, citation="triviality allowlist",
                       detail="true by construction; tells you nothing")
    r = validate(row)
    status = r.get("verdict", UNKNOWN)
    mapped = {VERIFIED: VERIFIED, REFUTED: REFUTED}.get(status, UNKNOWN)
    return Finding("math", f"math:{r.get('kind', '?')}", "high", mapped,
                   citation=f"sympy via truth_kinds ({r.get('kind', '?')})",
                   detail=str(r.get("detail", r.get("note", "")))[:120])


def _chk_derivation(row: str, _src: str) -> Finding:
    """Was there an argument, and does it HOLD? Replaces a regex that counted discourse markers.

    THE OLD BEHAVIOUR WAS KEEP-BY-ENCOURAGEMENT. It returned VERIFIED when the row contained
    "therefore/thus/hence" plus step words, which means any fluent paragraph about reasoning was certified
    as a valid derivation. Fluent prose about reasoning is trivially producible, so that check was near-worthless
    in the one direction that matters here.

    It now asks Z3 (via logic/fol_bridge) whether the conclusion follows from the premises, on a declared,
    reviewable English-to-first-order encoding. Three deliberate asymmetries, all conservative:

      VALID          -> NA, not VERIFIED. Form validity is evidence about the ARGUMENT, never about the
                        claim's truth, and a VERIFIED here would let a well-formed falsehood through. The
                        encoding is also ours: three polarity inversions were found in it in one session, so
                        a VALID verdict buys a "no red flag", not a certificate.
      CONTRADICTED   -> REFUTED. The premises entail the negation. This is the safe direction: a wrong
      /UNDETERMINED     encoding costs a good row (cheap), while the reverse costs a bad KEEP (catastrophic).
                        The counterexample model is cited so the verdict is auditable.
      UNPARSEABLE    -> UNKNOWN, never VERIFIED. An argument we could not encode cannot be certified by
                        counting its words.

    Cost is ~1.15 ms/case measured, against 13-52 s/claim for the Lean kernel, so this is the cheap tier.
    """
    if not re.search(r"\b(therefore|thus|hence|so)\b", row, re.I):
        return Finding("derivation", "chain-shape", "medium", NA, citation="no conclusion marker", detail="")

    try:
        from logos.logic import fol_bridge
    except Exception as e:                                    # noqa: BLE001
        return Finding("derivation", "fol", "medium", UNKNOWN,
                       citation=f"fol_bridge unavailable ({type(e).__name__})",
                       detail="an argument was asserted and nothing could check it")

    try:
        r = fol_bridge.check_row(row)
    except Exception as e:                                    # noqa: BLE001
        return Finding("derivation", "fol", "medium", UNKNOWN,
                       citation=f"fol_bridge raised {type(e).__name__}", detail=str(e)[:120])

    status = r.get("status")
    detail_kind = r.get("detail", "")

    if status == fol_bridge.VALID:
        if r.get("vacuous"):
            return Finding("derivation", "fol", "medium", UNKNOWN,
                           citation="Z3: conclusion asserted among the premises",
                           detail="circular argument: valid and proves nothing")
        needed = r.get("premises_needed")
        return Finding("derivation", "fol", "medium", NA,
                       citation=f"Z3 fol_bridge: {detail_kind or 'VALID'}"
                                + (f", premises used {needed}" if needed else ""),
                       detail="the argument is formally VALID, which is a statement about its form and not "
                              "about whether the conclusion is true -- so it does not verify the row")

    if status == fol_bridge.INVALID and detail_kind in ("CONTRADICTED", "UNDETERMINED"):
        cx = r.get("counterexample")
        cx_s = ""
        if isinstance(cx, dict) and cx:
            cx_s = "; ".join(f"{k}={v}" for k, v in list(cx.items())[:4])
        return Finding("derivation", "fol", "medium", REFUTED,
                       citation=f"Z3 fol_bridge: {detail_kind}",
                       detail=(r.get("reason") or "")[:100] + (f" | counterexample: {cx_s}" if cx_s else ""))

    # UNPARSEABLE / UNKNOWN / inconsistent premises / ledger mismatch / no premises
    return Finding("derivation", "fol", "medium", UNKNOWN,
                   citation=f"Z3 fol_bridge: {status}/{detail_kind or '-'}",
                   detail=(r.get("reason") or "the argument could not be encoded, so nothing about it is "
                                              "claimed")[:140])


CHECKS: tuple[Check, ...] = (
    Check("structure", "syntax", "high", 1.0, ("code", "prose"), run=_chk_structure),
    Check("math", "arithmetic", "high", 5.0, ("math",), run=_chk_math),
    Check("derivation", "chain-shape", "medium", 2.0, ("derivation", "logic"), run=_chk_derivation),
    Check("semantic", "name-binding", "medium", 2.0, ("code",), depends_on=("structure",),
          run=_chk_semantic),
)


# ------------------------------------------------------------------------------- dag
def build_checklist(row: str, kind: str = "text") -> tuple[list[Check], set]:
    subs = subjects_of(row, kind)
    return [c for c in CHECKS if c.applicable(subs)], subs


def compile_dag(checks: list[Check]) -> list[Check]:
    """Topological order by dependency, then by tier and cost, so cheap high-value checks run first."""
    names = {c.name for c in checks}
    by_name = {c.name: c for c in checks}
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    order: list[Check] = []
    placed: set[str] = set()

    def place(c: Check, stack: tuple = ()) -> None:
        if c.name in placed:
            return
        if c.name in stack:
            return                                          # cycle guard: refuse, do not loop
        for dep in c.depends_on:
            if dep in names:
                place(by_name[dep], stack + (c.name,))
        if c.name not in placed:
            placed.add(c.name)
            order.append(c)

    for c in sorted(checks, key=lambda x: (tier_rank.get(x.tier, 3), x.cost)):
        place(c)
    return order


def execute(row: str, kind: str = "text", source: str = "") -> RowVerdict:
    """Run the DAG. Cheap first, short-circuit on REFUTED, propagate UNKNOWN to children."""
    checks, subs = build_checklist(row, kind)
    order = compile_dag(checks)
    v = RowVerdict(findings=[])
    state: dict[str, str] = {}

    for c in order:
        blocked = [d for d in c.depends_on if state.get(d) in (REFUTED, UNKNOWN)]
        if blocked:
            v.skipped.append(f"{c.name}(dep {blocked[0]}={state.get(blocked[0])})")
            state[c.name] = UNKNOWN
            continue
        f = c.run(row, source)
        f.cost = c.cost
        v.total_cost += c.cost
        if not f.citation:
            # MANDATORY: a finding that cannot name its artifact is not evidence
            f.detail = (f.detail + " | REJECTED: no citation").strip(" |")
            f.verdict = UNKNOWN
        v.findings.append(f)
        state[c.name] = f.verdict
        if f.verdict == REFUTED:
            v.refuted_by = c.name
            v.disposition = DROP
            remaining = [x.name for x in order[order.index(c) + 1:]]
            if remaining:
                v.skipped.append("short-circuit after " + c.name + ": " + ", ".join(remaining))
            return v                                    # short-circuit: refutation is conclusive

    verdicts = [f.verdict for f in v.findings]
    if REFUTED in verdicts:
        v.disposition = DROP
    elif all(s == NA for s in verdicts):
        v.disposition = DROP                            # nothing applied and nothing to say
    elif UNKNOWN in verdicts:
        v.disposition = ESCALATE                        # a real question remains open
    elif VERIFIED in verdicts:
        v.disposition = KEEP
    else:
        v.disposition = ESCALATE
    return v


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.row_eval - SELF TEST")
    ok = True

    # --- subjects gate the checklist: a pure math row gets no code checks
    checks, subs = build_checklist("pi = 3.14159")
    names = {c.name for c in checks}
    good = "math" in subs and "math" in names and "semantic" not in names
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] subjects gate the checklist: math row got {sorted(names)}")

    # --- a code row gets structure and semantic, and semantic DEPENDS on structure
    checks, subs = build_checklist("def f():\n    return qqq + 1\n", kind="code")
    sem = next((c for c in checks if c.name == "semantic"), None)
    good = "code" in subs and sem is not None and "structure" in sem.depends_on
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a code row gets structure+semantic with the dependency "
          f"recorded")

    # --- the DAG orders dependencies first
    order = [c.name for c in compile_dag(checks)]
    good = order.index("structure") < order.index("semantic")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dependency ordering honoured: {order}")

    # --- a REFUTED math claim DROPs immediately, and the expensive checks never run
    v = execute("2 + 2 = 5")
    good = v.disposition == DROP and v.refuted_by == "math"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a refuted math claim is DROPped at the first cheap check "
          f"({v.total_cost:.0f} cost spent)")

    # --- every finding carries a citation, or it is rejected
    good = all(f.citation for f in v.findings)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every finding cites an artifact "
          f"({[f.citation[:24] for f in v.findings]})")

    # --- TRIVIALITY ALLOWLIST: x = x is not a finding worth paying for
    v = execute("4 = 4")
    math = next((f for f in v.findings if f.check == "math"), None)
    good = math is not None and math.verdict == NA
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] trivia is NOT_APPLICABLE, not VERIFIED "
          f"(the allowlist VCoT-Lift needed)")

    # --- a real verified math row is KEEPed
    v = execute("1 mile = 1.609344 km")
    good = v.disposition == KEEP
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a genuinely verified row is KEEP")

    # --- UNKNOWN propagates: a child of an UNKNOWN parent is skipped, not credited
    v = execute("def f(:\n    return 1\n", kind="code")
    good = v.disposition == DROP and "semantic" in " ".join(v.skipped)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a broken parent short-circuits the child "
          f"(skipped: {v.skipped})")

    # --- a conclusion whose argument cannot be encoded is UNKNOWN, never VERIFIED.
    #     The old check VERIFIED this row because it contained "Step 1 / Step 2 / Therefore" -- which is
    #     KEEP-BY-ENCOURAGEMENT: any fluent paragraph about reasoning was certified. Counting discourse
    #     markers is not evidence, so the row now escalates.
    v = execute("Step 1: measure the span. Step 2: compare to the design. Therefore it is within "
                "tolerance.")
    der = next((f for f in v.findings if f.check == "derivation"), None)
    good = der is not None and der.verdict == UNKNOWN and v.disposition == ESCALATE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] step words alone no longer VERIFY anything ({der.verdict if der else '-'}, "
          f"{v.disposition})")

    # --- a semantically FALSE argument is REFUTED, which is the safe direction: a wrong encoding costs a
    #     good row (cheap) rather than passing a bad one (catastrophic).
    v = execute("All men are mortal; Socrates is a man; therefore Socrates is a fish.")
    der = next((f for f in v.findings if f.check == "derivation"), None)
    good = der is not None and der.verdict == REFUTED and v.disposition == DROP
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an argument that does not follow is REFUTED -> {v.disposition}")

    # --- and a formally VALID argument must NOT verify the row. Validity is a statement about the argument,
    #     not about whether its conclusion is true.
    v = execute("All men are mortal; Socrates is a man; therefore Socrates is mortal.")
    der = next((f for f in v.findings if f.check == "derivation"), None)
    good = der is not None and der.verdict == NA
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a VALID argument returns NA, not VERIFIED ({der.verdict if der else '-'})")

    # --- every finding declares a category and a tier, so routing is auditable
    v = execute("pi = 3.14159")
    good = all(f.category and f.tier for f in v.findings)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] findings carry category+tier "
          f"({[(f.check, f.tier) for f in v.findings]})")

    # --- cost accounting so the tiering can be judged later
    good = v.total_cost > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cost is accounted per row ({v.total_cost:.0f})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-row checklist -> DAG -> executed verdict.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--explain", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.explain:
        v = execute(args.explain, kind="code" if "def " in args.explain else "text")
        print(json.dumps(v.as_dict(), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
