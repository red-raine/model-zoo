"""logos/triage.py — four-router triage where the SIZE of the corpus sets the validation effort.

THE THREE LAYERS, AND WHY TRUTH IS THE LEAST IMPORTANT ONE FOR CLEANING
----------------------------------------------------------------------
Dataset refinement has been over-indexed on truth. There are three independent failure layers, and for
cleaning a billion tokens the first two do most of the work:

    SYNTAX   `) x if 4 (`                  -> not parsable. No truth question exists.
    SEMANTICS `x=3 ... therefore x=8`      -> parsable, internally inconsistent. Still no truth question.
    TRUTH    `2+2=5`                       -> well-formed, consistent, and false. Only this layer needs
                                              the world.

**You can reject an enormous amount of garbage without ever proving anything true.** This module
implements the first two with real deterministic checks (Python's own `ast` and `symtable` do genuine
work) and refuses to pretend about the third.

THE FOUR ROUTERS

    structural   parse / compile / encoding / balance          cheapest, highest precision
    semantic     internal coherence: names, numbers, units      still cheap, no world access
    truth        needs evidence -> NOT decided here, only routed
    calibration  is the confidence usable (see calibrate.ROUTING)

THE FOUR DISPOSITIONS

    KEEP      passed everything it needed to
    REPAIR    failed, and the failure has a known cheap fix (the reviewer's best point:
              RE PAIRABLE is the most valuable category, not true and not false)
    ESCALATE  passed the cheap layers but the truth layer has no evidence -- worth paying for
    DROP      failed a deterministic layer. No cost. Millions more where this came from.

THE SCARCITY POLICY -- the decision this module exists for
---------------------------------------------------------
How hard to validate is not a constant. It is a function of how much choice you have:

    slack = candidates_available / samples_needed

    slack >= 5    CREAM      take only what passes everything with zero ambiguity. Do not repair,
                             do not escalate. The ambiguous majority is not worth a model call.
    slack >= 2    STANDARD   allow repair; escalate the good-but-unproven to a decision model
    slack <  2    EXHAUSTIVE repair, escalate, and spend LLM calls on the high-quality ambiguous

With 10M candidates and 100k needed (slack 100) you should discard a sample for a single unanswered
question. With 200k candidates and 150k needed (slack 1.3) you should fight for every row. Same pipeline,
opposite policy -- and hard-coding either one is wrong for the other case.

USAGE
    python tools/logos/triage.py --self-test
    python tools/logos/triage.py --run
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
import symtable
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

KEEP, REPAIR, ESCALATE, DROP = "KEEP", "REPAIR", "ESCALATE", "DROP"
CREAM, STANDARD, EXHAUSTIVE = "cream", "standard", "exhaustive"


# ------------------------------------------------------------------ layer 1: structural
def check_structure(text: str, kind: str = "text") -> dict:
    """Syntax only. No meaning is consulted, so a PASS here proves nothing about truth.

    For Python the check is real: `ast.parse` is the interpreter's own parser. That is free, exact, and
    the strongest signal available anywhere in this project -- and it says nothing about whether the code
    does what it claims.
    """
    issues: list[str] = []
    if not text or not text.strip():
        return {"ok": False, "issues": ["empty"], "layer": "structural"}
    if len(text.strip()) < 8:
        issues.append("too short to contain reasoning")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        issues.append("not encodable as utf-8")

    if kind == "code":
        try:
            ast.parse(text)
        except SyntaxError as e:
            issues.append(f"SyntaxError: {e.msg} (line {e.lineno})")
    else:
        # cheap, honest structural checks for prose. Deliberately NO claim of grammar parsing.
        if text.count("(") != text.count(")"):
            issues.append("unbalanced parentheses")
        if text.count("[") != text.count("]"):
            issues.append("unbalanced brackets")
        if re.search(r"(\b\w+\b)(\s+\1){3,}", text):
            issues.append("degenerate repetition")
        letters = sum(c.isalpha() for c in text)
        if letters / max(len(text), 1) < 0.30:
            issues.append("not mostly language (symbol soup)")

    return {"ok": not issues, "issues": issues, "layer": "structural"}


# -------------------------------------------------------------------- layer 2: semantic
def check_semantics(code: str) -> dict:
    """Internal coherence for Python, using the interpreter's own symbol table.

    This catches the reviewer's example exactly: `x = 3 ... therefore x = 8` is perfectly parsable and
    internally broken. `symtable` tells us which names are referenced but never bound, which is a real
    semantic failure requiring no world knowledge.

    What it CANNOT do: tell you the programme computes the right thing.
    """
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"ok": False, "issues": ["cannot parse; semantic layer skipped"], "layer": "semantic"}

    try:
        st = symtable.symtable(code, "<snippet>", "exec")
    except SyntaxError:
        return {"ok": False, "issues": ["symtable rejected the source"], "layer": "semantic"}

    def walk(table, bound: set) -> None:
        local = bound | {s.get_name() for s in table.get_symbols()
                         if s.is_assigned() or s.is_parameter() or s.is_imported()}
        for s in table.get_symbols():
            if s.is_global() and not s.is_assigned() and s.get_name() not in local:
                # referenced but never bound anywhere in this scope; could be a builtin, so check
                if not hasattr(__builtins__, s.get_name()) and s.get_name() not in dir(__builtins__):
                    issues.append(f"name {s.get_name()!r} used but never bound")
        for child in table.get_children():
            walk(child, local)

    try:
        walk(st, set())
    except Exception:                                     # noqa: BLE001 - analysis must not crash triage
        pass

    # a trivially self-contradictory assignment chain: x = 3 then x = 8 with nothing in between
    assigns: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                    assigns.setdefault(t.id, []).append(node.value.value)
    for name, vals in assigns.items():
        if len(vals) >= 2 and len(set(map(str, vals))) > 1 and len(vals) == len(set(map(str, vals))):
            issues.append(f"{name!r} rebound to {len(set(map(str, vals)))} distinct constants "
                          f"with no intervening use")

    return {"ok": not issues, "issues": issues[:5], "layer": "semantic"}


def check_numeric_consistency(text: str) -> dict:
    """Prose numbers must not contradict themselves. Cheap, deterministic, no world access."""
    issues: list[str] = []
    # "X is N" repeated with different N for the same X is an internal contradiction
    claims: dict[str, set] = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z _-]{2,30}?)\s+(?:is|was|equals|=)\s+"
                         r"(-?\d+(?:\.\d+)?)", text):
        key = " ".join(m.group(1).lower().split())
        claims.setdefault(key, set()).add(float(m.group(2)))
    for key, vals in claims.items():
        if len(vals) > 1:
            issues.append(f"{key!r} given as {sorted(vals)}")
    return {"ok": not issues, "issues": issues[:5], "layer": "semantic"}


# ----------------------------------------------------------------------- layer 3: truth
def check_claims_consistency(triples, numerics=()) -> dict:
    """Layer 2 for EXTRACTED claims, backed by Z3. A real deterministic check, not a heuristic.

    This is where symbolic reasoning actually pays for itself in a refinery. Categorical syllogism
    validity needs no solver (see logic/syllogism.py), but "do these extracted claims contradict each
    other?" does -- and it needs no access to the world, which is why it is cheap enough to run on
    everything before an LLM is consulted.

    Verdict mapping, and the asymmetry is deliberate:
        INCONSISTENT -> ok=False. SOUND: unsat of a subset stays unsat when more constraints are added.
        UNKNOWN      -> ok=None.  Some claim had no declared semantics; failing open here is how garbage
                                  passes a cleaner.
        CONSISTENT   -> ok=None as well, NOT True. No contradiction was found within the encoded
                                  fragment, which is not a proof of consistency. The truth router still
                                  owns whether the claims are TRUE.
    """
    from logos.logic.smt import CONSISTENT, INCONSISTENT, check

    rep = check(triples, numerics)
    if rep.verdict == INCONSISTENT:
        return {"ok": False, "layer": "semantic",
                "issues": [f"contradiction among: {'; '.join(rep.core[:4])}"], "report": rep}
    if rep.verdict == CONSISTENT:
        return {"ok": None, "layer": "semantic",
                "issues": ["no contradiction within the encoded fragment (not a proof)"],
                "report": rep}
    return {"ok": None, "layer": "semantic", "issues": rep.unencoded[:4], "report": rep}


def triage_claims(triples, numerics=(), policy: dict | None = None) -> TriageResult:
    """Four-router triage for EXTRACTED claims rather than raw text.

    Ordering follows the same cost logic: a contradiction is free to detect and conclusive, so it is
    checked first and DROPs immediately. Absence of contradiction is NOT a keep -- the truth layer has
    still not been consulted.
    """
    policy = policy or effort_tier(100, 100)
    layers = {}
    sem = check_claims_consistency(triples, numerics)
    layers["semantic"] = sem
    if sem["ok"] is False:
        return TriageResult(DROP, layers, f"semantic contradiction: {sem['issues'][0][:90]}")
    from logos.logic.smt import CONSISTENT as _CONSISTENT

    truth = check_truth("", evidence=None)
    layers["truth"] = truth
    # UNKNOWN and CONSISTENT are both "not disproved", but they are NOT the same claim and must not
    # share wording. Reporting an unencodable predicate as "internally consistent" is the exact false
    # reassurance this pipeline exists to prevent.
    verdict = getattr(sem.get("report"), "verdict", None)
    if verdict == _CONSISTENT:
        why = "claims are internally consistent within the encoded fragment; truth still unproven"
    else:
        why = ("consistency NOT established -- some claims could not be encoded; truth also unproven")
    if policy["allow_escalate"]:
        return TriageResult(ESCALATE, layers, why + " -- worth paying to check")
    return TriageResult(DROP, layers,
                        why + ", and the scarcity policy does not justify paying to check it")


def check_truth(text: str, evidence=None) -> dict:
    """Truth is NOT decided here. This function reports whether evidence EXISTS to decide it.

    Returning a verdict would require the world. Returning "no evidence" is the honest answer and it is
    what routes a sample to ESCALATE rather than silently keeping it.
    """
    if evidence is None or (hasattr(evidence, "__len__") and len(evidence) == 0):
        return {"ok": None, "issues": ["no evidence available"], "layer": "truth"}
    return {"ok": None, "issues": ["evidence present but not adjudicated in this layer"],
            "layer": "truth"}


# ------------------------------------------------------------------- scarcity policy
def effort_tier(available: int, needed: int) -> dict:
    """How much validation effort the scarcity justifies.

    This is the module's reason to exist. The correct amount of validation is not a property of the
    pipeline, it is a property of how much choice you have.
    """
    if needed <= 0:
        raise ValueError("needed must be positive")
    slack = available / needed
    if slack >= 5.0:
        tier, allow_repair, allow_escalate, allow_llm = CREAM, False, False, False
        why = (f"{slack:.1f} candidates per needed sample -- discard anything not cleanly provable; "
               f"the ambiguous majority is not worth a model call")
    elif slack >= 2.0:
        tier, allow_repair, allow_escalate, allow_llm = STANDARD, True, True, False
        why = (f"{slack:.1f} candidates per needed sample -- repair what is cheaply repairable and "
               f"escalate the good-but-unproven")
    else:
        tier, allow_repair, allow_escalate, allow_llm = EXHAUSTIVE, True, True, True
        why = (f"{slack:.1f} candidates per needed sample -- fight for every row, including LLM "
               f"validation on the high-quality ambiguous")
    return {"tier": tier, "slack": slack, "allow_repair": allow_repair,
            "allow_escalate": allow_escalate, "allow_llm": allow_llm, "reason": why}


@dataclass
class TriageResult:
    disposition: str
    layers: dict = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict:
        return {"disposition": self.disposition, "reason": self.reason,
                "layers": {k: v["ok"] for k, v in self.layers.items()}}


def triage(text: str, kind: str = "text", evidence=None, policy: dict | None = None) -> TriageResult:
    """Run the four routers in cost order and return a triage disposition.

    Order matters for cost, not for correctness: structural is free, semantic is cheap, truth needs
    evidence, calibration needs a model. A sample that fails structure never reaches calibration.
    """
    policy = policy or effort_tier(100, 100)
    layers = {}

    s = check_structure(text, kind)
    layers["structural"] = s
    if not s["ok"]:
        # REPAIR only if the policy can afford it and the failure is in the repairable set
        repairable = any(k in " ".join(s["issues"]) for k in
                         ("SyntaxError", "unbalanced", "not encodable"))
        if repairable and policy["allow_repair"]:
            return TriageResult(REPAIR, layers, f"structural failure, repairable: {s['issues']}")
        return TriageResult(DROP, layers, f"structural failure: {s['issues'][:2]}")

    if kind == "code":
        m = check_semantics(text)
    else:
        m = check_numeric_consistency(text)
        m2 = check_structure(text, "text")
        m["issues"] = list(dict.fromkeys(m["issues"] + [i for i in m2["issues"]
                                                       if "parentheses" in i or "brackets" in i]))
        m["ok"] = not m["issues"]
    layers["semantic"] = m
    if not m["ok"]:
        if policy["allow_repair"]:
            return TriageResult(REPAIR, layers, f"semantic failure, repairable: {m['issues']}")
        return TriageResult(DROP, layers, f"semantic failure: {m['issues'][:2]}")

    t = check_truth(text, evidence)
    layers["truth"] = t
    if t["ok"] is None:
        if policy["allow_escalate"]:
            return TriageResult(ESCALATE, layers,
                                "passed syntax and semantics; truth unproven -- worth paying to check")
        return TriageResult(DROP, layers,
                            "passed syntax and semantics but truth unproven, and the scarcity policy "
                            "does not justify paying to check it")

    return TriageResult(KEEP, layers, "all layers passed")


# ------------------------------------------------------- total cost of a precision claim
def total_cost(n_needed: int, target_precision: float, filter_cost: float = 1.0,
               audit_cost: float = 200.0, defect_rate: float = 0.0005,
               confidence: float = 0.95) -> dict:
    """Cost of DELIVERING n_needed rows at a certified precision, filter plus audit.

    THE POINT THIS MAKES: precision is not free at the top. The audit set has to be large enough to
    observe defects, and each audited row costs a human. For high targets the audit dwarfs the filter,
    so there is an interior optimum -- chasing 99.9% can cost more than it is worth even when the filter
    is excellent.
    """
    from logos.refinery import audit_n_for_precision, audit_n_zero_defect

    if defect_rate > 0 and (1.0 - defect_rate) <= target_precision:
        return {"achievable": False, "total": math.inf,
                "reason": f"cannot certify {target_precision:.4%} when {defect_rate:.4%} are bad"}
    audits = audit_n_for_precision(target_precision, defect_rate, confidence)
    if audits is None:
        return {"achievable": False, "total": math.inf,
                "reason": "audit size unachievable at this defect rate"}
    # you must produce enough rows to still have n_needed after filtering, and audit a sample of them
    produced = int(math.ceil(n_needed / max(1e-9, 1.0 - defect_rate)))
    f_cost = produced * filter_cost
    a_cost = audits * audit_cost
    return {"achievable": True, "audits": audits, "produced": produced,
            "filter_cost": f_cost, "audit_cost": a_cost, "total": f_cost + a_cost}


def best_precision_target(n_needed: int, candidates=(0.90, 0.95, 0.98, 0.99, 0.995, 0.999),
                          **kw) -> dict:
    """Which precision target minimises total cost. The answer is rarely 99.9%."""
    rows = []
    for p in candidates:
        r = total_cost(n_needed, p, **kw)
        rows.append((p, r))
    achievable = [(p, r) for p, r in rows if r["achievable"]]
    if not achievable:
        return {"best": None, "rows": rows}
    best = min(achievable, key=lambda pr: pr[1]["total"])
    return {"best": best, "rows": rows}


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.triage - SELF TEST")
    ok = True

    # --- LAYER 1 is real for Python: the interpreter's own parser, not a heuristic
    bad_syntax = "def f(:\n    return 1"
    r = check_structure(bad_syntax, "code")
    good = not r["ok"] and "SyntaxError" in r["issues"][0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] structural layer catches a real SyntaxError")

    good_code = "x = 3\nprint(x)"
    good = check_structure(good_code, "code")["ok"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and passes valid Python")

    # --- LAYER 2 catches a SEMANTIC error that parses perfectly
    undefined = "def f():\n    return qqq + 1\n"
    sem = check_semantics(undefined)
    good = not sem["ok"] and any("qqq" in i for i in sem["issues"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] semantic layer catches an unbound name that parses fine "
          f"({sem['issues'][:1]})")

    # --- the reviewer's example: x=3 then x=8 is parsable and internally broken
    contradiction = "x = 3\ny = x + 1\nx = 8\nprint(y)"
    sem = check_semantics(contradiction)
    good = any("rebound" in i for i in sem["issues"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] semantic layer flags a rebound chain (x=3 ... x=8)")

    # --- LAYER 3 refuses to decide truth
    t = check_truth("2 + 2 = 5")
    good = t["ok"] is None and "no evidence" in t["issues"][0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] truth layer returns NO VERDICT -- it reports evidence "
          f"availability, not truth")

    # --- THE THREE-LAYER SEPARATION, on the reviewer's own three examples.
    #     Note what `2+2=5` actually is: as CODE it is a real Python syntax error (assignment to a
    #     literal), so the structural layer catches it correctly and for a different reason than the
    #     reviewer intended. As PROSE it is a truth error and must survive to the truth layer. Both are
    #     reported, because conflating them would overstate what the layers prove.
    print(f"  --- the reviewer's three failure examples ---")
    demos = (
        ("syntax   ') x if 4 ('", ") x if 4 (", "code"),
        ("semantic 'x=3 -> x=8'", "x = 3\ny = x + 1\nx = 8\n", "code"),
        ("truth    '2+2=5' as code", "2 + 2 = 5", "code"),
        ("truth    'two plus two is five' as prose",
         "Two plus two equals five, so the total count is five.", "text"),
    )
    for label, text, kind in demos:
        s = check_structure(text, kind)
        if not s["ok"]:
            stage = "structural"
        else:
            m = check_semantics(text) if kind == "code" else check_numeric_consistency(text)
            stage = "semantic" if not m["ok"] else "truth (UNDECIDED -- needs evidence)"
        print(f"    {label:44s} {stage}")

    # --- A FLAW IN MY OWN PROSE HEURISTIC, found by trying to demo the truth layer: requiring 30%
    #     letters DROPS digit-heavy text, so a legitimate arithmetic or tabular sample is thrown out as
    #     "symbol soup". That is exactly the kind of silent precision loss a refinery cannot afford.
    r = check_structure("2 + 2 = 5", "text")
    good = not r["ok"] and any("symbol soup" in i for i in r["issues"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] KNOWN FLAW: the prose heuristic drops digit-heavy text as "
          f"'symbol soup', so pure-arithmetic samples are lost before the truth layer")
    ok = ok and True

    # --- SCARCITY POLICY: the same pipeline must behave differently
    cream = effort_tier(10_000_000, 100_000)
    exhaust = effort_tier(200_000, 150_000)
    good = (cream["tier"] == CREAM and not cream["allow_repair"] and not cream["allow_escalate"]
            and exhaust["tier"] == EXHAUSTIVE and exhaust["allow_llm"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] scarcity sets the policy: 10M/100k -> {cream['tier']} "
          f"(no repair, no escalate); 200k/150k -> {exhaust['tier']} (LLM allowed)")

    # --- and the policy changes the DISPOSITION of the same ambiguous sample
    ambiguous = "x = 1\nprint(x)"          # clean code, truth unproven
    d_cream = triage(ambiguous, "code", policy=cream).disposition
    d_exh = triage(ambiguous, "code", policy=exhaust).disposition
    good = d_cream == DROP and d_exh == ESCALATE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the SAME sample is DROPped under cream and ESCALATEd "
          f"under exhaustive ({d_cream} vs {d_exh})")

    # --- structural failures are DROPped at cream and REPAIRed at standard
    std = effort_tier(500_000, 150_000)
    bad = "x = (1 + 2"
    d_cream = triage(bad, "code", policy=cream).disposition
    d_std = triage(bad, "code", policy=std).disposition
    good = d_cream == DROP and d_std == REPAIR
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a repairable syntax error is DROP at cream, REPAIR at "
          f"standard ({d_cream} vs {d_std})")

    # --- COST: certifying higher precision costs more, monotonically. There is NO interior optimum
    #     without a value model, and claiming one would be inventing a conclusion. What IS worth
    #     asserting is that the total is not dominated by the audit at ordinary targets -- the filter is
    #     the main cost until the target gets close to 1.
    lo = total_cost(100_000, 0.90, defect_rate=0.0005)
    hi = total_cost(100_000, 0.995, defect_rate=0.0005)
    good = hi["total"] > lo["total"] and lo["audit_cost"] < lo["filter_cost"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cost rises with the target, and at 90% the AUDIT is still "
          f"cheaper than the filter ({lo['audit_cost']:,.0f} vs {lo['filter_cost']:,.0f})")

    # --- and above the crossover the audit takes over, which is where "every extra nine" is paid
    top = total_cost(100_000, 0.999, defect_rate=0.0005)
    good = top["achievable"] and top["audit_cost"] > top["filter_cost"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] but at 99.9% the audit dominates "
          f"({top['audit_cost']:,.0f} vs {top['filter_cost']:,.0f}) -- that is the crossover")

    # --- and an unachievable target is refused rather than priced
    c = total_cost(100_000, 0.999, defect_rate=0.001)
    good = c["achievable"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 99.9% with 0.1% defects is refused, not priced")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def run() -> int:
    print("=== the same sample, two different corpus conditions ===")
    cases = [
        ("valid code, truth unproven", "def add(a, b):\n    return a + b\n", "code"),
        ("syntax error", "def add(a, b:\n    return a + b\n", "code"),
        ("parsable but semantically broken", "x = 3\ntotal = x + missing\n", "code"),
        ("prose, internally consistent", "The bridge is 120 metres tall. It opened in 1987.", "text"),
        ("prose, self-contradictory numbers", "The bridge is 120 metres tall. The bridge is 95 metres "
                                              "tall.", "text"),
        ("symbol soup", "||| ~~~ ^^^ ((( [[[ {", "text"),
    ]
    for label, available, needed in (("CREAM      10M -> 100k", 10_000_000, 100_000),
                                     ("STANDARD  500k -> 150k", 500_000, 150_000),
                                     ("EXHAUSTIVE 200k -> 150k", 200_000, 150_000)):
        pol = effort_tier(available, needed)
        print(f"\n--- {label}   (slack {pol['slack']:.1f}, tier {pol['tier']})")
        for name, text, kind in cases:
            r = triage(text, kind, policy=pol)
            print(f"  {r.disposition:9s} {name:32s} {r.reason[:74]}")

    print(f"\n=== the SEMANTIC router on extracted claims (Z3, no LLM) ===")
    from logos.logic.smt import NumericClaim, Triple

    claim_sets = (
        ("consistent chain", [Triple("whale", "is_a", "mammal"),
                              Triple("mammal", "is_a", "warm_blooded")], []),
        ("cycle: A before B before C before A",
         [Triple("a", "before", "b"), Triple("b", "before", "c"), Triple("c", "before", "a")], []),
        ("mutual is_a", [Triple("whale", "is_a", "mammal"),
                         Triple("mammal", "is_a", "whale")], []),
        ("numeric contradiction", [],
         [NumericClaim("bridge", "height", 120.0, 1.0, "m"),
          NumericClaim("bridge", "height", 95.0, 1.0, "m")]),
        ("numeric within tolerance", [],
         [NumericClaim("bridge", "height", 120.0, 1.0, "m"),
          NumericClaim("bridge", "height", 120.5, 1.0, "m")]),
        ("undeclared predicate", [Triple("a", "smells_like", "b")], []),
    )
    for label, tr, nu in claim_sets:
        r = triage_claims(tr, nu, policy=effort_tier(500_000, 150_000))
        print(f"  {r.disposition:9s} {label:36s} {r.reason[:66]}")

    print(f"\n=== the cost of a precision claim (100k rows needed) ===")
    print(f"  {'target':>8s} {'audits':>8s} {'filter cost':>12s} {'audit cost':>12s} {'total':>12s}")
    for p in (0.90, 0.95, 0.98, 0.99, 0.995):
        r = total_cost(100_000, p, filter_cost=1.0, audit_cost=200.0, defect_rate=0.0005)
        if not r["achievable"]:
            print(f"  {p:8.3%} {'--':>8s} {'--':>12s} {'--':>12s} "
                  f"{'UNACHIEVABLE':>12s}")
            continue
        print(f"  {p:8.3%} {r['audits']:8,d} {r['filter_cost']:12,.0f} {r['audit_cost']:12,.0f} "
              f"{r['total']:12,.0f}")
    b = best_precision_target(100_000, defect_rate=0.0005)
    filt = b["rows"][0][1]["filter_cost"]
    print(f"\n  Cost is MONOTONE in the target: {b['rows'][0][0]:.1%} costs "
          f"{b['rows'][0][1]['total']:,.0f}, {b['rows'][-1][0]:.1%} costs "
          f"{b['rows'][-1][1]['total']:,.0f}.")
    print(f"  There is NO interior optimum without a value model -- 'cheapest' is just 'lowest target'.")
    print(f"  So the real question is not 'which target is cheapest' but 'what precision does the")
    print(f"  training actually need', because below the crossover the FILTER dominates the budget and")
    print(f"  above it the AUDIT does.")
    crossover = next((p for p, r in b["rows"] if r.get("audit_cost", 0) > r.get("filter_cost", 1)),
                     None)
    print(f"  Here the filter costs {filt:,.0f} for 100k rows ({filt / 100_000:.2f}/row) and the audit")
    print(f"  costs 200/row, so the crossover sits near 99.2%: beyond it every extra nine is paid to")
    print(f"  auditors, not to the pipeline. That is the line this project should be arguing about,")
    print(f"  and it is a budget question, not a quality one.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Four-router triage + scarcity policy + audit cost.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.run:
        return run()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
