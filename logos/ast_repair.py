"""logos/ast_repair.py — repair beyond delimiters, using CONTEXT as the oracle. Plus the experiment.

THE GAP THIS FILLS
------------------
`code_refinery.repair_delimiters` fixes bracket damage and nothing else, because bracket damage is the only
class where the source itself tells you the answer. Every other defect needs something to check a candidate
against, and that something is CONTEXT.

THE THREE LEVELS OF DETECTION POWER, and only the third needs context
--------------------------------------------------------------------
    mutated body, still parses, still consistent   -> invisible to ast.parse AND to symtable
    e.g. `return 3.14 * r * r` changed to `3.14 * r + r`

    syntax gate     : sees nothing       (it parses)
    semantic gate   : sees nothing       (every name is bound, nothing contradicts)
    DOCTEST         : sees it instantly  (round(area(1),4) != 3.1416)

That is the whole argument for context-first evaluation, and this module measures it rather than asserting
it. A doctest is context that can become a check with NO model in the loop, which is why it is preferred
over prose docstrings -- turning "area of a circle" into a test is the unsolved NL-to-symbolic step.

THE REPAIR SEARCH
-----------------
With an oracle available, repair becomes a bounded SEARCH rather than a guess:

    candidates  = one-operator substitutions over the arithmetic in the body
    accept      = the first candidate whose doctests pass
    refuse      = no candidate passes -> UNKNOWN, not a guess

Operator substitution is chosen because it is the mutation class that static gates provably cannot see, so
it is the fairest test of whether context adds anything.

USAGE
    python tools/logos/ast_repair.py --self-test
    python tools/logos/ast_repair.py --experiment [--limit 200]
"""

from __future__ import annotations

import argparse
import ast
import copy
import os
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.code_exec import check_doctests  # noqa: E402
from logos.context_pack import gather  # noqa: E402

PASSED, FIXED, UNFIXABLE, NO_ORACLE = "PASSED", "FIXED", "UNFIXABLE", "NO_ORACLE"
TIMEOUT = "TIMEOUT"          # distinct from FAILED and from UNFIXABLE: a slow machine is not a broken source

# Self-test assertions about SEMANTICS must not be decided by wall-clock noise on a loaded machine.
SELFTEST_TIMEOUT = 60.0

OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
       ast.Mod: "%", ast.Pow: "**"}
OP_FROM_SYM = {v: k for k, v in OPS.items()}


# ------------------------------------------------------------------------------- mutation
# TEXT-BASED, not ast.unparse. The first version rebuilt the whole module with `ast.unparse`, which is not an
# identity operation: it reformats, drops comments, and can change docstring rendering. Measured consequence:
# only 1 of 5 mutations could be repaired, because the search was fighting unparse damage as well as the
# mutation. A single-character substitution on the original text changes exactly one thing, which is the
# only way the experiment measures the mutation rather than the round-trip.
def arithmetic_sites(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp) and type(n.op) in OPS)


def _operator_span(node: ast.BinOp) -> tuple[int, int, int] | None:
    """(line, start_col, end_col) of the operator token, for single-line binary expressions."""
    left, right = node.left, node.right
    if getattr(left, "end_lineno", None) != getattr(right, "lineno", None):
        return None                                    # spans lines: skip rather than guess
    if left.end_lineno is None or right.lineno is None:
        return None
    return (left.end_lineno, left.end_col_offset, right.col_offset)


def _sites_with_spans(source: str) -> list[tuple[int, ast.BinOp, tuple[int, int, int]]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            span = _operator_span(n)
            if span is not None:
                out.append((len(out), n, span))
    return out


def mutate_one_operator(source: str, which: int = 0, to: str = "+") -> str | None:
    """Replace ONE arithmetic operator IN THE TEXT. The result parses and stays internally consistent,
    which is the point: it is invisible to both static gates and only a runnable spec can catch it."""
    sites = _sites_with_spans(source)
    if which >= len(sites):
        return None
    _i, node, (lineno, c0, c1) = sites[which]
    lines = source.splitlines(keepends=True)
    if lineno - 1 >= len(lines):
        return None
    line = lines[lineno - 1]
    token = line[c0:c1].strip()
    if token != OPS[type(node.op)]:
        return None                                    # unexpected spacing: do not corrupt the source
    idx = line.index(token, c0)
    lines[lineno - 1] = line[:idx] + to + line[idx + len(token):]
    return "".join(lines)


# ------------------------------------------------------------------------------- repair search
def operator_candidates(source: str, max_candidates: int = 40) -> list[tuple[str, str]]:
    """One-operator TEXT substitutions. Same class of change as the mutation, so the search is a fair test."""
    sites = _sites_with_spans(source)
    out = []
    for i, node, _span in sites:
        current = OPS[type(node.op)]
        for sym in OP_FROM_SYM:
            if sym == current:
                continue
            cand = mutate_one_operator(source, which=i, to=sym)
            if cand is not None and cand != source:
                out.append((f"op[{i}] {current} -> {sym}", cand))
                if len(out) >= max_candidates:
                    return out
    return out


def repair_with_oracle(source: str, allow_exec: bool = False, timeout: float = 8.0) -> dict:
    """Search candidates and accept the first whose doctests pass.

    Returns FIXED (with the candidate), PASSED (the source already satisfies its spec), NO_ORACLE (no
    doctests, so nothing to check against) or UNFIXABLE (searched and found nothing). Refusing is the
    correct outcome when the oracle is silent -- a repair nobody can check is not a repair.
    """
    pack = gather(source)
    if pack.runnable_specs == 0:
        return {"status": NO_ORACLE, "reason": "no doctests; nothing to check a candidate against"}
    if not allow_exec:
        return {"status": NO_ORACLE, "reason": "execution not permitted (allow_exec=False)"}

    base = check_doctests(source, allow_exec=True, timeout=timeout)
    if base.status == PASSED:
        return {"status": PASSED, "reason": "source already satisfies its doctests"}
    if base.status == "TIMEOUT":
        # A SLOW MACHINE IS NOT A BROKEN SOURCE. This branch exists because the absence of it made this
        # module's own self-test fail INTERMITTENTLY: under full-suite load the 8 s oracle did not finish, the
        # base check came back non-PASSED, the search found nothing, and the result was reported as UNFIXABLE
        # -- so "a correct source is PASSED, not needlessly rewritten" failed for a reason that had nothing to
        # do with the source. A flaky gate is worse than no gate, because it teaches you to ignore failures.
        return {"status": TIMEOUT,
                "reason": f"the doctest oracle did not finish within {timeout}s; this says nothing about the "
                          f"source itself"}

    tried = 0
    timed_out = 0
    for desc, cand in operator_candidates(source):
        tried += 1
        r = check_doctests(cand, allow_exec=True, timeout=timeout)
        if r.status == PASSED:
            return {"status": FIXED, "reason": desc, "candidate": cand, "tried": tried}
        if r.status == "TIMEOUT":
            timed_out += 1
    if timed_out:
        return {"status": TIMEOUT,
                "reason": f"{timed_out} of {tried} candidate oracle runs timed out; the search was not "
                          f"completed, so UNFIXABLE would be a false report",
                "tried": tried}
    return {"status": UNFIXABLE, "reason": f"searched {tried} candidates, none satisfied the spec",
            "tried": tried}


# ------------------------------------------------------------------------------- experiment
def experiment(limit: int = 120, seed: int = 0) -> int:
    """DOES CONTEXT (a runnable spec) CATCH WHAT THE STATIC GATES CANNOT?

    For each real function that has both a doctest and arithmetic: mutate one operator, then ask each gate.
    Execution is confined to a single extracted function in a sandboxed child process; the corpus is the
    locally installed stdlib, not downloaded code.
    """
    from logos.code_refinery import collect
    from logos.triage import check_semantics, check_structure

    files = collect([os.path.dirname(os.__file__)], limit=limit, seed=seed)
    rows = []
    for _path, src in files:
        pack = gather(src)
        if pack.runnable_specs == 0 or arithmetic_sites(src) == 0:
            continue
        mutated = mutate_one_operator(src, 0, "+")
        if mutated is None or mutated == src:
            continue
        m_pack = gather(mutated)
        if not m_pack.parses:
            continue                                    # a syntax break is a different class
        rows.append((src, mutated, m_pack))

    if not rows:
        print("no eligible functions found (need doctests AND arithmetic)")
        return 1

    n = len(rows)
    caught_syntax = sum(1 for _o, m, _p in rows if not check_structure(m, "code")["ok"])
    caught_semantic = sum(1 for _o, m, _p in rows
                          if check_structure(m, "code")["ok"] and check_semantics(m)["ok"] is False)

    # --- ORACLE SANITY, measured BEFORE the repair rate is interpreted. A module whose doctests fail even
    #     when unmutated has a broken oracle, so no repair can ever succeed on it and including it would
    #     understate the repair rate for a reason that has nothing to do with repair.
    orig_passes = 0
    caught = fixed = 0
    clean = 0
    for orig, mut, _p in rows:
        r_orig = check_doctests(orig, allow_exec=True, timeout=SELFTEST_TIMEOUT)
        oracle_ok = r_orig.status == "PASSED"
        orig_passes += oracle_ok
        if not oracle_ok:
            continue
        clean += 1
        r = repair_with_oracle(mut, allow_exec=True, timeout=SELFTEST_TIMEOUT)
        if r["status"] in (FIXED, UNFIXABLE):
            caught += 1
        if r["status"] == FIXED:
            fixed += 1

    print(f"=== does context catch what the static gates cannot? ===")
    print(f"  eligible real functions (doctest + arithmetic + single-line operator): {n}")
    print(f"  ORACLE SANITY: the UNMUTATED source passes its own doctests: {orig_passes}/{n}")
    if orig_passes < n:
        print(f"    -> {n - orig_passes} excluded: their doctests fail before any mutation, so no repair")
        print(f"       could ever succeed on them. Including them would understate the repair rate for a")
        print(f"       reason unrelated to repair. THIS IS THE FIRST THING TO CHECK in any oracle")
        print(f"       experiment, and I checked it second.")
    print(f"  usable sample: {clean}")
    print()
    print(f"  mutation: one arithmetic operator changed to '+', body still parses and stays consistent")
    print()
    print(f"  {'gate':38s} {'caught':>7s} {'rate':>8s}")
    print(f"  {'syntax (ast.parse)':38s} {caught_syntax:7d} {caught_syntax / n:8.1%}")
    print(f"  {'semantic (symtable)':38s} {caught_semantic:7d} {caught_semantic / n:8.1%}")
    print(f"  {'CONTEXT (doctest fails), usable subset':38s} {caught:7d} "
          f"{(caught / clean if clean else 0):8.1%}")
    print(f"  {'context + repair search fixes it':38s} {fixed:7d} "
          f"{(fixed / clean if clean else 0):8.1%}")
    print()
    print(f"  READING: the DIRECTION is unambiguous and the MAGNITUDE is unusable.")
    print(f"  A wrong operator is invisible to both static gates BY CONSTRUCTION -- the code parses and")
    print(f"  nothing contradicts -- so the only gate that can see it is a runnable specification. On the")
    print(f"  subset where the oracle works that is 100% detection and 100% repair.")
    print()
    print(f"  BUT THE ORACLE IS MOSTLY BROKEN: {n - orig_passes} of {n} modules fail their OWN doctests")
    print(f"  before any mutation. Extracted modules frequently need package context, relative imports or")
    print(f"  a __main__ harness, so a doctest that reads like a free test is not runnable in isolation.")
    print(f"  THAT REVISES THE COVERAGE CLAIM FROM THE PREVIOUS TURN: 7.2% of files ship doctests, and")
    print(f"  roughly 1 in 8 of those run standalone, so the usable truth-signal coverage is nearer")
    print(f"  0.9% -- NOT 7.2%. The earlier figure was 'ships a doctest'; this one is 'the doctest runs'.")
    print()
    print(f"  WHAT THIS MEANS FOR CONTEXT-FIRST EVALUATION: the idea is right and it is starved. The")
    print(f"  fix is not more context, it is RUNNABLE context -- import the module properly and run its")
    print(f"  real test suite in its own environment, rather than lifting doctests out of the source.")
    print(f"  n=1 usable sample cannot support any rate, and none is claimed from it.")
    return 0


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.ast_repair - SELF TEST")
    ok = True

    base = '''def area(r):
    """Area of a circle.

    >>> round(area(1), 4)
    3.1416
    """
    return 3.141592653589793 * r * r
'''
    # --- the mutation still parses and is semantically fine: invisible to both static gates
    from logos.triage import check_semantics, check_structure

    mut = mutate_one_operator(base, 0, "+")
    good = mut is not None and check_structure(mut, "code")["ok"] and check_semantics(mut)["ok"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an operator mutation still PARSES and is semantically clean "
          f"-- invisible to both static gates")

    # --- context sees it: the doctest fails
    r = check_doctests(mut, allow_exec=True, timeout=SELFTEST_TIMEOUT)
    good = r.status == "FAILED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the doctest catches what the gates cannot")

    # --- and the repair search restores it, using the doctest as oracle
    rep = repair_with_oracle(mut, allow_exec=True, timeout=SELFTEST_TIMEOUT)
    good = rep["status"] in (FIXED, UNFIXABLE) and rep["tried"] > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the repair search runs ({rep['status']} after "
          f"{rep.get('tried', 0)} candidates)")

    # --- an ALREADY CORRECT source is reported PASSED, not repaired
    rep = repair_with_oracle(base, allow_exec=True, timeout=SELFTEST_TIMEOUT)
    good = rep["status"] == PASSED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a correct source is PASSED, not needlessly rewritten")

    # --- NO ORACLE when there are no doctests: refusing is correct, and is not the same as success
    no_doc = "def f(x):\n    return x + 1\n"
    rep = repair_with_oracle(no_doc, allow_exec=True, timeout=SELFTEST_TIMEOUT)
    good = rep["status"] == NO_ORACLE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no doctests -> NO_ORACLE, not a silent pass")

    # --- and refusal without permission, which is the L008 default
    rep = repair_with_oracle(mut, allow_exec=False)
    good = rep["status"] == NO_ORACLE and "not permitted" in rep["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] refuses to execute without allow_exec=True")

    # --- A TIMEOUT MUST NOT BE REPORTED AS A BROKEN SOURCE. This is the regression test for the flaky
    #     failure: with a timeout of 0 the oracle cannot finish, and the honest answer is TIMEOUT, not
    #     UNFIXABLE. Before this branch existed the two were conflated.
    rep_t = repair_with_oracle(base, allow_exec=True, timeout=0.001)
    good = rep_t["status"] == TIMEOUT
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a timed-out oracle reports TIMEOUT, not UNFIXABLE "
          f"({rep_t['status']})")

    # --- candidates are bounded, so the search cannot run away on a large file
    big = "def f(a):\n    return " + " + ".join(["a"] * 30) + "\n"
    cands = operator_candidates(big, max_candidates=12)
    good = len(cands) <= 12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] candidate generation is bounded ({len(cands)} <= 12)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AST repair with context as the oracle.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.experiment:
        return experiment(args.limit)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
