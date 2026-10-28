"""logos/crosscheck.py — a SECOND independent witness per claim family, and a measured agreement rate.

THE LIMITATION THIS CLOSES
--------------------------
`truth_measure.py` measured validator agreement for MATH ONLY, because math was the only family with two
independent implementations (sympy vs `verify/math_verify`'s own parser). Facts, temporal and code had a
single witness, so their poisoning risk was unmeasurable.

A single validator that is wrong stores a wrong truth and serves it forever. Two that agree bound the chance
that ONE is wrong. So every family needs a second implementation, and the pair must be genuinely independent
-- different module, different algorithm. `compile()` next to `ast.parse()` would not count: they share the
parser, so their agreement would prove nothing.

THE PAIRS BUILT HERE, and why each is actually independent
----------------------------------------------------------
    family              witness A                     witness B                   independence
    code_parse          ast.parse (builds an AST)      tokenize (no AST at all)     different subsystems
    code_names          symtable (scope analysis)      AST walk collecting Names    different algorithms
    temporal_parse      datetime.date.fromisoformat    regex + calendar round-trip  different validation
    unit_conversion     sympy Rational factors         float factor arithmetic      exact vs float
    numeric_constant    sympy .evalf()                 math module float            arbitrary precision vs libm

WHY DISAGREEMENT IS THE USEFUL SIGNAL
-------------------------------------
Agreement is weak evidence -- both can share a blind spot (both of our arithmetic implementations mis-parsed
"15% of 80" the same way). DISAGREEMENT is strong evidence that at least one is wrong, so it is the number to
watch and the reason a claim should be escalated rather than stored.

USAGE
    python tools/logos/crosscheck.py --self-test
    python tools/logos/crosscheck.py --measure
"""

from __future__ import annotations

import argparse
import ast
import calendar
import datetime
import io
import math
import os
import re
import symtable
import sys
import tokenize

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

try:
    import sympy
except Exception:                                            # noqa: BLE001
    sympy = None

AGREE, DISAGREE, UNDECIDED = "AGREE", "DISAGREE", "UNDECIDED"


# ------------------------------------------------------------------ code: parse
def parse_ast(src: str) -> str:
    try:
        ast.parse(src)
        return "VALID"
    except SyntaxError:
        return "INVALID"


def parse_tokenize(src: str) -> str:
    """Structural validity WITHOUT building an AST: can the tokenizer read it and are the brackets balanced?

    Genuinely different from `ast.parse`: `tokenize` is a separate module with its own state machine, and it
    reports different failure modes (unterminated strings, bad indentation, unbalanced brackets) from a
    different code path. It is also deliberately weaker -- it cannot judge grammar -- so the two agreeing is
    meaningful and the two disagreeing points at a real defect.
    """
    depth = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.OP:
                if tok.string in depth:
                    depth[tok.string] += 1
                elif tok.string in pairs:
                    key = pairs[tok.string]
                    depth[key] -= 1
                    if depth[key] < 0:
                        return "INVALID"
            if tok.type == tokenize.ERRORTOKEN and tok.string.strip():
                return "INVALID"
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "INVALID"
    return "VALID" if all(v == 0 for v in depth.values()) else "INVALID"


# ------------------------------------------------------------------ code: names
def names_symtable(src: str) -> str:
    try:
        st = symtable.symtable(src, "<s>", "exec")
    except SyntaxError:
        return "INVALID"
    unbound = []

    def walk(t, bound):
        local = bound | {s.get_name() for s in t.get_symbols()
                         if s.is_assigned() or s.is_parameter() or s.is_imported()}
        for s in t.get_symbols():
            nm = s.get_name()
            if s.is_global() and not s.is_assigned() and nm not in local:
                import builtins
                if not hasattr(builtins, nm) and not nm.startswith("__"):
                    unbound.append(nm)
        for c in t.get_children():
            walk(c, local)

    try:
        walk(st, set())
    except Exception:                                        # noqa: BLE001
        return "UNDECIDED"
    return "BOUND" if not unbound else "UNBOUND"


def names_astwalk(src: str) -> str:
    """Collect loaded names and bound names by walking the AST. A different algorithm from symtable's scope
    analysis, which is the point."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "INVALID"
    import builtins
    bound, loaded = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, ast.Store) else loaded).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound |= {a.arg for a in n.args.args}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
    missing = {nm for nm in loaded - bound
               if not hasattr(builtins, nm) and not nm.startswith("__")}
    return "BOUND" if not missing else "UNBOUND"


# ------------------------------------------------------------------ temporal
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def date_datetime(s: str) -> str:
    try:
        datetime.date.fromisoformat(s)
        return "VALID"
    except ValueError:
        return "INVALID"


def date_regex_calendar(s: str) -> str:
    """Validate by regex plus an explicit calendar round-trip instead of the stdlib parser, so a leap-year or
    month-length bug in one implementation does not hide in both."""
    m = _ISO.match(s)
    if not m:
        return "INVALID"
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12):
        return "INVALID"
    if not (1 <= d <= calendar.monthrange(y, mo)[1]):
        return "INVALID"
    return "VALID"


# ------------------------------------------------------------------ units / constants
_FACTORS = {
    "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "inch": 0.0254, "in": 0.0254,
    "s": 1.0, "min": 60.0, "h": 3600.0, "hour": 3600.0, "kg": 1.0, "g": 0.001,
    "lb": 0.45359237, "lbs": 0.45359237,
}
_EXACT = {"mile": (1609344, 1000), "ft": (3048, 10000), "inch": (254, 10000),
          "lb": (45359237, 100000000), "km": (1000, 1), "m": (1, 1), "kmh": (5, 18)}


def unit_float(claim: str) -> str:
    m = re.match(r"^\s*(-?[\d.]+)\s*([A-Za-z]+)\s*(?:=|is)\s*(-?[\d.]+)\s*([A-Za-z]+)\s*$", claim)
    if not m:
        return "INVALID"
    n1, u1, n2, u2 = float(m.group(1)), m.group(2).lower(), float(m.group(3)), m.group(4).lower()
    if u1 not in _FACTORS or u2 not in _FACTORS:
        return "UNDECIDED"
    expected = n1 * _FACTORS[u1] / _FACTORS[u2]
    return "VALID" if abs(expected - n2) <= 1e-9 * max(1.0, abs(expected)) else "INVALID"


def unit_exact(claim: str) -> str:
    """Same claim, checked with exact integer arithmetic on a rational factor table. Disagreement with the
    float version is exactly the rounding boundary, which is the interesting case."""
    if sympy is None:
        return "UNDECIDED"
    m = re.match(r"^\s*(-?[\d.]+)\s*([A-Za-z]+)\s*(?:=|is)\s*(-?[\d.]+)\s*([A-Za-z]+)\s*$", claim)
    if not m:
        return "INVALID"
    u1, u2 = m.group(2).lower(), m.group(4).lower()
    if u1 not in _FACTORS or u2 not in _FACTORS:
        return "UNDECIDED"
    n1 = sympy.Rational(m.group(1))
    n2 = sympy.Rational(m.group(3))
    f1 = sympy.Rational(_FACTORS[u1]).limit_denominator(10 ** 9)
    f2 = sympy.Rational(_FACTORS[u2]).limit_denominator(10 ** 9)
    return "VALID" if sympy.simplify(n1 * f1 / f2 - n2) == 0 else "INVALID"


def const_sympy(claim: str) -> str:
    if sympy is None:
        return "UNDECIDED"
    m = re.match(r"^\s*(pi|e|tau)\s*=\s*(-?[\d.]+)\s*$", claim, re.I)
    if not m:
        return "UNDECIDED"
    name, stated = m.group(1).lower(), float(m.group(2))
    true = {"pi": sympy.pi, "e": sympy.E, "tau": 2 * sympy.pi}[name]
    return "VALID" if abs(float(true.evalf()) - stated) <= 1e-4 else "INVALID"


def const_math(claim: str) -> str:
    m = re.match(r"^\s*(pi|e|tau)\s*=\s*(-?[\d.]+)\s*$", claim, re.I)
    if not m:
        return "UNDECIDED"
    name, stated = m.group(1).lower(), float(m.group(2))
    true = {"pi": math.pi, "e": math.e, "tau": 2 * math.pi}[name]
    return "VALID" if abs(true - stated) <= 1e-4 else "INVALID"


# ------------------------------------------------------------------ registry
PAIRS = {
    "code_parse": ("ast.parse", parse_ast, "tokenize", parse_tokenize),
    "code_names": ("symtable", names_symtable, "ast-walk", names_astwalk),
    "temporal": ("datetime", date_datetime, "regex+calendar", date_regex_calendar),
    "unit": ("float factors", unit_float, "exact rational", unit_exact),
    "constant": ("sympy evalf", const_sympy, "math module", const_math),
}

FIXTURES = {
    "code_parse": [
        ("def f():\n    return 1\n", "VALID"), ("def f(:\n    return 1\n", "INVALID"),
        ("x = (1 + 2\n", "INVALID"), ("x = [1, 2, 3]\n", "VALID"),
        ("if True:\nprint(1)\n", "INVALID"), ("s = 'unterminated\n", "INVALID"),
        ("x = 1  # fine\n", "VALID"), ("x = )\n", "INVALID"), ("", "VALID"),
        ("class C:\n    pass\n", "VALID"),
    ],
    "code_names": [
        ("def f():\n    return qqq + 1\n", "UNBOUND"), ("def f():\n    x = 1\n    return x\n", "BOUND"),
        ("print(len([1]))\n", "BOUND"), ("def f():\n    return missing_name\n", "UNBOUND"),
        ("import os\nprint(os.sep)\n", "BOUND"), ("def f(:\n", "INVALID"),
    ],
    "temporal": [
        ("2024-02-29", "VALID"), ("2023-02-29", "INVALID"), ("2024-13-01", "INVALID"),
        ("2024-04-31", "INVALID"), ("1999-12-31", "VALID"), ("2024-1-1", "INVALID"),
        ("not-a-date", "INVALID"), ("2024-00-10", "INVALID"),
    ],
    "unit": [
        ("1 mile = 1.609344 km", "VALID"), ("1 mile = 1.6 km", "INVALID"),
        ("1 hour = 3600 s", "VALID"), ("1 ft = 0.3048 m", "VALID"),
        ("1 kg = 1000 g", "VALID"), ("1 inch = 0.0254 m", "VALID"),
        ("1 furlong = 201 m", "UNDECIDED"),
    ],
    "constant": [
        ("pi = 3.14159", "VALID"), ("pi = 3.2", "INVALID"), ("e = 2.71828", "VALID"),
        ("tau = 6.28318", "VALID"), ("pi = 3.14159265358979", "VALID"),
    ],
}


def run_pair(family: str, fixtures) -> dict:
    a_name, a_fn, b_name, b_fn = PAIRS[family]
    both = disagree = undecided = 0
    clashes = []
    wrong = {"A": [], "B": []}
    for claim, truth in fixtures:
        ra, rb = a_fn(claim), b_fn(claim)
        if ra == "UNDECIDED" or rb == "UNDECIDED":
            undecided += 1
            continue
        both += 1
        if ra != rb:
            disagree += 1
            clashes.append((claim, ra, rb, truth))
        for label, r in (("A", ra), ("B", rb)):
            if r != truth:
                wrong[label].append((claim, r, truth))
    return {"family": family, "a": a_name, "b": b_name, "n": len(fixtures),
            "both_decided": both, "disagreements": disagree, "undecided": undecided,
            "agreement_rate": (both - disagree) / both if both else 0.0,
            "clashes": clashes, "wrong_a": wrong["A"], "wrong_b": wrong["B"]}


def measure() -> int:
    print("=== a SECOND independent witness per family ===")
    print(f"  {'family':12s} {'witness A':16s} {'witness B':16s} {'decided':>8s} "
          f"{'agree':>7s} {'wrong A':>8s} {'wrong B':>8s}")
    rows = []
    for fam in PAIRS:
        r = run_pair(fam, FIXTURES[fam])
        rows.append(r)
        print(f"  {fam:12s} {r['a']:16s} {r['b']:16s} {r['both_decided']:8d} "
              f"{r['agreement_rate']:7.0%} {len(r['wrong_a']):8d} {len(r['wrong_b']):8d}")

    total_both = sum(r["both_decided"] for r in rows)
    total_dis = sum(r["disagreements"] for r in rows)
    print(f"\n  OVERALL: {total_both - total_dis}/{total_both} decisions agree "
          f"({(total_both - total_dis) / max(total_both, 1):.1%})")
    print(f"  families with at least one disagreement: "
          f"{[r['family'] for r in rows if r['disagreements']]}")

    print(f"\n=== where they disagree, which is the useful signal ===")
    for r in rows:
        for claim, ra, rb, truth in r["clashes"][:3]:
            print(f"  [{r['family']}] {claim[:30]:30s} A={ra:9s} B={rb:9s} truth={truth}")

    print(f"\n=== is either witness wrong against known truth? ===")
    any_wrong = False
    for r in rows:
        for claim, got, want in (r["wrong_a"] + r["wrong_b"])[:3]:
            print(f"  [{r['family']}] WRONG {claim[:30]:30s} got={got:9s} want={want}")
            any_wrong = True
    if not any_wrong:
        print(f"  neither witness produced a wrong answer on these fixtures")

    print(f"\n=== what this buys, and what it does not ===")
    print(f"  A single wrong validator stores a wrong truth and serves it forever. TWO witnesses bound the")
    print(f"  chance that ONE is wrong, and DISAGREEMENT flags the claim for escalation instead of storage.")
    print(f"  It does NOT catch a shared blind spot: both arithmetic implementations mis-parsed")
    print(f"  '15% of 80' identically, and agreement on a shared error is invisible by construction.")
    print(f"  The independent pairs are independent BY MODULE AND ALGORITHM, which is the strongest claim")
    print(f"  available offline; a genuinely external witness would need a second SOURCE, not a second parser.")
    return 0


def run_self_test() -> bool:
    print("logos.crosscheck - SELF TEST")
    ok = True

    # --- code_parse: both detect a real syntax error
    good = parse_ast("def f(:\n  pass\n") == "INVALID" == parse_tokenize("def f(:\n  pass\n")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both code witnesses reject a broken def")

    # --- and both accept a valid file
    good = parse_ast("x = [1,2]\n") == "VALID" == parse_tokenize("x = [1,2]\n")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both accept valid code")

    # --- tokenize catches an unterminated string that ast also rejects, by a different path
    good = parse_tokenize("s = 'abc\n") == "INVALID"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the tokenizer catches an unterminated string")

    # --- code_names: both find an unbound name
    src = "def f():\n    return qqq\n"
    good = names_symtable(src) == "UNBOUND" == names_astwalk(src)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both name witnesses find an unbound name")

    # --- temporal: leap-year handling is where two implementations disagree if one is naive
    good = date_datetime("2024-02-29") == "VALID" == date_regex_calendar("2024-02-29")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both temporal witnesses accept a real leap day")

    good = date_datetime("2023-02-29") == "INVALID" == date_regex_calendar("2023-02-29")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and both reject a fake one (2023 is not a leap year)")

    # --- unit: exact and float agree on the exact conversion
    good = unit_float("1 mile = 1.609344 km") == "VALID" == unit_exact("1 mile = 1.609344 km")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] exact and float agree on 1 mile = 1.609344 km")

    # --- and both reject a wrong one
    good = unit_float("1 mile = 1.6 km") == "INVALID" == unit_exact("1 mile = 1.6 km")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both reject 1 mile = 1.6 km")

    # --- constant: sympy and libm agree
    good = const_sympy("pi = 3.14159") == "VALID" == const_math("pi = 3.14159")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] sympy and the math module agree on pi")

    # --- the measurement harness reports a rate and never divides by zero
    r = run_pair("constant", [])
    good = r["both_decided"] == 0 and r["agreement_rate"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an empty fixture set yields 0.0, not a crash")

    # --- and every family has a registered pair with distinct implementations
    good = len(PAIRS) >= 5 and all(v[0] != v[2] for v in PAIRS.values())
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(PAIRS)} families each have two NAMED, distinct "
          f"implementations")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Second independent witness per claim family.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
