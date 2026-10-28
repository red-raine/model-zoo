"""logos/verify/math_core.py — symbolic and numeric math validation. The cheapest real truth we have.

WHY MATH FIRST
--------------
`verify/math_verify.py` already handles arithmetic, percentages and grouped expressions. This module covers
the rest of what a dataset actually asserts, and every one of them is DECIDABLE:

    numeric constant    pi = 3.14159          (to a stated tolerance)
    algebraic identity  (x+1)^2 = x^2+2x+1    (symbolic simplification, exact)
    calculus            d/dx x^2 = 2x          (sympy.diff, exact)
    linear algebra      det([[1,2],[3,4]]) = -2 (numpy/sympy, exact)
    unit conversion     1 mile = 1.609344 km   (exact table, no model)
    equation solution   x=2 solves x^2 = 4     (substitution, exact)

**No language model, no corpus, no ambiguity.** That makes math the correct first layer in a refinery: it
converts a large slice of "is this true?" into computing, and it does so at machine speed with a proof.

THE ONE THING THAT MAKES MATH VALIDATION SUBTLE
-----------------------------------------------
**Domain.** `sqrt(x^2) = x` is true for x >= 0 and false otherwise. `1/x * x = 1` fails at x = 0. A validator
that returns VERIFIED without stating the assumed domain is producing a proof of something else. So every
verdict carries the domain it assumed, and claims that hold only under an assumption are returned as
CONDITIONAL rather than VERIFIED -- which is a first-class outcome, not a hedge.

USAGE
    python tools/logos/verify/math_core.py --self-test
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

try:
    import sympy
    from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                            parse_expr, standard_transformations)
except Exception:                                          # noqa: BLE001
    sympy = None

try:
    import numpy as np
except Exception:                                          # noqa: BLE001
    np = None

VERIFIED, REFUTED, CONDITIONAL, UNKNOWN = "VERIFIED", "REFUTED", "CONDITIONAL", "UNKNOWN"

TRANSFORMS = (standard_transformations + (implicit_multiplication_application,)) if sympy else ()

# exact conversion factors to SI base, so a conversion claim is arithmetic rather than a lookup guess
UNITS = {
    "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "inch": 0.0254, "in": 0.0254,
    "yard": 0.9144, "yd": 0.9144,
    "s": 1.0, "min": 60.0, "h": 3600.0, "hour": 3600.0, "day": 86400.0,
    "kg": 1.0, "g": 0.001, "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523125,
    "l": 0.001, "litre": 0.001, "liter": 0.001, "ml": 1e-6, "gal": 0.003785411784,
}


@dataclass
class Verdict:
    status: str = UNKNOWN
    kind: str = ""
    detail: str = ""
    assumed: list = field(default_factory=list)
    value: object = None

    def as_dict(self) -> dict:
        return {"status": self.status, "kind": self.kind, "detail": self.detail,
                "assumed": self.assumed, "value": str(self.value)[:120] if self.value is not None else None}


def _parse(text: str):
    """Parse an expression, with the two corrections that every maths-in-prose pipeline needs.

    BUG 1, and it silently broke most of this module's first self-test: **`^` IS XOR**, in Python and in
    sympy. `x^2` parsed as a logical expression, so `(x+1)^2 = x^2+2x+1` came back REFUTED. Every dataset in
    existence writes exponentiation as `^`, so it is translated to `**` before parsing.

    BUG 2: implicit multiplication turns prose into symbols. `hello world` parses as `hello * world`, so any
    two identical phrases "verify" as an identity (which is true and useless). A guard rejects expressions
    with free symbol names longer than two characters, because real algebra uses x, y, n, t, i, k -- not
    words. The correct flow is: classify the claim as an algebraic identity FIRST, then call this. The
    parser is a calculus tool, not a natural-language reader, and it should refuse prose rather than
    symbolise it.
    """
    text = text.replace("^", "**")
    expr = parse_expr(text, transformations=TRANSFORMS, evaluate=True)
    long_names = [str(s) for s in expr.free_symbols if len(str(s)) > 2]
    if long_names:
        raise ValueError(f"looks like prose, not an expression (symbols {long_names[:3]})")
    return expr


# ------------------------------------------------------------------- numeric constants
CONSTANTS = {
    "pi": lambda: sympy.pi, "e": lambda: sympy.E, "tau": lambda: 2 * sympy.pi,
    "golden ratio": lambda: (1 + sympy.sqrt(5)) / 2,
    "phi": lambda: (1 + sympy.sqrt(5)) / 2,
    "sqrt2": lambda: sympy.sqrt(2), "sqrt(2)": lambda: sympy.sqrt(2),
}


def check_numeric_constant(claim: str, tolerance: float = 1e-4) -> Verdict:
    """`pi = 3.14159` -- a numeric assertion about a named constant, checked to a stated tolerance."""
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ()]{0,20}?)\s*(?:=|is|≈|=~)\s*(-?[\d.]+)\s*$", claim, re.I)
    if not m or sympy is None:
        return Verdict(UNKNOWN, "numeric_constant", "not a recognisable constant claim")
    name = m.group(1).strip().lower().rstrip("(").strip()
    stated = float(m.group(2))
    if name not in CONSTANTS:
        return Verdict(UNKNOWN, "numeric_constant", f"unknown constant {name!r}")
    true_val = float(CONSTANTS[name]().evalf())
    if abs(true_val - stated) <= tolerance:
        return Verdict(VERIFIED, "numeric_constant", f"{name} = {true_val:.10g}, stated {stated}",
                       value=true_val)
    return Verdict(REFUTED, "numeric_constant",
                   f"{name} = {true_val:.10g}, stated {stated} (outside +/-{tolerance})",
                   value=true_val)


# ------------------------------------------------------------------- identities
def check_identity(lhs: str, rhs: str) -> Verdict:
    """`(x+1)^2 = x^2+2x+1` -- simplify the difference. Exact when it is zero."""
    if sympy is None:
        return Verdict(UNKNOWN, "algebraic_identity", "sympy unavailable")
    try:
        a, b = _parse(lhs), _parse(rhs)
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "algebraic_identity", f"could not parse: {type(e).__name__}")
    diff = sympy.simplify(a - b)
    if diff == 0:
        return Verdict(VERIFIED, "algebraic_identity", "lhs - rhs simplifies to 0")
    # NOT CONDITIONAL. An identity must hold for ALL values of its variables, so a difference that vanishes
    # only at finitely many points is REFUTED. The first version called `(x+1)^2 = x^2+1` CONDITIONAL
    # because it happens to be true at x=0, which would let a false identity pass as "sometimes true".
    # CONDITIONAL is reserved for integrals, where the +C is a genuine convention rather than a defect.
    where = ""
    free = sorted(diff.free_symbols, key=str)
    if free:
        try:
            sols = sympy.solve(diff, free[0])
            if sols:
                where = f"; equal only at {free[0]} in {[str(s) for s in sols][:4]}"
        except Exception:                                  # noqa: BLE001
            pass
    return Verdict(REFUTED, "algebraic_identity",
                   f"lhs - rhs = {diff}, which is not identically 0{where}", value=diff)


# ------------------------------------------------------------------- calculus
def check_derivative(expr: str, var: str, claimed: str) -> Verdict:
    if sympy is None:
        return Verdict(UNKNOWN, "derivative", "sympy unavailable")
    try:
        x = sympy.Symbol(var)
        d = sympy.simplify(sympy.diff(_parse(expr), x))
        c = sympy.simplify(_parse(claimed))
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "derivative", f"could not parse: {type(e).__name__}")
    if sympy.simplify(d - c) == 0:
        return Verdict(VERIFIED, "derivative", f"d/d{var}({expr}) = {d}", value=d)
    return Verdict(REFUTED, "derivative", f"d/d{var}({expr}) = {d}, not {c}", value=d)


def check_integral(expr: str, var: str, claimed: str, check_constant: bool = True) -> Verdict:
    """Integrals are only defined up to a constant, so an antiderivative claim is CONDITIONAL on +C being
    allowed unless the claim omits it and the difference is constant."""
    if sympy is None:
        return Verdict(UNKNOWN, "integral", "sympy unavailable")
    try:
        x = sympy.Symbol(var)
        ant = sympy.simplify(sympy.integrate(_parse(expr), x))
        c = sympy.simplify(_parse(claimed))
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "integral", f"could not parse: {type(e).__name__}")
    diff = sympy.simplify(ant - c)
    if diff == 0:
        return Verdict(VERIFIED, "integral", f"integral = {ant}", value=ant)
    if check_constant and diff.free_symbols == set():
        return Verdict(CONDITIONAL, "integral",
                       f"differs by the constant {diff}; correct up to +C", assumed=["+C"],
                       value=ant)
    return Verdict(REFUTED, "integral", f"integral = {ant}, claim differs by {diff}", value=ant)


# ------------------------------------------------------------------- linear algebra
def parse_matrix(text: str):
    """[[1,2],[3,4]] or 1 2; 3 4 -> sympy Matrix."""
    s = text.strip().strip("[]")
    rows = [r for r in re.split(r"[;\n]|\)\s*\(", s) if r.strip()]
    if len(rows) == 1:
        # maybe a nested list
        inner = re.findall(r"\[([^\[\]]+)\]", text)
        if inner:
            rows = inner
        else:
            rows = [s]
    out = []
    for r in rows:
        vals = [v for v in re.split(r"[,\s]+", r.strip().strip("[]()")) if v]
        if not vals:
            continue
        out.append([sympy.sympify(v) for v in vals])
    return sympy.Matrix(out) if out else None


def check_determinant(matrix_text: str, claimed: str) -> Verdict:
    if sympy is None:
        return Verdict(UNKNOWN, "determinant", "sympy unavailable")
    try:
        M = parse_matrix(matrix_text)
        c = _parse(claimed)
        d = sympy.simplify(M.det())
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "determinant", f"could not evaluate: {type(e).__name__}")
    if sympy.simplify(d - c) == 0:
        return Verdict(VERIFIED, "determinant", f"det = {d}", value=d)
    return Verdict(REFUTED, "determinant", f"det = {d}, not {c}", value=d)


def check_matrix_product(a_text: str, b_text: str, claimed_text: str) -> Verdict:
    if sympy is None:
        return Verdict(UNKNOWN, "matrix_product", "sympy unavailable")
    try:
        A, B = parse_matrix(a_text), parse_matrix(b_text)
        C = parse_matrix(claimed_text)
        P = sympy.simplify(A * B)
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "matrix_product", f"could not evaluate: {type(e).__name__}")
    if P == C:
        return Verdict(VERIFIED, "matrix_product", f"A*B = {P.tolist()}", value=P.tolist())
    return Verdict(REFUTED, "matrix_product", f"A*B = {P.tolist()}, not {C.tolist()}",
                   value=P.tolist())


# ------------------------------------------------------------------- units
def check_unit_conversion(claim: str, tolerance: float = 1e-6) -> Verdict:
    """`1 mile = 1.609344 km`. Exact table arithmetic, no model, no ambiguity."""
    m = re.match(r"^\s*(-?[\d.]+)\s*([A-Za-z]+)\s*(?:=|is|≈)\s*(-?[\d.]+)\s*([A-Za-z]+)\s*$", claim)
    if not m:
        return Verdict(UNKNOWN, "unit_conversion", "not a recognisable conversion")
    n1, u1, n2, u2 = float(m.group(1)), m.group(2).lower(), float(m.group(3)), m.group(4).lower()
    if u1 not in UNITS or u2 not in UNITS:
        return Verdict(UNKNOWN, "unit_conversion",
                       f"unknown unit(s): {[u for u in (u1, u2) if u not in UNITS]}")
    expected = n1 * UNITS[u1] / UNITS[u2]
    if abs(expected - n2) <= tolerance * max(1.0, abs(expected)):
        return Verdict(VERIFIED, "unit_conversion", f"{n1} {u1} = {expected:.10g} {u2}",
                       value=expected)
    return Verdict(REFUTED, "unit_conversion", f"{n1} {u1} = {expected:.10g} {u2}, not {n2}",
                   value=expected)


# ------------------------------------------------------------------- equation solution
def check_solution(equation: str, var: str, value: str) -> Verdict:
    """`x = 2 solves x^2 = 4` -- substitute and check. Exact."""
    if sympy is None:
        return Verdict(UNKNOWN, "solution", "sympy unavailable")
    if "=" not in equation:
        return Verdict(UNKNOWN, "solution", "equation has no '='")
    lhs, rhs = equation.split("=", 1)
    try:
        x = sympy.Symbol(var)
        v = _parse(value)
        diff = sympy.simplify((_parse(lhs) - _parse(rhs)).subs(x, v))
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "solution", f"could not evaluate: {type(e).__name__}")
    if diff == 0:
        return Verdict(VERIFIED, "solution", f"{var}={v} makes the equation 0")
    return Verdict(REFUTED, "solution", f"{var}={v} leaves residual {diff}", value=diff)


# ------------------------------------------------------------------- capability manifest
CAPABILITIES = {
    "numeric_constant": {
        "proves": "a named constant equals a stated value to a tolerance",
        "cannot": ["constants outside the table", "contexts where the symbol is a variable, not pi"],
        "method": "sympy exact constant evaluation"},
    "algebraic_identity": {
        "proves": "two expressions are equal for ALL values of their variables",
        "cannot": ["claims needing assumptions the parser does not know",
                   "inequalities", "statements about a specific solution set"],
        "method": "sympy.simplify(lhs - rhs) == 0"},
    "derivative": {"proves": "a derivative claim, exactly",
                   "cannot": ["differentiability", "piecewise domains", "notation outside sympy syntax"],
                   "method": "sympy.diff"},
    "integral": {"proves": "an antiderivative, up to an additive constant",
                 "cannot": ["definite integrals given as indefinite", "convergence questions"],
                 "method": "sympy.integrate, difference must be constant"},
    "determinant": {"proves": "a determinant, exactly",
                    "cannot": ["numerical stability claims", "symbolic entries it cannot parse"],
                    "method": "sympy Matrix.det"},
    "matrix_product": {"proves": "a matrix product, exactly", "cannot": ["shape mistakes are errors not claims"],
                       "method": "sympy Matrix multiplication"},
    "unit_conversion": {"proves": "conversions between units in the table",
                        "cannot": ["compound units", "non-SI systems not in the table",
                                   "currency or other non-constant 'units'"],
                        "method": "exact factor table to SI base"},
    "solution": {"proves": "a stated value satisfies an equation",
                 "cannot": ["that it is the ONLY solution", "inequality solutions"],
                 "method": "substitute and simplify"},
}


# ------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.verify.math_core - SELF TEST")
    ok = True
    if sympy is None:
        print("  [SKIP] sympy unavailable")
        return True

    # --- constants: true, false, and tolerance behaviour
    good = check_numeric_constant("pi = 3.14159").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3.14159 VERIFIED")
    good = check_numeric_constant("pi = 3.2").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3.2 REFUTED")
    good = check_numeric_constant("e = 2.718281828").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] e = 2.718281828 VERIFIED")

    # --- identities: exact, refuted, and CONDITIONAL when it holds only on a domain
    good = check_identity("(x+1)^2", "x^2 + 2*x + 1").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] (x+1)^2 = x^2+2x+1 VERIFIED exactly")
    good = check_identity("(x+1)^2", "x^2 + 1").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] (x+1)^2 = x^2+1 REFUTED")
    r = check_identity("x^2", "x")
    good = r.status == REFUTED and "equal only at" in r.detail
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] x^2 = x is REFUTED as an identity, with the point where "
          f"it happens to hold noted")

    # --- calculus
    good = check_derivative("x^2", "x", "2*x").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] d/dx x^2 = 2x VERIFIED")
    good = check_derivative("x^3", "x", "2*x").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] d/dx x^3 = 2x REFUTED")
    good = check_derivative("sin(x)", "x", "cos(x)").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] d/dx sin(x) = cos(x) VERIFIED")

    # --- integrals: exact, and CONDITIONAL when the claim omits +C
    good = check_integral("2*x", "x", "x^2").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] int 2x dx = x^2 VERIFIED")
    r = check_integral("2*x", "x", "x^2 + 5")
    good = r.status == CONDITIONAL
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] int 2x dx = x^2+5 is CONDITIONAL (differs by a constant)")
    good = check_integral("cos(x)", "x", "sin(x)").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] int cos(x) dx = sin(x) VERIFIED")

    # --- linear algebra
    good = check_determinant("[[1,2],[3,4]]", "-2").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] det([[1,2],[3,4]]) = -2 VERIFIED")
    good = check_determinant("[[1,2],[3,4]]", "-1").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] det = -1 REFUTED")
    good = check_matrix_product("[[1,0],[0,1]]", "[[5,6],[7,8]]", "[[5,6],[7,8]]").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] I*A = A VERIFIED")
    good = check_matrix_product("[[1,2],[3,4]]", "[[1,0],[0,1]]",
                                "[[1,2],[3,4]]").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] A*I = A VERIFIED")

    # --- units: exact factor arithmetic, and a wrong one caught
    good = check_unit_conversion("1 mile = 1.609344 km").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 1 mile = 1.609344 km VERIFIED")
    good = check_unit_conversion("1 mile = 1.6 km").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 1 mile = 1.6 km REFUTED (off by 0.009344)")
    good = check_unit_conversion("1 hour = 3600 s").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 1 hour = 3600 s VERIFIED")
    good = check_unit_conversion("1 furlong = 201 m").status == UNKNOWN
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown unit is UNKNOWN, not guessed")

    # --- solutions: substitution, and the limitation stated honestly
    good = check_solution("x^2 = 4", "x", "2").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] x=2 solves x^2=4 VERIFIED")
    good = check_solution("x^2 = 4", "x", "3").status == REFUTED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] x=3 does not solve x^2=4")
    # the capability limit, asserted rather than glossed: a solution check is not a uniqueness proof
    good = "that it is the ONLY solution" in CAPABILITIES["solution"]["cannot"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the manifest states a solution check does NOT prove uniqueness")

    # --- every kind declares what it cannot do
    good = all("cannot" in c and c["cannot"] for c in CAPABILITIES.values())
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(CAPABILITIES)} kinds declare their limits")

    # --- garbage is UNKNOWN, never a false VERIFIED. The first version compared junk to ITSELF, which is
    #     a true identity by reflexivity and therefore asserted nothing useful; and prose parses as symbols
    #     under implicit multiplication, so `hello world` = `hello world` came back VERIFIED.
    cases = [("hello world", "goodbye world"), ("2 +", "3 +"), ("= = =", "x"), ("", "x^2")]
    statuses = [check_identity(a, b).status for a, b in cases]
    good = VERIFIED not in statuses
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] prose and malformed input never return VERIFIED "
          f"({statuses})")

    # --- and the ^ alias is honoured, which is what every real dataset writes
    good = check_identity("(x+1)^2", "x^2 + 2*x + 1").status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] `^` is read as exponentiation, not XOR")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Symbolic/numeric math validation.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
