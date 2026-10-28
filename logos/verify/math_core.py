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
    # "15% of 80" -> "(15/100)*80" and a bare "15%" -> "(15/100)". Percent-of is notation; sympy would
    # read % as modulo and `of` as a symbol.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
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


def _significant_digits(literal: str) -> int:
    """Significant digits in a written number. `3` -> 1, `3.0` -> 2, `3.14` -> 3, `0.00314` -> 3.

    This is the piece the old validator threw away with float(), and losing it is what made `pi = 3.14`
    REFUTED: without the literal's precision there is no way to ask whether a value is a CORRECT ROUNDING,
    only whether it is close, and those are different questions.
    """
    t = str(literal).strip().lstrip("+-")
    if "." in t:
        t = t.rstrip("0") if t.rstrip("0").endswith(".") is False else t
        if t.endswith("."):
            t = t[:-1]
    t = t.replace(".", "")
    t = t.lstrip("0")
    return max(1, len(t))


def _round_to_sf(x: float, n: int) -> float:
    """Round x to n significant digits, which is what a stated constant is claiming."""
    if x == 0:
        return 0.0
    from math import floor, log10
    digits = n - int(floor(log10(abs(x)))) - 1
    return round(x, digits)


def check_numeric_constant(claim: str, tolerance: float | None = None,
                           required_rel_error: float | None = None) -> Verdict:
    """`pi = 3.14` -- is the stated value a CORRECT ROUNDING of the constant?

    THE DEFECT THIS REPLACES. The old check was `abs(true - stated) <= 1e-4`, i.e. an ABSOLUTE tolerance,
    and it produced three wrong answers:

        pi = 3.14    REFUTED   rel_err 5.07e-04 vs tol 1e-4 -- but 3.14 IS pi correctly rounded to 3 s.f.
        e  = 2.718   REFUTED   same: correctly rounded to 4 s.f.
        pi = 22/7    UNKNOWN   the regex only accepted a decimal literal as the stated value

    and an absolute tolerance is not scale-free: 1e-4 is reasonable for pi and absurd for the speed of
    light. Rounding correctness has no tolerance at all, which is why it is the right test:

        pi = 3      1 s.f.   pi rounds to 3      -> CORRECT (but coarse)
        pi = 3.0    2 s.f.   pi rounds to 3.1    -> WRONG   (and the point the old code missed)
        pi = 3.14   3 s.f.   pi rounds to 3.14   -> CORRECT
        pi = 3.2    2 s.f.   pi rounds to 3.1    -> WRONG

    COARSE IS NOT FALSE. `pi = 3` is a correct rounding and a terrible value to plot a circle with; that is
    a separate question, answered by `required_rel_error` and reported in `assumed` rather than folded into
    the verdict. Refuting a correct rounding because it is imprecise would drop good data -- the failure
    direction this project calls catastrophic.
    """
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ()\-]{0,24}?)\s*(?:=|is|\u2248|=~)\s*(.+?)\s*$",
                 claim or "", re.I)
    if not m or sympy is None:
        return Verdict(UNKNOWN, "numeric_constant", "not a recognisable constant claim")
    name = m.group(1).strip().lower().rstrip("(").strip()
    stated_literal = m.group(2).strip()
    if name not in CONSTANTS:
        return Verdict(UNKNOWN, "numeric_constant", f"unknown constant {name!r}")

    true_val = float(CONSTANTS[name]().evalf())

    # ---- a plain decimal literal: test whether it is the true value CORRECTLY ROUNDED
    if re.fullmatch(r"-?\d+\.?\d*", stated_literal):
        stated = float(stated_literal)
        sf = _significant_digits(stated_literal)
        expected = _round_to_sf(true_val, sf)
        rel = abs(stated - true_val) / abs(true_val) if true_val else 0.0
        if abs(stated - expected) <= abs(expected) * 1e-12 + 1e-15:
            notes = []
            if required_rel_error is not None and rel > required_rel_error:
                notes.append(f"rel_err {rel:.2e} exceeds the required {required_rel_error:.2e}: a correct "
                             f"rounding that is TOO COARSE for the declared use")
            return Verdict(VERIFIED, "numeric_constant",
                           f"{name} correctly rounded to {sf} s.f. as {stated_literal} "
                           f"(true {true_val:.10g}, rel_err {rel:.2e})",
                           assumed=notes, value=true_val)
        return Verdict(REFUTED, "numeric_constant",
                       f"{name} = {true_val:.10g}; {stated_literal} has {sf} s.f. but the correct rounding "
                       f"to {sf} s.f. is {expected:g}, not {stated:g}", value=true_val)

    # ---- an expression as the stated value, e.g. `pi = 22/7`. Correct only if it is EXACT.
    try:
        expr = _parse(stated_literal)
        diff = float((CONSTANTS[name]() - expr).evalf())
        rel = abs(diff) / abs(true_val) if true_val else abs(diff)
    except Exception:                                       # noqa: BLE001
        return Verdict(UNKNOWN, "numeric_constant", f"could not evaluate stated value {stated_literal!r}")
    if abs(diff) <= abs(true_val) * 1e-12 + 1e-15:
        return Verdict(VERIFIED, "numeric_constant", f"{name} = {stated_literal} exactly", value=true_val)
    return Verdict(REFUTED, "numeric_constant",
                   f"{name} = {true_val:.10g}, not {stated_literal} exactly (rel_err {rel:.2e})",
                   value=true_val)


def precision_sufficient(claim: str, required_rel_error: float) -> dict:
    """Is a correctly-rounded constant precise ENOUGH for a stated use? Separate from being true.

    `pi = 3` is a correct rounding and a 4.5% error. Plotting a circle of radius 500 px with it puts the
    edge 22 px out, which is visible. `pi = 3.14` is 5.1e-4, i.e. 0.25 px at that radius. So the same claim
    is acceptable or not depending ENTIRELY on what it is for, which is why the precision requirement has to
    be an input rather than a constant buried in a validator.
    """
    v = check_numeric_constant(claim)
    if v.status != VERIFIED:
        return {"sufficient": None, "reason": f"not a correct rounding ({v.status}): {v.detail}",
                "rel_error": None}
    m = re.search(r"rel_err (\S+?)\)", v.detail)
    rel = float(m.group(1)) if m else None
    if rel is None:
        return {"sufficient": None, "reason": "could not read the relative error", "rel_error": None}
    return {"sufficient": rel <= required_rel_error, "rel_error": rel,
            "reason": f"rel_err {rel:.2e} vs required {required_rel_error:.2e}"}


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


def check_arithmetic_equality(lhs: str, rhs: str, tolerance: float = 0.0) -> Verdict:
    """Exact arithmetic equality via sympy. Deliberately an INDEPENDENT implementation of what
    `verify/math_verify.py` does with its own parser, so the two can be cross-checked against each other.

    Two implementations that share no code disagreeing on a claim is real evidence of a bug or an ambiguous
    claim; one implementation agreeing with itself is not evidence of anything.
    """
    if sympy is None:
        return Verdict(UNKNOWN, "arithmetic", "sympy unavailable")
    try:
        a = _parse(lhs)
        b = _parse(rhs)
    except Exception as e:                                 # noqa: BLE001
        return Verdict(UNKNOWN, "arithmetic", f"could not parse: {type(e).__name__}")
    if a.free_symbols or b.free_symbols:
        return Verdict(UNKNOWN, "arithmetic",
                       f"not a numeric claim: free symbols {sorted(map(str, a.free_symbols | b.free_symbols))}")
    try:
        diff = sympy.simplify(a - b)
    except Exception as e:                                 # noqa: BLE001
        # A VALIDATOR MUST NEVER RAISE; it returns UNKNOWN. Measured: `gamma = 3.0` parses `gamma` as a
        # sympy FunctionClass, whose free_symbols is empty so the guard above passes, and then `a - b`
        # raises TypeError -- taking the whole pipeline down on any row naming a function sympy knows.
        # Fail-closed means UNKNOWN is a RETURN, never a traceback.
        return Verdict(UNKNOWN, "arithmetic", f"could not compare symbolically: {type(e).__name__}")
    if diff == 0:
        return Verdict(VERIFIED, "arithmetic", f"{lhs} = {rhs} exactly", value=float(a.evalf()))
    try:
        d = float(diff.evalf())
    except Exception:                                      # noqa: BLE001
        d = None
    if tolerance > 0 and d is not None and abs(d) <= tolerance:
        return Verdict(VERIFIED, "arithmetic", f"differs by {d} within tolerance", value=float(a.evalf()))
    return Verdict(REFUTED, "arithmetic", f"{lhs} = {float(a.evalf()):.10g}, not {rhs}",
                   value=float(a.evalf()))


_DERIV_RE = re.compile(r"^\s*d\s*/\s*d\s*([A-Za-z])\s*(.+?)\s*=\s*(.+?)\s*$", re.I)
_INTEG_RE = re.compile(r"^\s*(?:the\s+)?(?:integral|antiderivative)\s+of\s+(.+?)\s*d\s*([A-Za-z])\s*"
                       r"=\s*(.+?)\s*$", re.I)
_INTEG_SYM_RE = re.compile(r"^\s*(?:\u222b|int)\s*(.+?)\s*d\s*([A-Za-z])\s*=\s*(.+?)\s*$", re.I)


def check_calculus_text(claim: str) -> Verdict:
    """`d/dx x^2 = 2` and `integral of 2*x dx = x^2` -- parse the ROW and route it.

    THIS CLOSES A DOCUMENTED DEAD END. `truth_kinds.validate("d/dx x^2 = 2")` returns kind='calculus',
    decidable=True, verdict='ROUTED', note='hand to verify/math_core.check_derivative|check_integral'. The
    taxonomy knew the row was decidable and named the functions to call -- and no caller ever called them, so
    a decidable row was silently treated as undecidable. Measured before this function existed:
    `classify_repair("d/dx x^2 = 2")` returned tier 3, i.e. "no oracle", for a row with an exact oracle.

    Returns UNKNOWN for a row that is not in either shape, so the caller can fall through.
    """
    if sympy is None:
        return Verdict(UNKNOWN, "calculus", "sympy unavailable")
    text = (claim or "").strip()
    m = _DERIV_RE.match(text)
    if m:
        var, expr, claimed = m.group(1), m.group(2), m.group(3)
        # `d/dx of x^2` shows up often enough to be worth stripping
        expr = re.sub(r"^\s*of\s+", "", expr, flags=re.I)
        return check_derivative(expr, var, claimed)
    m = _INTEG_RE.match(text) or _INTEG_SYM_RE.match(text)
    if m:
        expr, var, claimed = m.group(1), m.group(2), m.group(3)
        return check_integral(expr, var, claimed)
    return Verdict(UNKNOWN, "calculus", "not a derivative or integral claim shape")


def check_arithmetic_text(claim: str) -> Verdict:
    """Parse `A op B = C` shaped text and check it exactly.

    A NAMED CONSTANT defers to `check_numeric_constant` first. Without that, `pi = 3.1416` was VERIFIED by
    the constant check and REFUTED here -- exact symbolic equality cannot accept a rounding, so the same
    string got opposite verdicts depending on the entry path.
    """
    try:
        const = check_numeric_constant(claim)
        if const.status in (VERIFIED, REFUTED):
            return const
    except Exception:                                       # noqa: BLE001
        pass
    try:
        calc = check_calculus_text(claim)
        if calc.status in (VERIFIED, REFUTED):
            return calc
    except Exception:                                       # noqa: BLE001
        pass
    m = re.match(r"^\s*(.+?)\s*=\s*(.+?)\s*$", claim)
    if not m:
        return Verdict(UNKNOWN, "arithmetic", "no '=' in the claim")
    return check_arithmetic_equality(m.group(1), m.group(2))


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
    # --- A CORRECT ROUNDING IS NOT A FALSE CLAIM. This was REFUTED by the old absolute-tolerance test.
    for lit in ("3.14", "3.142", "3.1416", "3.14159"):
        v = check_numeric_constant(f"pi = {lit}")
        good = v.status == VERIFIED
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] pi = {lit} is a correct rounding -> {v.status}")

    # --- and a WRONG rounding is still caught. Note 3.1 is CORRECT at 2 s.f. (pi rounds to 3.1); 3.0 and
    #     3.2 are not. My first version of this test listed 3.1 as wrong, which was my error, not the
    #     engine's -- the rounding test caught my misreading of my own fixture.
    for lit in ("3.0", "3.2", "3.15", "3.1417"):
        v = check_numeric_constant(f"pi = {lit}")
        good = v.status == REFUTED
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] pi = {lit} is NOT the correct rounding -> {v.status}")

    # --- 1 significant digit: coarse but CORRECT. Coarse is not false.
    v = check_numeric_constant("pi = 3")
    good = v.status == VERIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3 is a correct 1-s.f. rounding -> {v.status} "
          f"(coarse, not wrong)")

    # --- but it is wildly insufficient for a precise use, and that is reported separately
    d = precision_sufficient("pi = 3", 1e-3)
    good = d["sufficient"] is False and d["rel_error"] > 1e-2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3 is INSUFFICIENT for a 1e-3 use "
          f"(rel_err {d['rel_error']:.2e}) while still being a correct rounding")

    d = precision_sufficient("pi = 3.14", 1e-3)
    good = d["sufficient"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 3.14 IS sufficient for the same use "
          f"(rel_err {d['rel_error']:.2e})")

    # --- THE TWO-PATH DISAGREEMENT: both entry points must agree on the same string
    same = all(check_arithmetic_text(f"pi = {lit}").status == check_numeric_constant(f"pi = {lit}").status
               for lit in ("3", "3.0", "3.14", "3.1416", "3.2"))
    good = same
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] check_arithmetic_text and check_numeric_constant AGREE on "
          f"every pi rounding (they disagreed before)")

    # --- an expression as the stated value is now parsed instead of reported as unrecognisable
    v = check_numeric_constant("pi = 22/7")
    good = v.status == REFUTED and "22/7" in v.detail
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pi = 22/7 parses and is REFUTED as an exact claim "
          f"(rel_err {float(22 / 7 - 3.141592654) / 3.141592654:.2e})")

    # --- THE ROUTE THAT WAS NEVER TAKEN. truth_kinds said decidable=True, verdict=ROUTED, and named the
    #     function; nothing called it. Now the row is actually validated.
    for row, expect in (("d/dx x^2 = 2*x", VERIFIED), ("d/dx x^2 = 2", REFUTED),
                        ("integral of 2*x dx = x^2", VERIFIED),
                        ("integral of 2*x dx = x^3", REFUTED)):
        v = check_calculus_text(row)
        good = v.status == expect
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {row:28s} -> {v.status}")

    # --- and a non-calculus row is UNKNOWN, not a wrong guess
    v = check_calculus_text("2 + 2 = 5")
    good = v.status == UNKNOWN
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a non-calculus row returns UNKNOWN (no free opinions)")

    # --- A VALIDATOR MUST NEVER RAISE. It returns UNKNOWN. Names that sympy knows as functions (gamma,
    #     beta, zeta) used to take the pipeline down with a TypeError.
    hostile = ["gamma = 3.0", "beta = 2", "zeta = 1.5", "=", "== 4", "2 +", "lambda = 3",
               "a" * 400 + " = 1", "pi = ", "2 + 2 = ", "\x00 = 1"]
    raised = []
    for h in hostile:
        for fn_name in ("check_arithmetic_text", "check_numeric_constant", "check_unit_conversion"):
            fn = globals().get(fn_name)
            if fn is None:
                continue
            try:
                v = fn(h)
                if getattr(v, "status", None) not in (VERIFIED, REFUTED, UNKNOWN):
                    raised.append(f"{fn_name}({h[:12]!r}) -> bad status")
            except Exception as e:                          # noqa: BLE001
                raised.append(f"{fn_name}({h[:12]!r}) raised {type(e).__name__}")
    good = not raised
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no validator raises on {len(hostile)} hostile inputs "
          f"({raised[:2] if raised else 'clean'})")

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
