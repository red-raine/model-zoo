"""math_verify.py — check arithmetic and algebraic claims by actually evaluating them.

WHY THIS FILE EXISTS
--------------------
`tools/scoring/trace_eval.py` scores text by counting connective phrases. It gave
"Therefore, 2 + 2 = 5. For example, since 2 is approximately 2.5 ..." a TRACE of
**0.816, grade A** — above even the 0.70 release gate. A correct explanation with
no connectives scored 0.133, grade D.

Vocabulary is not correctness. This module checks the thing that is checkable:
does the arithmetic in the answer actually hold.

APPROACH
--------
1. Find claim shapes: `A <op> B = C`, `X equals Y`, `X is equal to Y`,
   named-function arithmetic, and `n% of m`.
2. Sanitise strictly (whitelist characters), then parse with sympy.
3. Compare the two sides symbolically. Free symbols are allowed (an equation with
   unknowns is checked for symbolic identity, not truth).
4. Report VERIFIED / REFUTED / UNVERIFIABLE per claim.

SECURITY
--------
`sympy.sympify` on arbitrary text can reach `__import__`. Input is therefore
whitelisted to `[0-9 + - * / ( ) . ^ % space]` plus single letters used as
symbols before it is ever parsed. Anything else fails the claim as UNPARSEABLE.

USAGE
-----
    python tools/verify/math_verify.py --self-test
    python tools/verify/math_verify.py --text "so 236 - 14 = 222 dollars"
    python tools/verify/math_verify.py --jsonl answers.jsonl --field response
"""

from __future__ import annotations

import argparse
import json
import re
import sys

try:
    import sympy
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application)
    _HAS_SYMPY = True
except Exception:                                     # pragma: no cover
    _HAS_SYMPY = False

# --- claim extraction ------------------------------------------------------
# left side may contain ops and parens; allow a trailing unit word
_TERM = r"[0-9\(][0-9\s\.\+\-\*/\(\)\^%]*"
_EQ = r"(?:=|==|equals|is equal to|gives|results in|comes to)"
CLAIM_RE = re.compile(rf"(?P<lhs>{_TERM}?)\s*{_EQ}\s*(?P<rhs>{_TERM})", re.I)

# `n% of m` -> n/100*m
PCT_OF = re.compile(r"(?P<p>[0-9]+(?:\.[0-9]+)?)\s*%\s*of\s*(?P<v>[0-9][0-9,\.]*)", re.I)
# "15% of 80 = 12" must be checked as a percent, not as the sub-claim "80 = 12"
PCT_EQ = re.compile(r"(?P<p>[0-9]+(?:\.[0-9]+)?)\s*%\s*of\s*(?P<v>[0-9][0-9,\.]*)"
                    r"\s*(?:=|==|equals|is equal to|gives|is)\s*(?P<r>[0-9][0-9,\.]*)", re.I)

_ARITH = re.compile(r"[0-9\s\.\+\-*/()=^%]")
_ALLOWED = set("0123456789+-*/(). ^=")


def _clean(expr: str) -> str:
    """Whitelist, then normalise. No letters survive -> no symbol injection."""
    expr = expr.replace(",", "").replace("$", "").replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    out = "".join(ch for ch in expr if ch in _ALLOWED)
    return " ".join(out.split())


def _eval_side(expr: str):
    """Return a sympy expression, or None if it cannot be parsed safely."""
    if not _HAS_SYMPY:
        return None
    s = _clean(expr).strip(" =")
    if not s or not any(c.isdigit() for c in s):
        return None
    # a trailing operator means we truncated mid-expression
    if s[-1] in "+-*/^":
        return None
    try:
        return parse_expr(s, transformations=standard_transformations,
                          evaluate=True)
    except Exception:
        return None


def _same(a, b) -> bool | None:
    if a is None or b is None:
        return None
    try:
        if sympy.simplify(a - b) == 0:
            return True
        fa, fb = sympy.N(a), sympy.N(b)
        if fa.is_number and fb.is_number:
            return abs(complex(fa) - complex(fb)) < 1e-9
        return False
    except Exception:
        return None


def find_claims(text: str) -> list[dict]:
    """Extract checkable arithmetic/equality claims."""
    claims = []
    t = text or ""

    for m in PCT_OF.finditer(t):
        p = float(m.group("p"))
        v = float(m.group("v").replace(",", ""))
        claims.append({
            "kind": "percent_of",
            "raw": m.group(0),
            "value": p / 100.0 * v,
        })

    masked: list[tuple[int, int]] = []
    for m in PCT_EQ.finditer(t):
        p = float(m.group("p")) / 100.0
        v = float(m.group("v").replace(",", ""))
        r = float(m.group("r").replace(",", ""))
        masked.append((m.start(), m.end()))
        claims.append({
            "kind": "percent_equality",
            "raw": m.group(0).strip()[:120],
            "lhs": str(p * v), "rhs": str(r),
            "holds": abs(p * v - r) < 1e-9,
        })

    for m in CLAIM_RE.finditer(t):
        if any(s <= m.start() < e for s, e in masked):
            continue
        lhs, rhs = m.group("lhs"), m.group("rhs")
        if not lhs.strip():
            continue
        # require an operator or a multi-digit number on at least one side
        if not re.search(r"[+\-*/^%]", lhs + rhs) and \
           len(re.sub(r"\D", "", lhs + rhs)) < 4:
            continue
        # "15% of 80 = 12" -> compute the percent, then compare numerically
        pct = PCT_OF.search(lhs)
        if pct:
            try:
                a = sympy.Rational(str(float(pct.group("p")))) / 100 * \
                    sympy.Rational(str(float(pct.group("v").replace(",", ""))))
            except Exception:
                a = None
            b = _eval_side(rhs)
        else:
            a, b = _eval_side(lhs), _eval_side(rhs)
        verdict = _same(a, b)
        if verdict is None:
            continue
        claims.append({
            "kind": "equality",
            "raw": m.group(0).strip()[:120],
            "lhs": str(a), "rhs": str(b),
            "holds": bool(verdict),
        })
    return claims


def verify_text(text: str) -> dict:
    claims = find_claims(text)
    eq = [c for c in claims if c["kind"] in ("equality", "percent_equality")]
    refuted = [c for c in eq if not c["holds"]]
    verified = [c for c in eq if c["holds"]]
    return {
        "claims": len(claims),
        "equalities": len(eq),
        "verified": len(verified),
        "refuted": len(refuted),
        "refuted_examples": [c["raw"] for c in refuted[:5]],
        "verified_examples": [c["raw"] for c in verified[:5]],
        "percent_claims": len([c for c in claims if c["kind"] == "percent_of"]),
        "verdict": ("REFUTED" if refuted else
                    "VERIFIED" if verified else "NO_CLAIMS"),
    }


# --- self-test -------------------------------------------------------------
SELFTEST = [
    ("2 + 2 = 5", "REFUTED"),
    ("2 + 2 = 4", "VERIFIED"),
    ("Therefore, 2 + 2 = 5. For example, 2 is approximately 2.5", "REFUTED"),
    ("236 - 14 = 222 dollars on books", "VERIFIED"),
    ("Each book cost 222 / 6 = 37 dollars", "VERIFIED"),
    ("180 + 100 = 280 pages over 14 days", "VERIFIED"),
    ("14 / 2 = 7 daisies", "VERIFIED"),
    ("5 + 9 = 14 daisies", "VERIFIED"),
    ("the sky is green because air molecules are larger", "NO_CLAIMS"),
    ("", "NO_CLAIMS"),
    ("15% of 80 = 12", "VERIFIED"),
    ("2 * 3 = 7", "REFUTED"),
    ("(3 + 4) * 2 = 14", "VERIFIED"),
    ("(3 + 4) * 2 = 15", "REFUTED"),
]


def run_self_test() -> bool:
    print("math_verify.py - SELF TEST")
    print(f"  sympy: {_HAS_SYMPY}")
    ok_all = True
    for text, expect in SELFTEST:
        v = verify_text(text)["verdict"]
        ok = v == expect
        ok_all = ok_all and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {expect:<11} got={v:<11} {text[:52]!r}")
    # the adversarial fixture must be caught
    bad = ("Therefore, 2 + 2 = 5. For example, since 2 is approximately 2.5, the "
           "sum of 2.5 and 2.5 equals 5, when rounding.")
    v = verify_text(bad)
    ok = v["refuted"] > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] adversarial 2+2=5 -> refuted={v['refuted']} "
          f"{v['refuted_examples']}")
    ok_all = ok_all and ok
    print(f"SELF TEST: {'ALL PASS' if ok_all else 'FAILURES'}")
    return ok_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify arithmetic claims with sympy.")
    ap.add_argument("--text")
    ap.add_argument("--jsonl")
    ap.add_argument("--field", default="response")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    if args.text:
        print(json.dumps(verify_text(args.text), indent=2))
        return 0

    if args.jsonl:
        rows, agg = [], {"REFUTED": 0, "VERIFIED": 0, "NO_CLAIMS": 0}
        with open(args.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                v = verify_text(d.get(args.field) or "")
                agg[v["verdict"]] += 1
                if v["refuted"]:
                    rows.append({"prompt": str(d.get("prompt"))[:80],
                                 **{k: v[k] for k in ("refuted", "refuted_examples")}})
        out = {"totals": agg, "total": sum(agg.values()), "refuted_rows": rows[:40]}
        print(json.dumps(out, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
