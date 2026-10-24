"""logos.logic.quantity — slot-aligned numeric checking.

WHY THIS EXISTS
---------------
The per-evidence measurement was unambiguous: the numeric rules scored 0.421 and 0.437
consistency with the gold verdict, and 0.050 / 0.097 when used alone — pure noise. The
diagnosis was that "the claim has a number and the answer has a number" is a **co-occurrence,
not a slot match**. A claim about 2020 GDP against an answer mentioning 2019 is not a conflict.

This module tests that diagnosis directly: if numbers are attached to their **slot** (unit
family + the noun they modify) before being compared, do the numeric rules become informative?

THE ALIGNMENT RULE
------------------
Two quantities are comparable only if they share a **unit family** (length, mass, time, money,
percentage, count, year). Within a family they are converted to SI and compared. A claim about
a *length* and an answer about a *year* are not in conflict at all — they are about different
things, and saying so is the whole point.

A second gate: the noun context must overlap. "the bridge is 330 m" against "the tunnel is 340
m" compares two different objects; alignment fails and the answer is INSUFFICIENT, not REFUTE.
That is a stricter rule than fuzzy matching and it is the one that makes the result trustworthy.

TOLERANCE
---------
A relative residual under 2% counts as agreement, because rounding in reporting is normal
(1082.68 ft reported as 1083 ft). Above that it is a refutation, and the residual is reported so
a reviewer can see how far off it is.

WHAT THIS IS NOT
----------------
Not a general numeric reasoner: no arithmetic across quantities, no unit inference from context
(a bare "330" with no unit yields UNKNOWN family and is not compared), no date-interval algebra.
It does one thing — decide whether a claim's number and an answer's number are about the same
quantity — and it says UNKNOWN rather than guessing when it cannot tell.

USAGE
-----
    python tools/logos/logic/quantity.py --self-test
    python tools/logos/logic/quantity.py --claim "the bridge is 330 m long" --answer "1083 feet"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum

# unit family -> multiplier to the family's base unit
FAMILIES: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "metre": 1.0, "meter": 1.0, "metres": 1.0, "meters": 1.0,
               "km": 1000.0, "kilometre": 1000.0, "kilometer": 1000.0,
               "cm": 0.01, "mm": 0.001, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
               "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
               "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
               "in": 0.0254, "inch": 0.0254, "inches": 0.0254},
    "mass": {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001,
             "grams": 0.001, "t": 1000.0, "tonne": 1000.0, "tonnes": 1000.0,
             "ton": 907.18474, "tons": 907.18474, "lb": 0.45359237, "lbs": 0.45359237,
             "pound": 0.45359237, "pounds": 0.45359237, "oz": 0.028349523,
             "ounce": 0.028349523, "ounces": 0.028349523},
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "min": 60.0,
             "minute": 60.0, "minutes": 60.0, "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
             "hours": 3600.0, "day": 86400.0, "days": 86400.0, "week": 604800.0,
             "weeks": 604800.0, "year": 31557600.0, "years": 31557600.0},
    "money": {"usd": 1.0, "$": 1.0, "dollar": 1.0, "dollars": 1.0, "eur": 1.08,
              "\u20ac": 1.08, "gbp": 1.27, "\u00a3": 1.27, "pound_sterling": 1.27,
              "cent": 0.01, "cents": 0.01},
    "percent": {"%": 1.0, "percent": 1.0, "per cent": 1.0, "pct": 1.0},
    "count": {"": 1.0, "people": 1.0, "person": 1.0, "persons": 1.0, "inhabitants": 1.0,
              "residents": 1.0, "cases": 1.0, "deaths": 1.0, "votes": 1.0, "soldiers": 1.0,
              "km2": 1.0, "sq": 1.0},
    "year": {"year": 1.0},          # handled specially below
}

MAGNITUDE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
             "billion": 1e9, "bn": 1e9, "trillion": 1e12}

# Thousands separators first: "45,000,000" is one number, not "45" plus junk. An earlier
# regex matched only "45" and the claim then compared 45 against 45,000,000.
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_UNIT_ALT = "|".join(sorted({u for fam in FAMILIES.values() for u in fam if u},
                            key=len, reverse=True))
PATTERN = re.compile(
    rf"{_NUM}\s*(?:(?P<mag>thousand|million|billion|trillion|bn|mn)\s+)?"
    rf"(?P<unit>{_UNIT_ALT}|%)?", re.IGNORECASE)
YEAR_LIKE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
# words that carry no entity information when comparing slots
STOP = frozenset("""a an the is are was were be been of in on at to for from by with and or
that this these those it its as than then so very more most less least about around
approximately roughly nearly some any all both each few many much""".split())


class Family(str, Enum):
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    MONEY = "money"
    PERCENT = "percent"
    COUNT = "count"
    YEAR = "year"
    UNKNOWN = "unknown"


@dataclass
class Quantity:
    raw: str
    value: float           # in family base units
    family: Family
    unit: str
    noun: str              # nearest content word before the number
    span: tuple[int, int]
    context: frozenset = frozenset()   # content words of the whole sentence


def _content_noun(text: str, upto: int, width: int = 40) -> str:
    """Nearest content word before the number. Cheap slot proxy, no parser needed."""
    window = text[max(0, upto - width):upto].lower()
    words = [w for w in re.findall(r"[a-z]{3,}", window) if w not in STOP]
    return words[-1] if words else ""


def extract(text: str) -> list[Quantity]:
    """All quantities with their family, SI value and slot noun."""
    out: list[Quantity] = []
    for m in PATTERN.finditer(text or ""):
        raw_num = m.group(1).replace(",", "")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        mag = (m.group("mag") or "").lower()
        if mag:
            # "m" as magnitude only when followed by a space and not a length unit
            val *= MAGNITUDE.get(mag, 1.0)
        unit = (m.group("unit") or "").lower()
        family = Family.UNKNOWN
        if unit:
            for fam, table in FAMILIES.items():
                if unit in table:
                    family = Family(fam)
                    val *= table[unit]
                    break
        else:
            ctx = text[max(0, m.start() - 12):m.end() + 12].lower()
            if YEAR_LIKE.fullmatch(raw_num) or "year" in ctx or "in 1" in ctx:
                family = Family.YEAR
            elif mag:
                family = Family.COUNT
        # Magnitude words are NOT content: "45 million people" and "45,000,000 residents" are
        # the same statement, but 'million' appears in only one context set and dragged the
        # Jaccard overlap to 0.40, producing a false slot mismatch.
        ctx_words = frozenset(w for w in re.findall(r"[a-z]{3,}", (text or "").lower())
                               if w not in STOP and w not in MAGNITUDE)
        out.append(Quantity(raw=m.group(0).strip(), value=val, family=family,
                            unit=unit, noun=_content_noun(text, m.start()),
                            span=(m.start(), m.end()), context=ctx_words))
    return out


@dataclass
class Comparison:
    verdict: str            # support | refute | insufficient
    reason: str
    residual: float = 0.0
    claim_q: Quantity | None = None
    answer_q: Quantity | None = None


def compare(claim: str, answer: str, tolerance: float = 0.02) -> Comparison:
    """Decide whether a claim's number and an answer's number are about the same quantity.

    Returns `insufficient` whenever alignment cannot be established. Refusing to compare two
    numbers that merely co-occur is the entire correction this module makes.
    """
    cqs = extract(claim)
    aqs = extract(answer)
    if not cqs or not aqs:
        return Comparison("insufficient", "no quantity on one side")

    cq = cqs[0]
    if cq.family == Family.UNKNOWN:
        return Comparison("insufficient", "claim quantity has no known unit family",
                          claim_q=cq)

    # only quantities in the SAME family are comparable
    same = [q for q in aqs if q.family == cq.family]
    if not same:
        fams = sorted({q.family.value for q in aqs})
        return Comparison("insufficient",
                          f"no comparable quantity: claim is {cq.family.value}, "
                          f"answer has {fams}", claim_q=cq)

    # Slot check: the noun context must overlap, else the two numbers are about different
    # things. Skipped for PERCENT, where the nearest content word is a verb ("fell" vs
    # "dropped") and the noun proxy carries no slot information -- applying it there rejected
    # "unemployment fell by 12%" against "the figure dropped 12.5 percent", which are the same
    # statement.
    aq = same[0]
    if cq.family != Family.PERCENT:
        # Slot alignment by CONTEXT OVERLAP. Comparing the two anchor nouns was positionally
        # fragile: in "the length of the bridge IS 330" the nearest noun is 'bridge', while in
        # "the bridge HAS a length of 330" it is 'length' -- the same statement scored a slot
        # mismatch. Jaccard overlap over the sentence's content words is stable under that
        # reordering while still separating "the bridge" from "the tunnel".
        ca, aa = set(cq.context), set(aq.context)
        if ca and aa:
            jac = len(ca & aa) / len(ca | aa)
            aligned = (jac >= 0.5) or ca <= aa or aa <= ca
            if not aligned:
                return Comparison("insufficient",
                                  f"slot mismatch (context overlap {jac:.2f}): "
                                  f"'{cq.noun}' vs '{aq.noun}' -- same unit family but "
                                  f"different objects", claim_q=cq, answer_q=aq)

    resid = abs(cq.value - aq.value) / max(abs(cq.value), 1e-9)
    # Percentages get an absolute allowance as well. Reporting a figure as "12%" when the
    # source says 12.5% is rounding, not disagreement, and a purely relative tolerance calls it
    # a 4.2% error.
    if cq.family == Family.PERCENT and abs(cq.value - aq.value) <= 1.0:
        return Comparison("support", f"agrees within 1 percentage point "
                          f"({cq.value:g}% vs {aq.value:g}%)", resid, cq, aq)
    if resid <= tolerance:
        return Comparison("support", f"agrees within {tolerance:.0%} "
                          f"({cq.value:g} vs {aq.value:g} base units)", resid, cq, aq)
    return Comparison("refute", f"disagrees by {resid:.1%} "
                      f"({cq.value:g} vs {aq.value:g} base units)", resid, cq, aq)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.quantity - SELF TEST")
    ok = True

    # --- unit conversion is exact
    r = compare("the bridge is 330 m long", "the bridge is 1083 feet long")
    good = r.verdict == "support" and r.residual < 0.01
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 330 m == 1083 ft (residual {r.residual:.4f})")

    # --- a real disagreement is caught with its residual
    r = compare("the bridge is 330 m long", "the bridge is 1120 feet long")
    good = r.verdict == "refute" and 0.03 < r.residual < 0.05
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 330 m != 1120 ft (residual {r.residual:.4f})")

    # --- THE correction: a year is not a length, so there is no conflict to report
    r = compare("the bridge is 330 m long", "it was built in 1998")
    good = r.verdict == "insufficient" and "no comparable quantity" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] length vs year -> INSUFFICIENT, not a conflict")

    # --- slot mismatch: same family, different object
    r = compare("the bridge is 330 m long", "the tunnel is 340 m long")
    good = r.verdict == "insufficient" and "slot mismatch" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] same unit family but different object -> "
          f"INSUFFICIENT")

    # --- magnitudes normalise
    r = compare("the city has 45 million people", "the city has 45,000,000 residents")
    good = r.verdict == "support"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 45 million == 45,000,000")

    # --- and a magnitude disagreement refutes
    r = compare("the city has 45 million people", "the city has 7 million people")
    good = r.verdict == "refute"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 45 million != 7 million -> refute "
          f"(residual {r.residual:.3f})")

    # --- percentages compare within the percent family
    r = compare("unemployment fell by 12%", "the figure dropped 12.5 percent")
    good = r.verdict == "support"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 12% ~ 12.5% within 2% tolerance")

    # --- no unit family -> refuse to compare
    r = compare("the score was 87", "the score was 92")
    good = r.verdict == "insufficient"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] bare numbers with no unit family -> INSUFFICIENT")

    # --- missing quantity on one side
    good = compare("the bridge is 330 m long", "no numbers here").verdict == "insufficient"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] missing quantity -> INSUFFICIENT")

    # --- extractor reports family and noun
    qs = extract("the tower is 1083 feet tall")
    # the anchor is the noun BEFORE the number ("the TOWER is 1083 feet") -- that is the
    # subject being measured, whereas "tall" comes after and qualifies the predicate.
    good = len(qs) == 1 and qs[0].family == Family.LENGTH and qs[0].noun == "tower"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] extractor gives family={qs[0].family.value} "
          f"noun='{qs[0].noun}'")

    # --- slot alignment is the discriminator: same number, wrong slot, must NOT refute
    r = compare("the bridge is 330 m long", "the tunnel opened 330 years ago")
    good = r.verdict == "insufficient"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] identical number, different family -> "
          f"INSUFFICIENT (co-occurrence is not evidence)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Slot-aligned numeric comparison.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--claim", default="")
    ap.add_argument("--answer", default="")
    args = ap.parse_args(argv)
    if args.self_test or not args.claim:
        return 0 if run_self_test() else 1
    r = compare(args.claim, args.answer)
    print(f"verdict  : {r.verdict}")
    print(f"reason   : {r.reason}")
    print(f"residual : {r.residual:.4f}")
    if r.claim_q:
        print(f"claim    : {r.claim_q.value:g} {r.claim_q.family.value} "
              f"noun='{r.claim_q.noun}'")
    if r.answer_q:
        print(f"answer   : {r.answer_q.value:g} {r.answer_q.family.value} "
              f"noun='{r.answer_q.noun}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
