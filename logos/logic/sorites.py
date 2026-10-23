"""logos.logic.sorites — chain arguments, and the confidence rule that matters.

WHY CHAINS GET THEIR OWN MODULE
-------------------------------
A sorites (polysyllogism) is a chain: A→B, B→C, C→D, therefore A→D. It is the shape of almost
every multi-hop factual claim, and it is where 2026 measurements say models fail hardest -- a
single-pass LM has finite per-pass capacity and accuracy "collapses once complexity exceeds
capacity", with sharp degradation past **two hops** (ICLR 2026; NovelHopQA, EMNLP 2025).

Two things a chain checker has to get right that a rule checker does not:

1. **Link integrity.** Consecutive steps must share the middle term *in matching position*. A
   chain whose links do not join is not a weak argument, it is not an argument.

2. **Confidence composition.** The correct combination is the **product**, not the minimum.
   With independent steps at 0.9 each, three hops leave 0.729 -- the min rule would still say
   0.9 and would badly overstate a long chain. That single difference is why multi-hop
   accuracy is overstated whenever uncertainty is propagated by weakest-link intuition.

WHAT THIS DOES NOT DO
---------------------
It does not decide whether each *step* is true; it checks whether the chain **as stated** is
valid and how confidence decays along it. Step truth is the evidence layer's job.

USAGE
-----
    python tools/logos/logic/sorites.py --self-test
    python tools/logos/logic/sorites.py --chain "P->Q" "Q->R" "R->S"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import propositional as prop  # noqa: E402


@dataclass
class Link:
    """One step of a chain."""

    index: int
    formula: str
    antecedent: str = ""
    consequent: str = ""
    confidence: float = 1.0
    joins_previous: bool = True


@dataclass
class ChainResult:
    valid: bool
    n_links: int
    links: list[Link] = field(default_factory=list)
    conclusion: str = ""
    derived_conclusion: str = ""
    confidence_product: float = 1.0
    confidence_min: float = 1.0
    weakest_link: int | None = None
    breaks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _split_imp(f: prop.Node) -> tuple[prop.Node, prop.Node] | None:
    if f[0] == "imp":
        return f[1], f[2]
    return None


def check_chain(step_texts: list[str], conclusion_text: str | None = None,
                confidences: list[float] | None = None) -> ChainResult:
    """Verify a chain of implications and report how confidence decays along it.

    A chain is valid when each step is an implication, consecutive steps share a middle term
    in matching position (`A→B` then `B→C`), and the stated conclusion is the composite
    `A→C`. Premise order is NOT reorderable here -- a chain is a sequence by definition, which
    is the opposite of the rule matcher's order-independence and the reason this is separate.
    """
    steps = [prop.parse(t) for t in step_texts]
    n = len(steps)
    conf = list(confidences) if confidences else [1.0] * n
    if len(conf) != n:
        raise ValueError("confidences must align with steps")

    res = ChainResult(valid=False, n_links=n,
                      confidence_product=float(math.prod(conf)),
                      confidence_min=min(conf) if conf else 1.0)
    if conf:
        res.weakest_link = int(min(range(n), key=lambda i: conf[i]))

    splits = [_split_imp(s) for s in steps]
    bad = [i for i, sp in enumerate(splits) if sp is None]
    if bad:
        res.breaks.append(f"step(s) {bad} are not implications; a chain is made of "
                          f"implications")
        return res

    for i, (ant, con) in enumerate(splits):
        res.links.append(Link(index=i, formula=prop.to_text(steps[i]),
                              antecedent=prop.to_text(ant),
                              consequent=prop.to_text(con), confidence=conf[i]))

    # link integrity: consequent of step i must equal antecedent of step i+1
    for i in range(n - 1):
        prev_con = splits[i][1]
        next_ant = splits[i + 1][0]
        if prev_con != next_ant:
            res.breaks.append(
                f"link {i}->{i + 1} does not join: step {i} concludes "
                f"{prop.to_text(prev_con)} but step {i + 1} assumes "
                f"{prop.to_text(next_ant)}")
            res.links[i + 1].joins_previous = False

    derived = ("imp", splits[0][0], splits[-1][1])
    res.derived_conclusion = prop.to_text(derived)

    if res.breaks:
        return res

    if conclusion_text is not None:
        stated = prop.parse(conclusion_text)
        res.conclusion = prop.to_text(stated)
        if stated != derived:
            # The chain is sound but the conclusion drawn is not the one it licenses. This is
            # the "non sequitur at the end of a valid chain" case, and it is worth naming
            # separately from a broken link.
            res.breaks.append(
                f"chain composes to {prop.to_text(derived)} but the stated conclusion is "
                f"{prop.to_text(stated)}")
            return res
    else:
        res.conclusion = res.derived_conclusion

    res.valid = True
    if n >= 3 and res.confidence_product < 0.5 * res.confidence_min:
        res.notes.append(
            f"{n} hops at mean p={res.confidence_product ** (1 / n):.3f} give "
            f"{res.confidence_product:.3f} product vs {res.confidence_min:.3f} min -- the "
            f"min rule overstates a long chain by "
            f"{res.confidence_min / max(res.confidence_product, 1e-12):.1f}x")
    return res


def categorical_chain(terms: list[str], universal: bool = True) -> ChainResult:
    """The classic sorites: All A are B, all B are C, ... therefore all A are Z.

    Rendered from a term list so the caller does not have to write the implications by hand.
    """
    if len(terms) < 3:
        raise ValueError("need at least 3 terms for a chain")
    if universal:
        steps = [f"({terms[i]} -> {terms[i + 1]})" for i in range(len(terms) - 1)]
        conc = f"({terms[0]} -> {terms[-1]})"
    else:
        steps = [f"({terms[i]} -> {terms[i + 1]})" for i in range(len(terms) - 1)]
        conc = f"({terms[0]} -> {terms[-1]})"
    return check_chain(steps, conc)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.sorites - SELF TEST")
    ok = True

    # --- a valid 4-link chain
    r = check_chain(["P -> Q", "Q -> R", "R -> S"], "P -> S")
    good = r.valid and r.n_links == 3 and r.derived_conclusion == "(P -> S)"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 3-link chain valid, composes to "
          f"{r.derived_conclusion}")

    # --- the whole chain is also valid propositionally (independent confirmation)
    prem = [prop.parse("P -> Q"), prop.parse("Q -> R"), prop.parse("R -> S")]
    good = prop.entails(prem, prop.parse("P -> S"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and the propositional engine agrees independently")

    # --- a BROKEN link: Q->R then S->T does not join
    r = check_chain(["P -> Q", "S -> T"], "P -> T")
    good = (not r.valid) and len(r.breaks) >= 1 and "does not join" in r.breaks[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] broken link detected: {r.breaks[0][:64]}...")

    # --- a valid chain with the WRONG conclusion drawn (non sequitur at the end)
    r = check_chain(["P -> Q", "Q -> R"], "P -> S")
    good = (not r.valid) and any("composes to" in b for b in r.breaks)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] valid chain + wrong conclusion is caught "
          f"separately from a broken link")

    # --- a non-implication step is not a chain step
    r = check_chain(["P -> Q", "R & S"], "P -> S")
    good = (not r.valid) and any("not implications" in b for b in r.breaks)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] non-implication step rejected")

    # --- CONFIDENCE COMPOSITION: product, not min. The analytic numbers.
    r = check_chain(["P -> Q", "Q -> R", "R -> S"], "P -> S", confidences=[0.9, 0.9, 0.9])
    want = 0.9 ** 3
    good = abs(r.confidence_product - want) < 1e-12 and abs(r.confidence_min - 0.9) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 3 hops at p=0.9 give {r.confidence_product:.4f} "
          f"product (0.729) vs {r.confidence_min:.2f} min -- the min rule overstates by "
          f"{0.9 / want:.2f}x")

    # --- longer chain, product decays fast, min does not
    confs = [0.9] * 8
    r = check_chain([f"P{i} -> P{i + 1}" for i in range(8)], "P0 -> P8", confidences=confs)
    good = r.confidence_product < 0.45 and r.confidence_min == 0.9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 8 hops at 0.9 = {r.confidence_product:.3f} "
          f"product while min still says {r.confidence_min:.2f}")

    # --- weakest link is identified
    r = check_chain(["P -> Q", "Q -> R", "R -> S"], "P -> S", confidences=[0.95, 0.4, 0.95])
    good = r.weakest_link == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] weakest link identified at index "
          f"{r.weakest_link} (p={r.links[1].confidence})")

    # --- the classic categorical sorites renders and validates
    r = categorical_chain(["Socrates", "man", "mortal", "living"])
    good = r.valid and r.n_links == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] categorical sorites over 4 terms valid "
          f"({r.derived_conclusion})")

    # --- term ORDER matters: reversing the chain must break it
    r = check_chain(["Q -> P", "Q -> R"], "Q -> P")
    good = (not r.valid)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a chain whose links do not join is rejected even "
          f"when each step is individually fine")

    # --- two hops is the measured cliff: assert the confidence drop is already material
    r = check_chain(["P -> Q", "Q -> R"], "P -> R", confidences=[0.9, 0.9])
    good = abs(r.confidence_product - 0.81) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] even the 2-hop cliff leaves only "
          f"{r.confidence_product:.2f} of the per-step confidence")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chain (sorites) validity + confidence decay.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chain", nargs="+", default=[], metavar="STEP")
    ap.add_argument("--conclusion", default="")
    ap.add_argument("--terms", nargs="+", default=[])
    args = ap.parse_args(argv)
    if args.self_test or (not args.chain and not args.terms):
        return 0 if run_self_test() else 1
    r = (categorical_chain(args.terms) if args.terms
         else check_chain(args.chain, args.conclusion or None))
    print(f"valid      : {r.valid}")
    print(f"composes to: {r.derived_conclusion}")
    print(f"confidence : product {r.confidence_product:.4f} · min {r.confidence_min:.4f}")
    if r.weakest_link is not None:
        print(f"weakest    : link {r.weakest_link}")
    for b in r.breaks:
        print(f"BREAK      : {b}")
    for n in r.notes:
        print(f"note       : {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
