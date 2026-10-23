"""logos.datasets — golden training sets whose labels are DERIVED, not judged.

THE RULE
--------
A teacher model's verdict is an opinion; a solver's output is a label. The verified-synthesis
pattern (arXiv:2508.03117) is: **construct the instance from the known answer**, then the label
comes for free and is checkable. So every generator here starts from a ground truth we already
own and renders a claim from it:

  syllogism  ground truth = the validated 24-form engine (256 mood x figure combos, 0 mismatches)
  numeric    ground truth = the fact table + sympy conversion
  lexical    ground truth = the WordNet relation between substituted words

No model is asked what the answer is, so no model can be wrong, and there is no label noise to
clean afterwards.

THE GATES ARE THE PRODUCT
-------------------------
Generated data is worthless if it leaks the test set or duplicates itself. Three gates run on
every row and each returns a typed rejection reason, because **a gate that rejects 0% is not
working** -- it is decoration. The self-test asserts non-zero rejection on all three.

  dedup     n-gram fingerprint against every row already accepted
  decontam  against a supplied dev/test text set (exact + n-gram overlap)
  temporal  evidence must be dated on or before the claim (AVeriTeC's own #1 trap)

USAGE
-----
    python tools/logos/datasets.py --self-test
    python tools/logos/datasets.py --task syllogism --n 256 --out B:\\datasets\\forge\\golden\\syllogism.jsonl
    python tools/logos/datasets.py --task numeric --n 200 --out ...
    python tools/logos/datasets.py --task lexical --n 200 --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from typing import Literal

from pydantic import BaseModel, Field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.lexical import Lexicon  # noqa: E402
from logos.logic import syllogism  # noqa: E402

Task = Literal["syllogism", "numeric", "lexical"]


# ------------------------------------------------------------------------------- schema
class GoldenRow(BaseModel):
    task: Task
    difficulty: int = Field(ge=1, le=5)
    prompt: str
    gold: str
    gold_source: str                    # where the label came from, for audit
    meta: dict = Field(default_factory=dict)


class GateReport(BaseModel):
    kept: int = 0
    rejected: dict[str, int] = Field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.kept + sum(self.rejected.values())

    @property
    def rejection_rate(self) -> float:
        return sum(self.rejected.values()) / max(self.total, 1)

    def summary(self) -> str:
        parts = [f"kept {self.kept}"]
        for k, v in sorted(self.rejected.items()):
            parts.append(f"{k} {v}")
        return f"{' · '.join(parts)} · rejection {self.rejection_rate:.1%}"


# -------------------------------------------------------------------------------- gates
def _tokens(text: str) -> list[str]:
    return [w.strip(".,;:!?()[]\"'").lower()
            for w in text.split() if w.strip(".,;:!?()[]\"'")]


def fingerprint(text: str, n: int = 4) -> set[str]:
    """Word n-gram fingerprints. Not MinHash -- at 10^3-10^4 rows an exact set is enough and
    is auditable, which mattered more here than speed."""
    toks = _tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class Gates:
    def __init__(self, dev_texts: list[str] | None = None, n: int = 4,
                 overlap_threshold: float = 1.0) -> None:
        self.n = n
        self.seen: set[str] = set()
        self.dev = [fingerprint(t, n) for t in (dev_texts or [])]
        self.overlap_threshold = overlap_threshold

    def check(self, text: str, evidence_date: str | None = None,
              claim_date: str | None = None) -> str | None:
        """Return a rejection reason, or None to accept."""
        fp = fingerprint(text, self.n)
        if not fp:
            return "dedup:empty"
        key = hashlib.sha1("|".join(sorted(fp)).encode()).hexdigest()
        if key in self.seen:
            return "dedup:exact"
        for dv in self.dev:
            if dv and len(fp & dv) / max(len(fp | dv), 1) >= self.overlap_threshold:
                return "decontam:dev_overlap"
        if evidence_date and claim_date and evidence_date > claim_date:
            return "temporal:evidence_after_claim"
        self.seen.add(key)
        return None


# ------------------------------------------------------------------- syllogism generator
NOUNS = [("cats", "mammals", "animals"), ("sparrows", "birds", "animals"),
         ("roses", "flowers", "plants"), ("sedans", "cars", "vehicles"),
         ("oaks", "trees", "plants"), ("trout", "fish", "animals"),
         ("violins", "instruments", "objects"), ("hammers", "tools", "objects"),
         ("novels", "books", "publications"), ("pines", "conifers", "trees")]
QUANT = {"A": ("All {a} are {b}", "all"), "E": ("No {a} are {b}", "none"),
         "I": ("Some {a} are {b}", "some"), "O": ("Some {a} are not {b}", "some_not")}
FIGURES = {1: (("M", "P"), ("S", "M")), 2: (("P", "M"), ("S", "M")),
           3: (("M", "P"), ("M", "S")), 4: (("P", "M"), ("M", "S"))}


def render_syllogism(mood: str, figure: int, nouns: tuple[str, str, str]) -> dict:
    s, m, p = nouns
    role = {"S": s, "M": m, "P": p}
    (maj_a, maj_b), (min_a, min_b) = FIGURES[figure]
    qm, qn, qc = QUANT[mood[0]], QUANT[mood[1]], QUANT[mood[2]]
    return {
        "major": qm[0].format(a=role[maj_a], b=role[maj_b]),
        "minor": qn[0].format(a=role[min_a], b=role[min_b]),
        "conclusion": qc[0].format(a=s, b=p),
        "prompt": (f"PREMISE 1: {qm[0].format(a=role[maj_a], b=role[maj_b])}\n"
                   f"PREMISE 2: {qn[0].format(a=role[min_a], b=role[min_b])}\n"
                   f"CONCLUSION: {qc[0].format(a=s, b=p)}\n"
                   f"Does the conclusion follow necessarily?"),
    }


def gen_syllogism() -> list[GoldenRow]:
    """All 256 mood x figure combinations. The label is the engine's ruling, so it cannot
    disagree with the engine -- which the self-test checks by round-trip."""
    rows: list[GoldenRow] = []
    for figure in (1, 2, 3, 4):
        for letters in itertools.product(syllogism.MOODS, repeat=3):
            mood = "".join(letters)
            valid = syllogism.is_valid(mood, figure)
            ok, violations = syllogism.check_rules(mood, figure)
            nouns = NOUNS[(hash((mood, figure)) % len(NOUNS))]
            r = render_syllogism(mood, figure, nouns)
            # difficulty is a property of the reasoning, not of the surface text:
            #   1 canonical figure-1 valid forms (Barbara, Celarent) -- the textbook cases
            #   2 other valid forms                        -- figure/order must be tracked
            #   3 invalid with >=2 rule violations         -- easy to spot
            #   5 invalid with exactly 1 violation         -- a near-miss, the hard case
            if valid and figure == 1 and mood in ("AAA", "EAE"):
                d = 1
            elif valid:
                d = 2
            elif len(violations) >= 2:
                d = 3
            else:
                d, = (5,)
            assert valid is ok, "is_valid and check_rules disagree"
            rows.append(GoldenRow(
                task="syllogism", difficulty=d, prompt=r["prompt"],
                gold="entails" if valid else "does_not_entail",
                gold_source="derived:syllogism.is_valid",
                meta={"mood": mood, "figure": figure, "valid": valid,
                      "violations": violations, "nouns": list(nouns)}))
    return rows


# ---------------------------------------------------------------------- numeric generator
# Conversion factors to SI. Kept explicit rather than pulling a units library so the arithmetic
# in a golden label is auditable with a calculator.
TO_M = {"m": 1.0, "km": 1000.0, "ft": 0.3048, "mi": 1609.344, "cm": 0.01}
HEDGES = ("about", "around", "approximately", "roughly", "nearly", "circa")


def gen_numeric(facts: list[tuple[str, float, str]] | None = None) -> list[GoldenRow]:
    """Numeric consistency pairs. The label is computed, never judged.

    A hedged claim ('about 330 m') NEVER yields a refutation: the claim does not commit to a
    precise value, so a precise mismatch is not evidence against it. That is a deliberate
    label rule, and it is the one most synthetic numeric datasets get wrong.
    """
    facts = facts or [("the bridge", 330.0, "m"), ("the tower", 1083.0, "ft"),
                      ("the tunnel", 57.0, "km"), ("the dam", 12.5, "mi"),
                      ("the mast", 240.0, "m")]
    rows: list[GoldenRow] = []
    for subject, value, unit in facts:
        si = value * TO_M[unit]
        for target in ("m", "ft", "km"):
            exact = si / TO_M[target]
            for delta_label, delta in (("support", 0.0), ("refute", 0.06),
                                       ("refute", -0.06)):
                shown = exact * (1.0 + delta)
                rounded = round(shown) if target != "km" else round(shown, 1)
                claim = f"{subject} is {rounded} {target} long"
                evidence = f"measured: {value:g} {unit}"
                residual = abs(shown - exact)
                rel = residual / max(exact, 1e-9)
                if delta_label == "support":
                    gold = "Supported"
                else:
                    gold = "Refuted" if rel > 0.02 else "Not Enough Evidence"
                rows.append(GoldenRow(
                    task="numeric", difficulty=1 if delta_label == "support" else 3,
                    prompt=f"CLAIM: {claim}\nEVIDENCE: {evidence}\nVerdict?",
                    gold=gold, gold_source="derived:unit_conversion",
                    meta={"claim_value": rounded, "unit": target, "si": si,
                          "relative_residual": round(rel, 6), "delta": delta}))
            # the hedged variant: never a refutation, by construction
            hedge = HEDGES[abs(hash((subject, target))) % len(HEDGES)]
            rows.append(GoldenRow(
                task="numeric", difficulty=2,
                prompt=(f"CLAIM: {subject} is {hedge} {round(exact)} {target} long\n"
                        f"EVIDENCE: measured: {value:g} {unit}\nVerdict?"),
                gold="Supported", gold_source="derived:hedge_rule",
                meta={"hedged": True, "hedge": hedge, "unit": target}))
    return rows


# ---------------------------------------------------------------------- lexical generator
CLAIM_TEMPLATES = [
    ("the {adj} dog crossed the road", "Supported"),
    ("the {adj} river flooded the valley", "Supported"),
    ("the {adj} engine failed inspection", "Refuted"),
]


def gen_lexical(lx: Lexicon | None = None, limit: int = 400) -> list[GoldenRow]:
    """Adversarial lexical pairs, with the label derived from the WordNet relation.

    synonym substitution -> the label MUST be unchanged (a model that flips here is keying on
    surface form). antonym substitution -> the label MUST flip. Both classes are needed; a
    dataset of only synonyms teaches invariance and never teaches sensitivity.
    """
    lx = lx or Lexicon()
    rows: list[GoldenRow] = []
    seeds = ["big", "hot", "large", "fast", "old", "strong", "heavy", "bright", "clean",
             "safe", "wide", "deep", "early", "cheap", "quiet"]
    for adj in seeds:
        for rel, subs, flip in (("synonym", sorted(lx.synonyms(adj)), False),
                                ("antonym", sorted(lx.antonyms(adj)), True)):
            for sub in subs[:6]:
                if len(rows) >= limit:
                    return rows
                tmpl, base_gold = CLAIM_TEMPLATES[hash((adj, sub)) % len(CLAIM_TEMPLATES)]
                gold = base_gold
                if flip:
                    gold = {"Supported": "Refuted", "Refuted": "Supported"}[base_gold]
                rows.append(GoldenRow(
                    task="lexical", difficulty=2 if rel == "synonym" else 4,
                    prompt=(f"CLAIM A: {tmpl.format(adj=adj)}\n"
                            f"CLAIM B: {tmpl.format(adj=sub)}\n"
                            f"Relation between the two claims?"),
                    gold=gold, gold_source=f"derived:wordnet_{rel}",
                    meta={"seed": adj, "substitute": sub, "relation": rel,
                          "label_flipped": flip}))
    return rows


# ------------------------------------------------------------------------------ apply gates
def apply_gates(rows: list[GoldenRow], gates: Gates,
                date_of=None) -> tuple[list[GoldenRow], GateReport]:
    rep = GateReport()
    kept: list[GoldenRow] = []
    for r in rows:
        cd, ed = (None, None)
        if date_of is not None:
            cd, ed = date_of(r)
        reason = gates.check(r.prompt, ed, cd)
        if reason:
            rep.reject(reason)
        else:
            kept.append(r)
            rep.kept += 1
    return kept, rep


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.datasets - SELF TEST")
    ok = True

    # --- syllogism: the label IS the engine's ruling, so round-trip must be exact
    syl = gen_syllogism()
    good = len(syl) == 256
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all 256 mood x figure combos generated "
          f"({len(syl)})")
    mismatch = sum(1 for r in syl if r.gold != ("entails" if r.meta["valid"]
                                                else "does_not_entail"))
    good = mismatch == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] labels round-trip against the engine "
          f"({mismatch} mismatches)")
    nvalid = sum(1 for r in syl if r.meta["valid"])
    print(f"  [INFO] valid rows: {nvalid}/256 · difficulty histogram "
          f"{ {d: sum(1 for r in syl if r.difficulty == d) for d in range(1, 6)} }")
    good = len({r.difficulty for r in syl}) >= 4
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] difficulty spans the easy->hard range")
    # a known textbook form must be the easy case
    barbara = [r for r in syl if r.meta["mood"] == "AAA" and r.meta["figure"] == 1][0]
    good = barbara.gold == "entails" and barbara.difficulty == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Barbara (AAA-1) is labelled entails at "
          f"difficulty 1")

    # --- numeric: exact arithmetic, no fudging
    num = gen_numeric()
    ft = [r for r in num if r.meta.get("unit") == "ft" and r.gold == "Supported"
          and r.meta.get("claim_value") is not None
          and abs(r.meta["claim_value"] - 1083) < 1e-9]
    good = len(ft) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 330 m -> 1083 ft is labelled Supported")
    bad = [r for r in num if r.meta.get("unit") == "ft" and r.gold == "Refuted"
           and r.meta.get("relative_residual") is not None]
    good = len(bad) >= 1
    ok = ok and good
    if bad:
        print(f"  [PASS] a 6% unit mismatch is labelled Refuted "
              f"(relative residual {bad[0].meta['relative_residual']:.3f})")
    hedged = [r for r in num if r.meta.get("hedged")]
    good = len(hedged) >= 1 and all(r.gold == "Supported" for r in hedged)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hedged claims never refute "
          f"({len(hedged)} rows, all Supported)")

    # --- lexical: synonym keeps the label, antonym flips it
    lex = gen_lexical(limit=200)
    syn = [r for r in lex if r.meta["relation"] == "synonym"]
    ant = [r for r in lex if r.meta["relation"] == "antonym"]
    good = len(syn) >= 5 and len(ant) >= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both classes present "
          f"(synonym {len(syn)}, antonym {len(ant)})")
    good = all(r.meta["label_flipped"] is False for r in syn) and \
           all(r.meta["label_flipped"] is True for r in ant)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] synonym rows are NOT flipped, antonym rows ARE")

    # --- gates must actually reject, or they are decoration
    # submit a row TWICE. A first version submitted only unique rows and asserted dedup would
    # fire, which was impossible -- rejecting nothing was the correct behaviour and the test
    # was wrong. Use a row that is not in the dev set so decontam cannot take the credit.
    g = Gates(dev_texts=[syl[0].prompt])
    kept, rep = apply_gates(syl + [syl[10]], g)
    good = rep.rejected.get("dedup:exact", 0) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dedup rejects re-submitted rows "
          f"({rep.rejected.get('dedup:exact', 0)})")
    g2 = Gates(dev_texts=[syl[0].prompt])
    kept2, rep2 = apply_gates(syl, g2)
    good = rep2.rejected.get("decontam:dev_overlap", 0) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] decontamination rejects dev-overlapping rows "
          f"({rep2.rejected.get('decontam:dev_overlap', 0)})")
    g3 = Gates()
    _, rep3 = apply_gates(num[:10], g3,
                          date_of=lambda r: ("2020-01-01", "2021-01-01"))
    good = rep3.rejected.get("temporal:evidence_after_claim", 0) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] temporal gate rejects post-dated evidence "
          f"({rep3.rejected.get('temporal:evidence_after_claim', 0)})")

    good = rep.rejection_rate > 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] report carries a rejection rate "
          f"({rep.rejection_rate:.1%})")
    print(f"  [INFO] gate summary: {rep.summary()}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Golden dataset generators with derived labels.")
    ap.add_argument("--task", choices=("syllogism", "numeric", "lexical", "all"),
                    default="all")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dev", default="", help="dev jsonl for decontamination")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    rows: list[GoldenRow] = []
    if args.task in ("syllogism", "all"):
        rows += gen_syllogism()
    if args.task in ("numeric", "all"):
        rows += gen_numeric()
    if args.task in ("lexical", "all"):
        rows += gen_lexical()
    if args.n:
        rows = rows[: args.n]

    dev_texts = []
    if args.dev and os.path.exists(args.dev):
        dev_texts = [json.loads(l).get("claim", "") for l in open(args.dev, encoding="utf-8")
                     if l.strip()]
        dev_texts = [t for t in dev_texts if t]

    kept, rep = apply_gates(rows, Gates(dev_texts=dev_texts))
    print(f"[gold] generated {len(rows)} · {rep.summary()}")
    dist: dict[str, dict[int, int]] = {}
    for r in kept:
        dist.setdefault(r.task, {})
        dist[r.task][r.difficulty] = dist[r.task].get(r.difficulty, 0) + 1
    for task, dd in dist.items():
        print(f"[gold] {task:10s} by difficulty {dict(sorted(dd.items()))}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(r.model_dump_json() + "\n")
        print(f"[gold] wrote {len(kept)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
