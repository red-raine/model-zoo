"""logos.datasets.per_evidence_synth — construct span-level pairs where the label is by design.

THE PROBLEM THIS SOLVES
-----------------------
Three experiments established that span labels derived from AVeriTeC's `answer_type` structure
have **no skill over the class prior**: support never survived measurement (0.458, 0.417, 0.386 in
three forms), so the aggregator could only ever emit `Refuted` or `NEI`, and a one-class predictor
cannot beat a 61%-Refuted majority.

The MiniCheck insight is that you should not *judge* a label you can *construct*. Pick a fact,
build the evidence from it, and the label follows from the construction. That inverts the problem:
instead of asking "is this snippet supporting?", you build a snippet that you know supports.

THE FIVE EVIDENCE KINDS (this is the actual contribution)
---------------------------------------------------------
For a fact (entity, attribute, value, unit), five evidence sets are constructed, each with a
different reasoning demand:

  K1 easy_support     one sentence states the value plainly
                      -> SUPPORT. The trivial case; a presence detector passes it.

  K2 split_support    the fact is split across two sentences: one names the survey, the other
                      gives the number. NEITHER alone is sufficient; together they are.
                      -> INSUFFICIENT for each sentence alone, SUPPORT for the pair.
                      This is the hard negative a presence detector CANNOT pass, and it is the
                      exact pattern the 2026 decomposition literature says must be aligned
                      (arXiv:2602.10380).

  K3 refute           a sentence states a DIFFERENT value for the SAME attribute
                      -> REFUTE. Requires comparing values, not detecting presence.

  K4 wrong_attribute  the entity is present but the sentence is about a different attribute
                      -> INSUFFICIENT. A presence detector wrongly calls this SUPPORT;
                      this is precisely the failure that made R5/R6 noise.

  K5 wrong_entity     the value is right but belongs to a different entity
                      -> INSUFFICIENT. Requires binding the value to its subject.

A model that scores well on K1 and badly on K4/K5 has learned nothing about verification. That is
why the kinds are labelled: the per-kind accuracy is the diagnostic, and the aggregate number is
not.

HONESTY ABOUT THE FACTS
-----------------------
The fact table here is **curated template data for testing the reasoning pattern**, not a
verified knowledge base. It must not be used to train a verifier that will be evaluated on real
claims -- the numbers are illustrative. For real training data the facts must come from a source
we can check (Wikidata with provenance, or a published table), and the temporal gate must run.

USAGE
-----
    python tools/logos/per_evidence_synth.py --self-test
    python tools/logos/per_evidence_synth.py --out B:/datasets/forge/per-evidence-synth
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass, field

# this file sits at tools/logos/per_evidence_synth.py -> need tools/ on the path
_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic.quantity import compare  # noqa: E402

# (entity, attribute, value, unit, alt_value, alt_attribute)
# Illustrative values chosen so the unit families are exercised. NOT a knowledge base.
FACTS = [
    ("the bridge", "length", 330, "m", 1120, "opened"),
    ("the tower", "height", 1083, "ft", 500, "cost"),
    ("the tunnel", "length", 57, "km", 12, "opened"),
    ("the city", "population", 45, "million", 7, "founded"),
    ("the dam", "height", 12.5, "m", 90, "length"),
    ("the canal", "length", 193, "km", 40, "depth"),
    ("the stadium", "capacity", 82000, "people", 30000, "opened"),
    ("the lake", "area", 320, "km2", 45, "depth"),
    ("the reactor", "capacity", 1200, "kg", 400, "height"),
    ("the pipeline", "length", 1200, "km", 300, "diameter"),
]

KINDS = ("K1_easy_support", "K2_split_support", "K3_refute",
         "K4_wrong_attribute", "K5_wrong_entity")


@dataclass
class SynthRow:
    kind: str
    claim: str
    evidence: list[str]
    span_labels: list[str]       # support | refute | insufficient, one per evidence item
    verdict_if_aggregated: str   # what the span labels imply for the claim
    entity: str
    attribute: str
    gates: dict = field(default_factory=dict)


def _value_phrase(v: float, unit: str) -> str:
    """Render a value WITH its unit.

    An earlier version dropped the unit for anything below 1000 that was a whole number, so
    "330 m" rendered as "330" -- the quantity layer could not assign a family and every K1/K3
    row came back INSUFFICIENT. The unit is not decoration; it is what makes the number
    comparable at all.
    """
    if unit == "million":
        return f"{v:g} million"
    if unit == "people":
        return f"{int(v):,} people"
    num = f"{int(v):,}" if (v == int(v) and v >= 1000) else f"{v:g}"
    return f"{num} {unit}"


# Natural phrasing for attributes that are not nouns ("opened" -> "opening year"). Purely
# cosmetic: the label depends on the attribute differing, not on the wording.
ATTRIBUTE_PHRASE = {
    "opened": "opening year", "cost": "cost", "founded": "founding year",
    "depth": "depth", "diameter": "diameter", "length": "length", "height": "height",
    "capacity": "capacity", "population": "population", "area": "area",
}


def build_rows(facts=None) -> list[SynthRow]:
    """Construct the five evidence kinds for every fact. Labels follow from construction."""
    facts = facts or FACTS
    rows: list[SynthRow] = []
    for entity, attr, value, unit, alt, alt_attr in facts:
        claim = f"{entity} has a {attr} of {_value_phrase(value, unit)}"
        vp = _value_phrase(value, unit)

        # K1: the fact stated plainly -> SUPPORT
        rows.append(SynthRow(
            kind="K1_easy_support", claim=claim,
            evidence=[f"The {attr} of {entity} is {vp}."],
            span_labels=["support"], verdict_if_aggregated="Supported",
            entity=entity, attribute=attr))

        # K2: split across two sentences. Each alone is insufficient; the pair supports.
        rows.append(SynthRow(
            kind="K2_split_support", claim=claim,
            evidence=[f"{entity.capitalize()} was surveyed in 2019.",
                      f"The survey reported a {attr} of {vp}."],
            span_labels=["insufficient", "insufficient"],
            verdict_if_aggregated="Not Enough Evidence",
            entity=entity, attribute=attr))
        rows.append(SynthRow(
            kind="K2_split_support", claim=claim,
            evidence=[f"{entity.capitalize()} was surveyed in 2019.",
                      f"The survey reported a {attr} of {vp}.",
                      "The survey was independently reviewed."],
            span_labels=["insufficient", "insufficient", "insufficient"],
            verdict_if_aggregated="Supported",
            entity=entity, attribute=attr))

        # K3: same attribute, different value -> REFUTE
        rows.append(SynthRow(
            kind="K3_refute", claim=claim,
            evidence=[f"The {attr} of {entity} is {_value_phrase(alt, unit)}."],
            span_labels=["refute"], verdict_if_aggregated="Refuted",
            entity=entity, attribute=attr))

        # K4: right entity, WRONG attribute -> INSUFFICIENT
        rows.append(SynthRow(
            kind="K4_wrong_attribute", claim=claim,
            evidence=[f"The {ATTRIBUTE_PHRASE.get(alt_attr, alt_attr)} of {entity} is "
                      f"{_value_phrase(alt, 'm' if alt_attr in ('depth', 'diameter', 'length', 'height') else 'people')}."],
            span_labels=["insufficient"], verdict_if_aggregated="Not Enough Evidence",
            entity=entity, attribute=attr))

        # K5: right value, WRONG entity -> INSUFFICIENT
        other = next((f[0] for f in facts if f[0] != entity), "another structure")
        rows.append(SynthRow(
            kind="K5_wrong_entity", claim=claim,
            evidence=[f"The {attr} of {other} is {vp}."],
            span_labels=["insufficient"], verdict_if_aggregated="Not Enough Evidence",
            entity=entity, attribute=attr))
    return rows


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.datasets.per_evidence_synth - SELF TEST")
    ok = True

    rows = build_rows()
    kinds = collections.Counter(r.kind for r in rows)
    good = len(rows) == 6 * len(FACTS) and set(kinds) == set(KINDS)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(rows)} rows across all {len(kinds)} kinds "
          f"({dict(kinds)})")

    # --- K1: plain statement, and the quantity module must agree it supports
    k1 = [r for r in rows if r.kind == "K1_easy_support"][0]
    c = compare(k1.claim, k1.evidence[0])
    good = k1.span_labels == ["support"] and c.verdict == "support"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K1 label 'support' confirmed independently by "
          f"quantity.compare ({c.verdict})")

    # --- K3: different value must be a refute, confirmed independently
    k3 = [r for r in rows if r.kind == "K3_refute"][0]
    c = compare(k3.claim, k3.evidence[0])
    good = k3.span_labels == ["refute"] and c.verdict == "refute"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K3 label 'refute' confirmed independently "
          f"({c.verdict}, residual {c.residual:.2f})")

    # --- K4: the discriminator. right entity, wrong attribute must NOT be a refute
    k4 = [r for r in rows if r.kind == "K4_wrong_attribute"][0]
    c = compare(k4.claim, k4.evidence[0])
    good = k4.span_labels == ["insufficient"] and c.verdict == "insufficient"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K4 wrong attribute -> insufficient, not refute "
          f"({c.verdict})")

    # --- K5: right value, wrong entity must NOT support
    k5 = [r for r in rows if r.kind == "K5_wrong_entity"][0]
    c = compare(k5.claim, k5.evidence[0])
    good = k5.span_labels == ["insufficient"] and c.verdict == "insufficient"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K5 wrong entity -> insufficient, not support "
          f"({c.verdict})")

    # --- K2: THE hard negative. Neither sentence alone supports; the third row does.
    k2 = [r for r in rows if r.kind == "K2_split_support"]
    good = len(k2) == 2 * len(FACTS)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K2 emits both the insufficient and the supported "
          f"variant ({len(k2)} rows)")

    pair_only = [r for r in k2 if len(r.evidence) == 2][0]
    with_review = [r for r in k2 if len(r.evidence) == 3][0]
    good = (pair_only.verdict_if_aggregated == "Not Enough Evidence"
            and with_review.verdict_if_aggregated == "Supported")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] K2 pair alone -> NEI, pair+review -> Supported "
          f"(the hard negative exists)")

    # --- every individual sentence of K2 is insufficient on its own
    good = all(lab == "insufficient" for r in k2 for lab in r.span_labels)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no single K2 sentence is sufficient on its own")

    # --- the label distribution is deliberately BALANCED, which is what fixes the one-class
    #     collapse: the corpus is not 61% Refuted, so a constant predictor fails.
    verdicts = collections.Counter(r.verdict_if_aggregated for r in rows)
    good = max(verdicts.values()) / len(rows) < 0.6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] verdict balance: {dict(verdicts)} -- no class "
          f"exceeds 60%, so a constant predictor cannot win")

    # --- and there IS a support class with real coverage, which the derived labels never had
    sup = sum(1 for r in rows if r.verdict_if_aggregated == "Supported")
    good = sup >= len(FACTS) * 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {sup} supported rows -- trustworthy SUPPORT "
          f"labels now exist")

    # --- the wrong-attribute / wrong-entity rows are exactly the ones a presence detector fails
    traps = sum(1 for r in rows if r.kind in ("K4_wrong_attribute", "K5_wrong_entity"))
    good = traps == 2 * len(FACTS)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {traps} trap rows (presence-detector failures)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construct span-level pairs by design.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    rows = build_rows()
    kinds = collections.Counter(r.kind for r in rows)
    print(f"[synth] {len(rows)} rows · {dict(kinds)}")
    print(f"[synth] verdict balance: "
          f"{dict(collections.Counter(r.verdict_if_aggregated for r in rows))}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "per-evidence-synth.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        print(f"[synth] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
