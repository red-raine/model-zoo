"""logos.verify.per_evidence — derive span-level verdicts, then test if aggregation recovers claims.

THE HYPOTHESIS THIS TESTS
-------------------------
The whole architecture rests on one claim: *conflict is an aggregation property, so per-evidence
verdicts plus asymmetric aggregation should beat a single claim-level forward pass.* That has
never been tested here. Building a 10k-pair training set and fine-tuning a model would cost GPU
hours; it would also confound "is the architecture right" with "did the training work".

So this file tests the architecture **before** any training, using only labels we can DERIVE
deterministically from AVeriTeC's own annotation structure. If derived span labels plus the
aggregator cannot reconstruct the claim verdict, training a span model is not the next step --
fixing the derivation or the aggregator is.

WHERE THE LABELS COME FROM (every rule named, so it can be audited)
------------------------------------------------------------------
AVeriTeC ships, per claim, a list of `questions`, each with one or more `answers` carrying an
`answer_type` in {Boolean, Extractive, Abstractive, Unanswerable}. That structure licenses
labels in a graded way:

  R1_unanswerable   answer_type == Unanswerable              -> INSUFFICIENT   (deterministic)
  R2_boolean_yes    Boolean "Yes", question not negated      -> SUPPORT        (deterministic)
  R3_boolean_no     Boolean "No",  question not negated      -> REFUTE         (deterministic)
  R4_boolean_inv    Boolean, question IS negated             -> inverted       (deterministic)
  R5_numeric_match  claim's number appears in the answer     -> SUPPORT        (deterministic)
  R6_numeric_miss   claim has a number, answer has a
                    DIFFERENT number, same slot              -> REFUTE         (deterministic)
  R7_no_signal      nothing above fired                      -> INSUFFICIENT   (honest default)

The important design choice is R7. Roughly two thirds of answers are Extractive/Abstractive and
their label genuinely does not follow from structure. Defaulting them to INSUFFICIENT keeps the
dataset honest: it means the coverage number reported below is a real measurement of how much of
this corpus the deterministic layer can reach, not an artefact of guessing.

WHY THE AGGREGATION TEST MATTERS MORE THAN THE DATASET
------------------------------------------------------
`aggregate.asymmetric_aggregate` already implements the rule that a single confidently refuted
snippet produces `Conflicting` when something else supports. Feeding it derived span labels and
checking how often the result matches the gold claim verdict is a direct, training-free test of
the composition claim -- and it produces a number that is comparable to the 0.808 the
fine-tuned encoder gets, on the same items.

USAGE
-----
    python tools/logos/verify/per_evidence.py --self-test
    python tools/logos/verify/per_evidence.py --split dev
    python tools/logos/verify/per_evidence.py --split train --out B:/datasets/forge/per-evidence
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from dataclasses import dataclass, field

# three levels up: tools/logos/verify/per_evidence.py -> tools/
_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.aggregate import asymmetric_aggregate  # noqa: E402
from logos.schema import EvidenceLabel, EvidenceVerdict, LABELS  # noqa: E402

RAW = r"B:\datasets\forge\downloads\averitec"
GOLD = {"Supported", "Refuted", "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking"}

NEG_Q = re.compile(r"\b(?:not|never|no longer|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|"
                   r"wasn'?t|weren'?t|cannot|can'?t|without|fail(?:ed|s)? to)\b", re.I)
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

# RULE GATING, measured on 500 real dev claims.
# Per-rule consistency with the claim's gold verdict (support-on-Supported or refute-on-Refuted):
#     R3_boolean_no      n=134  rate 0.903   <- the only rule that carries real signal
#     R6_numeric_miss    n=254  rate 0.437   <- noise
#     R5_numeric_match   n=107  rate 0.421   <- noise
#     R2_boolean_yes     n=115  rate 0.417   <- noise
#     R4_boolean_inv     n=  2  rate 0.500   <- too few to judge
# Used alone for aggregation, R5 scores 0.050 and R6 scores 0.097 -- far below the 0.61
# majority baseline. So the numeric rules are OFF by default.
#
# WHY R5/R6 FAIL: "the claim has a number and the answer has a number" is a co-occurrence, not
# a slot match. A claim about 2020 GDP against an answer mentioning 2019 is not a conflict. This
# is the same defect as the missing-denominator false positive, and the fix is the same: the
# NUMBER must be attached to the same quantity/unit before it means anything. Until slot
# alignment exists, numeric span labels are noise.
#
# WHY R2 FAILS: AVeriTeC's questions probe sub-facts, not the claim itself. "Did X happen?" ->
# "Yes" confirms a peripheral detail; the claim as a whole can still be Refuted. This is exactly
# the sub-claim-alignment (SAE) problem: evidence that is not aligned to the sub-claim does not
# transfer. R3 survives because a denied sub-fact usually does sink the whole claim.
RULES_ENABLED = ("R1_unanswerable", "R3_boolean_no", "R4_boolean_inv")
RULES_DISABLED = ("R2_boolean_yes", "R5_numeric_match", "R6_numeric_miss")


@dataclass
class Pair:
    claim: str
    question: str
    answer: str
    answer_type: str
    label: str                 # support | refute | insufficient
    rule: str
    confidence: float
    claim_gold: str
    source_url: str = ""


@dataclass
class ClaimResult:
    claim: str
    gold: str
    derived: str | None
    n_pairs: int = 0
    rules: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text):
        v = m.replace(",", "").rstrip(".")
        try:
            out.add(str(float(v)).rstrip("0").rstrip("."))
        except ValueError:
            pass
    return out


def derive(claim: str, question: str, answer: str, answer_type: str,
           enabled: tuple[str, ...] | None = None) -> tuple[str, str, float]:
    """Return (label, rule, confidence). Deterministic; every branch names its rule.

    `enabled` defaults to RULES_ENABLED. Passing it explicitly lets a test exercise the LOGIC
    of a gated-off rule without changing what ships, which is the difference between "this rule
    is wrong" (it is not) and "this rule measured near-chance" (it did).
    """
    rules = RULES_ENABLED if enabled is None else enabled
    at = (answer_type or "").strip().lower()
    a = (answer or "").strip()
    al = a.lower()

    # R1: the annotators recorded that no answer exists.
    if at == "unanswerable" or al in ("unanswerable", "no answer"):
        return ("insufficient", "R1_unanswerable", 0.9)

    # R2-R4: a Boolean answer is a direct probe, but question POLARITY decides the direction.
    # R2 ("Yes") is gated off by default: measured 0.417 consistency, i.e. noise, because
    # AVeryTeC questions probe sub-facts rather than the claim itself.
    if at == "boolean" or al in ("yes", "no"):
        neg_q = bool(NEG_Q.search(question or ""))
        if al.startswith("yes"):
            if neg_q:
                return ("refute", "R4_boolean_inv", 0.75)
            if "R2_boolean_yes" in rules:
                return ("support", "R2_boolean_yes", 0.8)
            return ("insufficient", "R7_no_signal", 0.3)
        if al.startswith("no"):
            return ("support" if neg_q else "refute",
                    "R4_boolean_inv" if neg_q else "R3_boolean_no",
                    0.75 if neg_q else 0.8)

    # R5/R6: numeric agreement. GATED OFF by default -- measured near-chance, see RULES_ENABLED.
    if "R5_numeric_match" in rules or "R6_numeric_miss" in rules:
        cn, an = _numbers(claim or ""), _numbers(a)
        if cn:
            if cn & an and "R5_numeric_match" in rules:
                return ("support", "R5_numeric_match", 0.7)
            if an and "R6_numeric_miss" in rules:
                return ("refute", "R6_numeric_miss", 0.55)

    # R7: nothing above licensed a label. Say so rather than guess.
    return ("insufficient", "R7_no_signal", 0.3)


def build(split: str = "dev") -> tuple[list[Pair], list[ClaimResult]]:
    path = os.path.join(RAW, f"{split}.json")
    rows = json.load(open(path, encoding="utf-8"))
    pairs: list[Pair] = []
    results: list[ClaimResult] = []
    for row in rows:
        claim = (row.get("claim") or "").strip()
        gold = (row.get("label") or "").strip()
        if not claim or gold not in GOLD:
            continue
        cr = ClaimResult(claim=claim, gold=gold, derived=None)
        evs: list[EvidenceVerdict] = []
        for q in row.get("questions") or []:
            question = q.get("question") or ""
            for ans in q.get("answers") or []:
                text = ans.get("answer") or ""
                atype = ans.get("answer_type") or ""
                lab, rule, conf = derive(claim, question, text, atype)
                pairs.append(Pair(claim=claim, question=question, answer=text,
                                  answer_type=atype, label=lab, rule=rule,
                                  confidence=conf, claim_gold=gold,
                                  source_url=ans.get("source_url") or ""))
                cr.n_pairs += 1
                cr.rules.append(rule)
                if lab != "insufficient":
                    evs.append(EvidenceVerdict(
                        evidence_id=f"{len(evs)}",
                        text=f"Q: {question} A: {text}",
                        label=EvidenceLabel.SUPPORT if lab == "support"
                        else EvidenceLabel.REFUTE,
                        confidence=conf,
                        source=ans.get("source_medium") or "",
                        # reliability is left at the neutral default on purpose: we have no
                        # per-source reliability estimate yet, and inventing one would make the
                        # aggregation result uninterpretable.
                        source_reliability=0.7))
        if evs:
            verdict, _w = asymmetric_aggregate(evs, claim_id=claim[:40])
            cr.derived = verdict
        results.append(cr)
    return pairs, results


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.verify.per_evidence - SELF TEST")
    ok = True

    # --- R1: unanswerable is deterministic INSUFFICIENT
    good = derive("c", "q", "Unanswerable", "Unanswerable")[:2] == ("insufficient",
                                                                   "R1_unanswerable")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R1 unanswerable -> insufficient")

    # --- R2/R3: boolean direction. R2 is gated OFF by default (measured 0.417 = noise), so
    #     its LOGIC is tested with it explicitly enabled, and the gating is tested separately.
    good = derive("c", "Did X happen", "Yes", "Boolean",
                  enabled=("R2_boolean_yes",))[:2] == ("support", "R2_boolean_yes")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R2 boolean Yes -> support (when enabled)")

    good = derive("c", "Did X happen", "Yes", "Boolean")[:2] == ("insufficient",
                                                                 "R7_no_signal")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R2 is DISABLED by default -> insufficient "
          f"(measured near-chance)")
    good = derive("c", "Did X happen", "No", "Boolean")[:2] == ("refute", "R3_boolean_no")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R3 boolean No -> refute")

    # --- R4: a NEGATED question inverts the direction
    good = derive("c", "Did X fail to happen", "Yes", "Boolean")[:2] == ("refute",
                                                                        "R4_boolean_inv")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R4 negated question inverts Yes -> refute")

    # --- R5/R6: numeric agreement and same-slot conflict
    num_on = ("R5_numeric_match", "R6_numeric_miss")
    good = derive("the bridge is 330 m long", "How long", "330 metres", "Extractive",
                  enabled=num_on)[:2] == ("support", "R5_numeric_match")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R5 number present in answer -> support "
          f"(when enabled)")

    good = derive("the bridge is 330 m long", "How long", "1120 feet", "Extractive",
                  enabled=num_on)[:2] == ("refute", "R6_numeric_miss")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R6 different number -> refute (when enabled)")

    good = derive("the bridge is 330 m long", "How long", "330 metres", "Extractive")[:2] == \
        ("insufficient", "R7_no_signal")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] numeric rules are DISABLED by default "
          f"(measured 0.050 and 0.097 when used alone -- noise)")

    # --- R7: no signal stays INSUFFICIENT, it does not guess
    good = derive("the policy is popular", "What did he say", "He said it was a good day",
                  "Abstractive")[:2] == ("insufficient", "R7_no_signal")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] R7 no structural signal -> insufficient "
          f"(does not guess)")

    # --- the honest default is the majority, and that is asserted so it cannot drift silently
    _p, results = build("dev")
    pairs = collections.Counter()
    for cr in results:
        for r in cr.rules:
            pairs[r] += 1
    total = sum(pairs.values())
    insuff = pairs["R1_unanswerable"] + pairs["R7_no_signal"]
    decisive = total - insuff
    good = total > 1000
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dev yields {total} pairs across "
          f"{len(results)} claims")

    good = 0 < decisive / max(total, 1) < 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] rule mix: {decisive}/{total} decisive "
          f"({decisive / max(total, 1):.1%}), {insuff} insufficient")
    print(f"  [INFO] rule counts: {dict(pairs.most_common())}")

    # --- aggregation runs and produces a verdict for claims with decisive evidence
    with_derived = [cr for cr in results if cr.derived]
    good = len(with_derived) > 100
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(with_derived)} claims produce a derived "
          f"verdict")

    dist = collections.Counter(cr.derived for cr in with_derived)
    print(f"  [INFO] derived verdict distribution: {dict(dist)}")

    # --- the self-test does NOT assert accuracy here. Whether derived labels + aggregation
    #     recover the gold verdict is the measurement, not a pass/fail. Asserting a threshold
    #     before seeing the number is how a test becomes a wish.
    # THE NUMBER THAT MATTERS IS THE MARGIN, NOT THE AGREEMENT.
    # Raw agreement here was 0.822, which looks better than the fine-tuned encoder's 0.808 --
    # until you notice 104 of 107 derived verdicts are 'Refuted' and the coverage subset is 83%
    # Refuted. Always-Refuted scores 0.832 on that subset, so the rule layer is 0.009 WORSE than
    # a constant. The baseline is therefore printed next to the agreement, permanently.
    agree = sum(1 for cr in with_derived if cr.derived == cr.gold)
    gold_dist = collections.Counter(cr.gold for cr in with_derived)
    maj_label, maj_n = gold_dist.most_common(1)[0]
    margin = agree / max(len(with_derived), 1) - maj_n / max(len(with_derived), 1)
    print(f"  [INFO] raw agreement with gold: {agree}/{len(with_derived)} "
          f"({agree / max(len(with_derived), 1):.3f})")
    print(f"  [INFO] majority baseline on the SAME subset: {maj_n}/{len(with_derived)} "
          f"({maj_n / max(len(with_derived), 1):.3f}, always '{maj_label}')")
    print(f"  [INFO] margin over majority: {margin:+.3f} -- THIS is the skill number")
    if margin <= 0.02:
        print(f"  [INFO] VERDICT: no measured skill. The rule layer reproduces the class "
              f"prior. Do NOT train a span model on these labels.")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Derive per-evidence labels; test aggregation.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    pairs, results = build(args.split)
    rules = collections.Counter(p.rule for p in pairs)
    labels = collections.Counter(p.label for p in pairs)
    total = len(pairs)
    insuff = rules["R1_unanswerable"] + rules["R7_no_signal"]
    print(f"[pe] {args.split}: {total} pairs · {len(results)} claims")
    print(f"[pe] labels: {dict(labels)}")
    print(f"[pe] rules : {dict(rules.most_common())}")
    print(f"[pe] decisive-label coverage: {(total - insuff) / max(total, 1):.1%} "
          f"({total - insuff}/{total})")

    with_d = [cr for cr in results if cr.derived]
    if with_d:
        from logos.harness import bootstrap_ci, macro_f1, paired_bootstrap
        import numpy as np
        L2I = {lab: i for i, lab in enumerate(LABELS)}
        g = np.array([L2I[cr.gold] for cr in with_d])
        p = np.array([L2I[cr.derived] for cr in with_d])
        m, per = macro_f1(g, p)
        ci = bootstrap_ci(g, p, n_boot=2000)
        print(f"[pe] derived-verdict accuracy: {ci['accuracy']:.4f} "
              f"[{ci['accuracy_ci'][0]:.3f},{ci['accuracy_ci'][1]:.3f}]")
        print(f"[pe] derived-verdict macro-F1: {ci['macro_f1']:.4f} "
              f"[{ci['macro_f1_ci'][0]:.3f},{ci['macro_f1_ci'][1]:.3f}]")
        print(f"[pe] per-class F1: {per.round(3).tolist()}")
        print(f"[pe] NOTE: this covers only the {len(with_d)} claims that received at least "
              f"one decisive span label -- it is NOT comparable to the encoder's 0.808 on all "
              f"500 until the same subset is scored for the encoder")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.split}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
        print(f"[pe] wrote {args.out}/{args.split}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
