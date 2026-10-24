# Per-evidence labels: the architecture test failed, and it saved a GPU run

**Date:** 2026-10-23
**Module:** `logos/verify/per_evidence.py` (15/15)
**Measurement:** AVeriTeC dev, 500 claims, 1,399 question-answer pairs

---

## What was tested, and why before training

The whole architecture rests on one claim: *conflict is an aggregation property, so span-level
verdicts plus asymmetric aggregation should beat a single claim-level forward pass.* That had
never been tested here. The plan was to build ~10k per-evidence pairs and fine-tune a model.

Building the training run first would have confounded two questions — "is the architecture
right" and "did the training work". So the architecture was tested **with no model at all**,
using labels derived deterministically from AVeriTeC's own annotation structure.

## The derivation rules, and what each measured

AVeriTeC ships per claim a list of `questions`, each with `answers` carrying an `answer_type`
in {Boolean, Extractive, Abstractive, Unanswerable}. That licenses labels in a graded way.
Each rule below was scored by whether its span label agrees with the **claim's** gold verdict
(support-on-Supported, or refute-on-Refuted counts as consistent):

| rule | n | consistency | verdict |
|---|---|---|---|
| **R3_boolean_no** — Boolean "No", question not negated | 134 | **0.903** | **real signal** |
| R6_numeric_miss — claim has a number, answer has a different one | 254 | 0.437 | noise |
| R5_numeric_match — claim's number appears in the answer | 107 | 0.421 | noise |
| R2_boolean_yes — Boolean "Yes", question not negated | 115 | 0.417 | noise |
| R4_boolean_inv — Boolean, negated question | 2 | 0.500 | too few to judge |

Used alone for aggregation: **R5 scores 0.050, R6 scores 0.097** — far below the 0.61 majority
baseline.

### Why R5/R6 fail

"The claim has a number and the answer has a number" is a **co-occurrence, not a slot match**. A
claim about 2020 GDP against an answer mentioning 2019 is not a conflict. This is the identical
defect to the missing-denominator false positive found earlier, and the fix is the same: the
number must be attached to the same quantity/unit before it means anything. Until slot alignment
exists, numeric span labels are noise.

### Why R2 fails, and R3 survives

AVeriTeC's questions probe **sub-facts, not the claim**. "Did X happen?" → "Yes" confirms a
peripheral detail while the claim as a whole can still be Refuted. That is exactly the
sub-claim-alignment (SAE) problem: evidence not aligned to the sub-claim does not transfer.

R3 survives because a denied sub-fact usually does sink the whole claim.

## The number that matters — and the one that looked good

With only the rules that measured trustworthy (R1 + R3 + R4):

```
coverage          : 107 / 500 claims (21.4%)
derived verdicts  : Refuted 104, Not Enough Evidence 3
raw agreement     : 88/107 = 0.822     <- looks better than the encoder's 0.808
MAJORITY baseline : 89/107 = 0.832     <- always 'Refuted', on the SAME subset
margin            : -0.009             <- THIS is the skill number
```

**0.822 is the class prior, not reasoning.** The coverage subset is 83% Refuted, so always
saying "Refuted" scores 0.832 and the entire rule layer is **0.9 points worse than a constant**.

This is now printed next to the agreement inside `--self-test`, permanently. A raw agreement
number without its subset majority is not a result.

## The planning consequence

**Do not train a per-evidence head on structurally-derived labels.** It would learn R5/R6 noise
and R2's prior, and the earlier finding that a claim-level encoder reaches 0.808 would make the
span model look like a regression rather than what it is — a target problem.

What the per-evidence head actually needs, in order of preference:

1. **LLM-judged span labels with solver verification** (the MiniCheck pattern): construct the
   pair from a known fact so the label is *derived*, not judged. This is the only route that has
   working precedent at our scale (770M reaching GPT-4 parity at 400× lower cost).
2. **Slot-aligned numeric extraction** — attach the number to its quantity and unit before
   comparing. That would rescue R5/R6, which are currently pure co-occurrence.
3. **R3 alone as a high-precision seed set**: 134 dev pairs at 0.903 is a genuinely usable seed,
   and the same rule on train yields roughly 600. That is a bootstrap set, not a training set.

## What this does NOT say

It does not say per-evidence + aggregation is the wrong architecture. It says **these
structurally-derived labels cannot test it**. The aggregator (`aggregate.py`, 16/16) and the
Mondrian layer (`abstain.py`, 12/12) are still untested against a real span model — they are
waiting for labels worth training on, not for a rewrite.

## Files

| file | self-test |
|---|---|
| `logos/verify/per_evidence.py` | 15/15 — 7 rule-logic tests, 3 gating tests, coverage + margin reporting |

Rule gating is a first-class constant (`RULES_ENABLED` / `RULES_DISABLED`) with the measured
numbers in the comment, and `derive(..., enabled=...)` lets a test exercise a gated-off rule's
logic without changing what ships. That is the difference between "this rule is wrong" (it is
not) and "this rule measured near-chance" (it did).
