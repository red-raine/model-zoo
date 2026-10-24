# Three experiments: slot alignment, the aggregate ceiling, and Lean's `sorry` hole

**Date:** 2026-10-23
**Modules:** `logos/logic/quantity.py` (13/13), `lean_bridge.py` (11/11, +zero-sorry)
**Experiments:** 3, on 500 real AVeriTeC dev claims and 1,399 question-answer pairs

---

## Experiment 1 — does slot alignment rescue the numeric rules?

Earlier the numeric rules scored 0.421 / 0.437 consistency, and 0.050 / 0.097 alone. The
diagnosis was co-occurrence rather than slot match. `quantity.py` attaches each number to a
**unit family** (length, mass, time, money, percent, count, year) and a **noun anchor**, converts
within the family, and refuses to compare across families.

| rule | n | consistency |
|---|---|---|
| OLD `R5_numeric_match` (co-occurrence) | 101 | 0.386 |
| OLD `R6_numeric_miss` (co-occurrence) | 200 | 0.555 |
| **NEW `S2_aligned_refute`** | 33 | **0.818** |
| NEW `S1_aligned_support` | 24 | 0.458 |
| combined OLD | 301 | 0.498 |
| **combined NEW** | 57 | **0.667** |

**Slot alignment helps: +0.168.** And it helps in exactly one direction — **refute 0.818**,
support 0.458. That is the sub-claim-alignment (SAE) pattern a third time: a genuine
contradiction is informative, a genuine agreement may only confirm a peripheral fact.

Coverage falls from 301 pairs to 57, which is the correct trade: refusing to compare a length
against a year is the whole point.

## Experiment 2 — does that move the aggregate margin?

| configuration | coverage | accuracy | majority | **margin** |
|---|---|---|---|---|
| arbitrary confidences (0.8 / 0.7), no slot | 104 | 0.846 | 0.846 | **+0.000** |
| arbitrary confidences, + slot refute | 126 | 0.730 | 0.794 | **−0.063** |
| confidences calibrated to measured consistency, + slot | 126 | 0.794 | 0.794 | **+0.000** |

Two findings, both uncomfortable:

**(a) The weight threshold decides the answer.** `asymmetric_aggregate` classifies a snippet as
STRONG when `confidence × reliability ≥ 0.55`. With my arbitrary 0.7 × 0.7 = 0.49, every
slot-aligned refute was treated as WEAK and downgraded to "Not Enough Evidence" — which is why
adding good labels made the result *worse*. The aggregate is more sensitive to an invented
confidence than to the quality of the labels.

**(b) And with the confidences calibrated, the margin is exactly zero.** Derived distribution:
`{'Refuted': 126}` — the rule layer predicts "Refuted" for every single covered claim.

## Why it is structurally zero, not merely tuneable

The derivation only ever produces **REFUTE** labels. SUPPORT never survives measurement
(0.458, 0.417, 0.386 across three attempts in three different forms). With refutes only, the
aggregator has exactly two possible outputs: `Refuted`, or `Not Enough Evidence`. **A one-class
predictor cannot beat the majority of a 61%-Refuted corpus**, no matter how the thresholds are
set.

So the conclusion is not "tune better". It is: **you cannot derive a working span-level verifier
from AVeriTeC's annotation structure alone.** You need trustworthy SUPPORT labels, which means
either LLM-judged span entailment (the MiniCheck pattern, with solver verification) or
constructed data where the support relation holds by construction.

This is the third independent experiment reaching the same place. The finding is now load-bearing
rather than provisional.

## Experiment 3 — what Lean can actually do, and the hole it left

| case | result |
|---|---|
| 5-link sorites chain (propositional) | **PROVED** |
| universal instantiation `∀x P(x) ⊢ P(a)` | **PROVED** (first-order works) |
| transitivity over a quantified relation | **PROVED** |
| `2 + 2 = 4` and `330·328084/100000 = 1082` via `decide` | **PROVED** — arithmetic a truth table cannot do |
| `2 + 2 = 5` | **REJECTED** ✓ |
| barber-style paradox via `exact False.elim (by sorry)` | **"PROVED"** ← the hole |

**Lean accepts `sorry` and exits 0.** It prints a warning and returns success, so any check that
asks "did it compile" reports a proof of *anything*. This is not hypothetical — it happened in
the boundary test above.

This is precisely the failure mode `antigravity/minion/formal/lean_runner.py` advertises a guard
against ("zero-sorry enforcement"), and precisely what a parser-based fallback cannot catch. A
`LeanRunner` that checks exit code without scanning for `sorry` is verifying nothing.

### Fixed

`lean_bridge.py` now:
- has `has_sorry(source)` and refuses to call anything trustworthy when `sorry`/`admit` appears
- sets `has_sorry` on every `LeanResult` from both proof paths
- asserts, as a test, that a `sorry`-tainted result is **not** trustworthy even when Lean
  accepted it

## Files

| file | self-test | what |
|---|---|---|
| `logos/logic/quantity.py` | 13/13 | slot-aligned numeric comparison: unit families, noun anchor, cross-family refusal |
| `logos/logic/lean_bridge.py` | 11/11 | + zero-sorry enforcement, `has_sorry` on every result |

Bugs caught this round: thousands separators parsed `45,000,000` as `45`; the noun-anchor check
wrongly applied to percentages (rejecting "fell by 12%" vs "dropped 12.5 percent"); a
`Comparison` dataclass indexed as a tuple; and the `sorry` hole itself.

## What this changes in the plan

1. **The span head is now unambiguously blocked on labels, not on architecture.** Verified three
   ways. Next attempt must be MiniCheck-style constructed or LLM-judged pairs — nothing derived
   from `answer_type` will work.
2. **Confidence must be calibrated, not invented.** The aggregate's behaviour was decided by an
   arbitrary 0.7. `quantity`'s measured 0.818 and R3's 0.903 are the numbers to use, and the
   fact that doing so changed the answer is itself evidence that the weights need deriving from
   measured consistency rather than chosen.
3. **Any "Lean-verified" claim in this project now means "compiled AND sorry-free".** The
   distinction is a test, not a comment.

## Possibly useful next

- **Walton CQ support detection is lexical** — worth checking against a human-labelled subset
  before it informs any verdict.
- **The aggregator's `strong_weight=0.55`** is hand-set. Everything above suggests it should be
  fitted on measured per-rule consistency: `weight_i = measured_consistency_i`.
