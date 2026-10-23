# Measurement harness results — and the claim it killed

**Date:** 2026-10-23
**Tool:** `tools/logos/harness.py` (self-test 9/9)
**Data:** AVeriTeC dev, n=500, both models trained on the same 2,566-row fit split

---

## The table that decides what we are allowed to say

| class | n | deberta-base F1 | ctu-large F1 | paired diff | 95% CI | verdict |
|---|---|---|---|---|---|---|
| Supported | 122 | 0.664 | **0.852** | **+0.188** | **[+0.124, +0.256]** | **SIGNIFICANT** |
| Refuted | 305 | 0.829 | **0.884** | **+0.055** | **[+0.025, +0.087]** | **SIGNIFICANT** |
| Not Enough Evidence | 35 | 0.643 | 0.615 | −0.027 | [−0.147, +0.079] | not distinguishable |
| **Conflicting** | **38** | **0.214** | **0.362** | **+0.148** | **[−0.038, +0.323]** | **NOT DISTINGUISHABLE** |

Overall: macro-F1 0.5876 → 0.6784, **diff +0.0908, CI [+0.0282, +0.1496]** — excludes zero.
Accuracy 0.7440 → 0.8080, diff +0.0640, CI [+0.0260, +0.1020].
McNemar on discordant pairs: 33 rows only base got right, 65 only CTU got right, **p = 0.0016**.

**The head swap is real.** That part stands.

---

## The claim it killed

I reported earlier, twice, that the CTU checkpoint "attacks the Conflicting tail directly" and
that the tail "more than doubled, `.214 → .362`". The point estimate did more than double. **The
95% interval on that difference is [−0.038, +0.323] and contains zero.** With 38 positives in
dev, a 0.148 F1 gain on that class is **not statistically established**, and I should not have
described it as an established effect.

This is the fourth time in this project that a number looked like a finding and dissolved under
inspection — and the first time a purpose-built tool caught it *before* it propagated into the
design. That is the entire argument for building the harness before more model work.

### What actually improved, and what did not

**Established:** the model got much better at `Supported` (+0.188, CI excludes 0 by a wide
margin) and somewhat better at `Refuted` (+0.055). Those two classes carry 85% of dev.

**Not established:** anything about `Conflicting` or `NEI`, in either direction. `NEI` moved
−0.027 with a CI spanning ±0.11.

### Why this matters for the plan

The strategy I wrote down was "the head swap attacks the tail". It doesn't, or at least we
cannot show it does. The tail fix has to come from the **aggregation layer** — the asymmetric
combiner with the `min(support, refute)` interaction that peaks at evidence balance — and that
component is already built and tested (`logos/aggregate.py`, 16/16). Its evaluation will inherit
exactly this measurement problem, so the item below is now the critical path, not a nicety.

---

## Conclusion: we cannot measure our most important class on this split

38 Conflicting positives. The paired interval is roughly ±0.18 wide. **Any Conflicting
improvement below ~0.18 F1 is unmeasurable on 500 dev rows.** That is larger than the entire
current F1 of the class.

Options, in order of cost:

1. **Cross-fit the evaluation** — 5-fold over the 3,066 train rows gives ~600 held-out rows per
   fold and ~1,500 Conflicting positives total instead of 38. Cheap, no new labels, and it is
   the only way to get a usable interval without touching a GPU.
2. **Report the class as `insufficient n`** rather than as a number, until 1 is done. A point
   estimate with a CI spanning zero is not a result.
3. **Pool dev with a held-out slice of train** *for the Conflicting class only* — acceptable if
   stated, never for the headline.

---

## Harness capabilities (self-test 9/9)

| test | analytic expectation | result |
|---|---|---|
| macro-F1 vs sklearn, 50 random cases | exact agreement | max err **5.6e-17** |
| perfect predictor | F1 = 1.0, CI **[1,1]** (degenerate) | pass |
| collapse to one class on balanced gold | exactly **0.10** macro-F1 (class 0 F1 = 0.4) | **0.100000000000** |
| model compared against itself | diff **exactly 0**, CI **[0,0]** | pass |
| McNemar on identical predictions | **p = 1.0** | pass |
| detects a real improvement | CI excludes 0, McNemar p < 1e-6 | diff +0.0600, **p=1.5e-36** |
| reports no difference when there is none | CI contains 0 | CI [−0.0002, +0.0005] |
| 5 stratified folds | balanced sizes | **[100,100,100,100,100]** |
| ablation flip rate | evidence-reading model 1.00 vs prior-only 0.00 | pass |

One real bug caught while building it: the fold assignment used round-robin `j % k` per class,
which balances class proportions but let fold sizes drift (101/100/100/100/99). Now greedy
assignment to the smallest fold, which is what sklearn's `StratifiedKFold` actually guarantees.

---

## Still open on this item

The **evidence-ablation flip rate** (REAL, arXiv:2609.08943) is implemented and unit-tested
against a stub model, but has **not been run on the real checkpoint** — that needs the CTU
weights on Sol and one inference pass over dev with the evidence half removed. Until it runs,
we do not know whether either model is verifying evidence or reciting a prior. Given the
`Supported` gain is where the head swap earned its keep, this check is the highest-value
remaining step and it is cheap.

---

# Part 2 — the ablation flip test ran, and the placebo control inverted the conclusion

**Checkpoint:** CTU AVeriTeC-large post-trained, 500 dev rows, three conditions.
**Artifact:** `/tmp/data/ablation/ablation.json` (predictions for all three conditions).

| condition | accuracy |
|---|---|
| full — `claim [SEP] evidence` | **0.8080** |
| **ablated** — claim only (REAL) | **0.3360** |
| **placebo** — `claim [SEP] unrelated council-bin-collection sentence` | **0.2360** |

| flip rate vs full | |
|---|---|
| evidence deleted | **0.6840** (342 rows) |
| **placebo distractor** | **0.7860** (393 rows) |

Flips by gold label: Supported 0.738 · Refuted 0.649 · NEI 0.571 · **Conflicting 0.895**.
Ablated prediction distribution collapses to NEI (279/500 = 56%).

## What this establishes

**It is NOT a prior model.** Deleting the evidence costs **47 accuracy points** (0.808 → 0.336).
The verdict genuinely depends on the evidence slot. That was the question REAL (arXiv:2609.08943)
asks, and the answer is reassuring.

## What it exposes

**A placebo distractor is WORSE than no evidence at all: 0.236 vs 0.336**, and the flip rate is
*higher* (0.786 > 0.684).

That is the opposite of what a relevance-discriminating verifier would do. If the model could
tell relevant from irrelevant evidence, a random sentence should behave like an absence —
landing near 0.336. Instead it actively hurts, which means the model treats whatever sits in the
evidence slot as *competing evidence* with no notion of relevance. Sensitivity to evidence is
**less** than baseline sensitivity to arbitrary text.

So the honest two-line summary:

1. **evidence-dependent** — established, large effect, not a prior model.
2. **not relevance-discriminating** — a distractor degrades it below having nothing.

This would have been invisible without the placebo arm. A single-condition ablation would have
reported "flip rate 0.684, plausible evidence dependence" and stopped. That is exactly the
failure mode the research warned about, and the control that catches it costs one extra forward
pass over the dev set.

## Consequence for the plan

Item "**evidence relevance reranker**" moves from *nice-to-have (+14–22 pts lever in the
literature)* to **the top-ranked data/model item**, because we now have a direct measurement of
the deficiency rather than a citation:

> the model cannot distinguish a relevant snippet from an irrelevant one, and an irrelevant one
> is worse than none.

Two further consequences:

- **Conflicting flips at 0.895** — nearly every Conflicting verdict depends on the evidence.
  Combined with the earlier finding that Conflicting's F1 improvement (0.214 → 0.362) is not
  statistically established at n=38, the class is simultaneously the most evidence-driven and the
  least measurable. Cross-fitting is the only way out.
- **The aggregator is now clearly on the critical path.** If the head cannot discriminate
  relevance, the per-evidence 3-way head plus asymmetric aggregation is not an optimisation; it is
  the mechanism that supplies the missing discrimination.

## Self-test status

`tools/verify/ablation_eval.py` — 4/4 (three-condition build verified: ablated has no `[SEP]`,
placebo carries the distractor and not the evidence, labels in range).
