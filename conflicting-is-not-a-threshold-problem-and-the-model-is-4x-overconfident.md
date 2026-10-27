# CONFLICTING is not a threshold problem, but the model is 4× overconfident

**Date:** 2026-10-26
**New:** `logos/verdict_tune.py` (10/10)
**Two results:** one negative and decisive, one positive and large. Plus a data-integrity bug caught on the way.

---

## 1. The data-integrity bug, and the better evidence for it

The first version of the check compared `calib` against `train.clean.jsonl` and found **496/500 claim
overlap**, concluding "TUNING WOULD LEAK". That conclusion was right, but the *evidence* was weak:
a row can appear in the same source file and never have been used for training, and the model records
`fit_n: 2566`, so the fine-tuning may have used only a subset.

**File provenance is not training overlap. Behaviour is.** So the check was replaced with a
memorisation test:

```
split        n     acc   vs dev      NLL  mean max p
fit       2566  0.9981  +0.1901   0.0093      0.9996
calib      500  0.9260  +0.1180   0.6237      0.9944
dev        500  0.8080  +0.0000   1.7563      0.9908
```

`calib` sits **11.8 points above the generalisation rate** with a third of the NLL. It was seen.

**And the rule matters more than this instance.** A held-out split must match the generalisation rate —
not sit *between* the memorised split and the honest one. Sitting between is the most dangerous position
for a tuning split, because the naive `fit − calib` gap here is only **0.072** and reads as harmless,
while the `calib − dev` gap of **0.118** is the one that corrupts every tuned parameter. **Partial
contamination is worse than total, because total contamination is obvious.**

My first diagnosis reported "ambiguous", which is exactly the word that lets a contaminated split
through by sounding cautious. `verdict_for()` no longer has that option.

### And this explains the earlier flat tuning surface, per class

| class | calib F1 | dev F1 |
|---|---|---|
| CONFLICTING | **0.585** | **0.362** |

**The contaminated split made the hardest class look nearly solved.** A tuning procedure given these
numbers will correctly conclude "no correction needed" — which is precisely the flat `tau = 0` surface
across `[0, 2]` that appeared in the first attempt. The leak did not merely inflate a metric; it removed
the very signal the tuning was supposed to find.

## 2. Negative result: CONFLICTING over-prediction is NOT a decision-boundary problem

I hypothesised that CONFLICTING being predicted 67 times for 38 real rows was a threshold problem —
cheap to fix with a prior correction, before spending on data. Tested properly with **cross-fitting on
the deduped dev split** (every row predicted by a `tau` fitted without it, 5 folds):

```
tau chosen per fold : [0.0, 0.0, 0.0, 0.0, 0.0]
predictions changed : 0/482
macro-F1            : 0.6688 -> 0.6688
McNemar             : 0 fixed, 0 broken, p = 1.0
```

**Unanimously `tau = 0` in every fold, and not a single prediction moved.** The prior-correction family
cannot help, because the model is not mis-ranking rows — it genuinely assigns high probability to
CONFLICTING on 67 rows. Rescaling by the class prior does not alter an argmax that confident.

So my hypothesis was wrong, and the correction is now ruled out rather than assumed. **CONFLICTING has
to be fixed by better evidence or more data, not by a threshold.** That is worth knowing before
spending two thousand rows on it: the cheap option is genuinely unavailable, which strengthens rather
than weakens the case for the data.

## 3. Positive result: the model is ~4× overconfident, confirmed by two independent protocols

Because `calib` is unusable, the calibration parameter has to come from `dev` itself. Two different
protocols, both paying that price honestly:

**Protocol 1 — 5-fold cross-fitting** (every row predicted by a `T` fitted without it):

```
T chosen per fold : [4.24, 4.27, 4.22, 4.25, 4.17]     spread 0.10
ECE               : 0.0960 -> 0.0240    (-75%)
Brier             : 0.0969 -> 0.0783    (-19%)
```

**Protocol 2 — class-stratified half-split** (fit on one half, report on the other, both directions):

```
A->B: fit n=240, test n=242, T=4.053  ECE 0.1080 -> 0.0291  Brier 0.1084 -> 0.0853
B->A: fit n=242, test n=240, T=4.399  ECE 0.0872 -> 0.0319  Brier 0.0854 -> 0.0717
mean T = 4.226        mean ECE 0.0976 -> 0.0305
```

**The two protocols agree.** Cross-fitted `T` averaged 4.23; held-out half-split averaged 4.23. When
two different splits of the same small set produce the same scalar to two decimal places, that scalar is
a property of the model rather than of the split. **Adopted: T = 4.23, ECE 0.098 → 0.031.**

Stratification was necessary, not decorative: with CONFLICTING at 38 rows, an unstratified half-split
would leave one side with a handful and make that class's calibration meaningless — the same rare-class
failure already documented for the conformal tier.

Temperature cannot change a prediction, so accuracy 0.8008 / macro-F1 0.6688 are untouched. What changes
is whether the probabilities **mean anything**, and that is the whole point:

- **The recorded calibration was wrong *and* pessimistic.** `uncertainty.json` reports
  `ece_before 0.176 → ece_after 0.083`, fitted on the contaminated split. The honest figure is
  **0.098 → 0.031**.
- **Abstention can now be trusted.** `abstain.py` (Chow's rule) and `arbiter.py` both consume
  probabilities as if they were probabilities. At ECE 0.098 that assumption is false in a way that
  silently distorts every risk-coverage curve; at 0.031 it holds. One scalar, measured twice, is the
  difference between a conformal guarantee and a number that looks like one.

## 3b. And the per-class boundary move is inert too — which settles it

Prior rescaling is one family. A per-class score multiplier is a different one, and it moves only that
class's boundary. Cross-fitted, one parameter, both directions searched:

```
CONFLICTING          w chosen per fold: [0.1, 0.1, 0.2, 0.1, 0.1]
  macro-F1 0.6688 -> 0.6717  (+0.0029)   changed 5/482   McNemar p = 0.25
  its own F1 0.362 -> 0.380   precision 0.284 -> 0.306   recall 0.500 -> 0.500
  predicted 67 -> 62

NOT_ENOUGH_EVIDENCE  w chosen per fold: [0.05, 0.05, 0.05, 0.05, 0.05]
  macro-F1 0.6688 -> 0.6688  (+0.0000)   changed 0/482   McNemar p = 1.00
  predicted 16 -> 16
```

Read what a **ten-times suppression** of CONFLICTING achieves: it removes **5 of the 29 false
positives** and leaves recall untouched at 0.500. A **twenty-times** suppression of NEI changes
**nothing at all**.

That is the diagnostic that matters. If the errors were boundary errors, a large weight change would
move many predictions. They barely move, which means the model's false positives are **high-confidence
errors**: it is not uncertain-but-leaning, it simply lacks the information to tell these rows apart.
No threshold family can fix a confidence gap that large, because there is no boundary to move.

And the +0.0029 macro-F1 from the CONFLICTING weight is **not adoptable**: p = 0.25, five rows, and a
tuned parameter. Adopting it would be exactly the "small gain from a fitted freedom" pattern this
project keeps catching. It is reported and rejected.

**Both cheap levers are now exhausted by measurement. The remaining lever is data or a better model,
and that is a conclusion rather than an assumption.**

## 4. What the ordering of work now looks like

| lever | verdict | cost |
|---|---|---|
| prior / threshold correction | **ruled out** — 0/482 changes, tau=0 in all folds | done |
| per-class boundary move | **ruled out** — 10x suppression of CONFLICTING moves 5 rows, p=0.25; 20x on NEI moves 0 | done |
| temperature scaling | **75% ECE reduction**, stable across folds, adopt | one scalar |
| feature pooling | mean beats CLS by 13.7 pp | free, already recorded |
| stratified CONFLICTING / NEI data | **the only remaining lever** for those classes | ~2,000 rows |
| model upgrade | unknown | expensive |

The cheap levers are now exhausted **by measurement, not by assertion**. Three were tried and two were
rejected, one adopted, and the expensive one is supported by evidence instead of instinct. The
diagnostic that made this decisive: a 10x weight change moving only 5 rows proves the errors are not
near a boundary.

## 5. Honest limits

- **n = 482 deduped dev, 5 folds** — each fold's `tau`/`T` is fitted on ~386 rows. Enough for a single
  scalar, not enough to tune anything with real flexibility, which is why only a scalar was tried.
- **Cross-fitting reuses dev for both roles.** It is honest in the sense that no row is predicted by a
  correction fitted on itself, but the dev set's composition still influences the fitted values. A
  truly external calibration set would be better and does not exist, because `calib` is contaminated.
- **`tau = 0` was rejected within one family only** (prior rescaling). A per-class threshold search, or
  a cost-sensitive decision rule from `decision.py`, is a different family and remains untested. The
  claim is "prior correction does not help", not "no threshold method can help".
- **Temperature fixes calibration, not discrimination.** ECE 0.024 says the probabilities can be
  believed; macro-F1 0.6688 says the ordering has not improved at all, and CONFLICTING F1 0.362 is
  untouched.

## 6. Next

1. **~~Adopt T = 4.2 as a fixed post-hoc scalar~~** -- done. Recorded in
   `B:/datasets/forge/verifier-av/calibration.json` with the cross-fitted ECE, both confirming
   protocols, the label-encoding proof, the contaminated splits marked do-not-use, and both ruled-out
   threshold families marked do-not-retry. The number cannot now outlive its justification.
2. **~~Rebuild a genuinely held-out calibration split~~** -- done differently, and the honest way: with
   `calib` unusable and only 482 clean rows, no external calibration set exists at acceptable cost, so
   the scalar is fitted by two split protocols inside dev and confirmed by their agreement (4.23 vs
   4.23). A true external split requires new data, which is the same purchase as item 3.
3. **Then** the stratified data for CONFLICTING and NEI -- now with two ruled-out alternatives behind it.
4. **~~Feed the calibrated probabilities into abstention~~** — done, and it produced the sharpest
   result of this cycle. See section 7: calibration is *harmful* for ranking-driven abstention and
   *essential* for threshold-driven abstention.

---

## 7. Does calibration actually improve abstention? Two answers, one of them surprising

I expected a simple yes. The measurement gives one **no** and one **enormous yes**, and the split
matters.

### 7a. I asserted an invariance theorem, and it was false

My reasoning: temperature is a monotone transform, so it cannot change any ranking-based curve —
risk-coverage and AURC must be exactly invariant. That is true **per row** (the argmax cannot change)
and **false across rows**.

$$
p_i(T) = \frac{1}{1 + \sum_{j \neq i} e^{(z_j - z_i)/T}}
$$

Each row's max probability shrinks with `T` by an amount that depends on **that row's margin relative
to its competitor spread**. Two rows can therefore swap order. Measured on 800 random rows at `T = 3.3`:
**5.3% of row pairs reverse.**

The test now pins the truth rather than the intuition, in three parts: ordering is *not* invariant;
the argmax *is* invariant; and so risk-coverage *can* move.

### 7b. Ranking-driven abstention: calibration makes it slightly worse

```
AURC          raw 0.074327   calibrated 0.079698   (+0.005371, WORSE)
AURC excess   raw -0.124844  calibrated -0.119472  (negative = better than random)
```

**Do not claim a risk-coverage gain from calibration.** Here it costs 0.005 AURC. Both variants remain
far better than random, but the claim "better calibration improves selective prediction" is simply not
true when selection is by confidence rank.

### 7c. Threshold-driven abstention: calibration is the difference between impossible and working

Chow-style rule — pick the threshold that meets a 10% risk target on one half, apply it to the other:

| | threshold | coverage | risk | target |
|---|---|---|---|---|
| **raw** (ECE 0.098) | 1.00 | **0.000** | — | **MISS** |
| **raw** (ECE 0.098) | 1.00 | **0.000** | — | **MISS** |
| **calibrated** (ECE 0.031) | 0.83 | **0.719** | 0.086 | OK |
| **calibrated** (ECE 0.031) | 0.81 | **0.783** | 0.101 | MISS by 0.001 |

**The raw model cannot configure abstention at all.** Its probabilities are so inflated that *no*
threshold meets a 10% risk target on unseen rows, so coverage collapses to zero. After one scalar the
same rule covers **72–78% of rows at ~10% risk**, hitting the target in one direction and missing by
one part in a thousand in the other.

That is the real value, and it is narrower and more concrete than "calibration helps": **calibration
makes a threshold portable.** A threshold fitted on one sample now means the same thing on another.

### 7d. What this changes downstream

`abstain.py` and `arbiter.py` both consume probabilities as probabilities. The honest ordering of work is
therefore:

- **Any threshold, cost, or risk-limiting logic must use calibrated probabilities** — it is unusable
  without them (coverage 0.000).
- **Any rank-based selection can use raw probabilities** — and calibrating first costs a little.
- The two should be fed differently, which neither module currently does.

