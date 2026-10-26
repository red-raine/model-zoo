# CONFLICTING is not a threshold problem, but the model is 4× overconfident

**Date:** 2026-10-26
**New:** `logos/verdict_tune.py` (10/10)
**Two results:** one negative and decisive, one positive and large. Plus a data-integrity bug caught on the way.

---

## 1. The data-integrity bug, found by a check I nearly did not write

Before tuning anything, `verdict_tune.py` checks whether the tuning split overlaps the splits it must
stay away from. Output:

```
calib n=500 (tuning only)   dev n=500 (reporting only, 18 deduped)
  calib/dev claim overlap: 1   calib/train overlap: 496   <- TUNING WOULD LEAK
```

**496 of 500 `calib` rows are in the training set.** The split named "calib" is not held out from
train, so every calibration or threshold number ever fitted on it was fitted on seen data — including
the recorded `ece_before 0.176 → ece_after 0.083` in `uncertainty.json`.

The symptom was already visible and I nearly read past it: the first tuning attempt chose **`tau = 0.0`
with a flat plateau across the entire `[0, 2]` range**. A flat objective is not a robust optimum; it is
what a model that has memorised its data looks like. The overlap check explained it in one line.

**The general lesson: a suspiciously flat or degenerate tuning surface is evidence of leakage, not
evidence that the hyperparameter does not matter.** The check costs four lines and it caught something
that would have silently invalidated every subsequent tuned number.

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

## 3. Positive result: the model is about 4× overconfident, and one scalar fixes it

Same cross-fitted protocol, now for temperature:

```
T chosen per fold  : [4.24, 4.27, 4.22, 4.25, 4.17]     spread 0.10 -> stable
ECE                : 0.0960 -> 0.0240    (-75%)
Brier              : 0.0969 -> 0.0783    (-19%)
```

Temperature scaling cannot change a prediction, so accuracy and macro-F1 are untouched at 0.8008 /
0.6688. What changes is whether the probabilities **mean anything** — and a 75% reduction in expected
calibration error, reproduced out of fold with a parameter that barely moves between folds, is a real
result rather than a fitted one.

Two consequences that matter more than the number itself:

- **The recorded calibration was both wrong and pessimistic.** `uncertainty.json` reports
  `ece_before 0.176 → ece_after 0.083`, fitted on the leaked `calib` split. The honest cross-fitted
  figure is **0.096 → 0.024**. The leaked fit understated how calibratable this model is.
- **The abstention machinery can now be trusted.** `abstain.py` (Chow's rule) and `arbiter.py` both
  consume probabilities as if they were probabilities. At ECE 0.096 that assumption is false in a way
  that silently distorts every risk-coverage curve; at 0.024 it holds. A single scalar, fitted out of
  fold, is the difference between a conformal guarantee and a number that looks like one.

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

1. **Adopt T ≈ 4.2 as a fixed post-hoc scalar** for this checkpoint, recorded with the dev cross-fitted
   ECE so nobody re-derives it from the contaminated split.
2. **Rebuild a clean calibration split.** `calib` is 496/500 inside train; until a genuinely held-out
   calibration set exists, every tuned scalar depends on cross-fitting inside dev.
3. **Then** the stratified data for CONFLICTING and NEI — now with two ruled-out alternatives behind it.

