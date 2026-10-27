# The arbiter on real labels: it is insurance, not an improvement — and only if confidences share a scale

**Date:** 2026-10-27
**New:** `logos/arbitrate_verdict.py` (self-tests pass), `Arbiter(weight_mode=...)`
**First honest test of the arbiter**, and it found a bug in `mcnemar` on the way.

---

## 1. Why this test had to happen

The arbiter had only ever been used on dated facts — where every "method" is an archive route, there is
no ground truth, and its value could only be measured against a proxy. On AVeriTeC verdicts there is
real gold, three genuinely different methods, and a baseline worth beating: the NLI model's
**macro-F1 0.6784** on the 500 dev rows.

## 2. A bug in `mcnemar` that this output exposed

First run:

```
arbiter 0.4528  vs  nli 0.6784   (-0.2257)
McNemar: 120 rows fixed, 40 rows broken, p=0.0000
```

**120 rows fixed and 40 broken, yet the score dropped by 0.23.** The numbers were right; the labels
attached to them were transposed. `baseline.mcnemar` had

```python
n01 = ~ok_a & ok_b     # a wrong, b right   -> labelled "b"
n10 = ok_a & ~ok_b     # a right, b wrong   -> labelled "c"
```

but the standard 2×2 convention is the reverse: **b = a right & b wrong**, **c = a wrong & b right**.
Anyone reading `b`/`c` without checking the implementation would have concluded that the arbiter
*improved* by 80 rows while its macro-F1 fell by a fifth. The only reason it was caught is that the
direction contradicted the score — which is the argument for printing the score next to the test in the
first place.

Fixed at the source, with the convention stated in the docstring and the three affected tests updated.

## 3. The result: arbitration LOSES, badly, and the reason is not accuracy

```
nli      macro-F1 0.6784   acc 0.8080     <- the strong method
probe    macro-F1 0.4528   acc 0.6480     <- linear probe on frozen embeddings
prior    macro-F1 0.1894   acc 0.6100     <- always predict the majority class

nli + probe, cross-fitted arbiter   0.4528   (-0.2257)   40 fixed / 120 broken
```

The arbiter scored **exactly the probe's own macro-F1**. It was not averaging two opinions — it was
obeying one, the weaker one. The learned reliabilities were *correctly ordered*
(`nli 0.787 > probe 0.640 > prior 0.609`); the ranking was right and the decision was still wrong.

Neither weight scheme changed it (`mean` and `logodds` both 0.4528), so the weights were not the cause.

## 4. The mechanism: pooling compares confidence *scales*, not accuracies

Log-linear pooling multiplies `conf_m ^ w_m` across methods. That treats the numbers as if they shared
a scale. They did not:

| method | right how often | says |
|---|---|---|
| NLI | **79%** | 0.55 *(flattened by T = 4.23)* |
| probe | 64% | **0.95** *(raw softmax from a linear head: overconfident)* |

The probe wins on *scale* while losing on *accuracy*, because `log(0.95) >> log(0.55)`.

**The fix, and it is one line of meaning:** map each method's confidence to **`P(this method is
correct)`**, fitted out of fold. Now the numbers mean the same thing before they are multiplied.

```
nli + probe, raw confidences          0.4528
nli + probe, calibrated confidences   0.6765   (+0.2237)
nli alone                             0.6784
```

**A 0.224 macro-F1 recovery from making the confidences commensurable.** That is larger than any model
improvement achieved in this project so far, and it came from fixing a units mismatch.

## 5. The limit, which is the more important finding

Even fully calibrated, the best configuration scores **0.6765 against 0.6784** for the best member alone
— a difference of 0.0019, i.e. neutral, not a win.

> **An arbiter cannot beat its best member. It is insurance against a bad member, not an improvement
> over a good one.**

That reframes what this component is for. When one method dominates, arbitration can only match it, and
it needs a calibration step to avoid being dragged *below* it. Its value must come from cases where no
single method dominates — which is the date-routing case (four routes, none reliable) and explicitly
not the verdict case (one fine-tuned model far ahead of a linear probe).

The `prior` as a "method" also contributed nothing: `nli+probe+prior` scored 0.6774, `nli+probe` 0.6765.
A marginal base rate is a **prior**, not a **voter** — it carries no independent information, and listing
it as a method invites it to vote twice with whatever it happens to agree with.

## 6. What changed in the code

- `mcnemar` convention fixed, docstring states it, tests updated (3 call sites).
- `Arbiter(weight_mode="mean" | "logodds")`. **Neither helped here** — `logodds` compresses to 1.44×
  where `mean` gives 1.29×, and 0.4528 either way. It is kept because it is the right currency for
  pooling in principle, and it is recorded as not having been the problem.
- `arbitrate_verdict.py`: the first test of the arbiter on real labels, cross-fitted, with the
  independence and scale diagnostics built in.
- `Candidate.from_probs` now raises a clear error on a scalar or single-class input instead of an
  IndexError three frames away.

## 7. Honest limits

- **One task, one dataset, 500 dev rows, two folds.** The cross-fit is 2-fold because 500 rows do not
  support more; every number has a wide interval that is not reported.
- **The probe is a straw man.** A linear head on mean-pooled frozen features versus a fine-tuned DeBERTa
  is not a fair contest, so "arbitration loses" partly reflects a deliberately weak second method. The
  *mechanism* (scale, not accuracy) does not depend on that and was confirmed by the 0.224 recovery.
- **The scale fix was fitted on the same 500 rows** it was evaluated on, out of fold. Two folds is thin.
- **`nli` alone is the best config**, so the honest reading is "no evidence arbitration helps on
  verdicts", not "arbitration is proven useless".

## 8. Next

1. **Find a task where no method dominates** — that is where the arbiter can earn its place. Candidates:
   `entity_lookup` (many heterogeneous sources) or date precedence across genuinely independent archives.
2. **Make commensurable confidences a hard precondition** in `arbiter.decide()`: refuse candidates whose
   confidence is not marked as calibrated. The 0.224 loss is too large to leave to convention.
3. **Never list a base rate as a method.** It is a prior.
