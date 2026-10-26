# The decision-model slot: the calibration harness is built, the slot is not yet fillable

**Date:** 2026-10-26
**New:** `logos/calibrate.py` (16/16), `logos/decision_slot.py`
**Finding:** negative, and worth more than a positive one.

---

## 1. Where a decision model belongs

Agreed on the placement: **closed questions with calibrated output** — the `usable` validator, and
verdict classification. Not the arbiter, not the router. The arbiter takes *opinions and scores them*;
a decision model produces *one opinion with a probability*. Different jobs.

`laya` markets itself as trained with **strictly proper scoring rules**. That is a claim about a
model none of us has run. `calibrate.py` exists so the claim can be checked on our data instead of
believed:

- does its 0.7 mean 70%?
- does calibrating it beat leaving it alone, **out of sample**?
- **is it better than the hand rule it would replace?** ← the one that decides whether it earns a slot

## 2. What the harness measures, and why each part is needed

| metric | what it catches |
|---|---|
| **Brier** | strictly proper: minimised by the truth, so an "improvement" is real |
| **Murphy decomposition** | separates *calibration* (reliability) from *sharpness* (resolution) — one number hides which is broken |
| **log loss** | punishes confident mistakes harder |
| **ECE + reliability table** | what people mean by "looks calibrated", and the weakest of the three, so it never stands alone |

**A bug worth recording:** Murphy's three-term identity
`Brier = reliability − resolution + uncertainty` **does not close** on continuous forecasts. It is
exact only when the forecast is *constant within a bin* — i.e. for discrete scores. With a continuous
score there is a within-bin covariance term. Measured here: off by **0.0022**, with covariance
**+0.0014**. Small enough to look fine in a report and wrong enough to matter. The module now carries
the term, and a test asserts the three-term form *fails* on continuous input and *succeeds* on
discrete input, so nobody "fixes" it back.

**A second discipline:** every comparison is k-fold, and **in-sample and out-of-sample are printed
together**. Measured on noise: isotonic's held-out Brier is **0.288** against an in-sample **0.259** —
the flexibility bought nothing that generalises. `Isotonic` is included to be *measured*, not trusted.

## 3. The negative result

Question: can a decision model beat `date <= claim_date` on the `usable` question?

**The question cannot be answered with the data we have, and trying to answer it would produce a
lie.** There is no independent label for "usable" — the only label is the temporal gate, which *is*
the rule. So the rule's Brier against that label is **exactly 0.000000000000** by construction.

Any model scores ≥ 0. So the "result" would always be *"the rule wins, 0.00 vs 0.44"* — a
spectacular-looking table that only says the rule was graded on the answer key it wrote. That is the
circularity this project keeps finding, and manufacturing a table from it would have been the worst
possible output: a number that looks like evidence for a decision I already wanted.

What the fixture *does* establish, without circularity:

```
31 dated answers, 17 usable (54.8%)
days relative to claim: min −4732, max +1246, median −46
usable is NOT a route artifact: wayback 14/23, arquivo 5/2, last_modified 1/3
```

Every route produces both outcomes, so "usable" is genuinely a property of the **date**, which the
rule already computes exactly. There is nothing left for a model to learn.

## 4. So where does it actually go

| question | exact rule? | verdict |
|---|---|---|
| does this date precede the claim? | **yes** | **rule. Brier 0 is a floor; a model can only match or lose** |
| does this snippet support this claim? | no | **the first real home** — and AVeriTeC dev's 500 rows already carry Supported/Refuted/NEI labels |
| is this entity the same entity? | no | second home; labels are cheap to construct at n=500 |
| is this date *close enough*? | no, but only in a ~2-day band | not worth a model; it would be fitting the boundary |

**The prerequisite for the `usable` slot is a labelled set, not a better model.** That is the whole
finding. Training a classifier against gate labels would have produced a model that had learned
arithmetic, been reported as a win, and quietly added a failure mode whenever the claim date moved.

## 5. Honest limits

- **The harness is tested on synthetic distortions**, not on a real decision model. Whether laya or
  an unsloth-trained model is calibrated is still unknown — that is now cheap to check, not checked.
- **n=31 dated answers.** The route breakdown is descriptive only; per-route shares at n=3–23 cannot
  support a claim about routes.
- **`compare_to_rule` has not been run on a real model**, only on synthetic rule/model pairs, so it
  is verified to be unbiased but not yet verified to catch a real overfit model.
- **AVeriTeC labels are the obvious next target** and are already in the repo (`verify/averitec-dev-s42.json`),
  so the calibration work can proceed without waiting for anything new.

## 6. Next

1. **Point `calibrate.py` at the AVeriTeC dev verdict task** — real labels, a real closed question,
   and the measurement that decides whether a decision model earns the verdict slot.
2. **Only then** consider laya/ollaya/unsloth, and only on that task, not on `usable`.
3. Keep `usable` as the rule. Revisit only if the claim kind changes to something with a genuine
   judgement boundary.
