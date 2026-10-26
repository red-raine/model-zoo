# Are we ready for a bigger training and eval set? Not yet — and size is the third problem

**Date:** 2026-10-26
**New:** `logos/eval_audit.py` (12/12), `B:\datasets\forge\verifier-av\dev.dedup.jsonl`
**Answer:** no, but not for the reason the question assumes.

---

## 0. A correction to my own last commit

I wrote that *"AVeriTeC dev's 500 rows with Supported/Refuted/NEI labels are already in this repo."*
**That was wrong.** `verify/averitec-dev-s42.json` is 1.3 KB — a *result summary* from a past run
(n=200, baselines, confusion matrix), not labels. I inferred contents from a filename, which is the
same error as the dead-archive-counted-as-a-backup, one domain over.

The real data does exist, elsewhere: `B:\datasets\forge\verifier-av\` holds **3066 gated training
rows** and **500 dev rows**, plus frozen embeddings, NLI logits and label-error audits.

## 1. What the splits actually contain

```
train 3066    dev 500     (dev deduped -> 482)

class                   train    dev   dev prev
REFUTED                  1742    305     61.0%
SUPPORTED                 848    122     24.4%
CONFLICTING               194     38      7.6%
NOT_ENOUGH_EVIDENCE       282     35      7.0%
```

Distribution shift train→dev is mild (**≤ 4.2 pp** on every class), so dev is a reasonable sample of
train. The baseline any model must beat:

```
majority class accuracy   0.610
predicting the dev prior  Brier 0.2356, log loss 1.0277
```

**The only historical model number in the repo is accuracy 0.245.** Against a majority baseline of
0.610 and a random baseline of 0.315, **that is below random.** Whatever produced it was broken, and
the honest reading is that we do not have a working model to scale yet. Its confusion matrix explains
it: 77 of 121 REFUTED claims were predicted SUPPORTED, and 27 more REFUTED→CONFLICTING. A collapsed
predictor, not a weak one.

## 2. Blocker one: 3.6% of dev is in train, and it is concentrated in the class we trusted

```
exact-normalised claim overlap  dev->train : 17/500  (3.4%)
13-gram containment             dev->train :  4/500  (0.8%)
combined dropped after dedup               : 18/500  (3.6%)
```

Removing them:

```
class                   before  after  lost
REFUTED                    305    303     2
NOT_ENOUGH_EVIDENCE         35     34     1
CONFLICTING                 38     38     0
SUPPORTED                  122    107    15   <-- 12% of the class
```

**83% of all leakage landed in SUPPORTED.** That is not random scatter. And SUPPORTED is precisely
the class where earlier work reported the one gain treated as solid: **+0.188, CI [+0.042, +0.333]**.

I am not claiming that result is wrong. I am claiming it **cannot be quoted again until it is re-run
on `dev.dedup.jsonl`**, because 12% of the rows it was measured on were in the training set. A gain
estimated on contaminated rows is an upper bound at best, and nobody checked which rows were
contaminated at the time.

**This is the first thing to fix, before any new data is bought.**

## 3. Blocker two: the eval set can measure exactly one class

The right test is **paired (McNemar)** — every model is scored on the same claims, so only the rows
where two models disagree carry information. That is far cheaper than an unpaired comparison:

```
+10pp at 20% discordance:  157 pairs  vs  776 independent rows   (4.9x)
```

Using the paired test at the deduped sizes, against a +10 pp target:

| class | n | smallest detectable | verdict |
|---|---|---|---|
| **REFUTED** | 303 | **7.2%** | **measurable** |
| SUPPORTED | 107 | 12.1% | not at +10pp |
| CONFLICTING | 38 | 20.3% | needs ~2,000 total rows |
| NOT_ENOUGH_EVIDENCE | 34 | 21.5% | needs ~2,200 total rows |

**Zero of these can see the Conflicting effect we were chasing.** At n=38 the smallest detectable
difference is ~20 pp, so the earlier "Conflicting doubling" with CI **[−0.038, +0.323]** was
unmeasurable *by construction*. That was predictable before the experiment ran, with the arithmetic
now in `eval_audit.py`, and it is the single most useful thing this module buys us: it refuses to let
us spend a month measuring a number the set cannot resolve.

## 4. So what is the actual answer

**Not ready — and buying a bigger set now would buy precision on a broken pipeline.**

Order of work:

1. **Remove the leakage** (done: `dev.dedup.jsonl`, 482 rows) and **re-run the SUPPORTED +0.188
   claim** on it. One afternoon. It may survive; if it does not, we just avoided building on it.
2. **Establish a baseline above majority 0.610.** A prior-only predictor gets Brier 0.2356 — that is
   the floor. Nothing should be trained until something clears it, because the current best recorded
   number is *below random* and any scaling of it measures the bug more precisely.
3. **Only then buy data**, and buy it *stratified*: the binding constraint is CONFLICTING and NEI at
   ~35 rows each. A dataset 4× larger with the same skew still could not measure them. **What we need
   is not more rows, it is more of the rare classes** — the difference between 500 more random rows
   and 500 correctly stratified rows is the difference between still-unmeasurable and measurable.
4. **Calibrate on the verdict task** with `calibrate.py` once step 2 exists — that is the closed
   question a decision model can actually own, and the place laya/ollaya/unsloth would be tested.

## 5. Honest limits

- **`min_detectable_delta_paired` assumes 20% discordance**, which is an assumption, not a measurement
  from our models. If two models agree 95% of the time, the detectable delta is smaller than shown;
  if they agree 60%, it is larger. The real discordance must be measured from a pair of actual
  predictions before these numbers are treated as final.
- **Power is not validity.** A set that can detect a difference does not guarantee the difference is
  the thing you think you measured.
- **The dedup is exact-normalised plus 13-gram**, so paraphrase-level leakage survives it. A semantic
  dedup would remove more.
- **3066 training rows is small** for fine-tuning anything, which is an argument for the frozen-embedding
  approach already present (`emb-frozen.npz`) over full training.

## 6. Next

1. Re-run the SUPPORTED claim on the deduped dev split. Highest value, lowest cost.
2. Measure the **actual discordance rate** between two real predictors, replacing the 20% assumption.
3. Establish a baseline that beats majority 0.610; publish it as the number to beat.
4. Only then a stratified expansion of CONFLICTING and NOT_ENOUGH_EVIDENCE.
