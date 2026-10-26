# Correction: there IS a working baseline (0.8008), the leak is small, and the rare classes are the real wall

**Date:** 2026-10-26
**New:** `logos/baseline.py` (13/13: `--run` frozen probe, `--nli` verification)
**Corrects:** the previous note's claim that "there is no working model to scale"

---

## 1. What I got wrong

I reported that the only model result on record was **accuracy 0.245** — below random — and concluded
there was no working model, so scaling would only measure a bug.

**That was wrong.** `B:\datasets\forge\verifier-av\nli-post-result.json` records
`ctu-aic/deberta-v3-large-AVeriTeC-nli` at **acc 0.808, macro-F1 0.678**. I read one stale summary
(`averitec-dev-s42.json`, from a different label-generation system) and concluded from it that
nothing worked, without looking at the NLI results sitting in the same directory.

Twice now the failure has been the same: **inferring the state of the project from one small file.**
The fix is in the module — `baseline.py --nli` reloads the per-row logits and recomputes from
scratch, so the number in the report cannot drift from the data.

## 2. Verified independently, not taken on trust

Recomputed from `nli-post-logits.jsonl` (3566 rows: 2566 fit / 500 calib / 500 dev):

```
raw accuracy 0.9613, best permutation (0,1,2,3) -> 0.9613
identity is optimal: True   (stored predictions are correctly ordered)
```

The stored predictions are fine. But an earlier stage had a real bug worth recording:

| field | value | reading |
|---|---|---|
| `zero_shot.acc` | **0.106** | looks like a destroyed model |
| `zero_shot_permuted.acc` | **0.796** | same predictions, reordered |
| `permutation` | `[1, 0, 2, 3]` | **REFUTED and SUPPORTED were swapped** |
| `permutation_is_identity` | `false` | the bug was caught at the time |

**A label-order mismatch was reported as a 0.106 model.** That is the third appearance of the same
fingerprint in this project: a near-chance number that is diagnosed as "the model is broken" when the
labels were never aligned. `0.244` was the earlier one. Any accuracy near class prevalence deserves a
permutation check before any conclusion.

My recomputation reproduces the recorded dev numbers **exactly** — 0.8080 accuracy and 0.6784
macro-F1 — which is what makes the rest of this file trustworthy.

## 3. The real baseline, and what the leak costs

| | full dev (500) | deduped (482) | cost of the leak |
|---|---|---|---|
| accuracy | 0.8080 | **0.8008** | −0.72 pp |
| macro-F1 | 0.6784 | **0.6688** | −0.96 pp |
| majority baseline | 0.6100 | 0.6286 | — |
| vs majority | p = 4.9e−15 | **p = 7.2e−12** | still decisive |

**This baseline is real and it is the number to beat.** My frozen-embedding linear probe reaches only
0.647 accuracy / 0.453 macro-F1, and does *not* significantly beat majority (McNemar p = 0.507) — so
the pooled-embedding route is not a substitute for the NLI model. Recorded for the record: **mean
pooling beats CLS pooling by 13.7 pp** on that probe, which is a genuine feature-choice finding.

### The leak cannot explain an 18.8 pp class gain, but it is not nothing

```
SUPPORTED            15 leaked of 122 rows = 12.3% of the class
REFUTED               2 leaked of 305 rows =  0.7%
NOT_ENOUGH_EVIDENCE   1 leaked of  35 rows =  2.9%
CONFLICTING           0 leaked of  38 rows =  0.0%
```

Measured cost on SUPPORTED: F1 0.852 → 0.830 (−2.2 pp), precision −1.5, recall −2.7.

The **maximum possible** inflation of a SUPPORTED-class claim is **12.3 pp** (15 of 122 rows), which
would require a configuration to have memorised all fifteen. So the earlier **+0.188 SUPPORTED** result
probably survives the leakage check — but it is not free of doubt, and it must be re-run on
`dev.dedup.jsonl` before being quoted again. My previous note overstated this as "invalid until
re-run"; the more accurate statement is **"at most 12.3 pp of it could be leak, so re-run to close a
12 pp hole, not to rescue the result."**

## 4. The real wall is the rare classes, and it is now measured from both sides

Per-class on the deduped split:

| class | n | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| REFUTED | 303 | 306 | 0.879 | 0.888 | **0.883** |
| SUPPORTED | 107 | 93 | 0.892 | 0.776 | **0.830** |
| NOT_ENOUGH_EVIDENCE | 34 | 16 | 0.938 | **0.441** | 0.600 |
| CONFLICTING | 38 | **67** | **0.284** | 0.500 | **0.362** |

Two distinct failure modes:

- **CONFLICTING is over-predicted**: 67 predictions for 38 real rows, precision 0.284. The model reaches
  for it and is usually wrong.
- **NEI is under-predicted**: 16 predictions for 34 rows, recall 0.441. The model avoids it.

And the two walls meet exactly where predicted. The conformal analysis in `uncertainty.json` reports:

```
mondrian class 3 (CONFLICTING): n_class=34, coverage 0.0, "too few calibration rows"
```

**The same 34 rows that make the model's F1 fragile also make conformal abstention impossible.** The
power analysis said these classes cannot be measured; the calibration says they cannot be guarded.
Two independent methods, one conclusion: **the binding constraint is rare-class row count, not model
quality and not total dataset size.**

## 5. What is actually true about readiness

| question | answer |
|---|---|
| is there a working baseline? | **yes — 0.8008 / 0.6688 deduped**, beats majority p=7e-12 |
| does it need a bigger dataset? | **not for REFUTED (F1 0.883) or SUPPORTED (0.830)** |
| where does it need data? | **CONFLICTING (F1 0.362, n=38) and NEI (F1 0.600, n=34)** |
| will a 4× bigger set with the same skew help? | **no** — it adds ~150 CONFLICTING rows at 8% prevalence, which buys ~5 pp resolution |
| does the leak invalidate anything? | it costs 0.7–1.0 pp overall, ≤12.3 pp on SUPPORTED, so re-run but do not panic |
| is calibration on the verdict task ready? | **yes** — real labels, real model, real probabilities |

## 6. Next, in order

1. **Re-run the SUPPORTED comparison on `dev.dedup.jsonl`** to close the 12.3 pp hole. Cheap.
2. **Attack CONFLICTING specifically** — it is the worst class (0.362) and the model over-predicts it
   67 vs 38, which is a decision-threshold problem before it is a data problem. A class-prior
   correction or Mondrian-balanced threshold is cheaper than more rows and should be tried first.
3. **Calibrate on the verdict task now** — `calibrate.py` has real labels and a real model to point at.
   The recorded calibration already moves ECE 0.176 → 0.083, so the harness can confirm or contradict
   that on its own terms.
4. **Only then** stratified data for CONFLICTING and NEI.

## 7. Honest limits

- **One model, one dataset.** `deberta-v3-large-AVeriTeC-nli` is a public checkpoint trained on this
  task, so 0.80 is a transfer number, not a result we produced. Beating it is the actual goal.
- **The frozen probe is a weak comparison**, not a candidate: a linear head on mean-pooled features
  versus a fine-tuned DeBERTa is not a fair contest, and it is reported only as the cheap floor.
- **The dedup is exact + 13-gram**, so paraphrase-level leakage survives.
- **`min_detectable_delta_paired` still runs on the measured 41.1% discordance**, which came from
  comparing CLS against mean pooling — two weak predictors. Real discordance between two *strong*
  models will be lower, which makes the eval set more capable than the current figure suggests. That
  has to be re-measured against a real pair.
