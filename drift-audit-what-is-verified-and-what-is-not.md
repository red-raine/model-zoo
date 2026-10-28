# Drift audit: what is verified, what is inherited, and the uncomfortable part

**Date:** 2026-10-28
**Trigger:** explicit instruction to question the experiments, look for hallucinations and project drift, and
re-read the original handoff.
**Result:** three inherited claims now verified live (one of them cold). Two overclaims corrected. One
structural drift named that is worse than any individual error.

---

## 1. Verified live in this audit (not inherited, not assumed)

| claim | status |
|---|---|
| syllogism rules reproduce **24/24 valid forms over all 256**, mismatches 0 | **verified now** by self-test |
| **51/51 lemmas machine-proved, 8/8 fallacies unprovable in real Lean** | **verified by a COLD run**, cache disabled, 470.5 s |
| softmax/attention fusion **violates monotonicity: 0.900 → 0.633** when a third expert joins | **verified now** by live self-test |

The Lean one matters most. The first check reported `cache 59 hits / 0 misses, elapsed 0.0s` — **I was
verifying a cache, not Lean.** With `--no-cache` it took 470 seconds of real Lean 4.35.0 and returned the
same result. The claim holds, but only after the audit forced the expensive path. A cache that is right
99 times is exactly the thing that hides the one time it is wrong.

## 2. Overclaims I am correcting

**a. "The model is ~4× overconfident."** What was measured is a temperature `T = 4.226` that minimises NLL,
i.e. **the raw logits are 4.23× too sharp.** "4× overconfident" is a loose gloss that suggests a ratio of
calibration error, which is a different quantity. The precise statement is the logit-scaling one.

**b. "99.9% precision is unreachable — the ceiling is 96.5%."** The 96.5% is a **point estimate on 200
rows of a development set that was used for tuning.** It is not a ceiling on the model, and it is
optimistically biased. The defensible statement is: *on this dev sample, at 40% coverage, the best
certifiable precision was 93.5% (95% LB) and the point estimate was 96.5%; both are dev estimates.*

## 3. Claims that are circular and labelled as such

- The refinement learning loop uses the temporal gate as its own label. It measures **route selection**,
  not truth. Labelled inline in `cascade.py` and in every report that quotes it.
- Recall on injected code defects is **circular** — we inject only the classes we can detect. Labelled.
- The 7.2% doctest figure comes from the local stdlib, which is unusually well documented. Labelled as an
  **over**statement of corpus coverage.
- AVeriTeC **dev is not a held-out set.** It fitted `T`, `tau`, weight modes, the review band and the
  strata edges. Every number from it is a development estimate. This has been stated but it deserves to be
  the loudest line in the file: **the project currently has no clean held-out set at all.**

## 4. The structural drift, which is the real finding

36 modules. 36 self-test suites. Real Lean, real Z3, real audit arithmetic, real corpus measurements.
And:

- **no trained model**
- **no cleaned corpus at scale**
- **no certified precision number** (because no held-out set and no human-labelled sheet)
- **no end-to-end run** that takes raw data and emits a validated corpus

The measurement apparatus became the project. That is the drift, and it is worse than any individual
overclaim because each module individually was justified. The instruction to *"question our experiments"*
is the correct response to it and should have come from me.

**The honest state:** we have a well-tested component library and no product. The next work must produce an
artifact, not another gate.

## 5. What the handoff actually said, and what I have drifted from

Re-reading `AGENTS.md` and the spine notes, the stated objectives are: clean datasets with our own
cascading tools, build our own corpora, add a fact-checking/uncertainty layer, and a training platform.
Two things in there I have under-served:

1. **"build our own corpora"** — I have measured other people's corpora (AVeriTeC, stdlib) extensively and
   built almost none of my own at scale.
2. **"a model training platform"** — untouched. `train/` exists from earlier work; I have added nothing.

And two things I have done that were not asked for and are now load-bearing: the arbiter (which cannot beat
its best member) and a calibration cascade for a model we did not train.

## 6. Hallucination check — specific numeric claims re-derived

| claim | how it was produced | verdict |
|---|---|---|
| dev 500 rows, train 3066 | read from files | **ok** |
| ACCURACY 0.8080 / macro-F1 0.6784 | recomputed from raw logits, reproduces recorded values exactly | **ok** |
| majority 0.610, random 0.315 | recomputed | **ok** |
| SUPPORTED 24.4% of dev | 122/500, read from label counts | **ok** |
| T = 4.226 | two independent protocols agreed 4.23 vs 4.23 | **ok** |
| ECE 0.098 → 0.031 | cross-fitted and half-split agreed | **ok** |
| audit 29 / 599 / 2995 / 29956 | closed form `ln(a)/ln(p)`, checked against Clopper-Pearson at the boundary | **ok** |
| structural reject 0/400, semantic 3/400 | re-run on seeded corpus | **ok** |
| doctests 7.2% | static scan, no execution | **ok** |
| **"the bottom quartile is at chance (50.4%)"** | band-0 precision on 125 dev rows | **ok as a dev statistic, NOT a model property** |

No fabricated numbers found. The errors were of **interpretation** (a cache read as a proof, a dev sample
read as a ceiling, a temperature read as an overconfidence ratio), not of arithmetic.

## 7. What this audit changes about the plan

1. **The next deliverable is an artifact, not a gate.** Either a cleaned corpus of a stated size with a
   certified precision, or a trained model — and the corpus comes first because it gates the model.
2. **A held-out set must exist before any number is reported again.** Cheapest: carve a never-touched slice
   of dev and stop looking at it. Correct: obtain AVeriTeC test labels.
3. **The arbiter stays out of the critical path** until it beats its best member on some task. It is
   currently a component looking for a reason.
4. **Every future claim must state its set** — dev, held-out, or synthetic — in the same sentence as the
   number. The audit found no wrong arithmetic and several wrong attributions, which is what you would
   expect from a project that stopped saying which set it was measuring.
