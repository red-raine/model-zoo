# Finding label errors: what failed, why, and what has to replace it

**Date:** 2026-10-22
**Tools:** `tools/verify/find_label_errors.py` (self-test 9/9), `tools/verify/train_verifier_nli.py`

---

## 1. I retracted my own "we are at the noise floor" claim

I said dev error (25.6%) ≈ `required_reannotation` rate (24.4%) means we are at the noise
floor. Decomposing it killed the claim:

| dev bucket | n | accuracy | error |
|---|---|---|---|
| reannotation-flagged | 121 | 0.6529 | **34.7%** |
| not flagged | 362 | 0.7735 | **22.7%** |
| **share of all our errors on flagged rows** | | | **33.9%** |

Flags carry a real **1.5× error enrichment** and are worth using. But they explain only a
third of our error. **22.7% error remains on labels nobody disputes.** The matching 24.4%
and 25.6% was a coincidence, and I built a conclusion on it.

(17 of 500 dev rows did not join to the raw file on exact claim text — probably punctuation
or Unicode normalisation. A 3.4% join miss is itself worth fixing before any per-row
analysis is trusted at the margin.)

---

## 2. The label-error finder, and its honest validation

`find_label_errors.py` scores each row on six signals ranked by how independent they are of
the model under evaluation:

| signal | weight | independence |
|---|---|---|
| `human_flag` (`required_reannotation`) | +0.30 | full — the dataset authors' own doubt |
| independent checkpoint contradicts gold, p ≥ 0.70 | +0.35 | intended as independent |
| two models confident on the same non-gold class | +0.45 | intended as independent |
| gold=SUPPORTED but evidence names source fake/satire/hoax | +0.40 | full — deterministic, auditable |
| gold=SUPPORTED but claim number absent from evidence | +0.25 | full — deterministic |
| all models unsure (max p < 0.45) | +0.10 | ambiguity, not mislabel |

### Bug 1: the first validation was circular

Scoring the finder against `required_reannotation` while `human_flag` is *an input* gave
**AUC = 1.0000**. That measured nothing but the fact that `0.30 > 0.0`. The finder now
returns two numbers — `suspicion` (all signals) and `suspicion_noflag` (independent signals
only) — and **only the second one is validated**.

### The real result

```
AUC(independent signals only, vs required_reannotation) = 0.5774
AUC(all signals, flag included — CIRCULAR)             = 0.8957
```

**0.577 is barely above noise.** The lexical, numeric and cross-model signals do **not**
find the label doubt that human annotators flagged. Reported as a negative result.

---

## 3. Why it failed: the "independent models" are not independent

The finder's strongest signal was "two independent models agree against gold". They are not
independent:

| measurement | value |
|---|---|
| agreement between our deberta-base and the CTU AVeriTeC-large | 0.708 |
| our errors | 128 |
| CTU zero-shot errors | 102 |
| both wrong | 58 |
| wrong the **same way** | 26 |
| **P(CTU wrong \| our model wrong)** | **0.453** |
| base error rate | 0.204 |
| **lift** | **2.22×** |

A 2.22× lift means the two models fail together far more often than chance. Both were
trained on AVeriTeC **train** — the same labels, the same annotation conventions, the same
evidence format. Their errors are correlated by construction.

**Therefore "two encoders agree against gold" is mostly shared inductive bias, not evidence
of a mislabel.** This is the whole reason AUC = 0.577, and no amount of reweighting fixes
it. More encoders trained on the same labels cannot arbitrate those labels.

This is the general form of a trap that has bitten this project repeatedly: **agreement
between models that share training data is evidence about the training data, not about the
world.**

---

## 4. What actually has to replace it: external evidence

If encoders cannot arbitrate labels, something outside the label distribution has to. The
signals that can, in order of strength:

1. **Primary sources** — the fact-checking article's own `justification` field, and the
   original claim URL. AVeriTeC ships both. The justification is *gold prose written by the
   annotator explaining their verdict*; a row where the justification contradicts the label
   is a demonstrable mislabel, not a model opinion.
2. **Retrieval against the live web** — `retrieval.py`'s 7 providers (wikipedia, wikidata,
   openalex, crossref, stackexchange, arxiv, searxng). If a claim says "45 million" and a
   primary source says 7 million, that is checkable.
3. **Deterministic channels** — `math_verify.py` (sympy, 15/15), `wikidata_relations.py`
   (35 relations), `logos.logic.syllogism` (all 24 valid forms). Exact, but settle only
   0.07% of AVeriTeC.
4. **Human flags** — `required_reannotation`, already available and already quantified above.

**Item 1 is free and unused.** `justification` and `fact_checking_article` are in the raw
JSON we already have on disk. Checking label-vs-justification consistency is a far stronger
error finder than any model agreement, and it costs no GPU and no API calls.

---

## 5. The cleaning experiment (designed, not yet run)

The decisive test of whether cleaning helps is **not** "does the finder rank flags" — it is
"does dropping rows improve dev accuracy". Dev must stay untouched or the result is
circular.

| arm | rows dropped from train | measures |
|---|---|---|
| baseline | 0 | the 0.7440 reference (already measured) |
| A — human flags only | 862 `required_reannotation` rows | does the authors' own doubt carry signal? |
| B — independent signals only | suspicion_noflag ≥ 0.40 | does OUR finder carry signal? |
| C — both | union | do they stack? |

Given AUC = 0.577 for arm B, my prediction is **arm A helps, arm B does nothing or hurts**.
Recording the prediction before running it is the point — it makes the result falsifiable
rather than narratable.

`find_label_errors.py --clean-train-out ... --clean-mode noflag|all` already emits the
filtered train files; the arms only need a retrain each (~28 min on the B50).

---

## 6. Correcting the compute claim: cloud GPU hours change the Mamba answer

**I was wrong to say Mamba pretraining is out of reach, and the reason matters.** My estimate
was for the Arc Pro B50 specifically. With hundreds of free cloud GPU hours per month:

| run | FLOPs (6ND) | A100 @ ~40% MFU (125 TF/s) | 4090/L40S-class @ ~35% MFU (28 TF/s) |
|---|---|---|---|
| 130M model, 10B tokens | 7.8e18 | **~17 h** | **~77 h** |
| 130M model, 30B tokens | 2.3e19 | ~52 h | ~231 h |
| 300M model, 10B tokens | 1.8e19 | ~40 h | ~179 h |

**A 130M Mamba continue-pretrain on 10B tokens is 17–77 GPU-hours.** That is inside a
monthly free allowance. The compute objection is withdrawn.

What remains true is narrower and more useful: **the measured bottleneck is not backbone
capacity.** It is (a) label noise we cannot yet separate from model error, and (b) retrieval
quality (+14–22 pts in the literature, versus a backbone swap that moves the tail by a few
points). Pretraining on Wikipedia does not teach a model whether *this* evidence contradicts
*this* claim — that is a pair-reasoning task with 2.5k examples.

Ranked use of those GPU-hours, best first:

1. **Harvest → SFT** on search trajectories from `retrieval.py` + `packed_judge.py` (tens of
   hours). Attacks the deployment distribution directly.
2. **k-fold cross-fitting** to turn 500 calibration rows into ~3k — recovers the 3.6 points
   lost to the carve and makes Mondrian viable. **No GPU needed at all.**
3. **Continue-pretrain the verifier on harvested claim-evidence pairs** (domain adaptation,
   hours).
4. **A small Mamba trained on verification data** — once we have 100k+ harvested pairs, not
   before. Training on the *task*, not on Wikipedia.
5. **Full Wikipedia + science pretrain** — last, and only if 1–4 plateau.

---

## 7. Files

| file | what |
|---|---|
| `tools/verify/find_label_errors.py` | six-signal suspicion scoring, non-circular validation, train cleaning. Self-test 9/9 |
| `tools/verify/train_verifier_nli.py` | CTU AVeriTeC-large post-training + zero-shot control + label-permutation detection. Self-test 5/5 |
| `B:\datasets\forge\verifier-av\label-errors.dev.json` | per-row suspicion + reason codes |
| `B:\datasets\forge\verifier-av\nli-zero-dev.jsonl` | CTU zero-shot per-row dev predictions (in ITS label order) |

Three bugs were caught in the making of this file, all of the same family as before — a
number that looked good and measured nothing:

1. Validating the finder against the human flag **while the flag was an input** → AUC 1.000.
2. Feeding the CTU checkpoint's **raw argmax** into the finder. Its head has Supported/Refuted
   **swapped** (`[1,0,2,3]`); unmapped, it invents ~250 phantom label errors.
3. Treating two models trained on the same labels as **independent voters** — measured lift
   2.22×, i.e. strongly correlated.
