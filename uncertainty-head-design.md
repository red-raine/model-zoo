# Uncertainty head + search harness — design, measurement, and what NOT to build

**Status:** built, measured, one negative result reported as such.
**Date:** 2026-10-22
**Tools:** `tools/verify/uncertainty.py` (self-test 7/7), `tools/verify/train_verifier_cls.py`
**Artifacts:** `B:\datasets\forge\verifier-av\cls-unc\{logits.jsonl,uncertainty.json,model.safetensors}`

---

## 0. First, a correction that changes what we are allowed to claim

`0.714` is **not** an accuracy ceiling.

```
0.714 = 1/4 + (3/4) × 0.619
```

That identity is `P_o = 1/K + (1 − 1/K)·κ` for K=4 with the reported free-marginal
κ = 0.619. It is **pairwise agreement between two single annotators**, not a bound on
accuracy against an aggregated gold label. I had been treating it as a ceiling and using it
to explain why 0.780 looked anomalous. It doesn't bound anything.

So the standing question is not "how did we beat 0.714" but "what is the noise floor", and
that is measurable. See §1.

---

## 1. The noise floor, measured

### Annotator-flagged label noise

AVeriTeC ships a `required_reannotation` boolean per claim — the dataset authors' own flag
that a label needs re-annotation.

| split | flagged | rate |
|---|---|---|
| train | 862 / 3068 | **28.1%** |
| dev | 122 / 500 | **24.4%** |

**Our dev error at full coverage is 25.6%.** The fraction of dev labels the dataset's own
authors flagged as needing re-annotation is 24.4%. Those two numbers are the same size.

That is the most important sentence in this document: **we are at the noise floor.** More
capacity will not buy accuracy here; it will fit disputes. Anyone reporting a number above
~0.76 on this dev split is measuring something other than claim verification.

(Do not confuse this 0.244 with the other 0.244 — the `Supported` dev prevalence. Both
appear in this project and only one of them is a bug fingerprint.)

### Leakage — measured, small, and reported rather than ignored

| check | count | effect |
|---|---|---|
| dev claims appearing verbatim in train | 5 / 500 = **1.0%** | all 5 predicted correctly |
| dev accuracy excluding those 5 | **0.7414** vs 0.7440 | inflation **0.26 points** |
| dev rows sharing a fact-checking *article* with train | 35 / 500 = **7.0%** | skewed Supported 23 / NEI 11 |
| duplicate claims inside dev itself | ≥1 | the Nigeria 52% claim appears twice |

The leakage is real but bounded under a point. Ablation and control results are unaffected.

---

## 2. Answering the architecture question: whole model head, or a numpy net?

**Decision: neither a new deep head nor evidential deep learning. Temperature scaling for
the probability, and a shallow logistic head only if it beats the free baseline — which on
this task it does not.**

The five candidates, and the specific failure each one has:

| approach | cost | why it is / isn't used here |
|---|---|---|
| **Softmax confidence** | free | Not a correctness probability. Measured ECE **0.189** on dev. 2.3× overconfident. |
| **Temperature scaling** | 1 parameter | **CHOSEN.** ECE 0.189 → **0.056**, argmax-invariant so accuracy cannot move. |
| **Evidential deep learning (Dirichlet/EDL)** | one extra head | Rejected. The evidence term is known to fit poorly and collapse; it produces confident-but-wrong rather than a calibrated correctness probability. It also estimates aleatoric/epistemic *label* uncertainty, which is not the same quantity as "is this argmax right". |
| **MC-dropout / deep ensembles** | 5–20× inference | Rejected. Reported to need >20% error at 60% coverage where selective risk needs ~10%. For a 3k-row text task the variance estimate is dominated by estimator noise. |
| **BH-style conformal selection** | cheap | Rejected. Provides no rejection at all in a large fraction of replicates — it fails open, which is the worst failure mode for a gate. |
| **Separate logistic correctness head** | ~40:1 on 500 rows | **BUILT, TESTED, LOST.** See §3. Reported as a negative result. |

**Why not "just make it a bigger model with an uncertainty head".** Because §1 says the error
is disputed labels, not capacity. A bigger head would fit the disputes more confidently.

---

## 3. The measurement (and the negative result)

Calibration split: **500 rows carved out of train** (seed 42), never trained on. Dev is
touched once. Verifier retrained on the remaining 2566 rows:

| | accuracy | macro-F1 |
|---|---|---|
| trained on all 3066 (previous run) | 0.7800 | 0.6130 |
| trained on 2566, dev evaluated | **0.7440** | 0.5876 |

**The clean calibration split costs 3.6 accuracy points.** That is the honest price of not
fitting the calibrator on the test set, and it is worth paying.

Per-class F1: Supported .664 · Refuted .829 · NEI .643 · **Conflicting .214**.

### Temperature scaling works

| split | ECE before | ECE after | Brier before | Brier after | accuracy |
|---|---|---|---|---|---|
| calib | 0.2061 | 0.0640 | 0.4761 | 0.4014 | 0.7200 → 0.7200 |
| **dev** | **0.1888** | **0.0557** | 0.4271 | 0.3669 | **0.7440 → 0.7440** |

T = **2.3194**, fitted on calib. Accuracy is bit-identical before and after (asserted in the
code) because positive scaling preserves the argmax. This is the "how likely is this true"
number, and it is essentially free.

### The learned correctness head LOSES to raw confidence

Candidates scored by AUC against "was the argmax correct", **selected on calib, never dev**:

| score | AUC calib | AUC dev |
|---|---|---|
| **`max_p` calibrated (free)** | **0.7455** | **0.7667** |
| `neg_entropy` (free) | 0.7443 | 0.7619 |
| logistic correctness head (15 features) | 0.7188 | 0.7338 |

**The 15-feature logistic head is worse than the free max-probability.** Feature ablation
says why:

| feature set | AUC calib OOF | AUC dev |
|---|---|---|
| 15 (logits + text stats) | 0.7188 | 0.7338 |
| 9 (logits only) | 0.7344 | 0.7426 |
| 6 (text stats only) | 0.5922 | 0.6320 |

The six cheap text features (word counts, digit density, evidence-item count) **actively
hurt**: adding them drops dev AUC from 0.7426 to 0.7338. With 500 calibration rows and a
72% base rate, the minority class ("wrong") has only ~140 effective examples, which is not
enough to estimate 15 coefficients well. The logistic head learns noise.

**Reported as a negative result.** Had I selected on dev I would have taken the head's
numbers at face value or, worse, tuned until it "won" on the test set. Selecting on calib is
the whole discipline.

---

## 4. The part that does work: a selective-risk controller

The value is not the probability, it is the **tested operating curve**. Threshold chosen on
calib against a one-sided Clopper–Pearson **upper confidence bound** on the accepted error
rate (δ = 0.10) — not against the observed error rate, which is how selective classifiers
end up promising less than they deliver.

| α | calib coverage | calib risk | UCB | **dev coverage** | **dev risk** | dev macro-F1 | promise |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.000 | — | 1.000 | **0.000** | — | — | nothing certifiable |
| 0.10 | 0.422 | 0.071 | 0.099 | **0.444** | **0.090** | 0.667 | HELD |
| 0.15 | 0.506 | 0.119 | 0.149 | **0.544** | **0.114** | 0.642 | HELD |
| 0.20 | 0.642 | 0.168 | 0.198 | **0.680** | **0.156** | 0.656 | HELD |
| 0.25 | 0.792 | 0.220 | 0.249 | **0.826** | **0.189** | 0.638 | HELD |

Every non-trivial operating point **HELD on dev**. α = 0.05 yields zero coverage because
500 calibration rows cannot certify a 5% error rate — reported as an empty row rather than
quietly picking a nicer threshold.

**Headline:** at full coverage the verifier is wrong **25.6%** of the time. Accept the 54%
it is most confident about and it is wrong **11.4%** of the time — a **2.2×** error
reduction, with the promise tested.

### Risk–coverage curve

| coverage | 5% | 10% | 20% | 30% | 50% | 70% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|
| error | 0.040 | 0.060 | 0.070 | 0.073 | 0.108 | 0.154 | 0.213 | **0.256** |

AURC **0.1222**, excess over a random ranking **−0.1338**. Monotone and well-behaved.

### Mondrian (class-conditional) thresholds: failed honestly

| class | threshold | dev accepted | dev risk |
|---|---|---|---|
| Supported | 0.849 | 24 | 0.167 |
| Refuted | 0.796 | 209 | 0.120 |
| Not Enough Evidence | 1.000 | **0** | — |
| Conflicting | 1.000 | **0** | — |

Coverage 0.466 vs 0.544 global — **Mondrian bought nothing**. The per-class calibration
subsets are 21–270 rows, too small to certify 15% risk, so it abstains entirely on the two
classes we most wanted to protect. The intent was right (stop the tail being coin-flipped);
the data cannot support it at n=500. Needs a larger calibration set, roughly 300+ per class.

---

## 5. "How likely true AND logical" — these are two signals, not one head

Truth and logical validity are different properties: a claim can be internally coherent and
false. They should not be one number.

- **truth_score** = temperature-scaled `max_p` (§3). Calibrated, measured, ECE 0.056.
- **logic_score** = agreement/veto from the **deterministic channels** — `sympy`,
  `wikidata_relations.py`, `logos.logic.syllogism` (all 24 valid syllogistic forms verified).

The conjunction rule is **veto, not average**: a deterministic contradiction overrides the
model verdict; a deterministic confirmation upgrades confidence. Averaging a proof with a
guess is how the guess contaminates the proof.

**Coverage caveat, measured earlier:** the deterministic channels settle **2 of 3,067**
AVeriTeC claims (0.07%), because AVeriTeC claims are narrative events, not triples. So
`logic_score` is a veto that almost never fires on *this* corpus. It will fire on FactKG
(108k claims with gold relation paths), which is where the Wikidata channel belongs.

---

## 6. Teaching it to search — the honest compute estimate

The operator's ask: prod it to search primary/academic sources, with a harness to train it.

| what it takes | reality |
|---|---|
| smallest trained search policies (Search-R1 lineage, R1-Searcher, ReARTeR) | **3B parameters minimum** |
| ASearcher-class RL for search | **~16,000 H800-hours** |
| our hardware | **1 × Intel Arc Pro B50, 16 GB** |
| our labelled data | 3,066 claims |

Training a search policy is out of reach by roughly four orders of magnitude of compute.
Any plan that says otherwise is wrong.

**The cheapest defensible version is a harness, not a trained policy:**

```
retrieval.py (7 providers)  ->  deterministic search policy, no training needed
        |
        v
packed_judge.py (30 items/call, 1-at-a-time, 50k/day cap)  ->  harvest (claim, our-evidence, verdict)
        |
        v
retrain the 4-way verifier on OUR retrieval distribution, not AVeriTeC's gold evidence
```

This is the ClaimCheck lesson applied as a **data engine** rather than as a student: the
research says retrieval is the +14–22 point lever, so the thing worth learning is *our*
retrieval distribution. Cost: **$0**, free lanes carry the volume, and every call goes
through the existing governor.

What it would buy, honestly: AVeriTeC's official baseline gets 0.49 with gold evidence; we
get 0.744. Harvesting our own evidence and retraining attacks the real deployment
distribution — where evidence is retrieved, not handed to us. That is the experiment worth
running next, and it does not need a single gradient through a search policy.

---

## 7. Mamba on all of Wikipedia + educational science — no

| quantity | value |
|---|---|
| tokens in "all of Wikipedia + science web" | ~50–100B |
| small Mamba (130M) training on 50B tokens | ~75–150 days on the B50 |
| free checkpoints of comparable size | available at 6× the token count, today |
| the actual measured bottleneck | **retrieval**, not the backbone (+14–22 pts) |

Spending 3–5 months of the only GPU to reach one-sixth of the tokens of a free checkpoint,
to chase a lever that is not the bottleneck, is the wrong order of operations. **Do not
build it.**

The defensible alternative if a backbone change is ever wanted: **continue-pretraining an
existing small checkpoint on a domain corpus** (days, not months), or swap to
`ctu-aic/deberta-v3-large-AVeriTeC-nli` — zero pretraining FLOPs and it attacks the
Conflicting tail (.214) directly.

---

## 8. What to do next, in order

1. **Ship the controller as-is.** `max_p` calibrated + CP threshold at α=0.10 or 0.15.
   Coverage 44–54%, error 9–11%, promise held. Free, no new model.
2. **Grow the calibration set to ~3k** by k-fold cross-fitting over train instead of a single
   500-row carve. That would (a) recover the 3.6 points lost to the carve and (b) make
   Mondrian viable with 300+ rows per class.
3. **Point the Wikidata channel at FactKG**, where it can actually fire.
4. **Run the harvest loop** (§6) and retrain on our own retrieval distribution.
5. **Quint spike** — express one of our own invariants ("the gate refuses training without a
   CLEAN gate.json") as a spec and model-check it. Candidate deterministic checker for
   protocol-shaped claims.
6. ~~Mamba pretrain~~ — no.

---

## 9. Files

| file | what |
|---|---|
| `tools/verify/uncertainty.py` | temperature scaling, correctness head, CP controller, AURC. Self-test 7/7 |
| `tools/verify/train_verifier_cls.py` | now carves `--calib`, saves checkpoint, dumps `logits.jsonl` |
| `B:\datasets\forge\verifier-av\cls-unc\logits.jsonl` | per-row logits for fit/calib/dev (3566 rows) |
| `B:\datasets\forge\verifier-av\cls-unc\uncertainty.json` | every number above, machine-readable |
| `B:\datasets\forge\verifier-av\cls-unc\model.safetensors` | the retrained verifier (713 MB) |

Two bugs were found and fixed while building this, both of the same family as the earlier
ones: `macro_f1_accepted` was scoring gold labels against themselves (always 1.000), and
`neg_entropy` was reporting positive entropy, giving a "weak" AUC of 0.24 that was actually
a correct score with the wrong sign. Both are the kind of error that produces a plausible
number and is invisible unless someone checks what the number means.
