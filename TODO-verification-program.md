# TODO — verification program (heads, datasets, pipelines)

**Status key:** ✅ done+tested · 🔨 built, not yet wired to real data · ⬜ todo · ⛔ dead end (do not rebuild)
**Program rule:** every item names the test that proves it and the baseline it must beat. No item is
"done" on an unlocked number.

---

## Phase 0 — foundation (✅ complete this session)

| # | item | status | test result |
|---|---|---|---|
| 0.1 | typed contract `logos/schema.py` | ✅ | quantity normalisation; `Warrant` forbids silent abstention |
| 0.2 | exact factor-graph inference `logos/pgm.py` | ✅ | **7/7** — VE ≡ brute force (7.8e-16); `P(B=1\|A=1)=sigmoid(1.7)` to 1e-12; 50 atoms in 20.5 ms |
| 0.3 | Bayesian atoms + PPI++ `logos/decision.py` | ✅ | **9/9** — PPI λ=0 exactly recovers classical; noise→λ 0; oracle cuts sd 0.034→0.003 |
| 0.4 | asymmetric aggregation + conformal `logos/aggregate.py` | ✅ | **16/16** — P(Conflicting) peaks at balance 0.023→0.085→0.000 |
| 0.5 | WordNet/FrameNet lexical layer `logos/lexical.py` | ✅ | **21/21** — transitive hypernymy, homonymy, FrameNet slots |
| 0.6 | log-linear fusion `logos/fusion.py` | ✅ | **11/11** — softmax gate **violates** monotonicity (0.900→0.633); log-linear holds it |
| 0.7 | cost-sensitive abstention `logos/abstain.py` | ✅ | **12/12** — Chow threshold exact; asymmetric costs abstain where max-prob answers |
| 0.8 | Mondrian conformal | ✅ | rare-class coverage 0.784 → 0.994; deviation 0.116 → 0.094 |
| 0.9 | 2-state HMM error locator `logos/locator.py` | ✅ | **8/8** — Viterbi 2.920559 hand-checked; fwd-bwd ≡ brute force 1.3e-15 |
| 0.10 | golden dataset generators `logos/datasets.py` | ✅ | **15/15** — 256 syllogisms, 0 label mismatches; 429 rows emitted |

---

## Phase 1 — the three agreed next steps

### 1.1 ⬜ Measurement harness (5-fold CV + bootstrap CI + ablation flip)
- **Why:** ~60 Conflicting positives in 500 dev rows ⇒ 1σ on that class's F1 ≈ ±0.08. **Any gain
  below ~0.10 on Conflicting is unmeasurable on one split.** Without this, every later item is
  unfalsifiable.
- **Test:** synthetic predictor with known macro-F1 must return it; perfect predictor → CI [1,1];
  label-shuffled → 1/K.
- **Must beat:** it must *shrink* the current single-split headlines into CIs. If it doesn't, it's
  miswired.
- Also includes the **evidence-ablation flip test** (REAL, arXiv:2609.08943): delete the cited span,
  the verdict must change. Behavioural test, not accuracy.

### 1.2 ⬜ `claim_ir` — deterministic spaCy IR
- **Test:** `"1.2 million"` == `"1,200,000"` (exact `Quantity` equality); negation scope excludes the
  entity ("the first man on the moon was **not** American" negates the predicate); `"early 1990s"` →
  interval with `precision=decade`.
- **Must beat:** coverage, not accuracy — target **≥5%** of AVeriTeC claims fully grounded, vs
  **0.07%** today.
- Uses `logos/lexical.py` for synonym/antonym/homonym/stopword expansion in `EntityRuler`, and
  FrameNet frames as typed logical forms.

### 1.3 ⬜ Per-evidence 3-way head (~10k derived pairs)
- **Why:** the missing layer. Gold evidence is worth **+14–22 pts** (arXiv:2608.25934); Conflicting
  is an aggregation failure, and the aggregator is already built (0.4).
- **Data:** ~4 evidence items × 2,566 claims ≈ 10k pairs, derived from AVeriTeC's own Q/A evidence.
- **Test:** 40 forced probes (number swap ⇒ refute, unit swap ⇒ refute, entity swap ⇒ refute,
  paraphrase ⇒ support, related-but-not-answering ⇒ insufficient) at **≥90%** with margin > 2 nats.
- **Must beat:** DeBERTa-v3-large NLI zero-shot on the same pairs.

---

## Phase 2 — dataset programme

| # | item | status | notes |
|---|---|---|---|
| 2.1 | syllogism curriculum (easy→hard) | ✅ | 256 rows, labels derived by the engine, difficulty 1/2/3/5 |
| 2.2 | numeric consistency pairs | ✅ | 6% unit mismatch ⇒ Refuted; **hedged claims never refute** |
| 2.3 | lexical adversarial pairs | ✅ | synonym keeps label, antonym flips (89 syn / 29 ant) |
| 2.4 | per-evidence 3-way pairs | ⬜ | needs 1.3; derive from AVeriTeC Q/A items |
| 2.5 | ternary yes/no/look-it-up set | ⬜ | from 0.7 + calibrated posteriors; **not** a 5th label class |
| 2.6 | HMM token-label set | ⬜ | needs span annotations; generate by evidence-coverage per token |
| 2.7 | unsatisfiable/trap set | ⬜ | fabricated entity, impossible quantity, era mismatch (Wikidata probes) |
| 2.8 | clean the existing corpora with our own tools | ⬜ | cross-fitted cleanlab (already validated: **+4.8 sd** over random thinning) |
| 2.9 | hard-negative mining | ⬜ | preferred over curriculum ordering — see ⛔ below |
| 2.10 | verified-synthesis gate on every generator | ✅ | dedup + decontam + temporal, each with a typed rejection reason |
| 2.11 | **temporal gate everywhere** | 🔨 | AVeriTeC-specific: evidence must predate the claim. Invisible in aggregate accuracy |
| 2.12 | n8n pipeline wrapping each generator | ⬜ | only after the generator's gates reject >5% (a gate that never fires is decoration) |

---

## Phase 3 — heads and combination

| # | item | status | test / baseline |
|---|---|---|---|
| 3.1 | log-linear fusion, fitted weights | 🔨 | next: fit `w ≥ 0` on **OOF** log-probs. Baseline = uniform weights. Must beat it on macro-F1 **and** ECE, else ship uniform |
| 3.2 | BCM baseline | ✅ | `bcm([.9,.1]×2, prior [.6,.4]) = 0.981818` exactly |
| 3.3 | OOF stacker (9 params) | ⬜ | test: 6 identical calibrated experts ⇒ weights equal, Σw=1, output unchanged; a noise expert ⇒ `\|w\|<1e-3` |
| 3.4 | abstain layer wired to costs | 🔨 | general 4×4 cost matrix; Chow is the uniform special case (tested) |
| 3.5 | Mondrian conformal per label | ✅ | must beat marginal on worst-class coverage |
| 3.6 | HMM locator on real features | ⬜ | features: text stats + NLI entailment + LM surprisal + **retrieval coverage per token** |
| 3.7 | semantic-entropy **probe** (1 forward pass) | ⬜ | cheaper than sampling; add as one feature to a 3-param logistic |
| 3.8 | combine: graph first, pool second | ⛔⚠️ | fusion belongs **inside** the factor graph as potentials, not after the heads |
| 3.9 | train heads separately then jointly | ⬜ | joint stage only after each head has a locked, CI-backed score |

---

## Phase 4 — training, if and only if Phase 2 and 3 justify it

| # | item | status |
|---|---|---|
| 4.1 | fine-tune CTU DeBERTa-v3-large on cleaned + golden data | ⬜ (currently 0.808/0.678 on the original split) |
| 4.2 | GEPA prompt tuning for the **generators** (DSPy 3.4.1 present) | ⬜ ≤200 rollouts; GEPA beats GRPO by 6% avg at 35× fewer rollouts |
| 4.3 | domain continue-pretrain | ⬜ hundreds of free GPU-hours makes this feasible (~17h A100 for 130M×10B tokens) |
| 4.4 | pretrain a small Mamba on **verification** data | ⬜ after ≥100k harvested pairs, not before |

---

## ⛔ Dead ends — do not rebuild (each with the reason)

| dead end | reason |
|---|---|
| Markov Logic Networks / PSL | `pracmln` last shipped **2019**; zero applied results 2022–2026. Use the built factor graph |
| t-norm / Łukasiewicz / LTN for calibration | no distribution over possible worlds; truth and belief conflated |
| DeepProbLog / Scallop as a training path | needs gradients through weighted model counting; pointless when the FFI is the required feature |
| **attention / softmax gate for head fusion** | **violates coordinate-wise monotonicity**: measured 0.900 → 0.633 when a third expert joins. Using log-linear instead |
| Deep Ensembles **sharing training data** as "independent" voters | measured 2.22× error-correlation lift; that's the design, not a discovery |
| logistic correctness head on summary features | falsified **3×** in this repo; use PPI++ (`decision.py`) |
| MC-dropout as a posterior | not a posterior. Ensembles first, last-layer Laplace second |
| Dempster–Shafer fusion | arXiv:2608.14617: below-chance confident wrongness on long chains |
| ternary / 1.58-bit **head** | BitNet's gains are at 1B+ params. Our head is ~480 bytes; arXiv:2609.29397 shows the small-scale claim reverses on one LR choice |
| abstain as a 5th **label** | conflates a verdict about the world (NEI) with a decision about cost |
| GRPO on the verdict head | trajectory optimiser for a tabular problem; needs thousands of rollouts |
| backbone shopping | within 5-fold noise at this corpus size (arXiv:2610.17077) |
| decomposition without sub-claim-aligned evidence | SRE "fails to improve and often degrades" (arXiv:2602.10380) |
| curriculum **ordering** for small discriminative models | no 2025–26 evidence; use difficulty-stratified sampling + hard negatives |
| textacy / fastcoref | abandoned 2023 |
| tuning Conflicting gaps < 0.10 | n≈60 positives, 1σ ≈ ±0.08 — that is noise |

---

## Two standing obligations

1. **Our 0.678 macro-F1 is not comparable to the AVeriTeC 2025 leaderboard** (winner 0.332 AVeriTeC
   score; fine-tuned Qwen3-32B F1 0.382). Ours is given **gold evidence**; theirs retrieves. State the
   evidence condition every time.
2. **The evidence-ablation flip test runs in CI.** If deleting the cited span doesn't change the
   verdict, the model is reading its own priors (arXiv:2609.08943) and every number above it is suspect.

---

## Gate rule for going to n8n

A generator ships to n8n only when:
- every gate rejects **> 5%** (dedup, decontam, temporal, solver)
- a 200-row hand audit shows label noise **< 5%**
- retraining on it beats the same-size random sample on **held-out** dev, with CIs
