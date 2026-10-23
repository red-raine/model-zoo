# Capability map: what is a skill, what needs search, what we train, what logic does

**Date:** 2026-10-22
**Built and tested:** `tools/logos/schema.py`, `pgm.py` (7/7), `decision.py` (9/9), `aggregate.py` (16/16)
**Research base:** 2026 sources cited inline; verified-before-use where marked.

---

## 0. Two corrections to the premise, stated up front

### "Have you been using a Bayesian head for logic?"

**No.** What exists in this repo before today: temperature scaling (one parameter, post-hoc),
a **logistic regression** head on 15 features (discriminative, not Bayesian), and cleanlab's
confident counting. Nothing generative, nothing relational, nothing Markov. The two things the
operator asked about are precisely the two I had not tried — I tried the cheap frequentist
workarounds around them three times, and they lost three times.

### "A Markov for logic"

The instinct is right; the tooling is dead. **Markov Logic Networks are the one instantiation
we must not use:**

| check | result |
|---|---|
| only Python MLN on PyPI (`pracmln`) | last release **2019-05-17** |
| `pymln` | 404 |
| Alchemy / Tuffy | unmaintained ~a decade |
| applied MLN/PSL on claim verification, 2022–2026 | **zero results** |
| 2026 SoK over 121 systems (arXiv:2609.23239) | formal machinery "has rarely been applied here" |

Domingos himself moved to *Tensor Logic* (arXiv:2510.12269). But the **2026 survey's own
decision matrix (arXiv:2610.07313, Table 4) is the useful finding**: for NL reasoning the only
*required* feature is a **foreign-function interface** to a neural model; probabilistic
semantics, recursion and differentiability are merely "beneficial", and *"probabilistic
inference remains the scalability bottleneck"*.

**So: not an MLN — a hand-written factor graph with exact inference.** That is MLN semantics
(`log-value = weight × rule satisfied`) in 30 lines of numpy, over the 10–50 atoms we actually
have. Treewidth is tiny because **we design the graph and we design it shallow**; exact
variable elimination lands in the 10⁵-numpy-op range. Choosing loopy BP or Gibbs here would be
choosing an approximation over an exact answer and then reporting its error as uncertainty.

### "A Bayesian for 'this looks fishy'"

Right, but "fishy" is **not one quantity**. It is four, and that is exactly why my 15-feature
head lost: it averaged them and no one component could be blamed.

| quantity | what it answers | engine | validation target |
|---|---|---|---|
| `p_verdict_wrong` | is the argmax right? | calibrated encoder (`max_p`) | selective risk on held-out |
| `p_prior_mismatch` | does the claim look fabricated? | claim-only model + surface probes | NEI-vs-rest F1 |
| `p_evidence_gap` | did retrieval fail? | **deterministic count** | nothing needed — it's a count |
| `p_epistemic` | is the encoder out of its depth? | seed-ensemble disagreement | calibration vs accuracy |

Combined with **noisy-OR, not a mean**: averaging lets three easy components hide one
screaming one. Built and tested in `decision.py`.

---

## 1. The capability map

For each sub-task: which engine, why, and the measured or cited basis.

| sub-task | engine | why | evidence |
|---|---|---|---|
| claim parsing: entities, numbers, dates, negation, modality | **deterministic NLP** (spaCy `EntityRuler` + `DependencyMatcher` + sympy) | only layer that can hand back *which token* carried the negation — the audit artefact | LLM numeric normalisation is unstable across runs; rules are byte-exact |
| entity grounding | **search** (Wikidata) — no model knows every entity | closed-world is false | — |
| numerical / date arithmetic | **symbolic** (sympy; interval algebra) | exact, checkable | pointwise BCE beats listwise at 0.6B/1.7B on CheckThat! 2025/26 Task 3 |
| relation checks over KGs | **search** (SPARQL, 35 relations) | exact | ours settle **0.07%** of AVeriTeC — a *coverage* problem, not a design problem |
| contradiction **between** sources | **span-level trained head + asymmetric aggregation** | conflict is a relation between snippets, invisible to one forward pass | HallDetect (arXiv:2608.05823) beats matched baselines on 3/4 |
| relevance of retrieved evidence | **small trained reranker** | retrieval is the bottleneck | gold evidence = **+14–22 pts** (arXiv:2608.25934) |
| when to abstain | **conformal set size**, not a third classifier | needs a guarantee, not an empirical curve | conservative abstention beats aggressive guessing (arXiv:2602.10380) |
| verifiability **before** retrieval | **deterministic KG probes + small calibrated scorer** | the only genuinely new information | +12–33.8% wF1 from post-hoc calibration (arXiv:2608.30731) |
| multi-hop composition | decomposition **with sub-claim-aligned evidence only** | SRE (repeated claim-level evidence) degrades | arXiv:2602.10380 |
| temporal reasoning | **interval algebra** | set operations, not learning | — |
| Conflicting (our F1 .362) | **aggregation**, not perception | it is a composition failure | see §3 |
| 4-way verdict | **fine-tuned encoder** — already done | at the noise floor | backbone shopping is within 5-fold noise (arXiv:2610.17077) |

### What we can realistically TRAIN (and what we must not)

| train? | sub-task | minimum data | basis |
|---|---|---|---|
| **yes, top priority** | per-evidence 3-way head | ~5–10k pairs (derivable: ~4 evidence items × 2.5k claims ≈ 10k) | MiniCheck: 770M reaches GPT-4-level at 400× lower cost (arXiv:2404.10774) |
| yes | evidence relevance reranker | ~5k pairs | +14–22 pts lever |
| yes | verifiability / check-worthiness | ~1–3k + post-hoc calibration | arXiv:2608.30731 |
| **no** | entity/number/date extraction | — | rules + zero-shot GLiNER |
| **no** | relation over KG | — | SPARQL |
| **no** | multi-hop reasoning | — | deterministic decomposition |
| **no** | temporal arithmetic | — | interval algebra |
| **no** | spawn relation extraction in spaCy | — | calOn 0.6B leads 40 RE benchmarks (arXiv:2610.17113) |

---

## 2. What I built this session, and what its tests prove

### `logos/schema.py` — the typed contract
Pydantic models: `ClaimIR`, `Quantity`, `TimeInterval`, `EntityMention`, `NegationScope`,
`EvidenceVerdict` (span-level), `ChannelResult`, `Belief` (four named atoms), `Warrant`.

Two design choices carry weight:
- **`Split.TRANSFER_CONTAMINATED` is a first-class enum value.** The CTU checkpoint scored
  **0.9260** on our train carve and **0.8080** on dev; calibrating on the carve made the
  controller promise 15% error and deliver 18.6%. That mistake now requires typing its own name.
- **`Warrant` refuses to be silent.** A model validator forbids an abstention with no recorded
  reason. The 2026 SoK's central criticism is that almost no system emits a warrant at all.

### `logos/pgm.py` — exact inference (7/7)
- `P(B=1|A=1) = sigmoid(w) = 0.845534734916` — exact to 1e-12 against the closed form.
- **VE == brute force on 12 random graphs, max error 7.8e-16.**
- 50-atom chain exact in 20.5 ms.
- Hard constraints produce no NaN.
- `explain()` emits per-factor warrants with the marginal shift if that factor is removed.

**One real bug caught here**, and it is the session's recurring theme: `broadcast` used
`reshape` without a `transpose`, so when two variables shared a cardinality but sat in a
different order, the table was silently mismapped. **My 2×2 all-binary test passed because
every axis had the same size — any permutation was invisible.** It only appeared when the
random test used mixed cardinalities (2 and 3), which is the situation every real label set is
in. A regression test for permuted 2/3/4 cardinalities is now locked in.

### `logos/decision.py` — the Bayesian half (9/9)
- Beta posterior mean matches closed form; credible interval brackets it; shrinkage pulls a
  1-of-1 source toward the population mean.
- **PPI++ with λ=0 recovers the classical estimate exactly** (0.552500000000). This is the
  no-loss property the logistic head never had.
- **Noise auxiliary → λ = 0.0000**, estimate unchanged. **Oracle auxiliary → across-seed sd
  0.0339 → 0.0033.** So weak signals can be added safely; they tune themselves off.
- Noisy-OR keeps one loud signal loud (`0.9595` vs mean `0.383`).

### `logos/aggregate.py` — aggregation + conformal (16/16)
- Forced cases all correct: 5 supports → Supported; 2/2 → Conflicting; empty → abstain *with a
  recorded reason*; a settled sympy channel overrides two supports.
- **Weak support cannot manufacture a conflict** against a strong refute (asymmetry).
- **P(Conflicting) peaks at evidence balance: 0.023 → 0.085 → 0.000** as support grows against
  one refute.
- APS conformal: held-out coverage 0.9623 ≥ 0.90; efficiency tracks model quality (set size
  3.03 weak → 2.05 strong); near-perfect model yields singletons.

**Two more real bugs, both caught by tests:**
1. Subtracting `v.mean()` "to normalise" made a supporting snippet a **negative** contribution
   to the Conflicting logit — support *lowered* conflict probability.
2. Implementing APS as "accumulate until mass **≥** q" instead of "include ranks with cumulative
   mass **≤** q". Symptom: coverage 1.000 with set size 2.65 on a model that should give
   singletons — **over-coverage looks like safety and is actually lost efficiency.**

And one test that was wrong rather than the code: I asserted P(Conflicting) was monotone in
support. It is not, and it should not be — **5-vs-1 is genuinely less conflicting than 1-vs-1.**
That failed test is what revealed that conflict is a **pairwise interaction**
(`min(support_mass, refute_mass)`), not an accumulation. Which is the whole argument for having
a factor graph at all: second-order structure is not expressible as a sum over snippets.

---

## 3. Why this matters for the `.362` Conflicting class

Conflict is not a property of any single (claim, evidence) pair. It is a property of a **set**.
A claim-level encoder reading `claim [SEP] ev1 | ev2 | ev3` compresses "ev1 supports, ev2
refutes" into one hidden state and is then asked to output a label meaning "these contradict" —
reporting the shape of an input it was never given separately.

That is a composition failure, and it is why the fix is aggregation, not a loss re-weight. With
the interaction term the mechanism is now explicit and measurable: conflict mass peaks exactly
at balance, and collapses to 0.000 when one side dominates.

**Honest size of the prize:** ~60 Conflicting positives in 500 dev rows ⇒ 1σ on that class's F1
is ≈ ±0.08, so a 95% CI spans roughly 0.19–0.53. **Any improvement below ~0.10 on Conflicting is
unmeasurable on this split.** The aggregator must be judged on 5-fold CV with CIs, not on one
dev split — otherwise I will report noise as progress.

---

## 4. The uncomfortable fact I have to keep stating

**Our 0.678 macro-F1 is not comparable to the AVeriTeC 2025 leaderboard.**

| system | AVeriTeC score | note |
|---|---|---|
| winner (CTU AIC) | 0.332 | 53 s/claim |
| fine-tuned Qwen3-32B veracity head | F1 0.382 / ACC 0.692 | does not fit 16 GB at 4-bit |
| **ours** | **macro-F1 0.678** | **gold evidence given, not retrieved** |

Different evidence condition. Not a win — a different task. Two guard tests follow from this and
both are cheap:

1. **Claim-only ablation** (already run: 0.574, below the 0.610 majority — so we are not purely
   prior-driven).
2. **Evidence-ablation flip test** (REAL, arXiv:2609.08943): delete the cited span and the
   verdict must change. If it doesn't, the model is reading its own priors, not the evidence.
   This is a *behavioural* test, not an accuracy test, and it belongs in CI.

---

## 5. Build order from here

| # | component | test | baseline to beat |
|---|---|---|---|
| 1 | **Measurement harness**: 5-fold CV + bootstrap CIs + evidence-ablation flip rate | synthetic predictor with known macro-F1 must return it; perfect predictor → CI [1,1]; shuffled → 1/K | our current single-split headlines — it must shrink them into CIs |
| 2 | **`claim_ir`**: spaCy EntityRuler + DependencyMatcher → typed IR | "1.2 million" == "1,200,000"; negation scope excludes the entity; "early 1990s" → interval | coverage: target ≥5% of claims fully grounded (vs 0.07% today) |
| 3 | **Per-evidence 3-way head** (~10k derived pairs) | 40 forced probes (number swap → refute, paraphrase → support) at ≥90%; **ablation must flip support→insufficient** | DeBERTa-v3-large NLI zero-shot on the same pairs |
| 4 | **Asymmetric aggregator** wired to a 4-way verdict | **built** — add the balance-peak test to CI | calibrated `max_p`, judged on Conflicting F1 **with CI** |
| 5 | **PPI++ decision layer** | **built** — λ=0 identity, noise→0, oracle→variance collapse | free `max_p` (ECE 0.084 at 81.7% coverage / 10.2% error). Tie acceptable; loss means bug |
| 6 | **Conformal abstention** | **built** — coverage guarantee + efficiency tracking | the temperature-scaled risk-coverage curve |
| 7 | **Factor-graph warrant layer** | **built** — VE ≡ brute force, sigmoid identity, 50 atoms < 50 ms | `max_p` on **ECE**, and it must produce a warrant. If accuracy drops and ECE doesn't improve, ship it as an explanation artefact and say so |
| 8 | **Verifiability pre-screen**: Wikidata existence probes + quantity plausibility + era consistency | injected "Albert Einstien", "14,000 km bridge", "Caesar's 1990 tweet" must all flag; **a real entity with a missing ID must return `unknown`, never `no`** | NEI-vs-rest F1 from the claim-level model |

### Dead ends — do not rebuild

| dead end | reason |
|---|---|
| MLN / PSL | `pracmln` 2019; zero applied results 2022–2026 |
| LTN / t-norm (Łukasiewicz, product, Gödel) for calibration | no distribution over possible worlds; "degrees of truth and belief are conflated" |
| DeepProbLog / Scallop as a training path | needs gradients through weighted model counting; pointless when the FFI is the required feature |
| logistic correctness head on summary features | falsified 3× in this repo; use PPI++ |
| MC-dropout as a posterior | not a posterior. Ensembles first, last-layer Laplace second |
| backbone shopping | within 5-fold noise at this corpus size (arXiv:2610.17077) |
| decomposition without sub-claim-aligned evidence | SRE "fails to improve and often degrades" (arXiv:2602.10380) |
| textacy, fastcoref | abandoned 2023 |
| tuning against Conflicting gaps < 0.10 | n≈60 positives, 1σ ≈ ±0.08 — that is noise |

---

## 6. Files

| file | self-test |
|---|---|
| `tools/logos/schema.py` | smoke: quantity normalisation, noisy-OR |
| `tools/logos/pgm.py` | **7/7** — VE ≡ brute force (7.8e-16), sigmoid identity, 50-atom chain, hard constraints, monotone support, warrants |
| `tools/logos/decision.py` | **9/9** — Beta conjugacy, PPI λ=0 identity, noise→λ 0, oracle→10× variance cut, noisy-OR, evidence gap, epistemic |
| `tools/logos/aggregate.py` | **16/16** — forced aggregation cases, asymmetry, balance peak, APS coverage + efficiency |
