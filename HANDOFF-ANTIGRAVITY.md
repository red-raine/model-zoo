# HANDOFF — Antigravity joins the `logos` engines

**Date:** 2026-10-28 · **From:** opencode · **To:** Antigravity (minion)
**Repo:** `http://192.168.8.217:3005/opencode/model-zoo` (Gitea) · mirror on GitHub `red-raine/model-zoo`
**Verify everything in one command:** `python logos/run_all.py --fast`

---

## 1. What this is, in one paragraph

`logos/` is a verification and dataset-cleaning SDK. It takes a row of data and decides whether to **KEEP**
it, **REPAIR** it, **ESCALATE** it to something more expensive, or **DROP** it — and it must be able to
*prove* whichever answer it gives. 42 modules, all self-tested. The rule the whole thing is built around:

> **Validity ≠ truth ≠ confidence ≠ calibrated probability.** A valid syllogism can be false. A confident
> sentence can be wrong. A probability from one protocol is not comparable to a probability from another.
> Every layer declares which of these it is allowed to claim (`logos/honesty.py`), and its self-test fails if
> that declaration drifts from the code.

---

## 2. Start here, in this order

| # | read | why |
|---|---|---|
| 1 | `logos/README.md` | the layer model (L0–L5) and the current verified inventory |
| 2 | `logos/honesty.py` | the capability manifest — what each layer proves and **cannot** prove |
| 3 | `drift-audit-what-is-verified-and-what-is-not.md` | what is actually verified vs inherited vs overclaimed |
| 4 | `objective-correction-ore-refining-not-theorem-proving.md` | why precision-first, and why 99.9% is unmeasurable at our corpus size |
| 5 | `logos/row_eval.py` | the per-row pipeline you will be extending |
| 6 | `TODO` sections in `openBitmamba-spine-todo.md` | the live work queue (v2–v5 sections) |

---

## 3. The architecture you are joining

```
row
 ├─ context_pack.gather()        context BEFORE judging: signature, doctests, ontology key,
 │                               RDF graph (rdflib), lemmatised terms
 ├─ truth_kinds.classify()       12 kinds of truth; decidable vs evidence-requiring
 ├─ row_eval.execute()
 │    subjects → checklist → DAG → verdict
 │    • cheap+high-tier first, short-circuit on REFUTATION
 │    • UNKNOWN propagates to children
 │    • EVERY finding must cite its artifact or it is rejected
 └─ truth_store.serve()          reuse a validated truth, but only if 2 INDEPENDENT validators agreed
```

Plus the math layer (`verify/math_core.py`, 8 decidable kinds), the logic layer
(`logic/syllogism.py`, `logic/smt.py`), and the consistency gate (`logic/smt.py`, Z3).

---

## 4. Where you should work — two engines, not one

### A. Logic / reasoning engine
- `logos/logic/syllogism.py` — 24/24 valid forms over 256, verified
- `logos/logic/propositional.py` — 35 inference + 16 replacement rules
- `logos/logic/badframes.py` — 8 formal defects, 5 LLM-specific frames
- `logos/logic/lean_bridge.py` — real Lean 4, `sorry`-free enforced
- `logos/logic/smt.py` — Z3 consistency: **INCONSISTENT is sound, CONSISTENT is fragment-relative**
- `logos/logic/quantity.py`, `sorites.py`, `channel.py`, `argumentation.py`

**Live question:** `smt.py` has 11 predicate semantics. Extending the predicate table (cause, part_of,
located_in, member_of) is high value and low risk. **Every new predicate needs a test with a hand-built
contradiction and a hand-built consistent case.**

### B. Dataset cleaning engine
- `logos/triage.py` — 4 routers (structural/semantic/truth/calibration), 4 dispositions, scarcity policy
- `logos/refinery.py` — asymmetric-cost routing + Clopper-Pearson audit arithmetic
- `logos/audit.py` — stratified precision audit, held-out sheets
- `logos/code_refinery.py`, `code_exec.py`, `ast_repair.py` — the code path
- `logos/ontology_bundle.py` — **OKF-native domain jargon** (new; see §6)

**Live question:** `row_eval.py` has only FOUR checks. Adding checks is the main throughput work. Each new
check must declare `category`, `tier` (high/medium/low), `cost`, `subjects` and — non-negotiable — **a
citation for every finding**.

---

## 5. Conventions that are not optional

1. **Every module is a program with `--self-test`.** `python logos/X.py --self-test` must print
   `SELF TEST: ALL PASS`. Then add it to `logos/run_all.py`.
2. **No number without its set.** Say "on 482 deduped AVeriTeC dev rows", not "93%". A number without its
   set is how this project produced two wrong claims.
3. **Circular evidence must be labelled inline** — in the code AND the output.
4. **Fail closed.** Unknown predicate → `UNKNOWN`, never `VERIFIED`. A dead archive counted as a backup is
   worse than no backup.
5. **Never execute untrusted code on the host** (L008). `code_exec.run_snippet()` defaults to `REFUSED`;
   process isolation is **not** a security boundary.
6. **Only Squid when needed.** `egress.py` learns per-source which path works; the proxy *breaks* wayback.
7. **Keys:** never the master key. Per-app keys via `tools/tm/key_store.py`. If a model 404/403/402s, STOP
   and request it — never silently switch keys.
8. **`--no-cache` for Lean at least once** before quoting a Lean result. A cache read as a proof is exactly
   how a claim survives being wrong.

---

## 6. The newest piece: OKF-native domain jargon (`ontology_bundle.py`)

The gap: `context_pack.normalize_terms()` lemmatises with **WordNet**, which is general English. Measured on
15 terms from this project's own documents:

```
known to WordNet : 6/15 (40%)
```

**60% of our own jargon is invisible to the dictionary** — and "known" is not "correct": WordNet's `bridge`
sense says nothing about a cantilever span. Worse, it will happily return an English sense that is wrong for
the domain, which is the same failure as an identifier attached to the wrong thing.

The fix is not a better lemmatiser, it is a **per-domain ontology**, and OKF gives the right shape for free.
Our `tools/okf_index.py` already validates the profile (required `type/title/status`; status ∈
`current|stale|archived|draft`).

**`draft` maps exactly onto "proposed but not curated"** — the anti-poisoning gate expressed in the format:

| event | status | served? |
|---|---|---|
| corpus uses a term nobody defined | `draft` | **no** — a proposal is not ground truth |
| a curator writes the definition | `current` | yes |
| the definition goes wrong | `stale` | **no** |

Measured: cold bundle 0/15 served → after curation **15/15**, with draft and stale terms correctly refused.

**This is where you can move fastest.** Each term is one markdown file with frontmatter, so a bundle is
greppable, diffable, and reviewable in a pull request like any other source file. Adding 100 engineering or
biomedical terms is a real contribution that costs no compute.

---

## 7. What NOT to do

- **Do not rebuild solved work.** Categorical syllogism validity is done. `2+2=4` is done. Adding Z3 to
  either is a regression in clarity, not an improvement.
- **Do not add a gate before an artifact exists.** The project's main failure mode has been building
  measurement infrastructure instead of product. See the drift audit: 42 modules, no trained model, no
  cleaned corpus at scale, no certified precision number.
- **Do not quote a dev number as if it were held out.** AVeriTeC dev fitted `T`, `tau`, weight modes, the
  review band and the strata edges. **We currently have no clean held-out set.** That is the largest
  structural defect in the pipeline and it is named, not hidden.
- **Do not trust `row_eval`'s four checks as a validation suite.** It is a scaffold.

---

## 8. Open questions we have not answered

1. **Shard valuation.** The plan is one MoE expert per shard, fine-tuned separately so a home lab can improve
   one shard at a time. The valuation method chosen is **canary-per-shard + leave-one-shard-out on a common
   held-out set** — because influence functions fail the budget test. **Unbuilt.**
2. **Output validation for code.** `code_behaviour` is classified but we validate that code *ran*, not that
   it returned the right thing. Half the signal is missing.
3. **A second validator per claim family.** Only math has two independent implementations (sympy vs
   `math_verify`), which is why math is the only family whose poisoning risk is measured.
4. **Ontologies fed to a decision model.** Argued to help; never measured.

---

## 9. How to verify I have not lied to you

```bash
python logos/run_all.py --fast          # 42 modules
python logos/logic/lean_verify_all.py --no-cache   # ~470s of real Lean, 51/51
python logos/refinery.py --run          # the audit arithmetic
python logos/ontology_bundle.py --measure
python logos/truth_measure.py           # validator agreement + cache worth
```

If any of those disagree with what a document claims, **the document is wrong and the code is right.**
That is the convention this repo runs on.


---

## Update 2026-10-28 -- live encoders, evidence gate, and the first end-to-end pipeline

### What is new
| module | what it does |
|---|---|
| `logos/rerank_client.py` | client for the two live cross-encoders + `evidence_sufficient()` gate |
| `logos/pipeline.py` | the single row -> KEEP/REPAIR/ESCALATE/DROP path that was missing |
| `logos/second_source.py` | four keyless public vocabularies for term corroboration |

### Live infrastructure, probed
```
:8084  TEI            BAAI/bge-reranker-v2-m3   max_input_length 8192
:8086  argus-reranker onnx, /rerank takes {"query","texts"}, returns LOGITS
:6333  qdrant         1.19.2   lab-research 1009 | lair-code 18852 | mc-wiki 87  (all 384-dim cosine)
```

### Findings worth carrying forward
1. **The two rerankers are one model.** TEI publishes probabilities, argus publishes logits, and after
   sigmoid they differ by 0.000024. Two services, two runtimes, ONE witness.
2. **A relevance score can only demote.** It ranks support above contradiction 5/5, which a token
   baseline cannot do at all -- but it *discards* contradictions rather than flagging them, so a single
   threshold would make REFUTED unreachable. The gate therefore returns its low band for validation.
3. **REPAIR was unreachable** in `row_eval` -- defined, never assigned. Now produced by a numeric repair
   that must re-pass the validator that refuted the original.
4. **My first repair produced a tautology** (`pi = 3.0` -> `pi = pi`, which verifies and destroys the
   claim). Verifying is necessary, not sufficient.
5. **Correction to an earlier number:** "38/60 orphan modules" was an import-matcher artifact. Re-measured:
   31 orphans, 25 are entry points, 6 are `__init__.py` -- dead library code is zero. The gap was never
   inventory, it was composition.
6. **Second-source corroboration is untrustworthy for our own jargon:** `macro-F1` matched "Macrobiotus
   sp. F104", `P585` matched a protein. 4 hits on our jargon, 0 of them relevant. Science vocabulary: 5/5
   corroborated and relevant. Counting works for public vocabulary, not for domain strings.

### Honest limits
- 16 constructed rows, labels written by me. The 16/16 proves the composition runs, not that it is accurate.
- **No clean held-out set.** Still the largest defect.
- Prose rows all reach ESCALATE because no check applies to "the Eiffel Tower is 330 metres tall". That
  abstain pile is the real measure of coverage, and it has not been sized on a real corpus yet.
- `ast_repair.repair_with_oracle` needs `allow_exec=True`, which L008 forbids by default, so it remains
  available but unused. The only active repair is numeric.


---

## Update 2026-10-28 (b) -- ONNX on the host, and an exact contradiction decider

### New modules
| module | what it does |
|---|---|
| `logos/onnx_rerank.py` | runs the int8 ONNX cross-encoder natively; no Docker, no HTTP |
| `logos/numeric_conflict.py` | decides a numeric contradiction exactly, by naming both values |

### The ONNX model was already on disk
`B:/ai-stack/models/argus-reranker/model_int8.onnx` -- 569.7 MB int8, `XLMRobertaForSequenceClassification`
24 layers x 1024 hidden, which identifies it as **bge-reranker-v2-m3** by architecture and parameter count
alone. `onnxruntime 1.20.1` is on the host, so it runs with no container:
**native == service on 5 pairs, max diff 0.000289.**
Native is **88.5 ms/pair vs 111.3 ms/pair (1.26x)** -- the model dominates and the HTTP hop is ~23 ms, so
the benefit is that a batch needs no container running, not speed.

### The embedder, and a trap worth knowing
`docker inspect tei` shows `--model-id BAAI/bge-small-en-v1.5`. That is what built the three 384-dim Qdrant
collections. `all-MiniLM-L6-v2` is also 384-dim and cached on this host, so using it against those
collections would return **confidently ranked nonsense** -- cosine is defined between any two vectors of
equal length, and it says nothing about whether they share a space. Same class of error as reading a logit
as a probability.

### Contradictions decide instead of vanishing
The measured gap was that the cross-encoder *discards* contradicting passages (4/5 below threshold, merged
with off-topic). `numeric_conflict.py` closes it with an exact rule:
```
contradicting passages flagged : 4/5   (named, with both values)
supporting or off-topic flagged: 0/10
missed: semantic contradiction with no numbers -- stays ESCALATE
```
Wired into the pipeline, this makes **DROP reachable from evidence**, which it was not before:
```
contradicting number               -> DROP      numeric_conflict
supporting / off-topic passage     -> ESCALATE
semantic contradiction, no numbers -> ESCALATE
```

### On training a reranker in 10 minutes -- priced rather than dismissed
The free labels an oracle can generate (mutate a number, flip a negation) teach a model to reproduce what
`sympy` already does at zero error across eight decidable kinds. The labels that would add capability are
real prose contradictions, which need human judgement, and we have none. Training now would also mean
evaluating on constructed data -- the circular-evidence defect this project already names as its biggest
weakness. **Compute is not the binding constraint; labels are.** A model becomes worth training when there
is judged data to train it on.

### Bugs found by measuring, in code written minutes earlier
1. A greedy optional group before a constrained one matched "330 metres tall" as unit="tall", so the
   contradiction was never extracted -- and the regex *succeeded*, so nothing backtracked.
2. Digit-only extraction missed "two premises" vs "three premises". Prose writes counts as words.
3. I wired the decider to the low band, assuming that is where contradictions land. **Measured: "1.609344
   km" vs "1.5 km" scores 0.867, above threshold**, because contradicting a claim is a way of being
   on-topic. It arrives in `sufficient`. The decider now runs on all candidates.
4. `cand["text"]` then raised KeyError (`all` entries carry `index`, only `low_band` carries `text`) and a
   `break` on exception silently disabled the entire refutation path.

### Honest limits
- 5 triples, hand-written, and the one missed contradiction is the semantic kind. 4/5 is illustrative of the
  mechanism, not a rate over a corpus.
- `onnx_rerank` loads 569.7 MB per process; it is not free in a short-lived script.
- **Retrieval is still missing.** The 18,852-point `lair-code` collection needs `bge-small-en-v1.5` on the
  host, which is not yet downloaded. Lexical candidate selection plus cross-encoder rerank is what exists.
- **No clean held-out set.** Still the largest defect.


---

## Update 2026-10-28 (c) -- encoder chosen by measurement: 19M, not 568M

### Hardware, measured
`i9-9980HK, 8c/16t, torch capability = AVX2 (no AVX-512/VNNI), 64 GB, no discrete GPU.`
Throughput from the built `llama-bench`:
```
LFM2.5-350M Q4  tg 44.8 tok/s
LFM2.5-2.6B Q4  tg  8.5 tok/s
LFM2.5-1.2B Q4  tg  2.5 tok/s   <- anomalous, slower than the 2.6B; under investigation
```
An LLM decision at ~450 tokens costs **~75 s/row** against 88.5 ms for a reranker: **~850x**, because it is
autoregressive decoding versus one forward pass.

### New module: `logos/encoder_bench.py`
30 pairs, 14 of 20 negatives HARD (same subject, no answer), token-F1 baseline 0.882:
```
19M  ms-marco-MiniLM-L-4-v2   AUC 0.985    10.2 ms
568M bge-reranker-v2-m3 int8  AUC 0.965   138.1 ms
568M bge-reranker-v2-m3 fp32  AUC 0.960  2776.1 ms
```
- the 568M is **4 of 200 comparisons** behind the 19M -- indistinguishable at this size
- **int8-vs-fp32 control**: int8 is 0.005 AHEAD on the same weights, so this is not an export artefact
- 10,000 rows: **1.7 min** vs 23.0 min
**Verdict: a cascade.** 19M for the bulk, 568M where a pair exceeds 512 tokens (MiniLM truncates) or the
small model is ambiguous. Not a preference -- a measurement.

### BitNet, priced from the repo
- embedding speedups **1.32x-1.74x** (270M), **1.42x-2.28x** (0.6B) over F16 prefill, 8 threads
- the headline **6.17x is against full precision**; we already run Q4_K_M, so it is not our baseline
- I2_S runs on x86 **AVX2** (VibeASR cites it, RTF<1)
- MTEB v2 66.26 vs bf16 twin 66.5; 67.49 vs 69.0 -- near-lossless, costs 0.24-1.51 points
- the HF card's llama.cpp example uses **`:BF16`**, so the dequantised path is what actually runs there
**Skip ternary for the encoder** -- it is already 10 ms and 64 GB makes bandwidth a non-issue. Measure it for
the repair LLM, where 8.5 tok/s is the real bottleneck.

### On a "decision model" (Reddit test)
Every route was blocked (403, captcha, Redlib 429/410/bot_blocked), so this rests on search snippets: 56
triage tasks, Qwen3.5-9B beating 14 "decision models", "every LLM that beat them reasons first", 96%.
Honest reading: 56 tasks is small, and *reasoning first* is exactly what costs 75 s/row here. A quality
result with no cost axis is not actionable on this machine. And a decision model is still a classifier
scored by macro-F1 + AUC against **labels we do not have**.

### CoT rewriting as self-repair -- right idea, named trap
Rewriting traces with a model injects fluent rationalisation into a training set. We already hit this at
small scale: the first numeric repair rewrote `pi = 3.0` to `pi = pi`, which verified and deleted the defect.
**Invariant: keep a rewrite only if it re-passes the same validators with the verdict unchanged.** The
mechanism is in `pipeline._numeric_repair`; extend it to prose with the LLM as proposer, validators as gate.
ThinkingCap compresses traces (10K -> 1.3K chars); it does not verify logic.

### Honest limits
- 30 constructed pairs; a 0.020 AUC gap is 4 comparisons. No claim of accuracy.
- ms-marco-MiniLM is trained for MS MARCO ranking and truncates at 512 tokens -- it cannot see our long rows.
- the 1.2B throughput anomaly is unexplained and may indicate a bad build rather than a bad model.
- **No held-out labelled set.** Every model decision above is contingent on that gap being fixed.


---

## Update 2026-10-28 (d) -- local LLM as decider: measured, degenerate

### The third-party evidence, read for what it is
modeltested.com's thread: 56 situations / 112 questions, local LLMs beating 14 commercial "decision models"
(Qwen3.5-9B 97.3%, Granite 4.2 8B 97.4% on hard cases; Jev 78.9, Clef 73.7, Clef Flash 68.4).
Two things matter for us:
- their test **states the rules in the prompt** and grades multiple choice, so it measures policy
  APPLICATION, not fact verification -- which is half of what our dispositions need
- the decision models fail exactly on "reading a rule carefully and doing a little math", which is our
  numeric-contradiction case and is already decided exactly by `numeric_conflict` + `math_core`
**Their hard-case column independently endorses the architecture we arrived at by measurement.**

### New module: `logos/llm_decider.py`
Runs the same ten rows through a persistent `llama-server`, temperature 0, and compares against the pipeline:
```
LLM accuracy      3/10
majority baseline 3/10  (always answer 'KEEP')
pipeline accuracy 10/10
distinct labels the LLM produced: 1 ['KEEP']
LLM median        2,178 ms/row      encoder 10.2 ms/row  (214x)
```
**One label for every row.** Its 3/10 is the class prior, not skill -- a two-line script printing `KEEP`
scores identically. The first run reported 3/6 = 0.50 on six rows containing three KEEPs, which is correct
arithmetic pointing at nothing. **A majority-class baseline now prints beside every accuracy figure, and
"distinct labels produced" is a first-class output.**

### What it does not prove
Collapsing to the majority class is precisely what supervised fine-tuning repairs, so this is not evidence
that fine-tuning would fail. What it shows: an off-the-shelf 2.6B carries no information about our rows,
reaches that state by answering INSTANTLY instead of reasoning, and the instant path still costs
**6 hours per 10,000 rows** to emit a constant.

### Correction to an earlier estimate
I priced an LLM decision at ~75 s/row from a reasoning budget (~208 h/10k). Measured it is **2.2 s**, because
it does not reason. So 850x was the price of reasoning and 214x is the price of not reasoning -- and they
trade exactly the way the thread documents: cheap decision models lose on hard cases.

### Honest limits
- 10 rows, labels written by me.
- a 2.6B is not their 9B. A 9B would likely score higher; on this CPU it would run near 2.5 tok/s.
- if this path is pursued, **any evaluation without the majority-class baseline will report ~30% for a
  constant and look like progress.**
- NO HELD-OUT SET. The model collapses for want of the labels that would fix it.


---

## Update 2026-10-28 (e) -- rounding is not error, repair is not cleaning, and coverage was 0% not 37.5%

### Reported precision was wrong in the direction that matters
`pi = 3.14` and `e = 2.718` were being REFUTED. Both are CORRECT ROUNDINGS. The validator compared against an
ABSOLUTE tolerance of 1e-4 -- which is also not scale-free. Replaced with stated-precision semantics: a value
with N significant digits is correct iff it equals the true value rounded to N s.f. No tolerance.
```
pi = 3   KEEP (correct 1 s.f., coarse)     pi = 3.0  DROP (2 s.f. is 3.1)
pi = 3.1 KEEP (correct 2 s.f.)             pi = 3.14 KEEP (correct 3 s.f.)    pi = 3.2 DROP
```
**Coarse is not false.** `precision_sufficient(claim, required_rel_error)` answers adequacy separately: pi = 3
is a correct rounding AND a 4.5% error, useless for a 500 px circle where pi = 3.14 is 0.25 px out.
Also fixed: two code paths that gave opposite verdicts on `pi = 3.1416`, and `pi = 22/7` which was
unrecognisable because only a decimal literal parsed.

### Repair default changed to DROP
I had `2 + 2 = 5` -> REPAIR because the correction is provable. The proof is irrelevant: a repair emits a row
the source never contained -- the source said 5, we emit 4, and we cannot know if the author meant 4. For a
training corpus that is authoring data. `repair_policy` now defaults to "drop"; repair is opt-in and altered
rows carry `altered=True`.

### New module: `logos/corpus_sizing.py` -- 200 real curated AVeriTeC claims
```
first run   : 10/200 false drops (5.0%)
after fixes :  0/200 false drops
coverage    : KEEP 0 | REPAIR 0 | DROP 0 | ESCALATE 200  = 0.0%
```
numeric_conflict was comparing two DATES ("1950 vs 2020"), reading "75,000 flu deaths last year in 2019" as
75,000 YEARS (the unit four words away), and matching "Forty percent" to 5,100. Fixed by making units
ADJACENT, excluding years, giving word-form numbers 1 s.f., and honouring rounding. Then the architectural
rule: **evidence may only refute a row that a validator can parse** -- a checkable `1 mile = 1.609344 km` is
refuted by contradicting evidence, a prose claim gets a flag and goes to review.

**And then coverage collapsed from 37.5% to 0%, because 37.5% was an illusion.** All 75 KEEPs came from the
`structure` check -- well-formedness -- including gold-REFUTED claims such as "New Zealand's new Food Bill
bans gardening" and "All USA Ballots Arriving After Election Day Will Be Thrown Out". KEEP was being issued
for shape and read as verification. For prose, structure is now a PRECONDITION, never a conclusion.

The corpus answers 93.5% of its own claims. So the gap is 93.5 points on this population and it is a
FACT-VERIFICATION gap, not a routing one. Arithmetic, units, constants and code rows remain decidable and are
a different population -- 0% must not be quoted as "the pipeline does nothing".

### A swallowed error that looked non-deterministic
`numeric_conflict` never had the `sys.path` bootstrap the other modules have, so its lazy
`logos.verify.math_core` import failed under `python file.py` and `except Exception: pass` discarded it: the
rounding check silently did not run, and one assertion passed on import and failed as a script. The failure is
now recorded in `rounding_check_errors` and asserted, because an unperformed test must not look like a
passing one.

### Cascade wired in
`evidence_sufficient` defaulted to the float32 TEI backend at 2,776 ms/pair; now the measured winner, the 19M
cross-encoder at 10.2 ms, with `truncated` flagged for >512-token pairs so the 568M arm takes those.

### My own errors, caught by the oracle three times
pi = 3.1 is a correct 2 s.f. rounding (I said wrong); 28 is the correct rounding of 27.63 (I said 27); and I
wrote prose rows for a rule requiring a checkable row. Testing against the oracle rather than intuition is
what surfaced each one.

### Honest limits
200 of 500 dev rows, and dev was already used for threshold fitting elsewhere -- not held out.
numeric_conflict still fires on 3.5% of real pairs; on prose it can now only raise a flag, never a drop.
**No held-out labelled set.** Unchanged, and now the single thing between these measurements and an accuracy
claim.

---

### TIERED REPAIR: COMPUTED CORRECTIONS ARE WORTH KEEPING (2026-10-28)

#### The objection that changed the design
I had made `2 + 2 = 5` a DROP, arguing a repair fabricates a row the source never contained. The correction
is right: the danger was never the CORRECTION, it was an UNMARKED correction rewriting provenance. Flag it and
basic arithmetic is worth keeping -- `2 + 2 = 5` has exactly one right answer and a machine computes it.

#### The tier boundary is NOT `basic maths vs calculus`
Measured: `d/dx x^2 = 2` is REFUTED and the engine already returns the true value `2*x`, so calculus is
computed just as exactly as arithmetic. And `a^2 + b^2 = c^2` is REFUTED as an IDENTITY -- correctly, since
it holds only GIVEN a right triangle: the missing piece is a PREMISE, and adding premises is authoring.
The line that matters is COMPUTE vs CHECK:
`
TIER 1  computed, one side authoritative by construction  -> repair + math_fix flag
TIER 2  check-only, so repairing means guessing which side -> drop
TIER 3  no oracle                                        -> drop
2 + 2 = 5        tier 1 -> "2 + 2 = 4"
pi = 3.2         tier 1 -> "pi = 3.1"
pi = 3.0         tier 1 -> "pi = 3.1"    <- this row used to be DROPped
1 mile = 1.5 km  tier 2 -> DROP           both sides are claims
a^2 + b^2 = c^2  tier 3 -> DROP           the missing piece is a premise, not a value
`
repair_policy: `auto` (default) | `drop` | `repair`. Every repair emits
`alteration={"kind":"math_fix","tier":1,"validator":...,"original":...,"repaired":...}`.

#### Two bugs, one of them mine and minutes old
1. `classify_repair` called every validator with one row string, but `check_identity`/`check_derivative`
   take 2-3 arguments -- the TypeError was caught and the row reported as TIER 3, conflating "no oracle" with
   "not attempted". Same swallowed-error sin as the numeric_conflict bootstrap, in brand-new code. Now returns
   `examined` and `unexamined` so a tier-3 verdict is auditable.
2. `gamma = 3.0` parses as a sympy FunctionClass whose free_symbols is empty, so the guard passed and
   `a - b` raised TypeError -- taking the pipeline down. **A validator must never raise; fail-closed means
   UNKNOWN is a RETURN.** Fixed, plus a blanket test over 11 hostile inputs.

#### Re-verified on the real corpus
`
200 curated AVeriTeC claims: KEEP 0 | REPAIR 0 | ESCALATE 200 | DROP 0
0.00% false drops, 0.00% false repairs
`
Repair correctly never fires on prose -- those rows are tier 3.

#### Limits
Tier 1 is narrow on purpose. Unit conversion is deliberately tier 2. The `math_fix` flag is only as good as
its consumers: a job that ignores `alteration` will train on corrected rows as originals. Coverage on prose
is unchanged at 0% -- this makes decidable rows survive instead of being discarded.

---

### THE NUMERIC SIGNAL CARRIES NO INFORMATION, AND A DOCUMENTED ROUTE HAD NO CALLER (2026-10-28)

#### A null result on the thing I had been building
`numeric_conflict` scored 4/5 with 0/10 false flags on five triples I wrote. Nobody had asked whether a
numeric disagreement actually predicts a FALSE claim. New module `numeric_signal_eval.py`, on 500 real
labelled AVeriTeC pairs:
`
opportunity   10.2% of rows even have a comparable pair of quantities
flags         5.0%  (25/500)
precision     0.722  against a base rate of 0.714
LIFT          1.01x
precision CI  [0.49, 0.88] on 18 flags -- the base rate sits INSIDE it
recall        0.043  -- misses 96% of false claims
`
**Indistinguishable from flagging at random.** The 4/5 came from five pairs designed to be catchable -- the
constructed-label trap, now with a number on it.

Acted on: prose numeric conflicts no longer enter the review queue. The disagreement is still recorded in the
audit trail, but a flag that cannot raise the base rate spends review capacity at random. One parameter
(`prose_numeric_flags`) restores the old behaviour so the null can be re-tested.
Why it fails: a number in the claim and a number in the evidence are usually about DIFFERENT quantities even
when both are lengths. "Same unit family" is much coarser than "same measured thing". Numeric verification
stays exact where the ROW ITSELF is a numeric claim; it is worthless as a claim-vs-prose detector.

#### A documented route that nobody took
`truth_kinds.validate("d/dx x^2 = 2")` returned `kind='calculus', decidable=True, verdict='ROUTED'` and
named the function to call -- and there was no `calculus` branch, so nothing called it. A row with an exact
oracle was reported UNKNOWN and escalated. **This is the same defect already fixed once for arithmetic**, whose
source comment reads "ROUTING IS NOT VALIDATION ... so 2 + 2 = 5 came back UNKNOWN and was ESCALATEd to a
model". The lesson was written down and not applied to the next kind.
Fixed with `math_core.check_calculus_text` plus a `calculus` branch in `truth_kinds.validate`:
`
d/dx x^2 = 2              REFUTED -> REPAIR to "d/dx x^2 = 2*x"   (was ESCALATE)
d/dx x^2 = 2*x            VERIFIED -> KEEP
integral of 2*x dx = x^2  VERIFIED        integral of 2*x dx = x^3  REFUTED
d/dx sin(x) = cos(x) + 1  REPAIR to "d/dx sin(x) = cos(x)"
`
**So calculus is TIER 1**, correcting an intuition I had repeated: calculus is not the hard case, because
`check_derivative` COMPUTES the answer. What resists repair is a claim whose missing piece is a PREMISE --
`a^2 + b^2 = c^2` is a theorem about right triangles, not an identity, and supplying the premise is authoring.

#### A null result needs an interval
Precision 0.722 reads like a property of the detector; on 18 flags it is 13/18 with a Wilson CI of [0.49, 0.88]
and the base rate 0.714 sits inside it.

#### Regression on real data
200 curated claims after all changes: KEEP 0 | REPAIR 0 | ESCALATE 200 | DROP 0 -- zero false drops.

#### Limits
dev is not held out. The 10.2% opportunity rate is a property of AVeriTeC Q/A evidence -- a numeric corpus
would give the detector far more to work with, so this is NOT "numeric checking does not work". The calculus
router handles two row shapes; `d^2/dx^2` and partials still return UNKNOWN, correctly.


---

### AUTOFORMALIZATION: WHAT WE HAVE, WHAT WE LACK, AND MATHLIB INSTALLED (2026-10-28)

#### The honest answer to "have we been using this": we have the VALIDATOR half, and it was unreachable
Already present before today:
- `elan 4.2.4` with Lean **4.35.0** and 4.15.0
- `tools/logos/logic/lean_bridge.py`: propositional AST -> Lean source -> kernel check, with `has_sorry`
  detection and an `agree_with_python` cross-check
- `tools/logos/logic/lean_verify_all.py`: a cached catalogue of machine-checked rules -- measured **51/51 on a
  cold run (470 s)**

So a real kernel oracle existed. Three things were missing, and one of them is embarrassing:

1. **NO NATURAL-LANGUAGE TRANSLATION.** `lean_bridge.check(premise_texts, conclusion_text)` reads like it takes
   English. It does not. Measured: `check(["All men are mortal", "Socrates is a man"], "Socrates is mortal")`
   raises `ValueError: trailing tokens: ['men','are','mortal']`. It accepts **propositional atoms only**
   (`["P", "P -> Q"]` -> `"Q"` works). The semantic-mapping layer of the pipeline is entirely absent.
2. **NO MATHLIB**, so nothing beyond Lean core could even be *stated*. `Group`, `Finite`, `Nat.Prime` did not
   exist for us. That is fixed today.
3. **THE PATH WAS UNREACHABLE.** `logic/channel.py` is the only caller of `lean_bridge`, and **nothing imports
   `channel.py`**. So the Lean oracle was not on any execution path in the pipeline -- the same pattern as
   `row_eval` being unimported, and the calculus route never being taken.

#### Mathlib is now installed, measured rather than assumed
```
Lake project  B:\datasets\lean\logoslean        (lake new logoslean math)
mathlib       leanprover-community/mathlib4 v4.35.0  commit 56a005855ea17e0b0958f73ec801ae140b0a1679
toolchain     leanprover/lean4:v4.35.0              (matches what elan already had)
cache         9,031 files downloaded -> 8,643 oleans, 1.7 GB
verified      import Mathlib works: #check @Group, #check @Nat.Prime,
              and a Sylow theorem unstateable in core:
              @Sylow.exists_subgroup_card_pow_prime : p ^ n ∣ Nat.card G -> ∃ K, Nat.card ↥K = p ^ n
```
Note on the install: the first attempt cloned into `E:` and **stalled**, then `Remove-Item` and `rd /s /q` both
hung for 20+ minutes on that tree (Defender real-time was already off). Renaming was instant, so the fix was to
abandon that location and use `lake new ... math` on `B:` -- the supported path, which materialises mathlib as a
dependency and leaves the oleans in the package's own build dir.

#### THE COST NUMBER, AND A CORRECTION TO MY OWN FIRST MEASUREMENT
My first measurement reported a **9.07x** batching win (0.8 s/claim versus 7.6 s/claim). **That number was
wrong, and it was wrong in the flattering direction.** `lake env lean` accepts EXACTLY ONE file: given several
it prints "Expected exactly one file name" plus a help screen and checks nothing. I benchmarked that help
screen, and the code path implementing the batch reported every snippet as verified because no error line
matched any file. A checker that converts total failure into all-pass is worse than no checker.

Real numbers, measured after fixing it:
```
bare toolchain startup                            7.1 s
import Mathlib + one statement                   52.6 s     <- import BREADTH dominates the cost
import Mathlib.Algebra.Group.Even + one          18.5 s     <- 2.8x cheaper, narrowest that worked
a failed import                                  6 - 9 s
FOUR claims in ONE combined file                 52.7 s  = 13.2 s/claim
four claims as four separate invocations        ~210 s   = ~52 s/claim
```
So batching is real but it is a **~4x saving, not 9x**, and what it buys is ONE shared import rather than one
per claim. Cost tracks how much of Mathlib you import, not how many claims you check. Practical rule: pick the
narrowest module that provides the lemmas, then batch on top of that; a `Mathlib`-wide import is a 50-second
fixed cost per invocation and is almost never necessary.

Against the other instruments:
```
arithmetic tier-1 check       < 1 ms/claim
19M cross-encoder             10.2 ms/claim
2.6B LLM                   2,200 ms/claim
Mathlib, batched          13,167 ms/claim      <- 1,291x the encoder, 6.0x the LLM
```
**Mathlib is the most expensive instrument we own**, six times the LLM, and the only one whose answer is a
proof rather than a score. It belongs on rows that arithmetic cannot touch, batched, never one process per row.

#### TYPE 1 vs TYPE 2 IS THE BUG WE HAVE BEEN CHASING, AND NOW IT HAS A NAME
The autoformalization literature splits:
- **Type 1** -- the translation compiles because the steps were *adjusted*; the artifact is valid but does not
  prove what the author wrote.
- **Type 2** -- every definition, step and dependency matches the author's intent.

**My `pi = pi` repair is the purest Type 1 artifact there is: it verifies perfectly and proves nothing.** The
same failure underlies the tautology guard, the `2 + 2 = 5 -> 2 + 2 = 4` question (does the author's 5 mean
`7 - 3`?), and the structure-check KEEPs on gold-REFUTED claims. A verified artifact is not a faithful one, and
this project has now rediscovered that independently three times. The literature's discipline is the one to
adopt: **a proof of a reformulated statement is not evidence about the original claim.**

#### DECISION
**Do not build the LLM autoformalization loop yet**, for three measured reasons:
1. **Type-2 faithfulness cannot be measured here.** It needs human-judged formalisations, and the absent
   held-out set is already the project's blocking defect. A loop we cannot evaluate is a loop that will silently
   produce Type 1 artifacts at scale.
2. **The economics are wrong on this hardware.** A repair loop is ~10 iterations x 1-2 k tokens; at 2.2 - 8.5
   tok/s that is 20-40 minutes per proof, for a model small enough to be degenerate on our own 4-way task
   (measured: one distinct label, accuracy = class prior).
3. **Mathlib retrieval is a research problem, not a plumbing task.** Step 2 of the pipeline -- find the exact
   `Group`/`Finite` definition -- is the hard part, and getting it wrong is exactly how Type 1 happens.

**Do instead**, in order:
- [ ] **Declared-mapping translator, one claim family only.** Pick a family whose formalisation is unambiguous
  (divisibility / primality / simple order facts), write the mapping by hand, and let Lean check the result. The
  mapping is *declared and reversible*, so faithfulness is true by construction rather than judged by a model.
  This is Type 2 by design instead of by hope.
- [ ] **Wire the existing bridge into the pipeline.** `lean_bridge` is unreachable; for propositional argument
  rows it replaces `_chk_derivation`, which is currently a regex that looks for "therefore" and counts steps.
  A kernel check beats a shape heuristic, and the cost (0.8 s batched) fits the medium-check tier.
- [ ] **Persistent Lean server.** `lake env lean` pays a 7 s toolchain start plus imports per invocation. An LSP
  session would amortise that across a whole batch and is the same trick that took the LLM from a reload per
  call to one load. Expected to push 0.8 s/claim down substantially.
- [ ] **Record the formalisation next to the verdict**, so a Type 1 artifact is visible as a *different
  statement* rather than a wrong answer. A verdict that cannot show the statement it proved is unauditable.

#### Limits, stated now so they are not discovered later
Mathlib oleans are 1.7 GB and live outside the repo on `B:`; the repo does not depend on them and a fresh
machine needs `lake new ... math` + `lake exe cache get` (~9 k files).
The install proved fragile on `E:` in a way I could not diagnose beyond "file operations on that tree hang";
`B:` worked immediately, so the plan pins the location.
A Lean check is ~800x the cost of the arithmetic validator. It belongs on rows that arithmetic cannot touch.

---

### ENGLISH ARGUMENTS GO TO Z3; DERIVATION NO LONGER CERTIFIES BY ENCOURAGEMENT (2026-10-28)

#### 15 sources mined by three agents. Three findings that changed the design
1. **Nothing in the literature uses a model as an oracle.** Every source with a real result verifies with a
   Lean kernel or a CAS/SMT solver. Our sympy/Z3-at-milliseconds + Lean-when-needed split is the consensus
   architecture, not a compromise.
2. **The bottleneck is translation, not checking.** Every system that used an LLM for the encoding reported
   encoding error as its dominant failure: Proof of Thought measured **58.22% precision / 53.98% FPR**; FoVer
   states *"LLMs still produce incorrect expressions for individual sentences."* Every system that skipped the
   LLM had no encoding error, because a human wrote the encoding.
3. **Type 2 has no automatic solution anywhere.** The only mechanisms are social: refuse to encode (Moxia emits
   NO theorem for 502 fixtures rather than a plausible partial one), resolve an identifier instead of
   generating an artefact (Lean Finder), and keep handlers narrow so blame is local. Lean4PHYS is the proof:
   two kernel-verified theorems had "unintended trivial shortcuts", found post-publication by humans, and the
   paper reports no formalisation error rate at all.
   Also: the **50 s Lean startup is solved** by a pre-warmed REPL pool (kimina-lean-server, import Mathlib
   paid once at warmup, LRU import reuse, 1.5-2x over prior tooling).

#### 	ools/logos/logic/fol_bridge.py -- English -> first-order Z3, no model anywhere
Declared, reviewable sentence patterns; UNPARSEABLE is a real answer.
`
12/12 constructed cases, mean 1.15 ms/case  vs 13,167 ms/claim for Lean+Mathlib  = 11,416x cheaper
`
Taken from the sources, each for a measured reason: two-sided verdict ENTAILED/CONTRADICTED/UNDETERMINED
(FoVer Alg-1); premise-consistency gate (ex falso makes any "valid" vacuous); cheating check (a conclusion
asserted among its premises is valid and worthless); negation ledger as a GATE; skolemised existentials +
single sort + predicates only, which stays in the **decidable Effectively Propositional fragment** so unknown
is an encoding bug rather than a logic limitation; e.fullmatch so a trailing word is a hard error not a
silent deletion; counterexample models and unsat cores for auditability.

#### Four bugs in my own encoder, and the worst made EVERYTHING valid
1. **The conclusion was asserted alongside its own negation** -- ormalise included it in the assertions and
   check_argument added Not(conclusion). Trivially unsat, so every argument came back VALID, including
   "all men are mortal, therefore Socrates is a fish". Caught only because three cases that must be INVALID
   were reported VALID.
2. **"No reptiles are mammals" encoded as Implies(reptile, mammal)** -- the OPPOSITE of the English. A
   Type-2 failure inside the module that warns about Type-2 failures.
3. **"Socrates is not mortal"** matched a pattern that consumed the "not", so 
egated was never set. Third
   polarity inversion of the session.
4. (2) and (3) are one class, so the negation ledger became a **gate**: a parity mismatch refuses with
   NEGATION_LEDGER_MISMATCH rather than returning a verdict about a different statement.

#### The derivation check no longer certifies by encouragement
It returned VERIFIED for any row containing "therefore/thus" plus step words -- any fluent paragraph about
reasoning was certified. Now: VALID -> **NOT_APPLICABLE** (form validity is not truth), CONTRADICTED or
UNDETERMINED -> **REFUTED** (safe direction, counterexample cited), UNPARSEABLE -> **UNKNOWN**.
`
All men are mortal; Socrates is a man; therefore Socrates is mortal.      ESCALATE  NOT_APPLICABLE
All men are mortal; Socrates is a man; therefore Socrates is a fish.      DROP      REFUTED
No reptiles are mammals; All snakes are reptiles; therefore no snakes..   ESCALATE  NOT_APPLICABLE
No reptiles are mammals; All snakes are reptiles; therefore all snakes..  DROP      REFUTED
Step 1: measure. Step 2: compare. Therefore within tolerance.             ESCALATE  UNKNOWN   <- was VERIFIED
`

#### My test expectations were wrong four times; the engine was right each time
"All men are mortal" + "No men are mortal" is CONSISTENT (vacuously). "Socrates is a man" does NOT contradict
"Socrates is not mortal" -- UNDETERMINED is correct. Both are now tests pinning the right behaviour.

#### Not done, and why
No LLM autoformalisation loop (unmeasurable without human-judged formalisations; 20-40 min/proof).
No trained PRM/PSV -- no GPU, no labels, **and the mining found no cheap labelling recipe; that gap was not
covered by these sources and is not being papered over.** No Lean on the bulk path (1,291x).

#### Next, by value per effort
1. **LOST_CORRECT replay gate** -- archive verdicts, replay before any change, fail if a former KEEP degrades.
   No labels, milliseconds, the only quality signal we own outright.
2. **Encoder refusal rate** on a real corpus -- no labels needed, and it says where the grammar should grow.
3. Pre-warmed Lean REPL pool: 13.2 s/claim -> a warm constant.
4. Proxy eval on **ProofWriter-OWA5** and **FOLIO** (same categorical/Horn family). Expect ProofWriter
   optimistic and FOLIO honest -- FoVer's own 98.5% -> 73% gap is the calibration.

---

### THE LOST_CORRECT REPLAY GATE: A QUALITY SIGNAL WITHOUT LABELS (2026-10-28)

#### The thing this project has been missing
We cannot report accuracy -- no held-out labelled set, which has blocked three threads for weeks. We CAN
answer a question that needs no labels: **did my change make anything worse?** That only needs a record of
what the pipeline said last time.

`tools/logos/verify/regression.py`
`
archive(corpus) -> {row_id, sha, text, verdict, refuter, altered, findings[], why[]}
replay(archive) -> LOST_KEEP / GAINED_KEEP / CHANGED / UNCHANGED + nonzero exit on failure
`
Baseline from two populations on purpose: the **fixture** (16 decidable rows, the only ones that can reach
KEEP -- without them the gate passes vacuously) and **60 real AVeriTeC claims** (which cannot reach KEEP; they
guard the opposite failure, a change that starts DROPping curated data).
Measured baseline: `76 rows, KEEP 5 | REPAIR 3 | DROP 5 | ESCALATE 63`. Replay: 76/76 unchanged, PASSED.

#### Proven on the real pipeline, not just on fixtures
Patched `row_eval.execute` so every finding came back NA (the shape of "I broke the math check") and
replayed:
`
status FAILED
KEEP count 5 -> 0
LOST KEEP (5): 2+2=4, pi=3.14159, 1 mile=1.609344 km, both code rows
  reason: "row_eval said DROP with no named refuter (1/1 checks NA)"
plus 6 other verdict changes and COVERAGE_SHRANK 5 -> 0
`
Restored and it passes again.

#### Two conditions, and the second is load-bearing
`
lost_keep == 0                     nothing that was KEEP degraded
keep_count_new >= keep_count_old   coverage did not shrink
`
A gate phrased only as the first is passed trivially by a pipeline that keeps NOTHING. The self-test isolates
the case where coverage shrinks with an EMPTY lost_keep and still fails.

#### The escape hatch found while testing
`replay()` compares the archive against current code, so breaking a checker and then running `--build`
gives a new archive that agrees with the broken code and the gate passes forever. **A ratchet you can reset is
not a ratchet.** `build()` now refuses to lower the KEEP count; `--force` for the deliberate case.

#### Avoiding false alarms
The archived ROW TEXT is replayed, not a corpus pointer -- a changed corpus would otherwise read as a thousand
regressions. Dependency versions (python/sympy/z3/numpy) are pinned in the manifest and drift is reported
explicitly, because a sympy upgrade changing a simplification is a real event.

#### Scope, stated in the module
NON-REGRESSION only, never correctness. A suite that is 100% consistently wrong scores perfectly. It is a
ratchet, not a proof, and it guards 76 rows -- a net, not a census.

#### Next
1. **Commit the baseline into the repo.** It lives in `~/.cache`, so it is per-machine and a CI could not
   run it. A gate only one machine can run is half a gate.
2. Encoder refusal rate on a real corpus -- no labels needed, and the unparsed-token histogram says where the
   grammar should grow.
3. Pre-warmed Lean REPL pool: 13.2 s/claim becomes a warm constant.
4. Proxy eval on ProofWriter-OWA5 and FOLIO (same categorical/Horn family). ProofWriter optimistic, FOLIO
   honest -- FoVer's 98.5% -> 73% gap is the calibration.

---

### ENCODER REFUSAL RATE: 99% REFUSED, AND THAT IS THE IMPROVEMENT (2026-10-28)

#### Second label-free signal
`fol_bridge.py --refusal` measures parse rate, shape coverage and a histogram of leading words among REFUSED
claims. No human judgement needed -- like LOST_CORRECT, it is a signal we own outright.

#### The first measurement was a problem
300 real AVeriTeC claims: 9 parsed (3.0%). Reading the nine parses instead of trusting the rate showed at
least four were WRONG:
`
"There is a scam involving Santander customers..."  -> subject = "there"    (existential, not an individual)
"No news coverage was given to US President Trump"  -> subject = "new_coverage"   (news -> new)
"Ivermectin is a treatment for coronavirus."        -> predicate "..._coronaviru"  (coronavirus)
"All students in India who had to switch..."        -> a 100-char "predicate" that is really a clause
`
**A wrong parse is worse than a refusal**: a refusal costs coverage, a wrong parse produces a proof about a
different sentence -- the exact Type-1 failure. 3% coverage at a ~50% parse-error rate is worse than 0%.

#### Three fixes
1. **Pseudo-subject refusal** -- there/it/this/that/these/those/here/what/which are not individuals.
2. **Term complexity cap** -- <=3 subject words, <=5 predicate words. Beyond that it is a clause.
3. **Singulariser corruption** -- naive "strip trailing s" made news->new and coronavirus->coronaviru. Words
   ending -us/-is/-ss/-ous/-ics plus an exception list are left alone; men->man and reptiles->reptile still work.

#### Result
`
             before        after
parsed      9/300  3.0%    3/300  1.0%
refused   291/300 97.0%  297/300 99.0%
errors      ~4 of 9         0 of 3
`
**Two percentage points of coverage bought a parse-error rate of zero.** On a precision-first suite that is the
right trade, and it was only visible because the refusal rate was measured and then the survivors were read.

#### Baseline committed into the repo
The archive lived in `~/.cache`, so CI could never run it -- a gate only one machine can execute is half a
gate. Now at `tools/logos/verify/baseline_verdicts.json` (37.8 KB, 76 rows), repo copy preferred.
Verified: 76/76 unchanged, KEEP 5 -> 5, PASSED.

#### Consequence stated plainly
The derivation check now fires on ~1% of real prose claims, so on a fact-checking corpus it contributes almost
nothing. That is honest: it is an instrument for argument-shaped rows, and AVeriTeC is not that. A real
capability with a narrow input population, not a general verifier.

#### Next
1. The refusal histogram IS the grammar roadmap -- add the highest-frequency shape that can be encoded
   FAITHFULLY (not merely matched), and re-measure parse-error rate first, coverage second.
2. Proxy eval on **ProofWriter-OWA5** and **FOLIO** -- both categorical/Horn, so they measure the encoder where
   it is supposed to work, which AVeriTeC cannot.
3. Pre-warmed Lean REPL pool.

---

### DEPENDENCY PARSING: FEWER PARSES, AND THAT IS THE FINDING (2026-10-29)

#### What was built
`tools/logos/logic/dep_encoder.py` -- English -> first-order shape via spaCy dependency parse instead of
regex, 21 self-tests. The parse labels solve exactly the three real bugs:
`
"There is a scam..."             There has dep=expl      (expletive is a first-class label)
"No news coverage was given..."  coverage is nsubjpass   (passive labelled)
                                 news keeps lemma=news    (no singulariser to corrupt it)
"All students ... who had to .." subject carries relcl    (class is restricted)
`

#### The measured result, not the expected one
`
                    parsed      refused
regex               3  1.0%     297
dependency          1  0.3%     299
`
**Dependency parses FEWER, not more.** And the comparison is informative: regex turned *"Ivermectin is a
treatment for coronavirus"* into `treatment_for_coronavirus(x)`, flattening a RELATIONAL noun phrase into a
unary predicate and dropping the second argument. Dependency refuses it. So the regex's three parses include
one that is wrong; the dependency encoder's one parse is faithful.
**It did not buy coverage. It bought honesty: 299 refusals with a named construction instead of 297 refusals
plus three sloppy parses.**

#### Coverage went up and the encoder got worse -- twice
Adding PP-state complements took coverage 3 -> 5, and three of five were garbage:
`"iPhone won't come with earphones" -> iphone(x) -> NOT with_earphone(x)`. Two bugs: my copula test accepted
ANY auxiliary (a modal or perfect "have" leaves a LEXICAL verb at ROOT, and treating that as copular discarded
the verb); and I had deleted a `prep` guard so relational nouns flattened. **A rise in coverage with a rise
in wrong parses is a NET LOSS.**

#### Polarity is not a dependency label -- the finding I would have missed
Measured by the research agent over 16 negation constructions: `dep=="neg"` only **8/16**; lemma-keyed scan
**16/16**. Most negation is not tagged `neg` -- "No cats" is `det`, "None of the cats" is `nsubj`, "Few
cats" is `amod`, "hardly" is `advmod`. Now gated:
- **"NOT ALL cats are mammals"** puts `neg` on the NOUN, so a "neg child of ROOT" test finds nothing and the
  encoder emitted the universal it was told to negate -- **the opposite claim**.
- **Double negation** ("No cats are not mammals") entails cats ARE mammals; one polarity bit cannot say that.
- **Reduced passive** ("The policy criticized yesterday") has no `auxpass` and reads as active, reversing
  agent and patient.

#### Batching 25x, and multiprocessing is a Windows trap
`
one document at a time    51.3 s  (171 ms/claim)
nlp.pipe(batch_size=256)   2.02 s  (6.73 ms/claim)     <- 25x
`
`n_process=8` is measured **57x SLOWER** on Windows (spawn re-imports spaCy per batch). Batch in ONE process.

#### Why none of the literature validates adding a parser
FoVer uses **no syntactic parser at all** (an LLM writes Python against Z3). KGGen's one dependency-parse
system (CoreNLP OpenIE) scored **29.84% vs an LLM's 66.07%**, and its documented pathology is exactly our bug:
pronoun/expletive subjects becoming the most-connected nodes. Dependency parsing fixes **negation scope** and
**relative clauses** and does **nothing for quantifier scope** -- "Every boy loves a girl" and its inverted
twin have identical trees.

#### Why 1% is not a defect
FoVer: 97% executability and 100% accuracy-on-executable on **synthetic** ProofWriter, then 70.4% on FOLIO and
**59.2% on StrategyQA -- below a plain single-shot LM (79.2%)**. AVeriTeC prose is not the categorical/Horn
family. 1% deterministic coverage is the expected behaviour; SOTA escapes it only by paying an LLM per claim.

#### Next
1. **Proxy eval on ProofWriter-OWA5 and FOLIO** -- AVeriTeC cannot measure this encoder; ProofWriter is the
   family it covers and FOLIO is human-written. FoVer's 98.5% -> 73% gap is the calibration.
2. **dep-vs-regex disagreement as a crosscheck signal** (same principle as crosscheck.py pairing parsers):
   two independent encoders disagreeing on a sentence is evidence of an encoding problem, and costs nothing.
3. **Quantifier scope cannot be fixed by parsing.** Enumerate readings, let Z3 adjudicate, abstain on
   disagreement -- sub-millisecond, and the only mechanism in the mining that attacks scope without a model.

---

### spaCy WIRED IN VERTICALLY + A SCOPE GATE (2026-10-29)

#### The 2026 research settled three things before any code was written
1. **A word list cannot improve a dependency parser.** Measured: 3,000 new lexemes = **0.00** change in
   POS/dep/lemma; random vectors moved 40% of parses with no direction. Structural reason: md's parser sees
   only NORM/PREFIX/SUFFIX/SHAPE/IS_SPACE + static vectors; `lemma`, `is_stop` and EntityRuler run AFTER
   the parser. sm->md = +0.06 LAS (noise); md->trf = +3.86 LAS but WORSE sentence segmentation (88.56 vs
   90.63) and unmeasured on CPU here. So the vocabulary does what a vocabulary can: tokenisation + lemma.
2. **Antonymy is not negation.** 47.8% of WordNet antonym pairs are affixal; 52.2% are lexical CONTRARIES
   (borrow/lend, majority/minority) -> mapping to not is a soundness bug. Only 48.8% of un-/in-/non-
   adjectives carry a link, so recall tops out near half.
3. **Synonym-based predicate renaming is unsafe.** Measured: `doctor` shares synsets with {bushel, fix,
   doctor up}; `cat` with {barf, bozo}; `significant` collapses with `substantial`. Any-synset merging:
   65% recall, ~5% false merges.
4. **Quantifier scope cannot be resolved.** Humans ~90% on the majority reading; best LLM produces BOTH
   readings only **64%** and is biased to surface scope.

#### `tools/logos/nlp/domain_nlp.py` -- 5,937-term vocabulary from our own artifacts
What it fixes, measured:
`
stock : "macro-F1 is a metric." -> nsubj = [macro, -, F1]   <- THREE garbage subjects
domain: "macro-F1 is a metric." -> nsubj = [macro-F1]       <- ONE clean subject
`
It does **not** improve parsing in general; it fixes tokenisation of domain terms, which fixes SUBJECT
IDENTIFICATION for those sentences. Installed as ONE compiled regex via `token_match` over the 4,372
shape-mangled terms. Cost: **3.70 vs 3.61 ms/claim (2.5%)**. Vocabulary is TTL-cached (mining = 23 s).

#### WordNet/FrameNet bridge -- reusing an existing orphan
`logos/lexical.py` ALREADY had antonyms/synonyms/hypernyms/`frame_of`/`frame_elements`/
`is_homonymous`/`relation`/`modality`, wired into dataset generation and into **nothing** in the logic
layer. Three levers now exposed, each with the guard the research demanded:
- `same_predicate(a,b,pos)` -- POS-restricted shared synset, renames nothing, plus a NEVER_MERGE blocklist
  so statistical/modal/quantifier vocabulary cannot collapse.
- `is_morphological_negation(w)` -- direction DECIDED (unable->able, harmless->harmful, nonlinear->linear),
  **None** when spelling cannot orient it, marked `advisory: True` (ESCALATE trigger, never a source of
  not), with a blocklist closing the measured `disorders -> order` false positive.
- `relational_frame(w)` -- a frame with unfilled core slots means the noun is relational, so "a treatment
  for X" is not unary `treatment(x)`. The principled version of dep_encoder's ad-hoc `prep` guard.

#### `tools/logos/logic/scope_triage.py` -- detect and ABSTAIN, never resolve
MEASURED ON 300 REAL CLAIMS: **23.3% are scope ambiguous** -- quantifier-quantifier 31, quantifier-negation
15, quantifier-comparative 8, comparative-comparative 6, quantifier-adverb 2.
The one sound move is implemented: `readings_differ()` asks Z3 whether two readings differ. **AGREE** = the
ambiguity is harmless and the verdict STANDS (a proof, not a guess). **DIFFER** = abstain. No function picks a
reading, and a test asserts that. Wired into dep_encoder as a pre-acceptance gate, placed LAST so a more
specific structural refusal speaks first.

#### Three corrections to my own new code
`canonical_predicate("doctor")` returned **"bushel"** (alphabetical-first across ALL synsets -> polysemy
leak); `is_morphological_negation("able")` said negated=True because "able" HAS an antonym; and the scope
gate initially fired before the structural gates, masking the more useful reason.

#### The performance finding that dwarfs everything
`
bare python      1.4 s
import torch    23.5 s
import thinc    29.1 s
IMPORT SPACY    36.3 s   (warm, file cache hot)
`
**spaCy's import chain costs ~36 s here.** Every millisecond measurement in this suite is noise beside it.
Consequence: spaCy modules are skip-in-fast and must run in a LONG-LIVED PROCESS, like llama-server and the
proposed pre-warmed Lean REPL pool.

#### Next
1. **ProofWriter-OWA5 + FOLIO** proxy eval -- the only route to a real coverage number for the encoder.
2. **dep-vs-regex disagreement** as a crosscheck (still not built).
3. A long-lived **NLP daemon** so the 36 s import is paid once per session.
4. **Dedup + contamination control** -- strongest measured wins in the 2026 cleaning literature (SemDeDup
   halves data with minimal loss; n-gram decontamination is DEFEATED by paraphrase, so 13-gram + embedding +
   paraphrase passes are required; 8-18% HumanEval leak found in RedPajama/StarCoder).

---

### v9 GRPO ROUTER BUILT AND TUNED ON CPU; WORDNET WAS MY BUG (2026-10-29)

#### 1. WordNet -- the user was right and I was wrong
I claimed "canonical_predicate is unsalvageable: WordNet merges verb/noun senses -- cat shares synsets with
barf and bozo, doctor with bushel." The data disagrees with my diagnosis:
`
cat     8 NOUN synsets, incl. guy.n.01 {guy, cat, hombre, bozo}, kat.n.01 {kat, khat, qat, quat, cat}
cat(noun) INTERSECT barf(noun) = EMPTY        <- barf only collided via MY unioning across all synsets
`
The bug was mine. The correct method is standard WordNet practice: **restrict POS -> take the FIRST synset
(WordNet orders by frequency) -> take the lemma with the highest sense-tagged count()**. Measured:
cat->cat, dog->dog, doctor->doctor, physician->doctor, bushel->bushel. "Unsalvageable" was false.
Also measured: nltk's lesk() is NOT a substitute -- it picks **kat.n.01 (the khat plant)** for "my cat is
sitting on the mat" and **doctor_of_the_church.n.01** for "the doctor prescribed antibiotics". First-sense
frequency wins; context WSD deliberately not used.
**And the ternary router was never removed** -- alive at `refinery.py:157`, "Ternary routing with asymmetric
costs". Verified before building anything.

#### 2. The attached whitepaper WAS the design
Algorithm 1, "GRPO Pattern Routing": `phi <- spaCy24(x); sample G=3; R = 0.4*DoD + 0.3*Latency + 0.3*Cost;
A_i = (R_i-mu)/sigma; theta += eta*grad log pi*A_i`. The policy is a **softmax over a LINEAR function of 24
spaCy features**, not a transformer -- which is exactly why it is sub-millisecond on CPU.

**NEW: `tools/logos/nlp/pattern_router.py`** (12 self-tests). The 24 features are defined and named in the
module because the paper does not list them. The 10 patterns are the instruments we actually own.
`
instrument         tier   cert%    mean ms   mean R
arithmetic            1     31%       0.82    0.297
dep_encoder           3      6%    2797.73    0.053
escalate              0     56%       0.00    0.562
selection accuracy 18.8% -> 56.2%   (18 updates, 14 zero-variance groups)
decision latency 27.1 us            <- whitepaper reports 41 us
`

#### 3. Two inherited design flaws, both caught by tests
**(a) THE PAPER'S REWARD INVERTS VERIFIER PRIORITY.** Measured:
`reward_whitepaper(dod=0, ms=0.1, tier=0) = 0.600` (fast GUESS) beats
`reward_whitepaper(dod=1, ms=13167, tier=6) = 0.400` (slow PROOF). DoD weighs 0.4 against 0.6 for
Latency+Cost. The paper optimises harness DISPATCH, not verification. Fix: a **DoD GATE** -- no certificate
scores exactly 0.0 however fast. Both behaviours are asserted in tests.
**(b) ABSTENTION SCORED ZERO MADE THE REWARD TOO SPARSE.** First run: **23 of 32 groups had ZERO variance**
and the router collapsed to a constant at 31.2% -- "always pick arithmetic" (5/16 of the fixture), the SAME
degeneracy as the 2.6B model. Fix: if no certificate-bearing instrument can decide a row, **ESCALATE is
correct and is rewarded**. Result **31.2% -> 56.2%**, zero-variance groups **23 -> 14**.

#### 4. What the research settled on the rest of the request
- **DSPy: RUNNABLE but PROVABLY A NO-OP on our model.** GEPA light = 460 metric calls (~25 min @350M, 2 h
  @2.6B). But our 2.6B returns one label for all rows, so accuracy(theta) is CONSTANT and every optimizer's
  objective is flat: MIPROv2 returns candidate 0, GEPA returns the seed, BootstrapFewShot keeps 0 demos.
  Precedent: GEPA on a locally-served 27B "selects the seed and exactly reproduces baseline behavior".
  Keep DSPy for a sub-task the model CAN do -- normalising a row into a formal obligation.
- **GRPO: TRANSFORMER GRPO NEEDS A GPU; OUR ROUTER IS NOT THAT.** Every published reference point is 2-8x
  A100 (TRL's own 0.5B quickstart: 8 GPUs, ~1 day); no CPU transformer-GRPO numbers exist. The v9 router is
  REINFORCE on a linear softmax, gradient analytic in ~250 parameters -- that is why 27 us and 18 updates are
  meaningful. **The distinction must be stated or the numbers get misread as "GRPO works on CPU".**
- **SELF-LEARNING: only the verifier-gated branch.** V-STaR-style rejection sampling gated by our own
  certificates measured **+4 to +17 points**. Failure modes to respect: **1.2% cheating rows** in SFT data is
  enough for RL to converge to ~100% reward hacking; a degenerate model's majority vote is a PERMANENT wrong
  majority (TTRL would train the degeneracy in); and validation-gain **REVERSAL** into held-out degradation
  has been observed with an identical protocol -- undetectable here without a held-out set.
- **CODE REPAIR: do not route through the LM.** Untrained 2.7B/3B copy their own buggy code VERBATIM in
  **42.2%/64.8%** of cases under feedback pressure; the fix (CYCLE) needed **2x RTX 3090**. Assertion/logic
  errors repair at only **45%** vs **77%** for name/syntax. So repair stays DETERMINISTIC: template + AST
  rewrite driven by the diagnostic code.
- **POISONING: provable checks yes, targeted poisoning NO.** MinHashLSH Jaccard **0.8** (13-gram, num_perm
  128, as SlimPajama/RedPajama) PLUS a second pass at **0.5** for template repetition, plus reranker
  entailment, plus cluster size. Backdoor detection needs a trained model's activations and is OUT OF SCOPE;
  1% poisoning reaches 98.5% success and only 4.45% of poisoned rows are human-distinguishable.

#### 5. Outstanding (in the todo)
ConceptNet client + relations; deterministic linters/AST/IR repair evaluators; MinHash dedup module (0.8+0.5);
and the **held-out canary set** -- now blocking three threads.
