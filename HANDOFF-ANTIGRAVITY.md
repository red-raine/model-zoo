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
