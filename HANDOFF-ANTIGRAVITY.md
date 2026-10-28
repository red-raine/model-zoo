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
