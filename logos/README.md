# `logos` — the reasoning + fact-checking SDK

**Status:** v0.1 · core logic layer built and tested · lexicon + API layers scaffolded
**Owner:** opencode · **Consumers:** mini-minion-0, model-zoo, trace-miner, any family app

---

## 0. The one rule this SDK is built around

> **A lexical knowledge graph supplies premises and type constraints. It never supplies
> validity, and it never supplies truth.**

Validity is a property of argument *form*. Truth is a property of the *world*. No resource
in our stack (WordNet, FrameNet, VerbNet, ConceptNet, HowNet) stores either one. What they
store is word-level meaning relations plus predicate–argument structure.

Every layer below therefore declares, machine-readably, **what it proves and what it does
not**. Downstream apps read that manifest instead of assuming. This exists because the
`dataset-builders` engine already produced a trace claiming *"The syllogistic form is:
'P or Q. Not P. Therefore, Q.'"* — derived from the substring `or` inside the word
`corrupt`.

---

## 1. Layers

```
 L5  api/            FastAPI + MCP  ──►  every family app, mini-minion tools
 L4  evidence/       Wikidata · SearXNG/crawl4ai · sympy     ← TRUTH
 L3  logic/          syllogism · selectional types · contradictions  ← VALIDITY
 L2  frames/         frame-semantic-transformer (FN 1.7)     ← STRUCTURE
 L1  lexicon/        WordNet · VerbNet · ConceptNet · HowNet  ← PREMISES + TYPES
 L0  honesty.py      the capability manifest
```

### What each layer can and cannot do

| layer | proves | does **not** prove |
|---|---|---|
| L1 lexicon | that `robin ⊂ bird` (hypernym chain); that `give` licenses Agent/Theme/Recipient with selectional restrictions | anything about truth; anything about validity |
| L2 frames | which situation a sentence evokes and who fills which role | whether the situation happened |
| **L3 logic** | **validity** of a categorical syllogism (decidable: 256 forms, 24 valid); type violations; `IsA` contradictions | truth of premises |
| L4 evidence | arithmetic (sympy, exact); entity facts (Wikidata, exact) | contested claims; anything needing specialist judgement |
| L5 api | nothing itself — it exposes the above and **propagates their limits** | — |

---

## 2. Built and tested today

### `logic/syllogism.py` — categorical syllogism validity

This is the real "logical thinking engine", and it is fully decidable.

- implements the **six classical validity rules** (undistributed middle, illicit
  process of major/minor, two negative premises, negative-premise/conclusion parity,
  two particular premises, particular-premise/conclusion parity)
- **self-test proves the rules reproduce the 24-form table over all 256 forms,
  mismatches = 0**
- catches named fallacies *with the specific rule violated*
- shallow NL front end: identifies the major premise by the conclusion's **predicate**
  (not by line order), then derives figure

```
All men are mortal; All greeks are men; Therefore all greeks are mortal.
  -> AAA-1 Barbara, valid
No reptiles are mammals; Some lizards are reptiles; Therefore some lizards are not mammals.
  -> EIO-1 Ferio, valid
Cron jobs on the same runner start the same reindex task.
  -> refused (not a categorical syllogism; will not guess)
```

Two bugs the self-test caught in my own front end: the four figure cases were in the
wrong order (Barbara → figure 4), and premises were assumed major-first. Both fixed.

### `tools/verify/` — evidence layer (built earlier this session)

| module | method | test |
|---|---|---|
| `math_verify.py` | sympy on arithmetic/percent/grouped expressions, strict whitelist | 15/15 |
| `fact_verify.py` | Wikidata SPARQL (deterministic) → SearXNG → local LLM judge | 11/11 |
| `lora_check.py` | decompose → check-worthiness → questions → pairwise → evidence ratio | 17/17 |
| `eval_factcheck.py` | labelled-set harness + self-check baseline + dangerous-error rate | 9/9 |

---

## 3. Measured state (honest, and worse than the demo implied)

On HaluEval qa (10 items, 5 true / 5 hallucinated), 1 run:

| system | coverage | accuracy on answered | dangerous (hallucination accepted) | s/claim |
|---|---|---|---|---|
| baseline ("just ask the model") | 0.20 | 1.000 | 0.000 | 48 |
| naive checker | **0.00** | — | — | 3.3 |
| LORA pipeline | **0.00** | — | — | 216 |

**Both verifiers abstain on everything.** Diagnosed causes:

1. **Retrieval fails on verbatim claims.** SearXNG returns 0 results for
   `"Reagan and Give Us Our Skeletons were both documentaries"` — the exact failure mode
   the literature predicts. Fixing this needs entity/relation queries, not claim strings.
2. **Decomposition returned `[]`** on question+answer claims, so LORA never reached
   retrieval and still burned 216 s/claim retrying the model. A fallback was added; it did
   **not** restore coverage (still 0.00 after the fix).
3. **The LORA upgrade cost 65× the naive checker for zero accuracy gain.** That is a
   negative result about my own work, recorded rather than hidden.

Until retrieval is fixed and this is re-run on AVeriTeC/FEVER at n≥200 with 3 runs, none
of these modules should gate anything.

---

## 4. Lexicon layer — what to install, and what to skip

Verified against live repos/docs this session.

**ADOPT**

| resource | how | size | why |
|---|---|---|---|
| **WordNet** (installed) | `nltk.download('wordnet')` | ~10 MB | hypernym chains → premise grounding |
| **VerbNet 3.3** | `nltk.download('verbnet3')` + `VerbnetCorpusReader` (the LazyCorpusLoader still resolves to 2.1) | 3.7 MB | **selectional restrictions — the only machine-checkable type layer.** `Patient[+concrete]`, `Agent[+animate]` |
| **frame-semantic-transformer** `small` | `pip install frame-semantic-transformer`, `use_gpu=False` | ~240 MB | only maintained, pip-installable, CPU-viable FN 1.7 parser. MIT code / Apache-2.0 models |
| **ConceptNet** | pruned Parquet from `conceptnet-assertions-5.7.0.csv.gz` (475 MB) → 9 relations, `/c/en/` only ≈ 50–80 MB | — | offline `IsA`/`PartOf`/`Causes`/`Antonym` lookups, no service |

**SKIP** (and why): MKGDB (Neo4j, 37M nodes, no licence), Mini-HowNet (link farm),
`Final_word_Similarity`/`SentenceSimilarity`/`SentimentAnalysisDictionary` (Chinese-only),
PSLQA (Gurobi licence, py2.7), ECNU-SenseMaker (torch 1.6, unreproducible),
`conceptnet-lite` (its own README says do not use), `concepcy` (calls the web API — not local),
**`llm4free`** (keyless tier reverse-engineers consumer endpoints; conflicts with the
Constitutional Key Law — use LiteLLM :4000 instead).

**Licence traps to pin now:**
- ConceptNet **data is CC BY-SA 4.0** (code is Apache-2.0). Shipping a derived DB means the
  derived DB is share-alike. Runtime inference results are fine.
- FrameNet **1.7 = CC BY 3.0** (commercial OK with attribution); FrameNet **1.5 =
  non-commercial only**. Pin 1.7.
- VerbNet ships under "distributed with permission of the author" — not an open licence.
  Fine internally; get clarity before commercial distribution.
- OpenHowNet is MIT code but the HowNet dictionary data has a separate research licence, and
  it is Chinese-first — skip for an English stack.

**Ideas worth stealing** (not code): FramenetParser's *joint* frame+FE decoding;
CoCo-Ex's linking thresholds (`len_diff_token>1`, `len_diff_char>10`, `dice<0.85`);
`rconceptnet`'s `--tree` k-hop expansion as a first-class verb; ecg_framenet's
frame-to-frame relations (Inheritance / Perspective_on / Using / See_also).

---

## 5. The honesty manifest (`honesty.py`)

Machine-readable, so a consumer cannot accidentally over-trust a layer:

```python
from logos import capabilities
capabilities()["logic.syllogism"]
# {'proves': 'validity of categorical syllogism form',
#  'cannot': ['truth of premises', 'non-categorical argument forms'],
#  'method': 'six classical rules, verified against all 256 forms',
#  'cost': '0 tokens, deterministic'}
```

Any app that calls a layer gets its limits back with the answer.

---

## 6. Integration with mini-minion-0

`mini-minion-0` exposes `MinimionBackend.run(prompt, cwd, model)` implementing the
ONE-protocol contract, with `CostGuard.assert_zero_tokens()` on every entry. `logos` fits
that contract exactly, because **L1–L3 are zero-token and deterministic**:

```
mini-minion task
   └─► logos.logic.syllogism      (0 tokens)   check argument form
   └─► logos.logic.types          (0 tokens)   VerbNet selectional violations
   └─► logos.evidence.math        (0 tokens)   sympy arithmetic
   └─► logos.evidence.wikidata    (0 tokens)   entity facts
   └─► logos.evidence.search      (1 LLM call) only when evidence is genuinely needed
```

So the SDK is **majority zero-token** and `CostGuard` will pass by default. That is the
"crazy things with minimion" path: deterministic gates first, model only for the residue.

---

## 7. Build order (what is next, in dependency order)

1. `honesty.py` + `__init__.py` — the manifest and public surface *(today)*
2. `logic/types.py` — VerbNet selectional restriction checker (needs `verbnet3`)
3. `logic/contradiction.py` — `IsA` conflicts across WordNet/ConceptNet (vote, flag, never silently merge)
4. `lexicon/conceptnet.py` — pruned Parquet loader
5. `frames/` — wrap `frame-semantic-transformer` small
6. **Fix retrieval** (the actual blocker): entity/relation queries + crawl4ai full-page fetch
7. `api/server.py` + `api/mcp.py` — expose to mini-minion and every app
8. **Measure on AVeriTeC/FEVER** at n≥200, 3 runs, with the self-check baseline

Item 6 is the one that decides whether the evidence layer works at all. Items 1–5 are
deterministic and cannot be wrong in the way retrieval can.

---

## 8. What this SDK will never claim

- that ConceptNet weight is a probability (it is crowd-agreement / source count, uncalibrated)
- that a hypernym chain proves a proposition (it licenses a substitution; proof needs a calculus)
- that frame parsing establishes truth conditions (it gives their scaffolding)
- that a valid syllogism is a true one
- that an LLM judge is ground truth
