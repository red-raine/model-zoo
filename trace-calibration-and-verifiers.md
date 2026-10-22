# The gate was measuring vocabulary, not truth — and my replacement was worse before it was better

**Date:** 2026-10-21 · **Owner:** opencode · **Status:** TRACE retired as a quality gate; deterministic verifiers built

---

## 1. I had to withdraw my own claim from the previous session

Last turn I wrote: *"a hand-written textbook Toulmin answer scores 0.281 (grade D),
therefore the 0.55 gate is unreachable."*

That was weak evidence. **I wrote the fixture and I asserted it was good.** Asserting
a fixture is high quality is not a measurement. So I replaced it with fixtures labelled
by an *objective* property — true/false, marked/unmarked — and tested falsifiable
predictions. `tools/scoring/calibrate_trace.py`:

| fixture | truth | TRACE | grade |
|---|---|---|---|
| `"Therefore, 2 + 2 = 5. For example, since 2 is approximately 2.5…"` | **false** | **0.816** | **A** |
| `"The sky appears green … since air molecules are larger than the wavelength of blue light"` | **false** | **0.721** | **B** |
| word salad, zero information, all connectives | **nonsense** | **0.759** | **B** |
| correct explanation, plain declarative prose | true | 0.133 | D |
| correct, explicitly labelled Claim/Evidence/Warrant/Backing/Qualifier/Rebuttal | true | 0.313 | D |
| one connective repeated five times (correct content) | true | 0.068 | D |

4 of 5 predictions hold; the one that fails (P4) fails *informatively* — the labelled
Toulmin answer loses because the scorer's marker sets contain connectives
(`therefore`, `however`, `moreover`) and **not** labels (`Claim:`, `Evidence:`).

**Mechanism, read from the source** (`tools/scoring/trace_eval.py`): each dimension is
`1.0 + min(6, distinct_marker_count × weight)` over literal word lists. It counts
*how many different connectives appear*. Nothing checks whether a claim is true.

**So the gate is worse than unmeasured — it is inverted.** It awards an **A** to
`2 + 2 = 5` and a **D** to a correct explanation. Any model optimised toward 0.55 would
be trained to produce confident, well-connected nonsense.

### Provenance of 0.55 (questioning authority)

`0.55` traces to `tools/tm/posttrain.py:60`:
```python
trace_filter: float = 0.55         # only keep TRACE >= threshold (A/B-ish)
```
It is a **training-data filter** — "is this trace good enough to train on" — sitting
between `GRADE_THRESHOLDS = {"A":0.80,"B":0.60,"C":0.42}`. The *release* gate is
`gates.py:67`, `trace > 0.70`. We took a data-filter number and used it as a
**model-quality** bar. That is a category error, and it was never recorded anywhere.

## 2. I also withdraw the "+81% improvement"

`0.140 → 0.253` came from **6 hand-picked probes, one run, no variance**. With a
per-probe std of ~0.08, the 95% CI half-width at n=6 is ±0.064:
`0.140 ± 0.064 = [0.076, 0.204]` vs `0.253 ± 0.064 = [0.189, 0.317]`. They barely
separate. It may be real; it is not established, and I reported it as a headline.

And I tested my own next guess too — that the "improvement" was a length artifact.
Across 350 scored rows, **Pearson r(answer_chars, trace) = 0.127** (r² = 0.016). So
that hypothesis is **wrong**; length explains 1.6% of the variance. Our small models
all produce similar connective density, which is why they all land near 0.11. The
metric is *saturated*, not length-driven.

## 3. The replacement: check correctness, not vocabulary

The operator's point was that "2 + 2 = 5" is trivially checkable — math libraries
solve it, and facts can be checked with search plus a local model. That is right, and
it is also what the fact-checking literature settled years ago.

Built, tested, in `tools/verify/`:

| tool | what it does | self-test |
|---|---|---|
| `math_verify.py` | extracts `A op B = C`, `p% of v`, grouped expressions; sanitises to a strict whitelist; evaluates with sympy; compares symbolically | **15/15** |
| `fact_verify.py` | sentence-level claims → Wikidata (deterministic) → SearXNG + local LLM judge | **11/11** plumbing; 8/8 live |
| `lora_check.py` | LORA pipeline: decompose → check-worthiness → question generation → pairwise verify → **evidence-ratio** aggregation | **16/16** |

`math_verify` catches the exact case TRACE graded A:
```
"Therefore, 2 + 2 = 5."  ->  REFUTED   (sympy, deterministic, 0 tokens)
```

`fact_verify` with a Wikidata channel makes entity claims deterministic:
```
Delaware is located in China.      -> CONTRADICTED  (P17 = United States)
Delaware is in the United States.  -> SUPPORTED
Paris is the capital of Germany.   -> CONTRADICTED  (capitals of Germany = berlin)
Paris is the capital of France.    -> SUPPORTED
```

## 4. Three bugs my own self-tests caught — including one that made things worse

**Bug A — the KG channel was silently dead.** `_wd_labels` selected `?label` while
binding `?v0…?vN`, so Wikidata returned HTTP 400, and a bare `except Exception:
return None` hid it. The whole Wikidata path had never worked. Fixed with
`VALUES ?p {…}` + `rdfs:label` + `FILTER(lang(?label)="en")`, and the error is now
printed instead of swallowed.

**Bug B — unescaped JSON in a prompt.** `DECOMPOSE_PROMPT` contained
`{"claims": [...]}`, so `str.format()` raised `KeyError('"claims"')` at runtime. Every
prompt now has a `.format()` test.

**Bug C — decomposition made the checker WORSE.** This is the important one:

| case | naive checker | first LORA version |
|---|---|---|
| "The sky appears green …" | **CONTRADICTED** ✓ | **SUPPORTED, ratio 1.0** ✗ |
| "Delaware is located in China" | CONTRADICTED ✓ | REFUTED ✓ (via web, KG dead) |

The decomposer collapsed *"The sky appears green because air molecules are larger than
the wavelength of blue light"* into *"The sky appears green."* — **it deleted the false
part**, then the pair judge supported the harmless residual. Decomposition can make a
claim easier to check by removing the thing that is wrong.

Fix: a **decomposition-coverage guard**. If less than 60% of the source's content words
survive into the claims, the densest source sentence is added as an extra claim. After
the fix:

```
F sky green  cov=0.75  counts={'SUPPORTED': 1, 'REFUTED': 1}  -> CONFLICTING (honest)
F Delaware   cov=1.00  wikidata=CONTRADICTED + web=REFUTED
F 2+2        math=REFUTED
F capital    wikidata=CONTRADICTED
T sky blue   counts={'NO_EVIDENCE': 1, 'SUPPORTED': 1}
T Paris      wikidata=SUPPORTED
```
All four false claims are caught; none is confidently supported. The sky-green case is
now reported as **CONFLICTING** — the judge is genuinely split, and saying so is correct.

## 5. What the literature says I still have wrong

I read `Libr-AI/OpenFactVerification`, `GAIR-NLP/factool`, the AVeriTeC pipeline,
`superwesleyhys-ux/factcircuit` and the bullshit-detector skill set. My first version had
**12 documented failure modes**; these are the ones not yet fixed:

1. **No benchmark.** Everything above is 8–16 hand-picked cases. Accuracy is **unknown**.
   AVeriTeC (CC BY-NC), FEVER, SciFact, VitaminC, HoVer, FactKG are the instruments.
2. **No baseline arm.** factool ships `run_self_check_dataset` — "just ask the model",
   0-shot and 3-shot CoT — precisely so a pipeline must beat it. I have no such number.
   (In factool, GPT-4 scores 75.60 claim-level but **43.33 response-level**, because the
   document metric is a conjunction.)
3. **No variance reporting.** A published eval found **~half of claims changed verdict
   across 3 runs of identical content**. I ran my 8 cases 3× and got 0/8 flips — but that
   is 8 easy cases at temperature 0, not evidence of stability.
4. **No origin collapsing.** Five syndicated reprints count as five sources.
5. **No temporal anchor / cutoff admission.** "X is CEO" was true in 2019; evidence
   published after the eval cutoff leaks.
6. **No span validation** of quoted evidence against the frozen source.

## 6. Revised position

- **TRACE is retired as the model-quality gate.** It remains usable as a *structural
  diagnostic* and as a *training-data filter* — but only with a threshold calibrated on
  data, and it must never stand between a good answer and a bad one.
- **Deterministic verifiers beat lexical metrics** wherever a claim is checkable:
  arithmetic (sympy), entity relations (Wikidata). Those two now cover the classes that
  matter most in our corpora and cost **zero model tokens**.
- **The LLM judge is the weak link** and it is not measured. Until it is benchmarked,
  `fact_verify`/`lora_check` produce *signals*, not verdicts.
- **Next, in order:** (1) download AVeriTeC + SciFact, (2) implement the self-check
  baseline, (3) measure claim-level and response-level accuracy with ≥3 runs and a
  bootstrap CI, (4) only then wire any of this into a gate.

## Artifacts

| path | what |
|---|---|
| `tools/scoring/calibrate_trace.py` | the adversarial TRACE calibration (8 fixtures, 5 predictions) |
| `tools/verify/math_verify.py` | sympy arithmetic/algebra checker, 15/15 |
| `tools/verify/fact_verify.py` | Wikidata + SearXNG + local LLM judge, 11/11 self-test |
| `tools/verify/lora_check.py` | LORA pipeline with evidence-ratio aggregation, 16/16 |
| `B:\datasets\forge\eval\trace-calibration.json` | the fixture scores |

## The lesson, again

Two sessions ago I concluded "the data was fine, the metric is the problem". That was
right but incomplete: **the metric was measuring the wrong thing entirely**, and my own
replacement, built without reading the literature, was measurably worse than the naive
version it replaced on the one case that mattered.

> Question the metric. Then question the fix. Then test the fix against the thing you
> replaced.

---

# Part 3 — retrieval was the blocker, and calibration made abstention the default

## The backend was dead, not my queries

SearXNG returned **0 results for every query**, including `"capital of france"`. Every
engine behind it was suspended:

```
brave        Suspended: too many requests
duckduckgo   CAPTCHA
google cse   "unusual traffic from your network"
youtube-cc   403
media-search crash
```

A single-provider retrieval layer converts the whole verification stack into a silent
no-op that reports *"No evidence found"* — which reads as an honest abstention. Worst
failure mode available: plausible and total.

**`tools/verify/retrieval.py`** — provider ladder `wikipedia -> wikidata -> searxng` with
per-process throttle, 429 backoff honouring `Retry-After`, and a disk cache at
`B:\datasets\forge\cache\retrieval`. Self-test includes *"a dead provider is skipped and
the next one still answers"*.

Wikipedia is the workhorse: free, unblocked, no key, and it answered the exact claim
SearXNG could not (`"Give Us Our Skeletons documentary"` → *"is a 1999 documentary film"*).

## Two more fatal bugs, both found by testing the fix

**Query was the verbatim claim.** For a claim about documentaries, Wikipedia returned
*Michael Jackson*, *RoboCop* and *Internment of Japanese Americans*. Replaced with
entity+predicate queries:

```
entities: ['Reagan', 'Give Us Our Skeletons']
queries : ['Reagan documentaries', 'Give Us Our Skeletons documentaries', ...]
```

**Conjunctions judged against one entity's evidence.** *"Reagan and Give Us Our Skeletons
were both documentaries"* was CONTRADICTED — evidence for `"Reagan documentaries"` is about
the **2024 biopic**, not the 2011 documentary. Added atomic decomposition:

```
Reagan is a documentary                 -> SUPPORTED (5 sources)
Give Us Our Skeletons is a documentary  -> SUPPORTED (5 sources)
```

Then the splitter over-fired: *"Paris is the capital of Germany"* has two entities but no
conjunction, and became *"Paris is a capital"* + *"Germany is a capital"*. Now requires an
explicit `and|both|or`.

## Calibration: SUPPORTED must be corroborated

The measured `dangerous = 1.000` (hallucinated answers accepted as TRUE) came from mapping
a single `SUPPORTED` judgement straight to `TRUE`. Fixed:

- judging is now **pairwise** (one call per snippet, capped at 4)
- `SUPPORTED` requires **>= 2 independent supporting snippets**; exactly one → `UNKNOWN`
  with the reason recorded (`"only 1 supporting source (below min_support=2)"`)
- any support *and* any refutation → `UNKNOWN` (`"conflicting 1S/1C"`), never a coin flip

Effect on the same claims:

| claim | before | after |
|---|---|---|
| Reagan is a documentary | CONTRADICTED | **UNKNOWN** (conflicting 1S/1C) |
| Give Us Our Skeletons is a documentary | — | **SUPPORTED** (2 sources) |
| Paris is the capital of Germany | CONTRADICTED (mangled) | **CONTRADICTED** (intact claim) |

Abstention is now the default under uncertainty, which is the correct behaviour for a
gate and the opposite of what it did before.

## Power: Wilson intervals, so n=2 cannot masquerade as a result

`wilson_ci()` added to `eval_factcheck.py` and printed per run:

```
0/2   -> [0.000, 0.658]      n=2 tells you almost nothing
0/40  -> [0.000, 0.088]      n=40 is a result
5/10  -> [0.237, 0.763]      textbook check
10/10 -> [0.723, 1.000]      never "100%"
```

The earlier "accuracy 0.000, dangerous 1.000" was **2 decisions**. The interval is
[0, 0.66]. Reporting the point estimate was wrong; the harness now refuses to let me.

## Bugs I caused while fixing things

- Deleted the **entire Wikidata block** replacing `_query_for` (it sat between
  `_query_for` and `search`). Restored, and that region now carries a warning comment.
- Broke my own CI self-test by expecting Wilson(0,2) > 0.70 when the correct value is
  0.658. The maths was right; the expectation was wrong.

Both are the same lesson: after editing a region, run the self-test for that region.

---

# Part 4 — the judge is fine; the pipeline around it is broken

## The n=20 calibrated run was invalid, and I have to say why

It reported coverage 0.000 on 20 items. That looked like the calibration being too strict.
It wasn't. **14 of the 20 items produced `no_claim`** — the extractor never got a claim:

```
"Which American worldwide automotive industry components and parts suppliers...? Lear Corporation."
```

HaluEval qa is **question + answer**, not a declarative claim. `extract_claims` requires a
declarative sentence with a finite verb, so it correctly refused — and the harness scored
that refusal as abstention.

**So coverage 0.000 was an ADAPTER artifact, not a verifier result.** I was measuring the
wrong thing, and the number was meaningless in both directions.

## The right test: judge given the evidence

HaluEval ships a `knowledge` passage per row. That isolates the actual weak link — the
judge — with no retrieval, no decomposition, no open-web variance:

```python
def knowledge_predict(item, endpoint, model):
    snip = {"title": "given knowledge", "content": item["knowledge"][:1800]}
    return fact_verify.judge_pair(item["claim"], snip, endpoint, model)
```

## RESULT — the first usable measurement in this line of work

n=40 (20 true / 20 hallucinated), 1 run, gpt-oss-20b, temperature 0:

| metric | value | Wilson CI95 |
|---|---|---|
| coverage | 0.875 (35/40) | — |
| **accuracy** | **0.886** | **[0.741, 0.955]** |
| **dangerous** (hallucination accepted) | **0.105** | **[0.029, 0.314]** |

**Given the right evidence, the judge is decent.** 88.6% accuracy on a 35-item sample, with
a lower bound of 74%. It accepts roughly 1 in 10 hallucinated answers.

## What this reframes

| configuration | coverage | accuracy | dangerous |
|---|---|---|---|
| judge + correct evidence | **0.875** | **0.886** | **0.105** |
| full pipeline (retrieval + decompose + judge), fixed retrieval | 0.200 | 0.000 | 1.000 |
| full pipeline, calibrated | 0.000 | — | — |
| baseline (no evidence) | 0.100 | 1.000 *(1 decision)* | 0.000 |

**The judge is not the bottleneck.** The pipeline loses everything in **claim extraction and
retrieval**, and then the aggregation faithfully reports the damage.

That is a completely different diagnosis from "the verifier doesn't work", and it changes
what to build next: stop tuning the judge, fix the claim adapter and the retrieval-to-claim
mapping.

## Honest limits of the new number

- **n=40, one run.** CI on dangerous reaches 0.314, so "about 1 in 10" could be 1 in 3.
- **Evidence was handed to it.** This measures the judge, not the system. A 10.5% dangerous
  rate with *perfect* evidence means the end-to-end rate will be worse.
- **HaluEval is hallucination detection**, not open-web claim verification. AVeriTeC/FEVER
  remain the right instruments for the full pipeline and are still not obtained (AVeriTeC
  gated, SciFact script-only).
- Single judge model (gpt-oss-20b Q4). No second judge, so no inter-judge agreement number.

## Next, now that the diagnosis is right

- [ ] **Fix the claim adapter**: HaluEval q+a must become a declarative claim before the
      extractor sees it (or the extractor must accept q+a). Without this the full-pipeline
      number stays meaningless.
- [ ] **n>=200 on the judge-only test** to tighten `dangerous` below a 0.15 upper bound.
- [ ] **Two judges + agreement**: route disagreements to abstention. A 10.5% dangerous rate
      with one judge should fall with an agreement gate.
- [ ] Re-run the full pipeline only after the adapter fix, and report it against the
      judge-only ceiling of 0.886 — the gap is the retrieval cost.

---

# Part 5 — retrieval v2 (all providers live) and the verifier-training bet

## Search stack status, measured

| component | state |
|---|---|
| SearXNG | **partially back** — 20 results for some queries; brave/google/duckduckgo/youtube still suspended |
| crawl4ai :11235 | **UP** — `/health` 200, `/md` returns 37,358 chars of clean markdown |
| Wikipedia / Wikidata API | **UP**, keyless |
| OpenAlex / Crossref / Stack Exchange / arXiv | **UP**, keyless, own index |
| Marginalia | 429 on the public key — optional, not core |
| trafilatura | **installed 2.3.1** (was the one missing dependency) |
| bs4 4.12 · lxml 5.4 · spaCy 3.8 + en_core_web_sm · httpx · playwright | all present |

**The SearXNG root cause is configuration, not network.** Defaults suspend an engine
**24 h on one CAPTCHA** and **15 days on one Cloudflare hit**
(`SearxEngineCaptcha: 86400`, `cf_SearxEngineCaptcha: 1296000`). One bad minute takes the
whole meta-search tier down for a fortnight. Two config fixes would restore it:

```yaml
search:
  suspended_times:
    SearxEngineTooManyRequests: 300      # 1h -> 5min
    SearxEngineCaptcha: 900              # 24h -> 15min
    cf_SearxEngineCaptcha: 3600          # 15d -> 1h
  formats: [html, json]
engines:
  - name: duckduckgo
    network:
      impersonate: "firefox"             # TLS fingerprint, not IP
```
Not applied: it needs a container restart, and stopping containers is the operator's call.

## retrieval.py v2 — organised by durability, not convenience

```
T1 OWN-INDEX   wikipedia · wikidata · openalex · crossref · stackexchange · arxiv
               keyless, legal, own index, effectively never blocked
T2 METASEARCH  searxng (bonus tier; partial success counts as success)
T3 CRAWLER     crawl4ai /md (JS pages, mild bot walls)   <- fetch, not search
```

All 7 providers verified live, 2 queries each, all returning results.

**New primitive `fetch_page(url)`** — the reason snippets were never enough:
```
tier 1  plain GET + trafilatura   ~50 ms, deterministic, no browser
tier 2  crawl4ai /md f=fit        JS pages, mild bot walls
tier 3  crawl4ai /md f=raw        last resort; noisy but never empty
```
The tier is recorded on every evidence row. Verified: Wikipedia page → 42,273 chars via
trafilatura. `deep_fetch` in `verify_text` replaces the top 2 snippets with real page text
before judging, so the judge sees evidence rather than a teaser.

**spaCy entities replace the regex extractor.** The regex produced
`"Reagan and Give Us Our Skeletons"` as one blob; spaCy returns `"Reagan"` and
`"Give Us Our Skeletons"` separately. That alone changes the query set.

## Repos assessed (ADOPT / SKIP)

| repo | verdict | why |
|---|---|---|
| **crawl4ai** | **ADOPT** (running) | `/md` with `f=fit`/`raw`; escalation tier only. v0.9 refuses per-request proxies (untrusted gate) |
| **Scrapling** | **ADOPT** (parser + `Fetcher`) | `Fetcher(impersonate='chrome')` uses `curl_cffi` TLS/JA3 — the actual DuckDuckGo CAPTCHA fix, since DDG blocks by TLS fingerprint, not IP |
| **trafilatura** | **ADOPT** | the missing dependency; best offline main-content extractor |
| WebSearchFree | adopt **pattern** | its `engines[]` with `ok`/`error`/`response_time` honesty is the best API design in the set; the binary is the same DDG/Brave scraping you're already blocked on |
| llm4free | **SKIP code** | its *LLM* tier calls reverse-engineered consumer endpoints (confirmed ToS-risky); its *search* adapters are ordinary SERP scraping — reference only |
| google-scraper | **SKIP** | needs paid CapSolver + residential proxy by its own README; keep only its plain-HTTP `/trends`, `/patents`, `/news/resolve-link` list |
| gemini-search-mcp | **SKIP** | automates Google's consumer AI Mode UI; its own probes show headless fails and profile priming is required |
| FreeKnowledge_AI | **SKIP** | 15–20× slower than DDG by its own benchmark; needs VPN for DDG + a SiliconFlow key |
| FreedomSearch | **SKIP** | abandoned; claims cache and rate limiting the code does not contain |

**The honest architectural conclusion:** do not build retrieval on a bot-wall bypass. Build
it on own-index APIs (Wikipedia/Wikidata/OpenAlex/Crossref/arXiv/StackExchange) plus
TLS-correct plain HTTP, and treat scraping as an opt-in tier with a hard budget.

## The training bet

HaluEval ships 10,000 rows with a question, a `right_answer`, a `hallucinated_answer`, and
the `knowledge` passage that settles it. That is **20,000 labelled (evidence, claim, label)
triples** — enough to fine-tune a small model to do this one job.

`build_verifier_data.py` → **18,000 train / 2,000 val**, labels balanced
(9,039 CONTRADICTED / 8,961 SUPPORTED), prompt chars p95 = **1,305** so 99.95% fit 512
tokens. Gated through the full cascade: 0 contamination, 1 PII redacted, 0 secrets, CLEAN.

Deliberately **two-way, not three-way**: HaluEval never supplies a merely-unrelated passage,
so a NEUTRAL class would teach the model to hedge on everything. The abstain path belongs in
the inference threshold, not in the training labels.

Training launched on sol: LFM2.5-350M, LoRA r16, 512 seq, 4,500 steps (~33 min).

**The bet, stated so it can fail:** a 350M model trained on 20k labelled pairs beats
gpt-oss-20b (accuracy 0.886, dangerous 0.105, ~25 s/claim) on this narrow task, at ~50 ms
per claim. If it does not, that is also a result — it would mean the task needs scale, and
the honest move is to stop trying to shrink it.

---

# Part 6 — WITHDRAWN: the judge measurement was measuring string matching

I reported earlier today:

> judge given the correct evidence: accuracy 0.886 [0.741, 0.955], dangerous 0.105
> "Given the right evidence, the judge is decent. It is not the bottleneck."

**That conclusion is withdrawn.** I trained a 350M verifier on HaluEval (18k pairs, loss
0.0134, 188 steps/min) and it scored **accuracy 1.000 on 200 held-out rows**. That is a red
flag, not a result, so I checked the dataset instead of celebrating:

```
right_answer   IS a verbatim substring of knowledge : 9,553 / 10,000  (95.5%)
hallucinated   IS a verbatim substring of knowledge :    53 / 10,000  ( 0.5%)
```

The task is **"does the answer appear in the passage?"** — string matching with a 95% base
rate. A two-line rule proves it:

| system (same 40 items) | accuracy | dangerous |
|---|---|---|
| **substring rule (2 lines of Python)** | **0.975** | **0.000** |
| gpt-oss-20b judge | 0.886 | 0.105 |
| LFM2.5-350M trained verifier | 1.000 *(extractive val)* | — |

**The 20B judge is worse than a substring check on this dataset.** Everything measured
against HaluEval qa — the 0.886, the 0.105, and the trained verifier's 1.000 — measures
substring matching. None of it transfers to real claims, where answers are not extractive.

## What survives

- `math_verify` (sympy) and the Wikidata channel are **deterministic and exact**. They are
  unaffected: they check the claim itself, not a passage against an answer.
- `retrieval.py` v2 (7 providers live, `fetch_page` with trafilatura/crawl4ai tiers, spaCy
  entities) is real infrastructure and works.
- The three retrieval bugs (dead backend, verbatim query, conjunction against one entity)
  were real and are fixed.
- The Wilson-CI harness and the "abstention is not accuracy" scoring are correct.

## What does not survive

- The judge accuracy number.
- The trained verifier as a verifier. It learned the shortcut. **Do not ship it.**
- Any claim that HaluEval is a verification benchmark.

## The pattern, three times in one session

1. "Coverage 0.00" → the search backend was dead.
2. "Coverage 0.00 after calibration" → the adapter produced no claims.
3. "Accuracy 1.000 / judge 0.886" → the dataset was extractive.

Each time the number looked like a verdict and dissolved when I checked how it was produced.
The rule that would have caught all three: **before trusting a metric, run the dumbest
possible baseline against the same data.** A two-line substring rule would have exposed
HaluEval immediately.

## Next

See `knowledge/BUDGET-REQUEST-cascade-and-verifier.md`. The job is no longer "fine-tune a
judge" — it is "build data where judging is not string matching". Our own 2,281 harvested
and 19,303 spine rows are non-extractive claims we already own; that is the free starting
point.

---

# Part 7 — the verifier bet FAILED, on a real benchmark

## Setup

AVeriTeC (GitHub raw, free) → 4-way verifier data:
- **train 3,067** — SUPPORTED 848 / REFUTED 1,742 / CONFLICTING 195 / NOT_ENOUGH_EVIDENCE 282
- **dev 500** — REFUTED 305 / SUPPORTED 122 / CONFLICTING 38 / NEI 35
- Extractiveness **0.0% dev / 0.1% train** (vs HaluEval 95.5%) — a real instrument
- Gated: **the gate caught 1 TruthfulQA contamination inside AVeriTeC train**, plus 8 PII redactions. 3,066 rows CLEAN.

Trained LFM2.5-350M, LoRA r16, 512 seq, 2,300 steps, 12.8 min, loss 0.071.

## Baselines first (the lesson from HaluEval)

| baseline | accuracy | CI95 |
|---|---|---|
| always-majority (`REFUTED`) | **0.610** | [0.567, 0.652] |
| substring rule | 0.608 | [0.565, 0.650] |
| random over 4 labels | ~0.25 | — |

The substring rule **collapsed from 0.975 on HaluEval to 0.608 here** — exactly chance.
That confirms AVeriTeC measures verification and HaluEval measured matching.

## Result

| model | accuracy | CI95 | macro-F1 | ms/claim |
|---|---|---|---|---|
| **350M verifier** | **0.245** | [0.191, 0.309] | 0.242 | 3,599 |

**−0.360 versus the majority baseline. It is worse than doing nothing.**

Confusion on the 200-row eval slice:

```
gold REFUTED   (121): SUPPORTED 77 · CONFLICTING 27 · REFUTED 4
gold SUPPORTED  (50): SUPPORTED 33 · CONFLICTING 14 · NEI 1 · REFUTED 2
```

It almost **never predicts REFUTED** — the opposite of the majority class. It did not
learn the trivial prior, let alone the task.

## Why it failed (three candidate causes, ranked)

1. **512-token truncation eats the evidence.** AVeriTeC prompts are ~1,600 chars of
   evidence plus the claim plus the template — roughly 500-600 tokens, so truncation
   removes the tail of the evidence. The model may be training on partial evidence and
   inferring from the claim alone.
2. **3,067 training examples is small** for 4-way with a 57/28/6/9 imbalance.
3. **350M may simply be too small** for 4-way claim verification with abstention.

Loss 0.071 on train versus 0.245 accuracy on dev is textbook memorisation: it fit 3,067
rows and generalised to nothing.

## What this establishes

- **HaluEval was the wrong dataset** and AVeriTeC is the right one. Confirmed by baseline.
- **A 350M model cannot do this task** from 3k examples at 512 tokens. That is a real,
  useful negative result — it means "shrink the judge" was the wrong instinct.
- **The deterministic channels are far weaker than I claimed.** They settled **2 of 3,067**
  claims (0.07%). I had said "Boolean and Extractive answers are what sympy and Wikidata
  can settle exactly" — wrong: `answer_type` describes the *evidence answer format*, not
  whether the claim is arithmetic or an entity relation. The Wikidata patterns only cover
  "X is in Y" and "X is the capital of Y", which almost no AVeriTeC claim matches.

## What to try next, in order

1. **Fix truncation first** — 2,048 seq so the evidence fits. This is the cheapest
   possible change and the most likely cause. (Costs ~5 steps/min on the Arc, so ~2 h.)
2. **Then scale the model** to LFM2.5-1.2B before concluding the task needs a frontier model.
3. **Widen the deterministic channels** — the current Wikidata patterns are far too narrow.
   A general entity-relation extractor plus SPARQL would cover far more than 0.07%.
4. **Do not re-run at 350M/512.** The result is unambiguous.

## The honest summary of this whole line of work

Four measurements today, each of which dissolved when checked:

1. judge accuracy 0.886 → **the dataset was 95.5% extractive**
2. trained verifier 1.000 → **it learned substring matching**
3. coverage 0.00 → **the adapter produced no claims**
4. 350M on AVeriTeC 0.245 → **worse than the majority baseline**

The one thing that survived every check is the **deterministic channels and the
infrastructure** (retrieval ladder, gate cascade, baselines harness, Wilson CIs). The
models have not earned a place in the pipeline yet.

---

# Part 8 — the failure was CODE, not model capacity (operator was right)

> Operator: *"the consistency of the failures points to bad code"*

Correct, and it was the right call. I had concluded "350M is too small for this task". That
conclusion was premature. The diagnostic signature I missed:

- accuracy **0.245**, macro-F1 **0.242** — *exactly* chance over 4 classes
- it **never predicted REFUTED** despite REFUTED being 57% of training — the opposite of
  the majority prior
- **`loss 0.071` reported by the trainer while per-row loss was 4.58–5.24**

A capacity-limited model still learns the majority prior. A model at exact chance with an
*inverted* bias is a broken input→output mapping. The 0.071-vs-4.9 contradiction was the
smoking gun and I should have caught it before writing "the bet failed".

## The bug

`train_arc.py` built training text by hand:

```python
f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
```

LFM2.5's **native** template is:

```
<|startoftext|><|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n
```

So every model we trained (v6, harvested, HaluEval, AVeriTeC) was trained on a chat format
the base model had **never seen**, with **no `<|im_end|>`**, so it never learned to stop.
Observed generations were degenerate repetition:

```
'CONFLICTING.SUPPORTED.SUPPORTED.SUPPORTED'
'REFREFREFREFREFREFREFREFREF...'
'ocumented immigrants in modern modern modernmo'
```

It was echoing my own label list back at me.

## The fix

`train_arc.py` and `eval_averitec.py` both now use `tok.apply_chat_template(...)`, so
training and inference share the model's own format and `<|im_end|>` is present.

## Measured effect (100-row overfit test, 400 steps)

| | accuracy @ 50 rows |
|---|---|
| hand-built `<|user|>` template | 0.270 |
| **native `apply_chat_template`** | **0.400** |

It is learning now. 400 steps on 100 rows is not enough to fully overfit, so this is
directional, not conclusive.

## Still unexplained

The trainer reports `loss ≈ 0.097` in **both** runs, while true per-row loss is ~4.9. The
reported loss is not the training objective I think it is — most likely TRL is masking most
tokens and computing loss over very few. **Do not trust `train_arc.py`'s loss number until
that is resolved.** Accuracy is the only signal that has been reliable.

## What this invalidates

- **"350M is too small"** — withdrawn. Not established.
- **The 0.245 AVeriTeC result** — produced by a model trained on a broken format. It
  measures the bug, not the architecture.
- **Every prior training run** used the same hand-built template, including the v6 run that
  produced TRACE 0.253. Those numbers are suspect for the same reason and should be re-run.

## The methodological lesson (fourth time today)

| symptom | what I first concluded | what it actually was |
|---|---|---|
| judge 0.886 | "the judge is fine" | dataset 95.5% extractive |
| verifier 1.000 | "it works" | substring matching |
| coverage 0.00 | "calibration too strict" | adapter produced no claims |
| accuracy 0.245 | "350M too small" | **hand-built chat template** |

The rule that catches all four: **when a model performs at exactly chance, suspect the
plumbing before the capacity.** And: *if two measurements of the same thing disagree
(trainer loss 0.07 vs per-row loss 4.9), stop and find out why before reporting either.*

---

# Part 9 — state at handoff (2026-10-22)

## Fixed and verified
- `train_arc.py` + `eval_averitec.py` now use the model's native `apply_chat_template`
- 100-row overfit test improved 0.270 -> 0.400 at 50 rows with the fix

## OPEN — the loss anomaly is NOT resolved
`train_arc.py` reports `train_loss ≈ 0.097` while true per-row loss measured ~4.9, in BOTH
the broken-template and fixed-template runs. Two attempts to measure
full-sequence vs completion-only loss locally **timed out**: CPU forward passes on ~500-token
sequences with a 350M model take minutes each, so 15 rows x 3 passes does not fit a shell
window.

**Do this on the GPU, not on CE:**
```python
# for 15 rows, on sol inside the unsloth image
pid = tok(apply_chat_template(...), return_tensors="pt").input_ids
cid = tok(completion, add_special_tokens=False).input_ids
ids = cat([pid, cid], 1)
lab = ids.clone(); lab[:, :pid.shape[1]] = -100
full = model(input_ids=ids, labels=ids).loss      # all tokens
comp = model(input_ids=ids, labels=lab).loss      # completion only
```
- if `comp ≈ 0.097` and argmax accuracy is low → TRL's number is fine, the bug is elsewhere
- if `comp ≈ 4.9` → **TRL's reported loss is bogus and must not be trusted**

**Until this is answered, treat `train_loss` from `train_arc.py` as meaningless.** A wrong
loss curve is how a dead run looks healthy — which is exactly what happened here: 0.097
looked like excellent convergence and it was measuring almost nothing.

## Re-run list (all invalidated by the template bug)
1. 100-row overfit with the fix, enough steps to reach ~1.0 (proves the pipeline works)
2. AVeriTeC train -> dev, 4-way, against the 0.610 majority baseline
3. v6-sft and harvested — their TRACE numbers used the same hand-built template

## Do not repeat
- Do not conclude "the model is too small" from a run that sits at exactly chance.
- Do not report a loss without checking it against a hand-computed per-row loss.

---

# Part 10 — the method was wrong, not the model (operator: "why can't you beat a pair of dice")

> Operator: *"figure out why you can beat a pair of dice youre clearly doing something wrong"*

Correct on both counts. Two independent errors, one a bug and one a wrong task shape.

## Error 1 — the harness bug (this is why 0.245 looked like chance)

AVeriTeC dev label prevalence:

```
Supported 24.4% · Refuted 61.0% · Conflicting 7.6% · Not Enough Evidence 7.0%
```

**My reported accuracy was 0.245.** That is the `Supported` prevalence to three decimal
places. Generations were degenerate (`'REFUTED.SUPPORTED.SUPPORTED'`) and my label parser
matched **any** label string found in that text, longest-first — so it reported a label
with no relation to the model's intent. **I never measured the model at all.**

Fix, and the rule: every generation must parse to exactly one of 4 labels or the run
**fails loudly**. No silent default. Log a confusion matrix and per-class F1 from run #1.
At 61% majority, accuracy alone cannot distinguish "learned something" from "collapsed".

## Error 2 — verdict prediction is classification, not generation

The field does not generate label strings below 7B. Measured, on the same gold evidence:

| system | AVeriTeC dev |
|---|---|
| AVeriTeC official baseline — BERT + Linear(4) + if-chain rules | **0.49** |
| AVeriTeC full pipeline (retrieved evidence) | 0.15 @ lambda=.25 |
| BLOOM-7B *generating* labels | macro-F1 **.43** (vs BERT-large .49) |
| **DeBERTa-v3-large + Linear(4), multi-evidence, argmax** | **0.63** |
| DeBERTa-v3-large, averaged over 10 shuffled QA orders | **0.65** |
| human annotator agreement ceiling | **0.714** |

Generation adds three failure modes classification cannot have: *surface-form competition*
(probability mass smeared over `Refuted`/`REFUTED`/`false`/`incorrect`), a generic-string
attractor, and decoding failure where no prediction exists at all.

**A 435M encoder with a 4-way head beats the entire official pipeline by 14 points.**

## The ceiling nobody tells you

AVeriTeC inter-annotator agreement is free-marginal kappa 0.619 → for 4 classes
`P_o = 0.619*0.75 + 0.25 = 0.714`. **Two trained annotators given the same evidence agree
only 71.4% of the time.** So 0.63 is within 8 points of human, and anything above ~0.66 at
sub-1B on dev would be publishable. Do not budget for more.

## What I built

`tools/verify/train_verifier_cls.py` — sequence classification, self-test 5/5:
- `microsoft/deberta-v3-base` + `Linear(4)`, argmax, no generation
- input is multi-evidence `[CLS] claim [SEP] q1 a1 [SEP] q2 a2 …` — **not** per-pair
  classification with if-chain rules (that is the 0.49 baseline, and it turns any single
  bad pair into a wrong verdict)
- **class-weighted CE** — measured weights `[0.71, 0.51, 3.85, 2.78]`, so NEI is upweighted
  3.85x and Conflicting 2.78x
- label order asserted against AVeriTeC `eval.py`
- reports macro-F1 and per-class F1, with the baselines printed next to the result

## Launch state — HUNG, not finished

The training container ran 17 minutes with 2 log lines and produced nothing. Most likely
stuck on the DeBERTa download through the container's HF cache mount. **No result yet.**

Next: pre-pull `microsoft/deberta-v3-base` into `/home/zcoder/hf-cache` explicitly, confirm
it is present, then re-run. Do not treat the hang as a model result.

## Also worth adopting from the research

1. **Average logits over 10 shuffled QA orders** — free +2 points (0.63 -> 0.65 published).
2. **`ctu-aic/deberta-v3-large-AVeriTeC-nli`** is a released checkpoint that already scores
   0.63. Use it as a reference arm; if our trained model cannot match it, that is the answer.
3. **`reverify`'s zero-weight rule set** (trivial / duplicate / echoed / known /
   self-referential → weight 0) is directly portable as a QA-pair information gate, and it
   neutralises a real AVeriTeC pathology: the shared task found that *submitting duplicate
   questions raises the score*, because the metric rewards duplicate evidence.
4. **Report the AVeriTeC score, not accuracy.** Verdict accuracy on gold evidence is a
   different number from the leaderboard.
5. **Do not do pair-level NLI + if-chains.** AIC CTU tried training 3-way DeBERTa for exactly
   that and reported "without a breakthrough result, motivating a radically different
   approach", then recommended omitting the step entirely.

## The rule from this one

**Check the label distribution before interpreting an accuracy.** 0.245 was not a bad model;
it was the prevalence of one class, which means the predictor was constant. A five-second
`collections.Counter(gold)` would have caught it before I wrote "the bet failed".

---

# Part 11 — what the 2026 literature says about the plan (and where it disagrees)

## The single most important number I found

**ClaimCheck (arXiv:2510.01226): Qwen3-4B with a modular pipeline scores 76.4% on AVeriTeC**,
beating LLaMA-3.1-70B and GPT-4o baselines.

And **arXiv:2608.25934**: replacing retrieved evidence with **gold evidence buys +14 to +22
points** across models, and *claim-only* models can beat zero-shot LLM pipelines. Their
conclusion: **"retrieval remains the primary bottleneck."**

So the 0.63 DeBERTa figure is **evidence-limited, not model-limited**. Every point of effort
spent on the model head is worth less than effort spent on evidence quality. That reorders
the whole plan.

## The distillation idea — right instinct, wrong order

Operator idea: distill DistilBERT into a reasoning decision model with a decision/JEV-like
head, then into a 0.6B Qwen.

What the evidence actually supports:

| finding | source | consequence |
|---|---|---|
| Distilling Step-by-Step gives **+0.5 to +6.0 points** on 220M–770M T5 | arXiv:2305.02301 | real, but measured *against label-only finetuning of the same model* |
| **A classifier is a better labeller than an LLM; an LLM is a better explainer than a classifier** | ARG, arXiv:2309.12247 (AAAI 2024) | **split the roles** — gold labels, LLM rationales |
| Rationale **quality dominates quantity** (+4.6% from selection, fewer rationales) | MoRSD, arXiv:2509.23574 | filter hard; drop 30–50% |
| **A retrieve-then-verify harness cannot be distilled into a small student** — representational, not a tuning problem (NL ⊄ TC⁰) | arXiv:2610.27157 | do not try to compress the pipeline into 0.6B |
| **In-Writing order** — free justification first, then one constrained verdict token — beats natural generation by up to 27% | arXiv:2601.07525 | this is the right output shape |

**Do not distill the 0.63 classifier's labels.** At 0.63 against a 0.61 majority baseline it
carries ~2 points of signal plus its errors; distilling it teaches the student the errors.
Use the **gold** labels and get rationales from a strong teacher.

**Expected outcome: 0.60–0.66 — at or below the classifier.** The whole task spans 0.61
(majority) to 0.714 (human agreement). A 0.6B generator cannot find 5 points that a 435M
classifier cannot. The generator buys **auditability**, not accuracy.

## The frame-logic question — honest answer

Of the seven families cited (world models, concept graphs, neuro-symbolic, latent-space, RL
agents, memory-augmented, cognitive architectures), **five have never been applied to
natural-language claim verification.** No MuZero/DreamerV3, no Cyc, no Logic Tensor Networks,
no SOAR/ACT-R/OpenCog/Sigma result exists on this task.

The one that *has* worked is **KG-as-retrieved-evidence**: FactKG (arXiv:2305.06590),
KG-GPT (arXiv:2310.11220), FactGenius (arXiv:2406.01311), SimGRAG (arXiv:2412.15272),
AgentKGV (arXiv:2607.09092). Note what it does — it **retrieves** a structured subgraph and
serialises it into the prompt. It does not inject an ontology into attention.

**"OWL2 ontology injected as a vector map with a dedicated concept attention head" — no
evidence, and one controlled comparison argues against it:** arXiv:2605.22641 (EMNLP 2026
Findings) compared early fusion, late fusion and cross-attention fusion of retrieved
knowledge and found **"early fusion outperforms both trainable variants."** The elaborate
attention mechanism is the part that does not pay.

**FrameNet as an intermediate representation for verification:** exactly one paper ever tried
it (SURFACE, arXiv:1810.13223, 2018), against baselines from a era when FEVER accuracy was in
the 50s–60s. Never replicated, never part of a competitive system. The current frame-semantics
benchmark (FrameBench, arXiv:2609.03370) finds **"small models struggle"** with frames at all.

**Verdict:** frames are a good *diagnostic vocabulary* (which is why `logos` uses them for
syllogistic validity) but there is **no evidence** they beat plain text as a verification
input. Do not build the attention head.

## Deterministic thinking — what actually works in 2026

| method | source | result |
|---|---|---|
| **ILP/logic verifier in the loop**, rewrite the offending step and regenerate | Reason Popper-ly, arXiv:2607.23019 | **up to +48 points for small models** on multi-hop |
| **Neurosymbolic primitive programs** (reasoning as an explicit DSL program) | Forethought, arXiv:2607.04096 | ~+30% relative; small models match frontier |
| **Process (step-level) reward** | arXiv:2607.02869 | Qwen2.5-0.5B GSM8K **63.73% vs 53.75%** outcome-only |
| **RLVR alone at tiny scale** | arXiv:2606.22189 | 135M model *decreased* 1.82% → 1.59% — negative |
| **Constrained single-label decoding** | Koa-decision, arXiv:2609.36115 | 85.5% on intent routing, 7.5× faster — good for the *verdict* token |
| **Verifier certificates** | arXiv:2610.09541 | mostly **abstain** at realistic calibration; a certificate is not a correctness guarantee |

The pattern: **the deterministic checker does the work; the model supplies grounded content
the checker cannot compute.** That is exactly the shape our pipeline already has (sympy,
Wikidata) — it just needs to be wider and actually fire.

## Decision-model API — free, useful, with two real caveats

`https://decision.vllm-sr.ai/keys` is the **vLLM Semantic Router "Decision API"**: six typed
models that answer with **calibrated probabilities and generate zero tokens**.

| model | params | Decision Index (text) | free limits |
|---|---|---|---|
| d3-turbo | 27B | 70.2 | 540/min · 648k/day |
| d3-flash | 9B | 55.79 | 720/min · 432k/day |
| d3-mini | 4B | 52.61 | 1,260/min · 756k/day |
| d3-nano | 2B | 38.86 | 1,440/min · 864k/day |
| d3-lite | 0.8B | 32.33 | 1,920/min · 1.15M/day |
| **d3-edge** | **0.6B** | **20.24** | 3,840/min · 2.3M/day |

**Caveat 1 — logging is on and cannot be turned off.** Their docs say logged data "may be
stored and used to… train and evaluate models. No request header turns logging off." So
**never send unpublished AVeriTeC dev claims or anything sensitive** through it.

**Caveat 2 — the 0.6B number is the reality check.** `d3-edge` at 0.6B scores **20.24** on
their own index versus 55.79 for 9B. A 0.6B typed decider is weak in general. And the
independent audit (arXiv:2609.32160, 28 papers) concludes: **"the typed readout itself has not
shown an independent accuracy advantage… Jev's clearest gains are in latency and cost."**

**How to use it:** as a cheap, fast, well-calibrated **deferral gate** — "is this claim worth
a full verification pass?" — not as a better classifier.

## Revised order of work

1. **Finish the classifier** (deberta-v3-base + Linear(4), already built) and get a real
   number. Target 0.63; the bar to beat is 0.61.
2. **Then attack evidence quality, not the model.** +14–22 points live there. That means
   wider deterministic channels (sympy + a general entity/relation → SPARQL extractor), not a
   better head.
3. **Then** the explainer: gold labels + LLM rationales, filtered by span-grounding and
   entailment, trained in In-Writing order (justification first, then one constrained verdict
   token). Expect 0.60–0.66 and treat it as an *explainer cross-checked against the
   classifier*, never as a replacement.
4. **Do not** build the OWL2 attention head, do not distil the pipeline into 0.6B, do not
   expect frames to beat plain text.

## Keys

- **Decision API**: free, no account, worth having for the deferral gate. Get one, but keep
  unpublished eval claims off it.
- **OpenRouter free tier**: we already hold the $10 top-up → 1,000 req/day, one shared bucket.
  Enough for the packed judge (~50 calls for the whole corpus).
- **No paid key is needed for step 1 or 2.** The classifier trains on the Arc, and evidence
  work is deterministic. Revisit payment only if step 2 stalls.

---

# Part 12 — both tracks: deterministic channel widened, classifier bugs fixed, and one structural finding

## Track B — the deterministic channel: built, and it does not fit AVeriTeC

`tools/verify/wikidata_relations.py` — 35 relations, spaCy entity spans, auditable verdicts.

Two real bugs found and fixed by the self-test:
1. **Six property directions were backwards.** "X is the capital of Y" means *Y* has capital
   *X*, so P36 lives on the OBJECT. Same for head-of-state (P35), CEO (P169), founded-by
   (P112), owns (P127), author (P50), director (P57), producer (P162), cast (P161).
2. **The P31 pattern over-matched.** `r"\bis (?:a|an|the)?\s*(?:kind of|type of)?\b"` has
   every group optional, so it matched a bare `" is "` and won the first-match race against
   every more specific relation — "Paris is the capital of France" extracted as P31.

Both fixed; self-test now 10/10 including explicit direction assertions.

### Then the structural finding

**Coverage on 50 AVeriTeC dev claims: 0.0.** Only 1 triple extracted in total.

Diagnosing it showed this is not a code bug — it is a **dataset/claim-style mismatch**:

```
"In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial."
"Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled..."
"UNESCO declared Nadar community as the most ancient race in the world."
"Republican Matt Gaetz was part of a company that had to pay 75 million in hospice fraud."
```

AVeriTeC claims are **narrative events with multiple clauses**. They are not
`subject --property--> object` facts. Wikidata stores *entity properties*, not *event
narratives*. No number of relations will make "Sean Connery refused to appear in an apple
commercial" a P-number.

**So: the deterministic channel needs a different dataset.** FactKG (arXiv:2305.06590,
108k claims, 5 reasoning types, **gold relation paths** as evidence) is built for exactly
this. That is where the Wikidata channel fires, and it is already on the operator's list.

Correct division of labour:
| channel | dataset | why |
|---|---|---|
| sympy + Wikidata relations | **FactKG** | claims ARE `S --relation--> O` with gold paths |
| DeBERTa classifier + retrieval | **AVeriTeC** | claims are narrative; needs evidence retrieval |

That is a genuine redirect, not a failure: I had the right tool pointed at the wrong corpus.

## Track A — the classifier: three code bugs, now training in fp32

| bug | symptom | fix |
|---|---|---|
| class weights left in fp32 while the model ran fp16 | `RuntimeError: expected scalar type Half but found Float` | cast weights to `logits.dtype` |
| DeBERTa-v3 fp16 numerical instability | **`eval_loss: NaN`**, every prediction collapsed to class 0 → accuracy 0.244 = the `Supported` prevalence | force `fp32` + `.float()`; 86M is cheap in fp32 |
| (earlier) hand-built `<|user|>` chat template | degenerate generation | `apply_chat_template` |

**Note the pattern repeating:** the NaN collapse produced *exactly* 0.244 again — the
`Supported` prevalence. That number has now appeared three times from three different bugs,
which is why "check the label distribution first" is the rule.

Training relaunched in fp32: 3,072 steps, ~1.6 it/s, ~25 min remaining at handoff.

## What is left, in order

1. **Let the fp32 classifier finish** and read `eval_acc` / `eval_macro_f1` / per-class F1.
   Bar to beat: **0.610**. Published reference: DeBERTa-v3-large **0.63**; human ceiling
   **0.714**.
2. **Point the deterministic channel at FactKG**, not AVeriTeC. That is where it can fire.
3. **Retrieval for AVeriTeC** — the +14–22 point lever, per arXiv:2608.25934.
4. Do not add more relations to fix AVeriTeC coverage. The mismatch is structural.

## The rule, now confirmed five times

| number | what it looked like | what it was |
|---|---|---|
| judge 0.886 | the judge works | dataset 95.5% extractive |
| verifier 1.000 | training works | substring matching |
| coverage 0.00 | calibration too strict | adapter produced no claims |
| accuracy 0.245 | 350M is too small | hand-built chat template |
| accuracy 0.244 (again) | — | **fp16 NaN, all predictions to one class** |

**Check the label distribution and the confusion matrix before interpreting any accuracy.**
0.244 is now a recognised fingerprint of a collapsed predictor.

---

# Part 13 — THE CLASSIFIER WORKS: 0.78 acc / 0.613 macro-F1

## Result (AVeriTeC dev, deberta-v3-base + Linear(4), fp32, 8 epochs)

```
eval_loss       2.594
eval_acc        0.780      majority 0.610 · published DeBERTa-v3-large 0.63 · human ceiling 0.714
eval_macro_f1   0.613      published macro-F1 0.39
per-class F1    Supported .737 · Refuted .865 · NEI .571 · Conflicting .277
```

**+17 points over the majority baseline, +15 over the published sub-1B reference, and above
the human annotator agreement ceiling (0.714).**

Three code bugs had to be fixed to get here — a broken chat template, fp16 NaN collapse, and
a dtype mismatch. None of them were model-capacity problems. The "350M is too small"
conclusion was wrong twice over.

### What plausibly produced the gain over the published 0.63
1. **Multi-evidence concatenation** in one example rather than per-pair classification plus
   if-chain aggregation (the if-chain is the 0.49 baseline; its Conflicting precision is 10%
   by the authors' own admission).
2. **Class-weighted CE.** Measured weights `[0.71, 0.51, 3.85, 2.78]` — NEI upweighted 3.85x,
   Conflicting 2.78x. This is most of the macro-F1 jump (0.39 -> 0.613) and lifts the two
   classes a 2-point-margin model would otherwise ignore.
3. fp32 instead of fp16.

### The caveat I do not yet have the answer to
**0.78 exceeds human agreement (0.714).** That is either a real result or a shortcut. Two
candidate shortcuts, one of which is being tested as this was written:

- **Isolation test (RUNNING):** `CLAIM_ONLY=1` drops the evidence and trains on the claim
  alone. If accuracy stays near 0.78, the model is memorising claims, not verifying them.
- **Annotator-phrasing leak:** the QA answers were written by annotators who had read the
  fact-check article, so wording like "published on [a fake news site]" telegraphs REFUTED.
  A model can learn that without verifying anything. Mitigation: strip/paraphrase the
  answers, or evaluate on freshly retrieved evidence rather than gold QA pairs.

**Do not publish the 0.78 until the claim-only ablation lands and the annotator-phrasing
leak is tested.** A number above the human ceiling demands the strongest possible scepticism.

## The three references

### arXiv:2604.09166 — NOT relevant (false friend)
"Automated **Batch Distillation** Process Simulation for a Large Hybrid Dataset for Deep
Anomaly Detection" — *Comput. Chem. Eng.* 215 (2026) 109827. "Batch distillation" is a
**chemical** unit operation, not model distillation. The paper generates time-series data
from a process simulator for chemical anomaly detection. Nothing transferable.

### ACL 2026 long 908 — HIGHLY relevant, and it names our exact risk
**"Distillation Traps and Guards: A Calibration Knob for LLM Distillability"** (Zhan et al.,
ACL 2026, pp. 19814–19833).

It identifies three **distillation traps**: **tail noise**, **off-policy instability**, and the
**teacher–student gap**. They manifest as *overconfident hallucinations, self-correction
collapse, and local decoding degradation*, which is why distillation "fails unpredictably".

**This maps onto us precisely:**

| their trap | our situation |
|---|---|
| **tail noise** | Conflicting F1 = **0.277**, NEI = 0.571, against Refuted 0.865. **That is the tail**, and it is exactly what a teacher will pollute. |
| **off-policy instability** | our plan was reverse-CoT (rationales conditioned on the gold answer) — textbook off-policy. |
| **teacher–student gap** | we planned to put a 4-way evidence task into 0.6B. The paper's whole point is that this gap is the failure mode, not a tuning detail. |

Their proposed guard — post-hoc calibration via RFT combining task utility + KL anchor +
tokenizer calibration — is for LLM teachers at scale, but the **diagnostic framing is
immediately usable**: before distilling, measure whether the *teacher* is distillable by
checking tail-class agreement, not just overall accuracy.

**Action:** before any distillation, report per-class agreement between teacher and student on
the two tail classes. If Conflicting agreement is near zero, the distillation will produce a
confident 3-class model that guesses on the fourth.

### model2vec — YES, usable, and cheap enough to be obviously worth it
`MinishLab/model2vec` (MIT, 2.3k stars). Distils any sentence transformer into **static
embeddings**: up to **50x smaller and 500x faster**, small quality drop, and distillation
takes **~30 seconds on CPU with no dataset** — it just forward-passes the vocabulary.

Directly useful in three places:

1. **`StaticModelForClassification`** — fine-tune a static-embedding 4-way classifier on our
   claim+evidence text and run it **without torch**. If it approaches DeBERTa's accuracy it
   becomes the fast tier of the cascade (and a strong candidate for the ONNX/llama.cpp
   deployment lane in `GOLDEN-MASTER-PLAN.md` tier 2).
2. **`potion-retrieval-32M`** as a **retrieval pre-filter** — 500x faster candidate selection
   before the expensive judge. Given the literature says retrieval is the +14–22 point lever,
   a cheap high-recall pre-filter is directly on the critical path.
3. **Dedup at scale** — static embeddings over the 41k-document corpus for semantic near-dup
   detection, cheaper than the current MinHash LSH pass (789 s for 19.5k rows).

License is MIT, dependencies are numpy only. No reason not to try it.

## Next, in order

1. **Read the claim-only ablation.** If it is near 0.78, the result is memorisation.
2. **Test the annotator-phrasing leak** — shuffle or paraphrase the QA answers and re-measure.
3. **Then** model2vec: train a static classifier as the fast tier and compare to DeBERTa.
4. **Then** distillation, with tail-class agreement measured first, per ACL 2026 long 908.

---

# Part 14 — model2vec results: a static model that beats the published encoder at 0.17 ms/claim

## Measured, same data, same 4-way labels, same dev split

| system | dev acc | macro-F1 | speed | notes |
|---|---|---|---|---|
| majority (`always Refuted`) | 0.610 | — | — | the bar |
| published DeBERTa-v3-large + head | 0.630 | 0.390 | — | Ullrich et al. 2024 |
| **model2vec static (potion-base-32M)** | **0.644** | **0.415** | **0.17 ms/claim** | **179 s fit, no torch** |
| **deberta-v3-base + Linear(4)** | **0.780** | **0.613** | ~100 ms/claim | ~20 min, torch, GPU |

Per-class F1 (static): Supported .436 · Refuted .763 · NEI .462 · **Conflicting .000**

## What this means

**The static model beats the published DeBERTa-v3-large reference — 0.644/0.415 vs
0.63/0.39 — at 0.17 ms per claim with no torch and a 179-second fit.**

That is a genuine engineering result, not a curiosity. For a pipeline that has to judge
thousands of rows, 0.644 at 0.17 ms is a better answer than 0.78 at 100 ms for the *fast
tier*, and the two-tier design writes itself:

```
tier 1  model2vec static      0.17 ms/claim, no torch     <- everything
tier 2  deberta-v3-base       0.78 acc                    <- only where tier 1 is unsure
```

If tier 1 is confident on 80% of rows at 0.644 and tier 2 handles the rest at 0.78, the
blended number beats either alone at a fraction of the cost. That is the cascade the
golden standard describes, and model2vec makes the cheap tier almost free.

**Conflicting collapses to F1 0.000** in the static model — the tail again, and the same
class ACL 2026 long 908 warns is the first casualty of any distillation or compression. A
static model has no capacity for a 5.7%-prevalence class that requires reading two
contradictory sources. Expected, and it bounds what the fast tier can be trusted with: it
should never be the final word on a Conflicting verdict.

## Where model2vec fits the toolset beyond this

1. **Retrieval pre-filter** — `potion-retrieval-32M` at 0.17 ms/claim over the 41k-document
   corpus. The literature says retrieval is the +14–22 point lever (arXiv:2608.25934); a
   pre-filter that costs nothing puts the expensive reranker only on plausible candidates.
2. **Semantic dedup** — replaces the MinHash LSH pass that took 789 s for 19.5k rows.
3. **The ONNX/llama.cpp lane** — `GOLDEN-MASTER-PLAN.md` tier 2 asks for sub-1B judges
   deployable without torch. A model2vec classifier is exactly that.

Install was `pip install "model2vec[train]"`, MIT, numpy-only in the base package.

## Still open — the claim-only ablation

At 67% when this was written. It answers the only question that matters about the 0.78:
does the classifier still score near 0.78 with the **evidence removed**? If yes, it is
memorising claims and the result is an artifact. Nothing gets published before that lands.

## Files
- `tools/verify/train_verifier_static.py` (self-test 5/5)
- `B:\datasets\forge\verifier-av\static-model\result.json`
