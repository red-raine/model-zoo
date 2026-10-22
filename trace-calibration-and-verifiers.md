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
