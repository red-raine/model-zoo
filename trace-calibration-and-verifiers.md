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
