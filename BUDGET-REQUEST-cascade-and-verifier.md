# BUDGET REQUEST — cascading cleaning + a verifier that measures verification

**Date:** 2026-10-21 · **App key:** `pool-model-zoo` · **Requested by:** opencode
**Status:** AWAITING OPERATOR GREEN-LIGHT (Law: nothing paid runs without a quote)

---

## 0. Why this request exists at all — and why the number is small

I was about to ask for a budget to fine-tune a verifier. Then I checked the dataset and
the request collapsed. Recording that first, because it changes the ask.

### The finding that killed my last result

HaluEval qa looked ideal: 10,000 rows, each with a question, a `right_answer`, a
`hallucinated_answer`, and the `knowledge` passage. I built 18,000 training pairs from it,
trained LFM2.5-350M, and it scored **accuracy 1.000** on 200 held-out rows.

That was a red flag, not a win. Measured:

```
right_answer   IS a verbatim substring of knowledge : 9,553 / 10,000  (95.5%)
hallucinated   IS a verbatim substring of knowledge :    53 / 10,000  ( 0.5%)
```

So the task is **"does the answer appear in the passage?"** — string matching with a 95%
base rate. And a two-line rule proves it:

| system (same 40 items) | accuracy | dangerous |
|---|---|---|
| **substring rule (2 lines of Python)** | **0.975** | **0.000** |
| gpt-oss-20b judge | 0.886 | 0.105 |
| LFM2.5-350M trained verifier | 1.000 *(extractive val)* | — |

**The 20B judge is worse than a substring check.** My "the judge is fine at 88.6%"
conclusion from earlier today is withdrawn. The trained verifier learned the shortcut and
is worthless for real claims, where answers are not extractive.

That is the third time this session that checking *how* a number was produced dissolved it
(dead backend → verbatim queries → conjunction against one entity → extractive dataset).

### Consequence for the budget

The expensive thing is **not** judging. It is **building data where judging is not string
matching**. That is a generation problem, and generation is free at the tiers we have.
So the ask is small — **$5 ceiling, $1.50 expected** — and most of it is a spot-check.

---

## 1. What the cascade will be (local first, paid last)

Follows `GOLDEN-MASTER-PLAN.md` (the Quality Ladder) and `design-patterns.md` Pattern 1
("run the free tiers first").

| tier | stage | method | cost | volume |
|---|---|---|---|---|
| **L0** | structural | `prose_validator` + `decontaminate_text` (13-gram vs 40,303 refs) + MinHash dedup + PII | **$0** | 22,442 rows, full |
| **L1** | deterministic truth | `math_verify` (sympy) + Wikidata SPARQL | **$0, exact** | every numeric/entity claim |
| **L2** | local judge | our 350M/1.2B on sol's Arc, or DMR `lfm2.5-1.2b` at :12434 | **$0** | full corpus |
| **L3** | thinkingcap rewrite | condense CoT traces, rewrite the rejects — free pool, batch | **$0** | ~15% of rows |
| **L4** | free cloud judge | Gemini Flash-Lite 150K RPD / OpenRouter 1,000/day, batch mode, `{"keep":"y"\|"n"}` only | **$0** | full corpus |
| **L5** | paid spot-check | 5% audit of what L2–L4 passed, frontier model, batch | **paid, capped** | ~1,100 rows |
| **L6** | human | operator review of the top slice | attention | ~140 rows |

L0–L4 are free and carry the volume. **L5 is the only paid tier and it is an audit, not a
worker** — exactly as `GOLDEN-MASTER-PLAN.md` tier 3 describes ("5% audits + gate-model
training labels").

---

## 2. The non-extractive training set (the actual job)

To retrain the verifier so it measures verification, the training data must break the
shortcut. Three ways, cheapest first:

| approach | what it does | cost |
|---|---|---|
| **a. Paraphrase the answers** | rewrite `right_answer` so it is not a substring of the passage; the claim then requires entailment, not matching | free pool |
| **b. Generate claims from evidence** | give a model the passage and ask it to state a claim the passage supports (and one it refutes), then check deterministically | free pool |
| **c. Use our own corpora** | harvested agent traces + spine already contain non-extractive claims; label with L1 deterministic channels + L2 | **$0** |

**(c) first, then (a).** Our own 2,281 harvested rows and 19,303 spine rows are real claims
with no extractive shortcut. That is free training signal we already own.

Target: **20,000 non-extractive pairs**, built mostly from (c) + (a), generated on the
Gemini free lane (150K RPD → one day, $0).

---

## 3. Call and token estimate

| stage | calls | in-tok | out-tok | tier |
|---|---|---|---|---|
| L2 local judge (22,442 rows) | 22,442 | — | — | local GPU, $0 |
| L3 rewrite (~15% = 3,366) | 3,366 | 3.4M | 1.7M | free |
| L4 free judge (22,442) | 22,442 | 22M | 0.9M | free (150K RPD) |
| claim generation (20,000) | 20,000 | 20M | 4M | free (150K RPD) |
| **L5 paid audit (5% ≈ 1,122)** | **1,122** | **0.9M** | **0.02M** | **paid** |

`max_tokens` is **not** truncated on free lanes — calls are billed per-request, not
per-token, and truncation starves thinking models into empty content
(`PROVIDER-RATE-LIMITS.md:83-90`). Judge prompts are single-JSON `{"keep":"y"}` at
temperature 0.

---

## 4. Cost, at real prices

**L5 audit, two options:**

| model | price /M | 0.9M in + 0.02M out | note |
|---|---|---|---|
| `deepseek/deepseek-v4-flash` | $0.064 / $0.129 | **$0.06** | sufficient for an audit |
| `openai/gpt-5.6-terra` | $1.00 / $6.00 | **$0.92** | frontier, for the disputed tail only |

Batch APIs are **50% cheaper** and bypass per-request rate limits, so the realistic L5 cost
is **$0.03–$0.46**.

**Total request:**

```
Expected:   $1.50
Ceiling:    $5.00      (hard stop, enforced by BudgetTracker)
Paid lanes: deepseek-v4-flash first, gpt-5.6-terra ONLY on the disputed tail
Free lanes: Gemini Flash-Lite 150K RPD, OpenRouter 1,000/day, AIHubMix 20RPM/50RPD
```

Compare with the pre-existing quote in `GREENLIGHT-BUDGET-AND-GOLDENDB.md`: $0 primary,
$25 paid fallback, $35 ceiling. **This request is 5× smaller** because the deterministic
channels (sympy, Wikidata) and our own corpora carry most of the work.

---

## 5. What I am requesting specifically

1. **Access to `pool-model-zoo` for this run**, leased via `tools/tm/key_lease.py`
   (checkout → run → checkin with spend). I will not touch `LITELLM_MASTER_KEY`.
2. **A $5.00 hard ceiling**, `--hard-budget 1.50` per paid pass, so a bug cannot drain it.
3. **Models on my key** (no additions requested): `deepseek/deepseek-v4-flash`,
   `openai/gpt-5.6-terra`, plus the existing free lane models.
4. **Batch mode enabled** for L4/L5 (50% cheaper, no per-request limits).
5. **Smoke gate before any batch >10**: 5 requests, must pass content non-null AND
   format-valid on ≥4/5 (`free-first-rotation-v1.yaml:27-32`). I will not launch a batch
   that fails the smoke.

---

## 6. What I will NOT do

- Will not read, pass, or fall back to the master key.
- Will not call a model that is not on `pool-model-zoo` without asking first.
- Will not truncate `max_tokens` on a free lane to "save" anything — it costs a call either
  way and starves the model.
- Will not run concurrent sweeps against the same key (that trips a 401 + 300 s cooldown).
- Will not report a dataset as clean without a `gate.json` whose `decontam_self_test` is
  `n/n`.
- Will not use HaluEval again as a verification benchmark. It is an extractive dataset and
  it made a 2-line substring rule look like a 97.5% verifier.

---

## 7. Open questions for the operator

1. **Green-light the $5 ceiling?** Or hold me to $1.50 with a re-ask before the audit?
2. **Which paid model for the audit** — `deepseek-v4-flash` ($0.06, likely enough) or
   `gpt-5.6-terra` ($0.92, frontier)? My recommendation: flash for the bulk audit, terra
   only on disagreements.
3. **Is there a non-extractive verification set already on disk** that I have not found?
   AVeriTeC is gated (401) and SciFact's claims are script-only. If you have either, that
   removes the 20,000-call generation job entirely.
4. **Gemini free lane for bulk generation** — confirm the 150K RPD tier is on my key, since
   that is what makes the generation step a one-day $0 job rather than a 20-day one.

---

## 8. Honest expectation

The best evidence I have is that this task is harder than it looks and my measurements keep
dissolving. So the stated expectation is modest:

- **Likely:** a non-extractive verifier that beats the 2-line substring rule on
  non-extractive data, at ~50 ms/claim instead of 25 s.
- **Possible:** it still underperforms a frontier judge, in which case the right answer is
  to keep the judge as an *audit* and rely on the deterministic channels (sympy, Wikidata)
  for the claims they can settle — which is most of what our corpora contain.
- **Certain:** the free tiers cover the volume. The paid spend is an audit, and if the
  audit disagrees with the free tiers more than 5% of the time, the pipeline is wrong and
  I will report that rather than tune the threshold until it agrees.
