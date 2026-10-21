# The zero-token trace factory — verified status, and three defects

**Date:** 2026-10-21 · **Owner:** opencode · **Engine:** `E:\dev\entropy_systems\dataset-builders\`
**Plan:** `C:\Users\myste\.commandcode\plans\conceptnet-fuzzy-frames-fingerprints.md` (11.1 KB, 2026-08-23)
**Pointer:** `C:\Users\myste\.zcode\msg\bitnet-runner\20261021-1000-zcode-frames-plan-pointer.md`

---

## 1. What I verified (not what the pointer claims)

The pointer said "build status unknown". It is now known. I ran the engine.

| claim in the plan | verified status |
|---|---|
| **Part A** — ConceptNet wired via shared `kb_utils.py` | **BUILT.** `kb_utils.py` exists (8.5 KB), index at `data/conceptnet/cn_en.tsv` = **43.3 MB**, built 2026-08-24 by `build_cn_index.py`. NLTK dropped its ConceptNet corpus, hence the local TSV. |
| **Part B** — frame detection upgraded to cue 0.5 / evidence 0.3 / availability 0.2 | **BUILT.** Confirmed in source: `confidence = 0.5*cue_score + 0.3*_kb_evidence_score(...) + 0.2*kb_fraction` |
| **Part C** — `build_complex_cot()` KB-driven planner | **BUILT and RUNS.** `reasoning_engine.py:1310`. Produced 3,336 chars with **zero model calls**. |
| **Part D** — trace fingerprints / StyleGuide | modules exist (`trace_fingerprint.py` in the broader tree); not exercised this session |

**So the engine is real.** The plan is not aspirational — A, B and C are implemented and executable.

## 2. But the output was noise, and I found why

First run, on a cron-job/concurrency scenario:

```
conceptnet: DerivedFrom:Onglaze, AtLocation:Resturant, CapableOf:Shine_light, DerivedFrom:Samesies
wordnet:    a.n.06, a.n.07, adenine.n.01, ampere.n.02, angstrom.n.01
frame:      Disjunctive Syllogism  conf=0.40  cues_found=['or']
```

Two defects, both small, both severe:

### Defect 1 — `extract_concepts` had no stopword filter

`wn.synsets("a")` returns `angstrom.n.01`, `ampere.n.02`, `a.n.06`, `a.n.07` — WordNet's
senses for the *letter* a. Every `a`, `or`, `in`, `the` in the input was being looked up
and answered with its trivial sense. A text about cron jobs therefore produced
`adenine.n.01`.

**Populated-but-noisy is worse than empty**, because it looks like grounding.

### Defect 2 — `detect_frames` matched cues as unescaped substrings

```python
if re.search(cue, text, re.IGNORECASE):   # cue "or" matches "cORrupt"
```
The cue `or` fired inside "c**or**rupt", which is how a factual scenario "detected"
Disjunctive Syllogism at confidence 0.40.

### Fixes applied (backup: `reasoning_engine.py.bak-20261021`)

1. `content_tokens()` — drops stopwords and tokens ≤2 chars; used by `extract_concepts`.
2. Plain-word cues are wrapped in `\b…\b`; genuine regex cues pass through untouched.

### Measured effect

| metric | before | after |
|---|---|---|
| trace generation time | **49.65 s** | **0.00 s** |
| concepts for the cron text | `adenine.n.01`, `Antonym:nand`, `AtLocation:Resturant` | `Lemma:Job`, `runner.n.10`, `DerivedFrom:Microtask`, `DerivedFrom:Index`, `HasA:Sixty_seconds_in` |
| false Disjunctive-Syllogism frame | fired (conf 0.40) | gone |
| frames found | 2 (both false) | **0** |

The 50 s → 0 s win is the same fix: the stopword lookups were doing thousands of
WordNet/ConceptNet queries per call.

## 3. Defect 3 — exposed by the fix, not yet fixed

With the false frames gone, **zero frames fire** on a factual scenario, and
`build_complex_cot` degrades badly:

```
Step 2: Before diving in, let me reflect on the reasoning approach (reasoning).
Step 2.1: The syllogistic form is: ''.        <- empty
```

It emits an empty syllogistic form instead of falling back to a factual/causal frame.
Arguably frame detection is now *correct* (a factual scenario is not a syllogism), so the
bug is in the fallback, not the detector.

**Decision needed before fixing:** which frame should a factual scenario get? Options:
a factual/causal frame (new), a causal-chain frame built from ConceptNet `Causes` edges,
or "no frame → emit a plain structured answer without syllogistic scaffolding". This is a
design choice, not a bug fix, so it is recorded rather than guessed.

## 4. Why this matters for the zoo

This engine is the only **deterministic trace factory** in the family — content derived
from a KB, zero model tokens at generation time. That is exactly what the ablation said we
need: Wave 9's prompt-diversity problem (87 → 10k prompts) is currently solved by spending
teacher tokens. A working KB planner would generate diversity for **$0 and no quota**.

But it can only be consumed once its grounding is trustworthy. Feeding the pre-fix output
to a student would teach it that cron jobs live in restaurants.

## 5. Status for the plan / todo

- [x] Verified A, B, C are built and executable (not aspirational)
- [x] Defect 1 (stopword concepts) fixed + measured
- [x] Defect 2 (substring cue match) fixed + measured
- [ ] **Defect 3** — no-frames fallback emits empty syllogistic form; needs a design decision
- [ ] **Quality gate on generated traces** — a zero-token trace must pass the same
      `gate.py` contract as any other corpus before it can be trained on
- [ ] **Scale test** — 3,336 chars/trace at 0.00 s means bulk generation is now cheap;
      measure 1,000 traces and check diversity + decontamination
- [ ] **Wire into Wave 9** — this is a prompt-diversity generator that costs no tokens;
      it should be generator #6 alongside mining, Evol-Instruct, PersonaHub, DreamCoder,
      contrastive pairs

## Artifacts

| path | what |
|---|---|
| `E:\dev\entropy_systems\dataset-builders\reasoning_engine.py` | fixed (2 defects) |
| `...\reasoning_engine.py.bak-20261021` | pre-fix backup |
| `...\kb_utils.py` | ConceptNet local index (43.3 MB TSV) |
| `...\build_cn_index.py` | index builder |
| `C:\Users\myste\.commandcode\plans\conceptnet-fuzzy-frames-fingerprints.md` | the spec |
