# Real facts with provenance, and the ontology layer

**Date:** 2026-10-24
**Modules:** `logos/ontology.py` (15/15), `logos/facts.py` (13/13)
**Data:** 43 real Wikidata facts at `B:\datasets\forge\facts\` → 258 constructed rows

---

## 1. Fixing the two problems

The previous corpus reached 1.000 on **template** facts with **no temporal gate**. Both are now
addressed.

### Provenance

`logos/facts.py` harvests real entity-attribute-value triples from Wikidata via SPARQL, carrying:

| field | source |
|---|---|
| `qid` | the entity URI |
| `ref_url` | `pr:P854` on the statement's reference node |
| `retrieved` | `pr:P813` (retrieved-at date) |
| `point_in_time` | `pq:P585` qualifier |
| `fetched_at` | when **we** fetched it — provenance about the provenance |

Real harvest (bridges, towers, dams, cities): **43 facts**, e.g. *Taizhou Yangtze River Bridge
height 192 m*, *Kőröshegy viaduct 88 m*, *Kanmon Bridge 140.8 m*.

### The temporal gate

`temporal_gate(fact, claim_date)` rejects any fact whose source is dated **after** the claim.
Measured against a claim dated 2022-01-01:

```
2 kept, 41 rejected
   38   no source date
    3   source postdates claim
```

**88% of the harvested facts cannot be used for a dated claim, because Wikidata has no date for
them.** That is the real provenance gap and it is the honest headline: the gate works, and what it
reveals is that the source data is thinner than assumed. 5 of 43 statements carry any date at all;
4 carry a reference URL.

## 2. The harvest was garbage before the class constraint

The first query was `?s wdt:P2048 ?v` with no class filter. It returned:

```
Fisher        height  -1193 m
Ringtail      height  -1191 m
Dormouse      height  -1191 m
Beebalm       height  -1280 m
```

Negative heights on entities named after Nevada Test Site subsidence craters. **A property is not
a fact filter — the class is.** Fixed with `?s wdt:P31/wdt:P279* wd:Q12280` (bridge, transitively)
plus a plausibility gate:

| gate | rejects |
|---|---|
| non-positive value | −1193 m |
| outside `[0.5, 5000] m` for length, `[100, 5e9]` for count | 98 000 m |
| no human-readable label (falls back to QID) | unlabelled entities |

A malformed fact in a training corpus is worse than a missing one, so the gate drops rather than
repairs.

## 3. Does the corpus survive real facts?

| fact source | rows | slot-aligned + compose | presence detector |
|---|---|---|---|
| template facts | 60 | **1.000** (+0.500) | 0.483 (−0.017) |
| **real Wikidata facts** | **258** | **0.973** (+0.473) | 0.473 (−0.027) |

Per-kind on real facts:

| kind | presence | slot + compose |
|---|---|---|
| K1 easy support | 1.00 | 0.95 |
| K2 split support | 0.50 | 0.98 |
| K3 refute | 0.84 | 0.93 |
| K4 wrong attribute | **0.00** | **1.00** |
| K5 wrong entity | **0.00** | **1.00** |

**The drop from 1.000 to 0.973 is the measure of how much the template phrasing was doing.**
2.7 points — small, which is reassuring, and real, which is why the experiment was worth running.
The remaining failures are on real entity names and real magnitudes inside the same reasoning
pattern.

The traps remain perfect and the presence baseline remains *below* the majority with **0.00 on
both trap kinds**, so the corpus is diagnostic on real data too.

## 4. The ontology layer

The operator's proposal — *pass the logical frames as an ontology to a model and ask it to write
according to it, or validate whether output matches* — is the strongest architectural idea of the
session. Implemented as `logos/ontology.py`.

An ontology here is **executable**, not prose:

```
Ontology(fields={name: Field(type, required, enum, min, max)},
         rules=[(name, predicate, message)],
         examples=[(valid, payload)])
```

`rules` are Python predicates, so a constraint is checkable rather than aspirational. A rule that
raises is reported as a violation, not a crash — a checker must not die on malformed input.

### The pair is the point

`render_prompt(onto)` and `validate(onto, obj)` are two halves of one idea:

- a **prompt without a validator** is a hope
- a **validator without a prompt** only sorts garbage

Together they let a tiny model be useful *and* auditable, and they let a big model be checked
rather than trusted. The rendered prompt carries every constraint **and** an explicit refusal
instruction:

> 3. If you cannot satisfy a constraint, output `{"refuse": true, "reason": ...}` instead of guessing.
> 4. A refused answer is acceptable. A wrong answer that looks conforming is not.

### Three ontologies ship

**`categorical_syllogism`** — two premises, three terms, middle term shared and absent from the
conclusion, mood matching the quantifiers. This catches the **fallacy of four terms**, the
commonest equivocation error, as a term-count violation rather than a judgment call.

**`span_evidence`** — a decisive span label requires a comparable quantity. This is the R5/R6
lesson encoded as a constraint: no `refute` without a slot-aligned comparison.

**`machine_checked_proof`** — `compiled AND NOT has_sorry`. This is the Lean `sorry` hole encoded
as a rule, so the difference between "compiled" and "verified" is a test rather than a comment.

## 5. On the datasets the operator named

`OpenMathReasoning`, `OpenCodeReasoning`, `KodCode-V1`, `OpenThoughts3-1.2M` are all large
reasoning-trace corpora. The ontology layer makes them **filterable rather than merely usable**:
traces can be admitted only where their final answer is checkable (sympy for math, execution for
code, the quantity layer for numerics), and where the trace's shape conforms to the ontology.

That is a real path to "collect the stuff we know is good and can prove it" — but it needs a
verifier per domain, and only the numeric one exists here. The honest sequencing is: numeric
first (works today), then math via sympy, then code via execution. Reasoning-trace filtering with
an unverified boundary is just a bigger corpus with the same defect.

## 6. Where this lands relative to the ambition

The operator's framing: *the LLM and the internet validate what we can't, and we validate what
they can't — together as good as a human with Google.*

That is now a concrete division of labour rather than a hope, and each side is measured:

| side | what it contributes | measured |
|---|---|---|
| **our deterministic layer** | slot-aligned comparison, composition, temporal gate, ontology conformance | 0.973 on constructed real-fact corpus; perfect on traps |
| **the neural layer** | evidence relevance, narrative claims, recall on AVeriTeC | 0.808 dev, but **not relevance-discriminating** |
| **Wikidata/Wikipedia** | ground facts with provenance | 43 facts, 4 with URLs, 5 with dates |

The gap that remains is not architectural — it is **provenance density in the source data** and
**relevance discrimination in the neural layer**. Both are addressable, and neither is addressed
by a bigger model.

## 7. Files

| file | self-test | what |
|---|---|---|
| `logos/ontology.py` | 15/15 | executable ontologies, validator, prompt renderer, 3 shipped ontologies |
| `logos/facts.py` | 13/13 | Wikidata harvest with provenance, class constraint, plausibility gate, temporal gate, cache |

## 8. Next

1. **Raise provenance density** — harvest statements that *have* references (`FILTER(BOUND(?refURL))`)
   instead of taking whatever comes back, and prefer properties whose statements are referenced.
2. **Filter a reasoning-trace corpus on the numeric verifier** — the only domain where the boundary
   is checkable today, and the demonstrated way to add data without adding noise.
3. **Feed ontology conformance into the factor graph** so a non-conforming generation is a
   discountable event rather than a silent pass.
