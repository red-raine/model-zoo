# Closing the provenance gap: sources registry, Wayback dating, and the field I had wrong

**Date:** 2026-10-24
**Modules:** `logos/sources.py` (11/11), `logos/facts.py` (16/16)
**Result:** usable-for-dated-claims corpus went from **5% → 35%** (2/43 → 19/54)

---

## 1. The registry: 121 sources across 29 categories

`logos/sources.py` catalogues where a verifier can check something, chosen for **niche coverage**
rather than the famous five. Each entry records what it holds, whether it carries dates, how it is
accessed, and its licence — so "scale" never silently means "scrape everything".

| | |
|---|---|
| sources | **121** |
| categories | **29** |
| carrying dates | **100 (83%)** |
| keyless-accessible | **83** |

Every category has at least two sources, which is a test rather than a claim. Categories include
`knowledge_graph`, `encyclopaedic`, `archive`, `news_archive`, `academic`, `medical`, `statistical`,
`climate`, `geoscience`, `biology`, `chemistry`, `legal`, `financial`, `patents`, `factcheck`,
`geography`, `transport`, `infrastructure`, `sport`, `music`, `film`, `books`, `company`,
`conflict`, `disaster`, `demographic`, `education`, `lexical`, `nlp`.

The infrastructure entries matter for the bridge facts this project keeps using: `structurae`,
`skyscraperpage`, and CTBUH's Skyscraper Center all carry construction and completion dates.

## 2. Wayback Machine: dates the citation, not the fact

`wayback_dates(url)` queries the CDX API for the **earliest and latest** capture of any URL. Fixed
a real bug here: the first version used `limit=-1`, which returns only the *last* record, so
"earliest capture" of Wikipedia's syllogism page came back as **2026-10-16**. Two cheap calls
(ascending `limit=1`, and `sort=reverse limit=1`) give the truth: **2003-12-09 → 2026-10-16**.

`resolve_date(fact_date, ref_url)` returns a **list of evidence, strongest first**, each naming its
source and strength:

| source | strength | means |
|---|---|---|
| `point_in_time` (P585) | publisher | when the value was measured |
| `wayback_first` | archive | earliest capture of the cited URL |
| `retrieved` (P813) | — | **when a Wikidata editor checked. Not a fact date.** |

## 3. The field I had wrong

The original gate used `point_in_time or retrieved`. Measured on real facts:

```
Hausmannsturm in Hartenfels Castle
   retrieved  = 2024-10-15     <- the old gate read this as the fact's date
   wayback    = 2019-07-21     <- the page demonstrably existed in 2019
```

**A `P813` "retrieved" date is an upper bound on availability** — it is when someone *looked*, not
when the fact *existed*. Treating it as the fact's date rejects facts that are perfectly usable, and
it is wrong in the conservative direction, which hid the error.

Corrected: precedence is established from the **earliest evidence of existence**
(`min(point_in_time, archive_first)`). A bare retrieval date is explicitly *not* evidence of
precedence, and the rejection message now says so:

> *"only a retrieval date (2023-11-29), which is when the source was CHECKED, not when the fact
> existed; it cannot establish precedence"*

## 4. The actual fix: harvest only referenced statements

Wayback dated **4 of 4** cited facts. But 39 of 43 facts had **no citation at all**, so no date
resolver could help them. **A date resolver dates citations; it cannot manufacture them.**

`harvest(..., referenced_only=True)` adds `FILTER(BOUND(?refURL))`:

| harvest | facts | with ref URL | gate-kept (2022 claim) |
|---|---|---|---|
| unfiltered | 43 | **4 (9%)** | 2 (5%) |
| **referenced-only** | **54** | **54 (100%)** | **19 (35%)** |

**Seven times more usable facts**, because the corpus now only contains facts that *can* be dated.
Trade volume for availability — the only trade that helps a verifier of dated claims.

## 5. The gate is now demonstrably correct, not just strict

From the sample, the kept and dropped facts are individually right:

| fact | archive | verdict |
|---|---|---|
| Luiz I Bridge 45 m | 2015-03-16 | **KEEP** |
| Crimean Bridge 80 m | 2017-08-03 | **KEEP** |
| Skye Bridge 35 m | 2017-08-16 | **KEEP** |
| **1915 Çanakkale Bridge 334 m** | **2022-02-06** | **drop** — the bridge did not exist for a 2022-01-01 claim |
| Eğiste Hadimi Viaduct 166 m | 2022-11-26 | **drop** |

That Çanakkale row is the gate earning its place: a structurally perfect fact, correctly excluded
because the structure postdates the claim. This is the AVeriTeC temporal trap, caught automatically.

Rejection reasons are now specific: 13 no archive capture, 6 retrieval-date-only, 1 genuinely
postdating.

## 6. Honest state

- **35% of a referenced-only harvest survives a 2022 claim.** Better than 5%, still a majority
  rejected. The residual is archive coverage (13 with no capture) and the retrieval-only rows.
- **121 sources catalogued, 1 integrated.** Wayback is the only resolver wired in. The registry is a
  map; wiring more is deliberate work, not a claim of coverage.
- **Reasoning-trace datasets** (`Fable-5.1-Max-Reasoning-Filtered-10000x`, `Primus-Instruct`,
  OpenMathReasoning, OpenCodeReasoning, KodCode-V1, OpenThoughts3-1.2M) are all *trace* corpora.
  They become filterable through the ontology layer and checkable through a per-domain verifier —
  and only the numeric verifier exists today. Numeric first, then math via sympy, then code via
  execution.

## 7. Files

| file | self-test | what |
|---|---|---|
| `logos/sources.py` | 11/11 | 121-source registry, Wayback CDX resolver, date cascade |
| `logos/facts.py` | 16/16 | corrected temporal gate, referenced-only harvest, class + plausibility gates |

## 8. Next

1. **Wire a second resolver** — `commoncrawl` and `archive.today` for the 13 URLs Wayback has no
   capture for.
2. **Scale the referenced-only harvest** by loosening the class constraint to more property/class
   pairs, now that the filter preserves quality.
3. **Numeric-verifier trace filter** — the one domain where a reasoning corpus can be checked today.
