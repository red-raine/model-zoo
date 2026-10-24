# The source registry at 197, three resolvers, and the reproducibility bug

**Date:** 2026-10-24
**Modules:** `logos/sources.py` (15/15)
**Result:** registry 121 → **197 sources / 40 categories**; gate-kept **35/54 (65%)**, now reproducible

---

## 1. The registry, expanded with the operator's list

The operator supplied ~100 sources across five taxonomies (fact-checking networks, authoritative
reference, niche academic search, history/geography, OSINT). All are now in the registry.

| | before | after |
|---|---|---|
| sources | 121 | **197** |
| categories | 29 | **40** |
| carrying dates | 100 | **156 (79%)** |
| keyless-accessible | 83 | **128** |

New categories: `factcheck_registry`, `media_bias`, `image_verification`, `geospatial`,
`computational`, `investigative`, `search`, `cyber`, `osint`, `history`, `reference`, `patents`,
`financial`, `conflict`, `disaster`, `demographic`, `education`.

Every category has **at least two** sources — asserted as a test, because "niche coverage" is
otherwise a claim with no content. Five categories arrived with a single entry
(`image_verification`, `geospatial`, `computational`, `investigative`, `search`) and were filled.

Entries worth calling out for this project's actual needs: `structurae`, `skyscraperpage` and
CTBUH for structure dates; `chroniclingamerica` and `trove` for archival newspapers;
`sanborn`-class map archives (`loc_maps`, `david_rumsey`, `old_maps_online`) for historical
geography; `slavevoyages`, `avalon_project`, `wilson_center` for primary documents; and
`suncalc` / `mapillary` / `invid` / `exiftool` for media forensics.

## 2. Three resolvers — and two contributed nothing

`resolve_date` now cascades **publisher → Wayback → Common Crawl → archive.today**, reporting
which source supplied each date and its strength.

Measured on the 54 referenced-only facts:

| resolver | facts dated |
|---|---|
| `wayback` | **45 / 54 (83%)** |
| `commoncrawl` | **0** |
| `archive_today` | **0** |
| undated | 8 |

**Two of three resolvers produced zero dates on this sample.** Common Crawl only indexes URLs it
crawled, and these are obscure structure pages; archive.today has no public API and its
`/timemap/` endpoint returned nothing usable. Both are implemented, both fail gracefully, and
neither is worth wiring deeper without evidence it helps this corpus.

That is the honest shape of the win: **Wayback alone accounts for all of it**, and the value of
the other two is unproven.

## 3. The bug that mattered: results were not reproducible

Running the same 54 facts repeatedly gave:

```
run 1   25 dated    19 gate-kept
run 2   47 dated    35 gate-kept
run 3   49 dated    37 gate-kept
```

**archive.org rate-limits.** A corpus built from uncached lookups changes between runs, and
provenance that changes between runs is not provenance. Fixed with a disk cache keyed by URL.

Then a second bug inside the fix:

```python
hit = _cache_get(url, "wayback")
if hit is not None:          # cached MISS is None -> indistinguishable from "not cached"
    return ...
```

`None` did double duty as "not in cache" and "cached as a miss", so every miss was **retried on
every run** — which is why the numbers drifted from 47 to 49 *with the cache supposedly warm*.
Fixed with a `_CACHE_ABSENT` sentinel; a cached miss is now a real, honoured value.

Three consecutive runs after the fix: **35 / 35 / 35**. Deterministic.

## 4. Where the provenance work now stands

| stage | measure |
|---|---|
| unfiltered Wikidata harvest | 4 / 43 facts carry a reference URL (9%) |
| **referenced-only harvest** | **54 / 54 (100%)** |
| Wayback dates them | 45 / 54 (83%) |
| **survive a 2022 claim date** | **35 / 54 (65%)** |

From 5% usable to **65% usable**, and now reproducible. The remaining 19 are 8 undated pages and
11 whose archive capture postdates the claim.

## 5. Honest limits

- **197 sources catalogued, one integrated.** The registry is a map. Wiring each needs its own
  client, key, and rate-limit handling; claiming "197 sources" as coverage would be false.
- **Common Crawl and archive.today are unproven** on this corpus.
- **The trace corpora** (`Fable-5.1-Max-Reasoning-Filtered-10000x`, `Primus-Instruct`,
  `IFM/Code-Reasoning`, OpenMathReasoning, OpenCodeReasoning, KodCode-V1, OpenThoughts3-1.2M) are
  still unprocessed. Only the **numeric** verifier is mature enough to filter them today; math via
  sympy and code via execution are the sequenced next domains.

## 6. Next

1. **Scale the referenced-only harvest** across more property × class pairs — the filter preserves
   quality, so volume is now safe to raise.
2. **Numeric-verifier trace filter** — the one domain where a reasoning corpus can be checked.
3. **Wire `resolve_date` into `facts.harvest`** so a fact arrives dated rather than being dated
   later by a separate pass.
