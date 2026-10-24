# Redundancy: 3³ access paths, and the measured ceiling on it

**Date:** 2026-10-24
**Modules:** `logos/redundancy.py` (12/12), `logos/sources.py` (15/15) → **209 sources / 41 categories**
**Data:** 24 real facts, 9 routes × 3 retries each

---

## 1. The federation answer does not exist

The canonical way to get archive redundancy is Memento TimeTravel, which federates 25+ archives
in one request. Probed 15 candidate archives against one URL:

| endpoint | result |
|---|---|
| **wayback CDX** | **WORKS** `["20031209141920", "20040506202930"]` |
| **arquivo.pt CDX** | **WORKS** memento list, keyless |
| **arquivo.pt timemap** | **WORKS** Memento protocol |
| `timetravel.mementoweb.org` | **DNS does not resolve** — the aggregator itself is dead |
| commoncrawl index name | 404 |
| archive.today | DNS does not resolve |
| loc-webarchive | 403 |
| swap.stanford.edu | 502 |
| bibalex / haw.nsk.hr | timeout |
| vefir.is / digar.ee | 404 |
| padi.cat | SSL certificate failure |
| trove | 410 Gone |

**3 of 15.** So redundancy cannot be bought by listing archives. It has to be bought by finding
independent routes to the *same fact*.

The dead endpoints are now in the registry tagged with what the probe found — because **a dead
endpoint counted as a backup is worse than no backup: it inflates the count.**

## 2. The 3³ structure, built

For one fact, three **tiers**, each with three **routes**, each retried three times — 27 attempts:

| tier | routes | what it is |
|---|---|---|
| **A** the statement itself | P585 point-in-time · P813 retrieved · revision id | publisher's own claim |
| **B** the citation's server | `Last-Modified` · `Date`/`Age` · ETag hints | the host's own state |
| **C** third-party archives | Wayback CDX · arquivo.pt CDX · arquivo.pt timemap | independent witness |

`quorum_dates()` runs tier C in parallel with retries and merges all three, returning the
per-tier dates, the consensus, the **spread in days**, and a strength label.

**Agreement is counted per tier, not per route.** Three routes inside the archive tier may be one
crawl observed three times, so they count as *one witness*. That is the difference between
redundancy and the appearance of it.

## 3. The measured ceiling

24 real facts through all 9 routes:

| agreement | facts |
|---|---|
| **single-tier** | **16** |
| two-tier | **3** |
| **three-tier** | **0** |
| none | 5 |

**Zero facts reached three independent witnesses, and two-thirds had exactly one.** Coverage was
19/24 dated (79%), gate-kept 13/24 (54%).

This is the real ceiling: **redundancy is bounded by how many genuinely independent routes exist,
not by how many we list.** For a 2015 archive capture of a Portuguese-indexed bridge page, there
is one witness on earth, and adding 24 more endpoint names does not create a second.

## 4. And the one "second witness" was often wrong

The two-tier cases exposed the problem:

| fact | tier B (Last-Modified) | tier C (archive) | spread |
|---|---|---|---|
| L'Isle-Jourdain Viaduct | 2021-06-09 | 2021-12-29 | 203 d |
| **Talbrücke Olbersdorf** | **2025-05-31** | **2016-08-07** | **3219 d** |

`Last-Modified` is a **server render date**, not a content date. On a Wikipedia page it returned
**today**. Two fixes:

1. A `Last-Modified` within **30 days of now** is rejected outright — it is a render/revalidation
   timestamp, and accepting it would let a brand-new page date an old claim, which is the exact
   failure the temporal gate exists to prevent.
2. `strength` now means **agreement**, not mere response. Two tiers nine years apart report
   **`conflicted`**, not `two-tier`. Dressing a conflict up as corroboration is the failure mode
   this whole module is supposed to prevent.

## 5. What is actually true about redundancy here

The ambition was 27 backups per fact. What the measurement says:

- **Built:** 9 routes × 3 retries = 27 attempts per fact, with per-tier quorum, spread detection,
  conflict labelling, retry, and disk caching.
- **Achieved:** 1–2 independent witnesses for most facts, 0 for 5 of 24.
- **The binding constraint is the data, not the architecture.** Most obscure facts have exactly
  one archived copy anywhere in the world. That is a property of the web, not of our code.

Where the architecture *does* buy real redundancy is the **date**, not the fact: 79% of facts got
a date from at least one route, and the tiers catch each other's failures — arquivo.pt answered
when Wayback was throttled, and vice versa.

## 6. Honest limits

- **Squid proxy not found running.** `docker ps` timed out enumerating the container set; the
  `b4tman/squid:5.8` image exists locally but no proxy container was confirmed up. Proxy rotation
  is therefore implemented as a documented next step, **not** as a working claim.
- **209 sources catalogued, one resolver tier integrated.** The registry is a map.
- **12 of 15 probed archives are unreachable from this host.** Some may be reachable through a
  different egress; that is testable with the proxy, and is not evidence they are dead.

## 7. Next

1. **Bring the Squid proxy up and re-probe** — 12 dead endpoints may be egress-blocked, not dead.
2. **Wire `quorum_dates` into `facts.harvest`** so facts arrive with their strength label attached.
3. **Accept `single-tier` as the normal case** and let the temporal gate consume the strength
   rather than requiring corroboration that often does not exist.
