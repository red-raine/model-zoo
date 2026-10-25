# Egress: the proxy was running all along, and it is not a magic wand

**Date:** 2026-10-25
**New module:** `logos/egress.py` (11/11 self-tests)
**Correction to two previous claims**

---

## 1. The correction I have to lead with

I reported, twice: *"Squid proxy not confirmed running. `docker ps` times out enumerating the
container set."*

Both halves were misleading.

```
docker version --format ...        -> server=29.8.2   (daemon fine)
docker ps -q --filter name=squid   -> 7e59e8629a71    (instant)
docker inspect ...                 -> name=/rotating-squid-proxy
                                      running=true
                                      started=2026-10-23T00:45:19Z
                                      restart=unless-stopped
                                     3128/tcp -> 0.0.0.0:3128
```

**It had been running for two days.** The full `docker ps` is slow because of how many containers
exist; a *filtered* query answers in milliseconds. So "unverified" got reported as "probably down",
and that cost two sessions of treating 12 archives as dead.

**The rule worth keeping: a timed-out enumeration is not evidence of absence.** Targetted queries
exist for exactly this reason, and reaching for the broad one first is what produced the wrong
conclusion.

## 2. And the proxy does not do what I hoped

Probed every failed endpoint twice — forced direct, then through `:3128`:

| source | direct | proxy | reading |
|---|---|---|---|
| wayback-cdx | **200** | 429 | proxy **breaks** a working endpoint |
| arquivo.pt | 200 | 200 | both work |
| archive.today | 429 | 429 | **alive both ways** — my earlier "DNS dead" was transient |
| archive.is | DNS-fail | **429** | proxy **recovers** this one |
| ghostarchive | **200** | 200 | **new working source** |
| conifer | **200** | 200 | **new working source** |
| swap.stanford | **200** | 405 | proxy breaks it (my earlier "502" was transient) |
| perma.cc | 403 | 403 | alive, needs auth |
| loc-webarchive | 403 | 403 | alive, blocked |
| bibalex | DNS-fail | **409** | Squid cannot reach its own upstream |
| vefir.is | 404 | 409 | 404 = alive, wrong path |
| haw.nsk.hr | DNS-fail | 409 | dead or blocked |
| padi.cat | SSL-fail | 409 | dead or blocked |
| memento-aggregator | DNS-fail | 409 | dead or blocked |

**The proxy is not a fallback that always helps.** It recovered `archive.is` and broke
`wayback-cdx` and `swap.stanford`. Routing everything through it would have *reduced* coverage.

### `409` is Squid talking, not the origin

A `409 Conflict` on a CONNECT tunnel means **Squid could not reach the origin through its
upstream**. The origin was never contacted. So:

| status | meaning | scored as |
|---|---|---|
| `429` | the origin answered: "slow down" | **ALIVE** (reachability success) |
| `403` | the origin answered: "no" | **ALIVE** |
| `409` | Squid's own tunnel failed | **proxy-path failure** |
| `None` | nobody answered | unreachable |

Conflating a 429 with a DNS failure is precisely how a working archive gets written off as dead —
and that, not archive.org, was the actual bug in the earlier count.

## 3. So egress became a bandit arm

Which path works is a **per-source fact that changes over time**. That is a bandit problem, not a
configuration problem. The arm name is now `source|path`:

```
wayback_cdx|direct   archive_is|proxy   swap_stanford|direct
```

Thompson sampling learns the preference per source without a table that silently rots.

### Two bugs found while testing it

**Bug 1 — mean-ranking is not a bandit.** The first version sorted paths by posterior **mean**.
Measured: the losing arm was tried **once in 20 calls** (`mean 0.33 = 1/(2+1)`) — so a path that
had recovered would never be rediscovered. Greedy ordering by mean is a lookup table with extra
steps. Fixed: rank by Thompson sample.

**Bug 2 — two arms explore too slowly even after the fix.** With a strong incumbent, Thompson gave
the loser ~2 samples in 20 (mean 0.25). For egress the probe costs one HTTP request, so a forced
probe is cheap: every 5th call the least-tried path goes first. Measured effect: **a path that
comes back is rediscovered on the same call it recovers**, instead of at some point in the next few
hundred.

## 4. `+3` keyless independent indexes we did not have

`lc/gau` queries OTX, Wayback, Common Crawl and URLScan. We had one of those four. Probed the rest:

| surface | result | what it gives |
|---|---|---|
| **`archive.org/wayback/available`** | **200, 246 bytes** | the closest snapshot in ONE cheap JSON call — we have only been using CDX, which is far heavier |
| **`urlscan.io/api/v1/search`** | **200, 5.8 KB, real results** | **a genuinely independent URL index that is not archive.org** |
| **`otx.alienvault.com/api/v1/indicators`** | **200** | keyless host/URL intel; empty for a Wikipedia page, which is expected — it is security-oriented |
| `index.commoncrawl.org/collinfo.json` | 200, 129 collections | index list works, but see below |

**URLScan is the real find.** It is the first non-archive.org index that answers keyless. Archive
redundancy was always going to be capped by archive.org's coverage; a second *crawler* is a
different population of URLs, not a second window on the same one.

**Common Crawl is not usable right now.** `collinfo` returns 200 with 129 collections, but every
per-collection CDX endpoint fails (`CC-MAIN-2026-43` → 502, `CC-MAIN-2026-39` → 502,
`CC-MAIN-2026-34` → 404). The index cluster is down, not our URL. My earlier "wrong index name"
diagnosis was also wrong — the name was fine.

## 5. Corrected archive census

**Reachable and answering (direct):** wayback CDX, wayback availability, arquivo.pt CDX,
arquivo.pt timemap, ghostarchive, conifer, urlscan, OTX, commoncrawl collinfo.
**Alive but gated:** archive.today/ph (429), archive.is (429 via proxy), perma.cc (403),
loc-webarchive (403), swap.stanford (200), vefir.is (404).
**Genuinely unreachable:** bibalex, haw.nsk.hr, padi.cat, timetravel.mementoweb.org.

So the honest count is **9 answering, 6 gated, 4 dead** — not the "3 of 15" I reported. The
difference came from three errors of mine: a timed-out `docker ps`, transient failures read as
permanent, and a 429 read as a DNS failure.

## 6. Honest limits

- **The proxy's upstream rotation is what makes it slow** (12 s for `example.com`) and what gets it
  429'd by archive.org. A different upstream set would change the table.
- **`409` bodies are not parsed.** Squid can say *why* in the response; treating every 409 the same
  loses information.
- **`explore_every=5` is hand-set.** It should be tuned against request cost, which we have not
  measured per source.
- **Only reachability is learned, not content quality.** That separation is deliberate — mixing them
  is what taught the router to prefer liars — but it means a source that always 200s with junk
  still looks healthy here. The content judgement belongs to `router.usable`.

## 7. Next

1. **Wire `egress_fetch` into `redundancy.py`**, so every archive route learns its own path.
2. **Integrate the wayback availability API as a cheap tier-0 probe** before spending a CDX call.
3. **Add URLScan as a fourth route in `redundancy`,** covering `entity_lookup` and `archive_capture`
   contexts — it is the only independent index we have.
4. **Parse Squid 409 bodies** to distinguish "upstream refused" from "upstream timed out".
