# The agentic router: learning which sources to ask, and the reward that decides what it learns

**Date:** 2026-10-24
**Modules:** `logos/router.py` (13/13 self-tests), integrated with `logos/redundancy.py`, `logos/facts.py`
**Data:** 24 real Wikidata facts, 4 candidate date routes, budget k=2 per fact

---

## 1. Why a bandit and not a trained model

The ask was "train a model to pick the top sources so we don't overwhelm fact-checking with too
much information". A neural ranker needs labelled `(context, source, outcome)` triples at a scale
we do not have — the same wall the per-evidence head hit. A **Beta posterior needs one bit per
observation and is exactly optimal for a Bernoulli arm.**

So: Thompson sampling. Every `(context, source)` pair carries `Beta(α, β)` over "this source yields
usable data for this *kind* of question". Selection draws one sample per candidate, sorts, takes
top-k. Three properties come for free:

| property | why the Beta gives it |
|---|---|
| **exploration** | an unseen arm is wide, so it sometimes wins the draw |
| **exploitation** | a proven arm's mass concentrates high |
| **no permanent bans** | a bad arm's mass concentrates low, but a lucky draw still rescues it — a source that was down for an hour is not lost forever |

`context` is the *kind of question* (`date_resolution`, `numeric_check`, `entity_lookup`,
`archive_capture`), not the claim text. Different tasks favour different sources, and one global
ranking would destroy that. The interface is `select` / `observe`, so a trained ranker can be
swapped in later when there is data to fit — a swap, not a rewrite.

## 2. It learns

Cold bandit, 24 facts, k=2:

```
wayback_cdx         mean=0.900  n=18
arquivo_pt_cdx      mean=0.400  n=3
arquivo_pt_timemap  mean=0.400  n=3
http_last_modified  mean=0.400  n=3
```

**Selection drift, first 12 facts → last 12:**

```
wayback_cdx          9 -> 9      (kept asking the one that works)
arquivo_pt_cdx       3 -> 0      (stopped entirely)
arquivo_pt_timemap   2 -> 1
http_last_modified   2 -> 1
```

It stopped asking the useless sources. That is the whole point: **2 requests per fact instead of
4, and it still returned 23/24 dates.**

## 3. The memory makes the repeat free

Second pass over the identical 24 facts:

```
memo hits  3 -> 26     23/24 served from memory
```

The 1 miss is the fact that no route could ever date. A remembered **None** is a real answer, and
it is cached as one — so an undatable fact costs nothing on the second run either.

The memo is content-addressed on `sha1(task|query)` with a 30-day TTL, and it doubles as the
bandit's persistence, so posteriors survive process restarts. **It genuinely keeps learning the
more it is used.**

## 4. The reward ablation — and why it matters more than the algorithm

First run rewarded a source for **returning a date**. Coverage looked excellent. Then fact 9:

```
Talbrücke Olbersdorf   http_last_modified -> 2025-05-31
```

That page was archived in 2016. `2025-05-31` is the server's render timestamp. **The bandit counted
it as a success.** A source that returns a plausible-looking wrong answer every time is worse than a
silent one — and an availability reward *actively selects for it*.

So the same 24 facts were run under two rewards:

| reward | dated | **gate-kept** | junk dates |
|---|---|---|---|
| **R1** returned a date (availability) | 23/24 (96%) | **11/24 (46%)** | **12** |
| **R2** date survives the temporal gate (usefulness) | 13–14/24 (54–58%) | **13–14/24 (54–58%)** | **0** |

Under R1, **more than half the "successes" were unusable.** The 38-point coverage gain was noise
being counted as data. Under R2, `dated == gate-kept` exactly: every date it returned could settle
the question. (The 13-vs-14 difference across runs is archive variance, not a code change — the
invariant `dated == gate-kept` held in both.)

Under R2 the ranking itself changes — the same sources, the same data, a different lesson:

| source | R1 mean | R2 mean | verdict |
|---|---|---|---|
| wayback_cdx | 0.900 | **0.565** | down — it was never as good as availability made it look |
| arquivo_pt_cdx | 0.400 | 0.222 | down |
| arquivo_pt_timemap | 0.400 | 0.200 | down |
| **http_last_modified** | **0.400** | **0.143** | **down 3× — the bandit learned it is a liar** |

**The reward decides what the system learns.** "Returned something" teaches it to prefer liars;
"survived the gate" teaches it to prefer witnesses. Same algorithm, opposite behaviour.

## 5. The efficiency result

Querying **2** bandit-chosen sources instead of **4** in parallel:

| | sources/fact | gate-kept |
|---|---|---|
| fixed quorum (previous) | 4 | 13/24 |
| **bandit, R2 reward** | **2** | **13–14/24** |

**Half the requests, equal-or-better precision.** That is the agentic win: not more coverage, but
the same coverage for a fraction of the budget — which is what makes scaling to hundreds of sources
affordable at all.

### 5b. Two bugs the live run exposed

1. **An unusable answer ended the search.** The first post-claim wayback capture terminated the loop
   while the second route held a pre-claim capture. Losing 4 of 24 facts their gate pass. Fixed: the
   first answer is kept as a fallback and the loop continues looking for one that settles the
   question. Regression test included.
2. **`usable` was not in the API at all.** `routed_lookup` observed on `got is not None`, so any
   caller doing the right thing had no way to say so. Now `usable(value) -> bool` is a parameter, and
   two tests pin both directions: with a validator the liar is demoted to 0.25 / witness promoted to
   0.98; **without** one the same liar scores **0.97**. That second test exists to fail loudly if
   anyone removes the validator.

## 6. Borrowed patterns (from the repos)

- **`lc/gau`, `xnl-h4ck3r/waymore`** — the right shape for source fan-out: *one interface, many
  backends, merge + dedupe, degrade gracefully per backend*. Our 4 routes follow that shape.
- **Hochstenbach's Memento algorithm** (`mementoweb.org/tools/memento_algorithm.txt`) — resource
  type detection (`URI-R` / `URI-G` / `URI-M`) via `Accept-Datetime`, `Memento-Datetime`,
  `Vary: Accept-Datetime`, and `Link: rel="timegate"`, with a redirect cap. Our archive routes are
  still datetime-*query* based; full negotiation is the next step.
- **`machawk1/awesome-memento`** — the catalogue of who actually runs a Memento endpoint. Useful,
  but our probe showed the listed federation aggregator itself has dead DNS; being listed is not
  being reachable.
- **`lissy93/web-check`, `jivoi/awesome-osint`** — the OSINT checklist shape (DNS, headers, TLS,
  redirects, archive availability per host) is the right *per-source* health probe for the registry.

## 7. Honest limits

- **One context measured.** `date_resolution` got 24 facts; the other five contexts are cold.
- **The gate is a proxy for correctness.** R2 measures "not obviously wrong", which is weaker than
  "corroborated by an independent tier". A bandit trained on real corroboration needs tier
  agreement, and only 3/24 facts ever had two tiers.
- **Web-check / gau / waymore not integrated** — patterns borrowed, no code taken.
- **209 sources, 4 routes wired.** The bandit is ready for the other 205; each needs its own client.

## 8. Next

1. **Health-probe every registry source** (web-check shape) so dead arms start with a pessimistic
   prior instead of burning exploration budget.
2. **Add corroboration as a reward term**, so the bandit prefers routes that produce *two-tier*
   agreement — the only real defence against a confident liar.
3. **Wire `routed_lookup` into `facts.harvest`** so harvested facts arrive dated, with the bandit
   having already learned the cheapest route that works.
