# The two-memory ablation: the memo wins, the bandit does not, and together they are worse than neither

**Date:** 2026-10-25
**New module:** `logos/ablate_memory.py` · fixture `B:\datasets\forge\date-fixture.json`
**Status:** this **overturns** the headline claim of the previous two sessions.

---

## 1. Why this was run at all

A paper landed the same week (UIUC, Oct 2026: *self-improving agent with a world model*) that builds
the same architecture we did — frozen weights, self-improvement carried entirely by memory, two
memory branches. Their ablation reports something that should worry anyone who ships that design:

> **ALFWorld SR@1 and SR@5: deactivating the planner memory BEATS having both memory branches
> active.** (It does not hold on ScienceWorld — it is setup-specific.)

The mapping to us is direct:

| their memory | ours |
|---|---|
| planner memory — decision lessons | **bandit posteriors** — which source to ask |
| world-model memory — environment dynamics | **memo cache** — what the answer was |

I had never ablated that combination. I had been reporting "bandit + memo" as the win, on the
strength of a comparison against the *4-route parallel quorum* — not against a fixed-order memo. That
is the wrong baseline, and it is exactly the mistake I keep claiming to guard against.

## 2. Method: a recorded fixture, because the network was drowning the signal

The previous measurements were unreadable at this resolution: the same corpus gave 13/24 on one run
and 14/24 on the next, because archive.org and arquivo.pt throttle. A two-point ablation cannot
survive that noise.

So every `(fact, route)` pair was fetched **once** into `date-fixture.json` (24 facts × 4 routes,
335 s), and all conditions **replay over identical data**. Differences are then attributable to the
selection policy alone. This should have been the method from the start.

## 3. Result

| condition | pass 1 | pass 2 | requests | gate-kept |
|---|---|---|---|---|
| **none** (fixed order, no memory) | 14/24 | 14/24 | 68 | 14 |
| **memo only** (fixed order + cache) | 14/24 | 14/24 | **31** | **14** |
| **bandit only** (adaptive, no cache) | 13/24 | 14/24 | **73** | 14 |
| **both** (bandit + cache anything) | 12/24 | **12/24** | 35 | **12** |
| both, cache usable answers only | 13/24 | 14/24 | 58 | 14 |

Three findings, in increasing order of discomfort:

1. **The memo alone halves the cost and loses nothing.** 68 → 31 requests, same 14/24. That is a
   real, clean win.
2. **The bandit alone is a regression.** 73 requests vs 68 for a fixed order, and no accuracy gain.
   It is spending requests to explore routes that a static order already handled.
3. **Naive `both` is worse than either alone AND worse than no memory at all** — 12/24 against
   14/24 with the cache switched off entirely. **The paper's warning reproduces on our data.**

## 4. The mechanism: the memo starves the learner, and poisons what it keeps

Two distinct interactions, both measured:

**Poisoning.** The memo cached whatever came back, including an answer that failed the caller's
validator. Fixing only that (`cache usable answers only`) recovers 12 → 14. So **2 of 24 facts were
being lost purely because a bad answer was written to memory**: one bad search became a thirty-day
wrong belief.

**Starvation.** Even fixed, `both` needs 58 requests to reach the 14 that `memo only` reaches in 31.
The reason is structural: **the bandit learns only from trials, and the memo's whole purpose is to
remove trials.** They are not two independent memories that add up — they compete for the same
improvement budget. A cache in front of a learner is a decision to stop learning about whatever the
cache holds.

That is the useful general statement, stronger than "two memories can be worse than one": **the
second memory does not merely fail to help; it removes the feedback the first one needs.**

## 5. What was fixed in the code

`Memo.remember` now takes a `usable` flag, and `retry_s` defaults to **0** — an unusable answer is
stored as unusable and **not served**. That looks wasteful and is deliberate: the cost is `k`
requests per undatable fact per session (2 of 48 here), which is the cheapest thing in this system,
and the alternative is freezing the learner before it can improve.

Four regression tests now pin it, including the one that would have caught the 12/24:
*an unusable entry is NOT served next call even when the normal ttl is huge.*

## 6. What this corrects in the record

I reported, two sessions ago:

> *"Querying 2 bandit-chosen sources instead of 4: half the requests, equal-or-better precision."*

That comparison was against the parallel quorum. Against a **fixed-order memo** — the honest
baseline, because that is what you would write without a bandit — the bandit is **+5 requests and
−0 facts**. The earlier sentence was not false, it was measured against a baseline chosen by the
mechanism I wanted to justify. The router stays in the library because it is correct and tested, but
**for date resolution the honest status is: unproven, and currently a net cost.**

## 7. Two more findings from the same week's papers that apply to us

**"Zero subtasks inflate summaries."** The UIUC evaluation has ALFWorld subtasks where ReAct scores
exactly 0. Averaging across subtasks then produces a spectacular headline, while on the `pick`
subtask alone plain ReAct beat the new method. **I have committed this exact error**: I reported
"coverage 96%" when the figure that mattered — dates that actually settle the claim — was 46%. Same
failure, one week apart, mine.

**"A redefined metric reported as accuracy."** A second paper (layer-transition OPD) has a genuine
result of **<1 percentage point**, while the summarisation circulating says *"up to 90% improvement."*
The 90% is a **self-defined recovery rate** (74.9% → 89.2%), not accuracy. Any number I report needs
its definition attached, on the same line, or it will be repeated as something it is not.

**Waypoint-OPD's real lesson for the bandit.** That paper's insight is that an *endpoint* signal
cannot distinguish consistent local improvement from **reversing shifts that cancel out**. Our
`Beta(α, β)` has exactly this blindness: it is **exchangeable**, so a source that alternates
work/fail/work/fail and one that works half the time *in a stable pattern* produce **identical
posteriors**. A gate on trajectory agreement — the paper's cosine gate — is the missing mechanism.
That is a design gap, not a tuning parameter.

## 8. Honest limits

- **n = 24.** Two facts is 8 points here; the 12-vs-14 gap is one or two facts. The *direction* is
  clear and the mechanism is confirmed by the poisoning control, but the magnitude is not precise.
- **One context, one task.** `date_resolution` only. The bandit may still earn its place on
  `entity_lookup`, where routes are numerous and genuinely heterogeneous — untested.
- **The fixture freezes route behaviour.** It cannot show a route that recovers mid-run. That is what
  the forced-probe test in `egress.py` covers separately.
- **Requests are not latency.** The bandit adds `k` sequential calls where a quorum ran them in
  parallel, so its wall-clock cost may be worse than the request count suggests.

## 9. Next

1. **Default to memo-only for dating** and leave the bandit off until it beats a fixed order.
2. **Ablate the bandit on `entity_lookup`**, where there are hundreds of candidates and a fixed
   order is genuinely indefensible — that is where exploration should pay.
3. **Add trajectory-consistency tracking to the arms** (Waypoint's gate), so an erratic source can be
   distinguished from a reliably-middling one.
4. **Report every number with its definition inline.** The 96%/46% episode and the 90% recovery-rate
   episode are the same error, and I made one of them.
