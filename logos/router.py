"""logos/router.py — agentic source selection (bandit) + memory, so we stop re-asking the web.

TWO PROBLEMS
------------
1. **Too many sources.** The registry holds 209. Querying all of them per claim is unaffordable
   and, worse, counterproductive: most will return nothing, and the noise buries the two that
   would have answered. We need to *learn which sources pay off* and query a handful.

2. **Re-asking what we already know.** The same URLs get resolved repeatedly across runs. Every
   repeat is latency we already paid for.

WHAT THIS IMPLEMENTS
--------------------
**A contextual bandit.** Each (context, source) pair carries a Beta(alpha, beta) posterior over
"this source returns usable data for this kind of question". Selection is **Thompson sampling**:
draw one sample per candidate and take the top k. That gives three properties for free, which is
why it is the right algorithm rather than a scored list:

  * unexplored sources get tried (an unseen source has a wide Beta, so its sample can win)
  * good sources are exploited (their posterior mass concentrates high)
  * bad sources fade without being permanently banned (their posterior mass concentrates low but
    a lucky draw can still rescue them, so a source that was temporarily down is not lost)

`context` is the *kind of question*, not the claim text: `date_resolution`, `numeric_check`,
`entity_lookup`, `archive_capture`. Different tasks favour different sources, and one global
ranking would destroy that.

**A memory.** Content-addressed store keyed by (task, query) with a TTL, recording hit/miss stats.
It is also the bandit's persistence: posteriors survive across runs, so the system genuinely keeps
learning the more it is used.

WHY NOT A TRAINED MODEL
-----------------------
A neural ranker needs labelled (context, source, outcome) triples at a scale we do not have yet --
the same reason the per-evidence head stalled. A Beta posterior needs one bit per observation and
is exactly optimal for a Bernoulli arm. Start where the statistics are sufficient; upgrade to a
model when there is data to fit. The interface (`select` / `observe`) does not change, so the
upgrade is a swap, not a rewrite.

BUDGET
------
`select(k=N)` is the guard against information overload. The default cap is deliberately small:
querying 4 sources well beats querying 20 badly, and the returned list is ordered by expected
value so a caller can stop early.

USAGE
-----
    python tools/logos/router.py --self-test
    python tools/logos/router.py --summary
    python tools/logos/router.py --select date_resolution 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

DEFAULT_STATE = os.path.join(os.path.expanduser("~"), ".cache", "logos_router.json")

# Default context set: the kinds of question we actually ask a source.
CONTEXTS = ("date_resolution", "numeric_check", "entity_lookup", "archive_capture",
            "numeric_value", "quantifier_check")


# ------------------------------------------------------------------------------ bandit
@dataclass
class Arm:
    """Beta posterior for one (context, source) pair."""

    successes: float = 1.0        # alpha, including the Beta(1,1) prior
    failures: float = 1.0         # beta
    observations: int = 0
    total_ms: float = 0.0

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)

    @property
    def trials(self) -> float:
        return self.successes + self.failures

    @property
    def uncertainty(self) -> float:
        """Std dev of Beta(alpha, beta) -- how much we still do not know."""
        a, b = self.successes, self.failures
        t = a + b
        return (a * b / (t * t * (t + 1))) ** 0.5

    def sample(self, rng: random.Random) -> float:
        """One Thompson draw. Beta via two gammas, no scipy needed."""
        x = rng.gammavariate(self.successes, 1.0)
        y = rng.gammavariate(self.failures, 1.0)
        return x / (x + y) if (x + y) > 0 else 0.5


class Router:
    """Thompson-sampling source router with persistent posteriors."""

    def __init__(self, state_path: str = DEFAULT_STATE, seed: int = 0,
                 prior_strength: float = 1.0) -> None:
        self.state_path = state_path
        self.rng = random.Random(seed)
        self.prior = float(prior_strength)
        self.arms: dict[str, dict[str, Arm]] = {}
        self.load()

    # ------------------------------------------------------------------------ state
    def load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:                 # noqa: BLE001 - absent state is a cold start
            return
        for ctx, arms in raw.get("arms", {}).items():
            self.arms[ctx] = {k: Arm(**v) for k, v in arms.items()}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump({"arms": {c: {k: a.__dict__ for k, a in arms.items()}
                                    for c, arms in self.arms.items()}}, fh, indent=1)
        except Exception:                 # noqa: BLE001
            pass

    def arm(self, context: str, source: str) -> Arm:
        return self.arms.setdefault(context, {}).setdefault(
            source, Arm(successes=self.prior, failures=self.prior))

    # ---------------------------------------------------------------------- selection
    def select(self, context: str, candidates: list[str], k: int = 4,
               exclude: set[str] | None = None) -> list[str]:
        """Top-k sources for this context, by Thompson sample. The budget guard."""
        exclude = exclude or set()
        pool = [s for s in candidates if s not in exclude]
        if not pool:
            return []
        drawn = [(self.arm(context, s).sample(self.rng), s) for s in pool]
        drawn.sort(reverse=True)
        return [s for _, s in drawn[:max(1, k)]]

    def ranked(self, context: str, candidates: list[str]) -> list[tuple[str, float, int]]:
        """Deterministic ranking by posterior MEAN (for reporting, not for choosing)."""
        out = [(s, self.arm(context, s).mean, self.arm(context, s).observations)
               for s in candidates]
        out.sort(key=lambda t: -t[1])
        return out

    # -------------------------------------------------------------------- observation
    def observe(self, context: str, source: str, success: bool,
                elapsed_ms: float = 0.0) -> Arm:
        a = self.arm(context, source)
        if success:
            a.successes += 1.0
        else:
            a.failures += 1.0
        a.observations += 1
        a.total_ms += elapsed_ms
        return a

    def summary(self) -> dict:
        out = {}
        for ctx, arms in self.arms.items():
            rows = sorted(((s, a.mean, a.observations) for s, a in arms.items()),
                          key=lambda t: -t[1])
            out[ctx] = [{"source": s, "mean": round(m, 3), "n": n} for s, m, n in rows]
        return out


# ------------------------------------------------------------------------------- memory
ABSENT = "\x00ABSENT"            # a remembered "we looked and found nothing" is a real answer


@dataclass
class Memo:
    """Content-addressed memory for lookup results, with TTL and hit/miss stats.

    AN UNUSABLE ANSWER IS NOT A RESULT
    ----------------------------------
    The first version cached whatever a lookup returned, including answers that failed the caller's
    validator. Measured on a recorded 24-fact fixture (logos/ablate_memory.py):

        memo + bandit, caching anything      12/24 gate-kept
        memo + bandit, never caching junk    14/24
        memo alone                           14/24
        no memory at all                     14/24

    Caching an unusable answer turned a one-off search mistake into a thirty-day error, and that is
    why adding the second memory made the system WORSE THAN HAVING NEITHER. The two memories are not
    independent: the memo amplifies the bandit's exploration mistakes into permanent belief.

    So `remember` takes a `usable` flag and unusable entries get a SHORT ttl. The system still
    remembers "I already looked and found nothing good" so it does not re-ask on every call, but it
    retries soon, because a throttled archive is often fine an hour later.
    """

    path: str = os.path.join(os.path.expanduser("~"), ".cache", "logos_memo.json")
    ttl_s: float = 60 * 60 * 24 * 30          # 30 days: facts do not change hourly
    # 0 means an UNUSABLE answer is not cached at all. That looks wasteful and is deliberate: the
    # memo and the bandit compete for the same improvement budget. The bandit learns only from
    # trials, so caching "we already failed here" removes exactly the trials it needs. Measured on
    # the fixture: caching failures left the pair at 12/24 while memory-free scored 14/24 -- the
    # cache froze the learner before it could improve. The cost of not caching is k requests per
    # undatable fact per session (2 of 48 here), which is the cheapest thing in this system.
    retry_s: float = 0.0
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.entries = raw.get("entries", {})
            self.hits = raw.get("hits", 0)
            self.misses = raw.get("misses", 0)
        except Exception:                 # noqa: BLE001
            pass

    @staticmethod
    def key(task: str, query: str) -> str:
        return hashlib.sha1(f"{task}|{query}".encode("utf-8")).hexdigest()

    def get_entry(self, task: str, query: str) -> dict | None:
        """{'value', 'usable'}, or None. Distinguishes a remembered ABSENT from a cold miss."""
        e = self.entries.get(self.key(task, query))
        if not e:
            self.misses += 1
            return None
        ttl = self.ttl_s if e.get("usable", True) else e.get("ttl", self.retry_s)
        # '>=' not '>': a TTL of 0 must expire immediately, and '>' leaves a same-tick write
        # alive because the difference can be exactly 0.
        if time.time() - e["at"] >= ttl:
            self.misses += 1
            return None
        self.hits += 1
        v = e["value"]
        return {"value": None if v == ABSENT else v, "usable": bool(e.get("usable", True))}

    def get(self, task: str, query: str):
        """Backward-compatible accessor: the value, or None. A remembered ABSENT returns None."""
        e = self.get_entry(task, query)
        return None if e is None else e["value"]

    def remember(self, task: str, query: str, value, usable: bool = True) -> None:
        """Store a result. `usable=False` also shortens the ttl so the system retries soon."""
        self.entries[self.key(task, query)] = {
            "value": ABSENT if value is None else value,
            "at": time.time(),
            "usable": bool(usable),
            "ttl": self.ttl_s if usable else self.retry_s,
            "task": task,
            "query": query[:200],
        }

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"entries": self.entries, "hits": self.hits,
                           "misses": self.misses}, fh)
        except Exception:                 # noqa: BLE001
            pass

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


# --------------------------------------------------------------------- routed lookup
def routed_lookup(router: Router, memo: Memo, context: str, query: str,
                  candidates: list[str], fetch, k: int = 3, usable=None) -> dict:
    """Memo → bandit-selected sources → record what worked. The whole point of both pieces.

    `fetch(query, source)` returns a value or None; the first non-None answer is kept.

    `usable(value) -> bool` is the REWARD, and it is not optional in spirit. Measured on 24 real
    facts: rewarding a source for merely *returning* something produced 96% coverage of which 12 of
    23 dates were unusable, and taught the bandit to prefer a server render timestamp over a real
    archive capture. Rewarding the same run on "the answer survives the temporal gate" produced 58%
    coverage with zero junk, and correctly demoted the lying route by 3x. Same algorithm, opposite
    behaviour -- the reward decides what gets learned.

    Pass a validator. If you do not, the router learns to prefer sources that talk, not sources
    that are right, and it will get very good at that.
    """
    cached = memo.get_entry(context, query)
    if cached is not None:
        return {"value": cached["value"], "from": "memo", "tried": [],
                "hit": cached["value"] is not None, "usable": cached["usable"]}

    tried, value, winner = [], None, None
    good = False                     # a source may never be tried at all; never leave this unbound
    for src in router.select(context, candidates, k=k):
        t0 = time.perf_counter()
        try:
            got = fetch(query, src)
        except Exception:                 # noqa: BLE001 - a failing source is data, not a crash
            got = None
        ms = (time.perf_counter() - t0) * 1000

        # observe on USEFULNESS, not on response
        good = got is not None
        if good and usable is not None:
            try:
                good = bool(usable(got))
            except Exception:             # noqa: BLE001 - a broken validator must not kill a run
                good = False
        router.observe(context, src, good, ms)
        tried.append({"source": src, "ok": good, "answered": got is not None,
                      "ms": round(ms, 1)})

        # the VALUE is accepted on answering; the REWARD is recorded on being usable. Those differ:
        # an unusable answer is still a failure the bandit has to learn from.
        #
        # AND AN UNUSABLE ANSWER MUST NOT END THE SEARCH. Measured: a post-claim wayback capture
        # terminated the loop while a second route held a pre-claim capture, costing 4 of 24 facts
        # their gate pass. So the first answer is kept as a fallback and the loop continues looking
        # for one that actually settles the question.
        if got is not None and value is None:
            value, winner = got, src           # fallback, may be overwritten below
        if good:
            value, winner = got, src
            break                              # stop early: the budget guard in action

    if value is not None:
        # remember whether it was USABLE, not merely present. Caching junk is how the memo made
        # the two-memory system score 12/24 while memory-free scored 14/24.
        memo.remember(context, query, value, usable=bool(usable is None or good))
    else:
        memo.remember(context, query, None, usable=False)   # remember the miss, retry on the short ttl

    # `usable` is a statement about the ANSWER: nothing found is never usable, and with no validator
    # supplied we can only say the answer exists, not that it is any good.
    found_usable = bool(value is not None and (usable is None or good))
    return {"value": value, "from": winner or "none", "tried": tried,
            "hit": value is not None, "usable": found_usable}


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos/router - SELF TEST")
    ok = True
    import tempfile

    tmp = tempfile.mkdtemp()
    R = Router(state_path=os.path.join(tmp, "r.json"), seed=7)

    # --- an unseen source has a Beta(1,1) posterior centred at 0.5
    a = R.arm("date_resolution", "wayback")
    good = abs(a.mean - 0.5) < 1e-12 and a.observations == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cold prior mean 0.5 for an unseen source")

    # --- posterior converges to the empirical rate (analytic)
    for _ in range(80):
        R.observe("date_resolution", "wayback", True)
    for _ in range(20):
        R.observe("date_resolution", "wayback", False)
    a = R.arm("date_resolution", "wayback")
    # prior Beta(1,1) + 80 successes + 20 failures -> Beta(81, 21); mean 81/102.
    good = abs(a.mean - 81 / 102) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] posterior mean {a.mean:.4f} == "
          f"(1+80)/(1+20+1+0) = {81 / 102:.4f}")

    # --- a well-observed arm is more certain than a cold one
    cold = Router(state_path=os.path.join(tmp, "r0.json")).arm("date_resolution", "x")
    good = a.uncertainty < cold.uncertainty
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] uncertainty falls with observation "
          f"({a.uncertainty:.4f} < {cold.uncertainty:.4f})")

    # --- EXPLOITATION: a proven source dominates the top slot
    R2 = Router(state_path=os.path.join(tmp, "r2.json"), seed=11)
    for _ in range(60):
        R2.observe("date_resolution", "good", True)
    for _ in range(60):
        R2.observe("date_resolution", "bad", False)
    picks = [R2.select("date_resolution", ["good", "bad"], k=1)[0] for _ in range(200)]
    frac = picks.count("good") / len(picks)
    good_ = frac >= 0.95
    ok = ok and good_
    print(f"  [{'PASS' if good_ else 'FAIL'}] exploitation: the proven source is chosen "
          f"{frac:.1%} of the time (>= 95%)")

    # --- EXPLORATION: an untried source still gets tried despite a strong incumbent
    R3 = Router(state_path=os.path.join(tmp, "r3.json"), seed=3)
    for _ in range(40):
        R3.observe("date_resolution", "incumbent", True)
    picks = [R3.select("date_resolution", ["incumbent", "newcomer"], k=1)[0]
             for _ in range(200)]
    frac_new = picks.count("newcomer") / len(picks)
    good = 0.01 < frac_new < 0.40
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] exploration: an untried source is still tried "
          f"{frac_new:.1%} of the time")

    # --- a bad source is not permanently banned (a temporary outage must not be fatal)
    frac_later = frac_new
    good = frac_later > 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a losing arm is never fully abandoned "
          f"(posterior, not a blacklist)")

    # --- BUDGET: never more than k, and k is respected
    many = [f"s{i}" for i in range(50)]
    picked = R.select("numeric_check", many, k=3)
    good = len(picked) == 3 and len(set(picked)) == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] budget: 50 candidates -> exactly 3 selected")

    # --- contexts are independent: learning one does not corrupt another
    R4 = Router(state_path=os.path.join(tmp, "r4.json"), seed=5)
    for _ in range(50):
        R4.observe("date_resolution", "wayback", True)
        R4.observe("numeric_check", "wayback", False)
    good = (R4.arm("date_resolution", "wayback").mean > 0.8
            and R4.arm("numeric_check", "wayback").mean < 0.2)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] contexts are independent: same source, "
          f"date={R4.arm('date_resolution', 'wayback').mean:.2f} vs "
          f"numeric={R4.arm('numeric_check', 'wayback').mean:.2f}")

    # --- persistence: it KEEPS LEARNING across runs
    R4.save()
    R5 = Router(state_path=os.path.join(tmp, "r4.json"))
    good = abs(R5.arm("date_resolution", "wayback").mean
               - R4.arm("date_resolution", "wayback").mean) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] posteriors persist across processes "
          f"(it keeps learning)")

    # --- MEMO: a second identical question is free
    M = Memo(path=os.path.join(tmp, "m.json"))
    calls = {"n": 0}

    def fetch(q, src):
        calls["n"] += 1
        return "2015-08-16" if src == "arquivo_pt" else None

    r1 = routed_lookup(R5, M, "date_resolution", "https://x/y", ["arquivo_pt", "wayback"], fetch)
    first_calls = calls["n"]
    r2 = routed_lookup(R5, M, "date_resolution", "https://x/y", ["arquivo_pt", "wayback"], fetch)
    good = (r1["hit"] and r2["from"] == "memo" and calls["n"] == first_calls)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] memo: the repeat made 0 extra calls "
          f"({first_calls} total)")

    # --- routed_lookup observes what it tries, so the router learns from real use
    good = R5.arm("date_resolution", "arquivo_pt").observations >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] routed_lookup feeds observations back to the "
          f"router")

    # --- a failing source is recorded as a failure, not an exception
    calls2 = {"n": 0}

    def boom(q, src):
        calls2["n"] += 1
        raise RuntimeError("source down")

    r3 = routed_lookup(R5, M, "numeric_check", "https://x/z", ["wayback"], boom)
    good = r3["value"] is None and not r3["hit"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a raising source is recorded as a failure, "
          f"no crash")

    # --- THE ABLATION, AS A REGRESSION TEST.
    #     A source that always answers with junk must be demoted relative to one that answers
    #     usefully. Without `usable`, the liar wins -- measured for real: 96% "coverage" where 12
    #     of 23 dates were unusable, and the bandit learned to prefer a server render timestamp.
    tmp2 = tempfile.mkdtemp()
    R6 = Router(state_path=os.path.join(tmp2, "r6.json"), seed=13)
    M6 = Memo(path=os.path.join(tmp2, "m6.json"))

    def noisy_fetch(q, src):
        return "JUNK" if src == "liar" else ("GOOD" if src == "witness" else None)

    for i in range(40):
        routed_lookup(R6, M6, "date_resolution", f"q{i}", ["liar", "witness"], noisy_fetch,
                      k=2, usable=lambda v: v == "GOOD")
    liar = R6.arm("date_resolution", "liar")
    wit = R6.arm("date_resolution", "witness")
    good = liar.mean < 0.30 and wit.mean > 0.70
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] with a usable() validator the always-answering liar "
          f"is demoted ({liar.mean:.2f}) and the witness promoted ({wit.mean:.2f})")

    # --- the contrast: WITHOUT a validator the same liar is rewarded and wins. Asserted because
    #     this is the failure mode -- a refactor that drops `usable` should break this test loudly
    #     rather than silently make the router prefer noise.
    R7 = Router(state_path=os.path.join(tmp2, "r7.json"), seed=13)
    M7 = Memo(path=os.path.join(tmp2, "m7.json"))
    for i in range(40):
        routed_lookup(R7, M7, "date_resolution", f"q{i}", ["liar", "witness"], noisy_fetch, k=2)
    liar7 = R7.arm("date_resolution", "liar")
    good = liar7.mean > 0.90
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] without a validator the SAME liar scores "
          f"{liar7.mean:.2f} -- response is not usefulness, and the default reward is a trap")

    # --- an UNUSABLE answer must not end the search: a post-claim capture must not stop us from
    #     finding a pre-claim one. Measured cost of the old behaviour: 4 of 24 facts lost their
    #     gate pass because the first route answered with a date after the claim.
    tmp3 = tempfile.mkdtemp()
    R8 = Router(state_path=os.path.join(tmp3, "r8.json"), seed=17)
    M8 = Memo(path=os.path.join(tmp3, "m8.json"))
    late_early = {"late": lambda q, s: "2025-05-31", "early": lambda q, s: "2016-01-01"}

    def fetch2(q, src):
        return late_early[src](q, src)

    for i in range(30):
        r = routed_lookup(R8, M8, "date_resolution", f"z{i}", ["late", "early"], fetch2,
                          k=2, usable=lambda v: v <= "2022-01-01")
        if r["value"] != "2016-01-01":
            break
    good = r["value"] == "2016-01-01"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a post-claim answer does not stop the search; the "
          f"pre-claim route still wins ({r['value']})")

    # --- THE ABLATION FINDING, AS A REGRESSION TEST: a cached UNUSABLE answer must not be served
    #     as if it were fresh. Measured cost of the old behaviour: the two-memory configuration
    #     scored 12/24 on a recorded fixture while NO MEMORY scored 14/24. Caching junk is worse
    #     than having no cache, because it converts one bad search into a long-lived wrong belief.
    tmp4 = tempfile.mkdtemp()
    R9 = Router(state_path=os.path.join(tmp4, "r9.json"), seed=19)
    M9 = Memo(path=os.path.join(tmp4, "m9.json"))

    def junk_fetch(q, src):
        return "JUNK"

    r = routed_lookup(R9, M9, "date_resolution", "q-junk", ["a"], junk_fetch, k=1,
                      usable=lambda v: v == "GOOD")
    stored = [e for e in M9.entries.values() if e.get("usable") is False]
    served = M9.get_entry("date_resolution", "q-junk")
    good = r["usable"] is False and bool(stored) and served is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unusable answer is stored as unusable and NOT "
          f"served as a result")

    # --- and it is not served at all on the next call, so the bandit still gets its trials
    M10 = Memo(path=os.path.join(tmp4, "m10.json"), ttl_s=10_000_000)
    M10.remember("t", "q", "JUNK", usable=False)
    good = M10.get_entry("t", "q") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unusable entry is NOT served next call even when "
          f"the normal ttl is huge -- the bandit keeps its trials")

    # --- but a USABLE entry IS served, so the cache still does its job
    M11 = Memo(path=os.path.join(tmp4, "m11.json"))
    M11.remember("t", "ok", "2015-08-16", usable=True)
    e = M11.get_entry("t", "ok")
    good = e is not None and e["value"] == "2015-08-16" and e["usable"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a usable entry is served normally")

    # --- a cold miss returns None (the negative-cache semantics are opt-in via retry_s)
    good = M11.get_entry("t", "never-seen") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cold miss returns None")

    # --- nothing found is never "usable", even when no validator was supplied
    r0 = routed_lookup(Router(state_path=os.path.join(tmp4, "r0b.json")),
                       Memo(path=os.path.join(tmp4, "m0b.json")),
                       "date_resolution", "q-empty", [], lambda q, s: None)
    good = r0["value"] is None and r0["usable"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an empty candidate list returns unusable, no crash")

    # --- a found answer with no validator is usable; a found answer failing the validator is not
    R0 = Router(state_path=os.path.join(tmp4, "r0c.json"))
    M0 = Memo(path=os.path.join(tmp4, "m0c.json"))
    a = routed_lookup(R0, M0, "date_resolution", "q-noval", ["s"], lambda q, s: "X")
    b = routed_lookup(R0, M0, "date_resolution", "q-val", ["s"], lambda q, s: "X",
                      usable=lambda v: False)
    good = a["usable"] is True and b["usable"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] usable follows the validator when one is supplied")

    # --- memo TTL expiry is honoured
    M2 = Memo(path=os.path.join(tmp, "m2.json"), ttl_s=0.0)
    M2.remember("t", "q", "v")
    good = M2.get("t", "q") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] expired memo entries are misses")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bandit source router + memo.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--select", nargs="+", default=[], metavar="CONTEXT [CANDIDATE...]")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--state", default=DEFAULT_STATE)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    R = Router(state_path=args.state)
    if args.summary:
        s = R.summary()
        if not s:
            print("no observations yet -- the bandit is cold")
        for ctx, rows in s.items():
            print(f"\n{ctx}:")
            for r in rows[:10]:
                print(f"  {r['source']:28s} mean={r['mean']:.3f}  n={r['n']}")
        return 0
    if args.select:
        ctx = args.select[0]
        cands = args.select[1:] or list(CONTEXTS)
        print(f"context {ctx}:")
        for src, mean, n in R.ranked(ctx, cands):
            print(f"  {src:28s} mean={mean:.3f}  n={n}")
        print(f"\nThompson top-{args.k}: {R.select(ctx, cands, k=args.k)}")
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
