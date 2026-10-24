"""logos/dating.py — attach a defensible date to every harvested fact, choosing routes by bandit.

THE GAP THIS CLOSES
-------------------
`facts.harvest` produces facts with provenance but often no usable date, and `redundancy.quorum`
can date one citation at a time. Nothing joined them, so every experiment re-derived dates by hand
and the temporal gate was fed whatever happened to be lying around.

`date_facts()` runs the whole thing: for each fact, ask the **bandit** which routes are worth
trying (2 instead of 4), take the first usable answer, record the outcome so the router learns, and
remember the result so a repeat costs nothing.

THE REWARD IS TASK-RELATIVE, AND THAT IS THE POINT
--------------------------------------------------
The validator is not "did a source answer". It is **"is this date usable as precedence evidence for
*this* claim date"**:

    usable(d) = d <= claim_date

Measured consequence on 24 real facts: with an availability reward, coverage looked like 96% while
12 of 23 dates were unusable and the bandit learned to prefer a server render timestamp
(`http_last_modified`, mean 0.400). With the task-relative reward the same run scored 58% coverage
with zero junk and demoted that route 3x (0.143). Same sources, same day, opposite lesson.

A post-claim date is not a bad *fact* about the page -- it is a useless *answer* to "what existed
before this claim". Scoring it as success teaches the router to fetch dates that cannot settle the
question, which is worse than fetching nothing.

WHAT A DATE HERE DOES AND DOES NOT PROVE
----------------------------------------
  * an archive capture date proves the CITED PAGE existed then -- a lower bound on the claim
  * it does NOT prove the page said what Wikidata claims it said
  * `http_last_modified` is a server state, not a content date, and is rejected when fresh

So every result carries `gate_ok` AND `strength`, and a weakly-sourced date is still marked weak.

USAGE
-----
    python tools/logos/dating.py --self-test
    python tools/logos/dating.py --run 24 --claim 2022-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.redundancy import (arquivo_pt_first, arquivo_pt_timemap,  # noqa: E402
                              http_last_modified, wayback_first)
from logos.router import Memo, Router, routed_lookup  # noqa: E402

CONTEXT = "date_resolution"

ROUTES = {
    "wayback_cdx": wayback_first,
    "arquivo_pt_cdx": arquivo_pt_first,
    "arquivo_pt_timemap": arquivo_pt_timemap,
    "http_last_modified": http_last_modified,
}


@dataclass
class DatedFact:
    qid: str
    entity: str
    attribute: str
    value: float
    date: str = ""
    source: str = ""
    strength: str = "none"          # none | single-route
    gate_ok: bool = False
    reason: str = ""
    tried: list = field(default_factory=list)


def date_facts(facts, claim_date: str, k: int = 2,
               router: Router | None = None, memo: Memo | None = None,
               fetchers: dict | None = None, timeout: float = 20.0,
               online: bool = True) -> list[DatedFact]:
    """Date a batch of facts, choosing routes by bandit and validating by the precedence question.

    `fetchers` lets a test inject fakes; `online=False` with no fetchers means every route returns
    None, which is the honest cold-start case rather than a crash.
    """
    fetchers = fetchers if fetchers is not None else (ROUTES if online else {})
    router = router or Router()
    memo = memo or Memo()
    out: list[DatedFact] = []

    for f in facts:
        ref = getattr(f, "ref_url", "") or ""
        row = DatedFact(qid=getattr(f, "qid", ""), entity=getattr(f, "entity", ""),
                        attribute=getattr(f, "attribute", ""),
                        value=float(getattr(f, "value", 0.0)))

        # tier A: the statement's own point-in-time needs no network and always wins if present
        pit = (getattr(f, "point_in_time", "") or "")[:10]
        if pit:
            row.date, row.source, row.strength = pit, "tier_a:P585", "single-tier"
            row.gate_ok = pit <= claim_date[:10]
            row.reason = ("point-in-time qualifier, the publisher's own statement"
                          if row.gate_ok else
                          f"point-in-time {pit} postdates claim {claim_date[:10]}")
            out.append(row)
            continue

        if not ref:
            row.reason = "no reference URL; a date resolver dates citations, it cannot invent them"
            out.append(row)
            continue

        # the reward: usable AS PRECEDENCE EVIDENCE FOR THIS CLAIM
        usable = (lambda d, _c=claim_date[:10]: bool(d) and d[:10] <= _c)

        def _fetch(query: str, source: str) -> str | None:
            fn = fetchers.get(source)
            if fn is None:
                return None
            try:
                return fn(query, timeout)
            except TypeError:
                return fn(query)

        res = routed_lookup(router, memo, CONTEXT, ref, list(fetchers), _fetch,
                            k=k, usable=usable)
        row.tried = res["tried"]
        if res["value"]:
            row.date = res["value"]
            row.source = res["from"]
            row.strength = "single-route"
            row.gate_ok = row.date[:10] <= claim_date[:10]
            row.reason = (f"archived/cited page existed {row.date} via {row.source}, "
                          f"which predates the claim" if row.gate_ok else
                          f"{row.source} returned {row.date}, after the claim {claim_date[:10]}; "
                          f"this cannot establish precedence")
            if res["from"] == "memo":
                row.reason = f"from memory: {row.date} (no network call this time)"
        else:
            row.reason = ("no route returned a usable pre-claim date"
                          if res["tried"] else "no routes available")
        out.append(row)

    return out


def report(rows: list[DatedFact], claim_date: str) -> dict:
    n = len(rows) or 1
    dated = [r for r in rows if r.date]
    kept = [r for r in rows if r.gate_ok]
    from collections import Counter
    by_source = Counter(r.source for r in dated)
    return {
        "total": len(rows),
        "dated": len(dated),
        "dated_pct": round(len(dated) / n, 3),
        "gate_kept": len(kept),
        "gate_kept_pct": round(len(kept) / n, 3),
        "by_source": dict(by_source),
        "rejected": [(r.entity, r.date, r.reason) for r in dated if not r.gate_ok],
    }


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.dating - SELF TEST")
    ok = True
    import tempfile

    class _F:
        def __init__(self, qid, entity, ref="", pit=""):
            self.qid, self.entity, self.ref_url, self.point_in_time = qid, entity, ref, pit
            self.attribute, self.value = "height", 100.0

    tmp = tempfile.mkdtemp()

    def fresh(n="r"):
        return (Router(state_path=os.path.join(tmp, f"{n}.json"), seed=1),
                Memo(path=os.path.join(tmp, f"{n}_m.json")))

    # --- tier A short-circuits: no network at all
    R, M = fresh("a")
    rows = date_facts([_F("Q1", "Bridge A", ref="https://x/a", pit="2015-01-01")], "2022-01-01",
                      router=R, memo=M, fetchers={})
    good = rows[0].date == "2015-01-01" and rows[0].gate_ok and rows[0].source == "tier_a:P585"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a point-in-time qualifier is used with no network "
          f"call and passes the gate")

    # --- tier A after the claim is rejected, with the reason naming the date
    R, M = fresh("b")
    rows = date_facts([_F("Q2", "Bridge B", ref="https://x/b", pit="2023-06-01")], "2022-01-01",
                      router=R, memo=M, fetchers={})
    good = rows[0].date == "2023-06-01" and not rows[0].gate_ok and "postdate" in rows[0].reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a post-claim point-in-time is recorded but fails the "
          f"gate")

    # --- no reference URL: says so plainly, invents nothing
    R, M = fresh("c")
    rows = date_facts([_F("Q3", "Bridge C")], "2022-01-01", router=R, memo=M, fetchers={})
    good = rows[0].date == "" and "cannot invent" in rows[0].reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no reference URL -> no date invented, reason states why")

    # --- injection works and a good route is used
    R, M = fresh("d")
    fakes = {"wayback_cdx": lambda q, t: "2015-08-16",
             "arquivo_pt_cdx": lambda q, t: None}
    rows = date_facts([_F("Q4", "Bridge D", ref="https://x/d")], "2022-01-01",
                      router=R, memo=M, fetchers=fakes, k=2)
    good = rows[0].date == "2015-08-16" and rows[0].gate_ok
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an injected route dates the fact and passes the gate")

    # --- the reward is task-relative: a post-claim date is a FAILURE for the bandit
    R, M = fresh("e")
    late = {"late_source": lambda q, t: "2025-05-31"}
    rows = date_facts([_F("Q5", "Bridge E", ref="https://x/e")], "2022-01-01",
                      router=R, memo=M, fetchers=late, k=1)
    arm = R.arm(CONTEXT, "late_source")
    good = rows[0].date == "2025-05-31" and not rows[0].gate_ok and arm.mean < 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a post-claim date is kept as information but scored a "
          f"FAILURE (arm mean {arm.mean:.2f})")

    # --- memory: the second identical fact makes no call
    R, M = fresh("f")
    calls = {"n": 0}

    def counting(q, t):
        calls["n"] += 1
        return "2019-07-21"

    fakes2 = {"wayback_cdx": counting}
    f = _F("Q6", "Bridge F", ref="https://x/f")
    date_facts([f], "2022-01-01", router=R, memo=M, fetchers=fakes2)
    first = calls["n"]
    date_facts([f], "2022-01-01", router=R, memo=M, fetchers=fakes2)
    good = calls["n"] == first and first >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the repeat made 0 extra calls ({first} total)")

    # --- the bandit converges on the route that actually works
    R, M = fresh("g")
    fakes3 = {"good": lambda q, t: "2018-01-01", "bad": lambda q, t: None}
    for i in range(25):
        date_facts([_F(f"Q{i}", f"B{i}", ref=f"https://x/{i}")], "2022-01-01",
                   router=R, memo=M, fetchers=fakes3, k=1)
    good_arm = R.arm(CONTEXT, "good")
    bad_arm = R.arm(CONTEXT, "bad")
    good = good_arm.mean > 0.8 and bad_arm.mean < 0.4
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] after 25 facts the working route is preferred "
          f"({good_arm.mean:.2f} vs {bad_arm.mean:.2f})")

    # --- report arithmetic
    R, M = fresh("h")
    rows = date_facts([_F("Q7", "A", ref="https://x/1"), _F("Q8", "B")], "2022-01-01",
                      router=R, memo=M, fetchers={"wayback_cdx": lambda q, t: "2016-01-01"})
    rep = report(rows, "2022-01-01")
    good = rep["total"] == 2 and rep["dated"] == 1 and rep["gate_kept"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] report counts dated and gate-kept separately")

    # --- nothing available must not raise
    R, M = fresh("i")
    rows = date_facts([_F("Q9", "C", ref="https://x/9")], "2022-01-01",
                      router=R, memo=M, online=False)
    good = rows[0].date == "" and bool(rows[0].reason)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] with no routes at all the fact is undated, no crash")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agentic fact dating via bandit route selection.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", type=int, default=0, help="date N real facts")
    ap.add_argument("--claim", default="2022-01-01")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args(argv)
    if args.self_test or not args.run:
        return 0 if run_self_test() else 1

    from logos.facts import load
    facts = load(r"B:\datasets\forge\facts-ref")[:args.run]
    t0 = time.time()
    rows = date_facts(facts, args.claim, k=args.k)
    rep = report(rows, args.claim)
    print(f"{len(facts)} facts, claim {args.claim}, k={args.k}, {time.time() - t0:.0f}s")
    print(f"  dated     : {rep['dated']}/{rep['total']} ({rep['dated_pct']:.0%})")
    print(f"  gate-kept : {rep['gate_kept']}/{rep['total']} ({rep['gate_kept_pct']:.0%})")
    print(f"  by source : {rep['by_source']}")
    for e, d, why in rep["rejected"][:6]:
        print(f"    REJECT {e[:28]:28s} {d}  {why[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
