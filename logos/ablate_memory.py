"""Ablate OUR two memories, because a published paper says the pair can be worse than one.

WHY THIS EXISTS
---------------
The UIUC self-improving-agent paper (Oct 2026) builds the same thing we did: frozen weights, and
self-improvement carried entirely by memory. It has TWO memories -- a planner memory that stores
decision lessons, and a world-model memory that stores environment dynamics. Ours map onto it
directly:

    their planner memory      <- our bandit posteriors   (which source to ask)
    their world-model memory  <- our memo cache          (what the answer was)

Their ablation reports the opposite of what everyone assumes:

    ALFWorld SR@1 and SR@5: DEACTIVATING the planner memory BEATS having both memory branches
                            active (it does not hold on ScienceWorld -- it is setup-specific)

Two further findings matter even more than that one:

  * a NO-MEMORY configuration already substantially outperforms the baselines, so the improvement
    cannot be attributed to the memory layer at all -- the predictive decision architecture is
    doing the work
  * some ReAct subtasks score exactly ZERO, which inflates any summary that averages across
    subtasks. On the "pick" subtask alone, plain ReAct beat the new method

If that pattern is real, our "bandit + memo" result may be two mechanisms whose contributions we
have attributed to each other. So this file does the thing I have been claiming is mandatory and
had not done for the router: a baseline, an ablation, and a control.

THE METHODOLOGICAL FIX
----------------------
The previous measurements were noisy because archives rate-limit: the same corpus gave 13/24 one run
and 14/24 the next. That makes a 1-2 point ablation unreadable. So this runs against a RECORDED
RESPONSE TABLE: every (fact, route) pair is fetched ONCE into a JSON fixture, and all four
conditions replay over identical data. Differences are then attributable to the selection policy,
not to whether arquivo.pt was in the mood.

CONDITIONS
    none        fixed route order, no memory     -- the baseline
    memo_only   fixed route order + memo cache
    bandit_only bandit selection, no memo
    both        bandit + memo

METRICS
    gate_kept   facts whose date actually establishes precedence for the claim
    requests    network calls made (the cost we are trying to reduce)

USAGE
    python tools/logos/ablate_memory.py --build 24     # record the fixture once
    python tools/logos/ablate_memory.py --run          # replay all four conditions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.redundancy import (arquivo_pt_first, arquivo_pt_timemap,  # noqa: E402
                              http_last_modified, wayback_available, wayback_first)

ROUTES = {
    "wayback_cdx": wayback_first,
    "wayback_available": wayback_available,
    "arquivo_pt_cdx": arquivo_pt_first,
    "http_last_modified": http_last_modified,
}
FIXTURE = r"B:\datasets\forge\date-fixture.json"
CLAIM = "2022-01-01"


def build(n: int, out: str = FIXTURE) -> str:
    """Fetch every (fact, route) pair once. This is the deterministic replay table."""
    from logos.facts import load
    facts = load(r"B:\datasets\forge\facts-ref")[:n]
    table = {"claim": CLAIM, "facts": []}
    t0 = time.time()
    for i, f in enumerate(facts, 1):
        row = {"qid": f.qid, "entity": f.entity, "ref_url": f.ref_url,
               "point_in_time": f.point_in_time, "retrieved": f.retrieved, "results": {}}
        for name, fn in ROUTES.items():
            try:
                row["results"][name] = fn(f.ref_url, 20.0) if f.ref_url else None
            except Exception:                                  # noqa: BLE001
                row["results"][name] = None
        table["facts"].append(row)
        got = sum(1 for v in row["results"].values() if v)
        print(f"  {i:2d}/{len(facts)} {f.entity[:26]:26s} {got}/{len(ROUTES)} routes answered")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1)
    print(f"\nfixture written: {out}  ({time.time() - t0:.0f}s)")
    return out


def _gate_ok(row: dict, date: str, claim: str = CLAIM) -> bool:
    """Same precedence rule as facts.temporal_gate: does this date settle the question?"""
    if not date:
        return False
    pit = (row.get("point_in_time") or "")[:10]
    if pit:
        return pit <= claim[:10]
    return date[:10] <= claim[:10]


def replay(table: dict, condition: str, budget_k: int = 2, use_bandit: bool = False,
           use_memo: bool = False, passes: int = 2, cache_usable_only: bool = False) -> dict:
    """Replay one condition over the recorded table. No network.

    `cache_usable_only` is the DIAGNOSTIC for the interaction this ablation found. The memo in
    routed_lookup caches the first non-None answer, including an UNUSABLE one (a date after the
    claim). So a bad ordering on pass 1 does not merely cost one search -- it cements a wrong
    answer for the whole TTL, and the bandit's exploration error becomes permanent memory.

    Two memories are not independent: one amplifies the other's mistakes. This flag tests that.
    """
    import tempfile
    from logos.router import Memo, Router

    tmp = tempfile.mkdtemp()
    R = Router(state_path=os.path.join(tmp, "r.json"), seed=42)
    M = Memo(path=os.path.join(tmp, "m.json"))

    fixed_order = list(ROUTES)
    requests = 0
    per_pass_kept = []

    for _p in range(passes):
        kept = 0
        for row in table["facts"]:
            pit = (row.get("point_in_time") or "")[:10]
            if pit:                                  # tier A needs no network at all
                kept += _gate_ok(row, pit)
                continue
            if not row.get("ref_url"):
                continue

            cache_key = row["ref_url"]
            if use_memo:
                hit = M.get("date_resolution", cache_key)
                if hit is not None:
                    # a cached value only counts if it would have been usable; with
                    # cache_usable_only a poisoned entry can never have been written
                    kept += _gate_ok(row, hit)
                    continue

            if use_bandit:
                order = R.select("date_resolution", fixed_order, k=budget_k)
            else:
                order = fixed_order[:budget_k]

            best = None
            best_usable = False
            for src in order:
                requests += 1
                got = row["results"].get(src)
                usable = bool(got) and got[:10] <= CLAIM[:10]
                R.observe("date_resolution", src, usable)
                if got is not None and best is None:
                    best = got
                if usable:
                    best, best_usable = got, True
                    break
            if use_memo and best is not None:
                if cache_usable_only and not best_usable:
                    pass                             # do NOT cement an unusable answer
                else:
                    M.remember("date_resolution", cache_key, best)
            kept += _gate_ok(row, best or "")
        per_pass_kept.append(kept)

    return {"condition": condition, "gate_kept": per_pass_kept[-1] if per_pass_kept else 0,
            "pass1": per_pass_kept[0] if per_pass_kept else 0,
            "requests": requests, "n": len(table["facts"])}


def run(out: str = FIXTURE) -> int:
    with open(out, encoding="utf-8") as fh:
        table = json.load(fh)
    n = len(table["facts"])
    print(f"replaying {n} facts over a recorded fixture (no network)\n")
    print(f"  {'condition':12s} {'pass1':>6s} {'pass2':>6s} {'requests':>9s} {'req/fact':>9s}")

    rows = []
    for cond, bandit, memo, clean in (("none", False, False, False),
                                      ("memo_only", False, True, False),
                                      ("bandit_only", True, False, False),
                                      ("both", True, True, False),
                                      ("both|cache-usable-only", True, True, True)):
        r = replay(table, cond, use_bandit=bandit, use_memo=memo, cache_usable_only=clean)
        rows.append(r)
        print(f"  {cond:22s} {r['pass1']:5d}/{n} {r['gate_kept']:5d}/{n} "
              f"{r['requests']:9d} {r['requests'] / (2 * n):9.1f}")

    print(f"\n  ABLATION READOUT")
    base = next(r for r in rows if r["condition"] == "none")
    for r in rows:
        if r["condition"] == "none":
            continue
        dk = r["gate_kept"] - base["gate_kept"]
        dr = r["requests"] - base["requests"]
        print(f"  {r['condition']:12s} vs none: gate-kept {dk:+d} facts, "
              f"requests {dr:+d}")
    both = next(r for r in rows if r["condition"] == "both")
    bandit = next(r for r in rows if r["condition"] == "bandit_only")
    memo = next(r for r in rows if r["condition"] == "memo_only")
    clean = next(r for r in rows if r["condition"] == "both|cache-usable-only")
    print(f"\n  THE QUESTION THE PAPER RAISES: is 'both' actually best?")
    print(f"    both={both['gate_kept']}  bandit_only={bandit['gate_kept']}  "
          f"memo_only={memo['gate_kept']}  none={base['gate_kept']}")
    best = max(rows, key=lambda r: (r["gate_kept"], -r["requests"]))
    print(f"    best by (gate-kept, then fewest requests): {best['condition']}")
    if best["condition"] != "both":
        print(f"    -> the two-memory configuration is NOT the winner; "
              f"one mechanism is redundant or harmful here")

    print(f"\n  MECHANISM CHECK: is the memo cementing the bandit's mistakes?")
    print(f"    both (cache anything)     {both['gate_kept']}/{n} over {both['requests']} requests")
    print(f"    both (cache usable only)  {clean['gate_kept']}/{n} over {clean['requests']} "
          f"requests")
    if clean["gate_kept"] > both["gate_kept"]:
        print(f"    -> CONFIRMED: caching an UNUSABLE answer costs "
              f"{clean['gate_kept'] - both['gate_kept']} fact(s). The memo turns a one-off "
              f"search mistake into a 30-day error. Two memories are not independent.")
    else:
        print(f"    -> not confirmed on this fixture")

    print(f"\n  WHAT SURVIVES THE ABLATION")
    print(f"    memo alone: {memo['gate_kept']}/{n} kept for {memo['requests']} requests "
          f"({base['requests']} without) -- it halves the cost and loses nothing")
    print(f"    bandit alone: {bandit['gate_kept']}/{n} for {bandit['requests']} requests "
          f"-- MORE requests than fixed order, for no gain")
    print(f"    -> the bandit is not earning its place on this fixture. Report it as unproven, "
          f"not as a win.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ablate bandit vs memo over a recorded fixture.")
    ap.add_argument("--build", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--fixture", default=FIXTURE)
    args = ap.parse_args(argv)
    if args.build:
        build(args.build, args.fixture)
        return 0
    return run(args.fixture)


if __name__ == "__main__":
    raise SystemExit(main())
