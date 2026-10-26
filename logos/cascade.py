"""logos/cascade.py — the cascading envelope: every method votes, the arbiter decides.

THE SHAPE
---------
Not one method, and not a router that wins by default. Each route that answers becomes a
**candidate**; its confidence is its own track record (the arbiter's Beta posterior for
`date_precedence`), not a hand-set number. The arbiter then returns one of:

    keep        a date clears the Chow bar and is asserted
    escalate    methods disagree past the bar -> go search / ask an LLM
    unknowable  no method can address this claim kind -> REMOVE OR REWRITE the fact

The last one is the disposition the whole build was missing. A fact nobody can settle should not sit
in the corpus looking settled; it should leave, or be rewritten into a claim that can be settled.

THE HONEST PROBLEM WITH THIS EVALUATION
---------------------------------------
Learning needs a label. We have no ground-truth publication dates for these bridges, so the temporal
gate is used as a proxy outcome. That is circular: the arbiter learns to predict *which route
produces gate-passing dates*, which is useful for route selection, but it is NOT learning truth and
must never be reported as if it were. Both the cold-prior and the learned pass are reported, and the
circularity is stated next to the numbers.

USAGE
    python tools/logos/cascade.py --fixture B:\\datasets\\forge\\date-fixture.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.arbiter import ESCALATE, KEEP, UNKNOWABLE, Arbiter, Candidate  # noqa: E402

CLAIM_KIND = "date_precedence"
FIXTURE = r"B:\datasets\forge\date-fixture.json"
CLAIM = "2022-01-01"


def _usable(date: str | None) -> bool:
    return bool(date) and date[:10] <= CLAIM[:10]


def candidates_for(row: dict, arbiter: Arbiter, use_records: bool) -> list[Candidate]:
    """One candidate per route that answered. Confidence is the route's own record.

    `answers` carries the relevance fact, which capability alone cannot express: a capture date that
    postdates the claim is a perfectly good capture date and a useless piece of precedence evidence.
    """
    out = []
    for method, date in (row.get("results") or {}).items():
        if not date:
            continue
        conf = arbiter.arm(CLAIM_KIND, method).mean if use_records else 0.5
        out.append(Candidate(method=method, value=date, confidence=conf,
                             claim_kind=CLAIM_KIND, claims=(CLAIM_KIND,),
                             answers=_usable(date)))
    return out


def run(fixture: str = FIXTURE, passes: int = 6) -> int:
    with open(fixture, encoding="utf-8") as fh:
        table = json.load(fh)
    facts = table["facts"]

    for label, use_records in (("COLD PRIORS (nothing learned)", False),
                               ("LEARNED (proxy label = temporal gate, CIRCULAR)", True)):
        A = Arbiter(state_path=os.path.join(os.path.expandvars("%TEMP%"), "cascade_arb.json"),
                    seed=7)
        # rebuild cold each pass set so the two reports are comparable
        A.arms.clear()

        print(f"\n=== {label} ===")
        for p in range(passes):
            disp = {KEEP: 0, ESCALATE: 0, UNKNOWABLE: 0}
            kept_facts = 0
            for row in facts:
                if (row.get("point_in_time") or "")[:10]:
                    # tier A needs no arbitration: the publisher stated it
                    kept_facts += _usable(row["point_in_time"])
                    continue
                cands = candidates_for(row, A, use_records)
                d = A.decide(cands, CLAIM_KIND)
                disp[d.disposition] = disp.get(d.disposition, 0) + 1
                if d.disposition == KEEP and _usable(d.value):
                    kept_facts += 1
                # learn from the gate as a PROXY: each capable method is scored on whether its own
                # answer passed. Circular, and labelled as such above.
                if use_records and cands:
                    A.observe_decision(CLAIM_KIND, cands,
                                       next((c.value for c in cands if _usable(c.value)), None))
            print(f"  pass {p + 1}: keep={disp[KEEP]:2d} escalate={disp[ESCALATE]:2d} "
                  f"unknowable={disp[UNKNOWABLE]:2d}  |  dates asserted & gate-passing: "
                  f"{kept_facts}/{len(facts)}")

        if use_records:
            print(f"\n  learned route reliability for {CLAIM_KIND}:")
            for r in sorted(({"m": m, "mean": a.mean, "n": a.observations}
                             for m, a in A.arms.get(CLAIM_KIND, {}).items()),
                            key=lambda r: -r["mean"]):
                print(f"    {r['m']:22s} mean={r['mean']:.3f}  n={r['n']}")

    print(f"\n=== WHAT THE CASCADE SENDS BACK ===")
    print(f"  numbers above are gate-passing dates, NOT truth. The learned column uses the gate as")
    print(f"  its own label, which is circular by construction -- it measures route selection, not")
    print(f"  correctness. Reported together so neither can be quoted alone.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cascading envelope over the dating methods.")
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--passes", type=int, default=6)
    args = ap.parse_args(argv)
    return run(args.fixture, args.passes)


if __name__ == "__main__":
    raise SystemExit(main())
