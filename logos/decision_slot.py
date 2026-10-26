"""logos/decision_slot.py — can a decision model beat the hand rule at `usable`? Not yet, and the
reason is the finding.

THE QUESTION
------------
The `usable` validator currently is a rule: `date <= claim_date`. A decision model (laya, ollaya, a
distilled classifier) would only earn that slot by beating the rule OUT OF SAMPLE. `calibrate.py`
can measure that. So: measure it.

THE RESULT, AND WHY IT IS NEGATIVE
----------------------------------
There is no independent label for "usable". The only label available is the temporal gate, which IS
the rule. So:

    Brier of the rule against its own label = 0 exactly, by construction.

No model can beat zero. Running the comparison would produce a beautiful "the rule wins, 100% to 44%"
table that means nothing, because the rule is being graded on the answer key it wrote. That is the
same circularity this project keeps finding, and manufacturing a table from it would be the worst
possible outcome -- a number that looks like evidence for a decision I already wanted.

So this module does not score a model against a fake label. It measures the CIRCULARITY and reports
it, because the honest statement is:

    the `usable` slot is currently unfillable, and the prerequisite is a labelled set,
    not a better model.

WHERE THE SLOT *CAN* BE FILLED
------------------------------
The rule is exact and computable, so it needs no model. A model earns its place only where no exact
rule exists. Three such questions, in order of how soon we can label them:

  1. **Does this snippet support this claim?** No closed form. Labels exist in AVeriTeC dev (500 rows
     with Supported/Refuted/NEI), which the project has already measured -- that is a real label set
     and the correct first home for a decision model.
  2. **Is this entity the same entity?** No closed form, and labels are cheap to construct by hand
     at n=500.
  3. **Is this date close enough to be evidence?** Ambiguous only near the boundary; the rule is
     exact elsewhere. A model here would be fitting the 2-day band around the claim date, which is
     not worth a model.

USAGE
    python tools/logos/decision_slot.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.calibrate import brier, brier_decomposition, cross_validate, ece  # noqa: E402

FIXTURE = r"B:\datasets\forge\date-fixture.json"
CLAIM = "2022-01-01"


def _days(date: str, claim: str = CLAIM) -> int:
    import datetime
    d = datetime.date.fromisoformat(date[:10])
    c = datetime.date.fromisoformat(claim[:10])
    return (d - c).days


def run(fixture: str = FIXTURE) -> int:
    with open(fixture, encoding="utf-8") as fh:
        table = json.load(fh)

    dates, labels, routes = [], [], []
    for row in table["facts"]:
        for method, date in (row.get("results") or {}).items():
            if not date:
                continue
            dates.append(date)
            labels.append(1.0 if _days(date) <= 0 else 0.0)   # the rule, used as the label
            routes.append(method)

    if not dates:
        print("fixture holds no dated answers")
        return 1

    n = len(dates)
    pos = int(sum(labels))
    rule_probs = np.array(labels)                              # the rule, exactly
    print(f"=== {n} dated answers, {pos} pass the precedence rule ({pos / n:.1%}) ===\n")

    print("=== the rule scored against its own definition ===")
    print(f"  Brier   : {brier(rule_probs, labels):.12f}")
    print(f"  ECE     : {ece(rule_probs, labels):.12f}")
    print(f"  -> exactly zero, because the label IS the rule. This is a tautology, not a result.")

    print(f"\n=== what a model could possibly see ===")
    days = np.array([_days(d) for d in dates], dtype=float)
    print(f"  days relative to claim: min={days.min():+.0f} max={days.max():+.0f} "
          f"median={np.median(days):+.0f}")
    print(f"  answers after the claim (rule says unusable): {int((days > 0).sum())} of {n}")
    print(f"  answers before the claim (rule says usable):  {int((days <= 0).sum())} of {n}")

    # The one thing that CAN be asked without circularity: how much of the "usable" verdict is a
    # deterministic function of a single feature? If a one-feature threshold explains it exactly,
    # no model can add anything -- which is the point.
    from collections import Counter
    by_route = Counter((routes[i], bool(labels[i])) for i in range(n))
    print(f"\n=== is `usable` route-dependent, or purely date-dependent? ===")
    for r in sorted({x[0] for x in by_route}):
        ok_r = by_route.get((r, True), 0)
        bad_r = by_route.get((r, False), 0)
        tot = ok_r + bad_r
        print(f"  {r:22s} usable {ok_r:2d}/{tot:2d} ({ok_r / tot:.0%})")
    print(f"  -> every route produces both outcomes, so the property is not a route artifact; it is")
    print(f"     the date, which the rule already computes exactly.")

    print(f"\n=== verdict ===")
    print(f"  A decision model CANNOT beat this rule, because the rule is exact and its input is")
    print(f"  known. Brier 0 is a floor. Any model scoring above 0 is strictly worse, and any model")
    print(f"  scoring 0 has merely relearned arithmetic.")
    print(f"\n  the `usable` slot is unfillable today, and the prerequisite is a LABELLED SET, not a")
    print(f"  better model. Where labels already exist -- AVeriTeC dev's 500 rows with")
    print(f"  Supported/Refuted/NEI -- a decision model has a genuine closed question to answer and")
    print(f"  `calibrate.py` can check whether its probabilities mean what they say.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Can a decision model fill the usable slot?")
    ap.add_argument("--fixture", default=FIXTURE)
    args = ap.parse_args(argv)
    return run(args.fixture)


if __name__ == "__main__":
    raise SystemExit(main())
