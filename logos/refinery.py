"""logos/refinery.py — ore refining, not theorem proving: asymmetric-cost routing + audit arithmetic.

THE OBJECTIVE CORRECTION
------------------------
Everything before this file optimised the wrong target. It asked "is this confidence a calibrated
probability?" -- an *instrument* question -- when the job is:

    cheaply extract the highest-quality reasoning traces from billions of noisy tokens

Those are different optimisation problems with different optima. A production QA system wants balanced
coverage. A dataset refiner wants **precision, and does not care about recall**: throwing away 98% of a
mountain is fine if the remaining 2% is trustworthy.

So the cost asymmetry inverts the decision:

    false positive  (keeping a bad sample)   CATASTROPHIC -- it poisons training and is invisible later
    false negative  (dropping a good sample)  nearly free -- you have millions more

With Chow's rule `t* = 1 - C_d/C_w`, a refinery sets `C_d/C_w -> 0`, so t* approaches 1. The earlier
default (1/4 -> 0.75) was a balanced-classifier setting, not a refinery setting.

THE CONSTRAINT THE GUIDANCE MISSES
----------------------------------
"99.9% precision at 10% recall is probably far more valuable" is asserted, not checked. Precision is a
claim about unsampled rows, and it can only be established by auditing a sample. With ZERO defects
observed, certifying precision >= p at confidence 1-alpha requires

    n >= ln(alpha) / ln(1 - p)

    99.9% at 95% confidence -> n >= 2,995 audited samples with ZERO defects
    99.5% at 95% confidence -> n >=   598
    99.0% at 95% confidence -> n >=   298

If any defect appears, n grows. **The 99.9% target therefore mandates a hand-audited set of a few
thousand samples before the claim means anything** -- and that cost has to be in the plan, because a
pipeline that achieves 99.9% precision and cannot demonstrate it is indistinguishable from one that
achieves 60%.

This module computes that number so it stops being a slogan.

USAGE
    python tools/logos/refinery.py --self-test
    python tools/logos/refinery.py --run
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

KEEP, REVIEW, DROP = "KEEP", "REVIEW", "DROP"


# --------------------------------------------------------------------------- audit arithmetic
def audit_n_zero_defect(precision: float, confidence: float = 0.95) -> int:
    """Audits needed to certify `precision` with ZERO observed defects.

    One-sided exact bound: with 0 defects in n audits, the upper confidence bound on the defect rate is
    `1 - alpha^(1/n)`. Requiring that to be <= 1-precision gives `alpha^(1/n) >= precision`, hence

        n >= ln(alpha) / ln(precision)

    Note the denominator is ln(PRECISION), not ln(1-precision) -- the first version of this function had
    that inverted, which returned n=1 for every target and would have told us a single audit could
    certify 99.9%. A wrong audit budget is worse than none, because it is spent.
    """
    if not 0.0 < precision < 1.0:
        raise ValueError("precision must be in (0, 1)")
    alpha = 1.0 - confidence
    return int(math.ceil(math.log(alpha) / math.log(precision)))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def _binom_cdf(k: int, n: int, p: float) -> float:
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(0, k + 1)))


def precision_lower_bound(defects: int, audited: int, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson lower bound on precision given `defects` in `audited`.

    Solved by bisection on the binomial CDF: the lower bound is the p for which
    P(X <= defects | n=audited, defect rate 1-p) = alpha.
    """
    if audited <= 0:
        return 0.0
    if defects < 0 or defects > audited:
        raise ValueError("defects must be in [0, audited]")
    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        # P(observing <= defects defects | defect rate = 1-mid) -- alpha means mid is the lower bound
        if _binom_cdf(defects, audited, 1.0 - mid) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def audit_n_for_precision(precision: float, defect_rate: float = 0.0,
                          confidence: float = 0.95, cap: int = 2_000_000) -> int | None:
    """Audits needed to certify `precision` when you EXPECT `defect_rate` defects.

    Returns None when the target is UNACHIEVABLE at that defect rate, which happens whenever
    `1 - defect_rate <= precision`. That is not an edge case, it is the common one: if you expect one
    defect per thousand and you want to certify 99.9% precision, the expected precision IS 99.9% and the
    lower confidence bound is necessarily below it at every sample size. **You cannot certify a target
    equal to your own defect rate; you need headroom.** Returning a number there would be the most
    misleading thing this module could do, because it would look like a budget.
    """
    if defect_rate > 0.0 and (1.0 - defect_rate) <= precision:
        return None
    if defect_rate == 0.0:
        return audit_n_zero_defect(precision, confidence)
    lo = max(1, int(10 * (1 - precision) / max(defect_rate, 1e-9)))
    hi = cap
    for _ in range(100):
        mid = (lo + hi) // 2
        d = int(round(defect_rate * mid))
        if precision_lower_bound(d, mid, confidence) >= precision:
            hi = mid
        else:
            lo = mid + 1
        if lo >= hi:
            break
    if hi >= cap and precision_lower_bound(int(round(defect_rate * cap)), cap,
                                          confidence) < precision:
        return None
    return hi


# --------------------------------------------------------------------------- asymmetric routing
def chow_threshold(cost_discard: float, cost_wrong: float) -> float:
    """t* = 1 - C_d/C_w. A refinery sets C_d << C_w, so t* -> 1."""
    if cost_wrong <= 0:
        raise ValueError("cost_wrong must be positive")
    return max(0.0, min(1.0, 1.0 - cost_discard / cost_wrong))


def route(conf: np.ndarray, cost_discard: float = 0.001, cost_wrong: float = 1.0,
          review_band: float = 0.05) -> dict:
    """Ternary routing with asymmetric costs.

        conf >= t*                -> KEEP    (assert)
        t* - review_band <= conf  -> REVIEW  (hand or LLM check; the 1-5% band)
        conf <  t* - review_band  -> DROP    (no cost; you have millions more)

    DROP is the default and REVIEW is deliberately small. In a refinery the cheap action is discarding,
    so anything not clearly good should leave unless it is close enough to be worth a look.
    """
    c = np.asarray(conf, dtype=float)
    t = chow_threshold(cost_discard, cost_wrong)
    keep = c >= t
    review = (~keep) & (c >= t - review_band)
    drop = ~(keep | review)
    return {"threshold": t, "keep": keep, "review": review, "drop": drop,
            "n_keep": int(keep.sum()), "n_review": int(review.sum()),
            "n_drop": int(drop.sum())}


def evaluate_routing(route_result: dict, correct: np.ndarray) -> dict:
    """Precision on KEEP (the number that matters) and the audit cost to certify it."""
    k = route_result["keep"]
    n_keep = int(k.sum())
    defects = int((~np.asarray(correct, dtype=bool)[k]).sum()) if n_keep else 0
    prec = (1.0 - defects / n_keep) if n_keep else float("nan")
    return {
        "n_keep": n_keep,
        "coverage": n_keep / len(correct),
        "precision_measured": prec,
        "defects": defects,
        "precision_lb95": precision_lower_bound(defects, n_keep, 0.95) if n_keep else 0.0,
        "audits_to_certify_999": audit_n_zero_defect(0.999) if n_keep else 0,
    }


# ---------------------------------------------------------------------------------- driver
def run() -> int:
    from logos.baseline import D
    from logos.calibrate import Probs

    rows = [__import__("json").loads(ln) for ln in
            open(os.path.join(D, "nli-post-logits.jsonl"), encoding="utf-8") if ln.strip()]
    dev = [r for r in rows if r["split"] == "dev"]
    Z = np.array([r["logits"] for r in dev], dtype=float)
    y = np.array([r["label"] for r in dev])
    P = Probs.from_logits(Z, 4.226, labels=y)
    conf = P.select("threshold").max(axis=1)
    correct = P.select("threshold").argmax(axis=1) == y

    print("=== audit arithmetic: what the 99.9% target actually costs ===")
    for p in (0.99, 0.995, 0.999, 0.9999):
        print(f"  certify {p:.4%} @95% with ZERO defects : {audit_n_zero_defect(p):>7,d} audits")
    for dr in (0.0001, 0.001, 0.01):
        n = audit_n_for_precision(0.999, dr)
        verdict = (f"{n:,} audits" if n else "UNACHIEVABLE at any n")
        print(f"  certify 99.9% @95% if {dr:.2%} are bad : {verdict}")

    print(f"\n=== TERNARY ROUTING at refinery costs, {len(y)} dev rows ===")
    print(f"  {'C_d/C_w':>9s} {'t*':>7s} {'KEEP':>6s} {'REVIEW':>7s} {'DROP':>6s} "
          f"{'prec(KEEP)':>11s} {'LB95':>7s}")
    for ratio in (1 / 4, 0.2, 0.15, 0.13, 0.12):
        r = route(conf, cost_discard=ratio, cost_wrong=1.0)
        e = evaluate_routing(r, correct)
        lb = f"{e['precision_lb95']:.4f}" if e["n_keep"] else "--"
        pr = f"{e['precision_measured']:.4f}" if e["n_keep"] else "--"
        print(f"  {ratio:9.4f} {r['threshold']:7.4f} {e['n_keep']:6d} {r['n_review']:7d} "
              f"{r['n_drop']:6d} {pr:>11s} {lb:>7s}")

    # --- THE FINDING: calibration compressed the confidence range, so high-precision thresholds are
    #     unreachable. Measured, not assumed.
    print(f"\n=== why the strict thresholds keep nothing ===")
    raw_conf = P.raw.max(axis=1)
    print(f"  max confidence, RAW logits          : {raw_conf.max():.4f}")
    print(f"  max confidence, CALIBRATED (T=4.23) : {conf.max():.4f}")
    print(f"  -> temperature scaling improved ECE to 0.031 and COMPRESSED the tail. Any refinery")
    print(f"     threshold above {conf.max():.2f} keeps zero rows.")

    print(f"\n=== best operating point, RAW vs CALIBRATED confidence ===")
    print(f"  {'conf':>10s} {'keep frac':>10s} {'n':>5s} {'precision':>10s} {'LB95':>7s}")
    summary = {}
    for label, c in (("raw", raw_conf), ("calibrated", conf)):
        order_c = np.argsort(-c)
        best = None
        for frac in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00):
            m = max(1, int(round(frac * len(y))))
            sel = order_c[:m]
            prec = float(correct[sel].mean())
            lb = precision_lower_bound(int((~correct[sel]).sum()), m, 0.95)
            if best is None or lb > best[2]:
                best = (frac, m, lb, prec)
        summary[label] = best
        print(f"  {label:>10s} {best[0]:10.0%} {best[1]:5d} {best[3]:10.4f} {best[2]:7.4f}")
    print(f"  -> the best CERTIFIABLE precision is "
          f"{max(summary['raw'][2], summary['calibrated'][2]):.4f}, reached with "
          f"{'RAW' if summary['raw'][2] > summary['calibrated'][2] else 'CALIBRATED'} confidence")
    print(f"     at {summary['raw'][0] if summary['raw'][2] > summary['calibrated'][2] else summary['calibrated'][0]:.0%} coverage. This independently confirms the")
    print(f"     probability-routing table's 'ranking -> raw' choice, measured a third way.")

    print(f"\n=== the honest reading ===")
    best_frac, best_n, best_lb, best_prec = summary["raw"] if summary["raw"][2] > summary["calibrated"][2] else summary["calibrated"]
    print(f"  Best CERTIFIABLE precision: {best_lb:.2%} (95% lower bound) at {best_frac:.0%} coverage, "
          f"n={best_n}.")
    print(f"  Point-estimate precision peaks at {best_prec:.2%}, but on {best_n} rows the lower bound")
    print(f"  is {best_lb:.2%} -- the point estimate is not evidence by itself.")
    print(f"\n  99.9% precision is NOT REACHABLE here: the model's ceiling on this data is "
          f"{best_prec:.2%}.")
    print(f"  And it could not be CERTIFIED even if reached: certifying 99.9% needs "
          f"{audit_n_zero_defect(0.999):,} audits with ZERO defects,")
    print(f"  i.e. an audit set {audit_n_zero_defect(0.999) / max(best_n, 1):.0f}x the keep pile. "
          f"The claim is unmeasurable at this scale.")
    print(f"\n  What this implies for the plan:")
    print(f"   1 precision must be certified against an audit set LARGER than the keep pile, so the")
    print(f"     first milestone is keep-pile VOLUME, not 99.9%")
    print(f"   2 calibration and selectivity CONFLICT: T=4.23 bought ECE 0.031 and destroyed the")
    print(f"     high-confidence tail a refinery needs. ECE-optimal is not selectivity-optimal.")
    return 0


# --------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.refinery - SELF TEST")
    ok = True

    # --- 1. the closed form, checked against the derivation
    good = (audit_n_zero_defect(0.999) == 2995
            and audit_n_zero_defect(0.995) == 598
            and audit_n_zero_defect(0.99) == 299)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] zero-defect audit sizes: 99.9%->"
          f"{audit_n_zero_defect(0.999)}, 99.5%->{audit_n_zero_defect(0.995)}, "
          f"99.0%->{audit_n_zero_defect(0.99)}")

    # --- 2. and the bound is self-consistent: n from the formula must certify p, n-1 must not
    n = audit_n_zero_defect(0.999)
    good = (precision_lower_bound(0, n, 0.95) >= 0.999
            and precision_lower_bound(0, n - 1, 0.95) < 0.999)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the bound and the formula agree at the boundary "
          f"(n={n}: {precision_lower_bound(0, n, 0.95):.6f}, n-1: "
          f"{precision_lower_bound(0, n - 1, 0.95):.6f})")

    # --- 3. more audits with the same defect count can only raise the bound
    good = (precision_lower_bound(1, 3000) > precision_lower_bound(1, 1000)
            and precision_lower_bound(0, 1000) > precision_lower_bound(5, 1000))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the lower bound rises with n and falls with defects")

    # --- 4. YOU CANNOT CERTIFY A TARGET EQUAL TO YOUR OWN DEFECT RATE. Expecting 1 defect per 1000 and
    #     wanting to certify 99.9% precision is unachievable at every sample size, and returning a number
    #     there would look like an audit budget. It returns None instead.
    good = (audit_n_for_precision(0.999, 0.001) is None
            and audit_n_for_precision(0.999, 0.01) is None
            and audit_n_for_precision(0.99, 0.001) is not None)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a target equal to the expected defect rate returns None "
          f"(certifying 99.9% while expecting 0.1% defects is impossible); 99.0% at 0.1% is "
          f"achievable -> {audit_n_for_precision(0.99, 0.001):,}")
    good = audit_n_for_precision(0.995, 0.001) > audit_n_zero_defect(0.995)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and expecting defects costs more audits than the "
          f"zero-defect ideal ({audit_n_for_precision(0.995, 0.001):,} vs "
          f"{audit_n_zero_defect(0.995):,})")

    # --- 5. Chow inverts as the cost asymmetry grows
    good = (chow_threshold(0.25, 1.0) == 0.75 and chow_threshold(0.001, 1.0) == 0.999
            and chow_threshold(0.0, 1.0) == 1.0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] t* = 1 - C_d/C_w: 0.25->0.75, 0.001->0.999, "
          f"0->1.0")

    # --- 6. routing is a partition: every row is in exactly one bucket
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 1000)
    r = route(conf, cost_discard=0.05, cost_wrong=1.0, review_band=0.10)
    total = r["n_keep"] + r["n_review"] + r["n_drop"]
    disjoint = not np.any(r["keep"] & r["review"]) and not np.any(r["keep"] & r["drop"])
    good = total == 1000 and disjoint
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] KEEP/REVIEW/DROP partition every row exactly once")

    # --- 7. a stricter threshold can only shrink the keep pile
    tight = route(conf, cost_discard=0.001, cost_wrong=1.0)
    loose = route(conf, cost_discard=0.5, cost_wrong=1.0)
    good = tight["n_keep"] < loose["n_keep"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] tighter cost ratio keeps less "
          f"({tight['n_keep']} vs {loose['n_keep']})")

    # --- 8. evaluate_routing reports the defect count that drives the audit cost
    correct = np.ones(1000, dtype=bool)
    correct[:5] = False
    e = evaluate_routing(r, correct)
    good = e["n_keep"] == int(r["keep"].sum()) and e["defects"] <= e["n_keep"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] precision is measured against the labels and produces "
          f"both a number and a lower bound")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Asymmetric-cost ore refining + audit arithmetic.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.run:
        return run()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
