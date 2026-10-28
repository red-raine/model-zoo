"""logos/numeric_signal_eval.py — does a numeric disagreement actually PREDICT a false claim?

THE QUESTION, AND WHY IT IS THE ONE THAT MATTERS
------------------------------------------------
`numeric_conflict` was validated against five triples I wrote. Then against 200 real rows it produced ten
false drops, and after fixes it produces flags instead of drops. But nobody has asked the question that
decides whether the flag is worth anything:

    when numeric_conflict says the claim and the evidence disagree about a number,
    is the claim actually FALSE more often than chance?

AVeriTeC has human gold verdicts, so this is answerable with real labels rather than constructed ones:

    gold REFUTED    the evidence contradicts the claim      <- a numeric conflict SHOULD predict this
    gold SUPPORTED  the evidence establishes the claim      <- a numeric conflict SHOULD NOT fire here

The other two classes (NOT_ENOUGH_EVIDENCE, CONFLICTING) are excluded from the confusion matrix because
neither is a clean negative: an absence of evidence is not evidence the numbers agree.

WHAT IS MEASURED
    opportunity   how many rows even HAVE a comparable pair of quantities (the detector cannot speak
                  about a row with no numbers, and that denominator is the honest one)
    precision     of the rows it flags, what fraction really are REFUTED
    recall        of the REFUTED rows, what fraction it flags
    lift          precision against the base rate, because 61% of this corpus is REFUTED and a detector
                  that flags everything scores 0.61 precision while being worthless
    by severity   whether severity=high is better than severity=low, or whether the severity is decoration

dev is NOT held out -- thresholds were fitted on it elsewhere in this project. That limits this to "the signal
carries information", not "the signal is calibrated".

USAGE
    python tools/logos/numeric_signal_eval.py --self-test
    python tools/logos/numeric_signal_eval.py --measure [--limit 500]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

CORPUS = os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")


def _confusion(flags, gold, positive_gold="REFUTED", negative_gold="SUPPORTED") -> dict:
    """2x2 with the standard names. Returns counts plus precision/recall/base rate, all None-safe.

    A detector with no flags has undefined precision, and reporting 0.0 there would look like "wrong every
    time" instead of "said nothing". None is the honest value and the caller decides what to do with it.
    """
    tp = fp = fn = tn = 0
    for f, g in zip(flags, gold):
        if g == positive_gold:
            if f:
                tp += 1
            else:
                fn += 1
        elif g == negative_gold:
            if f:
                fp += 1
            else:
                tn += 1
    n_pos = tp + fn
    n_neg = fp + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pos": n_pos, "n_neg": n_neg,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / n_pos) if n_pos else None,
        "fpr": (fp / n_neg) if n_neg else None,
        "base_rate": (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) else None,
    }


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval. Chosen over the normal approximation because it behaves at small n and at the
    extremes, which is exactly where a detector like this lives: 13/18 and 0/25 are the shapes that occur."""
    if n == 0:
        return (None, None)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _lift(c: dict):
    """Precision / base rate. 1.0 means the flag is no better than flagging everything."""
    if c["precision"] is None or not c["base_rate"]:
        return None
    return c["precision"] / c["base_rate"]


def evaluate(limit: int = 500, corpus: str = CORPUS) -> int:
    from logos import corpus_sizing, numeric_conflict

    if not os.path.exists(corpus):
        print(f"corpus not found: {corpus}")
        return 1
    rows = corpus_sizing.load_corpus(corpus, limit)
    print(f"=== does a numeric disagreement predict a false claim? ({len(rows)} real rows) ===")
    print(f"  {corpus}")
    gold_all = collections.Counter(r["gold"] for r in rows)
    print(f"  gold distribution: {dict(gold_all)}\n")

    rows = [r for r in rows if r["evidence"]]
    flags, sev, gold, n_disagree = [], [], [], []
    opportunity = 0
    for r in rows:
        nc = numeric_conflict.conflicts(r["claim"], r["evidence"])
        flags.append(bool(nc["conflict"]))
        sev.append(nc.get("severity", "none"))
        n_disagree.append(len(nc.get("disagreements") or []))
        gold.append(r["gold"])
        # opportunity = at least one quantity on each side could be compared at all
        if nc.get("compared"):
            opportunity += 1

    n = len(rows)
    print(f"  rows with evidence: {n}")
    print(f"  rows where any two quantities were COMPARABLE: {opportunity} ({opportunity / n:.1%})")
    print(f"  rows flagged as conflicting: {sum(flags)} ({sum(flags) / n:.1%})")
    print(f"  severity: {dict(collections.Counter(sev))}\n")

    overall = _confusion(flags, gold)
    print("=== the confusion matrix, on the gold REFUTED vs SUPPORTED rows ===")
    print(f"  {'':22s} {'gold REFUTED':>13s} {'gold SUPPORTED':>15s}")
    print(f"  {'flagged':22s} {overall['tp']:13d} {overall['fp']:15d}")
    print(f"  {'not flagged':22s} {overall['fn']:13d} {overall['tn']:15d}")
    print()
    prec = overall["precision"]
    rec = overall["recall"]
    fpr = overall["fpr"]
    base = overall["base_rate"]
    lift = _lift(overall)
    print(f"  base rate (REFUTED share) {base:.3f}" if base is not None else "  base rate: n/a")
    print(f"  precision                 {prec:.3f}" if prec is not None else "  precision: UNDEFINED (no flags)")
    print(f"  recall                    {rec:.3f}" if rec is not None else "  recall: n/a")
    print(f"  false-positive rate       {fpr:.3f}" if fpr is not None else "  fpr: n/a")
    print(f"  LIFT over base rate       {lift:.2f}x" if lift is not None else "  lift: n/a")
    if prec is not None:
        lo, hi = wilson(overall["tp"], overall["tp"] + overall["fp"])
        spans = base is not None and lo <= base <= hi
        print(f"  precision 95% CI          [{lo:.2f}, {hi:.2f}] on {overall['tp'] + overall['fp']} flags"
              f"{'  <- the base rate sits INSIDE it, so no signal' if spans else ''}")

    print("\n=== is the SEVERITY doing anything, or is it decoration? ===")
    for level in ("high", "low"):
        sel = [f for f, s in zip(flags, sev) if s == level]
        if not sel:
            print(f"  severity={level:5s} no rows")
            continue
        sub = [f for f, s in zip(flags, sev) if s == level]
        subgold = [g for g, s in zip(gold, sev) if s == level]
        c = _confusion(sub, subgold)
        p = f"{c['precision']:.3f}" if c["precision"] is not None else "n/a"
        print(f"  severity={level:5s} {len(sub):4d} rows   precision {p}"
              f"   of those, {c['tp']} REFUTED / {c['fp']} SUPPORTED")

    print("\n=== THE HONEST READING ===")
    if lift is None:
        print("  The detector never fired, so it has no measurable signal on this corpus. That is not a")
        print("  verdict on the mechanism -- it is a statement that prose claims and prose evidence rarely")
        print("  state the same quantity, which is exactly what the opportunity rate above shows.")
    elif lift >= 1.5:
        print(f"  A numeric disagreement raises the chance the claim is REFUTED by {lift:.2f}x. The signal")
        print(f"  carries real information and is worth flagging for review -- but at precision {prec:.2f} it")
        print(f"  is NOT worth dropping a row on, which is why the pipeline only flags from prose.")
    elif lift < 1.2:
        if prec is not None:
            lo, hi = wilson(overall["tp"], overall["tp"] + overall["fp"])
        else:
            lo, hi = (None, None)
        inside = base is not None and lo is not None and lo <= base <= hi
        print(f"  LIFT IS {lift:.2f}x -- NO USABLE SIGNAL. Precision {prec:.2f} against a base rate of "
              f"{base:.2f},")
        if inside:
            print(f"  and the base rate sits INSIDE the precision CI [{lo:.2f}, {hi:.2f}]. On "
                  f"{overall['tp'] + overall['fp']} flags the detector is")
            print(f"  indistinguishable from flagging at random, so its output should not enter a review")
            print(f"  queue at all. That is a null result and it is worth as much as a positive one: the")
            print(f"  4/5-with-0-false-flags figure came from five pairs I designed to be catchable.")
        else:
            print(f"  the CI excludes the base rate, so there may be a weak signal -- but {overall['tp'] + overall['fp']}")
            print(f"  flags is too few to act on, and recall is low enough that the queue would be nearly")
            print(f"  empty anyway.")
    else:
        print(f"  LIFT IS {lift:.2f}x -- BELOW 1.0, meaning a flag makes REFUTED LESS likely than the base")
        print(f"  rate. The numeric signal is anti-correlated with falsity on this corpus, and the flag is")
        print(f"  actively misleading. That is worth knowing before building on it.")

    if rec is not None and rec < 0.15:
        print(f"\n  Recall {rec:.3f}: it misses almost every false claim, because most REFUTED rows are")
        print(f"  contradicted by FACTS (a quote, a person, a decision) rather than by numbers. Numeric")
        print(f"  verification is a narrow instrument and this measures how narrow.")

    print("\n=== limits ===")
    print("  dev is NOT held out -- thresholds elsewhere in this project were fitted on it.")
    print("  AVeriTeC evidence is Q/A prose, so fewer quantities line up than in a numeric corpus. The")
    print("  opportunity rate above is the denominator this signal can ever serve.")
    print("  Conflicting/NotEnoughEvidence rows are excluded: absence of a numeric conflict is not")
    print("  evidence that the numbers agree.")
    return 0


def run_self_test() -> bool:
    print("logos.numeric_signal_eval - SELF TEST")
    ok = True

    # --- a perfect detector
    c = _confusion([True, True, False, False], ["REFUTED", "REFUTED", "SUPPORTED", "SUPPORTED"])
    good = c["precision"] == 1.0 and c["recall"] == 1.0 and c["fpr"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a perfect detector scores precision 1.0 and recall 1.0")

    # --- a detector that flags everything: perfect recall, useless precision
    c = _confusion([True] * 4, ["REFUTED", "REFUTED", "SUPPORTED", "SUPPORTED"])
    good = c["recall"] == 1.0 and c["precision"] == 0.5 and _lift(c) == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] flagging EVERYTHING gets recall 1.0, precision = base rate, "
          f"and lift 1.00 -- which is why lift is reported")

    # --- a detector that never fires has UNDEFINED precision, not 0.0
    c = _confusion([False] * 4, ["REFUTED", "REFUTED", "SUPPORTED", "SUPPORTED"])
    good = c["precision"] is None and c["recall"] == 0.0 and _lift(c) is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a silent detector reports precision None, not 0.0 "
          f"(it said nothing; it did not say 'no')")

    # --- an anti-correlated detector is detectable: flagging only the SUPPORTED rows gives lift 0
    c = _confusion([False, False, True, True], ["REFUTED", "REFUTED", "SUPPORTED", "SUPPORTED"])
    good = c["precision"] == 0.0 and _lift(c) == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a detector that fires only on TRUE claims scores lift 0.00")

    # --- excluded classes must not leak into either cell
    c = _confusion([True], ["NOT_ENOUGH_EVIDENCE"])
    good = (c["tp"], c["fp"], c["fn"], c["tn"]) == (0, 0, 0, 0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an ambiguous gold class lands in no cell")

    if not os.path.exists(CORPUS):
        print(f"  [SKIP] corpus not present: {CORPUS}")
    else:
        rows = __import__("logos.corpus_sizing", fromlist=["x"]).load_corpus(CORPUS, 3)
        good = len(rows) == 3 and all("gold" in r for r in rows)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] loads real rows with gold labels")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Does a numeric disagreement predict a false claim?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return evaluate(args.limit, args.corpus)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
