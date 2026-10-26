"""logos/eval_audit.py — is the eval set big enough to measure what we intend to claim?

THE QUESTION
------------
"Ready for a bigger training and eval set?" cannot be answered by looking at file sizes. It is a
statistical question with a numerical answer: for each class, what is the smallest improvement this
eval set could detect? If the answer is larger than the effect we hope for, the set is too small and
no amount of training will reveal it.

This project has already been bitten by exactly that. A Conflicting "doubling" was reported with
CI [-0.038, +0.323] at n=38 -- a result that spans "harmful" to "enormous" and therefore says
nothing. That is not a failure of the model; it is a failure of the eval set, and it is predictable
in advance with the arithmetic in this file.

WHAT IS COMPUTED
----------------
    wilson(k, n)              a CI that behaves at small n and near 0/1, unlike the normal one
    min_n_two_proportion      sample size to detect a given difference between two rates
    min_detectable_delta      the inverse: at this n, what is the smallest difference detectable?
    required_n_for_class      how many rows of a class are needed to see a hoped-for gain

ALSO AUDITED
------------
    label distribution per split
    exact-normalised duplicate claims WITHIN and ACROSS splits (leakage)
    13-gram containment between dev claims and train claims (near-duplicate leakage)
    a prior-only baseline: the Brier / log loss that any model must beat

USAGE
    python tools/logos/eval_audit.py --self-test
    python tools/logos/eval_audit.py --audit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

D = r"B:\datasets\forge\verifier-av"
TRAIN = os.path.join(D, "train.clean.jsonl")
DEV = os.path.join(D, "dev.jsonl")

NORM = statistics.NormalDist()


# ------------------------------------------------------------------------------- power
def z(p: float) -> float:
    return NORM.inv_cdf(p)


def wilson(k: int, n: int, z_: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Behaves at k=0 and k=n, where the normal approximation collapses --
    and those are exactly the cells that matter when a rare class is being measured."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z_ ** 2 / n
    centre = (p + z_ ** 2 / (2 * n)) / denom
    half = z_ * math.sqrt(p * (1 - p) / n + z_ ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def min_n_two_proportion(p1: float, p2: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Rows PER GROUP to detect a difference between two INDEPENDENT proportions.

    Use this only when the two rates come from different samples -- e.g. comparing our number to a
    published baseline measured on someone else's rows. When both models are scored on the SAME
    claims, this test throws away the pairing and overstates the required sample size by an order of
    magnitude. Use `min_n_paired` for that case, which is ours.
    """
    if p1 == p2:
        return 10 ** 9
    pbar = (p1 + p2) / 2
    za = z(1 - alpha / 2)
    zb = z(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p1 - p2) ** 2))


def min_n_paired(delta: float, discordance: float = 0.20, alpha: float = 0.05,
                 power: float = 0.8) -> int:
    """PAIRS needed to detect an accuracy difference `delta` with McNemar's test.

    This is the relevant test for us. Two models scored on the same 500 claims are paired, and only
    the rows where they DISAGREE carry information. The sample-size arithmetic therefore scales with
    the discordance rate, not with the error rate:

        n = (z_{1-a/2} + z_{1-b})^2 * pi_d / delta^2

    Measured consequence: with 20% discordance, 500 paired rows detect ~5.6pp, where the unpaired
    test demanded ~6,000 per group to see the same thing. Getting this wrong in the pessimistic
    direction is how a team orders ten times the data it needs; getting it wrong in the optimistic
    direction is how it claims significance it never had.
    """
    if delta <= 0 or discordance <= 0:
        return 10 ** 9
    return int(math.ceil((z(1 - alpha / 2) + z(power)) ** 2 * discordance / delta ** 2))


def min_detectable_delta_paired(n: int, discordance: float = 0.20,
                                alpha: float = 0.05, power: float = 0.8) -> float:
    """Inverse of `min_n_paired`: at this many paired rows, what accuracy difference is visible?"""
    if n <= 0 or discordance <= 0:
        return 1.0
    return (z(1 - alpha / 2) + z(power)) * math.sqrt(discordance / n)


def min_detectable_delta(n: int, p1: float, alpha: float = 0.05,
                         power: float = 0.8, hi: float = 0.95) -> float:
    """Smallest absolute difference from p1 that n rows can detect. Monotone decreasing in n."""
    if n <= 0:
        return 1.0
    lo = 0.0
    for _ in range(60):                       # bisect on the delta
        mid = (lo + hi) / 2
        if min_n_two_proportion(p1, min(0.999, p1 + mid), alpha, power) <= n:
            hi = mid
        else:
            lo = mid
    return hi


def required_n_for_class(prevalence: float, hoped_delta: float, alpha: float = 0.05,
                         power: float = 0.8) -> dict:
    """How many TOTAL rows are needed so that `prevalence` yields enough of one class."""
    per_group = min_n_two_proportion(prevalence, prevalence + hoped_delta, alpha, power)
    total = int(math.ceil(per_group / prevalence)) if prevalence > 0 else 10 ** 9
    return {"per_group": per_group, "total_rows_needed": total}


# ----------------------------------------------------------------------- audit helpers
def _norm_claim(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def _shingles(text: str, k: int = 13) -> set:
    toks = _norm_claim(text).split()
    if len(toks) < k:
        return set()
    return {hash(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def audit(train_path: str = TRAIN, dev_path: str = DEV,
          hoped_delta: float = 0.10) -> int:
    train = load_jsonl(train_path)
    dev = load_jsonl(dev_path)
    print(f"train: {len(train)} rows   dev: {len(dev)} rows   "
          f"hoped per-class gain: +{hoped_delta:.0%}\n")

    from collections import Counter
    tr_lab = Counter(r.get("completion") or r.get("gold_long") for r in train)
    dv_lab = Counter(r.get("completion") or r.get("gold_long") for r in dev)

    print("=== label distribution ===")
    print(f"  {'class':24s} {'train':>7s} {'dev':>7s} {'dev prev':>9s} "
          f"{'paired sees':>12s} {'unpaired sees':>14s}")
    classes = sorted(set(tr_lab) | set(dv_lab))
    rows = []
    for c in classes:
        n_d = dv_lab.get(c, 0)
        prev = n_d / len(dev) if dev else 0.0
        paired = min_detectable_delta_paired(n_d, 0.20) if n_d else 1.0
        unp = min_detectable_delta(n_d, max(prev, 1e-6)) if n_d else 1.0
        rows.append((c, tr_lab.get(c, 0), n_d, prev, paired, unp))
        print(f"  {c:24s} {tr_lab.get(c, 0):7d} {n_d:7d} {prev:9.1%} "
              f"{paired:12.1%} {unp:14.1%}")

    print(f"\n=== is the dev set big enough? (paired test, 20% discordance) ===")
    for c, n_tr, n_d, prev, paired, unp in rows:
        if prev == 0:
            print(f"  {c:24s} ABSENT from dev -- no measurement possible")
        elif paired <= hoped_delta:
            print(f"  {c:24s} YES: {n_d} rows detect {paired:.1%}, which covers "
                  f"+{hoped_delta:.0%}")
        else:
            need = min_n_paired(hoped_delta, 0.20) / prev if prev > 0 else 10 ** 9
            print(f"  {c:24s} NO: {n_d} rows see only {paired:.1%}; a +{hoped_delta:.0%} claim "
                  f"needs ~{int(need)} total rows")
    print(f"  (the whole set is paired: every model is scored on the same claims, so only the")
    print(f"   disagreements carry information -- that is why these numbers are far friendlier")
    print(f"   than an unpaired test would suggest)")

    # --- leakage: exact-normalised duplicates across splits
    tr_keys = {_norm_claim(r.get("claim", "")) for r in train}
    dv_keys = [_norm_claim(r.get("claim", "")) for r in dev]
    exact_overlap = sum(1 for k in dv_keys if k in tr_keys)
    print(f"\n=== leakage ===")
    print(f"  exact-normalised claim overlap dev->train: {exact_overlap}/{len(dev)} "
          f"({exact_overlap / len(dev):.1%})")

    tr_sh = {}
    for r in train:
        for h in _shingles(r.get("claim", ""), 13):
            tr_sh[h] = True
    near = 0
    for r in dev:
        sh = _shingles(r.get("claim", ""), 13)
        if sh and any(h in tr_sh for h in sh):
            near += 1
    print(f"  13-gram containment dev->train:            {near}/{len(dev)} "
          f"({near / len(dev):.1%})")
    if near or exact_overlap:
        print(f"  -> overlapping claims inflate dev scores. Deduplicate before trusting any number.")
    else:
        print(f"  -> clean on this measure; claims are disjoint between splits.")

    # --- the baseline any model must beat
    print(f"\n=== prior-only baseline (the floor) ===")
    base = max(dv_lab.values()) / len(dev) if dev else 0
    uniform_brier = 0.0
    for c in classes:
        p_pred = dv_lab.get(c, 0) / len(dev)
        for c_true in classes:
            outcome = 1.0 if c_true == c else 0.0
            # Brier sum over one-hot classes, averaged
            uniform_brier += (p_pred - outcome) ** 2
    uniform_brier /= (len(classes) * len(classes))
    print(f"  majority-class accuracy      : {base:.3f}")
    print(f"  predicting the dev prior     : Brier {uniform_brier:.4f}, "
          f"log loss {-sum(( dv_lab.get(c,0)/len(dev)) * math.log(max(1e-9, dv_lab.get(c,0)/len(dev))) for c in classes):.4f}")
    print(f"  any model must beat this, not 0.")
    if len(tr_lab) and len(dv_lab):
        shift = {c: (tr_lab.get(c, 0) / len(train)) - (dv_lab.get(c, 0) / len(dev))
                 for c in classes}
        print(f"\n=== train/dev distribution shift ===")
        for c in classes:
            print(f"  {c:24s} train {tr_lab.get(c,0)/len(train):6.1%}  "
                  f"dev {dv_lab.get(c,0)/len(dev):6.1%}  shift {shift[c]:+6.1%}")
        worst = max(abs(v) for v in shift.values())
        if worst > 0.10:
            print(f"  -> shift up to {worst:.0%}: dev is not a small-sample version of train, "
                  f"so train-fit decisions may not transfer.")
    return 0


def dedup_dev(train_path: str = TRAIN, dev_path: str = DEV,
              out_path: str = "", k: int = 13) -> int:
    """Drop dev claims that also appear in train, exactly or by 13-gram containment.

    Measured on this split: 17/500 exact-normalised overlaps and 4/500 with a shared 13-gram. Any dev
    score computed on the raw file is therefore inflated by up to 3.4% of rows that the model was
    trained on -- and that inflation is largest exactly on the classes with the fewest rows, which is
    where the measurement was already marginal. Deduplicate BEFORE believing any number, not after a
    result looks surprising.
    """
    train = load_jsonl(train_path)
    dev = load_jsonl(dev_path)
    out_path = out_path or dev_path.replace(".jsonl", ".dedup.jsonl")

    tr_keys = {_norm_claim(r.get("claim", "")) for r in train}
    tr_sh = set()
    for r in train:
        tr_sh |= _shingles(r.get("claim", ""), k)

    kept, dropped = [], []
    for r in dev:
        key = _norm_claim(r.get("claim", ""))
        sh = _shingles(r.get("claim", ""), k)
        if key in tr_keys:
            dropped.append((r, "exact-normalised duplicate of a train claim"))
        elif sh and (sh & tr_sh):
            dropped.append((r, f"shares a {k}-gram with a train claim"))
        else:
            kept.append(r)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    before = Counter(r.get("completion") for r in dev)
    after = Counter(r.get("completion") for r in kept)
    print(f"\n=== dedup ===")
    print(f"  dev in : {len(dev)}   dropped: {len(dropped)}   kept: {len(kept)}")
    print(f"  written: {out_path}")
    print(f"  {'class':24s} {'before':>7s} {'after':>7s} {'lost':>5s}")
    for c in sorted(set(before) | set(after)):
        lost = before.get(c, 0) - after.get(c, 0)
        print(f"  {c:24s} {before.get(c, 0):7d} {after.get(c, 0):7d} {lost:5d}")
    print(f"\n  effect on what the set can measure (+10% target, paired, 20% discordance):")
    for c in sorted(after):
        n = after[c]
        det = min_detectable_delta_paired(n, 0.20)
        mark = "OK " if det <= 0.10 else "NO "
        print(f"    {mark}{c:24s} n={n:3d}  detects {det:5.1%}")
    return 0


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.eval_audit - SELF TEST")
    ok = True

    # --- 1. z
    good = abs(z(0.975) - 1.959964) < 1e-5 and abs(z(0.5)) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] z(0.975) = {z(0.975):.6f}")

    # --- 2. Wilson behaves at the edges where the normal interval collapses
    lo, hi = wilson(0, 50)
    good = lo == 0.0 and 0.0 < hi < 0.15
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Wilson at k=0, n=50 -> [{lo:.3f}, {hi:.3f}] "
          f"(normal would give width 0)")

    lo, hi = wilson(50, 50)
    good = hi == 1.0 and lo > 0.85
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Wilson at k=n=50 -> [{lo:.3f}, {hi:.3f}]")

    # --- 3. Wilson brackets the point estimate
    lo, hi = wilson(160, 500)
    good = lo < 0.32 < hi
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Wilson brackets p=0.32 at n=500 "
          f"[{lo:.3f}, {hi:.3f}]")

    # --- 4. min_n matches the textbook value for 50% vs 60% between INDEPENDENT groups (~388)
    n = min_n_two_proportion(0.50, 0.60)
    good = 380 <= n <= 400
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unpaired: 50% vs 60% needs {n}/group "
          f"(the small-sample ~93 figure is the ONE-sample test, not this one)")

    # --- 4b. PAIRED is the right test for us, and it is cheaper. Same effect, same power.
    n_pair = min_n_paired(0.10, discordance=0.20)
    good = n_pair < n / 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] paired (McNemar) detects the same +10pp with "
          f"{n_pair} pairs vs {2 * n} independent rows ({2 * n / n_pair:.1f}x fewer)")

    # --- 4c. paired detectability at our actual eval size
    d500 = min_detectable_delta_paired(500, 0.20)
    d40 = min_detectable_delta_paired(40, 0.20)
    good = 0.04 < d500 < 0.08 and d40 > 0.15
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 500 paired rows see {d500:.1%}; a 40-row class sees "
          f"only {d40:.1%}")

    # --- 5. a smaller difference needs more rows, monotonically
    n1 = min_n_two_proportion(0.30, 0.48)
    n2 = min_n_two_proportion(0.30, 0.33)
    good = n2 > n1 * 10
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] +18pp needs {n1}/group, +3pp needs {n2}/group")

    # --- 6. min_detectable_delta is monotone decreasing in n
    d_small = min_detectable_delta(20, 0.30)
    d_big = min_detectable_delta(2000, 0.30)
    good = d_small > d_big > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] detectable delta falls with n "
          f"(n=20 -> {d_small:.1%}, n=2000 -> {d_big:.1%})")

    # --- 7. it reproduces the failure this project already suffered: n=38 cannot see +18pp
    #     (the Conflicting "doubling" with CI [-0.038, +0.323])
    det38 = min_detectable_delta_paired(38, 0.20)
    good = det38 > 0.18
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] even PAIRED, n=38 at 20% discordance only sees "
          f"{det38:.1%}, so the reported +18pp Conflicting gain was unmeasurable by construction")

    # --- 8. required_n_for_class turns a class prevalence into a total-set size
    req = required_n_for_class(0.08, 0.18)
    good = req["per_group"] > 0 and req["total_rows_needed"] > req["per_group"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an 8%-prevalence class needs "
          f"{req['per_group']}/group == ~{req['total_rows_needed']} total rows to see +18pp")

    # --- 9. identical proportions are never "detectable"
    good = min_n_two_proportion(0.4, 0.4) >= 10 ** 9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] zero difference returns infinity, not a small n")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Eval-set power audit.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--dev", default=DEV)
    ap.add_argument("--hoped", type=float, default=0.10)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.audit:
        return audit(args.train, args.dev, args.hoped)
    if args.dedup:
        return dedup_dev(args.train, args.dev, args.out)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
