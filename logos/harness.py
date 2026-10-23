"""logos.harness — measurement that can say "no difference", or nothing downstream is real.

WHY THIS COMES BEFORE ANY NEW MODEL CODE
----------------------------------------
Our dev split has 500 rows and `Conflicting` is ~7.6% of it: roughly **60 positives**. One
standard deviation on that class's F1 is about **±0.08**, so a 95% interval spans something
like 0.19-0.53. That means most of the numbers we have been comparing — 0.744 vs 0.808,
Conflicting .214 vs .362 — could be partly or wholly noise, and a single-split ranking cannot
tell the difference. The 2026 result is blunt about it: at this corpus size five-fold variance
is wide enough that authors call their own backbone comparison "underpowered rather than
settled" (arXiv:2610.17077).

So this module exists to do three things before anything else is believed:

  1. **Bootstrap CIs** on accuracy, macro-F1 and EVERY per-class F1.
  2. **PAIRED** comparisons. Two models scored on the same 500 rows are correlated; resampling
     items and recomputing *both* gives a far tighter interval on the difference than two
     independent CIs would, and it is the only honest way to say "B is better than A".
  3. **McNemar's test** on the discordant pairs, which is the right question: among the rows
     where the two models disagree, is the split lopsided?

Plus the behavioural test that guardrails everything: the **evidence-ablation flip rate**
(REAL, arXiv:2609.08943). If deleting the cited supporting span does not change the verdict,
the model is reading its own priors rather than the evidence, and no accuracy number above it
means what it appears to mean.

USAGE
-----
    python tools/logos/harness.py --self-test
    python tools/logos/harness.py --compare A=path1.jsonl B=path2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

N_CLASSES = 4
LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting"]


# ------------------------------------------------------------------------------ metrics
def macro_f1(gold: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-F1 and per-class F1, computed directly so the definition is auditable."""
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)
    per = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        if tp == 0:
            per[c] = 0.0
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        per[c] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return float(per.mean()), per


def accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(gold) == np.asarray(pred)).mean())


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Reported alongside bootstrap for accuracy because it is exact
    for a binomial proportion and does not need resampling."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, n)


def bootstrap_ci(gold: np.ndarray, pred: np.ndarray, n_boot: int = 2000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap over ITEMS (rows), which is the unit of exchangeability here."""
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = len(gold)
    rng = np.random.default_rng(seed)
    accs, macs, per = [], [], np.zeros((n_boot, N_CLASSES))
    for b in range(n_boot):
        idx = bootstrap_indices(n, rng)
        g, p = gold[idx], pred[idx]
        accs.append(accuracy(g, p))
        m, pc = macro_f1(g, p)
        macs.append(m)
        per[b] = pc
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    m0, per0 = macro_f1(gold, pred)
    return {
        "n": int(n),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci": (float(np.percentile(accs, lo)), float(np.percentile(accs, hi))),
        "accuracy_wilson": wilson(int((gold == pred).sum()), n),
        "macro_f1": m0,
        "macro_f1_ci": (float(np.percentile(macs, lo)), float(np.percentile(macs, hi))),
        "per_class_f1": per0,
        "per_class_ci": [(float(np.percentile(per[:, c], lo)),
                          float(np.percentile(per[:, c], hi))) for c in range(N_CLASSES)],
        "per_class_n": [int((gold == c).sum()) for c in range(N_CLASSES)],
    }


def paired_bootstrap(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int = 4000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """CI on (B - A) by resampling ITEMS and recomputing both. The correlated part cancels.

    Two independent CIs that overlap slightly tell you almost nothing; this does not make that
    mistake because it never treats the two models as independent samples.
    """
    gold = np.asarray(gold, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    n = len(gold)
    rng = np.random.default_rng(seed)
    d_mac, d_acc = [], []
    d_per = [[] for _ in range(N_CLASSES)]
    for _ in range(n_boot):
        idx = bootstrap_indices(n, rng)
        g = gold[idx]
        ma, pa_ = macro_f1(g, a[idx])
        mb, pb_ = macro_f1(g, b[idx])
        d_mac.append(mb - ma)
        d_acc.append(accuracy(g, b[idx]) - accuracy(g, a[idx]))
        for c in range(N_CLASSES):
            d_per[c].append(float(pb_[c] - pa_[c]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    m_a, per_a = macro_f1(gold, a)
    m_b, per_b = macro_f1(gold, b)
    ci_mac = (float(np.percentile(d_mac, lo)), float(np.percentile(d_mac, hi)))
    excl0 = (ci_mac[0] > 0) or (ci_mac[1] < 0)
    per_out = []
    for c in range(N_CLASSES):
        ci = (float(np.percentile(d_per[c], lo)), float(np.percentile(d_per[c], hi)))
        per_out.append({"class": LABELS[c], "f1_a": float(per_a[c]),
                        "f1_b": float(per_b[c]), "diff": float(per_b[c] - per_a[c]),
                        "diff_ci": ci,
                        "significant": bool(ci[0] > 0 or ci[1] < 0),
                        "n": int((gold == c).sum())})
    return {
        "macro_f1_a": m_a, "macro_f1_b": m_b, "diff_macro_f1": m_b - m_a,
        "diff_macro_f1_ci": ci_mac,
        "diff_accuracy": accuracy(gold, b) - accuracy(gold, a),
        "diff_accuracy_ci": (float(np.percentile(d_acc, lo)),
                             float(np.percentile(d_acc, hi))),
        "per_class": per_out,
        "significant": bool(excl0),
        "verdict": ("B better (CI excludes 0)" if excl0 and ci_mac[0] > 0 else
                    "A better (CI excludes 0)" if excl0 else
                    "NO detectable difference (CI contains 0)"),
    }


def mcnemar(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar on the discordant pairs: among rows where the two models differ, is the
    split lopsided? This is the standard paired test for two classifiers on one test set."""
    gold = np.asarray(gold, dtype=int)
    a_ok = np.asarray(pred_a, dtype=int) == gold
    b_ok = np.asarray(pred_b, dtype=int) == gold
    b_only = int((~a_ok & b_ok).sum())      # A wrong, B right
    a_only = int((a_ok & ~b_ok).sum())      # A right, B wrong
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "n_discordant": 0}
    try:
        from scipy.stats import binomtest
        p = float(binomtest(b_only, n, 0.5).pvalue)
    except Exception:                        # noqa: BLE001 - normal approximation fallback
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p_value": p}


# ------------------------------------------------------------- evidence-ablation flip test
def ablate_evidence(text: str) -> str:
    """Remove the evidence half of a `claim [SEP] evidence` example."""
    claim, sep, _ = text.partition("[SEP]")
    return claim if sep else text


def flip_rate(texts: list[str], preds: list[int], predict_fn) -> dict:
    """REAL (arXiv:2609.08943): does the verdict change when the cited evidence is ablated?

    A model that ignores evidence is not a verifier; it is a prior with a citation attached.
    `predict_fn(list[str]) -> list[int]` runs the model. A flip rate near 0 is a red flag; a
    flip rate far above the evidence-dependence baseline means the model keys on the surface.

    Reported together with the same model's flip rate under a *placebo* ablation (a random
    sentence appended), because a model that flips on any change is not reading evidence either.
    """
    ablated = [ablate_evidence(t) for t in texts]
    new = predict_fn(ablated)
    flips = int(sum(1 for a, b in zip(preds, new) if int(a) != int(b)))
    return {"n": len(texts), "flips": flips, "flip_rate": flips / max(len(texts), 1),
            "ablated_preds": new}


# ---------------------------------------------------------------------------------- io
def load_preds(path: str, split: str, remap: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    gold, pred = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] != split:
                continue
            p = int(r["pred"])
            gold.append(int(r["label"]))
            pred.append(remap[p] if remap else p)
    return np.asarray(gold), np.asarray(pred)


def format_report(name: str, ci: dict) -> str:
    lines = [f"{name}: n={ci['n']}  acc {ci['accuracy']:.4f} "
             f"[{ci['accuracy_ci'][0]:.3f},{ci['accuracy_ci'][1]:.3f}]  "
             f"macro-F1 {ci['macro_f1']:.4f} "
             f"[{ci['macro_f1_ci'][0]:.3f},{ci['macro_f1_ci'][1]:.3f}]"]
    for c in range(N_CLASSES):
        lo, hi = ci["per_class_ci"][c]
        lines.append(f"    {LABELS[c]:12s} F1 {ci['per_class_f1'][c]:.3f} "
                     f"[{lo:.3f},{hi:.3f}]  n={ci['per_class_n'][c]:3d}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.harness - SELF TEST")
    ok = True
    rng = np.random.default_rng(0)

    # --- macro-F1 must match sklearn on random inputs (independent implementation check)
    from sklearn.metrics import f1_score
    worst = 0.0
    for _ in range(50):
        n = int(rng.integers(20, 200))
        g = rng.integers(0, 4, n)
        p = rng.integers(0, 4, n)
        mine, _ = macro_f1(g, p)
        ref = float(f1_score(g, p, average="macro", labels=list(range(4)), zero_division=0))
        worst = max(worst, abs(mine - ref))
    good = worst < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] macro-F1 matches sklearn on 50 random cases "
          f"(max err {worst:.2e})")

    # --- perfect predictor: F1 = 1 and a DEGENERATE interval [1,1]
    g = rng.integers(0, 4, 200)
    ci = bootstrap_ci(g, g, n_boot=300)
    good = ci["macro_f1"] == 1.0 and ci["macro_f1_ci"] == (1.0, 1.0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] perfect predictor -> macro-F1 1.0 with CI [1,1]")

    # --- collapse to one class on BALANCED gold: exact analytic macro-F1 = 0.1
    # 25 per class, predict all class 0: F1_0 = 2*.25*1/1.25 = 0.4, others 0 -> macro 0.1
    gb = np.repeat(np.arange(4), 25)
    pb = np.zeros_like(gb)
    m, per = macro_f1(gb, pb)
    good = abs(m - 0.1) < 1e-12 and abs(per[0] - 0.4) < 1e-12 and per[1] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] collapse-to-one-class on balanced gold is "
          f"exactly 0.10 macro-F1 ({m:.12f}, class0 F1 {per[0]:.3f})")

    # --- a model compared with ITSELF must show exactly zero difference, CI [0,0]
    g = rng.integers(0, 4, 300)
    p = rng.integers(0, 4, 300)
    pb2 = paired_bootstrap(g, p, p, n_boot=300)
    good = pb2["diff_macro_f1"] == 0.0 and pb2["diff_macro_f1_ci"] == (0.0, 0.0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] paired bootstrap of a model against itself -> "
          f"diff exactly 0.0, CI [0,0]")

    # --- McNemar on identical predictions: no discordant pairs, p = 1
    mc = mcnemar(g, p, p)
    good = mc["n_discordant"] == 0 and mc["p_value"] == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] McNemar on identical predictions -> p = 1.0")

    # --- and it must DETECT a genuinely better model: B fixes 20% of A's errors
    truth = rng.integers(0, 4, 2000)
    a = truth.copy()
    wrong = rng.choice(2000, 600, replace=False)      # A gets 600 wrong (70% acc)
    a[wrong] = (a[wrong] + 1) % 4
    b = a.copy()
    fix = rng.choice(wrong, 120, replace=False)       # B fixes 120 of them (76% acc)
    b[fix] = truth[fix]
    pb3 = paired_bootstrap(truth, a, b, n_boot=500)
    mc3 = mcnemar(truth, a, b)
    good = pb3["diff_macro_f1_ci"][0] > 0 and mc3["p_value"] < 1e-6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] detects a real improvement: diff macro-F1 "
          f"{pb3['diff_macro_f1']:+.4f} CI "
          f"[{pb3['diff_macro_f1_ci'][0]:+.4f},{pb3['diff_macro_f1_ci'][1]:+.4f}], "
          f"McNemar p={mc3['p_value']:.2e}")

    # --- and it must FAIL to detect a null: B differs only on rows both get wrong anyway
    c = a.copy()
    both_wrong = np.where(a != truth)[0][:200]
    c[both_wrong] = (c[both_wrong] + 1) % 4            # different errors, same accuracy-ish
    pb4 = paired_bootstrap(truth, a, c, n_boot=500)
    good = not pb4["significant"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] reports NO difference when there is none "
          f"(CI {pb4['diff_macro_f1_ci'][0]:+.4f},{pb4['diff_macro_f1_ci'][1]:+.4f}) -> "
          f"{pb4['verdict']}")

    # --- stratification: every fold holds every class in proportion
    g = rng.choice(4, 500, p=[0.24, 0.61, 0.07, 0.08])
    folds = stratified_folds(g, 5, seed=0)
    good = len(folds) == 5 and all(len(f) > 0 for f in folds)
    ok = ok and good
    sizes = [len(f) for f in folds]
    good = good and sum(sizes) == len(g) and max(sizes) - min(sizes) <= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 5 stratified folds partition the dev set "
          f"{sizes}")

    # --- the ablation flip test has the two signatures it must distinguish
    texts = [f"claim {i} [SEP] evidence supporting {i}" for i in range(50)]
    base = [0] * 50
    # a model that reads evidence: removing it flips every verdict
    ev_dependent = lambda ts: [1 if "[SEP]" not in t else 0 for t in ts]   # noqa: E731
    r1 = flip_rate(texts, base, ev_dependent)
    # a model that ignores evidence: removing it changes nothing
    prior_only = lambda ts: [0 for _ in ts]                               # noqa: E731
    r2 = flip_rate(texts, base, prior_only)
    good = r1["flip_rate"] == 1.0 and r2["flip_rate"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ablation flip rate separates an evidence-reading "
          f"model (1.00) from a prior-only model (0.00)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def stratified_folds(y: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Class-stratified index folds balancing BOTH class proportions and fold size.

    Round-robin over `j % k` per class balances proportions but lets fold sizes drift (measured
    101/100/100/100/99 on 500 rows). Greedy assignment to the currently smallest fold keeps
    sizes within 1 while still placing each class's items across all k folds, which is what
    sklearn's StratifiedKFold guarantees. Inlined rather than delegated so the assignment is
    reproducible from (y, seed) alone and auditable.
    """
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    per_class: list[dict[int, int]] = [dict() for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        for i in idx:
            j = min(range(k), key=lambda f: (len(folds[f]), per_class[f].get(int(c), 0)))
            folds[j].append(int(i))
            per_class[j][int(c)] = per_class[j].get(int(c), 0) + 1
    return [np.asarray(sorted(f)) for f in folds]


def cv_score(y: np.ndarray, pred_factory, k: int = 5, seed: int = 0) -> dict:
    """5-fold CV where `pred_factory(train_idx) -> preds_on_all`. Returns per-fold macro-F1 so
    the spread is visible instead of collapsed into a mean."""
    y = np.asarray(y, dtype=int)
    folds = stratified_folds(y, k, seed)
    per_fold = []
    for f in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[f] = False
        preds = pred_factory(np.where(mask)[0])
        m, _ = macro_f1(y[f], np.asarray(preds)[f])
        per_fold.append(m)
    return {"k": k, "fold_macro_f1": per_fold, "mean": float(np.mean(per_fold)),
            "sd": float(np.std(per_fold)), "min": float(np.min(per_fold)),
            "max": float(np.max(per_fold))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CIs, paired tests, ablation flip rate.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--logits", default="", help="single model report")
    ap.add_argument("--compare", nargs=2, default=[],
                    metavar=("A", "B"), help="two logits files: A=path B=path")
    ap.add_argument("--remap", default="", help="e.g. 1,0,2,3 for the CTU checkpoint")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args(argv)
    if args.self_test or (not args.logits and not args.compare):
        return 0 if run_self_test() else 1

    if args.logits:
        g, p = load_preds(args.logits, args.split)
        print(format_report(os.path.basename(args.logits), bootstrap_ci(g, p, args.boot)))
        return 0

    spec = {}
    for part in args.compare:
        k, _, v = part.partition("=")
        spec[k] = v
    remap = [int(x) for x in args.remap.split(",")] if args.remap else None
    names = list(spec)
    ga, pa = load_preds(spec[names[0]], args.split, remap)
    gb, pb = load_preds(spec[names[1]], args.split, remap)
    if len(ga) != len(gb):
        print(f"[h] split sizes differ ({len(ga)} vs {len(gb)}); comparing is invalid")
        return 2
    print(format_report(names[0], bootstrap_ci(ga, pa, args.boot)))
    print()
    print(format_report(names[1], bootstrap_ci(gb, pb, args.boot)))
    print()
    d = paired_bootstrap(ga, pa, pb, args.boot)
    mc = mcnemar(ga, pa, pb)
    print(f"[h] PAIRED comparison {names[1]} vs {names[0]}:")
    print(f"[h]   macro-F1 {d['macro_f1_a']:.4f} -> {d['macro_f1_b']:.4f}  "
          f"diff {d['diff_macro_f1']:+.4f} "
          f"CI [{d['diff_macro_f1_ci'][0]:+.4f},{d['diff_macro_f1_ci'][1]:+.4f}]")
    print(f"[h]   accuracy diff {d['diff_accuracy']:+.4f} "
          f"CI [{d['diff_accuracy_ci'][0]:+.4f},{d['diff_accuracy_ci'][1]:+.4f}]")
    print(f"[h]   McNemar: only-{names[0]}-right {mc['a_only']} · "
          f"only-{names[1]}-right {mc['b_only']} · p = {mc['p_value']:.3g}")
    print(f"[h]   VERDICT: {d['verdict']}")
    print(f"[h]   PER-CLASS paired difference (this is where claims usually break):")
    for row in d["per_class"]:
        flag = "SIGNIFICANT" if row["significant"] else "not distinguishable"
        print(f"[h]     {row['class']:12s} {row['f1_a']:.3f} -> {row['f1_b']:.3f}  "
              f"diff {row['diff']:+.3f} "
              f"CI [{row['diff_ci'][0]:+.3f},{row['diff_ci'][1]:+.3f}]  n={row['n']:3d}  "
              f"{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
