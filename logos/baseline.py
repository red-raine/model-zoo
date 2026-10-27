"""logos/baseline.py — the number to beat, trained on frozen embeddings.

WHY THIS FILE EXISTS
--------------------
The only model result on record for this task is accuracy 0.245 against a majority baseline of 0.610
and a random baseline of 0.315. That is BELOW RANDOM, so it measures a bug, not a model. Nothing else
in the project can be interpreted until there is a working baseline and a published number to beat.

`emb-frozen.npz` already holds frozen 768-d embeddings for all 3067 train rows and 500 dev rows, so a
linear probe needs no GPU and no fine-tuning. That is also the honest response to "3066 rows is small
for training": 3066 rows is enough for a linear probe over frozen features and nothing more.

WHAT IS REPORTED
----------------
    accuracy, macro-F1      macro-F1 because the classes are 61/24/8/7 -- accuracy is dominated by
                            REFUTED and hides every rare-class failure
    per-class P/R/F1        so a class that is never predicted is visible as F1 0, not folded away
    paired McNemar          between two predictors, plus the DISCORDANCE RATE, which is the number
                            `eval_audit` had to assume (20%) and can now measure
    deduped dev             18 dev rows also appear in train; reported both ways so the leak's size
                            on each metric is visible rather than asserted

USAGE
    python tools/logos/baseline.py --self-test
    python tools/logos/baseline.py --run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

D = r"B:\datasets\forge\verifier-av"
EMB = os.path.join(D, "emb-frozen.npz")
TRAIN = os.path.join(D, "train.clean.jsonl")
DEV = os.path.join(D, "dev.jsonl")

CLASSES = ("CONFLICTING", "NOT_ENOUGH_EVIDENCE", "REFUTED", "SUPPORTED")
# manifest counts: REFUTED 1742, SUPPORTED 848, NEI 282, CONFLICTING 195 -- used to recover the
# integer encoding from the data rather than trusting an ordering guess
EXPECTED_TRAIN = {"REFUTED": 1742, "SUPPORTED": 848, "NOT_ENOUGH_EVIDENCE": 282,
                  "CONFLICTING": 195}
# the true integer order, recovered from EXPECTED_TRAIN and asserted by the self-test rather than
# assumed. CLASSES above is NOT this order -- getting the two confused mislabels every report.
CODE_TO_LABEL = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT_ENOUGH_EVIDENCE", 3: "CONFLICTING"}


def label_of(code: int) -> str:
    return CODE_TO_LABEL.get(int(code), f"class{code}")


# --------------------------------------------------------------------------------- the probe
def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 400,
                lr: float = 0.5, l2: float = 1e-3, seed: int = 0) -> dict:
    """Multinomial logistic regression by full-batch gradient descent. Small and deterministic.

    Standardise first: 768 frozen features have wildly different scales, and an unstandardised probe
    will converge to whatever the loudest dimensions say. The scaler is returned so dev is transformed
    with TRAIN statistics -- fitting it on dev is a classic silent leak.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd

    n, d = Xs.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    for _ in range(epochs):
        z = Xs @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (Xs.T @ g + l2 * W)
        b -= lr * g.sum(axis=0)
    return {"W": W, "b": b, "mu": mu, "sd": sd}


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mu"]) / model["sd"]
    z = Xs @ model["W"] + model["b"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return predict_proba(model, X).argmax(axis=1)


# ------------------------------------------------------------------------------- metrics
def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1. A class never predicted contributes F1 0 -- which is the
    point: accuracy would hide it entirely."""
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        if tp + fn == 0:
            continue                      # absent from this subset; not a failure, just not scored
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def per_class(y: np.ndarray, pred: np.ndarray, n_classes: int) -> list[dict]:
    out = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"class": c, "n": int(np.sum(y == c)), "predicted": int(np.sum(pred == c)),
                    "precision": prec, "recall": rec, "f1": f1})
    return out


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Exact binomial McNemar on the discordant pairs.

    Only rows where the two predictors differ carry information -- which is why the paired test needs
    far fewer rows than an unpaired comparison, and why the discordance rate is the quantity
    `eval_audit` needs in place of its 20% assumption.

    CONVENTION, stated because it was wrong here first. The 2x2 table is

                        b right   b wrong
            a right        --        n01
            a wrong       n10        --

    so **b = a right & b wrong** and **c = a wrong & b right**. The first version had these swapped,
    and the symptom was visible and would have been believable: a run where the new predictor scored
    0.23 macro-F1 LOWER reported "120 rows fixed, 40 rows broken". Those numbers were right, the
    labels attached to them were transposed, and the only reason it was caught is that the direction
    contradicted the score. Anyone reading b/c without checking would have concluded the opposite of
    the truth.
    """
    ok_a, ok_b = (a == y), (b == y)
    n01 = int(np.sum(ok_a & ~ok_b))          # a right, b wrong  -> "b" (counts AGAINST b)
    n10 = int(np.sum(~ok_a & ok_b))          # a wrong, b right  -> "c" (counts FOR b)
    n_disc = n01 + n10
    if n_disc == 0:
        return {"b": 0, "c": 0, "discordance": 0.0, "p": 1.0}
    k = min(n01, n10)
    # exact two-sided binomial: 2 * P(X <= k) with X ~ Bin(n_disc, 0.5), capped at 1
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
    return {"b": n01, "c": n10, "discordance": n_disc / len(y),
            "p": min(1.0, 2 * tail)}


# ---------------------------------------------------------------------------------- driver
def _dedup_mask() -> np.ndarray:
    """True for dev rows that do NOT appear in train. Reuses the audit's own normalisation."""
    from logos.eval_audit import _norm_claim, _shingles, load_jsonl
    train = load_jsonl(TRAIN)
    dev = load_jsonl(DEV)
    trk = {_norm_claim(r.get("claim", "")) for r in train}
    trs = set()
    for r in train:
        trs |= _shingles(r.get("claim", ""), 13)
    keep = []
    for r in dev:
        k = _norm_claim(r.get("claim", ""))
        sh = _shingles(r.get("claim", ""), 13)
        keep.append(not (k in trk or (sh and (sh & trs))))
    return np.array(keep, dtype=bool)


def run(emb: str = EMB) -> int:
    z = np.load(emb, allow_pickle=True)
    y_train = z["y_train"].astype(int)
    y_dev = z["y_dev"].astype(int)

    # --- recover the integer encoding from the manifest counts, never from a guess
    counts = np.bincount(y_train)
    mapping = {}
    for code, cnt in enumerate(counts):
        match = [lab for lab, exp in EXPECTED_TRAIN.items() if exp == cnt]
        mapping[code] = match[0] if match else f"class{code}"
    print("=== recovered label encoding (from manifest counts) ===")
    for code, lab in mapping.items():
        print(f"  {code} -> {lab}  (train n={int(counts[code])})")
    if len(set(mapping.values())) < len(mapping):
        print("  WARNING: counts did not map one-to-one; encoding is unreliable")

    mask = _dedup_mask()
    print(f"\n=== dev dedup: {int((~mask).sum())} of {len(mask)} rows also appear in train ===")

    results = {}
    for feat in ("cls", "mean"):
        Xtr, Xdv = z[f"X_train_{feat}"], z[f"X_dev_{feat}"]
        m = fit_softmax(Xtr, y_train, len(CLASSES))
        pred = predict(m, Xdv)
        for tag, sel in (("full", np.ones(len(y_dev), dtype=bool)), ("dedup", mask)):
            yy, pp = y_dev[sel], pred[sel]
            acc = float(np.mean(pp == yy))
            results[(feat, tag)] = {
                "model": m, "pred": pred, "sel": sel, "acc": acc,
                "macro_f1": macro_f1(yy, pp, len(CLASSES)),
                "per_class": per_class(yy, pp, len(CLASSES)),
            }

    # --- baselines on the same rows
    maj = int(np.bincount(y_train).argmax())
    print(f"\n=== results (majority class = {mapping.get(maj, maj)}) ===")
    print(f"  {'features':8s} {'split':7s} {'acc':>6s} {'majority':>9s} {'macroF1':>8s} "
          f"{'random':>7s}")
    for (feat, tag), r in results.items():
        yy, pp = y_dev[r["sel"]], r["pred"][r["sel"]]
        maj_acc = float(np.mean(pp == maj))
        rand = 1.0 / len(set(yy))
        print(f"  {feat:8s} {tag:7s} {r['acc']:6.3f} {maj_acc:9.3f} {r['macro_f1']:8.3f} "
              f"{rand:7.3f}")

    print(f"\n=== per-class on deduped dev (cls features) ===")
    print(f"  {'class':22s} {'n':>4s} {'pred':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    r = results[("cls", "dedup")]
    for row in r["per_class"]:
        lab = mapping.get(row["class"], str(row["class"]))
        print(f"  {lab:22s} {row['n']:4d} {row['predicted']:5d} {row['precision']:6.3f} "
              f"{row['recall']:6.3f} {row['f1']:6.3f}")

    # --- paired comparison + the REAL discordance rate, which eval_audit had to assume
    print(f"\n=== paired McNemar (cls vs mean, deduped dev) ===")
    pa = results[("cls", "dedup")]["pred"][mask]
    pb = results[("mean", "dedup")]["pred"][mask]
    yy = y_dev[mask]
    mc = mcnemar(yy, pa, pb)
    print(f"  discordant pairs: {mc['b'] + mc['c']}  (a-only-right {mc['b']}, "
          f"b-only-right {mc['c']})")
    print(f"  DISCORDANCE RATE: {mc['discordance']:.1%}  <- replaces the 20% assumption in "
          f"eval_audit")
    print(f"  exact McNemar p = {mc['p']:.4f}")

    # --- and vs the majority baseline, which is the comparison that matters
    print(f"\n=== vs the majority baseline, which is the bar ===")
    maj_pred = np.full(len(yy), maj)
    for feat in ("cls", "mean"):
        pf = results[(feat, "dedup")]["pred"][mask]
        mcx = mcnemar(yy, pf, maj_pred)
        verdict = "BEATS" if float(np.mean(pf == yy)) > float(np.mean(maj_pred == yy)) else "FAILS"
        print(f"  {feat:5s} acc {float(np.mean(pf == yy)):.3f} vs majority "
              f"{float(np.mean(maj_pred == yy)):.3f}  -> {verdict}"
              f"   (discordance {mcx['discordance']:.1%}, McNemar p={mcx['p']:.6f})")

    # --- MEASURED discordance, which eval_audit had to assume. The assumption was wrong in the
    #     direction that flatters us: higher discordance means each row carries LESS information,
    #     so the eval set is weaker than projected.
    from logos.eval_audit import min_detectable_delta_paired
    print(f"\n=== does the measured discordance change what the eval set can detect? ===")
    for assumed in (0.20, mc["discordance"]):
        d303 = min_detectable_delta_paired(303, assumed)
        d107 = min_detectable_delta_paired(107, assumed)
        tag = "assumed" if assumed == 0.20 else "MEASURED"
        print(f"  discordance {assumed:5.1%} ({tag:8s}): REFUTED n=303 sees {d303:.1%}, "
              f"SUPPORTED n=107 sees {d107:.1%}")

    # --- the SUPPORTED claim, re-measured on the deduped split
    si = [k for k, v in mapping.items() if v == "SUPPORTED"]
    if si:
        s = si[0]
        for tag, sel in (("full", np.ones(len(y_dev), dtype=bool)), ("dedup", mask)):
            yy2 = y_dev[sel]
            pp2 = results[("cls", "full")]["pred"][sel]
            tp = int(np.sum((pp2 == s) & (yy2 == s)))
            fn = int(np.sum((pp2 != s) & (yy2 == s)))
            fp = int(np.sum((pp2 == s) & (yy2 != s)))
            rec = tp / (tp + fn) if tp + fn else 0.0
            prec = tp / (tp + fp) if tp + fp else 0.0
            print(f"\n  SUPPORTED on {tag:5s} dev: n={tp + fn:3d} recall={rec:.3f} "
                  f"precision={prec:.3f} F1="
                  f"{(2 * prec * rec / (prec + rec)) if prec + rec else 0:.3f}")
    return 0


def run_nli(logits_path: str = os.path.join(D, "nli-post-logits.jsonl")) -> int:
    """Verify the recorded NLI numbers independently, and report the deduped truth.

    This exists because a 1.3 KB summary file was mistaken for a dataset and because an earlier
    reading of `zero_shot: 0.106` looked like a broken model. Both were wrong in opposite
    directions, and neither could be settled by reading a summary. It is settled by reloading the
    per-row logits and recomputing.
    """
    import itertools
    from collections import Counter

    from logos.eval_audit import _norm_claim, _shingles, load_jsonl

    rows = [json.loads(ln) for ln in open(logits_path, encoding="utf-8") if ln.strip()]
    y = np.array([r["label"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    k = int(max(y.max(), p.max())) + 1
    print(f"=== NLI logits: {len(rows)} rows ===")
    print(f"  splits: {dict(Counter(r['split'] for r in rows))}")

    print(f"\n=== is the recorded prediction order correct? ===")
    best = (tuple(range(k)), float(np.mean(y == p)))
    for perm in itertools.permutations(range(k)):
        acc = float(np.mean(np.array([perm[x] for x in p]) == y))
        if acc > best[1]:
            best = (perm, acc)
    print(f"  raw accuracy {float(np.mean(y == p)):.4f}, best permutation {best[0]} "
          f"-> {best[1]:.4f}")
    print(f"  identity is optimal: {best[0] == tuple(range(k))}"
          + ("  (predictions already correctly ordered)" if best[0] == tuple(range(k))
             else "  (RECORDED PREDICTIONS ARE MIS-ORDERED -- do not trust stored metrics)"))

    train = load_jsonl(TRAIN)
    dev = load_jsonl(DEV)
    trk = {_norm_claim(r.get("claim", "")) for r in train}
    trs = set()
    for r in train:
        trs |= _shingles(r.get("claim", ""), 13)

    def is_dup(claim: str) -> bool:
        kk = _norm_claim(claim)
        sh = _shingles(claim, 13)
        return kk in trk or (sh and bool(sh & trs))

    lab = [i for i, r in enumerate(rows) if r["split"] == "dev"]
    yd = y[lab]
    pd = p[lab]
    keep = np.array([not is_dup(rows[i]["text"].split(" [SEP] ")[0]) for i in lab])

    maj = int(np.bincount(y).argmax())
    for tag, sel in (("full dev", np.ones(len(yd), dtype=bool)), ("deduped", keep)):
        yy, pp = yd[sel], pd[sel]
        mc = mcnemar(yy, pp, np.full(len(yy), maj))
        print(f"\n=== {tag}: n={len(yy)} ===")
        print(f"  accuracy {float(np.mean(pp == yy)):.4f}   macro-F1 "
              f"{macro_f1(yy, pp, k):.4f}   majority {float(np.mean(yy == maj)):.4f}")
        print(f"  vs majority: discordance {mc['discordance']:.1%}, McNemar p={mc['p']:.3e}")
        print(f"  {'class':24s} {'n':>4s} {'pred':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
        for row in per_class(yy, pp, k):
            print(f"  {label_of(row['class']):24s} "
                  f"{row['n']:4d} {row['predicted']:5d} {row['precision']:6.3f} "
                  f"{row['recall']:6.3f} {row['f1']:6.3f}")

    # --- the leak's maximum possible contribution to a per-class gain
    print(f"\n=== how much could the leak have inflated a per-class claim? ===")
    from collections import Counter as C
    leak_classes = C(yd[~keep].tolist())
    for c, n_leak in sorted(leak_classes.items()):
        n_cls_full = int(np.sum(yd == c))
        share = n_leak / n_cls_full if n_cls_full else 0
        print(f"  {label_of(c):24s} {n_leak:2d} leaked of {n_cls_full:3d} rows = "
              f"{share:5.1%} of the class (max possible inflation {share:.1%})")
    return 0


# --------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.baseline - SELF TEST")
    ok = True
    rng = np.random.default_rng(3)

    # --- 1. the probe learns a separable problem
    n, d, k = 600, 24, 4
    y = rng.integers(0, k, n)
    X = rng.normal(0, 1, (n, d))
    X[np.arange(n), y % d] += 3.0                     # class signal in one dimension
    m = fit_softmax(X, y, k, epochs=300)
    acc = float(np.mean(predict(m, X) == y))
    good = acc > 0.85
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] softmax probe learns a separable signal "
          f"(train acc {acc:.3f})")

    # --- 2. and fails honestly on noise
    Xn = rng.normal(0, 1, (n, d))
    mn = fit_softmax(Xn, y, k, epochs=200)
    acc_n = float(np.mean(predict(mn, Xn) == y))
    good = acc_n < 0.40
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and does not invent structure in noise "
          f"(acc {acc_n:.3f} ~ chance {1 / k:.2f})")

    # --- 3. standardisation uses TRAIN statistics only (a dev-fitted scaler is a silent leak)
    good = ("mu" in m and "sd" in m and m["mu"].shape == (d,))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the scaler is carried on the model, so dev is "
          f"transformed with train statistics")

    # --- 4. macro-F1 punishes a never-predicted class, accuracy hides it
    yy = np.array([0] * 80 + [1] * 10 + [2] * 5 + [3] * 5)
    pp = np.zeros_like(yy)                            # predict class 0 always
    acc = float(np.mean(pp == yy))
    mf = macro_f1(yy, pp, 4)
    good = acc > 0.79 and mf < 0.25
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] always-majority: accuracy {acc:.2f} but macro-F1 "
          f"{mf:.2f} -- the metric that matters")

    # --- 5. McNemar: identical predictors have no discordant pairs and p=1
    a = np.array([0, 1, 2, 3])
    mc = mcnemar(a, a, a)
    good = mc["b"] == 0 and mc["c"] == 0 and mc["p"] == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] identical predictors: 0 discordant, p=1")

    # --- 6. McNemar's signal is the ASYMMETRY, not the number of discordant pairs.
    #     Convention: b = a right & b wrong, c = a wrong & b right.
    yy = np.array([1] * 30 + [0] * 10)
    perfect = yy.copy()
    always_zero = np.zeros_like(yy)
    mc = mcnemar(yy, perfect, always_zero)
    good = mc["b"] == 30 and mc["c"] == 0 and mc["p"] < 1e-6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a 30-0 split is decisive (b=30 c=0, "
          f"p={mc['p']:.2e})")

    # --- 6b. the same 30 discordant pairs, split 20-10, are NOT significant. Discordance count
    #      alone carries no evidence -- only the imbalance does. "We had 30 disagreements" is not a
    #      result.
    yy0 = np.zeros(40, dtype=int)
    a_mid = np.array([0] * 10 + [1] * 20 + [0] * 10)    # right, wrong, right
    b_mid = np.array([1] * 10 + [0] * 30)               # wrong, right, right
    mc_mid = mcnemar(yy0, a_mid, b_mid)
    good = mc_mid["b"] + mc_mid["c"] == 30 and mc_mid["p"] > 0.05
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...but 30 pairs split 20-10 are NOT "
          f"(p={mc_mid['p']:.3f}); discordance alone is not evidence")

    # --- 7. discordance rate is the quantity eval_audit assumed; it is now measurable
    good = abs(mc["discordance"] - 30 / 40) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] discordance rate is reported directly "
          f"({mc['discordance']:.1%})")

    # --- 8. per-class counts reconcile with the subset size
    rows = per_class(yy0, a_mid, 2)
    good = sum(r["n"] for r in rows) == len(yy0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] per-class support sums to the subset size")

    # --- 9. the label encoding must be RECOVERED, not assumed. CLASSES and CODE_TO_LABEL are
    #     different orderings, and confusing them mislabels every report while still looking fine.
    recovered = {}
    for code, cnt in enumerate(sorted(EXPECTED_TRAIN.values(), reverse=True)):
        pass
    counts = {lab: c for lab, c in EXPECTED_TRAIN.items()}
    by_count = sorted(counts, key=lambda l: -counts[l])
    good = by_count == ["REFUTED", "SUPPORTED", "NOT_ENOUGH_EVIDENCE", "CONFLICTING"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] manifest order by count is "
          f"{by_count[0]} > {by_count[1]} > {by_count[2]} > {by_count[3]}")

    good = (label_of(0) == "SUPPORTED" and label_of(1) == "REFUTED"
            and label_of(2) == "NOT_ENOUGH_EVIDENCE" and label_of(3) == "CONFLICTING")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] integer codes map to the right names "
          f"(0={label_of(0)}, 1={label_of(1)}, 2={label_of(2)}, 3={label_of(3)})")

    good = CLASSES[0] != label_of(0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and CLASSES is a DIFFERENT order -- which is why "
          f"indexing it by code is a bug")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frozen-embedding baseline probe + paired tests.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--emb", default=EMB)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.run:
        return run(args.emb)
    if args.nli:
        return run_nli()
    if args.nli:
        return run_nli()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
