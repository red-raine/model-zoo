"""uncertainty.py — a correctness head, calibration, and a selective-risk controller.

WHAT THIS ANSWERS
-----------------
The 4-way verifier reports a verdict. It does not report *how much to trust that verdict*.
Those are different questions and the second one is what makes a system deployable:

    verdict   = argmax P(class | claim, evidence)         <- already have it (0.780 dev)
    trust     = P(the verdict is CORRECT | features)      <- this file
    abstain   = control the error rate on what we ACCEPT   <- this file

WHY A SEPARATE HEAD AND NOT THE SOFTMAX
---------------------------------------
Softmax confidence is not a correctness probability. It is fit to match labels on the
training distribution, it is systematically overconfident under distribution shift, and it
says nothing about whether *this particular* argmax is right. The literature is blunt about
the failure modes: evidential deep learning (Dirichlet/EDL) is known to fit the evidence
poorly and collapse, MC-dropout needs >20% error at 60% coverage where selective risk needs
10%, and BH-style conformal selection provides no rejection in a large fraction of
replicates. So the design here is deliberately boring and checkable:

  1. **Temperature scaling** — one parameter, fit on a held-out calibration split. Gives
     the "how likely is this true" number. Argmax-invariant, so accuracy cannot move.
  2. **A small logistic correctness head** — predicts P(prediction is correct) from the
     logits plus cheap text statistics, cross-fitted so the reported AUC is not the AUC of
     a model scoring its own training rows.
  3. **A selective-risk controller** — chooses an acceptance threshold with a
     Clopper-Pearson upper confidence bound on the accepted error rate, instead of an
     arbitrary softmax cutoff. This is the part that lets the system say "accept 78% of
     claims at <=15% error, escalate the rest" with a number that was *tested*, not tuned
     on the test set.

THE RULE THIS FILE OBEYS
------------------------
**Everything is fitted on the calibration split, never on dev.** Dev is measured once and
reported once. A conformal quantile fitted on the data it is evaluated on guarantees
nothing — that is the single most common way to publish a perfect and meaningless number.

THE HONEST CAVEAT
-----------------
The correctness head is *learned*, so the threshold is not a textbook split-conformal
guarantee (those need a fixed, a-priori score). Cross-fitting makes it approximately
valid; the dev measurement is the actual test of whether it held. `report()` prints both
the calib-side bound and the dev-side measurement, and the gap between them is the honest
error bar on the whole procedure.

USAGE
-----
    python tools/verify/uncertainty.py --self-test
    python tools/verify/uncertainty.py --logits B:\\datasets\\forge\\verifier-av\\cls-unc\\logits.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting"]
N_CLASSES = 4


# --------------------------------------------------------------------------- calibration
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(T, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(T, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    lp = log_softmax(logits, T)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """1-D golden-section search on log T. NLL is convex in log T for a softmax.

    Bounded to [0.05, 20]: outside that range the "calibration" is a story about a broken
    model, not a temperature, and silently returning T=0.001 would hide it.
    """
    lo, hi = math.log(0.05), math.log(20.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(logits, y, math.exp(c)), nll(logits, y, math.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logits, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logits, y, math.exp(d))
        if b - a < 1e-10:
            break
    return float(math.exp(0.5 * (a + b)))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on the confidence of the argmax."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    hit = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1.0
    return float(((probs - oh) ** 2).sum(axis=1).mean())


# ------------------------------------------------------------------------------ features
def text_stats(text: str) -> dict:
    """Cheap, non-model features. No tokeniser, no download, no cost.

    These exist because the logits alone cannot see the things that actually predict failure
    on this task: a claim with one short answer is easier than a claim with six contradicting
    ones, and a claim carrying numbers invites a numeric mismatch the encoder will miss.
    """
    claim, _, ev = text.partition("[SEP]")
    cw = claim.split()
    ew = ev.split()
    nums = sum(ch.isdigit() for ch in ev)
    return {
        "claim_words": len(cw),
        "ev_words": len(ew),
        "ev_claim_ratio": len(ew) / max(len(cw), 1),
        "ev_digits": nums / max(len(ev), 1),
        "ev_items": max(ev.count("|"), ev.count("Q:"), 1),
        "claim_has_digit": 1.0 if any(ch.isdigit() for ch in claim) else 0.0,
    }


def build_features(logits: np.ndarray, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """14 numeric features per claim: 8 from the model, 6 from the text."""
    p = softmax(logits)
    srt = np.sort(p, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / math.log(N_CLASSES)
    lp = log_softmax(logits)
    pred = p.argmax(axis=1)
    cols: dict[str, np.ndarray] = {
        "max_p": top1,
        "margin": top1 - top2,
        "entropy_norm": ent,
        "logp_pred": lp[np.arange(len(p)), pred],
        "p_sup": p[:, 0],
        "p_ref": p[:, 1],
        "p_nei": p[:, 2],
        "p_con": p[:, 3],
        "pred_is_refuted": (pred == 1).astype(np.float64),
    }
    stats = [text_stats(t) for t in texts]
    for k in stats[0].keys() if stats else []:
        cols[k] = np.array([s[k] for s in stats], dtype=np.float64)
    names = list(cols.keys())
    X = np.column_stack([cols[n] for n in names])
    return X, names


# ------------------------------------------------------------------------ correctness head
def cross_fit_probs(X: np.ndarray, y: np.ndarray, k: int = 5,
                    seed: int = 42) -> np.ndarray:
    """Out-of-fold P(correct). The reported AUC must not be a model scoring its own rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def fit_head(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    pipe.fit(X, y)
    return pipe


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney). Implemented directly so it cannot silently reorder classes."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------- selective risk controller
def cp_upper(errors: int, n: int, delta: float = 0.10) -> float:
    """One-sided (1-delta) Clopper-Pearson UPPER bound on a binomial rate.

    Used so the acceptance threshold is chosen against a *confidence bound* on the error
    rate, not against the observed error rate. Picking the threshold that looks best on the
    calibration split is how selective classifiers end up with worse dev risk than promised.
    """
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    try:
        from scipy.stats import beta as _beta
        return float(min(1.0, _beta.ppf(1.0 - delta, errors + 1, n - errors)))
    except Exception:  # noqa: BLE001 - numeric fallback, same definition
        def cdf(p: float) -> float:
            return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                       for k in range(errors + 1))
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if cdf(mid) > delta:
                lo = mid
            else:
                hi = mid
        return float(min(1.0, hi))


def select_threshold(score: np.ndarray, correct: np.ndarray, alpha: float,
                     delta: float = 0.10, min_accept: int = 30) -> dict:
    """Highest-coverage threshold whose CP upper bound on error is <= alpha.

    Sweeps candidate thresholds over the observed score grid, so it returns a threshold that
    actually corresponds to a real operating point rather than an interpolated fantasy.
    """
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct).astype(int)
    order = np.unique(score)
    best = {"threshold": 1.0 + 1e-9, "coverage": 0.0, "risk": 0.0,
            "risk_ucb": 1.0, "n_accept": 0}
    for t in order:
        acc = score >= t
        n = int(acc.sum())
        if n < min_accept:
            continue
        err = int((1 - correct[acc]).sum())
        ucb = cp_upper(err, n, delta)
        if ucb <= alpha and n > best["n_accept"]:
            best = {"threshold": float(t), "coverage": n / len(score),
                    "risk": err / n, "risk_ucb": ucb, "n_accept": n}
    return best


def mondrian_thresholds(score: np.ndarray, correct: np.ndarray, pred: np.ndarray,
                        alpha: float, delta: float = 0.10,
                        min_accept: int = 25) -> dict:
    """Per-predicted-class thresholds.

    Global thresholds are dominated by the majority class: with 61% `Refuted` the system can
    hit 15% overall error while accepting almost nothing but Refuted and silently guessing on
    `Conflicting`. Class-conditional thresholds cost coverage and buy the thing we actually
    want, which is that the tail is not quietly coin-flipped.
    """
    out = {}
    for c in range(N_CLASSES):
        m = pred == c
        if m.sum() < min_accept:
            out[c] = {"threshold": 1.0 + 1e-9, "coverage": 0.0, "risk": 0.0,
                      "risk_ucb": 1.0, "n_accept": 0, "n_class": int(m.sum()),
                      "note": "too few calibration rows"}
            continue
        r = select_threshold(score[m], correct[m], alpha, delta, min_accept=15)
        r["n_class"] = int(m.sum())
        out[c] = r
    return out


# ------------------------------------------------------------------------ risk/coverage
def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by decreasing score; at each prefix report coverage and cumulative error."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    c = np.asarray(correct)[order].astype(int)
    err = np.cumsum(1 - c)
    cov = np.arange(1, len(c) + 1) / len(c)
    return cov, err / np.arange(1, len(c) + 1)


def aurc(score: np.ndarray, correct: np.ndarray) -> tuple[float, float, float]:
    """Area under the risk-coverage curve, plus the random-ranking reference and the excess.

    AURC alone is not comparable across datasets: a model on an easy dataset has low AURC at
    any quality. `excess` = AURC - overall_error is the part that comes from the *ranking*
    and is the number that can be compared. Deliberately NOT called AUGRC: AUGRC in
    arXiv:2407.01032 is a specific normalisation I have not reimplemented, and inventing a
    formula under that name would be the exact sloppiness this project keeps catching.
    """
    cov, risk = risk_coverage(score, correct)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    a = float(_trapz(risk, cov)) if len(cov) > 1 else 0.0
    overall = float(1.0 - np.asarray(correct).mean())
    return a, overall, a - overall


def selected_stats(score: np.ndarray, correct: np.ndarray, pred: np.ndarray,
                   label: np.ndarray, threshold: float) -> dict:
    acc = np.asarray(score) >= threshold
    n = int(acc.sum())
    d: dict = {"n_accept": n, "coverage": n / len(score),
               "abstain_rate": 1.0 - n / len(score)}
    if n == 0:
        d.update({"risk": 0.0, "accuracy_accepted": 0.0, "macro_f1_accepted": 0.0})
        return d
    d["risk"] = float(1.0 - np.asarray(correct)[acc].mean())
    d["accuracy_accepted"] = float(np.asarray(correct)[acc].mean())
    from sklearn.metrics import f1_score
    # macro-F1 ON THE ACCEPTED SUBSET, against the GOLD labels. An earlier version scored
    # labels against themselves and reported a satisfying 1.000 -- the same class of mistake
    # as the parser that matched any label in a degenerate generation.
    d["macro_f1_accepted"] = float(f1_score(np.asarray(label)[acc], np.asarray(pred)[acc],
                                            average="macro", labels=list(range(N_CLASSES)),
                                            zero_division=0))
    d["risk_ucb"] = cp_upper(int((1 - np.asarray(correct)[acc]).sum()), n, 0.10)
    return d



# -------------------------------------------------------------------------------- report
def load_logits(path: str) -> dict:
    splits: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            s = splits.setdefault(r["split"], {"logits": [], "label": [], "pred": [],
                                               "text": []})
            s["logits"].append(r["logits"])
            s["label"].append(r["label"])
            s["pred"].append(r["pred"])
            s["text"].append(r["text"])
    return {k: {kk: np.asarray(vv) if kk != "text" else vv for kk, vv in v.items()}
            for k, v in splits.items()}


def run_self_test() -> bool:
    print("uncertainty.py - SELF TEST")
    ok = True
    rng = np.random.default_rng(0)

    # --- temperature scaling recovers a known miscalibration and never moves argmax
    # The base distribution must be EXACTLY calibrated, otherwise "the true temperature" is
    # not T_true and a correct fitter looks broken. Two earlier versions of this test were
    # wrong: random labels make the optimum undefined, and a boosted-logit construction
    # leaves the base underconfident (grid search confirmed T=1.19 was the real argmin, so
    # the fitter was right and the test was wrong). Sampling p from a Dirichlet, drawing y
    # from p, and using log p as the logits gives a provably calibrated base.
    n, T_true = 20000, 3.0
    p_true = rng.dirichlet(np.full(4, 0.4), n)
    y = np.array([rng.choice(4, p=row) for row in p_true])
    base = np.log(np.clip(p_true, 1e-9, 1.0))      # softmax(base) == p_true exactly
    hot = base * T_true                            # overconfident by a known factor
    T_hat = fit_temperature(hot, y)
    good = abs(T_hat - T_true) / T_true < 0.10
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] recovers T (true {T_true}, fitted {T_hat:.3f})")



    p_before, p_after = softmax(hot), softmax(hot, T_hat)
    good = np.array_equal(p_before.argmax(1), p_after.argmax(1))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] argmax invariant under scaling")

    e0, e1 = ece(p_before, y), ece(p_after, y)
    good = e1 < e0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ECE falls {e0:.3f} -> {e1:.3f}")

    # --- AUC sanity: a perfect score is 1.0, a constant score is 0.5
    c = rng.integers(0, 2, 500)
    good = auc(c.astype(float), c) == 1.0 and abs(auc(np.ones(500), c) - 0.5) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC bounds (perfect 1.0, constant 0.5)")

    # --- CP upper bound: must exceed the observed rate, and shrink with n
    u_small = cp_upper(15, 100, 0.10)
    u_big = cp_upper(1500, 10000, 0.10)
    good = u_small > 0.15 and u_big > 0.15 and u_big < u_small
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] CP upper bound {u_small:.3f} (n=100) "
          f"> {u_big:.3f} (n=10k), both > observed 0.15")

    # --- the controller actually controls: synthetic dev with a known-good score
    # py is the TRUE per-item correctness probability. It must be biased high, otherwise a
    # 15% error rate simply does not exist in the population and "zero coverage" is the
    # CORRECT answer -- a Beta(2,5) version of this test failed for exactly that reason and
    # was wrong, not the controller.
    m = 20000
    py = rng.beta(6, 2, m)
    corr = (rng.random(m) < py).astype(int)
    score = np.clip(py + rng.normal(0, 0.02, m), 0, 1)
    cal, devi = slice(0, 5000), slice(5000, m)
    thr = select_threshold(score[cal], corr[cal], alpha=0.15, delta=0.10)
    cov_dev = float((score[devi] >= thr["threshold"]).mean())
    dev_risk = float(1 - corr[devi][score[devi] >= thr["threshold"]].mean()) if cov_dev else 1.0
    good = cov_dev > 0.10 and dev_risk <= 0.18
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] controller holds on held-out data "
          f"(alpha .15 -> dev risk {dev_risk:.3f}, coverage {cov_dev:.3f})")


    # --- a random score must NOT be accepted at any threshold worth having
    rnd = rng.random(m)
    thr_r = select_threshold(rnd[cal], corr[cal], alpha=0.15, delta=0.10)
    good = thr_r["coverage"] < 0.10
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] noise score rejected "
          f"(coverage {thr_r['coverage']:.3f})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Correctness head + selective risk control.")
    ap.add_argument("--logits", default=r"B:\datasets\forge\verifier-av\cls-unc\logits.jsonl")
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="target error rate on accepted claims")
    ap.add_argument("--delta", type=float, default=0.10,
                    help="confidence for the Clopper-Pearson bound")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    S = load_logits(args.logits)
    if "calib" not in S or "dev" not in S:
        print(f"[unc] need calib and dev splits, got {list(S)}")
        return 2
    out: dict = {"alpha": args.alpha, "delta": args.delta}

    # ---------------------------------------------------------------- 1. temperature
    cal, dev = S["calib"], S["dev"]
    T = fit_temperature(cal["logits"], cal["label"])
    out["temperature"] = T
    print(f"[unc] temperature T = {T:.4f} (fitted on calib only)")
    for name, sp in (("calib", cal), ("dev", dev)):
        pb = softmax(sp["logits"])
        pa = softmax(sp["logits"], T)
        accb = float((pb.argmax(1) == sp["label"]).mean())
        acca = float((pa.argmax(1) == sp["label"]).mean())
        print(f"[unc] {name}: ECE {ece(pb, sp['label']):.4f} -> {ece(pa, sp['label']):.4f} · "
              f"Brier {brier(pb, sp['label']):.4f} -> {brier(pa, sp['label']):.4f} · "
              f"acc {accb:.4f} -> {acca:.4f}")
        out[f"{name}_calibration"] = {
            "ece_before": ece(pb, sp["label"]), "ece_after": ece(pa, sp["label"]),
            "brier_before": brier(pb, sp["label"]), "brier_after": brier(pa, sp["label"]),
            "acc_before": accb, "acc_after": acca}
        assert abs(acca - accb) < 1e-12, "temperature scaling moved accuracy"

    # ------------------------------------------------- 2. correctness head, cross-fitted
    Xc, names = build_features(cal["logits"], cal["text"])
    Xd, _ = build_features(dev["logits"], dev["text"])
    yc = (cal["pred"] == cal["label"]).astype(int)
    yd = (dev["pred"] == dev["label"]).astype(int)
    print(f"[unc] features {len(names)}: {names}")
    print(f"[unc] base correctness: calib {yc.mean():.4f} · dev {yd.mean():.4f}")

    ablate = {}
    for tag, idx in (("all", list(range(len(names)))),
                     ("logits_only", [i for i, n in enumerate(names)
                                      if n not in ("claim_words", "ev_words",
                                                   "ev_claim_ratio", "ev_digits",
                                                   "ev_items", "claim_has_digit")]),
                     ("text_only", [i for i, n in enumerate(names)
                                    if n in ("claim_words", "ev_words", "ev_claim_ratio",
                                             "ev_digits", "ev_items", "claim_has_digit")])):
        oof = cross_fit_probs(Xc[:, idx], yc)
        a_cal = auc(oof, yc)
        head = fit_head(Xc[:, idx], yc)
        p_dev = head.predict_proba(Xd[:, idx])[:, 1]
        a_dev = auc(p_dev, yd)
        ablate[tag] = {"auc_calib_oof": a_cal, "auc_dev": a_dev}
        print(f"[unc] head[{tag:11s}] AUC calib-OOF {a_cal:.4f} · dev {a_dev:.4f}")
    out["head_ablation"] = ablate

    # --------------------------------------------- 3. which score? decided on CALIB, not dev
    # Three candidates. Selecting on dev would tune the headline on the test set, so the
    # winner is chosen by CALIB AUC and the dev column is reported afterwards, win or lose.
    oof = cross_fit_probs(Xc, yc)
    head = fit_head(Xc, yc)
    pc, pd_ = softmax(cal["logits"], T), softmax(dev["logits"], T)
    cand = {
        "correctness_head": (oof, head.predict_proba(Xd)[:, 1]),
        "max_p_calibrated": (pc.max(1), pd_.max(1)),
        # NEGATIVE entropy: higher = more confident = more likely correct. Written with the
        # sign that makes "higher is better" hold for every candidate, because a score whose
        # AUC comes out at 0.24 is not a weak score, it is a score with the wrong sign, and
        # reporting 0.24 without noticing that would be a silent error.
        "neg_entropy": ((pc * np.log(np.clip(pc, 1e-12, 1.0))).sum(1),
                        (pd_ * np.log(np.clip(pd_, 1e-12, 1.0))).sum(1)),
    }
    tbl = {}
    print()
    for k, (c, d) in cand.items():
        tbl[k] = {"auc_calib": auc(c, yc), "auc_dev": auc(d, yd)}
        print(f"[unc] score[{k:17s}] AUC calib {tbl[k]['auc_calib']:.4f} · "
              f"dev {tbl[k]['auc_dev']:.4f}")
    out["score_table"] = tbl
    winner = max(tbl, key=lambda k: tbl[k]["auc_calib"])
    print(f"[unc] SELECTED ON CALIB: {winner}  (its dev AUC {tbl[winner]['auc_dev']:.4f})")
    if winner != "correctness_head":
        print("[unc] NOTE: the learned head did NOT win; the free score did.")
    out["selected_score"] = winner
    s_cal, s_dev = cand[winner]
    out["auc_calib"] = tbl[winner]["auc_calib"]
    out["auc_dev"] = tbl[winner]["auc_dev"]

    print("\n[unc] === GATE: correctness-score AUC (selected score) ===")
    print(f"[unc] calib {out['auc_calib']:.4f} · dev {out['auc_dev']:.4f}")
    if out["auc_dev"] < 0.70:
        print("[unc] AUC < 0.70 -> cannot support a useful controller. STOP.")
    elif out["auc_dev"] < 0.80:
        print("[unc] 0.70-0.80 -> weak; usable at alpha >= 0.20.")
    else:
        print("[unc] AUC >= 0.80 -> the controller should fire at alpha=0.15.")

    # operating points: one sweep, so the trade-off is visible instead of one cherry-picked a
    print(f"\n[unc] === selective risk sweep (delta={args.delta}) ===")
    print(f"[unc] {'alpha':>5} {'calib cov':>9} {'calib risk':>10} {'ucb':>6} "
          f"{'DEV cov':>8} {'DEV risk':>8} {'DEV F1':>7}  verdict")
    sweep = {}
    for a in (0.05, 0.10, 0.15, 0.20, 0.25):
        thr = select_threshold(s_cal, yc, a, args.delta)
        dd = selected_stats(s_dev, yd, dev["pred"], dev["label"], thr["threshold"])
        sweep[str(a)] = {"threshold": thr["threshold"], "calib_coverage": thr["coverage"],
                         "calib_risk": thr["risk"], "calib_ucb": thr["risk_ucb"],
                         "calib_n": thr["n_accept"], **dd}
        v = "HELD" if dd["risk"] <= a else "VIOLATED"
        print(f"[unc] {a:5.2f} {thr['coverage']:9.3f} {thr['risk']:10.3f} "
              f"{thr['risk_ucb']:6.3f} {dd['coverage']:8.3f} {dd['risk']:8.3f} "
              f"{dd['macro_f1_accepted']:7.3f}  {v}")
    out["alpha_sweep"] = sweep

    print(f"\n[unc] === the requested alpha={args.alpha} operating point ===")
    thr = select_threshold(s_cal, yc, args.alpha, args.delta)
    out["threshold"] = thr
    d = selected_stats(s_dev, yd, dev["pred"], dev["label"], thr["threshold"])
    out["dev_global"] = d
    print(f"[unc] global threshold {thr['threshold']:.4f} · calib coverage "
          f"{thr['coverage']:.4f} · calib risk {thr['risk']:.4f} "
          f"(UCB {thr['risk_ucb']:.4f}, n={thr['n_accept']})")
    print(f"[unc] DEV: coverage {d['coverage']:.4f} · abstain {d['abstain_rate']:.4f} · "
          f"risk {d['risk']:.4f} · UCB {d['risk_ucb']:.4f} · "
          f"macro-F1 on accepted {d['macro_f1_accepted']:.4f}")
    print(f"[unc] dev risk vs the calib promise: "
          f"{'HELD' if d['risk'] <= args.alpha else 'VIOLATED'}")

    mth = mondrian_thresholds(s_cal, yc, cal["pred"], args.alpha, args.delta)
    out["mondrian"] = {str(k): v for k, v in mth.items()}
    print(f"\n[unc] Mondrian (per predicted class, alpha={args.alpha}):")
    acc_any = np.zeros(len(s_dev), dtype=bool)
    for c in range(N_CLASSES):
        t = mth[c]["threshold"]
        m = dev["pred"] == c
        sel = m & (s_dev >= t)
        acc_any |= sel
        n, e = int(sel.sum()), int((1 - yd[sel]).sum())
        note = "  <-- abstains entirely" if n == 0 and int(m.sum()) > 0 else ""
        print(f"[unc]   {LABELS[c]:12s} thr {t:.3f} · calib n={mth[c]['n_class']:3d} · "
              f"dev n={int(m.sum()):3d} · accepted {n:3d} · "
              f"risk {e / max(n, 1):.3f}{note}")
    n_m, e_m = int(acc_any.sum()), int((1 - yd[acc_any]).sum())
    out["dev_mondrian"] = {"coverage": acc_any.mean(), "risk": e_m / max(n_m, 1),
                           "n_accept": n_m, "risk_ucb": cp_upper(e_m, n_m, 0.10)}
    print(f"[unc]   Mondrian overall: coverage {out['dev_mondrian']['coverage']:.4f} · "
          f"risk {out['dev_mondrian']['risk']:.4f} "
          f"(UCB {out['dev_mondrian']['risk_ucb']:.4f})")
    if out["dev_mondrian"]["coverage"] <= d["coverage"] + 1e-9:
        print("[unc]   Mondrian bought nothing here -- per-class calib subsets are too "
              "small (31-281 rows) to certify the target risk.")

    # ------------------------------------------------------------ 4. risk-coverage curve
    a_sel, overall, excess = aurc(s_dev, yd)
    out["aurc"] = {"selected": a_sel, "overall_error": overall, "excess": excess}
    print("\n[unc] === risk-coverage ===")
    print(f"[unc] overall dev error {overall:.4f}")
    print(f"[unc] AURC {a_sel:.4f} (excess vs random ranking {excess:+.4f})")
    cov, risk = risk_coverage(s_dev, yd)
    print("[unc] curve (coverage -> risk):")
    for target in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00):
        i = min(int(target * len(cov)) - 1, len(cov) - 1)
        if i >= 0:
            print(f"[unc]   {cov[i]:.2f} -> {risk[i]:.4f}")
    out["risk_coverage"] = {"coverage": cov[::10].tolist(), "risk": risk[::10].tolist()}


    out["head_majority_baseline_auc"] = 0.5
    out["n"] = {"calib": int(len(yc)), "dev": int(len(yd))}
    p = args.out or os.path.join(os.path.dirname(args.logits), "uncertainty.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[unc] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
