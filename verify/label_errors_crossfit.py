"""label_errors_crossfit.py — label-error detection done the way the field does it.

WHAT I GOT WRONG FIRST, AND WHAT THIS DOES INSTEAD
--------------------------------------------------
I measured "two models agree against gold" using deberta-v3-base fine-tuned on AVeriTeC
train and the CTU large fine-tuned on AVeriTeC train -- **the same corpus** -- and reported
the resulting 2.22x error-correlation lift as if it were a fact about model ensembles. It
was a property of my experimental design. Correlated errors in ensembles that share training
data is textbook; it is the reason confident learning (Northcutt et al., JAIR 2022 / NeurIPS
2021 "Pervasive Label Errors") uses **cross-validated out-of-fold** predictions instead of
two models trained on everything, and the reason ensemble papers measure diversity at all.

So this file does two things:

  A. QUANTIFIES the design error. Same frozen backbone, two heads, evaluated on a common
     holdout. One arm trains the heads on DISJOINT subsets, the other on the SAME data.
     If the disjointness matters, the disjoint arm's error correlation must fall.

  B. FINDS LABEL ERRORS the standard way: freeze the backbone, cross-fit small heads so
     every row is scored by a head that never saw it, then rank rows with cleanlab's
     confident-learning scores. No hand-set weights.

THE VALIDATION THAT ACTUALLY PROVES SOMETHING
---------------------------------------------
Validating against `required_reannotation` is a proxy -- it asks "does this find rows the
humans doubted", not "does this find rows that are wrong". So the self-test injects KNOWN
label corruption into synthetic data and measures recovery precision@k. If the pipeline is
broken, precision collapses to the base rate (the fraction corrupted). That is a real test;
an AUC against a proxy is not.

USAGE
-----
    python tools/verify/label_errors_crossfit.py --self-test
    python tools/verify/label_errors_crossfit.py --embeddings B:\\...\\emb-frozen.npz ^
        --clean-out B:\\...\\train.crossfit-cleaned.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")
NUM_LABELS = 4
I2L = {0: "Supported", 1: "Refuted", 2: "Not Enough Evidence", 3: "Conflicting"}


# ------------------------------------------------------------------------------- metrics
def error_lift(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """P(B wrong | A wrong) / P(B wrong) -- how much B's failure is predicted by A's.

    1.0 means genuinely independent failures. Higher means the two heads fail together.
    """
    ea, eb = pred_a != y, pred_b != y
    base = float(eb.mean())
    if ea.sum() == 0 or base == 0:
        return {"lift": float("nan"), "agree": float((pred_a == pred_b).mean()),
                "err_a": float(ea.mean()), "err_b": base}
    joint = float((ea & eb).sum()) / float(ea.sum())
    return {"lift": joint / base, "agree": float((pred_a == pred_b).mean()),
            "err_a": float(ea.mean()), "err_b": base, "joint_given_a": joint}


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient between two boolean vectors -- correlation of the two error patterns."""
    a = a.astype(bool)
    b = b.astype(bool)
    n11 = float((a & b).sum())
    n10 = float((a & ~b).sum())
    n01 = float((~a & b).sum())
    n00 = float((~a & ~b).sum())
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if den <= 0:
        return float("nan")
    return (n11 * n00 - n10 * n01) / np.sqrt(den)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    n1 = int(np.sum(y))
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ------------------------------------------------------------------------- head training
def make_head(seed: int = 0, hidden: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if hidden > 0:
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=400,
                                           random_state=seed, early_stopping=False))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=3000, random_state=seed))


def fit_predict(Xtr, ytr, Xte, hidden=0, seed=0) -> np.ndarray:
    h = make_head(seed, hidden)
    h.fit(Xtr, ytr)
    return np.asarray(h.predict(Xte))


def fit_proba(Xtr, ytr, Xte, hidden=0, seed=0) -> np.ndarray:
    h = make_head(seed, hidden)
    h.fit(Xtr, ytr)
    p = np.asarray(h.predict_proba(Xte))
    # align columns to the global label set: a fold may miss a class entirely
    if p.shape[1] != NUM_LABELS:
        full = np.full((len(Xte), NUM_LABELS), 1e-6)
        for col, cls in enumerate(h.classes_):
            full[:, int(cls)] = p[:, col]
        p = full / full.sum(1, keepdims=True)
    return p


def oof_proba(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0,
              hidden: int = 0) -> np.ndarray:
    """Cross-fitted out-of-fold probabilities: every row scored by a head that never saw it."""
    from sklearn.model_selection import StratifiedKFold
    oof = np.full((len(y), NUM_LABELS), 1.0 / NUM_LABELS)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        oof[te] = fit_proba(X[tr], y[tr], X[te], hidden=hidden, seed=seed)
    return oof


def cleanlab_issues(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Ranked suspect indices, lowest label quality first. Uses cleanlab when available."""
    try:
        from cleanlab.filter import find_label_issues
        return np.asarray(find_label_issues(
            labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence"))
    except Exception as e:  # noqa: BLE001
        print(f"[cf] cleanlab unavailable ({e}); falling back to self-confidence ranking")
        return np.argsort(probs[np.arange(len(y)), y])          # worst p(gold) first


def label_quality(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-row score in [0,1], higher = label more trustworthy."""
    try:
        from cleanlab.rank import get_label_quality_scores
        return np.asarray(get_label_quality_scores(labels=y, pred_probs=probs))
    except Exception:  # noqa: BLE001
        return probs[np.arange(len(y)), y]


# ------------------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("label_errors_crossfit.py - SELF TEST")
    ok = True
    rng = np.random.default_rng(0)

    # --- synthetic data with KNOWN label corruption: the only validation with ground truth
    n, d, k = 3000, 64, 4
    centroids = rng.normal(0, 1.0, (k, d))
    y_true = rng.integers(0, k, n)
    X = centroids[y_true] + rng.normal(0, 1.6, (n, d))
    y_noisy = y_true.copy()
    n_bad = int(0.10 * n)
    bad = rng.choice(n, n_bad, replace=False)
    y_noisy[bad] = (y_noisy[bad] + rng.integers(1, k, n_bad)) % k     # flip to a wrong class

    probs = oof_proba(X, y_noisy, k=5, seed=0)
    issues = cleanlab_issues(y_noisy, probs)
    top = set(issues[:n_bad].tolist())
    prec = len(top & set(bad.tolist())) / n_bad
    good = prec > 0.50
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] recovers injected label noise: "
          f"precision@{n_bad} = {prec:.3f} (base rate {n_bad/n:.2f})")

    # --- and it must NOT flag clean rows at a high rate
    clean_prec = len(top - set(bad.tolist())) / n_bad
    good = clean_prec < 0.50
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] does not mostly flag clean rows "
          f"({clean_prec:.3f} of the top-k are actually clean)")

    # --- disjoint vs shared training data: the arm the earlier experiment skipped
    idx = rng.permutation(n)
    s1, s2, s3 = idx[: int(0.4 * n)], idx[int(0.4 * n): int(0.8 * n)], idx[int(0.8 * n):]
    pa = fit_predict(X[s1], y_noisy[s1], X[s3])
    pb = fit_predict(X[s2], y_noisy[s2], X[s3])
    dis = error_lift(pa, pb, y_noisy[s3])
    qa = fit_predict(X[idx[: int(0.8 * n)]], y_noisy[idx[: int(0.8 * n)]], X[s3])
    qb = fit_predict(X[idx[: int(0.8 * n)]], y_noisy[idx[: int(0.8 * n)]], X[s3], seed=7)
    shr = error_lift(qa, qb, y_noisy[s3])
    print(f"  [INFO] disjoint-data heads: lift {dis['lift']:.2f} · phi "
          f"{phi(pa != y_noisy[s3], pb != y_noisy[s3]):+.3f} · agree {dis['agree']:.3f}")
    print(f"  [INFO] shared-data heads:   lift {shr['lift']:.2f} · phi "
          f"{phi(qa != y_noisy[s3], qb != y_noisy[s3]):+.3f} · agree {shr['agree']:.3f}")
    good = dis["lift"] <= shr["lift"] + 0.25
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] disjoint training does not increase error "
          f"correlation vs shared training")

    good = auc(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1])) == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC sanity")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-fitted label-error detection.")
    ap.add_argument("--embeddings", default=os.path.join(DATA, "emb-frozen.npz"))
    ap.add_argument("--pool", default="cls", choices=("cls", "mean"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=0, help="MLP head width (0 = logistic)")
    ap.add_argument("--raw", default=r"B:\datasets\forge\downloads\averitec\train.json")
    ap.add_argument("--clean-out", default="")
    ap.add_argument("--precision-at", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    z = np.load(args.embeddings, allow_pickle=True)
    Xtr = z[f"X_train_{args.pool}"]
    ytr = z["y_train"]
    Xdv = z[f"X_dev_{args.pool}"]
    ydv = z["y_dev"]
    ttr = [str(t) for t in z["text_train"]]
    print(f"[cf] embeddings {Xtr.shape} train · {Xdv.shape} dev · pool={args.pool} "
          f"· head={'MLP'+str(args.hidden) if args.hidden else 'logistic'}")

    # ------------------------------------------------ A. disjoint vs shared training data
    rng = np.random.default_rng(0)
    n = len(ytr)
    idx = rng.permutation(n)
    s1, s2, s3 = idx[: int(0.4 * n)], idx[int(0.4 * n): int(0.8 * n)], idx[int(0.8 * n):]
    print(f"\n[cf] === A. does training-data overlap inflate error correlation? ===")
    print(f"[cf] split: S1 {len(s1)} · S2 {len(s2)} · S3 holdout {len(s3)}")
    pa = fit_predict(Xtr[s1], ytr[s1], Xtr[s3], hidden=args.hidden, seed=0)
    pb = fit_predict(Xtr[s2], ytr[s2], Xtr[s3], hidden=args.hidden, seed=1)
    dis = error_lift(pa, pb, ytr[s3])
    u = np.concatenate([s1, s2])
    qa = fit_predict(Xtr[u], ytr[u], Xtr[s3], hidden=args.hidden, seed=0)
    qb = fit_predict(Xtr[u], ytr[u], Xtr[s3], hidden=args.hidden, seed=1)
    shr = error_lift(qa, qb, ytr[s3])
    print(f"[cf] DISJOINT train sets (S1 vs S2), eval on S3:")
    print(f"[cf]   agreement {dis['agree']:.3f} · errA {dis['err_a']:.3f} "
          f"· errB {dis['err_b']:.3f} · LIFT {dis['lift']:.2f}")
    print(f"[cf]   phi(errorA, errorB) {phi(pa != ytr[s3], pb != ytr[s3]):+.4f}")
    print(f"[cf] SHARED train set (S1+S2 for both):")
    print(f"[cf]   agreement {shr['agree']:.3f} · errA {shr['err_a']:.3f} "
          f"· errB {shr['err_b']:.3f} · LIFT {shr['lift']:.2f}")
    print(f"[cf]   phi(errorA, errorB) {phi(qa != ytr[s3], qb != ytr[s3]):+.4f}")
    delta = shr["lift"] - dis["lift"]
    print(f"[cf] lift penalty from sharing data: {delta:+.2f} "
          f"({'overlap inflates correlation, as expected' if delta > 0.1 else 'no clear effect here'})")

    # ------------------------------------------------------------- B. find label errors
    print(f"\n[cf] === B. cross-fitted label-error detection ({args.folds}-fold) ===")
    probs = oof_proba(Xtr, ytr, k=args.folds, seed=0, hidden=args.hidden)
    oof_acc = float((probs.argmax(1) == ytr).mean())
    print(f"[cf] out-of-fold train accuracy {oof_acc:.4f} (honest: no head saw its own row)")
    issues = cleanlab_issues(ytr, probs)
    q = label_quality(ytr, probs)
    suspicion = 1.0 - q
    print(f"[cf] flagged {len(issues)} suspect rows by cleanlab ranking")

    # per-class breakdown of what is suspected
    cc = collections.Counter(I2L[int(ytr[i])] for i in issues)
    print(f"[cf] suspected by gold label: {dict(cc)}")

    # validate against human flags if the raw file is present
    if os.path.exists(args.raw):
        raw = json.load(open(args.raw, encoding="utf-8"))
        flag = {r["claim"].strip().lower(): bool(r.get("required_reannotation")) for r in raw}
        yflag, have = [], 0
        for t in ttr:
            c = t.split("[SEP]")[0].strip().lower()
            if c in flag:
                yflag.append(int(flag[c]))
                have += 1
            else:
                yflag.append(0)
        yflag = np.asarray(yflag)
        if yflag.sum() > 0:
            a = auc(suspicion, yflag)
            print(f"[cf] VALIDATION vs required_reannotation ({have} joined, "
                  f"{int(yflag.sum())} flagged): AUC = {a:.4f}")
            print(f"[cf]   earlier hand-rolled finder scored 0.5774 on dev. This is the number "
                  f"to beat.")

    if args.precision_at > 0:
        kbad = args.precision_at
        print(f"[cf] top {kbad} suspects:")
        for i in issues[:kbad]:
            print(f"[cf]   p(gold)={probs[i, ytr[i]]:.3f} gold={I2L[int(ytr[i])]:12s} "
                  f"pred={I2L[int(probs[i].argmax())]:12s} {ttr[i].split('[SEP]')[0][:70]}")

    # evaluate a head trained WITHOUT the suspects to see if it helps dev
    drop = set(issues[: int(0.10 * n)].tolist())
    keep = [i for i in range(n) if i not in drop]
    p_kept = fit_predict(Xtr[keep], ytr[keep], Xdv, hidden=args.hidden, seed=0)
    p_all = fit_predict(Xtr, ytr, Xdv, hidden=args.hidden, seed=0)
    print(f"\n[cf] === C. does dropping the top 10% suspects help HELD-OUT dev? ===")
    print(f"[cf] dev acc with all train rows      {float((p_all == ydv).mean()):.4f}")
    print(f"[cf] dev acc without the top {len(drop):4d} suspects "
          f"{float((p_kept == ydv).mean()):.4f}")
    print(f"[cf] (dev was never used for fitting or for choosing the suspects)")

    if args.clean_out:
        suspect_texts = {ttr[i] for i in drop}
        src = os.path.join(DATA, "train.jsonl")
        kept = dropped = 0
        with open(src, encoding="utf-8") as fh, open(args.clean_out, "w",
                                                     encoding="utf-8") as out:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ev = (r.get("prompt") or "").split("EVIDENCE:")[-1]
                ev = ev.split("Answer with exactly one word")[0].strip()
                t = f"{(r.get('claim') or '').strip()} [SEP] {ev}"
                if t in suspect_texts:
                    dropped += 1
                else:
                    kept += 1
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[cf] cleaned train -> {args.clean_out}: kept {kept}, dropped {dropped}")


    outp = os.path.join(DATA, "label-errors.crossfit.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({
            "pool": args.pool, "folds": args.folds, "oof_acc": oof_acc,
            "disjoint": dis, "shared": shr,
            "phi_disjoint": phi(pa != ytr[s3], pb != ytr[s3]),
            "phi_shared": phi(qa != ytr[s3], qb != ytr[s3]),
            "top_suspects": [{"claim": ttr[i].split("[SEP]")[0],
                              "gold": I2L[int(ytr[i])],
                              "pred": I2L[int(probs[i].argmax())],
                              "p_gold": float(probs[i, ytr[i]]),
                              "quality": float(q[i])} for i in issues[:200]],
        }, fh, indent=2, default=float)
    print(f"[cf] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
