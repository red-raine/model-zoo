"""clean_controls.py — the two controls the cleaning result is not valid without.

Two flaws in the first pass, both caught before reporting:

  A. ARM A WAS DEGENERATE. "Two heads trained on the same data" were two logistic
     regressions on the same features, so they produced *identical* predictions
     (agreement 1.000, phi 1.0000). Comparing that to disjoint heads measures "identical
     models are identical", which is tautological. Fixed here by giving both arms the SAME
     head-type pair (logistic vs MLP) so the only variable is training-data overlap.

  B. ARM C HAD NO CONTROL. Dropping the 306 most-suspect train rows raised held-out dev
     accuracy by 3.8 points. But dropping ANY 306 rows changes the fit, and a weak
     logistic-on-frozen-embeddings head is high-variance. Without a random-drop control the
     gain could be noise. Fixed here: 10 random-drop replicates, same size, compare means.

This is the difference between "cleaning helped" and "I removed some rows and the number
moved".

USAGE
-----
    python tools/verify/clean_controls.py --embeddings B:\\...\\emb-frozen.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_errors_crossfit import (auc, cleanlab_issues, error_lift,  # noqa: E402
                                   fit_predict, oof_proba, phi)

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Controls for the cleaning result.")
    ap.add_argument("--embeddings", default=os.path.join(DATA, "emb-frozen.npz"))
    ap.add_argument("--pool", default="cls")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drop-frac", type=float, default=0.10)
    ap.add_argument("--replicates", type=int, default=10)
    args = ap.parse_args(argv)

    z = np.load(args.embeddings, allow_pickle=True)
    Xtr, ytr = z[f"X_train_{args.pool}"], z["y_train"]
    Xdv, ydv = z[f"X_dev_{args.pool}"], z["y_dev"]
    n = len(ytr)
    print(f"[ctl] train {Xtr.shape} · dev {Xdv.shape} · pool={args.pool}")

    # ------------------------------------------------- A'. fair disjoint-vs-shared design
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    s1 = idx[: int(0.4 * n)]
    s2 = idx[int(0.4 * n): int(0.8 * n)]
    s3 = idx[int(0.8 * n):]
    print(f"\n[ctl] === A'. data overlap only (both arms: logistic vs MLP) ===")
    print(f"[ctl] S1 {len(s1)} · S2 {len(s2)} · holdout S3 {len(s3)}")

    a_log = fit_predict(Xtr[s1], ytr[s1], Xtr[s3], hidden=0, seed=0)
    a_mlp = fit_predict(Xtr[s2], ytr[s2], Xtr[s3], hidden=args.hidden, seed=0)
    dis = error_lift(a_log, a_mlp, ytr[s3])
    print(f"[ctl] DISJOINT  logistic(S1) vs MLP(S2):   agree {dis['agree']:.3f} · "
          f"errA {dis['err_a']:.3f} errB {dis['err_b']:.3f} · LIFT {dis['lift']:.2f} · "
          f"phi {phi(a_log != ytr[s3], a_mlp != ytr[s3]):+.3f}")

    u = np.concatenate([s1, s2])
    b_log = fit_predict(Xtr[u], ytr[u], Xtr[s3], hidden=0, seed=0)
    b_mlp = fit_predict(Xtr[u], ytr[u], Xtr[s3], hidden=args.hidden, seed=0)
    shr = error_lift(b_log, b_mlp, ytr[s3])
    print(f"[ctl] SHARED    logistic(U) vs MLP(U):     agree {shr['agree']:.3f} · "
          f"errA {shr['err_a']:.3f} errB {shr['err_b']:.3f} · LIFT {shr['lift']:.2f} · "
          f"phi {phi(b_log != ytr[s3], b_mlp != ytr[s3]):+.3f}")
    print(f"[ctl] phi: disjoint {phi(a_log != ytr[s3], a_mlp != ytr[s3]):+.3f} vs "
          f"shared {phi(b_log != ytr[s3], b_mlp != ytr[s3]):+.3f}")
    dphi = phi(b_log != ytr[s3], b_mlp != ytr[s3]) - phi(a_log != ytr[s3], a_mlp != ytr[s3])
    print(f"[ctl] delta phi from sharing training data: {dphi:+.3f} "
          f"({'sharing inflates error correlation' if dphi > 0.05 else 'no clear effect either way'})")

    # ------------------------------------ B'. does dropping suspects beat dropping at random?
    print(f"\n[ctl] === B'. cleaned vs RANDOMLY-THINNED at the same size ===")
    probs = oof_proba(Xtr, ytr, k=args.folds, seed=0, hidden=0)
    issues = cleanlab_issues(ytr, probs)
    k = int(args.drop_frac * n)
    suspects = issues[:k]

    base = float((fit_predict(Xtr, ytr, Xdv, hidden=0, seed=0) == ydv).mean())
    keep_s = np.setdiff1d(np.arange(n), suspects)
    acc_s = float((fit_predict(Xtr[keep_s], ytr[keep_s], Xdv, hidden=0, seed=0) == ydv).mean())
    print(f"[ctl] all {n} train rows                 dev acc {base:.4f}")

    rnd = []
    for r in range(args.replicates):
        g = np.random.default_rng(1000 + r)
        drop = g.choice(n, k, replace=False)
        keep = np.setdiff1d(np.arange(n), drop)
        rnd.append(float((fit_predict(Xtr[keep], ytr[keep], Xdv, hidden=0, seed=0)
                          == ydv).mean()))
    rnd = np.asarray(rnd)
    print(f"[ctl] cleanlab top {k} dropped           dev acc {acc_s:.4f}")
    print(f"[ctl] RANDOM {k} dropped ({args.replicates} reps)  dev acc "
          f"{rnd.mean():.4f} +- {rnd.std():.4f}  [min {rnd.min():.4f} max {rnd.max():.4f}]")
    gain = acc_s - rnd.mean()
    z_score = gain / max(rnd.std(), 1e-6)
    verdict = "CLEANING HELPS" if gain > 2 * rnd.std() else "WITHIN NOISE"
    print(f"[ctl] gain over random dropping: {gain:+.4f} ({z_score:+.1f} sd) -> {verdict}")
    print(f"[ctl] reference: baseline uncorrected dev acc {base:.4f}; "
          f"drop-random average {rnd.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
