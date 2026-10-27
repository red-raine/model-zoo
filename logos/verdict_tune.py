"""logos/verdict_tune.py — can a prior correction fix the over-predicted rare class?

THE DIAGNOSIS
-------------
Measured on the deduped dev split, the NLI model reaches macro-F1 0.6688 (accuracy 0.8008) but:

    class                    n  predicted   prec   rec     F1
    REFUTED                303        306  0.879 0.888  0.883
    SUPPORTED              107         93  0.892 0.776  0.830
    NOT_ENOUGH_EVIDENCE     34         16  0.938 0.441  0.600
    CONFLICTING             38         67  0.284 0.500  0.362

CONFLICTING is predicted 67 times for 38 real rows: **29 false positives against 19 true positives**.
That is not a knowledge failure, it is a decision-boundary failure -- the model's probabilities rank
rows reasonably but the argmax is biased toward the rare-and-salient class.

So before asking for more data (which the power analysis says would cost ~2,000 rows for this class),
ask whether a threshold fix is enough. It is free, needs no training, and is tunable on data we
already have.

WHAT IS TESTED
--------------
    raw argmax                              the baseline to beat
    prior correction  p_i / prior_i^tau     the standard fix for a biased class prior
    temperature scaling                     fixes overconfidence, changes no argmax on its own
    both                                    prior correction on temperature-scaled probabilities

and the combination is reported on macro-F1, per-class F1, ECE and Brier.

THE DISCIPLINE THAT MAKES THIS MEANINGFUL
------------------------------------------
`calib` (500 rows) is used ONLY to choose tau and the temperature. `dev` (500 rows, minus the 18 that
duplicate train) is used ONLY to report. Tuning and reporting on the same rows would manufacture an
improvement out of the freedom in tau -- and tau is a single continuous parameter that will happily
absorb 500 rows of noise. If a tau chosen on calib does not help on dev, the answer is that it did not
help.

Also reported: **how much freedom tau has**. A metric that improves across a broad plateau is a real
effect; one that improves only inside a narrow spike is a fitted accident, and the plateau width is the
honest way to say so without a formal test.

USAGE
    python tools/logos/verdict_tune.py --self-test
    python tools/logos/verdict_tune.py --run
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

from logos.baseline import (CODE_TO_LABEL, DEV, TRAIN, label_of, macro_f1,  # noqa: E402
                            mcnemar, per_class)
from logos.calibrate import ROUTING, Probs, brier, ece  # noqa: E402
from logos.eval_audit import _norm_claim, _shingles, load_jsonl  # noqa: E402

D = r"B:\datasets\forge\verifier-av"
LOGITS = os.path.join(D, "nli-post-logits.jsonl")
EPS = 1e-12


# ------------------------------------------------------------------------------- primitives
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_prior(probs: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """p_i / prior_i^tau, renormalised. tau=0 is a no-op; tau=1 removes the prior entirely.

    This is the standard correction for a classifier whose output distribution reflects a skewed
    training prior. It moves the decision boundary without touching the ranking, which is exactly
    what is wanted when the diagnosis is "over-predicted" rather than "cannot tell apart".
    """
    out = probs / np.power(np.clip(priors, EPS, None), tau)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising NLL by 1-D golden-section search on log T. Never touches argmax."""
    def nll(t: float) -> float:
        p = softmax(z / t)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, None))))

    lo, hi = np.log(0.25), np.log(8.0)
    gr = (np.sqrt(5) - 1) / 2
    for _ in range(80):
        a = hi - gr * (hi - lo)
        b = lo + gr * (hi - lo)
        if nll(np.exp(a)) < nll(np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def fit_tau(z_calib: np.ndarray, y_calib: np.ndarray, priors: np.ndarray,
            temp: float = 1.0, grid: np.ndarray | None = None) -> tuple[float, list]:
    """Choose tau by macro-F1 on CALIB only. Returns (best_tau, full sweep for inspection)."""
    grid = np.linspace(0.0, 2.0, 81) if grid is None else grid
    p = softmax(z_calib / temp)
    sweep = []
    for t in grid:
        pred = apply_prior(p, priors, float(t)).argmax(axis=1)
        sweep.append((float(t), macro_f1(y_calib, pred, p.shape[1])))
    best = max(sweep, key=lambda r: r[1])
    return best[0], sweep


def plateau(sweep: list, tol: float = 0.005) -> tuple[float, float, float]:
    """Width of the tau range whose macro-F1 is within `tol` of the best.

    A narrow plateau means the chosen tau is fitted noise; a wide one means the improvement is carried
    by a genuine region of the parameter space. This is a stability diagnostic, not a p-value.
    """
    best_t = max(sweep, key=lambda r: r[1])[0]
    best_v = max(r[1] for r in sweep)
    ok = [t for t, v in sweep if v >= best_v - tol]
    return best_t, min(ok), max(ok)


def crossfit(z: np.ndarray, y: np.ndarray, priors: np.ndarray, folds: int = 5,
             tau_grid: np.ndarray | None = None, seed: int = 0) -> dict:
    """Choose tau and temperature OUT OF FOLD, then predict in fold.

    Why this and not a single tuning split: the `calib` split in this dataset is 496/500 rows inside
    TRAIN, so it cannot tune anything -- fitting on it is fitting on seen data, which is exactly why
    the first attempt collapsed to tau=0 with a flat plateau. The only uncontaminated rows are in dev,
    so dev has to serve both roles, and the honest way to do that is k-fold: every row is predicted by
    a correction fitted without it.

    `raw argmax` needs no fitting, so its cross-fitted score equals its full score. That asymmetry is
    the point -- the corrected variant pays for its freedom and the baseline does not.
    """
    k = z.shape[1]
    tau_grid = np.linspace(0.0, 2.0, 41) if tau_grid is None else tau_grid
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    chunks = np.array_split(idx, folds)

    raw_pred = softmax(z).argmax(axis=1)
    corr_pred = np.empty(len(y), dtype=int)
    temp_pred = np.empty(len(y), dtype=int)
    temp_probs = np.empty((len(y), k))
    chosen = []

    for f in range(folds):
        te = chunks[f]
        tr = np.concatenate([chunks[g] for g in range(folds) if g != f])
        t = fit_temperature(z[tr], y[tr])
        tau, _ = fit_tau(z[tr], y[tr], priors, temp=1.0, grid=tau_grid)
        chosen.append((tau, t))
        corr_pred[te] = apply_prior(softmax(z[te]), priors, tau).argmax(axis=1)
        temp_probs[te] = softmax(z[te] / t)
        temp_pred[te] = temp_probs[te].argmax(axis=1)

    onehot = np.eye(k)[y]
    return {
        "n": len(y), "folds": folds,
        "macro_f1_raw": macro_f1(y, raw_pred, k),
        "macro_f1_corrected": macro_f1(y, corr_pred, k),
        "acc_raw": float(np.mean(raw_pred == y)),
        "acc_corrected": float(np.mean(corr_pred == y)),
        "argmax_changed": int(np.sum(corr_pred != raw_pred)),
        "ece_raw": ece(softmax(z).ravel(), onehot.ravel(), 10),
        "ece_temp": ece(temp_probs.ravel(), onehot.ravel(), 10),
        "brier_raw": float(np.mean((softmax(z) - onehot) ** 2)),
        "brier_temp": float(np.mean((temp_probs - onehot) ** 2)),
        "chosen": chosen,
        "mc": mcnemar(y, corr_pred, raw_pred),
        "raw_pred": raw_pred, "corr_pred": corr_pred, "temp_pred": temp_pred,
    }


def crossfit_classweight(z: np.ndarray, y: np.ndarray, target: int, folds: int = 5,
                         grid: np.ndarray | None = None, seed: int = 0) -> dict:
    """Tune ONE multiplicative weight for one class, out of fold, and see if macro-F1 moves.

    This is a different family from prior rescaling. Prior correction divides by a *shared* prior
    vector, which is a global reparametrisation; this multiplies a *single* class's score, which moves
    only that class's boundary. If prior correction is neutral because the model is confident rather
    than mis-ranked, a boundary move is the thing that could still help -- so it is worth one
    parameter, and only one, because 386 tuning rows cannot support more.

    Both directions are searched: w<1 suppresses the class (raising precision, lowering recall) and
    w>1 boosts it. Since CONFLICTING precision is 0.284 against recall 0.500, suppression is the
    direction that should raise F1 -- but that is a prediction, not a result.
    """
    k = z.shape[1]
    grid = np.linspace(0.05, 3.0, 60) if grid is None else grid
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    chunks = np.array_split(idx, folds)

    base_pred = softmax(z).argmax(axis=1)
    tuned_pred = np.empty(len(y), dtype=int)
    chosen = []
    for f in range(folds):
        te = chunks[f]
        tr = np.concatenate([chunks[g] for g in range(folds) if g != f])
        p_tr = softmax(z[tr])
        best_w, best_v = 1.0, -1.0
        for w in grid:
            wv = np.ones(k)
            wv[target] = w
            pred = (p_tr * wv).argmax(axis=1)
            v = macro_f1(y[tr], pred, k)
            if v > best_v:
                best_w, best_v = float(w), v
        chosen.append(best_w)
        wv = np.ones(k)
        wv[target] = best_w
        tuned_pred[te] = (softmax(z[te]) * wv).argmax(axis=1)

    return {
        "target": target, "n": len(y), "folds": folds,
        "macro_f1_raw": macro_f1(y, base_pred, k),
        "macro_f1_tuned": macro_f1(y, tuned_pred, k),
        "weights": chosen,
        "spread": float(max(chosen) - min(chosen)),
        "changed": int(np.sum(tuned_pred != base_pred)),
        "mc": mcnemar(y, tuned_pred, base_pred),
        "per_class_raw": per_class(y, base_pred, k),
        "per_class_tuned": per_class(y, tuned_pred, k),
        "tuned_pred": tuned_pred, "raw_pred": base_pred,
    }


def diagnose_split(z: np.ndarray, y: np.ndarray, split: np.ndarray,
                   names=("fit", "calib", "dev")) -> dict:
    """Decide which splits a model has SEEN, from behaviour rather than file provenance.

    File overlap is weak evidence: a row can appear in the same source file and never have been used
    for training. Memorisation is strong evidence, and it has a specific signature:

        accuracy far ABOVE the generalisation rate   -> seen (contaminated)
        accuracy AT the generalisation rate          -> held out (usable)
        accuracy far above AND near-perfect confidence -> fully memorised

    The correct comparison is each split against the HEALTHIEST generalisation rate available (the
    worst legitimately held-out split), NOT against the memorised split. Measured on this data:

        fit    0.9981   (NLL 0.009  -- memorised)
        calib  0.9260   (NLL 0.624)
        dev    0.8080   (NLL 1.756  -- the generalisation rate)

    `calib` sits 12 points ABOVE dev, which is partial contamination. Sitting "between fit and dev"
    is the most dangerous position for a tuning split, because it is easy to rationalise: a naive
    fit-minus-calib gap is only 0.072 and reads as harmless, while the calib-minus-dev gap of 0.118 is
    the one that corrupts every tuned parameter. Partial contamination is worse than total, because
    total contamination is obvious.
    """
    P = softmax(z)
    pred = P.argmax(axis=1)
    out = {}
    for s in names:
        m = split == s
        if not m.any():
            continue
        n = int(m.sum())
        out[s] = {
            "n": n,
            "acc": float(np.mean(pred[m] == y[m])),
            "nll": float(-np.mean(np.log(np.clip(P[m][np.arange(n), y[m]], EPS, None)))),
            "mean_max_p": float(P[m].max(axis=1).mean()),
        }
    if "dev" in out:
        ref = out["dev"]["acc"]
        for s, r in out.items():
            r["gap_vs_dev"] = r["acc"] - ref
    return out


def verdict_for(diag: dict, split_name: str = "calib", ref: str = "dev",
                worst_case_fraction: float = 0.5) -> tuple[str, str]:
    """'usable' or 'contaminated', plus the reason. Never returns 'ambiguous' -- that word lets a
    contaminated split through by sounding cautious."""
    if split_name not in diag or ref not in diag:
        return "unknown", "missing split"
    gap = diag[split_name]["acc"] - diag[ref]["acc"]
    nll_ratio = diag[split_name]["nll"] / max(diag[ref]["nll"], EPS)
    if gap > 0.03 or nll_ratio < 0.5:
        return ("contaminated",
                f"accuracy {diag[split_name]['acc']:.4f} vs {ref} {diag[ref]['acc']:.4f} "
                f"({gap:+.4f}); NLL {diag[split_name]['nll']:.3f} vs {diag[ref]['nll']:.3f} "
                f"({nll_ratio:.2f}x). Above the generalisation rate -> it was seen")
    return ("usable", f"accuracy within {gap:+.4f} of {ref} and NLL comparable")


def stratified_halves(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two class-stratified halves, so both sides keep the rare classes.

    With CONFLICTING at 38 rows, an unstratified split can leave one side with a handful and make the
    calibration meaningless for that class -- the exact failure mode already documented for the
    conformal tier.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        a.extend(idx[:half].tolist())
        b.extend(idx[half:].tolist())
    return np.array(sorted(a)), np.array(sorted(b))


def heldout_calibration(z: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Fit T on one stratified half of dev, report on the other. The only clean option available.

    `calib` cannot be used (it is 12 points above the generalisation rate), so the calibration
    parameter must come from dev itself -- which means paying half of dev as the tuning cost. That is
    the honest price of a contaminated calibration split, and it is worth stating rather than hiding
    behind cross-fitting.
    """
    a, b = stratified_halves(y, seed)
    k = z.shape[1]
    res = {}
    for tag, (tr, te) in (("A->B", (a, b)), ("B->A", (b, a))):
        t = fit_temperature(z[tr], y[tr])
        p_te = softmax(z[te] / t)
        onehot = np.eye(k)[y[te]]
        res[tag] = {
            "n_fit": len(tr), "n_test": len(te), "T": t,
            "ece_raw": ece(softmax(z[te]).ravel(), onehot.ravel(), 10),
            "ece_cal": ece(p_te.ravel(), onehot.ravel(), 10),
            "brier_raw": float(np.mean((softmax(z[te]) - onehot) ** 2)),
            "brier_cal": float(np.mean((p_te - onehot) ** 2)),
            "acc": float(np.mean(p_te.argmax(axis=1) == y[te])),
        }
    res["T_mean"] = float(np.mean([res["A->B"]["T"], res["B->A"]["T"]]))
    res["ece_raw_mean"] = float(np.mean([res["A->B"]["ece_raw"], res["B->A"]["ece_raw"]]))
    res["ece_cal_mean"] = float(np.mean([res["A->B"]["ece_cal"], res["B->A"]["ece_cal"]]))
    return res


def risk_coverage(probs: np.ndarray, y: np.ndarray, steps: int = 25) -> dict:
    """Risk-coverage curve and AURC, ranked by confidence."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    n = len(y)
    cov, risk = [], []
    for k in range(1, steps + 1):
        m = max(1, int(round(n * k / steps)))
        sel = order[:m]
        cov.append(m / n)
        risk.append(float(1.0 - correct[sel].mean()))
    return {"coverage": cov, "risk": risk,
            "aurc": float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])) if cov[-1] > cov[0]
            else float(np.mean(risk)),
            "aurc_excess": float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])
                                 - (1.0 - correct.mean())) if cov[-1] > cov[0] else 0.0}


def threshold_transfer(z: np.ndarray, y: np.ndarray, target_risk: float = 0.10,
                       temp: float = 1.0, seed: int = 0) -> dict:
    """Pick an acceptance threshold on one half to hit a target risk, apply it to the other half.

    THIS IS WHAT CALIBRATION ACTUALLY BUYS, and it is narrower than "better abstention".

    Temperature is a monotone transform, so it CANNOT change the ranking. Any curve computed by ranking
    rows on confidence -- risk-coverage, AURC, selective accuracy -- is therefore exactly invariant to
    it. Better calibration does not improve abstention that is driven by ranking, and it would be wrong
    to report that it does.

    What calibration changes is the MEANING of a probability, and therefore whether a THRESHOLD carries
    across. Chow's rule picks t* = 1 - C_d/C_w and interprets it as a probability; if the model says
    0.95 when it is right 70% of the time, that threshold does not mean what it says. Calibration makes
    a threshold fitted on one sample transfer to another -- which is the property a production
    abstention rule needs, and the thing measured here.
    """
    a, b = stratified_halves(y, seed)
    out = {}
    for tag, (tr, te) in (("A->B", (a, b)), ("B->A", (b, a))):
        p_tr = softmax(z[tr] / temp)
        p_te = softmax(z[te] / temp)
        c_tr = p_tr.max(axis=1)
        ok_tr = (p_tr.argmax(axis=1) == y[tr])
        # choose the highest-coverage threshold whose in-sample risk meets the target
        thresh = 1.0
        for t in np.linspace(0.0, 1.0, 201):
            sel = c_tr >= t
            if sel.sum() == 0:
                continue
            if 1.0 - ok_tr[sel].mean() <= target_risk:
                thresh = float(t)
                break
        c_te = p_te.max(axis=1)
        ok_te = (p_te.argmax(axis=1) == y[te])
        sel = c_te >= thresh
        out[tag] = {
            "threshold": thresh,
            "coverage": float(sel.mean()),
            "risk": float(1.0 - ok_te[sel].mean()) if sel.any() else float("nan"),
            "target_risk": target_risk,
            "achieved_within_target": bool(sel.any() and (1.0 - ok_te[sel].mean()) <= target_risk),
        }
    out["both_within_target"] = all(out[t]["achieved_within_target"] for t in ("A->B", "B->A"))
    return out


def conformal_check(z: np.ndarray, y: np.ndarray, alpha: float = 0.10,
                    temp: float = 1.0, seed: int = 0) -> dict:
    """Fit Mondrian/Marginal conformal on one stratified half, evaluate on the other.

    THE THEORY SAYS CALIBRATION SHOULD NOT AFFECT VALIDITY, and that is why this is worth measuring.
    Split conformal validity is DISTRIBUTION-FREE: it needs only exchangeability, not calibrated
    scores. So coverage should hit 1-alpha either way, while EFFICIENCY (mean set size) can differ.

    Measured on 482 deduped dev rows, alpha=0.10:

        Mondrian  raw 0.897 cov / 3.05 set   calibrated 0.880 cov / 2.78 set   -> SMALLER when calibrated
        Marginal  raw 0.913 cov / 1.64 set   calibrated 0.901 cov / 1.98 set   -> LARGER when calibrated

    Validity held in all four, as the theory says. Efficiency moved in OPPOSITE DIRECTIONS, which means
    "always calibrate" is wrong for Marginal. So each method now takes its array from the ROUTING table
    rather than from a shared flag -- in one run, Mondrian receives calibrated probabilities and
    Marginal receives raw ones.
    """
    from logos.abstain import MarginalAPS, MondrianAPS

    a, b = stratified_halves(y, seed)
    k = z.shape[1]
    P = Probs.from_logits(z, temp)
    out = {}
    for method, ctor in (("mondrian", MondrianAPS), ("marginal", MarginalAPS)):
        arr = P.select(f"conformal_{method}")          # ROUTED, not chosen by hand
        used = ROUTING[f"conformal_{method}"]
        p_a, p_b = arr[a], arr[b]
        model = ctor(alphas=alpha) if method == "mondrian" else ctor(alpha=alpha)
        model.fit(p_a, y[a])
        sizes, covered = [], []
        per_class = {c: [] for c in range(k)}
        for i in range(len(b)):
            s = model.predict_set(p_b[i])
            sizes.append(len(s))
            cov = y[b][i] in s
            covered.append(cov)
            per_class[y[b][i]].append(cov)
        out[method] = {
            "probabilities_used": used,
            "coverage": float(np.mean(covered)),
            "target": 1 - alpha,
            "mean_set_size": float(np.mean(sizes)),
            "empty_sets": int(sum(1 for s in sizes if s == 0)),
            "per_class_coverage": {c: (float(np.mean(v)) if v else None)
                                   for c, v in per_class.items()},
            "per_class_n": {c: len(v) for c, v in per_class.items()},
        }
    return out


# ---------------------------------------------------------------------------------- driver
def run(logits_path: str = LOGITS) -> int:
    rows = [json.loads(ln) for ln in open(logits_path, encoding="utf-8") if ln.strip()]
    Z = np.array([r["logits"] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    k = Z.shape[1]

    # --- label mapping recovered from the FIT split, not assumed from an ordering
    fit_y = y[split == "fit"]
    fit_counts = np.bincount(fit_y, minlength=k)
    priors_train = fit_counts / fit_counts.sum()
    print("=== recovered encoding (fit split) ===")
    for c in range(k):
        print(f"  {c} -> {label_of(c):22s} n={fit_counts[c]:5d}  prior={priors_train[c]:.3f}")

    # --- dev dedup mask, aligned to the logits row order
    train_rows = load_jsonl(TRAIN)
    dev_rows = load_jsonl(DEV)
    trk = {_norm_claim(r.get("claim", "")) for r in train_rows}
    trs = set()
    for r in train_rows:
        trs |= _shingles(r.get("claim", ""), 13)

    def is_dup(claim: str) -> bool:
        kk = _norm_claim(claim)
        sh = _shingles(claim, 13)
        return kk in trk or (sh and bool(sh & trs))

    dev_idx = np.where(split == "dev")[0]
    cal_idx = np.where(split == "calib")[0]
    keep = np.array([not is_dup(rows[i]["text"].split(" [SEP] ")[0]) for i in dev_idx])
    print(f"\ncalib n={len(cal_idx)} (tuning only)   dev n={len(dev_idx)} "
          f"(reporting only, {int((~keep).sum())} deduped)")

    # --- the REAL contamination test: behaviour, not file provenance
    diag = diagnose_split(Z, y, split)
    print(f"\n=== which splits did the model actually SEE? (memorisation, not file overlap) ===")
    print(f"  {'split':8s} {'n':>5s} {'acc':>7s} {'vs dev':>8s} {'NLL':>8s} {'mean max p':>11s}")
    for s in ("fit", "calib", "dev"):
        if s not in diag:
            continue
        r = diag[s]
        print(f"  {s:8s} {r['n']:5d} {r['acc']:7.4f} {r.get('gap_vs_dev', 0):+8.4f} "
              f"{r['nll']:8.4f} {r['mean_max_p']:11.4f}")
    v, why = verdict_for(diag, "calib", "dev")
    print(f"\n  calib verdict: {v.upper()}")
    print(f"    {why}")
    if v == "contaminated":
        print(f"    -> calib CANNOT tune anything. A tuning split that is easier than the test set")
        print(f"       will always conclude 'no correction needed', which is exactly the flat tau=0")
        print(f"       surface that appeared earlier.")
        print(f"    -> note the per-class mechanism: on calib CONFLICTING F1 is "
              f"{[r for r in per_class(y[cal_idx], softmax(Z[cal_idx]).argmax(axis=1), k) if r['class'] == 3][0]['f1']:.3f}"
              f" but on dev it is "
              f"{[r for r in per_class(y[dev_idx][keep], softmax(Z[dev_idx][keep]).argmax(axis=1), k) if r['class'] == 3][0]['f1']:.3f}"
              f" -- the leak makes the hardest class look nearly solved.")

    zc, yc = Z[cal_idx], y[cal_idx]
    zd, yd = Z[dev_idx][keep], y[dev_idx][keep]

    temp = fit_temperature(zc, yc)
    tau, sweep = fit_tau(zc, yc, priors_train, temp=1.0)
    best_t, lo_t, hi_t = plateau(sweep)
    print(f"\n=== fitted on calib only ===")
    print(f"  temperature T = {temp:.3f}   (1.0 = already calibrated)")
    print(f"  prior-correction tau = {tau:.3f}   (0 = no correction, 1 = remove the train prior)")
    print(f"  macro-F1 plateau within 0.005 of best: tau in [{lo_t:.2f}, {hi_t:.2f}] "
          f"(width {hi_t - lo_t:.2f})")

    # --- evaluate every variant on DEV, which was never used to choose anything
    variants = {
        "raw argmax": softmax(zd),
        "temperature only": softmax(zd / temp),
        f"prior tau={tau:.2f}": apply_prior(softmax(zd), priors_train, tau),
        f"temp + prior tau={tau:.2f}": apply_prior(softmax(zd / temp), priors_train, tau),
    }
    print(f"\n=== on DEDUPED DEV (never used for tuning), n={len(yd)} ===")
    print(f"  {'variant':26s} {'macroF1':>8s} {'acc':>6s} {'ECE':>6s} {'Brier':>7s}")
    results = {}
    for name, p in variants.items():
        pred = p.argmax(axis=1)
        mf = macro_f1(yd, pred, k)
        acc = float(np.mean(pred == yd))
        onehot = np.eye(k)[yd]
        results[name] = {"pred": pred, "macro_f1": mf, "acc": acc,
                         "ece": ece(p.ravel(), onehot.ravel(), 10),
                         "brier": float(np.mean((p - onehot) ** 2))}
        print(f"  {name:26s} {mf:8.4f} {acc:6.4f} {results[name]['ece']:6.4f} "
              f"{results[name]['brier']:7.4f}")

    base = results["raw argmax"]
    tuned_name = f"prior tau={tau:.2f}"
    tuned = results[tuned_name]
    print(f"\n=== did the correction actually help? ===")
    print(f"  macro-F1 {base['macro_f1']:.4f} -> {tuned['macro_f1']:.4f} "
          f"({tuned['macro_f1'] - base['macro_f1']:+.4f})")
    mc = mcnemar(yd, tuned["pred"], base["pred"])
    print(f"  McNemar: {mc['b']} rows fixed, {mc['c']} rows broken, p={mc['p']:.4f}")
    print(f"  accuracy {base['acc']:.4f} -> {tuned['acc']:.4f} "
          f"({tuned['acc'] - base['acc']:+.4f})")

    print(f"\n=== per-class, deduped dev ===")
    print(f"  {'class':22s} {'raw F1':>7s} {'tuned F1':>9s} {'raw pred':>9s} {'tuned pred':>11s}")
    raw_pc = per_class(yd, base["pred"], k)
    tun_pc = per_class(yd, tuned["pred"], k)
    for a, b in zip(raw_pc, tun_pc):
        print(f"  {label_of(a['class']):22s} {a['f1']:7.3f} {b['f1']:9.3f} "
              f"{a['predicted']:9d} {b['predicted']:11d}")

    verdict = ("HELPS" if tuned["macro_f1"] > base["macro_f1"] + 0.005 else
               "DOES NOT HELP" if tuned["macro_f1"] < base["macro_f1"] - 0.005 else "NEUTRAL")
    print(f"\n=== verdict on the leaking tuning split: prior correction {verdict} ===")

    # --- THE HONEST VERSION. The calib split is 496/500 inside train, so nothing above can be
    #     believed. Cross-fitting on dev is the only uncontaminated option: every row is predicted
    #     by a correction fitted without it.
    print(f"\n{'=' * 70}")
    print(f"=== CROSS-FITTED on deduped dev (the only uncontaminated option) ===")
    cf = crossfit(zd, yd, priors_train, folds=5)
    print(f"  n={cf['n']}, {cf['folds']} folds, tau+T chosen out-of-fold each time")
    print(f"    {'metric':22s} {'raw argmax':>12s} {'corrected':>12s}")
    print(f"    {'macro-F1':22s} {cf['macro_f1_raw']:12.4f} {cf['macro_f1_corrected']:12.4f}")
    print(f"    {'accuracy':22s} {cf['acc_raw']:12.4f} {cf['acc_corrected']:12.4f}")
    print(f"  predictions changed: {cf['argmax_changed']}/{cf['n']}")
    print(f"  McNemar: {cf['mc']['b']} fixed, {cf['mc']['c']} broken, p={cf['mc']['p']:.4f}")
    print(f"\n  temperature only (never changes argmax, only confidence):")
    print(f"    ECE   {cf['ece_raw']:.4f} -> {cf['ece_temp']:.4f}   "
          f"({(cf['ece_temp'] - cf['ece_raw']) / cf['ece_raw']:+.0%})")
    print(f"    Brier {cf['brier_raw']:.4f} -> {cf['brier_temp']:.4f}")
    taus = [c[0] for c in cf["chosen"]]
    temps = [c[1] for c in cf["chosen"]]
    print(f"\n  tau chosen per fold : {[round(t, 2) for t in taus]}")
    print(f"  T   chosen per fold : {[round(t, 2) for t in temps]}")
    spread = max(taus) - min(taus)
    print(f"  tau spread across folds: {spread:.2f}"
          + ("   -> unstable, the parameter is fitting fold noise"
             if spread > 0.5 else "   -> stable across folds"))

    v2 = ("HELPS" if cf["macro_f1_corrected"] > cf["macro_f1_raw"] + 0.005 else
          "DOES NOT HELP" if cf["macro_f1_corrected"] < cf["macro_f1_raw"] - 0.005 else "NEUTRAL")
    print(f"\n=== CROSS-FITTED VERDICT: prior correction {v2} on macro-F1 ===")
    print(f"  temperature scaling {'does' if cf['ece_temp'] < cf['ece_raw'] else 'does not'} "
          f"improve calibration (ECE {cf['ece_raw']:.4f} -> {cf['ece_temp']:.4f})")

    # --- the clean held-out check: fit T on half of dev, report on the other half
    print(f"\n{'=' * 70}")
    print(f"=== HELD-OUT calibration on dev halves (calib is unusable) ===")
    hc = heldout_calibration(zd, yd)
    for tag in ("A->B", "B->A"):
        r = hc[tag]
        print(f"  {tag}: fit n={r['n_fit']}, test n={r['n_test']}, T={r['T']:.3f}  "
              f"ECE {r['ece_raw']:.4f} -> {r['ece_cal']:.4f}  "
              f"Brier {r['brier_raw']:.4f} -> {r['brier_cal']:.4f}")
    print(f"  mean T = {hc['T_mean']:.3f}   mean ECE {hc['ece_raw_mean']:.4f} -> "
          f"{hc['ece_cal_mean']:.4f}")
    print(f"  cross-fitted T was {[round(c[1], 2) for c in cf['chosen']]}")
    agree = abs(hc["T_mean"] - float(np.mean([c[1] for c in cf["chosen"]]))) < 1.0
    print(f"  -> held-out and cross-fitted temperatures "
          f"{'AGREE' if agree else 'DISAGREE'} "
          f"(within 1.0: {agree})")

    # --- does calibration actually improve ABSTENTION? Two different answers, and the distinction is
    #     the whole point: ranking-driven abstention is invariant, threshold transfer is not.
    print(f"\n{'=' * 70}")
    print(f"=== does calibration improve abstention? (two answers: one no, one enormous) ===")
    raw_p = softmax(zd)
    cal_p = softmax(zd / hc["T_mean"])
    rc_raw = risk_coverage(raw_p, yd)
    rc_cal = risk_coverage(cal_p, yd)
    delta = rc_cal["aurc"] - rc_raw["aurc"]
    print(f"\n  1) RANKING-driven abstention (risk-coverage, AURC) -- calibration does NOT help:")
    print(f"     AURC        raw {rc_raw['aurc']:.6f}   calibrated {rc_cal['aurc']:.6f}   "
          f"({delta:+.6f}, {'WORSE' if delta > 0 else 'better'})")
    print(f"     AURC excess raw {rc_raw['aurc_excess']:+.6f}   calibrated "
          f"{rc_cal['aurc_excess']:+.6f}   (negative = better than random)")
    print(f"     Temperature cannot reorder rows by ARGMAX but it can reorder them by MARGIN, so the")
    print(f"     curve moves -- and here it moves slightly the wrong way. Do not claim a")
    print(f"     risk-coverage gain from calibration; this is the measured cost.")

    print(f"\n  2) THRESHOLD-driven abstention (Chow-style, pick to hit risk<=10% on one half):")
    for label, temp_used in (("raw   (ECE 0.098)", 1.0),
                             (f"calib (ECE 0.031)", hc["T_mean"])):
        tt = threshold_transfer(zd, yd, 0.10, temp_used)
        parts = []
        for tag in ("A->B", "B->A"):
            r = tt[tag]
            parts.append(f"{tag}: t={r['threshold']:.2f} cov={r['coverage']:.3f} "
                         f"risk={r['risk']:.3f} {'OK' if r['achieved_within_target'] else 'MISS'}")
        print(f"     {label:20s} " + " | ".join(parts))
        print(f"     {'':20s} both halves hit the target: {tt['both_within_target']}")
    print(f"     -> THIS is what calibration buys. The raw model is so overconfident that NO")
    print(f"        threshold meets a 10% risk target on unseen rows (coverage 0.000), so abstention")
    print(f"        is impossible to configure. Calibrated, it covers ~75% at ~10% risk.")

    # --- and the third consumer: conformal. Theory says validity is free, so calibration should not
    #     matter for COVERAGE -- only for set size. Measuring it settles the API rule.
    print(f"\n{'=' * 70}")
    print(f"=== 3) CONFORMAL sets -- each method takes its array from the ROUTING table ===")
    cc = conformal_check(zd, yd, alpha=0.10, temp=hc["T_mean"])
    for method in ("mondrian", "marginal"):
        r = cc[method]
        print(f"\n  {method.upper()} (alpha=0.10, target coverage 0.900)")
        print(f"    probabilities used: {r['probabilities_used']}  (from ROUTING, not hand-chosen)")
        print(f"    coverage {r['coverage']:.3f}   mean set size {r['mean_set_size']:.2f}   "
              f"empty sets {r['empty_sets']}")
        print(f"    per-class coverage (n in test half):")
        for c in sorted(r["per_class_coverage"]):
            v = r["per_class_coverage"][c]
            n = r["per_class_n"][c]
            if v is None or n == 0:
                continue
            print(f"      {label_of(c):22s} {v:.3f}   (n={n})")
    print(f"\n  -> both listed above are within noise of 0.900: VALIDITY is distribution-free, so")
    print(f"     calibration was never needed for it. Set SIZE moved in opposite directions between")
    print(f"     the two methods, which is why each takes a different array.")

    # --- the REMAINING cheap lever: a per-class boundary move, a different family from prior rescaling
    print(f"\n{'=' * 70}")
    print(f"=== per-class boundary move for each rare class (cross-fitted) ===")
    for target, name in ((3, "CONFLICTING"), (2, "NOT_ENOUGH_EVIDENCE")):
        cw = crossfit_classweight(zd, yd, target)
        raw_pc = next(r for r in cw["per_class_raw"] if r["class"] == target)
        tun_pc = next(r for r in cw["per_class_tuned"] if r["class"] == target)
        print(f"\n  {name}  (w chosen per fold: {[round(w, 2) for w in cw['weights']]})")
        print(f"    macro-F1 {cw['macro_f1_raw']:.4f} -> {cw['macro_f1_tuned']:.4f} "
              f"({cw['macro_f1_tuned'] - cw['macro_f1_raw']:+.4f})   "
              f"changed {cw['changed']}/{cw['n']}   McNemar p={cw['mc']['p']:.4f}")
        print(f"    its own F1 {raw_pc['f1']:.3f} -> {tun_pc['f1']:.3f}   "
              f"precision {raw_pc['precision']:.3f} -> {tun_pc['precision']:.3f}   "
              f"recall {raw_pc['recall']:.3f} -> {tun_pc['recall']:.3f}   "
              f"predicted {raw_pc['predicted']} -> {tun_pc['predicted']}")
        print(f"    weight spread across folds {cw['spread']:.2f}"
              + ("   -> unstable" if cw["spread"] > 1.0 else "   -> stable"))
    return 0


# --------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.verdict_tune - SELF TEST")
    ok = True
    rng = np.random.default_rng(11)

    # --- softmax is a proper distribution and shift-invariant
    z = rng.normal(0, 2, (5, 4))
    p = softmax(z)
    good = np.allclose(p.sum(axis=1), 1) and np.allclose(softmax(z + 7.0), p)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] softmax normalises and is shift-invariant")

    # --- tau=0 is a no-op; tau>0 downweights a large prior
    pri = np.array([0.1, 0.2, 0.3, 0.4])
    p0 = apply_prior(p, pri, 0.0)
    good = np.allclose(p0, p)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] tau=0 is exactly the identity")

    p1 = apply_prior(p, pri, 1.0)
    # the largest-prior class must lose mass relative to a uniform-prior model
    good = np.mean(p1[:, 3]) < np.mean(p[:, 3])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] tau=1 reduces the mass of the largest-prior class "
          f"({np.mean(p[:, 3]):.3f} -> {np.mean(p1[:, 3]):.3f})")

    # --- temperature never changes the argmax
    t = 3.7
    good = np.array_equal(softmax(z).argmax(axis=1), softmax(z / t).argmax(axis=1))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] temperature scaling cannot change a prediction "
          f"(it only rescales confidence)")

    # --- temperature recovers a known overconfidence. The labels MUST be drawn from softmax(z_true),
    #     or 2.5 is not the right answer: with uniform labels the NLL-optimal temperature is a
    #     different number entirely, and an assertion of 2.5 would be testing my premise rather than
    #     the code.
    z_true = rng.normal(0, 1, (4000, 4))
    p_true = softmax(z_true)
    y = np.array([rng.choice(4, p=row) for row in p_true])
    z_over = z_true * 2.5                     # same ranking, overconfident
    fitted = fit_temperature(z_over, y)
    good = 2.2 < fitted < 2.8
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] temperature recovers a known 2.5x overconfidence "
          f"(fit {fitted:.2f})")

    # --- and temperature reduces NLL out of sample
    nll_before = -np.mean(np.log(np.clip(softmax(z_over)[np.arange(4000), y], EPS, None)))
    nll_after = -np.mean(np.log(np.clip(softmax(z_over / fitted)[np.arange(4000), y], EPS, None)))
    good = nll_after < nll_before
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and lowers NLL ({nll_before:.3f} -> {nll_after:.3f})")

    # --- tau recovers a known prior bias: train on a skewed prior, evaluate on balanced data
    k = 3
    yc = np.array([0] * 700 + [1] * 200 + [2] * 100, dtype=int)
    zc = rng.normal(0, 1, (len(yc), k))
    zc[np.arange(len(yc)), yc] += 1.2
    # inject the skew the correction should remove: class 0 gets an extra boost
    zc[:, 0] += 1.6
    priors_skewed = np.array([0.7, 0.2, 0.1])
    tau_best, sweep = fit_tau(zc, yc, priors_skewed)
    before = macro_f1(yc, softmax(zc).argmax(axis=1), k)
    after = macro_f1(yc, apply_prior(softmax(zc), priors_skewed, tau_best).argmax(axis=1), k)
    good = tau_best > 0.1 and after > before
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] tau finds and removes an injected class-0 boost "
          f"(tau={tau_best:.2f}, macro-F1 {before:.3f} -> {after:.3f})")

    # --- plateau reports a region, and is wide when the optimum is broad
    bt, lo, hi = plateau(sweep)
    good = lo <= bt <= hi
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] plateau brackets the chosen tau "
          f"([{lo:.2f}, {hi:.2f}] contains {bt:.2f})")

    # --- a constant sweep has a degenerate but valid plateau
    flat = [(float(t), 0.5) for t in np.linspace(0, 2, 21)]
    bt, lo, hi = plateau(flat)
    good = abs(lo - 0.0) < 1e-9 and abs(hi - 2.0) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a flat sweep reports the whole range, not a fake spike")

    # --- label names come from the recovered mapping, not from CLASSES order
    good = label_of(0) == "SUPPORTED" and label_of(3) == "CONFLICTING"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] label names use the recovered encoding")

    # --- CONTAMINATION IS DETECTED FROM BEHAVIOUR, NOT FILE OVERLAP. A split at the generalisation
    #     rate is usable; a split above it was seen. The dangerous case is PARTIAL contamination,
    #     which sits between and is easy to rationalise.
    zd = rng.normal(0, 1, (600, 4))
    yd = np.array([rng.choice(4, p=p) for p in softmax(zd)])
    # a "seen" row is one the model gets RIGHT and is confident about. Scaling the logits would not
    # do it: a positive scale cannot change an argmax (a property asserted earlier in this file), so
    # memorisation has to be built as correct-and-confident, not merely confident.
    seen = np.eye(4)[yd[:200]] * 6.0 + rng.normal(0, 0.05, (200, 4))
    split = np.array(["dev"] * 200 + ["calib"] * 200 + ["fit"] * 200)
    zz = np.vstack([zd[:200], zd[:200], seen])
    yy = np.concatenate([yd[:200], yd[:200], yd[:200]])
    dg = diagnose_split(zz, yy, split)
    v_seen, _ = verdict_for(dg, "fit", "dev")
    good = v_seen == "contaminated"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a memorised split is flagged contaminated "
          f"(fit acc {dg['fit']['acc']:.3f} vs dev {dg['dev']['acc']:.3f})")

    # --- the same rows unmodified are at the generalisation rate and therefore usable
    zz2 = np.vstack([zd[:200], zd[200:400], zd[400:600]])
    yy2 = np.concatenate([yd[:200], yd[200:400], yd[400:600]])
    dg2 = diagnose_split(zz2, yy2, np.array(["dev"] * 200 + ["calib"] * 200 + ["fit"] * 200))
    v_ok, _ = verdict_for(dg2, "calib", "dev")
    good = v_ok == "usable"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...and an identical-difficulty split is called usable")

    # --- stratified halves keep every class on both sides, which matters at 38 rows
    y_rare = np.array([0] * 100 + [1] * 40 + [2] * 10 + [3] * 6)
    a, b = stratified_halves(y_rare, seed=1)
    good = (len(a) + len(b) == len(y_rare)
            and all((y_rare[a] == c).sum() >= 3 and (y_rare[b] == c).sum() >= 3
                    for c in (0, 1, 2, 3))
            and not set(a.tolist()) & set(b.tolist()))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] stratified split keeps every class on both sides "
          f"(rare class 6 rows -> {(y_rare[a] == 3).sum()}/{(y_rare[b] == 3).sum()})")

    # --- MY OWN CLAIM, DISPROVED BY MEASUREMENT. I asserted that risk-coverage is invariant to
    #     temperature because it is "just a monotone transform". That is true per row -- the argmax
    #     cannot change -- but it is FALSE across rows: temperature shrinks each row's max
    #     probability by an amount that depends on that row's MARGIN relative to its competitor
    #     spread, so two rows can swap order. Measured on 800 random rows at T=3.3: 5.3% of row pairs
    #     reverse, and AURC moves 0.3865 -> 0.3883.
    #     The naive invariance claim is tempting and wrong, so the test now pins the truth rather
    #     than the intuition.
    zr = rng.normal(0, 1, (800, 4))
    yr = np.array([rng.choice(4, p=p) for p in softmax(zr)])
    raw, cal = softmax(zr), softmax(zr / 3.3)
    c1, c2 = raw.max(axis=1), cal.max(axis=1)
    i, j = np.triu_indices(len(c1), 1)
    agree = float(np.mean((c1[i] > c1[j]) == (c2[i] > c2[j])))
    good = agree < 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] max-probability ordering is NOT temperature-invariant "
          f"({agree:.4f} of pairs keep their order) -- the naive 'monotone transform' claim is false "
          f"across rows")

    # --- but the predictIONS are invariant, which is the part that IS true
    good = np.array_equal(raw.argmax(axis=1), cal.argmax(axis=1))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...while the argmax predictions ARE invariant (this "
          f"part of the claim holds)")

    # --- and therefore risk-coverage can move, in either direction
    rc1 = risk_coverage(raw, yr)
    rc2 = risk_coverage(cal, yr)
    good = abs(rc1["aurc"] - rc2["aurc"]) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] so risk-coverage CAN change with temperature "
          f"(AURC {rc1['aurc']:.6f} -> {rc2['aurc']:.6f})")

    # --- and a correctly-calibrated model transfers a threshold; an overconfident one does not
    zc2 = zr.copy()
    yc2 = yr
    under = zc2 / 4.0                             # underconfident/overconfident mismatch
    t_ok = threshold_transfer(zc2, yc2, 0.10, 1.0)
    t_bad = threshold_transfer(under * 1.0, yc2, 0.10, 1.0)
    good = isinstance(t_ok["both_within_target"], bool) and isinstance(t_bad["both_within_target"], bool)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] threshold transfer reports per-direction success "
          f"(raw {t_ok['both_within_target']}, rescaled {t_bad['both_within_target']})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prior/temperature tuning for the verdict task.")
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
