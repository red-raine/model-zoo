"""logos/calibrate.py — the calibration and proper-scoring harness for decision models.

WHY THIS EXISTS
---------------
A decision model belongs in the closed-question slots: the `usable` validator ("does this answer
address the claim?"), and verdict classification. Those are yes/no questions with a probability
attached, which is exactly what a System-1 decision model is for -- and exactly what a chat model is
bad at.

But "calibrated" is a claim, and `laya` markets itself as *trained with strictly proper scoring
rules*. That is a claim about a model we have not run. The purpose of this file is to make such
claims checkable on OUR data:

    * does the model's 0.7 mean 70%?
    * does calibrating it beat leaving it alone, OUT OF SAMPLE?
    * is a calibrated model actually better than the hand rule it would replace?

The third question is the one that decides whether a decision model earns its slot, and a harness
that cannot ask it is decoration.

WHAT IS MEASURED
----------------
    Brier score            mean squared error of the probability. Strictly proper: minimised by the
                           truth, so any model that "improves" it has genuinely improved.
    Murphy decomposition   Brier = reliability - resolution + uncertainty. A model can be badly
                           calibrated (high reliability term) and still useful (high resolution); a
                           single number hides which is which.
    Log loss               harsher on confident mistakes.
    ECE                    expected calibration error over probability bins -- what people mean by
                           "looks calibrated", and the weakest of the three, so it never stands alone.

WHY OUT OF SAMPLE OR IT DID NOT HAPPEN
--------------------------------------
A calibrator has parameters, and fitting them on the same data used to report a gain is the exact
self-deception this project keeps finding. Every comparison here is k-fold, and the in-sample number
is reported next to the out-of-sample one SO THAT OVERFITTING IS VISIBLE. A test asserts that a
calibrator fitted to noise shows a large in-sample gain and a ~0 out-of-sample gain, and that the
harness says so.

USAGE
    python tools/logos/calibrate.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------------------- metrics
def brier(probs, labels) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs, labels) -> float:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_decomposition(probs, labels, bins: int = 10) -> dict:
    """Brier = reliability - resolution + uncertainty + within-bin variance (Murphy).

    reliability  how far each bin's mean prediction is from its observed frequency  (want 0)
    resolution   how far each bin's observed frequency is from the base rate           (want HIGH)
    uncertainty  base-rate variance -- a property of the DATA, not the model           (fixed)
    bin_variance forecast spread INSIDE a bin
    covariance   the term everyone forgets. Expanding E[(p-y)^2] around the bin mean gives a
                 within-bin cross term, so Murphy's three-term identity is exact only for DISCRETE
                 forecasts (constant per bin). With a continuous score it is off by this much --
                 measured at 0.0014 here, small enough to look fine and wrong enough to matter.
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0,
                "bin_variance": 0.0, "covariance": 0.0, "brier": 0.0}
    base = float(np.mean(y))
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel = res = binvar = cov = 0.0
    for b in range(bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        mp = float(p[m].mean())
        my = float(y[m].mean())
        rel += nb / n * (mp - my) ** 2
        res += nb / n * (my - base) ** 2
        binvar += float(np.sum((p[m] - mp) ** 2)) / n
        cov += 2.0 * float(np.sum((p[m] - mp) * (mp - y[m]))) / n
    unc = base * (1 - base)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "bin_variance": binvar, "covariance": cov,
            "brier": rel - res + unc + binvar + cov}


def ece(probs, labels, bins: int = 10) -> float:
    """Expected calibration error: weighted |mean prediction - observed frequency| per bin."""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        nb = int(m.sum())
        if nb:
            total += nb / n * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def reliability_table(probs, labels, bins: int = 10) -> list[dict]:
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        nb = int(m.sum())
        if not nb:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": nb,
                     "pred": round(float(p[m].mean()), 3),
                     "obs": round(float(y[m].mean()), 3),
                     "gap": round(float(p[m].mean() - y[m].mean()), 3)})
    return rows


# ---------------------------------------------------------------------------- calibrators
class Identity:
    """No calibration. This is the baseline every calibrator must beat."""

    name = "identity"

    def fit(self, probs, labels) -> "Identity":
        return self

    def __call__(self, probs):
        return np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    """sigmoid(a * logit(p) + b), fitted by IRLS. One parameter pair, hard to overfit."""

    name = "platt"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, probs, labels) -> "Platt":
        X = np.column_stack([_logit(probs), np.ones(len(probs))])
        y = np.asarray(labels, dtype=float)
        w = np.zeros(2)
        for _ in range(100):                     # Newton / IRLS
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = X.T @ (p - y)
            W = np.clip(p * (1 - p), 1e-9, None)
            H = X.T @ (X * W[:, None])
            try:
                step = np.linalg.solve(H + 1e-6 * np.eye(2), g)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def __call__(self, probs):
        z = self.a * _logit(probs) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BetaCalibration:
    """logit(q) = a*log(p) - b*log(1-p) + c. Three params; handles asymmetric distortions.

    Recommended over Platt when a model is overconfident in one direction only, which is the common
    case for an over-trained classifier.
    """

    name = "beta"

    def __init__(self) -> None:
        self.coef = np.array([1.0, 1.0, 0.0])

    def fit(self, probs, labels) -> "BetaCalibration":
        p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p), np.ones(len(p))])
        y = np.asarray(labels, dtype=float)
        w = np.array([1.0, 1.0, 0.0])
        for _ in range(100):
            z = X @ w
            pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = X.T @ (pr - y)
            W = np.clip(pr * (1 - pr), 1e-9, None)
            H = X.T @ (X * W[:, None])
            try:
                step = np.linalg.solve(H + 1e-6 * np.eye(3), g)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.coef = w
        return self

    def __call__(self, probs):
        p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
        z = self.coef[0] * np.log(p) - self.coef[1] * np.log(1 - p) + self.coef[2]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Isotonic:
    """Pool-adjacent-violators. Most flexible, most able to overfit -- included to be measured, not
    trusted by default."""

    name = "isotonic"

    def __init__(self) -> None:
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def fit(self, probs, labels) -> "Isotonic":
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        order = np.argsort(p)
        xs, ys = p[order], y[order]
        # PAV
        vals = ys.astype(float).tolist()
        wts = [1.0] * len(ys)
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1]:
                tw = wts[i] + wts[i + 1]
                nv = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tw
                vals[i:i + 2] = [nv]
                wts[i:i + 2] = [tw]
                ys = np.delete(ys, i + 1)
                xs = np.delete(xs, i + 1)
                i = max(0, i - 1)
            else:
                i += 1
        self.x, self.y = xs, np.asarray(vals)
        return self

    def __call__(self, probs):
        p = np.asarray(probs, dtype=float)
        if len(self.x) == 0:
            return np.clip(p, EPS, 1 - EPS)
        out = np.interp(p, self.x, self.y)
        return np.clip(out, EPS, 1 - EPS)


CALIBRATORS = {"identity": Identity, "platt": Platt, "beta": BetaCalibration,
               "isotonic": Isotonic}


# ------------------------------------------------------------------------- out-of-sample
def cross_validate(probs, labels, calibrator: str = "platt", folds: int = 5, seed: int = 0,
                   bins: int = 10) -> dict:
    """Fit the calibrator on k-1 folds, score the held-out fold. Compare against identity.

    Returns in-sample and out-of-sample Brier, because reporting only the first is how a calibrator
    fitted to noise looks like an improvement.
    """
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n < folds * 2:
        return {"ok": False, "reason": f"n={n} too small for {folds} folds", "n": n}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, folds)

    oos_pred = np.empty(n)
    oos_mask = np.zeros(n, dtype=bool)
    ins_brier, ins_ident = [], []

    for f in range(folds):
        test = chunks[f]
        train = np.concatenate([chunks[g] for g in range(folds) if g != f])
        cal = CALIBRATORS[calibrator]().fit(p[train], y[train])
        oos_pred[test] = cal(p[test])
        oos_mask[test] = True
        ins_brier.append(brier(cal(p[train]), y[train]))
        ins_ident.append(brier(p[train], y[train]))

    oos_b = brier(oos_pred[oos_mask], y[oos_mask])
    base_b = brier(p, y)
    return {
        "ok": True, "n": n, "folds": folds, "calibrator": calibrator,
        "brier_identity": base_b,
        "brier_oos": oos_b,
        "delta_oos": oos_b - base_b,                 # negative = calibration helped
        "brier_in_sample": float(np.mean(ins_brier)),
        "brier_in_sample_identity": float(np.mean(ins_ident)),
        "delta_in_sample": float(np.mean(ins_brier) - np.mean(ins_ident)),
        "ece_identity": ece(p, y, bins),
        "ece_oos": ece(oos_pred[oos_mask], y[oos_mask], bins),
        "logloss_identity": log_loss(p, y),
        "logloss_oos": log_loss(oos_pred[oos_mask], y[oos_mask]),
        "decomposition_identity": brier_decomposition(p, y, bins),
        "decomposition_oos": brier_decomposition(oos_pred[oos_mask], y[oos_mask], bins),
    }


def compare_to_rule(rule_probs, model_probs, labels, folds: int = 5, seed: int = 0) -> dict:
    """Does the model beat the hand rule it would replace? Out of sample, on the same items.

    A model only earns the slot if it wins here. Reporting an improvement over "do nothing" is worth
    almost nothing when a hand rule already exists.
    """
    rp = np.clip(np.asarray(rule_probs, dtype=float), EPS, 1 - EPS)
    mp = np.clip(np.asarray(model_probs, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=float)
    cvr = cross_validate(rp, y, "platt", folds, seed)
    cvm = cross_validate(mp, y, "platt", folds, seed)
    if not (cvr.get("ok") and cvm.get("ok")):
        return {"ok": False, "reason": "sample too small", "n": len(y)}
    return {
        "ok": True, "n": len(y),
        "rule_brier_oos": cvr["brier_oos"], "model_brier_oos": cvm["brier_oos"],
        "delta": cvm["brier_oos"] - cvr["brier_oos"],       # negative = model wins
        "rule_ece_oos": cvr["ece_oos"], "model_ece_oos": cvm["ece_oos"],
        "winner": "model" if cvm["brier_oos"] < cvr["brier_oos"] else "rule",
    }


# ------------------------------------------------------------------- routing by consumer
# MEASURED, NOT ASSUMED. Each consumer needs a different array, and the directions were measured on
# 482 deduped AVeriTeC dev rows (see knowledge/conflicting-is-not-a-threshold-problem...):
#
#   threshold / cost / risk-limiting   calibrated   raw cannot meet a 10% risk target at all
#                                                   (coverage 0.000 vs 0.72-0.78 calibrated)
#   rank-based selection (AURC)        raw          calibrating COSTS 0.005 AURC
#   conformal, Mondrian                calibrated   smaller sets: 3.05 -> 2.78
#   conformal, Marginal                raw          smaller sets: 1.64 vs 1.98 calibrated
#   class posterior for fusion         calibrated   consumed AS a probability in log-linear pooling
#
# "Always calibrate" is wrong for two of five. "Never calibrate" is wrong for three. The point is
# that the choice cannot be a global switch, so it is a table with a guard.
#
# A WARNING ABOUT THE MARGINAL ROW, because a routing table invites blind compliance. Marginal
# conformal makes a MARGINAL guarantee by construction, and it under-covers the rare classes no matter
# which array it is given: measured per-class coverage 0.737 for CONFLICTING and 0.706 for
# NOT_ENOUGH_EVIDENCE against a 0.90 target, with raw probabilities and the smallest sets of any
# configuration. The row says "raw" because that minimises set size GIVEN that you are using Marginal;
# it does not say Marginal is appropriate. If you need per-class coverage, use Mondrian, and Mondrian
# takes calibrated.
ROUTING: dict[str, str] = {
    "threshold": "calibrated",
    "cost": "calibrated",
    "risk_limiting": "calibrated",
    "ranking": "raw",
    "conformal_mondrian": "calibrated",
    "conformal_marginal": "raw",
    "fusion": "calibrated",
}


@dataclass
class Probs:
    """Both probability arrays, so a consumer must state which it needs.

    A bare ndarray cannot say whether it is calibrated, and that ambiguity is the actual defect: two
    modules in this project consume probabilities directly and neither can be audited for which one it
    holds. Carrying both and routing explicitly makes the question answerable.
    """

    raw: np.ndarray
    calibrated: np.ndarray
    temperature: float = 1.0
    ece_raw: float | None = None
    ece_calibrated: float | None = None

    @classmethod
    def from_logits(cls, logits: np.ndarray, temperature: float,
                    labels: np.ndarray | None = None, bins: int = 10) -> "Probs":
        raw = _softmax_np(logits)
        cal = _softmax_np(logits / temperature)
        ece_r = ece_c = None
        if labels is not None:
            onehot = np.eye(raw.shape[1])[labels]
            ece_r = ece(raw.ravel(), onehot.ravel(), bins)
            ece_c = ece(cal.ravel(), onehot.ravel(), bins)
        return cls(raw=raw, calibrated=cal, temperature=temperature,
                   ece_raw=ece_r, ece_calibrated=ece_c)

    def select(self, consumer: str) -> np.ndarray:
        """Return the array this consumer needs, or raise. Unknown consumers FAIL CLOSED.

        Failing open here would put an unaudited array into a threshold rule, which is the exact
        failure this class exists to prevent.
        """
        if consumer not in ROUTING:
            raise KeyError(f"unknown probability consumer {consumer!r}; known: "
                           f"{sorted(ROUTING)}. Refusing to guess which array to use.")
        return self.calibrated if ROUTING[consumer] == "calibrated" else self.raw

    def describe(self) -> dict:
        return {"temperature": self.temperature, "ece_raw": self.ece_raw,
                "ece_calibrated": self.ece_calibrated,
                "routing": {c: ROUTING[c] for c in ROUTING}}


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.calibrate - SELF TEST")
    ok = True
    rng = np.random.default_rng(7)

    # --- 1. Brier sanity: perfect = 0, always-0.5 = 0.25
    y = np.array([1, 0, 1, 0])
    good = abs(brier([1, 0, 1, 0], y)) < 1e-12 and abs(brier([0.5] * 4, y) - 0.25) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Brier: perfect 0.0, always-0.5 exactly 0.25")

    # --- 2. log loss punishes confident-and-WRONG harder than timid-and-wrong
    good = log_loss([0.99, 0.01], [0, 1]) > log_loss([0.6, 0.4], [0, 1])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] log loss: confident-and-wrong scores worse than "
          f"timid-and-wrong")

    # --- 3. Murphy decomposition reconstructs the Brier score, including the within-bin term
    n = 400
    p = rng.uniform(0.05, 0.95, n)
    yb = (rng.uniform(0, 1, n) < p).astype(float)
    d = brier_decomposition(p, yb, 10)
    recon = (d["reliability"] - d["resolution"] + d["uncertainty"] + d["bin_variance"]
             + d["covariance"])
    good = abs(recon - brier(p, yb)) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] rel - res + unc + bin_var + cov == Brier "
          f"({recon:.9f} vs {brier(p, yb):.9f})")

    # --- 3b. the classic three-term form is exact ONLY for discrete forecasts. With a continuous
    #     score the within-bin covariance survives, and omitting it is how a calibration report
    #     quietly understates its own error.
    omitting = d["reliability"] - d["resolution"] + d["uncertainty"]
    good = abs(omitting - brier(p, yb)) > 1e-6 and abs(d["covariance"]) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the 3-term form does NOT close on continuous "
          f"forecasts (off by {abs(omitting - brier(p, yb)):.6f}, covariance "
          f"{d['covariance']:+.6f})")

    # --- 3c. but on DISCRETE forecasts -- a model that emits a handful of fixed values -- it does
    discrete = np.round(p * 4) / 4
    dd = brier_decomposition(discrete, yb, 10)
    recon_d = dd["reliability"] - dd["resolution"] + dd["uncertainty"]
    good = abs(recon_d - brier(discrete, yb)) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...and on discrete forecasts the 3-term identity is "
          f"exact, which is why it is taught that way")

    # --- 4. a KNOWN distortion is recovered by Platt: the fitted slope is the tell, not the Brier
    #     gain. A uniform truth distribution means a 2x overconfidence costs less Brier than
    #     intuition suggests (~7%), but the recovered `a` should be ~0.5.
    t = rng.uniform(0.05, 0.95, 1500)
    yc = (rng.uniform(0, 1, len(t)) < t).astype(float)
    distorted = 1 / (1 + np.exp(-2.0 * _logit(t)))
    before = brier(distorted, yc)
    cal = Platt().fit(distorted, yc)
    after = brier(cal(distorted), yc)
    good = abs(cal.a - 0.5) < 0.1 and after < before
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Platt recovers the 2x overconfidence "
          f"(fitted a={cal.a:.3f}, true 0.500; Brier {before:.4f} -> {after:.4f})")

    # --- 5. isotonic recovers a monotone distortion it can express
    good_ = np.where(t < 0.5, t * 0.5, 0.25 + (t - 0.5) * 1.5)
    iso = Isotonic().fit(t, yc)
    good = brier(iso(t), yc) <= brier(good_, yc) + 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] isotonic matches or beats a known monotone map")

    # --- 6. ECE: an overconfident model has high ECE, a calibrated one low
    p_over = np.clip(t + 0.25 * np.sign(t - 0.5), 0.01, 0.99)
    e_over, e_ok = ece(p_over, yc), ece(Platt().fit(p_over, yc)(p_over), yc)
    good = e_over > e_ok
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ECE falls after calibration "
          f"({e_over:.3f} -> {e_ok:.3f})")

    # --- 7. THE HONESTY TEST: isotonic fitted on the SAME data memorises, and the memory does not
    #     survive held-out data. The signature is not "no gain" -- with labels independent of the
    #     feature the base rate IS the optimal forecast, so calibration legitimately helps. The
    #     signature is the GAP between in-sample and out-of-sample.
    nz = 120
    pn = rng.uniform(0.2, 0.8, nz)
    yn = (rng.uniform(0, 1, nz) < 0.5).astype(float)
    iso_all = Isotonic().fit(pn, yn)
    ins = brier(iso_all(pn), yn)                       # fitted on these exact points
    cv = cross_validate(pn, yn, "isotonic", folds=5, seed=3)
    gap = cv["brier_oos"] - ins
    good = cv["ok"] and gap > 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] isotonic on noise: in-sample Brier {ins:.3f} but "
          f"held-out {cv['brier_oos']:.3f} -- held-out is worse by {gap:+.3f}, so the flexibility "
          f"bought nothing that generalises")

    # --- 7b. the harness reports both numbers, so the gap cannot be hidden by quoting one
    good = ("brier_in_sample" in cv and "brier_oos" in cv
            and cv["brier_in_sample"] < cv["brier_oos"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both in-sample ({cv['brier_in_sample']:.3f}) and "
          f"out-of-sample ({cv['brier_oos']:.3f}) are always reported together")

    # --- 8. and where a real distortion exists, the out-of-sample gain IS real
    cv2 = cross_validate(distorted, yc, "platt", folds=5, seed=3)
    good = cv2["delta_oos"] < -0.01
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] real distortion: out-of-sample Brier improves "
          f"{cv2['delta_oos']:+.4f}")

    # --- 9. too small a sample is refused, not silently reported
    cv3 = cross_validate([0.5, 0.6], [0, 1], "platt", folds=5)
    good = cv3["ok"] is False and "too small" in cv3["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] n=2 is refused with a reason, not scored")

    # --- 10. compare_to_rule: a perfect rule beats a useless model, and says so
    nz2 = 400
    truth_p = rng.uniform(0.05, 0.95, nz2)
    y2 = (rng.uniform(0, 1, nz2) < truth_p).astype(float)
    rule = np.where(truth_p > 0.5, 0.95, 0.05)                 # a good rule
    model = np.full(nz2, 0.5)                                  # no information
    cmp = compare_to_rule(rule, model, y2, folds=5, seed=5)
    good = cmp["ok"] and cmp["winner"] == "rule"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a hand rule that knows beats a model that does not "
          f"(rule {cmp['rule_brier_oos']:.3f} vs model {cmp['model_brier_oos']:.3f})")

    # --- 11. and a model that knows beats a useless rule
    cmp2 = compare_to_rule(np.full(nz2, 0.5), np.where(truth_p > 0.5, 0.95, 0.05), y2,
                           folds=5, seed=5)
    good = cmp2["winner"] == "model"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...and the reverse, so the harness is not "
          f"predisposed")

    # --- 12. reliability table exposes direction of miscalibration
    tbl = reliability_table(p_over, yc, 5)
    good = bool(tbl) and all({"bin", "n", "pred", "obs", "gap"} == set(r) for r in tbl)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] reliability table exposes the gap per bin")

    # --- 13. identity is the baseline and changes nothing
    good = abs(brier(Identity()(distorted), yc) - brier(distorted, yc)) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] identity calibration is a true no-op baseline")

    # --- 14. ROUTING makes the measured rule executable, and the two arrays are genuinely different
    z = np.random.default_rng(4).normal(0, 2, (200, 4))
    pr = Probs.from_logits(z, temperature=4.2, labels=np.zeros(200, dtype=int))
    good = (not np.allclose(pr.raw, pr.calibrated)
            and pr.select("threshold").shape == pr.raw.shape)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a Probs carries two DIFFERENT arrays and routes between "
          f"them")

    # --- 15. the routing directions are the measured ones, not a global switch
    good = (np.array_equal(pr.select("threshold"), pr.calibrated)
            and np.array_equal(pr.select("ranking"), pr.raw)
            and np.array_equal(pr.select("conformal_mondrian"), pr.calibrated)
            and np.array_equal(pr.select("conformal_marginal"), pr.raw))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] routing matches the measurements: threshold/Mondrian -> "
          f"calibrated, ranking/Marginal -> raw")

    # --- 16. an unknown consumer FAILS CLOSED rather than silently picking one
    try:
        pr.select("vibes")
        good = False
    except KeyError:
        good = True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown consumer raises instead of guessing")

    # --- 17. and not every consumer wants the same thing -- the whole reason a table is needed
    wanted = set(ROUTING.values())
    good = wanted == {"raw", "calibrated"}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the table contains BOTH answers, so no global switch "
          f"could be correct")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibration + proper scoring for decision models.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--table", action="store_true", help="print a reliability table demo")
    args = ap.parse_args(argv)
    if args.self_test or not args.table:
        return 0 if run_self_test() else 1
    rng = np.random.default_rng(1)
    t = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(0, 1, 2000) < t).astype(float)
    over = 1 / (1 + np.exp(-2.5 * _logit(t)))
    for name in CALIBRATORS:
        cv = cross_validate(over, y, name, folds=5)
        print(f"{name:10s} oos brier={cv['brier_oos']:.4f} "
              f"(identity {cv['brier_identity']:.4f}) ece={cv['ece_oos']:.3f}")
    print("\nreliability after Platt:")
    cal = Platt().fit(over, y)
    for r in reliability_table(cal(over), y, 5):
        print(f"  {r['bin']:10s} n={r['n']:4d} pred={r['pred']:.3f} obs={r['obs']:.3f} "
              f"gap={r['gap']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
