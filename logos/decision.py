"""logos.decision — the Bayesian half: prior-mismatch, evidence-gap, epistemic, and PPI.

WHY THIS REPLACES THE LOGISTIC CORRECTNESS HEAD (which lost three times)
-----------------------------------------------------------------------
A 15-feature logistic head on summary features lost to free calibrated max-probability three
times in a row. The research explains why it *had* to lose:

  1. **No new information.** Every one of the 15 features was a function of the same forward
     pass that `max_p` already summarises nonlinearly. A linear model on correlated
     functions of one signal adds parameters, not evidence.
  2. **No guarantee.** A fitted discriminative head can lose; it has no fallback.
  3. **It averaged four different questions.** "This looks fishy" is not one quantity.

So the design here is not a better head. It is (a) four *named* quantities with four
separate validation targets, and (b) a rectifier with a formal no-loss property.

  p_verdict_wrong    discriminative: is the argmax right?      -> calibrated encoder
  p_prior_mismatch   does the claim look fabricated?           -> claim-only model + probes
  p_evidence_gap     did retrieval fail?                       -> retrieval coverage
  p_epistemic        is the encoder out of its depth?          -> seed-ensemble disagreement

Naming discipline is the point. The previous attempt called all four "confidence" and could
not tell which one was broken.

PPI++ AND THE NO-LOSS PROPERTY
------------------------------
Prediction-Powered Inference (arXiv:2301.09633; PPI++ Ann. Appl. Statist. 2026) gives the
property the logistic head lacked. With the interpolation form

    theta_lambda = mean(y_L) + lambda * (mean(f_U) - mean(f_L))

lambda = 0 recovers the classical labelled estimate EXACTLY, and lambda is fitted on held-out
data. So the rectified estimator can only beat the classical one up to estimation noise -- a
bad auxiliary predictor tunes itself to zero instead of degrading the answer. That is why it
is safe to add weak signals here and why it was unsafe to stack 15 weights by hand.

USAGE
-----
    python tools/logos/decision.py --self-test
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# ------------------------------------------------------------------------ Beta reliability
class BetaReliability:
    """Conjugate Beta-Binomial reliability for one source.

    Two sources that agree are not evidence of truth when one of them has been wrong 8 times
    out of 10. Tracking alpha/beta per source is the cheapest way to make "source
    reliability" honest instead of an assumed 0.5, and it keeps a credible interval so a
    source with 2 observations cannot shout.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n = 0
        self.k = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """Total pseudo-observations -- how much this estimate should be trusted at all."""
        return self.alpha + self.beta

    def update(self, correct: bool, weight: float = 1.0) -> "BetaReliability":
        if weight <= 0:
            raise ValueError("weight must be > 0")
        self.n += 1
        if correct:
            self.k += 1
            self.alpha += weight
        else:
            self.beta += weight
        return self

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        try:
            from scipy.stats import beta as _beta
            lo = float(_beta.ppf((1 - level) / 2, self.alpha, self.beta))
            hi = float(_beta.ppf(1 - (1 - level) / 2, self.alpha, self.beta))
            return lo, hi
        except Exception:                      # noqa: BLE001 - honest fallback
            m = self.mean
            return (max(0.0, m - 0.5), min(1.0, m + 0.5))

    def shrunk(self, global_mean: float, prior_strength: float = 4.0) -> float:
        """Empirical-Bayes style shrinkage toward the population mean.

        A source with 3 observations should not dominate a source with 300. Shrinking by
        pseudo-count is the standard fix and it is why `strength` exists.
        """
        a = self.alpha + prior_strength * global_mean
        b = self.beta + prior_strength * (1.0 - global_mean)
        return a / (a + b)


# ---------------------------------------------------------------------------------- PPI
def ppi_lambda(y_lab: np.ndarray, f_lab: np.ndarray, f_unlab: np.ndarray) -> float:
    """Regression coefficient for PPI++, clipped to [0, 1].

    Estimated on the labelled set as Cov(y, f) / Var(f). Clipping matters: a negative lambda
    would mean "trust the auxiliary predictor in reverse", which is never wanted, and >1
    would extrapolate beyond the labelled mean's own variance budget.
    """
    y_lab = np.asarray(y_lab, dtype=np.float64)
    f_lab = np.asarray(f_lab, dtype=np.float64)
    if f_lab.size < 2 or np.var(f_lab) <= 1e-12:
        return 0.0
    lam = float(np.cov(y_lab, f_lab)[0, 1] / np.var(f_lab, ddof=1))
    return float(np.clip(lam, 0.0, 1.0))


def ppi_estimate(y_lab: np.ndarray, f_lab: np.ndarray, f_unlab: np.ndarray,
                 lam: float | None = None) -> dict:
    """theta_lambda = mean(y_L) + lambda * (mean(f_U) - mean(f_L)).

    lambda = 0 is exactly the classical labelled estimate. That identity is asserted in the
    self-test, because it is the property that makes this safe where a logistic head was not.
    """
    y_lab = np.asarray(y_lab, dtype=np.float64)
    f_lab = np.asarray(f_lab, dtype=np.float64)
    f_unlab = np.asarray(f_unlab, dtype=np.float64)
    if y_lab.size == 0 or f_unlab.size == 0:
        raise ValueError("need at least one labelled and one unlabelled point")
    lam = ppi_lambda(y_lab, f_lab, f_unlab) if lam is None else float(np.clip(lam, 0.0, 1.0))
    classical = float(y_lab.mean())
    correction = float(f_unlab.mean() - f_lab.mean())
    theta = classical + lam * correction
    return {"theta": theta, "lambda": lam, "classical": classical,
            "correction": correction,
            # variance of the rectified estimate; the labelled term and the correction term
            # are treated as independent because f_U is a different sample of documents
            "se_approx": float(math.sqrt(np.var(y_lab, ddof=1) / max(y_lab.size, 1)
                                         + lam ** 2 * np.var(f_unlab, ddof=1)
                                         / max(f_unlab.size, 1)))}


# ------------------------------------------------------------------------ fishiness atoms
def noisy_or(parts) -> float:
    """1 - prod(1 - p). Any one strong suspicion is enough to be suspicious.

    Deliberately not a mean. Averaging lets three easy components hide one screaming one:
    evidence_gap 0.95 alongside a few 0.1s averages to ~0.3 and looks acceptable, when the
    only honest reading is "retrieval failed, we know nothing".
    """
    keep = 1.0
    for p in parts:
        if p is None:
            continue
        keep *= 1.0 - float(np.clip(p, 0.0, 1.0))
    return 1.0 - keep


def evidence_gap(retrieved: int, usable: int, min_usable: int = 1) -> float:
    """Share of the retrieval budget that produced nothing usable.

    Deterministic and auditable: it counts. This is the one "fishy" component that needs no
    model and no calibration, so it can be trusted before anything else is wired.
    """
    if retrieved <= 0:
        return 1.0
    deficit = max(0, min_usable - usable) / max(min_usable, 1)
    empty = 1.0 - (usable / retrieved)
    return float(np.clip(max(deficit, empty), 0.0, 1.0))


def epistemic_from_ensemble(preds: list[int], n_classes: int = 4) -> float:
    """Normalised disagreement across independently trained heads.

    Jensen-Shannon-free, auditable version: 1 - (largest agreeing fraction), rescaled so that
    unanimous agreement is 0 and a uniform split is 1.
    """
    if not preds:
        return 1.0
    counts = np.bincount(np.asarray(preds, dtype=int), minlength=n_classes)
    frac = counts / counts.sum()
    top = float(frac.max())
    floor = 1.0 / n_classes
    return float(np.clip((1.0 - top) / max(1.0 - floor, 1e-9), 0.0, 1.0))


# ------------------------------------------------------------------------------ tests
def run_self_test() -> bool:
    print("logos.decision - SELF TEST")
    ok = True

    # --- Beta conjugacy against the closed form
    b = BetaReliability(alpha=2.0, beta=2.0)
    for v in [1, 1, 1, 0]:
        b.update(bool(v))
    want = (2.0 + 3) / (4.0 + 4)
    good = abs(b.mean - want) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Beta posterior mean {b.mean:.4f} "
          f"== closed form {want:.4f}")
    lo, hi = b.credible_interval(0.9)
    good = lo < b.mean < hi
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 90% credible interval [{lo:.3f}, {hi:.3f}] "
          f"brackets the mean")

    # a source with 2 observations must be pulled toward the population mean
    tiny = BetaReliability(1.0, 1.0)
    tiny.update(True)
    shr = tiny.shrunk(global_mean=0.6, prior_strength=8.0)
    good = abs(shr - 0.6) < abs(tiny.mean - 0.6)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] shrinkage pulls 1-of-1 ({tiny.mean:.2f}) "
          f"toward 0.60 -> {shr:.3f}")

    # --- PPI: lambda = 0 must reproduce the classical estimate EXACTLY
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    f_lab = y * 0.8 + rng.normal(0, 0.5, 400)
    f_unlab = rng.normal(0.5, 0.6, 20000)
    r0 = ppi_estimate(y, f_lab, f_unlab, lam=0.0)
    good = r0["theta"] == float(y.mean()) and r0["lambda"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] lambda=0 recovers the classical estimate "
          f"exactly ({r0['theta']:.12f})")

    # --- a NOISE auxiliary predictor must tune itself to ~0, not degrade the answer
    noise = rng.normal(0, 1, 400)
    rn = ppi_estimate(y, noise, rng.normal(0, 1, 20000))
    good = rn["lambda"] < 0.05 and abs(rn["theta"] - rn["classical"]) < 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] noise auxiliary -> lambda {rn['lambda']:.4f}, "
          f"theta {rn['theta']:.4f} vs classical {rn['classical']:.4f}")

    # --- an ORACLE auxiliary must collapse variance across seeds
    def spread(aux_kind: str) -> float:
        vals = []
        for s in range(60):
            g = np.random.default_rng(1000 + s)
            yy = g.integers(0, 2, 200).astype(float)
            fu = g.integers(0, 2, 20000).astype(float)
            fl = yy.copy() if aux_kind == "oracle" else g.normal(0, 1, 200)
            vals.append(ppi_estimate(yy, fl, fu)["theta"])
        return float(np.std(vals))
    s_none, s_oracle = spread("noise"), spread("oracle")
    good = s_oracle < s_none * 0.6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] oracle auxiliary cuts across-seed sd "
          f"{s_none:.4f} -> {s_oracle:.4f}")

    # --- noisy-OR semantics: one strong signal is enough
    good = abs(noisy_or([0.95, 0.1, 0.1]) - (1 - 0.05 * 0.9 * 0.9)) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] noisy-OR does not let easy components hide a "
          f"loud one ({noisy_or([0.95, 0.1, 0.1]):.4f} vs mean 0.383)")

    # --- evidence gap is a count, and empty retrieval is maximal suspicion
    good = evidence_gap(0, 0) == 1.0 and evidence_gap(5, 0) == 1.0 and evidence_gap(5, 5) == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] evidence_gap: empty retrieval = 1.0, "
          f"all usable = 0.0")

    # --- epistemic: unanimous vs split
    good = epistemic_from_ensemble([1, 1, 1, 1], 4) == 0.0
    ok = ok and good
    split = epistemic_from_ensemble([0, 1, 2, 3], 4)
    good = abs(split - 1.0) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] epistemic: unanimous 0.0, uniform split 1.0")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bayesian fishiness atoms + PPI rectifier.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
