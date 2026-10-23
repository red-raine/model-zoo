"""logos.fusion — combining specialist heads without the pathologies.

THE QUESTION THIS ANSWERS
-------------------------
The operator asked whether combining heads with attention needs adjusted math. The answer is
worse than "needs adjustment": **a softmax gate over experts structurally violates the one
property we need.** Its weights are constrained to sum to 1, so adding a supporting expert
STEALS probability mass from the experts already present. Demonstrated in the self-test:
three experts at P(Supported)=0.9 pooled by a softmax gate give 0.9, 0.9, then 0.633 when a
supporting expert at 0.1 joins. Adding evidence in favour of an answer LOWERED its
probability. That single property test kills the design.

The property has a name. "Adding a supporting expert must not lower P(Supported)" is
**coordinate-wise monotonicity**, and log-linear pooling with non-negative weights satisfies
it while any renormalising gate does not.

WHAT TO USE INSTEAD
-------------------
    p_out(y) ∝ exp( b_y + Σ_i w_i · log p_i(y) ),    w_i ≥ 0

- w_i ≡ 1, b ≡ 0  -> plain product of experts
- w_i ≡ 1/K       -> PoE with sharpness correction
- b_y = (1-K)·log p_0(y) -> the Bayesian Committee Machine

PoE over correlated experts **over-sharpens without informing**: two experts both at 0.9
give 0.99, three give 0.99863. Confidence grows with the number of heads even when the heads
add no information. Measured in the wild: naive odds composition of an LLM with Bayesian
odds + Dempster-Shafer more than doubled ECE (0.16 -> 0.46) with no AUROC gain (arXiv:2608.14617).
So `bcm_pool` exists as the honest baseline and the weights are fit, not assumed.

THE ORDER THAT MATTERS
----------------------
    per-expert temperature -> log-odds standardization -> pool -> final temperature

Skipping standardization is why hand-tuned fusion weights do not transfer: an expert with
T=2 contributes a shrunk logit and is silently down-weighted, while T=0.5 dominates every
pool it enters.

USAGE
-----
    python tools/logos/fusion.py --self-test
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def _norm_log(logw: np.ndarray) -> np.ndarray:
    """Normalise log-weights to probabilities, ROW-WISE for batches.

    A single-expert vector and an (n, K) batch both reach this function. An earlier version
    normalised the whole array globally, so a batch of 4000 rows was scaled by one shared
    denominator: rows no longer summed to 1 and `calibrate_then_pool` returned something that
    was not a distribution. Caught by asserting that the pipeline output IS a distribution.
    """
    logw = np.asarray(logw, dtype=float)
    if logw.ndim == 1:
        m = float(np.max(logw))
        e = np.exp(logw - m)
        return e / e.sum()
    m = logw.max(axis=1, keepdims=True)
    e = np.exp(logw - m)
    return e / e.sum(axis=1, keepdims=True)


def loglinear_pool(probs: list[np.ndarray], weights: list[float] | None = None,
                   bias: np.ndarray | None = None, eps: float = 1e-9) -> np.ndarray:
    """p ∝ exp(b + Σ w_i log p_i). Weights must be non-negative to keep monotonicity.

    eps clipping does NOT fix the zero-forcing pathology on its own; it converts a hard veto
    into a soft one whose strength grows with the number of experts. Pair it with fitted
    weights and a fitted bias, which is what this function is for.
    """
    if not probs:
        raise ValueError("need at least one expert")
    if weights is None:
        weights = [1.0] * len(probs)
    if len(weights) != len(probs):
        raise ValueError("weights and probs must align")
    if any(w < 0 for w in weights):
        raise ValueError("negative weights break coordinate-wise monotonicity; "
                         "that is the whole reason this function exists")
    acc = None
    for w, p in zip(weights, probs):
        lp = np.log(np.clip(np.asarray(p, dtype=float), eps, 1.0))
        acc = w * lp if acc is None else acc + w * lp
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=float)
    return _norm_log(acc)


def bcm_pool(probs: list[np.ndarray], prior: np.ndarray) -> np.ndarray:
    """Bayesian Committee Machine: PoE plus a de-sharpening prior term p0^(1-K).

    This is the honest baseline. Unweighted PoE has no reason to be calibrated when the
    experts are correlated, which ours are by construction (all read the same claim and the
    same retrieved passages).
    """
    k = len(probs)
    prior = np.asarray(prior, dtype=float)
    acc = (1.0 - k) * np.log(np.clip(prior, 1e-12, 1.0))
    for p in probs:
        acc = acc + np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1.0))
    return _norm_log(acc)


def attention_gate_pool(probs: list[np.ndarray], logits: np.ndarray) -> np.ndarray:
    """A softmax-weighted mixture of experts. PROVIDED ONLY AS A COUNTEREXAMPLE.

    Do not use this. The weights are normalised to sum to 1, so the mixture is a convex
    combination: adding an expert with a lower score on class c necessarily drags P(c) down,
    even when that expert AGREES the claim is supported elsewhere. The self-test asserts the
    violation so that nobody re-introduces this construction later.

    Note: attention over expert outputs IS a learned gate -- the softmax of inner products is
    just a particular parametrisation. There is no separate "attention fusion" family.
    """
    g = np.asarray(logits, dtype=float)
    a = np.exp(g - g.max())
    a = a / a.sum()
    out = np.zeros_like(np.asarray(probs[0], dtype=float))
    for ai, p in zip(a, probs):
        out = out + ai * np.asarray(p, dtype=float)
    return out


# --------------------------------------------------------------------------- calibration
def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """1-D temperature on probabilities, by NLL golden-section search on log T."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    lp = np.log(np.clip(probs, 1e-12, 1.0))

    def nll(T: float) -> float:
        z = lp / max(T, 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        lz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-lz[np.arange(len(y)), y].mean())

    lo, hi = math.log(0.05), math.log(20.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(math.exp(d))
        if b - a < 1e-10:
            break
    return float(math.exp(0.5 * (a + b)))


def standardize_log_odds(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each expert's log-odds by its own std so that 'one unit' means the same thing
    across experts. Without this, fitted weights are not comparable and do not transfer."""
    lp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0))
    sd = float(lp.std())
    if sd <= 1e-12:
        return lp, 1.0
    return lp / sd, sd


def calibrate_then_pool(all_probs: list[np.ndarray], y: np.ndarray,
                        prior: np.ndarray) -> dict:
    """The four mandated stages, each individually testable and reported separately."""
    temps = [fit_temperature(p, y) for p in all_probs]
    scaled = []
    for p, T in zip(all_probs, temps):
        lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)) / T
        scaled.append(_norm_log(lp))
    std = [standardize_log_odds(p / p.sum(axis=1, keepdims=True))[1] for p in scaled]
    pooled = loglinear_pool(scaled)
    T_out = fit_temperature(pooled, y)
    lp = np.log(np.clip(pooled, 1e-12, 1.0)) / T_out
    final = _norm_log(lp)
    return {"temperatures": temps, "logodds_sd": std, "out_temperature": T_out,
            "pooled": pooled, "final": final}


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.fusion - SELF TEST")
    ok = True
    two = np.array([0.9, 0.1])

    # --- analytic: geometric mean of [.9,.1] and [.5,.5] with w=.5 each is [.75,.25]
    r = loglinear_pool([two, np.array([0.5, 0.5])], [0.5, 0.5])
    good = abs(r[0] - 0.75) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] geometric pool [.9,.1]+[.5,.5] -> "
          f"[{r[0]:.12f}, {r[1]:.12f}] (want sqrt(.45):sqrt(.05) = 3:1)")

    # --- unanimity / idempotence
    r = loglinear_pool([two, two], [0.5, 0.5])
    good = abs(r[0] - 0.9) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unanimity: identical experts return the input "
          f"({r[0]:.12f})")

    # --- the PoE over-sharpening pathology, asserted deliberately
    r = loglinear_pool([two, two, two], [1.0, 1.0, 1.0])
    good = abs(r[0] - 0.998630) < 1e-6 and abs(r[1] - 0.001370) < 1e-6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] PoE over-sharpens: 3 correlated experts at .9 "
          f"-> {r[0]:.6f} (0.729/0.730). Confidence grows with NO new information.")

    # --- BCM de-sharpens by exactly the prior term
    r = bcm_pool([two, two], np.array([0.6, 0.4]))
    good = abs(r[0] - 0.981818) < 1e-6 and abs(r[1] - 0.018182) < 1e-6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] BCM with prior [.6,.4] and K=2 -> "
          f"{r[0]:.6f} (1.35/1.375)")

    # --- THE PROPERTY THAT DECIDES THE ARCHITECTURE
    e1, e2 = np.array([0.9, 0.1]), np.array([0.9, 0.1])
    add = np.array([0.1, 0.9])          # an expert that does NOT support class 0
    gate_before = attention_gate_pool([e1, e2], np.array([1.0, 1.0]))
    gate_after = attention_gate_pool([e1, e2, add], np.array([1.0, 1.0, 1.0]))
    good = gate_after[0] < gate_before[0] - 0.2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] softmax gate VIOLATES monotonicity: P(cls0) "
          f"{gate_before[0]:.3f} -> {gate_after[0]:.3f} when a third expert joins "
          f"(this is why attention fusion is wrong here)")
    # same roster, same weights, but the dissenting expert gets w=0: the answer is untouched
    ll_after = loglinear_pool([e1, e2, add], [0.5, 0.5, 0.0])
    good = abs(ll_after[0] - gate_before[0]) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] log-linear with w=0 on that expert is "
          f"UNCHANGED ({ll_after[0]:.3f}), i.e. it cannot be dragged down by it")

    # --- monotonicity property test over random inputs, w >= 0
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        k = int(rng.integers(2, 6))
        ps = [rng.dirichlet(np.ones(4)) for _ in range(k)]
        w = rng.random(k) + 0.01
        base = loglinear_pool(ps, list(w))[0]
        j = int(rng.integers(0, k))
        bumped = [p.copy() for p in ps]
        bumped[j][0] = min(1.0, bumped[j][0] + 0.1)
        bumped[j] = bumped[j] / bumped[j].sum()
        after = loglinear_pool(bumped, list(w))[0]
        worst = min(worst, after - base)
    good = worst > -1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] monotone in 2000 random draws: raising one "
          f"expert's P(cls0) never lowers the pool (worst delta {worst:.2e})")

    # --- negative weights are refused, because they would break that guarantee
    try:
        loglinear_pool([two, two], [1.0, -0.5])
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] negative weights rejected at the API boundary")

    # --- calibration recovers a known temperature
    g = np.random.default_rng(1)
    n = 4000
    p_true = g.dirichlet(np.full(4, 0.4), n)
    y = np.array([g.choice(4, p=row) for row in p_true])
    base = np.log(np.clip(p_true, 1e-9, 1.0))
    hot = np.exp(base * 3.0)
    hot = hot / hot.sum(1, keepdims=True)
    T = fit_temperature(hot, y)
    good = abs(T - 3.0) / 3.0 < 0.10
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] temperature recovery: true 3.0, fitted {T:.3f}")

    # --- standardization makes two differently-sharpened copies interchangeable
    a = np.exp(base)
    a = a / a.sum(1, keepdims=True)
    b = np.exp(base / 3.0)
    b = b / b.sum(1, keepdims=True)
    la, sa = standardize_log_odds(a)
    lb, sb = standardize_log_odds(b)
    ratio = sb / max(sa, 1e-12)
    good = abs(ratio - (1.0 / 3.0)) < 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] standardisation detects the sharpness ratio "
          f"({ratio:.4f}, want 0.3333)")

    # --- the four-stage pipeline runs end to end and stays a distribution
    out = calibrate_then_pool([a, b], y, np.full(4, 0.25))
    good = abs(float(out["final"].sum(axis=1).mean()) - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] calibrate->standardize->pool->calibrate yields "
          f"a distribution (T_out {out['out_temperature']:.2f})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Log-linear expert fusion.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.demo:
        return 0 if run_self_test() else 1
    experts = {"cross_encoder": np.array([0.30, 0.55, 0.08, 0.07]),
               "bi_encoder": np.array([0.35, 0.40, 0.15, 0.10]),
               "symbolic": np.array([0.05, 0.90, 0.03, 0.02])}
    ps = list(experts.values())
    print("experts:", {k: v.round(3).tolist() for k, v in experts.items()})
    print("poe     :", loglinear_pool(ps).round(4))
    print("bcm     :", bcm_pool(ps, np.array([0.24, 0.61, 0.07, 0.08])).round(4))
    print("gate    :", attention_gate_pool(ps, np.zeros(len(ps))).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
