"""logos.abstain — the three-action decision: YES / NO / LOOK IT UP.

THE CORRECTION THIS FILE MAKES
------------------------------
The operator asked for a "3-bit head: yes, no, abstain (look it up)". The instinct is right
and the packaging is wrong, and the difference matters:

  **NEI is a verdict about the world.** The evidence does not settle the claim.
  **Abstain is a decision about cost.** We choose not to commit and to go look it up.

These are orthogonal axes. Putting abstain into the label set as a fifth class means a claim
that is *confidently* Not Enough Evidence can never abstain, and an unconfident Supported
claim is forced to masquerade as NEI. So: keep the 4-way verdict distribution, and put
abstention on top as a separate cost-sensitive decision.

THE RULE
--------
Answer c iff the expected cost of answering is below the cost of deferring:

    E[cost(answer c)] = Σ_y C(c, y) · p(y)      with C(c,c) = 0
    E[cost(abstain)]  = C_d
    action = argmin over {answer each c, abstain}

When the cost matrix is *uniform* over errors (every mistake costs C_w) this collapses to
Chow's rule with a closed-form threshold:

    t* = 1 − C_d / C_w        answer iff max_y p(y) ≥ t*

and the self-test asserts both paths agree in that case. But the uniform case is a special
case we must not ship: calling a refuted claim "Supported" is worse than calling it NEI, so
the general cost matrix is the real interface. The self-test includes a fixture where the
asymmetric matrix abstains while max-prob would answer.

MONDRIAN CONFORMAL, NOT MARGINAL
--------------------------------
Marginal conformal prediction looks fine overall and collapses on the rare class: across
3,150 runs on imbalanced data, marginal CP minority coverage fell as low as 0.5%, while
class-conditional (Mondrian) CP restored it by +61.7 points on average (arXiv:2607.27143).
Our rare class is `Conflicting` (7.6% of dev) — exactly the one that decides macro-F1. So
`MondrianAPS` sets a separate threshold per label, and the self-test encodes the collapse as
a regression test: marginal APS must FAIL to cover the rare class while Mondrian covers it.

USAGE
-----
    python tools/logos/abstain.py --self-test
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# `abstain.py` is runnable as a script (`python tools/logos/abstain.py --self-test`), so the package
# root has to be on the path before the intra-package import. Every other module in this package does
# the same; skipping it works under pytest and fails from the command line, which is the worst
# combination.
import os as _os
import sys as _sys

_TOOLS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _TOOLS not in _sys.path:
    _sys.path.insert(0, _TOOLS)

from logos.calibrate import coerce

N_CLASSES = 4


# ------------------------------------------------------------------ cost-sensitive action
def expected_costs(p, cost: np.ndarray) -> np.ndarray:
    """Expected cost of each of the 5 actions: answer 0..3, then abstain last.

    Accepts a `calibrate.Probs`, in which case the COST consumer's array is used -- and the
    routing table says that is the CALIBRATED one, because expected cost is only meaningful if
    the probabilities mean what they say.
    """
    p = coerce(p, "cost")
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"cost matrix must be {N_CLASSES}x{N_CLASSES}, got {cost.shape}")
    if np.any(np.diag(cost) != 0):
        raise ValueError("a correct answer must cost 0 on the diagonal")
    if np.any(cost < 0):
        raise ValueError("negative costs are nonsense")
    return (p[None, :] * cost).sum(axis=1)


def abstain_action(p, cost: np.ndarray, c_defer: float) -> dict:
    """argmin over {answer each class, abstain}. Returns the action and all expected costs.

    Accepts a `calibrate.Probs` and routes to the CALIBRATED array. Measured necessity: with raw
    (overconfident, ECE 0.098) probabilities no threshold met a 10% risk target on unseen rows at
    all -- coverage 0.000 -- while the calibrated array covered 0.72-0.78. A Chow threshold is a
    statement about probability, so it needs probabilities that mean what they say.
    """
    p = coerce(p, "threshold")
    if abs(float(p.sum()) - 1.0) > 1e-6:
        raise ValueError(f"posterior must be normalised, sums to {p.sum()}")
    ec = expected_costs(p, cost)
    all_costs = np.concatenate([ec, [float(c_defer)]])
    best = int(np.argmin(all_costs))
    action = "abstain" if best == N_CLASSES else int(best)
    return {"action": action, "expected_costs": ec, "cost_defer": float(c_defer),
            "best_cost": float(all_costs[best]),
            "margin": float(np.sort(all_costs)[1] - all_costs[best])}


def chow_threshold(c_wrong: float, c_defer: float) -> float:
    """Closed form for the uniform-cost case: t* = 1 - C_d / C_w."""
    if c_wrong <= 0:
        raise ValueError("c_wrong must be > 0")
    if c_defer < 0:
        raise ValueError("c_defer must be >= 0")
    return float(1.0 - c_defer / c_wrong)


def uniform_cost(c_wrong: float = 1.0) -> np.ndarray:
    c = np.full((N_CLASSES, N_CLASSES), float(c_wrong))
    np.fill_diagonal(c, 0.0)
    return c


# ------------------------------------------------------------------------ Mondrian APS
class MondrianAPS:
    """Class-conditional Adaptive Prediction Sets.

    Per label c, the threshold is the ⌈(n_c + 1)(1 − α_c)⌉-th smallest calibration score
    *among rows whose true label is c*, capped at n_c. Reporting per-class coverage is
    mandatory: a marginal number can look perfect while the rare class is uncovered.
    """

    def __init__(self, alphas: dict[int, float] | float = 0.10) -> None:
        if isinstance(alphas, (int, float)):
            self.alphas = {c: float(alphas) for c in range(N_CLASSES)}
        else:
            self.alphas = {int(k): float(v) for k, v in alphas.items()}
        for a in self.alphas.values():
            if not 0 < a < 1:
                raise ValueError("alpha must be in (0,1)")
        self.q: dict[int, float] = {}
        self.n: dict[int, int] = {}

    @staticmethod
    def _score(p: np.ndarray, y: int) -> float:
        order = np.argsort(-np.asarray(p, dtype=float))
        mass = 0.0
        for c in order:
            mass += float(p[c])
            if int(c) == int(y):
                return mass
        return 1.0

    def fit(self, probs, y: np.ndarray) -> "MondrianAPS":
        probs = coerce(probs, "conformal_mondrian")
        y = np.asarray(y, dtype=int)
        for c in range(N_CLASSES):
            m = y == c
            nc = int(m.sum())
            self.n[c] = nc
            if nc == 0:
                self.q[c] = 1.0
                continue
            scores = np.array([self._score(probs[i], y[i]) for i in np.where(m)[0]])
            scores.sort()
            idx = min(int(math.ceil((nc + 1) * (1.0 - self.alphas[c]))), nc) - 1
            self.q[c] = float(scores[max(idx, 0)])
        return self

    def predict_set(self, p) -> set[int]:
        """{c : score_c <= q_c}, i.e. each label judged against its OWN threshold.

        Routed to "conformal_mondrian", which the table sets to CALIBRATED: measured set size
        3.05 -> 2.78 with no loss of validity (coverage 0.897 -> 0.880 against a 0.900 target).
        """
        p = coerce(p, "conformal_mondrian")
        order = np.argsort(-p)
        out: set[int] = set()
        mass = 0.0
        for c in order:
            mass += float(p[c])
            if mass <= self.q[int(c)]:
                out.add(int(c))
        if not out:
            out.add(int(order[0]))
        return out

    def covers(self, p: np.ndarray, y: int) -> bool:
        return int(y) in self.predict_set(p)


class MarginalAPS:
    """One shared threshold. Kept only to demonstrate the rare-class collapse."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = float(alpha)
        self.q = 1.0

    def fit(self, probs, y: np.ndarray) -> "MarginalAPS":
        probs = coerce(probs, "conformal_marginal")
        y = np.asarray(y, dtype=int)
        scores = np.sort(np.array([MondrianAPS._score(probs[i], y[i])
                                   for i in range(len(y))]))
        n = len(scores)
        idx = min(int(math.ceil((n + 1) * (1.0 - self.alpha))), n) - 1
        self.q = float(scores[max(idx, 0)])
        return self

    def predict_set(self, p) -> set[int]:
        # routed to "conformal_marginal", which the table sets to RAW: measured set size 1.64 vs
        # 1.98 calibrated. BUT see the warning in calibrate.ROUTING -- Marginal under-covers the rare
        # classes by 16-19 points whatever array it gets (CONFLICTING 0.737, NEI 0.706 against a
        # 0.90 target). Use Mondrian if per-class coverage is what you need.
        p = coerce(p, "conformal_marginal")
        order = np.argsort(-p)
        out, mass = set(), 0.0
        for c in order:
            mass += float(p[c])
            if mass > self.q:
                break
            out.add(int(c))
        if not out:
            out.add(int(order[0]))
        return out

    def covers(self, p: np.ndarray, y: int) -> bool:
        return int(y) in self.predict_set(p)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.abstain - SELF TEST")
    ok = True

    # --- Chow's rule is the uniform-cost special case
    t = chow_threshold(1.0, 0.3)
    good = abs(t - 0.7) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Chow threshold with C_w=1.0, C_d=0.3 is "
          f"t* = {t:.12f}")

    uni = uniform_cost(1.0)
    p_abstain = np.array([0.45, 0.15, 0.30, 0.10])       # max 0.45 < 0.7
    r = abstain_action(p_abstain, uni, 0.3)
    good = r["action"] == "abstain"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] p max=0.45 -> {r['action']} (costs "
          f"{np.round(r['expected_costs'],3).tolist()} vs defer 0.3)")

    p_answer = np.array([0.72, 0.10, 0.12, 0.06])        # max 0.72 >= 0.7
    r2 = abstain_action(p_answer, uni, 0.3)
    good = r2["action"] == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] p max=0.72 -> answer class {r2['action']} "
          f"(expected cost {r2['best_cost']:.3f} vs defer 0.3)")

    # --- the general and closed-form paths must AGREE on the uniform case
    agree = True
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p = rng.dirichlet(np.ones(N_CLASSES))
        r = abstain_action(p, uni, 0.3)
        closed = "abstain" if p.max() < 0.7 else int(np.argmax(p))
        agree = agree and (r["action"] == closed or abs(p.max() - 0.7) < 1e-9)
    good = agree
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] general cost argmin == Chow threshold on 2000 "
          f"random posteriors")

    # --- ASYMMETRIC costs: max-prob answers, the cost matrix abstains
    asym = uniform_cost(1.0)
    asym[0, 1] = 2.0            # calling a Refuted claim "Supported" costs double
    p = np.array([0.55, 0.30, 0.10, 0.05])
    r = abstain_action(p, asym, 0.5)
    e_sup = 2.0 * 0.30 + 1.0 * 0.10 + 1.0 * 0.05
    e_ref = 1.0 * 0.55 + 1.0 * 0.10 + 1.0 * 0.05
    good = (r["action"] == "abstain" and abs(e_sup - 0.75) < 1e-12
            and abs(e_ref - 0.70) < 1e-12)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] asymmetric costs: E[Supported]={e_sup:.3f}, "
          f"E[Refuted]={e_ref:.3f}, E[defer]=0.500 -> {r['action']} "
          f"(max-prob would have answered Supported)")

    # --- cost-matrix validation is not decorative
    bad = uniform_cost(1.0)
    bad[0, 0] = 5.0
    try:
        expected_costs(np.full(4, 0.25), bad)
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a non-zero diagonal is rejected")

    # --- Mondrian threshold index is exact, not approximate
    g = np.random.default_rng(3)
    probs = g.dirichlet(np.ones(N_CLASSES) * 2.0, 400)
    y = np.array([g.choice(N_CLASSES, p=row) for row in probs])
    mono = MondrianAPS(0.10).fit(probs, y)
    nc = mono.n[0]
    if nc > 0:
        sc = np.sort(np.array([MondrianAPS._score(probs[i], y[i])
                               for i in np.where(y == 0)[0]]))
        idx = min(int(math.ceil((nc + 1) * 0.9)), nc) - 1
        good = abs(mono.q[0] - float(sc[idx])) < 1e-12
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] Mondrian class-0 threshold is exactly the "
              f"ceil(({nc}+1)*0.9)={idx+1}-th of {nc} scores")

    # --- THE COLLAPSE: marginal APS must under-cover the rare class, Mondrian must fix it
    n_cal, n_test = 1200, 4000
    # The rare class must be HARDER, or the collapse cannot reproduce. A first version gave
    # every class the same +2.2 boost, so the model was class-conditionally homogeneous and
    # marginal CP covered everything. That is not our setting: Conflicting F1 is .362 while
    # Refuted is .884. The boost now tracks real per-class difficulty.
    BOOST = {0: 2.0, 1: 2.8, 2: 1.4, 3: 0.6}      # Supported / Refuted / NEI / Conflicting
    def draw(n, g):
        yy = g.choice(N_CLASSES, size=n, p=[0.24, 0.61, 0.07, 0.08])   # AVeriTeC-like
        b = g.normal(0, 1.3, (n, N_CLASSES))
        b[np.arange(n), yy] += np.array([BOOST[int(c)] for c in yy])
        e = np.exp(b - b.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), yy
    pc, yc = draw(n_cal, g)
    pt, yt = draw(n_test, g)
    marg = MarginalAPS(0.10).fit(pc, yc)
    mond = MondrianAPS(0.10).fit(pc, yc)

    def class_cov(model, c):
        idx = np.where(yt == c)[0]
        return float(np.mean([model.covers(pt[i], yt[i]) for i in idx]))

    cov_m = {c: class_cov(marg, c) for c in range(N_CLASSES)}
    cov_d = {c: class_cov(mond, c) for c in range(N_CLASSES)}
    print(f"  [INFO] marginal per-class coverage: "
          f"{ {c: round(v,3) for c, v in cov_m.items()} }")
    print(f"  [INFO] Mondrian per-class coverage: "
          f"{ {c: round(v,3) for c, v in cov_d.items()} }")
    worst_m = min(cov_m.values())
    worst_d = min(cov_d.values())
    good = worst_d > worst_m + 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Mondrian lifts the WORST-class coverage "
          f"{worst_m:.3f} -> {worst_d:.3f} (the arXiv:2607.27143 failure, reproduced)")
    good = cov_m[3] < 0.88
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and the marginal rule really does under-cover "
          f"the rare class ({cov_m[3]:.3f} < 0.88) while Mondrian holds it "
          f"({cov_d[3]:.3f})")

    # APS coverage is only APPROXIMATELY valid per class (it is not exact split conformal on
    # a regression score), and with ~96 calibration rows for the rarest class the finite-sample
    # correction is genuinely conservative. So the fair claim is that Mondrian EQUALISES
    # coverage across classes -- not that every class hits exactly 0.90.
    dev_m = max(abs(v - 0.90) for v in cov_m.values())
    dev_d = max(abs(v - 0.90) for v in cov_d.values())
    good = dev_d < dev_m
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Mondrian equalises coverage: worst deviation "
          f"from 0.90 {dev_m:.3f} -> {dev_d:.3f}")

    # --- ROUTING IS ENFORCED, NOT DOCUMENTED. The same call, the same cost matrix, the same defer
    #     cost -- but handed a Probs it must use the consumer's array, and those arrays differ.
    #     raw is peaked so answering is cheap; calibrated is flat so abstaining wins. If routing were
    #     ignored, both calls would return the same action and this test would fail.
    from logos.calibrate import Probs

    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    flat = np.array([0.55, 0.15, 0.15, 0.15])
    # a single posterior, so shape (4,) -- abstain_action takes one row, not a batch
    pr = Probs(raw=peaked.copy(), calibrated=flat.copy(),
               temperature=4.2, ece_raw=0.098, ece_calibrated=0.031)
    cm = uniform_cost(c_wrong=1.0)
    a_cal = abstain_action(pr, cm, c_defer=0.2)              # threshold consumer -> calibrated
    a_bare = abstain_action(flat, cm, c_defer=0.2)           # bare array passes through
    a_raw = abstain_action(peaked, cm, c_defer=0.2)
    good = (a_cal.get("action") == a_bare.get("action")
            and a_cal.get("action") != a_raw.get("action"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] abstain_action routes a Probs to the CALIBRATED array "
          f"(cal frame -> {a_cal.get('action')}, raw would -> {a_raw.get('action')})")

    # --- and the conformal pair route in OPPOSITE directions from the SAME Probs object
    rng2 = np.random.default_rng(19)
    yy2 = rng2.integers(0, 4, 300)
    zz2 = rng2.normal(0, 1, (300, 4))
    zz2[np.arange(300), yy2] += 1.5
    raw2 = np.exp(zz2) / np.exp(zz2).sum(axis=1, keepdims=True)
    zz3 = zz2 / 4.2
    cal2 = np.exp(zz3) / np.exp(zz3).sum(axis=1, keepdims=True)
    pr2 = Probs(raw=raw2, calibrated=cal2, temperature=4.2)
    mond = MondrianAPS(alphas=0.10).fit(pr2, yy2)
    marg = MarginalAPS(alpha=0.10).fit(pr2, yy2)
    mond_cal = MondrianAPS(alphas=0.10).fit(cal2, yy2)
    marg_raw = MarginalAPS(alpha=0.10).fit(raw2, yy2)
    good = (np.allclose(mond.q[3], mond_cal.q[3], equal_nan=True)
            and abs(marg.q - marg_raw.q) < 1e-12)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Mondrian takes CALIBRATED and Marginal takes RAW from the "
          f"same Probs (their quantiles match the hand-split fits)")

    # --- an unknown consumer must not silently pick one
    try:
        pr2.select("nonsense")
        bad = False
    except KeyError:
        bad = True
    good = bad
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown consumer raises rather than guessing")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cost-sensitive abstention + Mondrian conformal.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.demo:
        return 0 if run_self_test() else 1
    from logos.schema import LABELS
    asym = uniform_cost(1.0)
    asym[0, 1] = 2.0
    for name, p in (("confident", np.array([0.72, 0.10, 0.12, 0.06])),
                    ("ambiguous", np.array([0.45, 0.15, 0.30, 0.10])),
                    ("near-tie", np.array([0.55, 0.30, 0.10, 0.05]))):
        r = abstain_action(p, asym, 0.5)
        act = r["action"] if r["action"] == "abstain" else LABELS[r["action"]]
        print(f"  {name:10s} -> {act:24s} (cost {r['best_cost']:.3f}, "
              f"margin {r['margin']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
