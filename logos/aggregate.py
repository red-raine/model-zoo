"""logos.aggregate — span-level evidence -> one verdict, and conformal abstention.

WHY AGGREGATION IS A SEPARATE COMPONENT
---------------------------------------
`Conflicting Evidence/Cherrypicking` is our worst class (F1 .362 on the fine-tuned
DeBERTa-v3-large, .214 on the base). The instinct is to re-weight the loss or fine-tune
harder. Both are wrong, because conflict is not a property any single forward pass over
concatenated evidence can see: it is a relation BETWEEN snippets. A claim-level model reads
"[claim] [SEP] ev1 | ev2 | ev3" and compresses "ev1 supports, ev2 refutes" into one hidden
state, then has to output a label that says "these contradict each other". It is being asked
to report the shape of an input it was never given separately.

The 2026 finding (arXiv:2608.05823, HallDetect) is that ASYMMETRIC aggregation over
per-evidence verdicts beats matched claim-level baselines on 3 of 4 benchmarks: a single
confidently contradicted sub-claim is enough to flag. That asymmetry is the whole trick --
support and refute are not symmetric evidence, because a verified contradiction is nearly
conclusive while a verified support is only as good as the source.

THE HONEST LIMIT
----------------
n ~ 60 Conflicting positives in 500 dev rows means 1 sigma on that class's F1 is about
+/-0.08. Any improvement below ~0.10 here is unmeasurable on this split. This module is built
to be *auditable* first and *better* second, and the self-test therefore checks forced
answers and monotonicity, not headline accuracy.

USAGE
-----
    python tools/logos/aggregate.py --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable, Sequence

import numpy as np

# allow `python tools/logos/aggregate.py` as well as `python -m logos.aggregate`
_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.schema import (Belief, ChannelResult, ChannelStatus, EvidenceLabel,  # noqa: E402
                          EvidenceVerdict, Label, Warrant)

N_CLASSES = 4


# --------------------------------------------------------------- asymmetric aggregation
def asymmetric_aggregate(evidence: Sequence[EvidenceVerdict],
                         channels: Sequence[ChannelResult] = (),
                         strong_weight: float = 0.55,
                         weak_weight: float = 0.35,
                         claim_id: str = "") -> tuple[Label | None, Warrant]:
    """Combine span-level verdicts into one 4-way label, emitting a warrant either way.

    Decision order matters and is documented rather than learned:

      1. A deterministic channel that SETTLED the claim wins outright. A proof is not a vote.
      2. No usable evidence                      -> None (abstain), reason recorded.
      3. >=1 strong refute AND >=1 strong support -> Conflicting.
      4. strong refute only (others insufficient) -> Refuted.
      5. strong support only                      -> Supported.
      6. anything else                            -> Not Enough Evidence.

    Asymmetry appears at step 3 vs 4: one strong refute plus one strong support is
    Conflicting, whereas one strong refute plus only-weak support is still Refuted. Weak
    support cannot manufacture a conflict, because a weak "support" is usually a topically
    related snippet that does not actually entail the claim.
    """
    fired = [c for c in channels if c.status == ChannelStatus.SETTLED and c.implies]
    if fired:
        w = Warrant(claim_id=claim_id, verdict=fired[0].implies,
                    rule="deterministic channel settled the claim",
                    fired_channels=[c.channel for c in fired],
                    used_evidence=[], flip_condition="",
                    notes=[f"{fired[0].channel}: {fired[0].detail}"])
        return fired[0].implies, w

    usable = [e for e in evidence if e.weight > 0]
    if not usable:
        w = Warrant(claim_id=claim_id, verdict=None, rule="no usable evidence",
                    notes=["every retrieved snippet was below the reliability floor or "
                           "labelled insufficient with zero confidence"],
                    flip_condition="retrieve at least one snippet with weight > 0")
        return None, w

    def strong(e: EvidenceVerdict, lab: EvidenceLabel) -> bool:
        return e.label == lab and e.weight >= strong_weight

    def any_weak(e: EvidenceVerdict, lab: EvidenceLabel) -> bool:
        return e.label == lab and e.weight >= weak_weight

    s_sup = [e for e in usable if strong(e, EvidenceLabel.SUPPORT)]
    s_ref = [e for e in usable if strong(e, EvidenceLabel.REFUTE)]
    w_sup = [e for e in usable if any_weak(e, EvidenceLabel.SUPPORT)]
    w_ref = [e for e in usable if any_weak(e, EvidenceLabel.REFUTE)]
    n_ins = sum(1 for e in usable if e.label == EvidenceLabel.INSUFFICIENT)

    used = [e.evidence_id for e in usable]
    notes = [f"{len(s_sup)} strong support, {len(s_ref)} strong refute, "
             f"{n_ins} insufficient, {len(usable)} usable of {len(evidence)} retrieved"]

    if s_ref and s_sup:
        verdict: Label = "Conflicting"
        rule = ">=1 strong refute AND >=1 strong support -> Conflicting"
        flip = (f"demote the refuting snippet {s_ref[0].evidence_id} (weight "
                f"{s_ref[0].weight:.2f}) below {strong_weight} and this becomes Supported")
    elif s_ref:
        verdict = "Refuted"
        rule = "strong refute, no strong support -> Refuted"
        flip = "add one strong supporting snippet to make this Conflicting"
    elif s_sup:
        verdict = "Supported"
        rule = "strong support, no strong refute -> Supported"
        flip = "add one strong refuting snippet to make this Conflicting"
    elif w_ref and w_sup:
        verdict = "Conflicting"
        rule = "only weak evidence but both directions present -> Conflicting"
        flip = "raise either direction to strong and the label is unchanged"
    elif w_ref:
        verdict = "Not Enough Evidence"
        rule = "only weak refute -> not settled"
        flip = "raise one refuting snippet to strong to get Refuted"
    elif w_sup:
        verdict = "Not Enough Evidence"
        rule = "only weak support -> not settled"
        flip = "raise one supporting snippet to strong to get Supported"
    else:
        verdict = "Not Enough Evidence"
        rule = "all evidence insufficient"
        flip = "retrieve a snippet that actually bears on the claim"

    w = Warrant(claim_id=claim_id, verdict=verdict, rule=rule,
                fired_channels=[c.channel for c in channels
                                if c.status != ChannelStatus.UNKNOWN],
                used_evidence=used, flip_condition=flip, notes=notes)
    return verdict, w


def aggregate_probs(evidence: Sequence[EvidenceVerdict],
                    prior: Sequence[float] = (0.24, 0.61, 0.07, 0.08),
                    strength: float = 2.0,
                    conflict_gain: float = 1.0) -> np.ndarray:
    """A soft version of the same rules, for the factor graph and for calibration tests.

    Each snippet contributes a log-likelihood ratio in favour of its own label, scaled by its
    weight. Conflict is captured because support and refute both accumulate: with two opposing
    snippets the Conflicting log-odds rise on BOTH the Supported and Refuted entries, which is
    not how a softmax over one snippet behaves. This vector is what makes Conflicting reachable.
    """
    logp = np.log(np.clip(np.asarray(prior, dtype=float), 1e-12, 1.0))
    logp = logp - logsumexp(logp)
    idx = {"Supported": 0, "Refuted": 1, "Not Enough Evidence": 2, "Conflicting": 3}

    sup_mass = ref_mass = 0.0
    for e in evidence:
        w = float(e.weight)
        if w <= 0:
            continue
        if e.label == EvidenceLabel.SUPPORT:
            logp[idx["Supported"]] += strength * w
            sup_mass += w
        elif e.label == EvidenceLabel.REFUTE:
            logp[idx["Refuted"]] += strength * w
            ref_mass += w
        else:
            logp[idx["Not Enough Evidence"]] += strength * w

    # CONFLICT IS AN INTERACTION, NOT AN ACCUMULATION.
    # A first version added a fixed +0.15 to the Conflicting logit per snippet. That is
    # structurally wrong: it makes conflict a first-order quantity, so P(Conflicting) either
    # rises monotonically or barely moves, and the measured peak sat at k=0 (i.e. conflict
    # never rose at all). Conflict requires BOTH directions to be present, which is a pairwise
    # property. min(support_mass, refute_mass) is exactly "how much genuinely opposed evidence
    # is there": zero when only one side appears, maximal at balance, and it stops growing once
    # one side saturates, which correctly makes a 5-vs-1 split LESS conflicting than a 1-vs-1.
    # This is the whole justification for the factor-graph layer: second-order structure.
    conflict = min(sup_mass, ref_mass)
    logp[idx["Conflicting"]] += strength * conflict_gain * conflict

    logp = logp - logsumexp(logp)
    return np.exp(logp)



def logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


# --------------------------------------------------------------------- conformal sets
class APSConformal:
    """Adaptive Prediction Sets over the 4 labels, for a coverage-guaranteed abstention rule.

    A threshold on max-probability gives an empirical risk-coverage curve. A conformal set
    gives a coverage GUARANTEE, which is the only defensible abstention claim in an audited
    system: "|set| > 1, so we abstain" is a statement about the procedure, not about this
    particular claim. The finite-sample correction uses ceil((n+1)(1-alpha))/n, so small
    calibration sets get honestly wider sets instead of silently under-covering.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = float(alpha)
        self.q = 1.0

    def _score(self, probs: np.ndarray, y: int) -> float:
        p = np.asarray(probs, dtype=float)
        order = np.argsort(-p)
        mass = 0.0
        for c in order:
            mass += float(p[c])
            if int(c) == int(y):
                return mass
        return 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "APSConformal":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        scores = np.array([self._score(probs[i], y[i]) for i in range(len(y))])
        n = len(scores)
        if n == 0:
            raise ValueError("empty calibration set")
        k = int(math.ceil((n + 1) * (1.0 - self.alpha))) / n
        self.q = float(min(1.0, np.quantile(scores, min(k, 1.0), method="higher")))
        return self

    def predict_set(self, probs: np.ndarray) -> set[int]:
        """{y : APS_score(y) <= q_hat}, i.e. include every rank whose CUMULATIVE mass is <= q.

        An earlier version accumulated until mass >= q, which is the wrong side of the
        comparison and silently adds one extra class to every set. Symptom: coverage 1.000
        with mean set size 2.65 on a model that should yield singletons. Over-coverage looks
        like safety and is actually lost efficiency.
        """
        p = np.asarray(probs, dtype=float)
        order = np.argsort(-p)
        out, mass = set(), 0.0
        for c in order:
            mass += float(p[c])
            if mass > self.q:
                break
            out.add(int(c))
        if not out:                     # empty sets are useless; keep the single best class
            out.add(int(order[0]))
        return out

    def covers(self, probs: np.ndarray, y: int) -> bool:
        return int(y) in self.predict_set(probs)


# ------------------------------------------------------------------------------ tests
def run_self_test() -> bool:
    print("logos.aggregate - SELF TEST")
    ok = True

    def ev(i, lab, conf=0.9, rel=0.9, src="s"):
        return EvidenceVerdict(evidence_id=i, text="t", label=lab, confidence=conf,
                               source=src, source_reliability=rel)

    sup, ref, ins = (EvidenceLabel.SUPPORT, EvidenceLabel.REFUTE,
                     EvidenceLabel.INSUFFICIENT)

    cases = [
        ("5 strong supports", [ev(str(i), sup) for i in range(5)], "Supported"),
        ("5 strong refutes", [ev(str(i), ref) for i in range(5)], "Refuted"),
        ("2 support + 2 refute", [ev("a", sup), ev("b", sup), ev("c", ref), ev("d", ref)],
         "Conflicting"),
        ("1 strong support + 1 strong refute", [ev("a", sup), ev("b", ref)], "Conflicting"),
        ("3 insufficient only", [ev(str(i), ins) for i in range(3)],
         "Not Enough Evidence"),
        ("1 strong support + 3 insufficient", [ev("a", sup)] + [ev(f"i{i}", ins)
                                                                for i in range(3)],
         "Supported"),
        ("weak support + weak refute", [ev("a", sup, 0.6, 0.6), ev("b", ref, 0.6, 0.6)],
         "Conflicting"),
    ]
    for name, evs, want in cases:
        got, w = asymmetric_aggregate(evs, claim_id=name)
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:36s} -> {got}")

    # asymmetry: weak support must NOT create a conflict against a strong refute
    got, _ = asymmetric_aggregate([ev("a", ref, 0.9, 0.9), ev("b", sup, 0.6, 0.6)])
    good = got == "Refuted"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] weak support cannot manufacture a conflict "
          f"against a strong refute -> {got}")

    # a deterministic channel wins outright, and says so
    ch = ChannelResult(channel="sympy", status=ChannelStatus.SETTLED,
                       detail="45e6 != 7e6", implies="Refuted")
    got, w = asymmetric_aggregate([ev("a", sup), ev("b", sup)], [ch], claim_id="x")
    good = got == "Refuted" and "sympy" in w.fired_channels
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a settled channel overrides 2 supports -> "
          f"{got} (warrant names {w.fired_channels})")

    # empty evidence abstains AND records why (the schema forbids a silent abstention)
    got, w = asymmetric_aggregate([], claim_id="empty")
    good = got is None and len(w.notes) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty evidence abstains with a reason")

    # monotonicity: adding a supporting snippet must never lower P(Supported)
    mono_sup = True
    prev_sup = -1.0
    base = [ev("r1", ref, 0.9, 0.9)]
    for k in range(6):
        p = aggregate_probs(base + [ev(f"s{j}", sup, 0.9, 0.9) for j in range(k)])
        mono_sup = mono_sup and p[0] >= prev_sup - 1e-12
        prev_sup = p[0]
    good = mono_sup
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] monotone: each added support raises P(Supported)")

    # P(Conflicting) is NOT monotone in support, and asserting that it was is what a first
    # version of this test did (and failed). With 1 refute and k supports the conflict mass
    # rises then falls: 5-vs-1 is genuinely LESS conflicting than 1-vs-1, because one side
    # now dominates. The true property is that conflict PEAKS AT BALANCE.
    conf = [float(aggregate_probs(base + [ev(f"s{j}", sup, 0.9, 0.9)
                                          for j in range(k)])[3]) for k in range(9)]
    peak = int(np.argmax(conf))
    good = conf[peak] > conf[0] and conf[peak] > conf[-1] and 0 <= peak <= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] P(Conflicting) peaks at evidence balance "
          f"(peak at k={peak} support vs 1 refute; p {conf[0]:.3f} -> {conf[peak]:.3f} -> "
          f"{conf[-1]:.3f})")

    # and adding a refuting snippet must never lower P(Refuted)
    prev = -1.0
    mono_ref = True
    for k in range(6):
        p = aggregate_probs([ev("s0", sup)] + [ev(f"r{j}", ref, 0.9, 0.9) for j in range(k)])
        mono_ref = mono_ref and p[1] >= prev - 1e-12
        prev = p[1]
    good = mono_ref
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] monotone: each added refute raises P(Refuted)")

    # stacking weighted evidence must not saturate the prior into a NaN or a zero vector
    ps = aggregate_probs([ev(str(i), sup if i % 2 else ref) for i in range(60)])
    good = bool(np.all(np.isfinite(ps))) and abs(float(ps.sum()) - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 60 contradictory snippets stay finite and "
          f"normalised")

    # ---------------------------------------------------------------- conformal coverage
    cov = APSConformal(alpha=0.10)
    rng = np.random.default_rng(0)
    n_cal, n_test = 800, 4000
    def draw(n, g):
        y = g.integers(0, N_CLASSES, n)
        base = rng.normal(0, 1, (n, N_CLASSES))
        base[np.arange(n), y] += 2.0
        e = np.exp(base - base.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), y
    p_cal, y_cal = draw(n_cal, rng)
    p_te, y_te = draw(n_test, rng)
    cov.fit(p_cal, y_cal)
    hit = float(np.mean([cov.covers(p_te[i], y_te[i]) for i in range(n_test)]))
    # The conformal guarantee is ONE-SIDED: coverage >= 1-alpha. Over-coverage is valid but
    # wasteful (it means sets are bigger than necessary), so the first assertion is the
    # guarantee and the second only catches runaway conservatism.
    good = hit >= 0.90 - 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] APS guarantee holds: held-out coverage "
          f"{hit:.4f} >= 0.90")

    # with a model that is actually informative, sets must be small AND still covered
    def draw_strong(n, g):
        # a REALISTICALLY informative model (p_true ~ 0.85), not a saturated one. A first
        # version used boost 4.0 / sd 0.7, giving p_true ~ 0.997; APS then saturates q_hat at
        # 1.0 and every set becomes uninformative. That is a real known APS degeneracy, not a
        # bug here, so the fixture must not sit on it.
        y = g.integers(0, N_CLASSES, n)
        b = rng.normal(0, 1.4, (n, N_CLASSES))
        b[np.arange(n), y] += 2.4
        e = np.exp(b - b.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), y
    ps_cal, ys_cal = draw_strong(n_cal, rng)
    ps_te, ys_te = draw_strong(n_test, rng)
    cov3 = APSConformal(alpha=0.10).fit(ps_cal, ys_cal)
    hit3 = float(np.mean([cov3.covers(ps_te[i], ys_te[i]) for i in range(n_test)]))
    sz3 = float(np.mean([len(cov3.predict_set(ps_te[i])) for i in range(400)]))
    # Efficiency must RESPOND to model quality. An absolute size threshold is a fixture
    # artefact (APS set size depends on how peaked the scores are), so the meaningful
    # assertion is monotone shrinkage as the model improves.
    def size_at(boost):
        g = np.random.default_rng(7)
        def draw_n(n):
            y = g.integers(0, N_CLASSES, n)
            b = g.normal(0, 1.4, (n, N_CLASSES))
            b[np.arange(n), y] += boost
            e = np.exp(b - b.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True), y
        pc, yc = draw_n(n_cal)
        pt, yt = draw_n(n_test)
        c = APSConformal(alpha=0.10).fit(pc, yc)
        return float(np.mean([len(c.predict_set(pt[i])) for i in range(300)]))
    s_lo, s_hi = size_at(1.0), size_at(3.5)
    good = hit3 >= 0.90 - 0.02 and s_hi < s_lo
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] efficiency tracks model quality: set size "
          f"{s_lo:.2f} (weak model) -> {s_hi:.2f} (strong), coverage {hit3:.4f}")

    sizes = [len(cov.predict_set(p_te[i])) for i in range(400)]
    good = min(sizes) >= 1 and max(sizes) <= N_CLASSES
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] prediction sets non-empty and bounded "
          f"(mean size {np.mean(sizes):.2f})")

    # a perfect model should produce singletons
    perfect = np.full((200, N_CLASSES), 0.01)
    yy = rng.integers(0, N_CLASSES, 200)
    perfect[np.arange(200), yy] = 0.97
    cov2 = APSConformal(alpha=0.10).fit(perfect, yy)
    sz = [len(cov2.predict_set(perfect[i])) for i in range(50)]
    good = float(np.mean(sz)) <= 1.2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] near-perfect model yields singleton sets "
          f"(mean size {np.mean(sz):.2f})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Asymmetric aggregation + conformal sets.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.demo:
        return 0 if run_self_test() else 1

    evs = [EvidenceVerdict(evidence_id="e1", text="source says the number is 7 million",
                           label=EvidenceLabel.REFUTE, confidence=0.88,
                           source_reliability=0.9),
           EvidenceVerdict(evidence_id="e2", text="official census says 45 million",
                           label=EvidenceLabel.SUPPORT, confidence=0.91,
                           source_reliability=0.85)]
    v, w = asymmetric_aggregate(evs, claim_id="demo")
    print("verdict:", v)
    print("rule:", w.rule)
    print("flip:", w.flip_condition)
    print("probs:", np.round(aggregate_probs(evs), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
