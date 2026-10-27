"""logos/arbiter.py — a learned, capability-gated arbiter over competing methods.

THE DESIGN ERROR THIS REPLACES
------------------------------
I framed a decision model as the thing that would *replace* the router. That is exactly backwards.
The router is one opinion. What is missing is the layer that takes the fixed order, the classifier,
the ontology check, the router's pick, and the archive routes -- all of them -- scores **which one is
most likely right**, and is allowed to say *"all of these look suspect, go search or ask an LLM."*

It must also be allowed to say **"the truth cannot be known here."** That is a different sentence
from "I abstain": abstaining is a decision about cost, unknowability is a property of the claim. A
method that cannot address a claim kind does not become able to by being outvoted.

THE THREE STRUCTURAL RULES
--------------------------
1. **A capability gate, not a vote.** Each method declares which claim kinds it can address
   (`honesty.py`). A method that cannot address this claim kind gets **weight 0**, which in
   log-linear pooling means it contributes a multiplicative identity -- the *absence* of an opinion,
   not a neutral vote. Five incapable methods agreeing must never outvote one capable method.
   This is the rule that makes the ensemble honest, and it is the one a plain majority vote gets
   wrong.

2. **Fusion is log-linear pooling, never attention.** Measured earlier in this project: a softmax
   gate over the same evidence scored 0.900 and collapsed to 0.633, because it violates
   coordinate-wise monotonicity -- adding support for a verdict could *lower* that verdict. Log-linear
   pooling has no such failure: more support never reduces the fused mass.

3. **Weights are LEARNED, not set.** The `aggregate` module was decided by a hand-set
   `strong_weight=0.55`, which is how a pipeline quietly becomes a hard-coded opinion. Here every
   `(method, claim_kind)` pair carries a `Beta` posterior over "this method is right about this kind
   of claim", updated from outcomes. It starts ignorant and earns its weight.

WHAT IT RETURNS
---------------
    keep        a value clears the confidence bar, and it is the fused winner
    escalate    no value clears the bar, so the answer is "go find out" -- search, or ask an LLM
    unknowable  no capable method exists for this claim kind; nothing can settle it, and the fact
                should be removed from the set or rewritten into a claim that CAN be settled

`unknowable` and `escalate` are deliberately distinct. Escalation buys information. Unknowability
means there is nothing to buy -- more evidence of a kind nobody can evaluate changes nothing.

USAGE
    python tools/logos/arbiter.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

DEFAULT_STATE = os.path.join(os.path.expanduser("~"), ".cache", "logos_arbiter.json")

KEEP = "keep"
ESCALATE = "escalate"
UNKNOWABLE = "unknowable"

EPS = 1e-6


# ------------------------------------------------------------------------------- beta arm
@dataclass
class Reliability:
    """Beta posterior over 'this method is right about this claim kind'."""

    right: float = 1.0
    wrong: float = 1.0
    observations: int = 0

    @property
    def mean(self) -> float:
        return self.right / (self.right + self.wrong)

    @property
    def uncertainty(self) -> float:
        a, b = self.right, self.wrong
        t = a + b
        return (a * b / (t * t * (t + 1))) ** 0.5

    def observe(self, correct: bool) -> None:
        if correct:
            self.right += 1.0
        else:
            self.wrong += 1.0
        self.observations += 1


# ----------------------------------------------------------------------------- candidate
@dataclass
class Candidate:
    """One method's opinion. `claims` is the claim kinds it can address on THIS item.

    A method that was never able to address this claim kind should either not be listed at all, or
    be listed with `claims` empty -- both mean "no capability", and both yield weight 0.

    `answers` is a DIFFERENT question and the distinction is the whole reason this field exists:

        claims=False   "I cannot judge this kind of claim at all"   -> capability failure
        answers=False  "I can judge this kind, but my answer does not address it"  -> relevance
                       failure, e.g. a post-claim capture date offered as evidence of precedence

    Measured before this field existed: the cascade kept 15 dates of which only 8 actually settled
    the claim. Seven were post-claim capture dates that the methods agreed on. Agreement is not
    correctness, and a fusion that cannot tell "we all agree" from "we all answered the wrong
    question" will confidently assert the wrong question's answer.
    """

    method: str
    value: object
    confidence: float = 0.5
    claim_kind: str = ""
    claims: tuple[str, ...] = ()
    answers: bool = True

    @classmethod
    def from_probs(cls, method: str, probs, value: object | None = None,
                   claim_kind: str = "", claims: tuple[str, ...] = (),
                   answers: bool = True) -> "Candidate":
        """Build a candidate whose confidence comes from the FUSION array of a `calibrate.Probs`.

        Confidence here is consumed as a probability inside log-linear pooling, so it must be a
        probability that means what it says. The routing table sets `fusion -> calibrated`: the raw
        array is 4.23x overconfident (ECE 0.098), and feeding that in would make every method look more
        certain than it is, in proportion to how sharp its logits happen to be rather than how often it
        is right.

        `value` defaults to the argmax class, which is the common case; pass it explicitly when the
        method's answer is not simply the most likely class.
        """
        import numpy as _np

        from logos.calibrate import coerce as _coerce

        p = _coerce(probs, "fusion")
        p = _np.asarray(p, dtype=float)
        row = p if p.ndim == 1 else p[0]
        top = int(_np.argmax(row))
        return cls(method=method, value=top if value is None else value,
                   confidence=float(row[top]), claim_kind=claim_kind, claims=claims,
                   answers=answers)


@dataclass
class Decision:
    value: object = None
    fused: dict = field(default_factory=dict)
    disposition: str = UNKNOWABLE
    reason: str = ""
    weights: dict = field(default_factory=dict)
    capable: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {"value": self.value, "disposition": self.disposition, "reason": self.reason,
                "confidence": round(self.confidence, 4),
                "fused": {str(k): round(v, 4) for k, v in self.fused.items()},
                "weights": {k: round(v, 3) for k, v in self.weights.items()},
                "capable": self.capable, "excluded": self.excluded}


# -------------------------------------------------------------------------------- arbiter
class Arbiter:
    """Learned, capability-gated fusion over competing methods.

    `cost_discard` is the cost of dropping a claim that was actually settleable; `cost_wrong` is the
    cost of asserting a wrong verdict. The escalation threshold comes from those, not from taste:

        t* = 1 - cost_discard / cost_wrong

    which is Chow's rule. With the defaults (discard 1, wrong 4) it is 0.75: assert only when the
    fused mass clears three quarters, otherwise go and find out.
    """

    def __init__(self, state_path: str = DEFAULT_STATE, cost_discard: float = 1.0,
                 cost_wrong: float = 4.0, seed: int = 0, prior: float = 1.0) -> None:
        self.state_path = state_path
        self.cost_discard = cost_discard
        self.cost_wrong = cost_wrong
        self.prior = prior
        self.rng = random.Random(seed)
        self.arms: dict[str, dict[str, Reliability]] = {}
        self.load()

    # ------------------------------------------------------------------------ state
    def load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:                     # noqa: BLE001
            return
        for kind, arms in raw.get("arms", {}).items():
            self.arms[kind] = {m: Reliability(**v) for m, v in arms.items()}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump({"arms": {k: {m: a.__dict__ for m, a in arms.items()}
                                    for k, arms in self.arms.items()}}, fh, indent=1)
        except Exception:                     # noqa: BLE001
            pass

    def arm(self, claim_kind: str, method: str) -> Reliability:
        return self.arms.setdefault(claim_kind, {}).setdefault(
            method, Reliability(self.prior, self.prior))

    @property
    def threshold(self) -> float:
        if self.cost_wrong <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.cost_discard / self.cost_wrong))

    # ----------------------------------------------------------------------- decide
    def decide(self, candidates: list[Candidate], claim_kind: str = "") -> Decision:
        if not candidates:
            return Decision(disposition=UNKNOWABLE,
                            reason="no method offered an opinion; nothing to arbitrate")

        # --- RULE 1: the capability gate. Capability is a property of the METHOD, not of its
        # confidence. A method that cannot address this claim kind contributes weight 0.
        capable, excluded, weights, irrelevant = [], [], {}, []
        for c in candidates:
            knows = bool(c.claim_kind or c.claims)
            covers = (not c.claims) or (claim_kind in c.claims)
            if knows and not covers:
                excluded.append(c.method)
                weights[c.method] = 0.0
                continue
            if not knows:
                # Undeclared capability is treated as NO capability, not as universal capability.
                # Failing open here would let an unrelated method vote on anything it liked.
                excluded.append(c.method)
                weights[c.method] = 0.0
                continue
            if not c.answers:
                # capable, but the answer does not address the question posed. This is the case that
                # made the cascade assert 15 dates when only 8 settled the claim.
                irrelevant.append(c.method)
                weights[c.method] = 0.0
                continue
            capable.append(c.method)
            weights[c.method] = self.arm(claim_kind, c.method).mean

        if not capable and irrelevant:
            return Decision(disposition=UNKNOWABLE, weights=weights, capable=[],
                            excluded=excluded + irrelevant,
                            reason=(f"{len(irrelevant)} method(s) can judge {claim_kind!r} but "
                                    f"none of their answers addresses it; the claim cannot be "
                                    f"settled as stated -- remove it from the set or rewrite it "
                                    f"into a claim that these methods could settle"))

        if not capable:
            return Decision(disposition=UNKNOWABLE, weights=weights, capable=[], excluded=excluded,
                            reason=("no method declares capability for claim kind "
                                    f"{claim_kind!r}; more votes cannot supply a capability "
                                    f"nobody has -- remove or rewrite the claim"))

        # a capable method whose reliability has been driven to ~0 still cannot settle it
        live = [m for m in capable if weights.get(m, 0.0) > EPS]
        if not live:
            return Decision(disposition=UNKNOWABLE, weights=weights, capable=capable,
                            excluded=excluded,
                            reason="every capable method has been driven to zero reliability by "
                                   "its own record; the claim needs a new method, not more votes")

        # --- RULE 2: log-linear pooling over the candidate values.
        values: list = []
        for c in candidates:
            if c.method in capable and all(c.value != v for v in values):
                values.append(c.value)

        logp: dict = {}
        for v in values:
            total = 0.0
            for c in candidates:
                if c.method not in capable:
                    continue
                w = weights.get(c.method, 0.0)
                if w <= EPS:
                    continue
                conf = max(EPS, min(1.0 - EPS, float(c.confidence)))
                # a method's opinion as a distribution over the candidate values: `conf` on its own
                # value, the remainder spread over the others
                share = conf if c.value == v else (1.0 - conf) / max(1, len(values) - 1)
                total += w * _log(max(EPS, share))
            logp[v] = total

        m = max(logp.values()) if logp else 0.0
        exps = {v: _exp(lp - m) for v, lp in logp.items()}       # stabilised softmax
        z = sum(exps.values()) or 1.0
        fused = {v: e / z for v, e in exps.items()}

        best = max(fused, key=fused.get)
        top = fused[best]
        t = self.threshold

        # --- RULE 3 (disposition): assert / escalate / refuse.
        if top >= t:
            disp, why = KEEP, (f"fused mass {top:.3f} clears the escalation threshold {t:.2f} "
                               f"(Chow: 1 - {self.cost_discard:g}/{self.cost_wrong:g})")
        else:
            disp, why = ESCALATE, (f"fused mass {top:.3f} is below {t:.2f} -- the methods disagree "
                                   f"enough that the cheapest correct move is to go and find out")

        return Decision(value=best if disp == KEEP else None, fused=fused, disposition=disp,
                        reason=why, weights=weights, capable=capable, excluded=excluded,
                        confidence=top)

    # ------------------------------------------------------------------------ learn
    def observe(self, claim_kind: str, method: str, correct: bool) -> Reliability:
        """Tell the arbiter whether a method turned out to be right. This is the whole learning."""
        a = self.arm(claim_kind, method)
        a.observe(correct)
        return a

    def observe_decision(self, claim_kind: str, candidates: list[Candidate], truth) -> dict:
        """Update every CAPABLE method from one resolved decision.

        A method is scored on whether ITS OWN vote matched the truth -- not on whether the fused
        decision was right. Otherwise a strong method gets blamed for being outvoted, and the
        arbiter slowly learns to distrust exactly the opinions it should keep. Incapable methods are
        not updated at all: they were never asked, so they cannot be scored.
        """
        updated = {}
        for c in candidates:
            covers = (not c.claims) or (claim_kind in c.claims)
            if not (c.claim_kind or c.claims) or not covers:
                continue
            self.observe(claim_kind, c.method, bool(truth is not None and c.value == truth))
            updated[c.method] = self.arm(claim_kind, c.method).mean
        return updated

    def summary(self) -> dict:
        out = {}
        for kind, arms in self.arms.items():
            rows = sorted(((m, a.mean, a.observations) for m, a in arms.items()),
                          key=lambda t: -t[1])
            out[kind] = [{"method": m, "mean": round(v, 3), "n": n} for m, v, n in rows]
        return out


def _log(x: float) -> float:
    import math
    return math.log(x)


def _exp(x: float) -> float:
    import math
    return math.exp(x)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.arbiter - SELF TEST")
    ok = True
    import tempfile

    tmp = tempfile.mkdtemp()

    def fresh(n="a", **kw):
        return Arbiter(state_path=os.path.join(tmp, f"{n}.json"), **kw)

    A = fresh("a")

    # --- 1. THE CAPABILITY GATE. Five incapable methods agreeing must not outvote one capable one.
    cands = [Candidate(f"junk{i}", "WRONG", 0.99, "date_precedence", ("numeric",))
             for i in range(5)]
    cands.append(Candidate("ontology", "RIGHT", 0.60, "date_precedence", ("date_precedence",)))
    d = A.decide(cands, "date_precedence")
    good = d.value == "RIGHT" and set(d.excluded) == {f"junk{i}" for i in range(5)}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] five incapable methods at 0.99 do NOT outvote one "
          f"capable at 0.60 (winner={d.value})")

    # --- 2. UNKNOWABLE is not abstention. If nobody can judge the claim kind, voting is meaningless
    #     and the fact should be removed or rewritten -- more votes cannot supply a capability.
    d = A.decide([Candidate("x", "A", 0.99, "date_precedence", ("numeric",)),
                  Candidate("y", "A", 0.98, "date_precedence", ("numeric",))], "date_precedence")
    good = d.disposition == UNKNOWABLE and d.value is None and "rewrite" in d.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no capable method -> UNKNOWABLE, not a confident "
          f"guess ({d.fused})")

    # --- 3. undeclared capability fails CLOSED, never open
    d = A.decide([Candidate("undeclared", "A", 0.99, "", ())], "date_precedence")
    good = d.disposition == UNKNOWABLE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a method that declares no capability gets weight 0 "
          f"(fails closed)")

    # --- 4. RULE 2: coordinate-wise monotonicity. Adding support for a value must never reduce it.
    #     This is the property a softmax attention gate violated (0.900 -> 0.633) in this project.
    base = [Candidate("a", "V", 0.6, "k", ("k",)), Candidate("b", "W", 0.5, "k", ("k",))]
    d1 = A.decide(base, "k")
    boosted = [Candidate("a", "V", 0.9, "k", ("k",)), Candidate("b", "W", 0.5, "k", ("k",))]
    d2 = A.decide(boosted, "k")
    good = (d2.fused.get("V", 0) >= d1.fused.get("V", 0) - 1e-12
            and d2.fused.get("W", 0) <= d1.fused.get("W", 0) + 1e-12)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] monotone: more support for V never lowers V "
          f"({d1.fused.get('V', 0):.3f} -> {d2.fused.get('V', 0):.3f})")

    # --- 5. RULE 3: escalation threshold comes from costs (Chow), not taste
    good = (abs(fresh("t1", cost_discard=1, cost_wrong=4).threshold - 0.75) < 1e-12
            and abs(fresh("t2", cost_discard=1, cost_wrong=2).threshold - 0.5) < 1e-12
            and abs(fresh("t3", cost_discard=0, cost_wrong=4).threshold - 1.0) < 1e-12)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] threshold = 1 - C_d/C_w (0.75, 0.50, 1.00 for "
          f"1/4, 1/2, 0/4)")

    # --- 6. disagreement escalates rather than guessing
    A2 = fresh("b")
    d = A2.decide([Candidate("m1", "X", 0.55, "k", ("k",)),
                   Candidate("m2", "Y", 0.55, "k", ("k",))], "k")
    good = d.disposition == ESCALATE and d.value is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a close split escalates instead of guessing "
          f"(top={d.confidence:.3f} < {A2.threshold:.2f})")

    # --- 7. and agreement keeps, with a clear winner
    d = A2.decide([Candidate("m1", "X", 0.95, "k", ("k",)),
                   Candidate("m2", "X", 0.90, "k", ("k",))], "k")
    good = d.disposition == KEEP and d.value == "X"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] agreement clears the bar and keeps "
          f"(top={d.confidence:.3f})")

    # --- 8. LEARNED WEIGHTS: a method that is repeatedly wrong loses its say.
    #     Note what is and is not claimed. Learning flips the PREFERENCE onto the method with a
    #     record (weights 0.97 vs 0.03). It does not by itself clear the 0.75 assertion bar, because
    #     the trusted method is only 0.55-confident on this item -- so the honest outcome is a
    #     flipped preference AND an escalation. Asserting here would be over-reach.
    A3 = fresh("c")
    for _ in range(30):
        A3.observe("k", "bad", False)
        A3.observe("k", "good", True)
    d = A3.decide([Candidate("bad", "WRONG", 0.99, "k", ("k",)),
                   Candidate("good", "RIGHT", 0.55, "k", ("k",))], "k")
    good = (d.fused.get("RIGHT", 0) > d.fused.get("WRONG", 0)
            and d.weights["good"] > d.weights["bad"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] learned weights flip the preference onto the method "
          f"with a record ({d.weights['good']:.2f} vs {d.weights['bad']:.2f}; "
          f"fused RIGHT={d.fused.get('RIGHT', 0):.3f} > WRONG={d.fused.get('WRONG', 0):.3f})")

    # --- 8b. and once the trusted method is confident enough, the same item is asserted
    d = A3.decide([Candidate("bad", "WRONG", 0.99, "k", ("k",)),
                   Candidate("good", "RIGHT", 0.97, "k", ("k",))], "k")
    good = d.disposition == KEEP and d.value == "RIGHT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ...and at 0.97 confidence the same item clears the "
          f"bar and is kept")

    # --- 8c. impossible truth is not credited: a method cannot be scored against None
    A3b = fresh("c2")
    upd = A3b.observe_decision("k", [Candidate("m", "X", 0.9, "k", ("k",))], None)
    good = A3b.arm("k", "m").wrong == 2.0 and "m" in upd
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a vote against unknown truth counts as wrong, never "
          f"as right")

    # --- 17. ROUTING: a candidate built from a Probs takes the FUSION array, which the table sets to
    #        calibrated. Raw is 4.23x overconfident, so a candidate built from raw would look more
    #        certain than it is -- and that would propagate silently through log-linear pooling.
    import numpy as _np

    from logos.calibrate import Probs as _Probs
    peaked = _np.array([0.95, 0.03, 0.01, 0.01])
    flat = _np.array([0.45, 0.25, 0.20, 0.10])
    pr = _Probs(raw=peaked, calibrated=flat, temperature=4.23)
    c_cal = Candidate.from_probs("m", pr, claim_kind="k", claims=("k",))
    good = abs(c_cal.confidence - 0.45) < 1e-12 and c_cal.value == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Candidate.from_probs takes the CALIBRATED array "
          f"(confidence {c_cal.confidence:.2f}, not the raw {peaked[0]:.2f})")

    # --- 17b. and that difference actually moves the FUSION. Two candidates are required: with one,
    #         log-linear pooling normalises to 1.0 regardless of the input confidence, so a
    #         single-candidate test would assert nothing. (I wrote it that way first.)
    pr_b = _Probs(raw=_np.array([0.30, 0.60, 0.05, 0.05]),
                  calibrated=_np.array([0.20, 0.40, 0.25, 0.15]), temperature=4.23)
    A9 = fresh("g2")
    d_cal = A9.decide([Candidate.from_probs("A", pr, claim_kind="k", claims=("k",)),
                       Candidate.from_probs("B", pr_b, claim_kind="k", claims=("k",))], "k")
    d_raw = A9.decide([Candidate("A", 0, 0.95, "k", ("k",)),
                       Candidate("B", 1, 0.60, "k", ("k",))], "k")
    diff = abs(d_cal.fused.get(0, 0) - d_raw.fused.get(0, 0))
    good = diff > 0.05
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] raw confidence would shift the fused mass by "
          f"{diff:.3f} ({d_cal.fused.get(0, 0):.3f} calibrated vs {d_raw.fused.get(0, 0):.3f} raw)")

    # --- 9. weights are NOT hand-set: they start at 0.5 and move only from observations
    A4 = fresh("d")
    w0 = A4.arm("k", "m").mean
    A4.observe("k", "m", True)
    w1 = A4.arm("k", "m").mean
    good = abs(w0 - 0.5) < 1e-12 and w1 > w0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] weights start ignorant (0.50) and move only from "
          f"outcomes (-> {w1:.3f})")

    # --- 10. reliability is per (method, claim kind), so a method good at one thing is not
    #      credited for another
    A5 = fresh("e")
    for _ in range(20):
        A5.observe("numeric", "calc", True)
        A5.observe("date_precedence", "calc", False)
    good = (A5.arm("numeric", "calc").mean > 0.9
            and A5.arm("date_precedence", "calc").mean < 0.1)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] capability is per claim kind, not per method "
          f"(numeric {A5.arm('numeric', 'calc').mean:.2f}, "
          f"date {A5.arm('date_precedence', 'calc').mean:.2f})")

    # --- 11. determinism: same inputs, same answer
    d1 = A2.decide([Candidate("m1", "X", 0.8, "k", ("k",))], "k")
    d2 = A2.decide([Candidate("m1", "X", 0.8, "k", ("k",))], "k")
    good = d1.value == d2.value and d1.fused == d2.fused
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] deterministic: identical inputs give identical fusion")

    # --- 12. zero confidence carries no information
    d = A2.decide([Candidate("m1", "X", 0.0, "k", ("k",)),
                   Candidate("m2", "Y", 0.9, "k", ("k",))], "k")
    good = d.value == "Y"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a zero-confidence method does not drag the fusion")

    # --- 13. persistence: it keeps learning across processes
    A3.save()
    A6 = Arbiter(state_path=os.path.join(tmp, "c.json"))
    good = abs(A6.arm("k", "good").mean - A3.arm("k", "good").mean) < 1e-12
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] posteriors persist across processes")

    # --- 14. the four methods the user named must be arbitrable together
    A7 = fresh("f")
    cands = [Candidate("fixed_order", "2016-08-07", 0.55, "date_precedence", ("date_precedence",)),
             Candidate("classifier", "2017-01-01", 0.61, "date_precedence", ("date_precedence",)),
             Candidate("ontology", "2016-08-07", 0.70, "date_precedence", ("date_precedence",)),
             Candidate("router", "2025-05-31", 0.65, "date_precedence", ("date_precedence",),
                       answers=False)]
    d = A7.decide(cands, "date_precedence")
    good = d.disposition in (KEEP, ESCALATE) and d.capable == ["fixed_order", "classifier",
                                                              "ontology"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] fixed_order + classifier + ontology + router all "
          f"arbitrate together ({d.disposition}, winner={d.value})")

    # --- 15. CAPABLE BUT IRRELEVANT. All methods can judge the claim kind, and they all agree --
    #      but none of their answers addresses the question. Unanimity must not rescue it.
    #      Measured before this rule: the cascade kept 15 dates and only 8 settled the claim.
    A8 = fresh("g")
    d = A8.decide([Candidate("wayback", "2025-05-31", 0.95, "date_precedence",
                             ("date_precedence",), answers=False),
                   Candidate("arquivo", "2024-01-01", 0.90, "date_precedence",
                             ("date_precedence",), answers=False)], "date_precedence")
    good = d.disposition == UNKNOWABLE and d.value is None and "rewrite" in d.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unanimous but IRRELEVANT -> unknowable, not kept "
          f"(the 15-vs-8 bug)")

    # --- 16. one relevant answer beats several irrelevant ones
    d = A8.decide([Candidate("wayback", "2025-05-31", 0.99, "date_precedence",
                             ("date_precedence",), answers=False),
                   Candidate("arquivo", "2016-08-07", 0.55, "date_precedence",
                             ("date_precedence",))], "date_precedence")
    good = d.value == "2016-08-07" and d.capable == ["arquivo"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one relevant answer outweighs several irrelevant "
          f"ones (winner={d.value})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Learned capability-gated arbiter over methods.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.summary:
        return 0 if run_self_test() else 1
    A = Arbiter()
    s = A.summary()
    if not s:
        print("no observations yet -- the arbiter is cold (every method at 0.50)")
    for kind, rows in s.items():
        print(f"\n{kind}:")
        for r in rows:
            print(f"  {r['method']:20s} mean={r['mean']:.3f}  n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
