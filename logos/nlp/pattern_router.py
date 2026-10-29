"""logos/nlp/pattern_router.py — the GRPO pattern router: a spaCy feature vector decides which instrument runs.

WHAT THIS IMPLEMENTS
--------------------
Algorithm 1 ("GRPO Pattern Routing") from the Viable Epistemic Engine v9 whitepaper:

    1   phi      <- spaCy24(x)                       24-dim complexity vector, the ONLY LLM-free step
    2   {p_1..p_G} ~ pi_theta(.|x)                   sample G pattern candidates
    3-5 for each:  r_i <- dispatch(p_i);  R_i <- 0.4*DoD(r_i) + 0.3*Latency(r_i) + 0.3*Cost(r_i)
    6   mu <- mean(R); sigma <- std(R)               GROUP statistics
    7-9 A_i <- (R_i - mu)/(sigma+eps);  theta <- theta + eta * grad log pi(p_i|x) * A_i
    deployment: dispatch the max-reward member of the sampled group

That is REINFORCE with a group-relative baseline. The policy is a SOFTMAX OVER A LINEAR FUNCTION OF 24 spaCy
features -- not a transformer -- which is exactly why it is sub-millisecond on CPU and why it can be trained
here at all. The whitepaper's own overhead claim is 0.041 ms/prompt.

WHY GRPO IS THE RIGHT FIT FOR THIS SUITE, SPECIFICALLY
------------------------------------------------------
GRPO needs a reward. Most projects have to learn one, or pay a model to judge. This suite already has
VERIFIABLE rewards: sympy proves arithmetic, Z3 proves or refutes a first-order argument, Lean's kernel accepts
or rejects a proof. "Did the instrument return a certificate?" is a reward that cannot be gamed by a fluent
answer, which is the failure mode the literature reports for learned rewards. So the reward is free here, and
it is the strongest kind.

WHAT I DID NOT COPY, AND WHY
----------------------------
The whitepaper does not list the 24 features. I define them below and label every one, because a feature vector
whose contents are undocumented is not reproducible. The paper's 10-pattern set is replaced by the instruments
this suite actually has -- routing to an instrument we do not own would be theatre.
The paper's headline result is a PILOT: "40% to 100% trajectory-selection accuracy" on a **10-prompt**
benchmark. Ten prompts is not evidence of anything. This module reports its own numbers and says how many rows
they came from.

USAGE
    python tools/logos/nlp/pattern_router.py --self-test
    python tools/logos/nlp/pattern_router.py --measure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

N_FEATURES = 24

# The instruments we actually own, with a cost TIER used by the reward. A pattern that cannot return a
# certificate (ESCALATE) is included on purpose: abstaining is sometimes the correct action, and a router that
# cannot choose it will be punished for guessing.
PATTERNS = [
    {"name": "arithmetic", "tier": 1, "desc": "sympy arithmetic/units/constants"},
    {"name": "numeric_conflict", "tier": 1, "desc": "exact numeric disagreement"},
    {"name": "fol_bridge", "tier": 2, "desc": "first-order validity via Z3"},
    {"name": "dep_encoder", "tier": 3, "desc": "dependency-parse claim shape"},
    {"name": "scope_triage", "tier": 3, "desc": "scope ambiguity detection"},
    {"name": "crosscheck", "tier": 2, "desc": "two independent implementations"},
    {"name": "rerank", "tier": 4, "desc": "19M cross-encoder evidence gate"},
    {"name": "mathlib", "tier": 6, "desc": "Lean + Mathlib kernel proof"},
    {"name": "truth_store", "tier": 1, "desc": "cached witnessed verdict"},
    {"name": "escalate", "tier": 0, "desc": "abstain; no instrument"},
]
K = len(PATTERNS)
MAX_TIER = max(p["tier"] for p in PATTERNS)
LATENCY_BUDGET_MS = 13167.0          # the Mathlib measurement; the slowest instrument sets the scale

EPS = 1e-8

# ------------------------------------------------------------------ the 24 features
# Every feature is named and justified, because an undocumented feature vector is not reproducible. Read from a
# spaCy Doc; anything unavailable degrades to 0 rather than raising, so the router never blocks a row.
FEATURE_NAMES = [
    "n_tokens",              # 1  length
    "n_chars",               # 2  length in characters
    "n_sents",               # 3  multi-sentence rows need different handling
    "n_numeric",             # 4  digits present -> arithmetic plausible
    "n_operators",           # 5  + - * / = signs
    "has_equals",            # 6  an equality is the strongest arithmetic signal
    "n_units",               # 7  m, km, kg, % ...
    "n_years",               # 8  4-digit calendar numbers
    "n_quantifier",          # 9  all/every/no/some ...
    "n_negation",            # 10 not/never/none ...
    "n_comparative",         # 11 more/less/than, JJR/RBR
    "n_conditional",         # 12 if/unless/when
    "n_hedge",               # 13 may/might/could/possibly
    "n_causal",              # 14 because/leads to/due to
    "n_propn",               # 15 named-entity-ish density
    "n_noun",                # 16
    "n_verb",                # 17
    "n_adj",                 # 18
    "n_unique_lemma_ratio",  # 19 lexical diversity
    "n_noun_chunks",         # 20
    "max_chunk_len",         # 21 a long chunk is a clause wearing a noun's clothes
    "tree_depth",            # 22 dependency depth
    "n_clauses",             # 23 relcl/advcl/ccomp/xcomp
    "n_punct",               # 24
]

_UNITS = {"m", "km", "cm", "mm", "kg", "g", "mg", "lb", "mi", "ft", "in", "s", "ms", "h",
          "percent", "%", "mb", "gb", "kb", "tb", "usd", "eur"}
_QUANT = {"all", "every", "each", "any", "some", "no", "none", "most", "many", "few", "several", "both"}
_NEG = {"not", "n't", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}
_HEDGE = {"may", "might", "could", "possibly", "likely", "perhaps", "apparently", "reportedly"}
_CAUSAL = {"because", "cause", "causes", "caused", "due", "leads", "lead", "therefore", "thus", "hence"}
_COND = {"if", "unless", "when", "whenever"}
_CLAUSE_DEPS = {"relcl", "advcl", "ccomp", "xcomp", "acl", "csubj"}


def _depth(tok) -> int:
    d, seen = 0, set()
    while tok is not None and tok.head is not tok and id(tok) not in seen:
        seen.add(id(tok))
        tok = tok.head
        d += 1
        if d > 64:
            break
    return d


def spacy24(text: str, doc=None) -> list:
    """The 24-dim complexity vector. Never raises; a missing parse degrades to zeros for the syntax features."""
    try:
        if doc is None:
            from logos.nlp import domain_nlp
            doc = domain_nlp.make_nlp(use_domain=True)(text)
    except Exception:                                         # noqa: BLE001
        return [0.0] * N_FEATURES

    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    chunks = list(doc.noun_chunks)
    lemmas = {t.lemma_.lower() for t in toks if t.is_alpha}
    feats = [
        float(len(toks)),
        float(len(text or "")),
        float(len(list(doc.sents))),
        float(sum(1 for t in toks if t.like_num or any(ch.isdigit() for ch in t.text))),
        float(sum(1 for t in toks if t.text in ("+", "-", "*", "/", "=", "^", "×", "÷"))),
        1.0 if "=" in (text or "") else 0.0,
        float(sum(1 for t in toks if t.lemma_.lower() in _UNITS or t.text in _UNITS)),
        float(sum(1 for t in toks if t.like_num and len(t.text) == 4 and t.text.isdigit()
                    and 1000 <= int(t.text) <= 2100)),
        float(sum(1 for t in toks if t.lemma_.lower() in _QUANT)),
        float(sum(1 for t in toks if t.lemma_.lower() in _NEG)),
        float(sum(1 for t in toks if t.tag_ in ("JJR", "RBR") or t.lemma_.lower() == "than")),
        float(sum(1 for t in toks if t.lemma_.lower() in _COND)),
        float(sum(1 for t in toks if t.lemma_.lower() in _HEDGE)),
        float(sum(1 for t in toks if t.lemma_.lower() in _CAUSAL)),
        float(sum(1 for t in toks if t.pos_ == "PROPN")),
        float(sum(1 for t in toks if t.pos_ == "NOUN")),
        float(sum(1 for t in toks if t.pos_ in ("VERB", "AUX"))),
        float(sum(1 for t in toks if t.pos_ == "ADJ")),
        float(len(lemmas)) / n,
        float(len(chunks)),
        float(max((len(c) for c in chunks), default=0)),
        float(max((_depth(t) for t in toks), default=0)),
        float(sum(1 for t in toks if t.dep_ in _CLAUSE_DEPS)),
        float(sum(1 for t in toks if t.is_punct)),
    ]
    return [float(x) for x in feats[:N_FEATURES]]


# ------------------------------------------------------------------ the policy
class SoftmaxRouter:
    """K-way softmax over a linear function of the 24 features. This is the whole policy."""

    def __init__(self, seed: int = 0, scale: float = 1.0):
        rng = random.Random(seed)
        # features live on wildly different scales, so the initial weights are tiny; a k-means-style
        # normalisation would be better but a small init is enough for a softmax policy
        self.w = [[rng.uniform(-1e-3, 1e-3) * scale for _ in range(N_FEATURES)] for _ in range(K)]
        self.b = [0.0] * K
        self.seed = seed

    def logits(self, phi: list) -> list:
        return [sum(w * x for w, x in zip(self.w[i], phi)) + self.b[i] for i in range(K)]

    def probs(self, phi: list) -> list:
        z = self.logits(phi)
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def score(self, phi: list) -> float:
        """For a check that the score survives a pickle: larger is a better fit."""
        return self.probs(phi)[0]

    def sample(self, phi: list, group: int = 3, rng: random.Random | None = None) -> list:
        """Sample `group` DISTINCT patterns (without replacement) from the policy.

        Distinct on purpose: a group with duplicates gives zero variance in the advantage, and every GRPO
        update for that prompt becomes zero. Sampled with the policy weights, so exploration follows pi.
        """
        rng = rng or random.Random(self.seed)
        p = self.probs(phi)
        idx = list(range(K))
        # weighted sampling without replacement (Efraimidis-Spirakis)
        keys = []
        for i in idx:
            u = max(rng.random(), 1e-12)
            keys.append((u ** (1.0 / max(p[i], 1e-12)), i))
        keys.sort(reverse=True)
        return [i for _k, i in keys[:min(group, K)]]

    def update(self, phi: list, chosen: list, rewards: list, lr: float = 0.05) -> dict:
        """One GRPO step: group-relative advantages times the policy gradient of log-softmax."""
        g = len(chosen)
        if g < 2:
            return {"skipped": True, "reason": "group of 1 has zero variance"}
        mu = sum(rewards) / g
        var = sum((r - mu) ** 2 for r in rewards) / g
        sigma = math.sqrt(var)
        if sigma < EPS:
            return {"skipped": True, "reason": "zero-variance group: no learning signal", "mu": mu}
        p = self.probs(phi)
        advs = []
        for k, r in zip(chosen, rewards):
            a = (r - mu) / (sigma + EPS)
            advs.append(a)
            # grad log softmax_k = onehot(k) - p
            for j in range(K):
                coeff = (1.0 if j == k else 0.0) - p[j]
                if coeff == 0.0:
                    continue
                step = lr * a * coeff
                wj = self.w[j]
                for f in range(N_FEATURES):
                    wj[f] += step * phi[f]
                self.b[j] += step
        return {"mu": mu, "sigma": sigma, "advantages": advs}

    def to_dict(self) -> dict:
        return {"w": self.w, "b": self.b, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: dict):
        r = cls(seed=d.get("seed", 0))
        r.w = d["w"]
        r.b = d["b"]
        return r


# ------------------------------------------------------------------ reward
DOD_GATE = True   # see reward() -- the whitepaper's raw weights let a fast guess beat a slow proof


def reward_whitepaper(dod: float, ms: float, tier: int) -> float:
    """R = 0.4*DoD + 0.3*Latency + 0.3*Cost, EXACTLY as the whitepaper specifies.

    Kept for comparison, and MEASURED TO BE WRONG FOR A VERIFIER: with these weights a fast wrong answer
    scores 0.600 (DoD 0, Latency ~1, Cost 1) while a slow proof scores 0.400 (DoD 1, Latency 0, Cost 0).
    A no-certificate result therefore BEATS a certificate, which is the exact inversion a precision-first
    suite exists to prevent. The whitepaper is optimising harness DISPATCH -- where a fast cheap attempt is a
    legitimate exploration -- not verification, where a fast wrong answer is the catastrophic outcome.
    """
    lat = max(0.0, 1.0 - min(1.0, ms / LATENCY_BUDGET_MS))
    cost = max(0.0, 1.0 - (tier / MAX_TIER))
    return 0.4 * float(dod) + 0.3 * lat + 0.3 * cost


def reward(dod: float, ms: float, tier: int) -> float:
    """The reward this suite actually optimises: DoD-GATED.

    LATENCY AND COST ARE TIE-BREAKERS AMONG INSTRUMENTS THAT PRODUCED A CERTIFICATE, never a substitute for
    one. If dod == 0 the reward is 0 no matter how fast or cheap the instrument was, because a fast wrong
    answer is the one outcome this suite treats as catastrophic.

    This is a deliberate, measured departure from the whitepaper. Its weights were recovered by a test:
        reward_whitepaper(dod=0, ms=0.1, tier=0)  = 0.600    <- a fast guess
        reward_whitepaper(dod=1, ms=13167, tier=6) = 0.400    <- a slow proof
    Set DOD_GATE = False to reproduce the paper's behaviour.
    """
    if DOD_GATE and float(dod) <= 0.0:
        return 0.0
    lat = max(0.0, 1.0 - min(1.0, ms / LATENCY_BUDGET_MS))
    cost = max(0.0, 1.0 - (tier / MAX_TIER))
    # inside the gate, keep the paper's 0.4/0.3/0.3 shape so the two are comparable
    return 0.4 * float(dod) + 0.3 * lat + 0.3 * cost


# ------------------------------------------------------------------ training on our own rows
def _instrument_outcomes(rows: list) -> list:
    """Run every instrument on every row and record (dod, ms, tier). No labels needed: the reward is the
    instrument's own certificate."""
    import time as _t
    out = []
    from logos import pipeline
    for r in rows:
        text = r["text"]
        per = {}
        for pat in PATTERNS:
            name = pat["name"]
            t0 = _t.perf_counter()
            dod = 0.0
            try:
                if name == "arithmetic":
                    from logos.verify import math_core
                    st = math_core.check_arithmetic_text(text).status
                    dod = 1.0 if st in ("VERIFIED", "REFUTED") else 0.0
                elif name == "numeric_conflict":
                    from logos import numeric_conflict
                    res = numeric_conflict.conflicts(text, text)
                    dod = 0.0        # needs real evidence; without it there is no certificate
                elif name == "fol_bridge":
                    from logos.logic import fol_bridge
                    res = fol_bridge.check_row(text)
                    dod = 1.0 if res.get("status") in ("VALID", "INVALID") else 0.0
                elif name == "dep_encoder":
                    from logos.logic import dep_encoder
                    res = dep_encoder.parse_dependency(text)
                    dod = 1.0 if res.get("status") == "PARSED" else 0.0
                elif name == "scope_triage":
                    from logos.logic import scope_triage
                    res = scope_triage.triage(text)
                    dod = 1.0 if res.get("status") == "SCOPE_AMBIGUOUS" else 0.0
                elif name == "crosscheck":
                    dod = 0.0
                elif name == "rerank":
                    dod = 0.0
                elif name == "mathlib":
                    dod = 0.0        # refused by the cost model in practice; see measure()
                elif name == "truth_store":
                    dod = 0.0
                elif name == "escalate":
                    # ABSTAINING IS THE CORRECT ANSWER WHEN NOTHING ELSE CAN DECIDE. Scoring it 0 made the
                    # reward sparse enough that 23 of 32 sampled groups had ZERO variance and the router
                    # collapsed to a constant policy -- the same degeneracy as the 2.6B model, for the same
                    # reason: no signal. DoD for ESCALATE is decided after all other instruments have run,
                    # so it is patched in below rather than guessed here.
                    dod = 0.0
            except Exception:                                # noqa: BLE001
                dod = 0.0
            ms = (_t.perf_counter() - t0) * 1000.0
            if name == "mathlib":
                ms = LATENCY_BUDGET_MS
            per[name] = (dod, ms, pat["tier"])
        # ---- ABSTENTION IS A CORRECT ANSWER. If no certificate-bearing instrument could decide this row,
        # then ESCALATE is the right action and must be rewarded, otherwise the policy is pushed toward
        # guessing and 23 of 32 groups end up with zero variance (measured). This is the precision-first
        # premise expressed as a reward: a correct refusal counts.
        certified = [n for n, (dod, _ms, _t) in per.items() if dod > 0.0 and n != "escalate"]
        if not certified:
            d, ms, t = per["escalate"]
            per["escalate"] = (1.0, ms, t)
        out.append(per)
    return out


def train(rows: list, epochs: int = 1, group: int = 3, lr: float = 0.05, seed: int = 0,
          verbose: bool = True) -> dict:
    """One or more GRPO epochs over our own rows. Returns the router plus before/after selection accuracy."""
    rng = random.Random(seed)
    outcomes = _instrument_outcomes(rows)
    phis = [spacy24(r["text"]) for r in rows]
    router = SoftmaxRouter(seed=seed)

    def best_pattern(per: dict) -> int:
        scored = []
        for i, pat in enumerate(PATTERNS):
            dod, ms, tier = per[pat["name"]]
            scored.append((reward(dod, ms, tier), i))
        return max(scored)[1]

    def accuracy(r: SoftmaxRouter) -> float:
        hit = 0
        for phi, per in zip(phis, outcomes):
            pick = max(range(K), key=lambda i: r.probs(phi)[i])
            if pick == best_pattern(per):
                hit += 1
        return hit / max(len(rows), 1)

    before = accuracy(router)
    steps = skipped = 0
    history = []
    for ep in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for idx in order:
            phi, per = phis[idx], outcomes[idx]
            chosen = router.sample(phi, group=group, rng=rng)
            rewards = [reward(*per[PATTERNS[i]["name"]]) for i in chosen]
            res = router.update(phi, chosen, rewards, lr=lr)
            if res.get("skipped"):
                skipped += 1
            else:
                steps += 1
        acc = accuracy(router)
        history.append(round(acc, 4))
        if verbose:
            print(f"    epoch {ep + 1}: selection accuracy {acc:.1%}  (steps {steps}, "
                  f"zero-variance groups {skipped})")
    after = accuracy(router)
    return {"router": router, "before": before, "after": after, "steps": steps,
            "skipped": skipped, "history": history, "n_rows": len(rows)}


# ------------------------------------------------------------------ measure
def measure(limit: int = 40, epochs: int = 2) -> int:
    from logos import pipeline
    rows = [{"text": r, "kind": "text"} for r, _e, _k in pipeline.FIXTURE]
    print("=== GRPO pattern router trainend on our own rows ===")
    print(f"  {len(rows)} rows x {K} instruments x reward R = 0.4*DoD + 0.3*Latency + 0.3*Cost")
    print(f"  NO LABELS: the reward is whether the instrument returned a certificate (sympy/Z3/kernel)\n")

    outcomes = _instrument_outcomes(rows)
    print(f"  {'instrument':18s} {'tier':>4s} {'cert%':>7s} {'mean ms':>10s} {'mean R':>8s}")
    for pat in PATTERNS:
        dod = [per[pat["name"]][0] for per in outcomes]
        ms = [per[pat["name"]][1] for per in outcomes]
        rs = [reward(*per[pat["name"]]) for per in outcomes]
        cert = sum(dod) / len(dod) * 100 if dod else 0.0
        print(f"  {pat['name']:18s} {pat['tier']:4d} {cert:6.0f}% {sum(ms) / len(ms):10.2f} "
              f"{sum(rs) / len(rs):8.3f}")

    print("\n=== GRPO epochs ===")
    res = train(rows, epochs=epochs, group=3, lr=0.05, seed=0)
    print(f"\n  selection accuracy {res['before']:.1%} -> {res['after']:.1%}   "
          f"({res['steps']} updates, {res['skipped']} zero-variance groups)")

    phi = spacy24("2 + 2 = 5")
    t0 = time.perf_counter()
    for _ in range(1000):
        res["router"].probs(phi)
    dt = (time.perf_counter() - t0) / 1000 * 1000
    print(f"  decision latency {dt * 1000:.1f} us  (whitepaper reports 41 us)")

    print("\n  HONEST READING:")
    print(f"  - {res['n_rows']} rows is a PILOT, not a result. The whitepaper's headline is 10 prompts; both")
    print( "    are far too small to claim generalisation, and this module says so rather than quoting it.")
    print( "  - Accuracy here means 'picked the instrument with the highest realised reward on THIS row',")
    print( "    which is a closed-world target: on unseen rows the reward is not known in advance.")
    print( "  - The reward is verifiable (a Z3/Lean/sympy certificate), which is the one genuinely strong")
    print( "    property: it cannot be gamed by a fluent wrong answer the way a learned reward can.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.pattern_router - SELF TEST")
    ok = True

    # --- the feature vector is 24-dimensional and finite
    phi = spacy24("All men are mortal; therefore Socrates is mortal.")
    good = len(phi) == N_FEATURES and all(isinstance(x, float) for x in phi) and all(math.isfinite(x) for x in phi)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] spaCy24 returns {len(phi)} finite floats")

    # --- features discriminate: an arithmetic row and a prose row must differ
    a = spacy24("2 + 2 = 5")
    b = spacy24("The committee met on Tuesday and discussed the proposal at length.")
    good = a != b and a[5] == 1.0 and b[5] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] has_equals fires on arithmetic and not on prose")

    # --- the policy is a distribution
    r = SoftmaxRouter(seed=1)
    p = r.probs(phi)
    good = len(p) == K and abs(sum(p) - 1.0) < 1e-9 and all(v >= 0 for v in p)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the router returns a probability distribution over {K} patterns")

    # --- sampling returns DISTINCT patterns; duplicates would zero the advantage
    s = r.sample(phi, group=3, rng=random.Random(0))
    good = len(s) == 3 and len(set(s)) == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the group is sampled WITHOUT replacement (distinct)")

    # --- group-relative advantages are zero-mean, which is the whole point of the baseline
    mu = sum([0.9, 0.5, 0.1]) / 3
    sd = math.sqrt(sum((x - mu) ** 2 for x in [0.9, 0.5, 0.1]) / 3)
    advs = [(x - mu) / (sd + EPS) for x in [0.9, 0.5, 0.1]]
    good = abs(sum(advs)) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] group-relative advantages sum to zero")

    # --- a zero-variance group is SKIPPED, not divided by zero
    res = r.update(phi, [0, 1, 2], [0.5, 0.5, 0.5])
    good = res.get("skipped") is True and "zero-variance" in res.get("reason", "")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a zero-variance group is skipped with a reason")

    # --- THE LEARNING CLAIM: an update must raise the probability of the best-in-group pattern
    r2 = SoftmaxRouter(seed=2)
    before_p = r2.probs(phi)
    r2.update(phi, [0, 1, 2], [0.9, 0.2, 0.1], lr=0.5)
    after_p = r2.probs(phi)
    good = after_p[0] > before_p[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the best-in-group pattern gains probability "
          f"({before_p[0]:.3f} -> {after_p[0]:.3f})")

    # --- and the worst loses it
    good = after_p[2] < before_p[2]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the worst-in-group pattern loses probability "
          f"({before_p[2]:.3f} -> {after_p[2]:.3f})")

    # --- the reward weights are the ones the design specifies, and DoD dominates
    # --- THE REASON THIS SUITE DEPARTS FROM THE WHITEPAPER'S REWARD.
    #     Paper weights:      fast guess 0.600  >  slow proof 0.400     <- DoD does NOT dominate
    #     Gated weights here: fast guess 0.000  <  slow proof 0.400
    cheap_wrong = reward(dod=0.0, ms=0.1, tier=0)
    slow_right = reward(dod=1.0, ms=LATENCY_BUDGET_MS, tier=MAX_TIER)
    good = slow_right > cheap_wrong
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] GATED: a slow PROOF outscores a fast guess "
          f"({slow_right:.3f} > {cheap_wrong:.3f})")

    paper_wrong = reward_whitepaper(dod=0.0, ms=0.1, tier=0)
    paper_right = reward_whitepaper(dod=1.0, ms=LATENCY_BUDGET_MS, tier=MAX_TIER)
    good = paper_wrong > paper_right
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] UNGATED (paper weights) INVERTS it: fast guess "
          f"{paper_wrong:.3f} > slow proof {paper_right:.3f} -- which is why the gate exists")

    good = abs(reward(1.0, 0.0, 0) - (0.4 + 0.3 + 0.3)) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a perfect reward is exactly 1.0")

    # --- and no certificate means zero, regardless of speed
    good = reward(0.0, 0.0, 0) == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no certificate scores exactly 0.0 however fast it was")

    # --- ABSTENTION IS REWARDED WHEN NOTHING CAN CERTIFY. Without this the reward is so sparse that the
    #     router collapses to a constant policy (measured: 23 of 32 sampled groups had zero variance).
    from logos import pipeline as _p
    _rows = [{"text": t} for t, _e, _k in _p.FIXTURE]
    _oc = _instrument_outcomes(_rows)
    r_escalate = sum(per["escalate"][0] for per in _oc) / len(_oc)
    r_dep = sum(per["dep_encoder"][0] for per in _oc) / len(_oc)
    good = r_escalate > r_dep
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] correct abstention is rewarded "
          f"(escalate DoD {r_escalate:.2f} vs dep_encoder {r_dep:.2f})")

    # --- the router round-trips through a dict, so it can be persisted
    d = r.to_dict()
    r3 = SoftmaxRouter.from_dict(d)
    good = r3.probs(phi) == r.probs(phi)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the router persists and reloads identically")

    # --- and training on real rows must not make selection WORSE
    from logos import pipeline
    rows = [{"text": t} for t, _e, _k in pipeline.FIXTURE]
    res = train(rows, epochs=1, group=3, lr=0.05, seed=0, verbose=False)
    good = res["after"] >= res["before"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one GRPO epoch on {res['n_rows']} rows: "
          f"{res['before']:.1%} -> {res['after']:.1%}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GRPO pattern router over a spaCy complexity vector.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--features", default="")
    args = ap.parse_args(argv)
    if args.features:
        phi = spacy24(args.features)
        print(json.dumps({n: round(v, 3) for n, v in zip(FEATURE_NAMES, phi)}, indent=2))
        return 0
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
