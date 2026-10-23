"""logos.locator — the Markov head: WHERE in the claim is it wrong.

WHY A 2-STATE HMM IS THE RIGHT SIZE HERE
----------------------------------------
The 2026 result that justifies this file (arXiv:2608.18115) fused per-token features (text
statistics + NLI entailment + LM surprisal) with a BiGRU and got **AUC 0.840 on RAGTruth**,
**+11 points over an independent per-token logistic regression**. The authors then ran the
decomposition that matters: *"most of the gain comes from temporal order rather than model
capacity"* and *"the same 0.845 ceiling recurs across recurrent, state-space (Mamba), and
attention architectures, locating the bottleneck in the feature set rather than the model."*

Read that twice. Architecture does not matter; **order** does, and it is worth ~11 AUC points.
A two-state HMM is the smallest model that captures order. At our data scale — 2.5k claims but
10^5-10^6 token labels — the HMM is trivially estimable and a transformer is not.

WHAT IT PREDICTS
----------------
Which spans of a claim are unsupported by the retrieved evidence. That is a different question
from the 4-way verdict, and it is the one a human reviewer actually needs: "this verdict is
wrong" is not actionable, "the 1998 in this claim is unsupported" is.

TWO HEADS, NOT ONE
------------------
arXiv:2507.20836 finds the FIRST unsupported token is far more detectable than subsequent ones.
A homogeneous model averages that away, so the emission model is exposed per-state and the
self-test pins both the order-capturing case and the "sticky prior overrides one dissenting
token" case — because that contrast IS the model.

USAGE
-----
    python tools/logos/locator.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

NEG_INF = -1e18
SUPPORTED, UNSUPPORTED = 0, 1


def _log_transition(p_stay: float = 0.5) -> np.ndarray:
    """2x2 transition matrix in log space. p_stay=0.5 is the 'no memory' control."""
    if not 0.0 < p_stay < 1.0:
        raise ValueError("p_stay must be in (0,1)")
    m = np.array([[p_stay, 1.0 - p_stay], [1.0 - p_stay, p_stay]])
    return np.log(m)


def emissions(o: np.ndarray) -> np.ndarray:
    """Per-token log-emissions from a 'supported' log-odds stream.

    o_t is evidence for state SUPPORTED. State SUPPORTED takes o_t, state UNSUPPORTED takes
    -o_t, so the model is symmetric and the only free signal is which side o_t favours.
    """
    o = np.asarray(o, dtype=float)
    return np.column_stack([o, -o])


def viterbi(o: np.ndarray, p_stay: float = 0.5,
            start: np.ndarray | None = None) -> tuple[float, list[int]]:
    """Exact best path. Returns (log-probability, state sequence)."""
    o = np.asarray(o, dtype=float)
    if o.ndim != 1:
        raise ValueError("o must be 1-D")
    T = len(o)
    if T == 0:
        return 0.0, []
    logA = _log_transition(p_stay)
    logB = emissions(o)
    logpi = np.log(np.asarray(start if start is not None else [0.5, 0.5], dtype=float))
    v = logpi + logB[0]
    bp = np.zeros((T, 2), dtype=int)
    for t in range(1, T):
        cand = v[:, None] + logA
        best_prev = np.argmax(cand, axis=0)
        v = cand[best_prev, np.arange(2)] + logB[t]
        bp[t] = best_prev
    last = int(np.argmax(v))
    score = float(v[last])
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t][path[-1]]))
    return score, list(reversed(path))


def forward_backward(o: np.ndarray, p_stay: float = 0.5) -> np.ndarray:
    """Exact per-token posteriors P(state_t = UNSUPPORTED). Verified against brute force."""
    o = np.asarray(o, dtype=float)
    T = len(o)
    logA = _log_transition(p_stay)
    logB = emissions(o)
    logpi = np.log(np.array([0.5, 0.5]))
    la = np.zeros((T, 2))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        m = la[t - 1][:, None] + logA
        mx = m.max(axis=0)
        la[t] = mx + np.log(np.exp(m - mx).sum(axis=0)) + logB[t]
    lb = np.zeros((T, 2))
    for t in range(T - 2, -1, -1):
        m = logA + logB[t + 1][None, :] + lb[t + 1][None, :]
        mx = m.max(axis=1)
        lb[t] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
    g = la + lb
    g = g - g.max(axis=1, keepdims=True)
    post = np.exp(g)
    post = post / post.sum(axis=1, keepdims=True)
    return post[:, UNSUPPORTED]


def brute_force_posteriors(o: np.ndarray, p_stay: float = 0.5) -> np.ndarray:
    """Enumerate all 2^T paths. Only for testing, T <= 12."""
    o = np.asarray(o, dtype=float)
    T = len(o)
    logA = _log_transition(p_stay)
    logB = emissions(o)
    logpi = np.log(np.array([0.5, 0.5]))
    scores = {}
    for path in itertools.product([0, 1], repeat=T):
        s = logpi[path[0]] + logB[0, path[0]]
        for t in range(1, T):
            s += logA[path[t - 1], path[t]] + logB[t, path[t]]
        scores[path] = s
    keys = list(scores)
    vals = np.array([scores[k] for k in keys])
    mx = vals.max()
    w = np.exp(vals - mx)
    w = w / w.sum()
    post = np.zeros((T, 2))
    for k, p in zip(keys, w):
        for t in range(T):
            post[t, k[t]] += p
    return post[:, UNSUPPORTED]


def spans_from_path(path: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of UNSUPPORTED as half-open token spans."""
    out, start = [], None
    for i, s in enumerate(path):
        if s == UNSUPPORTED and start is None:
            start = i
        elif s != UNSUPPORTED and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(path)))
    return out


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.locator - SELF TEST")
    ok = True

    # --- hand-computed Viterbi, exact
    # o = [2, -1, 2], uniform start, all transitions log 0.5
    #   t0: v=[1.306853, -2.693147]
    #   t1: best prev both 0 -> v0=-0.386294, v1=1.613706
    #   t2: v0 = max(-1.079441, 0.920559)+2 = 2.920559, from state 1
    o = np.array([2.0, -1.0, 2.0])
    score, path = viterbi(o, p_stay=0.5)
    good = abs(score - 2.920559) < 1e-5 and path == [0, 1, 0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hand-computed Viterbi: score {score:.6f} "
          f"(want 2.920559), path {path} (want [0,1,0])")

    # --- the same emissions with a STICKY prior: one dissenting token is overridden
    score_s, path_s = viterbi(o, p_stay=0.9)
    good = path_s == [0, 0, 0] and abs(score_s - 2.096133) < 1e-5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] sticky prior (p_stay=0.9) overrides the "
          f"dissent: path {path_s} (want [0,0,0]), score {score_s:.6f}")

    # --- forward-backward equals brute-force enumeration
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        T = int(rng.integers(3, 11))
        oo = rng.normal(0, 1.5, T)
        a = forward_backward(oo, p_stay=float(rng.uniform(0.2, 0.95)))
        b = brute_force_posteriors(oo, p_stay=0.5) if False else None
        # use the same p_stay for both
    worst = 0.0
    for _ in range(25):
        T = int(rng.integers(3, 11))
        oo = rng.normal(0, 1.5, T)
        ps = float(rng.uniform(0.2, 0.95))
        a = forward_backward(oo, p_stay=ps)
        b = brute_force_posteriors(oo, p_stay=ps)
        worst = max(worst, float(np.max(np.abs(a - b))))
    good = worst < 1e-10
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] forward-backward == brute force on 25 random "
          f"sequences (max err {worst:.2e})")

    # --- a dissenting single token in a long agreeing context must not be flagged
    o_long = np.array([1.5] * 5 + [-1.0] + [1.5] * 5)
    post = forward_backward(o_long, p_stay=0.9)
    # both must be LOW: the dissenting token is suppressed AND the agreeing tokens are not
    # flagged. An earlier version asserted post[0] > 0.5, i.e. that a SUPPORTED token should
    # look unsupported -- the assertion was backwards, not the model.
    good = post[5] < 0.5 and post[0] < 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one dissenting token among agreeing neighbours "
          f"keeps P(unsupported) below 0.5 ({post[5]:.3f})")

    # --- and a long disagreeing run IS flagged
    o_run = np.array([1.5] * 4 + [-1.0] * 4 + [1.5] * 4)
    post_run = forward_backward(o_run, p_stay=0.9)
    good = post_run[5] > 0.5 and post_run[0] < 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a 4-token dissenting run is flagged "
          f"(P={post_run[5]:.3f}) while its neighbours are not ({post_run[0]:.3f})")

    # --- span extraction is half-open and contiguous
    good = spans_from_path([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] spans_from_path -> "
          f"{spans_from_path([0, 1, 1, 0, 1])} (want [(1,3),(4,5)])")

    # --- pathological inputs must not produce NaN or inf
    finite = True
    for bad in (np.array([0.0]), np.array([1e6, -1e6]), np.array([-1e6, 1e6, 1e6])):
        s, pth = viterbi(bad)
        finite = finite and math.isfinite(s) and len(pth) == len(bad)
    good = finite
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] extreme inputs produce finite scores")

    # --- order genuinely matters: same number of dissenting tokens, different arrangement.
    # The first fixture ([2,2,-2,-2] vs [2,-2,2,-2]) was mirror-symmetric, so both means came
    # out at exactly 0.500 and the test could never have shown anything. These two are not
    # symmetric: adjacent dissents reinforce, separated ones get suppressed.
    agree_order = np.array([2.0, 2.0, -2.0, -2.0, 2.0, 2.0])
    mixed_order = np.array([2.0, -2.0, 2.0, 2.0, -2.0, 2.0])
    p1 = forward_backward(agree_order, p_stay=0.8).mean()
    p2 = forward_backward(mixed_order, p_stay=0.8).mean()
    good = abs(p1 - p2) > 1e-3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] order matters: same emissions in a different "
          f"order change mean P(unsupported) {p1:.3f} vs {p2:.3f}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2-state HMM error locator.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.demo:
        return 0 if run_self_test() else 1
    claim = "the bridge opened in 1998 and cost 45 million".split()
    o = np.array([1.2, 1.0, 1.4, -2.2, 0.9, 1.1, -1.8, 1.3, 1.0, 1.2])
    post = forward_backward(o, p_stay=0.8)
    score, path = viterbi(o, p_stay=0.8)
    print("claim:", " ".join(claim[:len(o)]))
    print("P(unsupported):", np.round(post, 2).tolist())
    print("spans:", [(claim[a:b]) for a, b in spans_from_path(path)])
    print("viterbi log-prob:", round(score, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
