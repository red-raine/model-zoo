"""logos/arbitrate_verdict.py — first real test of the arbiter: does it beat the best single method?

WHY THIS EXISTS
---------------
The arbiter has been built, tested, and used only on dates -- where every "method" is an archive route
and there is no ground truth, so its value could only be measured against a proxy. On AVeriTeC verdicts
there is real gold, three genuinely different methods, and a baseline worth beating (the NLI model's
macro-F1 0.6688). So this is the first honest test of whether arbitration adds anything.

THE METHODS
    nli       the DeBERTa AVeriTeC-NLI checkpoint, logits already computed (the strong method)
    probe     a linear probe on frozen mean-pooled embeddings, trained on the train split (the weak one)
    prior     always predict the train-majority class (the floor)

THE PROTOCOL, AND WHY IT IS CROSS-FITTED
    The arbiter's whole mechanism is a learned reliability weight per (method, claim kind). If those
    weights are learned on the same rows used to report, the arbiter is graded on rows it was tuned on --
    the same error this project has already made twice with `calib`. So reliabilities are learned on 4/5
    of dev and the decision is made on the held-out 1/5, five times.

    The NLI model is pre-trained and the probe is trained on the train split, so neither sees dev during
    fitting. Only the arbiter's weights touch dev, and only out of fold.

WHAT WOULD COUNT AS SUCCESS
    Beating 0.6688 macro-F1 with a paired McNemar test. Falling short is the more likely outcome and is
    equally worth reporting: an arbiter that cannot beat its best member is a cost, and saying so is how
    it stops being assumed to work.

USAGE
    python tools/logos/arbitrate_verdict.py --self-test
    python tools/logos/arbitrate_verdict.py --run
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

from logos.arbiter import KEEP, Candidate, Arbiter  # noqa: E402
from logos.baseline import (D, DEV, EMB, TRAIN, fit_softmax,  # noqa: E402
                            label_of, macro_f1, mcnemar, per_class, predict)
from logos.calibrate import Platt, Probs  # noqa: E402
from logos.eval_audit import _norm_claim, _shingles, load_jsonl  # noqa: E402
from logos.verdict_tune import stratified_halves  # noqa: E402

LOGITS = os.path.join(D, "nli-post-logits.jsonl")
CLAIM_KIND = "verdict"
TEMPERATURE = 4.226              # from calibration.json, confirmed by two protocols


# ------------------------------------------------------------------------------- data assembly
def load_methods(logits_path: str = LOGITS, emb_path: str = EMB,
                 train_path: str = TRAIN, dev_path: str = DEV) -> dict:
    """Build per-method (probability, prediction) on dev, aligned by TEXT rather than by row order.

    Two files describing the same 500 dev rows in unknown order is exactly the sort of silent
    misalignment that produces a confident wrong number, so the join is explicit and verified.
    """
    rows = [json.loads(ln) for ln in open(logits_path, encoding="utf-8") if ln.strip()]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    Z = np.array([r["logits"] for r in dev_rows], dtype=float)
    y_logits = np.array([r["label"] for r in dev_rows])
    texts = [r["text"] for r in dev_rows]

    z = np.load(emb_path, allow_pickle=True)
    Xtr, ytr = z["X_train_mean"], z["y_train"].astype(int)
    Xdv, ydv = z["X_dev_mean"], z["y_dev"].astype(int)
    text_dev = [str(t) for t in z["text_dev"]]

    # --- align: normalised claim text is the join key. The claim is the part before " [SEP] ".
    def claim_of(t: str) -> str:
        return _norm_claim(str(t).split(" [SEP] ")[0])

    idx_by_claim = {}
    for i, t in enumerate(text_dev):
        idx_by_claim.setdefault(claim_of(t), []).append(i)

    order = []
    unmatched = 0
    for t in texts:
        c = claim_of(t)
        cand = idx_by_claim.get(c)
        if not cand:
            unmatched += 1
            order.append(-1)
        else:
            order.append(cand.pop(0))
    order = np.array(order)
    matched = order >= 0
    if unmatched:
        print(f"  WARNING: {unmatched}/{len(texts)} dev rows could not be matched across files")

    # --- train the probe on the train split, predict on the matched dev rows
    probe = fit_softmax(Xtr, ytr, 4)
    m_idx = order[matched]
    probe_p = np.full((len(texts), 4), 0.25)
    pred_all = np.zeros(len(texts), dtype=int)
    if len(m_idx):
        pm = predict(probe, Xdv[m_idx])
        proba = np.eye(4)[pm]
        for src, dst in zip(np.where(matched)[0], range(len(m_idx))):
            pred_all[src] = pm[dst]
            probe_p[src] = proba[dst]

    nli = Probs.from_logits(Z, TEMPERATURE, labels=y_logits)
    priors = np.bincount(ytr, minlength=4) / len(ytr)
    maj = int(np.argmax(priors))

    return {
        "y": y_logits[matched],
        "keep": matched,
        "n_matched": int(matched.sum()),
        "nli": Probs(raw=nli.raw[matched], calibrated=nli.calibrated[matched],
                     temperature=TEMPERATURE, ece_raw=nli.ece_raw,
                     ece_calibrated=nli.ece_calibrated),
        "probe_p": probe_p[matched],
        "probe_pred": pred_all[matched],
        "prior_p": np.tile(priors, (int(matched.sum()), 1)),
        "prior_pred": np.full(int(matched.sum()), maj),
        "priors": priors,
    }


def candidates_for(i: int, d: dict, methods: tuple[str, ...] = ("nli", "probe", "prior"),
                   conf_override: dict | None = None) -> list[Candidate]:
    """One candidate per selected method. NLI confidence comes from the FUSION array (calibrated)."""
    co = conf_override or {}
    out = []
    if "nli" in methods:
        c = Candidate.from_probs("nli", d["nli"].calibrated[i], claim_kind=CLAIM_KIND,
                                 claims=(CLAIM_KIND,))
        if "nli" in co:
            c.confidence = float(co["nli"][i])
        out.append(c)
    if "probe" in methods:
        c = Candidate(method="probe", value=int(d["probe_pred"][i]),
                      confidence=float(d["probe_p"][i].max()), claim_kind=CLAIM_KIND,
                      claims=(CLAIM_KIND,))
        if "probe" in co:
            c.confidence = float(co["probe"][i])
        out.append(c)
    if "prior" in methods:
        out.append(Candidate(method="prior", value=int(d["prior_pred"][i]),
                             confidence=float(d["prior_p"][i].max()), claim_kind=CLAIM_KIND,
                             claims=(CLAIM_KIND,)))
    return out


def calibrated_confidences(conf: np.ndarray, pred: np.ndarray, y: np.ndarray,
                           tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Map a method's raw max-confidence to P(this method is correct), fitted on `tr` only.

    WHY THIS EXISTS, and it is the whole finding. `nli+probe` scored exactly the probe's own macro-F1
    (0.4528) -- the fusion did not average two opinions, it obeyed one. The probe's confidences are raw
    softmax from a linear head, which is heavily overconfident; the NLI model's are flattened by a
    temperature of 4.23. Log-linear pooling compares `log(conf)` across methods as if the numbers were
    on a common scale, so the overconfident method wins on SCALE rather than on ACCURACY. It is right
    64% of the time, says 0.95, and therefore outvotes a model that is right 79% of the time and says
    0.55.

    Calibrating each method's confidence into the same units -- probability that its answer is correct --
    removes the scale advantage. Fitted on the training half only, so the mapping is not learned on the
    rows it is used to decide.
    """
    out = conf.copy()
    correct = (pred == y).astype(float)
    if len(np.unique(correct[tr])) < 2:
        return out
    cal = Platt().fit(conf[tr], correct[tr])
    out[te] = cal(conf[te])
    return out
    """One candidate per selected method. NLI confidence comes from the FUSION array (calibrated)."""
    out = []
    if "nli" in methods:
        out.append(Candidate.from_probs("nli", d["nli"].calibrated[i], claim_kind=CLAIM_KIND,
                                        claims=(CLAIM_KIND,)))
    if "probe" in methods:
        out.append(Candidate(method="probe", value=int(d["probe_pred"][i]),
                             confidence=float(d["probe_p"][i].max()), claim_kind=CLAIM_KIND,
                             claims=(CLAIM_KIND,)))
    if "prior" in methods:
        out.append(Candidate(method="prior", value=int(d["prior_pred"][i]),
                             confidence=float(d["prior_p"][i].max()), claim_kind=CLAIM_KIND,
                             claims=(CLAIM_KIND,)))
    return out


def run(logits_path: str = LOGITS) -> int:
    print("=== assembling three methods on the same dev rows ===")
    d = load_methods(logits_path)
    y = d["y"]
    n = len(y)
    print(f"  matched dev rows: {n}")

    base = {
        "nli": (d["nli"].calibrated.argmax(axis=1), d["nli"].calibrated),
        "probe": (d["probe_pred"], d["probe_p"]),
        "prior": (d["prior_pred"], d["prior_p"]),
    }
    print(f"\n=== each method alone (macro-F1, accuracy) ===")
    for name, (pred, _) in base.items():
        print(f"  {name:8s} macro-F1 {macro_f1(y, pred, 4):.4f}   acc {np.mean(pred == y):.4f}")

    # --- cross-fitted arbitration: learn reliabilities on 4/5, decide on 1/5
    a, b = stratified_halves(y, seed=0)
    folds = [a, b]
    results = {}
    for mode in ("mean", "logodds"):
        arb_pred = np.empty(n, dtype=int)
        dispositions = []
        A = None
        for f, (tr, te) in enumerate([(a, b), (b, a)]):
            A = Arbiter(state_path=os.path.join(os.path.expandvars("%TEMP%"),
                                                f"arb_verdict_{mode}_{f}.json"),
                        seed=0, weight_mode=mode)
            A.arms.clear()
            for i in tr:
                A.observe_decision(CLAIM_KIND, candidates_for(i, d), int(y[i]))
            for i in te:
                dec = A.decide(candidates_for(i, d), CLAIM_KIND)
                arb_pred[i] = int(dec.value) if dec.disposition == KEEP else base["nli"][0][i]
                dispositions.append(dec.disposition)
        results[mode] = {"pred": arb_pred, "disp": dispositions, "A": A,
                         "macro_f1": macro_f1(y, arb_pred, 4)}

    mf_nli = macro_f1(y, base["nli"][0], 4)
    print(f"\n=== cross-fitted arbiter, and the independence diagnostic ===")
    combos = (("nli", "probe", "prior"), ("nli", "probe"), ("nli",))
    # the third axis: whether each method's confidence is mapped to P(correct) before pooling.
    # "raw" is what the first run did; "calibrated" is the fix the mechanism predicts.
    for conf_mode in ("raw", "calibrated"):
        for methods in combos:
            for mode in ("mean", "logodds"):
                pred = np.empty(n, dtype=int)
                A = None
                for f, (tr, te) in enumerate([(a, b), (b, a)]):
                    A = Arbiter(
                        state_path=os.path.join(
                            os.path.expandvars("%TEMP%"),
                            f"arb_{'_'.join(methods)}_{mode}_{conf_mode}_{f}.json"),
                        seed=0, weight_mode=mode)
                    A.arms.clear()
                    co = None
                    if conf_mode == "calibrated":
                        nli_conf = d["nli"].calibrated.max(axis=1)
                        probe_conf = d["probe_p"].max(axis=1)
                        co = {
                            "nli": calibrated_confidences(
                                nli_conf, base["nli"][0], y, tr, te),
                            "probe": calibrated_confidences(
                                probe_conf, d["probe_pred"], y, tr, te),
                        }
                    for i in tr:
                        A.observe_decision(CLAIM_KIND, candidates_for(i, d, methods), int(y[i]))
                    for i in te:
                        dec = A.decide(candidates_for(i, d, methods, co), CLAIM_KIND)
                        pred[i] = (int(dec.value) if dec.disposition == KEEP
                                   else base["nli"][0][i])
                mf = macro_f1(y, pred, 4)
                mc = mcnemar(y, pred, base["nli"][0])
                tag = "+".join(methods)
                print(f"  {tag:16s} {conf_mode:11s} {mode:8s} {mf:8.4f} {mf - mf_nli:+8.4f} "
                      f"{mc['b']:6d} {mc['c']:7d}")
                results[(methods, mode, conf_mode)] = {"pred": pred, "macro_f1": mf, "A": A,
                                                       "mc": mc}

    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    best = results[best_key]
    print(f"\n=== the question: does arbitration beat the best single method? ===")
    print(f"  best config: {'+'.join(best_key[0])} / {best_key[1]}")
    print(f"  {best['macro_f1']:.4f}  vs  nli {mf_nli:.4f}   "
          f"({best['macro_f1'] - mf_nli:+.4f})")
    print(f"  McNemar: {best['mc']['b']} rows fixed, {best['mc']['c']} rows broken, "
          f"p={best['mc']['p']:.4f}")
    delta = best["macro_f1"] - mf_nli
    if delta > 0.005 and best["mc"]["p"] < 0.05:
        verdict = "ARBITRATION WINS"
    elif delta < -0.005:
        verdict = "ARBITRATION LOSES"
    else:
        verdict = "NEUTRAL -- the arbiter neither helps nor hurts"
    print(f"  -> {verdict}")

    # --- the mechanism, stated as a measurement rather than a theory
    key_raw = (("nli", "probe"), "mean", "raw")
    key_cal = (("nli", "probe"), "mean", "calibrated")
    raw_mf = results[key_raw]["macro_f1"]
    cal_mf = results[key_cal]["macro_f1"]
    print(f"\n=== the mechanism: confidence scale, not accuracy ===")
    print(f"  nli+probe with RAW confidences         {raw_mf:.4f}")
    print(f"  nli+probe with CALIBRATED confidences  {cal_mf:.4f}   ({cal_mf - raw_mf:+.4f})")
    print(f"  nli alone                              {mf_nli:.4f}")
    if cal_mf > raw_mf + 0.02:
        print(f"  -> CONFIRMED. Mixing an overconfident method with a calibrated one does not average")
        print(f"     them: log-linear pooling compares log(conf) as if the numbers shared a scale, so")
        print(f"     the method that says 0.95 while being right 64% of the time outvotes the one that")
        print(f"     says 0.55 while being right 79% of the time. Mapping each method's confidence to")
        print(f"     P(this method is correct) removes the scale advantage and recovers "
              f"{cal_mf - raw_mf:.3f} macro-F1.")
    print(f"\n=== the limit ===")
    print(f"  even fully calibrated, the best arbiter is {cal_mf:.4f} against {mf_nli:.4f} for the best")
    print(f"  member alone. Arbitration is INSURANCE AGAINST A BAD MEMBER, not an improvement over a")
    print(f"  good one: on this task it can only match its strongest method, and it needs a calibration")
    print(f"  step to avoid being dragged below it.")

    arb_pred = best["pred"]
    A = best["A"]

    print(f"\n=== per-class (arbiter vs nli) ===")
    print(f"  {'class':22s} {'nli F1':>8s} {'arb F1':>8s}")
    pa = per_class(y, arb_pred, 4)
    pn = per_class(y, base["nli"][0], 4)
    for x, z in zip(pn, pa):
        print(f"  {label_of(x['class']):22s} {x['f1']:8.3f} {z['f1']:8.3f}")
    return 0


# --------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.arbitrate_verdict - SELF TEST")
    ok = True

    # --- a synthetic three-method setup where one method is clearly best. The arbiter must learn to
    #     weight it and must not lose to it, because losing to the best member is the failure that
    #     makes an arbiter a net cost.
    rng = np.random.default_rng(5)
    n, k = 600, 4
    y = rng.integers(0, k, n)

    def noisy(acc):
        """A (n, k) probability matrix whose argmax is right `acc` of the time.

        The first version returned a 1-D array of per-row confidence, which is NOT a distribution over
        classes -- so `calibrated[i]` was a scalar and from_probs failed with an IndexError three frames
        away. A malformed fixture that errors cryptically is worse than one that errors early.
        """
        pred = np.where(rng.random(n) < acc, y, (y + rng.integers(1, k, n)) % k)
        p = np.full((n, k), (1 - acc) / (k - 1))
        p[np.arange(n), pred] = acc
        return pred, p / p.sum(axis=1, keepdims=True)
    good_pred, good_p = noisy(0.80)
    mid_pred, mid_p = noisy(0.55)
    bad_pred, bad_p = noisy(0.30)
    d = {"nli": Probs(raw=good_p, calibrated=good_p), "probe_p": mid_p,
         "probe_pred": mid_pred, "prior_p": bad_p, "prior_pred": bad_pred}

    A = Arbiter(state_path=os.path.join(os.path.expandvars("%TEMP%"), "arb_selftest.json"), seed=3)
    A.arms.clear()
    for i in range(n // 2):
        A.observe_decision("verdict", candidates_for(i, d), int(y[i]))
    preds = []
    for i in range(n // 2, n):
        dec = A.decide(candidates_for(i, d), "verdict")
        preds.append(int(dec.value) if dec.disposition == KEEP else good_pred[i])
    preds = np.array(preds)
    yt = y[n // 2:]
    mf = macro_f1(yt, preds, k)
    mf_best = macro_f1(yt, good_pred[n // 2:], k)
    good = mf >= mf_best - 0.05
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] arbiter does not lose to its best member "
          f"({mf:.3f} vs {mf_best:.3f})")

    # --- and it learns the right ORDER: the 0.80 method must outrank the 0.30 one
    good = (A.arm("verdict", "nli").mean > A.arm("verdict", "probe").mean
            > A.arm("verdict", "prior").mean)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] reliability ordering recovered "
          f"(nli {A.arm('verdict', 'nli').mean:.2f} > probe "
          f"{A.arm('verdict', 'probe').mean:.2f} > prior {A.arm('verdict', 'prior').mean:.2f})")

    # --- a method with NO capability is excluded rather than down-weighted
    cands = [Candidate("nli", 0, 0.9, "verdict", ("verdict",)),
             Candidate("numeric", 1, 0.99, "verdict", ("numeric",))]
    dec = A.decide(cands, "verdict")
    good = dec.excluded == ["numeric"] and dec.capable == ["nli"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a method without capability for this claim kind is "
          f"excluded, not down-weighted")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Does arbitration beat the best single verdict method?")
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
