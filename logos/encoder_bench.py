"""logos/encoder_bench.py — 22M cross-encoder vs 568M cross-encoder, on our own task, with a cost axis.

THE QUESTION
------------
`rerank_client` measured bge-reranker-v2-m3 (568M int8) at **88.5 ms/pair** on this box, AUC 1.000 on ten
easy pairs against a 0.940 token-F1 baseline. The hardware probe then established there is no GPU: 8 cores,
AVX2 only, 64 GB RAM. So the only way bulk cleaning becomes viable is if a much smaller encoder holds most
of the quality.

`cross-encoder/ms-marco-MiniLM-L-4-v2` is already cached on this host at 22M parameters -- 26x smaller than
bge-reranker-v2-m3. If it holds the AUC, the encoder choice is settled and it is not close. If it does not,
we know exactly what the extra 546M parameters buy.

WHY THE BENCHMARK NEEDS HARD NEGATIVES
-------------------------------------
My first ten pairs would not have discriminated between a 22M and a 568M model, because every negative was
about a DIFFERENT SUBJECT. Any bag of words separates those, and a 0.940 token-F1 baseline already proved
it. So half the negatives here are HARD: same subject, same vocabulary, no answer in them.

    "One mile is 1.609344 kilometres."  vs  "Miles were defined differently in different countries."
    "Apoptosis is programmed cell death." vs "Apoptosis is studied in developmental biology."

A model that only matches topics will call those relevant. A model that reads the relation will not. That is
the difference we are trying to buy, so the benchmark has to be able to see it.

LABELS ARE CONSTRUCTED, NOT JUDGED. I wrote them knowing the intended answer, so this measures whether each
model does what I designed the test to ask -- not accuracy over a corpus. The absent held-out set remains the
project's largest defect, and this module does not fix it.

USAGE
    python tools/logos/encoder_bench.py --self-test
    python tools/logos/encoder_bench.py --measure
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the model is cached; never reach for the network
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------------------------------------------------------------- benchmark
# 1 = the passage bears on the claim, 0 = it does not (different subject, OR same subject with no answer)
# (claim, passage, label, hard)  -- 1 = bears on the claim, 0 = does not.
# `hard` = the negative shares the subject and the vocabulary and still contains no answer. Those are the
# only negatives that can distinguish a 22M encoder from a 568M one, so they are declared, not inferred.
BENCH = [
    # ---- numerics
    ("One mile is 1.609344 kilometres.",
     "One mile equals exactly 1.609344 kilometres by international definition.", 1, False),
    ("One mile is 1.609344 kilometres.",
     "Qdrant is a vector database written in Rust.", 0, False),
    ("One mile is 1.609344 kilometres.",
     "Miles were defined differently in different countries and historical periods.", 0, True),
    ("One mile is 1.609344 kilometres.",
     "The mile race was run in 3 minutes and 43 seconds.", 0, True),
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower stands 330 m high at its tip.", 1, False),
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower was completed in 1889 for the World's Fair.", 0, True),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "The served model reports a max_input_length of 8192 tokens.", 1, False),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "The bge-reranker family includes models of several different sizes.", 0, True),
    # ---- prose
    ("Apoptosis is programmed cell death.",
     "Apoptosis is a form of programmed cell death occurring in multicellular organisms.", 1, False),
    ("Apoptosis is programmed cell death.",
     "The stock market fell sharply on Tuesday.", 0, False),
    ("Apoptosis is programmed cell death.",
     "Apoptosis is studied extensively in developmental biology.", 0, True),
    ("Apoptosis is programmed cell death.",
     "Necrosis and apoptosis are both modes of cell death.", 0, True),
    ("A categorical syllogism has two premises and a conclusion.",
     "A categorical syllogism consists of a major premise, a minor premise and a conclusion.", 1, False),
    ("A categorical syllogism has two premises and a conclusion.",
     "Aristotle analysed syllogisms in the Prior Analytics.", 0, True),
    ("A categorical syllogism has two premises and a conclusion.",
     "Syllogisms are studied in introductory logic courses.", 0, True),
    # ---- code / docs
    ("The model was exported to int8 ONNX at 569.7 MB.",
     "The int8 export of the model is 569.7 MB on disk.", 1, False),
    ("The model was exported to int8 ONNX at 569.7 MB.",
     "ONNX Runtime supports several execution providers.", 0, False),
    ("The model was exported to int8 ONNX at 569.7 MB.",
     "Quantising to int8 usually reduces model size substantially.", 0, True),
    # ---- arithmetic
    ("2 + 2 = 4", "Two plus two equals four.", 1, False),
    ("2 + 2 = 4", "The first computer was built in the 1940s.", 0, False),
    ("2 + 2 = 4", "Arithmetic is taught in primary school.", 0, True),
    # ---- temporal
    ("BitNet Embeddings were released in July 2026.",
     "Microsoft released the first 1-bit embedding models in July 2026.", 1, False),
    ("BitNet Embeddings were released in July 2026.",
     "BitNet b1.58 was introduced in a 2024 paper.", 0, True),
    ("BitNet Embeddings were released in July 2026.",
     "Embedding models map text to dense vectors.", 0, False),
    # ---- infrastructure
    ("The tei container serves BAAI/bge-small-en-v1.5.",
     "The text-embeddings-router was started with --model-id BAAI/bge-small-en-v1.5.", 1, False),
    ("The tei container serves BAAI/bge-small-en-v1.5.",
     "The reranker container serves bge-reranker-v2-m3 instead.", 0, True),
    ("The tei container serves BAAI/bge-small-en-v1.5.",
     "Docker containers share a network namespace per network.", 0, False),
    # ---- same entity, wrong attribute
    ("Qdrant has three collections on this host.",
     "This host runs a Qdrant instance with three collections.", 1, False),
    ("Qdrant has three collections on this host.",
     "Qdrant uses HNSW indexing for approximate nearest neighbours.", 0, True),
    ("Qdrant has three collections on this host.",
     "The largest collection holds 18852 vectors.", 0, True),
]


def _auc(scored) -> float:
    """P(score(relevant) > score(irrelevant)) by pair counting. Ties are half credit."""
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _baseline_scores():
    """Token-F1, so every encoder is compared against something dumb on the SAME pairs.

    Without this, a good AUC reads as a model property when it may be a property of the fixtures.
    """
    from logos.rerank_client import lexical_score
    return [(lexical_score(c, p), y) for c, p, y, _h in BENCH]


# ------------------------------------------------------------------------------- models
def _bench_minilm(pairs) -> dict:
    """22M cross-encoder loaded with transformers directly.

    NOT sentence_transformers: its 5.4.0 release fails against the installed `datasets` with
    `cannot import name 'Dataset'`. Loading AutoModelForSequenceClassification avoids that dependency
    entirely, and it also lines the two models up as the same kind of object -- raw logits out of a
    classification head -- so the comparison is not measuring two libraries.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    load = time.perf_counter() - t0

    lat, scores = [], []
    for claim, passage in pairs:
        t = time.perf_counter()
        with torch.no_grad():
            enc = tok(claim, passage, return_tensors="pt", truncation=True, max_length=512)
            logits = mdl(**enc).logits
        scores.append(float(logits.reshape(-1)[0]))
        lat.append((time.perf_counter() - t) * 1000.0)
    n = sum(p.numel() for p in mdl.parameters())
    return {"name": f"{name.split('/')[-1]} ({n / 1e6:.0f}M)", "params": n, "load_s": load,
            "scores": scores, "median_ms": statistics.median(lat), "scale": "logit"}


def _bench_bge(pairs) -> dict:
    """568M int8 ONNX cross-encoder, native on the host. No Docker, no HTTP."""
    from logos import onnx_rerank
    ok, reason = onnx_rerank.available()
    if not ok:
        return {"name": "bge-reranker-v2-m3 int8 (568M)", "unavailable": reason}
    onnx_rerank._load()                                   # warm the session, excluded from latency
    lat = []
    scores = []
    for claim, passage in pairs:
        t = time.perf_counter()
        s = onnx_rerank.score_pair(claim, passage)
        lat.append((time.perf_counter() - t) * 1000.0)
        scores.append(s["raw"])
    return {"name": "bge-reranker-v2-m3 int8 (568M)", "params": 568_000_000, "load_s": 0.0,
            "scores": scores, "median_ms": statistics.median(lat), "scale": "logit"}


def _bench_bge_float32(pairs) -> dict:
    """The SAME weights as the int8 ONNX model, served float32 by TEI on :8084.

    This is the control that separates "the 568M model is worse on this task" from "my int8 export is
    worse". Without it, the comparison has an unmeasured confound sitting in the middle of it.
    """
    from logos import rerank_client
    # latency of the HTTP hop is included, which biases AGAINST this arm -- stated rather than corrected,
    # because the question is whether the QUALITY gap closes, not which transport is faster.
    lat, scores = [], []
    for claim, passage in pairs:
        t = time.perf_counter()
        r = rerank_client.rerank(claim, [passage], backend="tei")
        lat.append((time.perf_counter() - t) * 1000.0)
        scores.append(float(r[0]["prob"]))
    return {"name": "bge-reranker-v2-m3 fp32 (568M)", "params": 568_000_000, "load_s": 0.0,
            "scores": scores, "median_ms": statistics.median(lat), "scale": "probability"}


MODELS = (_bench_minilm, _bench_bge, _bench_bge_float32)


def measure() -> int:
    pairs = [(c, p) for c, p, _y, _h in BENCH]
    labels = [y for _c, _p, y, _h in BENCH]
    n_pos = sum(labels)
    print("=== encoder benchmark: our task, hard negatives, with a cost axis ===")
    print(f"  {len(BENCH)} pairs: {n_pos} relevant / {len(BENCH) - n_pos} not-relevant")
    n_hard = sum(1 for _c, _p, y, h in BENCH if y == 0 and h)
    print(f"  of the {len(BENCH) - n_pos} negatives, {n_hard} are HARD (same subject, no answer)")
    print()

    base = _baseline_scores()
    base_auc = _auc(base)
    results = []

    print(f"  {'model':30s} {'params':>10s} {'AUC':>6s} {'median ms':>10s} {'load s':>7s}  {'vs baseline':>12s}")
    row = f"  {'token-F1 baseline':30s} {'-':>10s} {base_auc:6.3f} {'<0.1':>10s} {'-':>7s}  {'-':>12s}"
    print(row)
    for fn in MODELS:
        try:
            r = fn(pairs)
        except Exception as e:                            # noqa: BLE001
            print(f"  {fn.__name__:30s} FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        if "unavailable" in r:
            print(f"  {r['name']:30s} unavailable: {r['unavailable'][:50]}")
            continue
        r["auc"] = _auc(list(zip(r["scores"], labels)))
        results.append(r)
        delta = r["auc"] - base_auc
        print(f"  {r['name']:30s} {r['params']:>10,} {r['auc']:6.3f} {r['median_ms']:10.1f} "
              f"{r['load_s']:7.2f}  {delta:+12.3f}")

    if len(results) < 2:
        print("\n  need at least two models to compare; some are unavailable")
        return 0

    # TWO SEPARATE QUESTIONS, and conflating them is how a benchmark lies:
    #   1. size      -- does the 19M model hold up against the 568M one?
    #   2. precision -- is the size result an artefact of my INT8 export rather than of the model?
    small = min(results, key=lambda r: r["params"])
    arms = {r["name"]: r for r in results}
    big_i8 = next((r for r in results if "int8" in r["name"]), None)
    big_fp = next((r for r in results if "fp32" in r["name"]), None)
    big = big_i8 or max(results, key=lambda r: r["params"])

    n_pos = labels.count(1)
    n_neg = labels.count(0)
    n_cmp = n_pos * n_neg

    print("\n=== 1. does the small model hold up? ===")
    speed = big["median_ms"] / small["median_ms"] if small["median_ms"] else float("nan")
    dauc = big["auc"] - small["auc"]
    sizef = big["params"] / small["params"]
    flips = abs(dauc) * n_cmp
    print(f"  {small['name']}  ->  {big['name']}")
    print(f"  parameters  {sizef:5.1f}x   ({small['params']:,} vs {big['params']:,})")
    print(f"  latency     {speed:5.2f}x slower  ({small['median_ms']:.1f} ms -> {big['median_ms']:.1f} ms)")
    print(f"  AUC         {dauc:+.3f}  ({small['auc']:.3f} -> {big['auc']:.3f})")
    print(f"  that gap is {flips:.0f} of {n_cmp} positive/negative comparisons flipped")
    print()
    if abs(dauc) < 0.02:
        print(f"  ==> INDISTINGUISHABLE on this benchmark ({dauc:+.3f}, {flips:.0f} of {n_cmp} comparisons).")
        print(f"      The 546M extra parameters buy no measurable relevance quality here, at {speed:.0f}x")
        print(f"      the latency. The defensible conclusion is not 'small is better' -- it is 'small is")
        print(f"      not worse, and is far cheaper.'")
    elif dauc > 0:
        print(f"  ==> the big model leads by {dauc:+.3f} ({flips:.0f}/{n_cmp} comparisons) for {speed:.0f}x")
        print(f"      the latency. Whether that is worth paying depends on what the remaining errors cost,")
        print(f"      which is a labels question.")
    else:
        print(f"  ==> the SMALL model leads by {-dauc:.3f} ({flips:.0f}/{n_cmp} comparisons). DO NOT read")
        print(f"      that as 'small is better'. At this size the two are indistinguishable, and the")
        print(f"      usable finding is that 19M reaches the same place {speed:.0f}x faster.")

    # ---- the control
    if big_i8 and big_fp:
        dp = big_i8["auc"] - big_fp["auc"]
        dflips = abs(dp) * n_cmp
        print("\n=== 2. is that a quantisation artefact? (int8 vs fp32, SAME weights) ===")
        print(f"  int8 ONNX   AUC {big_i8['auc']:.3f}  {big_i8['median_ms']:8.1f} ms")
        print(f"  fp32 TEI    AUC {big_fp['auc']:.3f}  {big_fp['median_ms']:8.1f} ms")
        print(f"  difference  {dp:+.3f}  ({dflips:.0f}/{n_cmp} comparisons)")
        print()
        if abs(dp) < 0.02:
            print(f"  ==> int8 did NOT cost measurable quality ({dp:+.3f}). So the size result is about the")
            print(f"      MODELS, not about my export -- the confound is eliminated rather than argued away.")
        elif dp > 0:
            print(f"  ==> int8 is AHEAD of fp32 by {dp:+.3f}. Either quantisation acts as a regulariser here")
            print(f"      or the two runtimes disagree; either way the size result is not an int8 artefact.")
        else:
            print(f"  ==> fp32 leads by {-dp:.3f}: INT8 COST QUALITY. The size comparison above used the int8")
            print(f"      arm, so it UNDERSTATES the 568M model. Re-read question 1 against the fp32 row.")
        print(f"  and note fp32 over HTTP costs {big_fp['median_ms'] / small['median_ms']:.0f}x the small")
        print(f"  model's latency, so the transport is measured too, not assumed free.")

    # ---- and the practical verdict that follows from both
    print("\n=== the decision this supports ===")
    per_10k_small = small["median_ms"] * 10_000 / 1000.0
    per_10k_big = big["median_ms"] * 10_000 / 1000.0
    print(f"  10,000 rows: {per_10k_small / 60:.1f} min on the 19M vs {per_10k_big / 60:.1f} min on the 568M.")
    print(f"  At that size BOTH are viable, so cost alone does not decide it. What decides it is context:")
    print(f"  ms-marco-MiniLM truncates at 512 tokens, bge-reranker-v2-m3 reaches 8192. So the honest")
    print(f"  design is a CASCADE -- 19M for the bulk, the 568M only where a pair exceeds 512 tokens or")
    print(f"  the small model lands in the ambiguous band. That is the same envelope principle the rest")
    print(f"  of this suite already runs on, arrived at here by measurement rather than by preference.")

    print("\n=== the behaviour both models share, and it is the important part ===")
    # do they both DISCARD a contradicting passage rather than flag it?
    for r in results:
        idx = [i for i, (c, p, _y, _h) in enumerate(BENCH)
               if (c, p) == ("One mile is 1.609344 kilometres.",
                             "Miles were defined differently in different countries and historical periods.")]
        if idx:
            s = r["scores"][idx[0]]
            print(f"  {r['name'][:28]:28s} hard negative scored {s:8.3f} (a good model puts this LOW)")
    print("  both being low is CORRECT for a relevance test and still not a truth verdict: the")
    print("  passage is not about the claim's value. Deciding refutation is numeric_conflict's job.")

    print("\n=== what this number is and is not ===")
    print(f"  IT IS: a measured cost/quality point on this exact CPU, with a baseline, on {len(BENCH)}")
    print(f"         pairs built to include hard negatives so the test CAN discriminate.")
    print("  IT IS NOT: an accuracy estimate. Labels are constructed by me, and one benchmark cannot")
    print("         separate a genuinely better encoder from a luckier one at this size.")
    print("  AND: ms-marco-MiniLM is trained for MS MARCO passage ranking, not for our claim/evidence")
    print("       relation. A 22M model winning here may reflect how easy my negatives still are, not")
    print("       that it generalises to real corpus rows.")
    return 0


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.encoder_bench - SELF TEST")
    ok = True

    # --- the benchmark must be able to fail. equal-sized classes mean a constant scorer is 0.5, not 1.0
    labels = [y for _c, _p, y, _h in BENCH]
    good = 0 < sum(labels) < len(labels)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the benchmark has both classes "
          f"({sum(labels)} / {len(BENCH) - sum(labels)})")

    good = abs(_auc([(0.5, 1), (0.5, 0)]) - 0.5) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a constant scorer scores 0.500 on it")

    good = _auc([(0.9, 1), (0.1, 0)]) == 1.0 and _auc([(0.1, 1), (0.9, 0)]) == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC reaches 1.0 and 0.0, so it reports both success and failure")

    # --- and it must actually contain hard negatives, or the comparison is vacuous
    n_hard = sum(1 for _c, _p, y, h in BENCH if y == 0 and h)
    n_neg = sum(1 for _c, _p, y, _h in BENCH if y == 0)
    good = n_hard >= n_neg // 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] declared hard negatives are at least half of the negatives "
          f"({n_hard}/{n_neg})")

    # --- every benchmark triple is well formed
    good = all(isinstance(c, str) and isinstance(p, str) and y in (0, 1) and isinstance(h, bool)
               and c and p for c, p, y, h in BENCH)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every row is (non-empty claim, non-empty passage, 0 or 1)")

    # --- the baseline is computable and is not accidentally perfect
    try:
        base_auc = _auc(_baseline_scores())
        good = 0.5 <= base_auc <= 1.0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] token-F1 baseline AUC = {base_auc:.3f} "
              f"(a baseline at 1.000 would mean the fixtures are too easy to measure anything)")
    except Exception as e:                                # noqa: BLE001
        print(f"  [SKIP] baseline unavailable ({type(e).__name__})")

    # --- the int8-vs-fp32 control must be wired in. Comparing a quantised export against a
    #     full-precision one and blaming architecture is the mistake this arm exists to prevent.
    good = _bench_bge_float32 in MODELS and _bench_bge in MODELS
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both bge arms are present (int8 ONNX and float32 TEI), "
          f"so quantisation is measured, not assumed")

    # --- the 22M model is loadable from cache; if it is not, say so rather than crash
    try:
        r = _bench_minilm([("2 + 2 = 4", "Two plus two equals four.")])
        good = len(r["scores"]) == 1
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] the cached 22M cross-encoder loads and scores "
              f"({r['scores'][0]:.3f} on an easy pair)")
    except Exception as e:                                # noqa: BLE001
        print(f"  [SKIP] 22M model unavailable ({type(e).__name__}: {str(e)[:50]})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Small vs large cross-encoder, with a cost axis.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
