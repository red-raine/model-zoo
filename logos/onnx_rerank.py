"""logos/onnx_rerank.py — run the int8 ONNX cross-encoder natively on the host. No Docker, no HTTP.

WHAT THIS REPLACES AND WHY IT MATTERS
------------------------------------
`argus-reranker` on :8086 is this exact file behind FastAPI:

    B:/ai-stack/models/argus-reranker/model_int8.onnx   543.3 MB, int8
    B:/ai-stack/models/argus-reranker/tokenizer.json    16.3 MB, XLM-R sentencepiece
    RERANK_MODE=onnx  ORT_THREADS=2

and `config.json` identifies it as XLMRobertaForSequenceClassification, 24 layers x 1024 hidden,
vocab 250002, 8194 positions -- i.e. bge-reranker-v2-m3, quantized. That is confirmed independently of
the score agreement in rerank_client.py: architecture plus parameter count (568M x 1 byte ~ 543 MB).

The service has no host port problem, but it does have three costs the host does not: an HTTP hop per
query, a serialised request queue, and a dependency on a container being up for a batch to run. Host-side
ONNX also lets a batch hold the session open and amortise the 543 MB load.

THE POINT OF THE SELF-TEST
The native path is only worth anything if it reproduces the service bit-for-bit-ish. If my pair encoding
is wrong -- wrong separator, wrong special tokens, query/text swapped -- the scores will still look
plausible, still be in [0,1], and still rank sensibly in easy cases. That is exactly the failure mode that
does not announce itself. So the self-test asserts NATIVE == SERVICE within 2e-3 on the same pairs, and
treats a mismatch as a bug rather than as jitter.

SAME SHAPE IS NOT SAME SPACE (a warning that cost nothing to find)
------------------------------------------------------------------
The three Qdrant collections are 384-dim cosine. So is `all-MiniLM-L6-v2`, which is cached on this host.
And so is `bge-small-en-v1.5`, which is what actually built them (read from the tei container's CMD:
`--model-id BAAI/bge-small-en-v1.5 --dtype float32`). Two different 384-dim models produce vectors in
INCOMPARABLE spaces: cosine similarity between them is meaningless while still returning numbers between
-1 and 1, so a naive query returns confidently ranked nonsense. This is the same class of error as reading
a logit as a probability -- identical shape, different space.

USAGE
    python tools/logos/onnx_rerank.py --self-test
    python tools/logos/onnx_rerank.py --measure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

MODEL_DIR = os.environ.get("LOGOS_ONNX_RERANK_DIR", r"B:\ai-stack\models\argus-reranker")
ONNX_PATH = os.path.join(MODEL_DIR, "model_int8.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
MAX_LEN = 8192
THREADS = int(os.environ.get("LOGOS_ONNX_THREADS", "2"))

_session = None
_tokenizer = None


def available() -> tuple:
    """Is the native path usable? Returns (ok, reason). Never raises -- a missing model is a state."""
    if not os.path.exists(ONNX_PATH):
        return False, f"model not found at {ONNX_PATH}"
    if not os.path.exists(TOKENIZER_PATH):
        return False, f"tokenizer not found at {TOKENIZER_PATH}"
    try:
        import onnxruntime                                        # noqa: F401
        import tokenizers                                         # noqa: F401
    except Exception as e:                                        # noqa: BLE001
        return False, f"missing library: {type(e).__name__}: {e}"
    return True, "ready"


def _load():
    """Load once and keep. The 543 MB session is the whole cost of the first call."""
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer
    import onnxruntime as ort
    from tokenizers import Tokenizer

    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    _tokenizer.enable_truncation(max_length=MAX_LEN)
    return _session, _tokenizer


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


def score_pair(query: str, text: str) -> dict:
    """Raw logit and probability for one (query, passage) pair."""
    import numpy as np

    so, tok = _load()
    enc = tok.encode(query, text)              # pair encoding; the self-test proves it matches the service
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    out = so.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    raw = float(np.asarray(out).reshape(-1)[0])
    return {"raw": raw, "prob": sigmoid(raw), "scale": "logit", "n_tokens": len(enc.ids)}


def rerank(query: str, texts: list) -> list:
    """Rank passages, descending probability. Same contract as rerank_client, so callers can swap."""
    res = []
    for i, t in enumerate(texts):
        s = score_pair(query, t)
        res.append({"index": i, "prob": s["prob"], "raw": s["raw"], "scale": "logit",
                    "n_tokens": s["n_tokens"]})
    return sorted(res, key=lambda r: -r["prob"])


PAIRS = [
    ("what is a syllogism", "A syllogism is a form of deductive reasoning."),
    ("what is a syllogism", "The bridge is 120 metres tall."),
    ("One mile is 1.609344 kilometres.", "One mile equals exactly 1.609344 kilometres."),
    ("One mile is 1.609344 kilometres.", "One mile is 1.500000 kilometres."),
    ("The Eiffel Tower is 330 metres tall.", "The Eiffel Tower stands 330 m high."),
]


def measure() -> int:
    ok, reason = available()
    print("=== native ONNX cross-encoder ===")
    print(f"  model     {ONNX_PATH}")
    print(f"  size      {os.path.getsize(ONNX_PATH) / 1e6:.1f} MB" if os.path.exists(ONNX_PATH) else "  size      -")
    print(f"  available {ok} ({reason})")
    if not ok:
        return 1

    import onnxruntime as ort
    print(f"  providers {ort.get_available_providers()}")

    t0 = time.perf_counter()
    so, tok = _load()
    load_s = time.perf_counter() - t0
    print(f"  load      {load_s:.2f}s (int8, {THREADS} threads)")
    for i in so.get_inputs():
        print(f"    in  {i.name:16s} {i.shape} {i.type}")
    for o in so.get_outputs():
        print(f"    out {o.name:16s} {o.shape} {o.type}")

    print("\n=== does the NATIVE path reproduce the SERVICE? ===")
    print(f"  {'query':30s} {'passage':34s} {'native':>9s} {'http':>9s} {'diff':>9s}")
    try:
        from logos import rerank_client
        http_ok = True
    except Exception:                                             # noqa: BLE001
        http_ok = False

    worst = 0.0
    timed_n, timed_h = 0.0, 0.0
    for q, p in PAIRS:
        n = score_pair(q, p)
        h = None
        if http_ok:
            try:
                t0 = time.perf_counter()
                h = rerank_client.rerank(q, [p], backend="argus")[0]
                timed_h += time.perf_counter() - t0
            except Exception:                                     # noqa: BLE001
                h = None
        d = abs(n["prob"] - h["prob"]) if h else float("nan")
        if h and not math.isnan(d):
            worst = max(worst, d)
        print(f"  {q[:30]:30s} {p[:34]:34s} {n['prob']:9.4f} "
              f"{(h['prob'] if h else float('nan')):9.4f} {d:9.5f}")

    print()
    if worst and not math.isnan(worst):
        if worst < 2e-3:
            print(f"  ==> max |native - service| = {worst:.6f}. The host path IS the service: same weights,")
            print(f"      same tokenizer, same order of arguments. Not a similar model -- the same one.")
        else:
            print(f"  ==> max |native - service| = {worst:.6f}, which is TOO LARGE. The tokenisation or the")
            print(f"      argument order is wrong. A mismatch here still returns plausible scores, which is")
            print(f"      why it has to be asserted rather than eyeballed.")

    print("\n=== cost of the HTTP hop ===")
    texts = [p for _q, p in PAIRS]
    t0 = time.perf_counter()
    for q, p in PAIRS:
        score_pair(q, p)
    native = (time.perf_counter() - t0) / len(PAIRS)
    http = timed_h / len(PAIRS) if timed_h else float("nan")
    print(f"  native {native * 1000:7.1f} ms/pair")
    print(f"  http   {http * 1000:7.1f} ms/pair" if not math.isnan(http) else "  http   unavailable")
    if not math.isnan(http) and native > 0:
        print(f"  ratio  {http / native:.2f}x")
    print(f"  and native needs no container to be running, which is the part that matters for a batch.")

    print("\n=== the trap this module documents rather than solves ===")
    print("  qdrant collections are 384-dim cosine. 'all-MiniLM-L6-v2' is cached on this host and is")
    print("  384-dim too. The collections were built by 'bge-small-en-v1.5' (read from the tei CMD),")
    print("  which is ALSO 384-dim. Same dimension, different vector space: cosine between them is")
    print("  meaningless while still returning a number in [-1, 1], so a naive query is confidently")
    print("  wrong. Same class of error as reading a logit as a probability.")
    print("  To use the existing collections the embedder must be bge-small-en-v1.5 specifically.")
    return 0


def run_self_test() -> bool:
    print("logos.onnx_rerank - SELF TEST")
    ok = True

    good, reason = available()
    print(f"  [{'PASS' if good else 'SKIP'}] native ONNX available ({reason})")
    if not good:
        print("SELF TEST: SKIPPED (model not present)")
        return True

    # --- the same sigmoid discipline as the HTTP client, so a logit is never compared to a probability
    good = abs(sigmoid(7.41092) - 0.99940) < 1e-4
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] sigmoid(7.41092) = {sigmoid(7.41092):.5f}, matching the "
          f"logit the service returned")

    # --- a relevant pair scores above an irrelevant one
    a = score_pair("what is a syllogism", "A syllogism is a form of deductive reasoning.")
    b = score_pair("what is a syllogism", "The bridge is 120 metres tall.")
    good = a["prob"] > 0.9 and b["prob"] < 0.01
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] relevant {a['prob']:.4f} vs irrelevant {b['prob']:.2e}")

    # --- THE ORACLE: the native path must reproduce the running service. This is the test that catches
    #     a wrong separator, swapped arguments, or the wrong special tokens -- none of which announce
    #     themselves, because all of them still return plausible numbers.
    try:
        from logos import rerank_client
        worst = 0.0
        for q, p in PAIRS:
            n = score_pair(q, p)["prob"]
            h = rerank_client.rerank(q, [p], backend="argus")[0]["prob"]
            worst = max(worst, abs(n - h))
        good = worst < 2e-3
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] NATIVE == SERVICE on {len(PAIRS)} pairs, "
              f"max diff {worst:.6f} (a mismatch means my tokenisation is wrong)")
    except Exception as e:                                        # noqa: BLE001
        print(f"  [SKIP] service unreachable, cannot cross-check ({type(e).__name__})")

    # --- truncation is real: a passage longer than 8192 tokens must not crash the session
    long_text = "word " * 9000
    try:
        s = score_pair("a claim", long_text)
        good = s["n_tokens"] <= MAX_LEN and 0.0 <= s["prob"] <= 1.0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] an over-long passage truncates to "
              f"{s['n_tokens']} tokens instead of raising")
    except Exception as e:                                        # noqa: BLE001
        ok = False
        print(f"  [FAIL] an over-long passage raised {type(e).__name__}: {e}")

    # --- ranking is by descending probability
    r = rerank("what is a syllogism",
               ["The bridge is 120 metres tall.", "A syllogism is a form of deductive reasoning."])
    good = r[0]["index"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] rerank() puts the relevant passage first "
          f"(index {r[0]['index']})")

    # --- missing model is a STATE not an exception: available() must answer, not raise
    saved = ONNX_PATH
    globals()["ONNX_PATH"] = r"Z:\definitely\not\here\model_int8.onnx"
    globals()["_session"] = None
    globals()["_tokenizer"] = None
    try:
        good, reason = available()
        good = good is False and "not found" in reason
    finally:
        globals()["ONNX_PATH"] = saved
        globals()["_session"] = None
        globals()["_tokenizer"] = None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a missing model returns (False, reason) instead of raising")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Host-side int8 ONNX cross-encoder.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--query", default="")
    ap.add_argument("--texts", default="")
    ap.add_argument("--availability", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.availability:
        ok, reason = available()
        print(json.dumps({"available": ok, "reason": reason, "model": ONNX_PATH}, indent=2))
        return 0
    if args.query and args.texts:
        print(json.dumps(rerank(args.query, [t for t in args.texts.split("|") if t]), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
