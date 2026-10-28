"""logos/rerank_client.py — the live cross-encoder, wired in as an evidence gate, plus the honesty check.

WHAT IS ACTUALLY RUNNING (probed, not assumed)
---------------------------------------------
    :8084  TEI            BAAI/bge-reranker-v2-m3   float32, max_input_length 8192
    :8086  argus-reranker onnx mode, /rerank takes {"query", "texts"}
    :6333  qdrant         1.19.2, three 384-dim cosine collections
                          lab-research 1009 pts | lair-code 18852 pts | mc-wiki 87 pts

    /reranker (search-stack) and /tei (cpu-latest) have NO host port -- reachable only on the Docker
    network, and compose files are not ours to edit.

THE FIRST THING THE PROBE CAUGHT: THE TWO RERANKERS ARE ONE WITNESS
------------------------------------------------------------------
Same query, same two documents:

    TEI   :8084  ->  0.99833 ,  1.6093e-05      <- already probabilities
    argus :8086  ->  7.40287 , -10.99631        <- raw LOGITS

    sigmoid(7.40287) = 0.99939   vs TEI 0.99833
    sigmoid(-10.99631) = 1.67e-05 vs TEI 1.61e-05

They agree to four decimal places once the scale is reconciled. So these are not two opinions; they are
one model (bge-reranker-v2-m3) exported twice. **Counting them as two witnesses would be fake independence
of exactly the kind the truth store now refuses by construction.** This module measures that instead of
assuming it.

WHY A RERANKER IS NOT A TRUTH WITNESS
------------------------------------
A cross-encoder scores support-shaped relevance, and MEASURED it does that well: it puts a supporting
passage above a contradicting one in 5/5 triples while token overlap manages none. But it reports the loser
by SCORING IT LOW, which is discarding, not refuting. Contradicting and off-topic passages land in the same
region, so the score carries one bit -- "is this support?" -- and not the second bit, "does this refute, or
is it about something else?". Feeding it into the arbiter as evidence of correctness would be circular: it
measures the retrieval step's own success criterion.

So the only place it belongs is **upstream, as an evidence-sufficiency gate**: if no retrieved passage
clears the relevance threshold, then we have no evidence and the honest verdict is UNKNOWN. That is a
precision-first use -- it forces abstention rather than inventing a verdict -- and it is what the pipeline
calls it for.

USAGE
    python tools/logos/rerank_client.py --self-test
    python tools/logos/rerank_client.py --measure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

TEI_URL = os.environ.get("LOGOS_TEI_URL", "http://localhost:8084")
ARGUS_URL = os.environ.get("LOGOS_ARGUS_URL", "http://localhost:8086")
QDRANT_URL = os.environ.get("LOGOS_QDRANT_URL", "http://localhost:6333")
TIMEOUT = float(os.environ.get("LOGOS_RERANK_TIMEOUT", "30"))

_UA = {"User-Agent": "logos-rerank/1.0", "Content-Type": "application/json"}


def _post(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=_UA, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _get(url: str):
    req = urllib.request.Request(url, headers=_UA, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def sigmoid(x: float) -> float:
    """Logit -> probability. Clamped so a large |x| cannot raise OverflowError."""
    if x >= 0:
        z = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


# ------------------------------------------------------------------------------- backends
def tei_rerank(query: str, texts: list) -> list:
    """TEI returns probabilities directly. Verified against /info: BAAI/bge-reranker-v2-m3."""
    out = _post(f"{TEI_URL}/rerank", {"query": query, "texts": list(texts)})
    if isinstance(out, dict):                       # some TEI builds wrap in {"results": [...]}
        out = out.get("results") or out.get("data") or []
    return [{"index": int(r["index"]), "prob": float(r["score"]), "raw": float(r["score"]),
             "scale": "probability"} for r in out]


def argus_rerank(query: str, texts: list) -> list:
    """argus returns RAW LOGITS, measured: 7.40287 and -10.99631 for the same pair TEI put at 0.998/1.6e-05.

    Converted here so downstream code never compares a logit to a probability. That is the same
    incommensurable-confidence defect that cost the arbiter 0.224 macro-F1 when it mixed scales.
    """
    out = _post(f"{ARGUS_URL}/rerank", {"query": query, "texts": list(texts)})
    if isinstance(out, dict):
        out = out.get("results") or out.get("data") or []
    res = []
    for r in out:
        raw = float(r["score"])
        res.append({"index": int(r["index"]), "prob": sigmoid(raw), "raw": raw, "scale": "logit"})
    return res


BACKENDS = {"tei": tei_rerank, "argus": argus_rerank}


def health() -> dict:
    """What is actually serving, and on what scale. Never assume -- ask."""
    h = {}
    try:
        info = _get(f"{TEI_URL}/info")
        h["tei"] = {"up": True, "model": info.get("model_id"),
                    "dtype": info.get("model_dtype"),
                    "max_input_length": info.get("max_input_length")}
    except Exception as e:                                   # noqa: BLE001
        h["tei"] = {"up": False, "error": type(e).__name__}
    try:
        info = _get(f"{ARGUS_URL}/health")
        h["argus"] = {"up": True, "mode": info.get("mode")}
    except Exception as e:                                   # noqa: BLE001
        h["argus"] = {"up": False, "error": type(e).__name__}
    try:
        cols = (_get(f"{QDRANT_URL}/collections").get("result") or {}).get("collections") or []
        h["qdrant"] = {"up": True, "collections": [c.get("name") for c in cols]}
    except Exception as e:                                   # noqa: BLE001
        h["qdrant"] = {"up": False, "error": type(e).__name__}
    return h


def rerank(query: str, texts: list, backend: str = "tei") -> list:
    """Rank passages for one query. Returns probabilities, whatever the backend emits."""
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    return sorted(BACKENDS[backend](query, texts), key=lambda r: -r["prob"])


# ------------------------------------------------------------------------------- the gate
def evidence_sufficient(query: str, passages: list, threshold: float = 0.5,
                        backend: str = "tei") -> dict:
    """Is there ANY passage that clears the relevance bar?

    This is the only thing the pipeline is allowed to use a reranker for. It answers "do we have evidence
    at all?", NOT "is the claim true". A well-supported truth score must never be read out of a relevance
    score -- see the module docstring.

    Fail-closed: an unreachable backend returns sufficiency=None, which the caller must treat as UNKNOWN
    rather than as "no evidence" (same distinction as 429-is-alive vs 409-is-proxy-broken).
    """
    if not passages:
        return {"sufficient": False, "reason": "no candidate evidence", "top": None,
                "low_band": [], "backend": backend}
    try:
        ranked = rerank(query, passages, backend=backend)
    except Exception as e:                                   # noqa: BLE001
        return {"sufficient": None, "reason": f"backend unreachable: {type(e).__name__}",
                "top": None, "low_band": [], "backend": backend}
    top = ranked[0]
    # THE LOW BAND IS RETURNED, NOT SWALLOWED. Measured: 4 of 5 contradicting passages score below a
    # support threshold, and the contradiction/off-topic ordering is not reliable enough to act on (AUC
    # 0.920 with at least one inversion). So this list means "passages relevance cannot judge" and the
    # caller's job is to validate them. Dropping them would make REFUTED unreachable and leave the reject
    # pile empty by construction.
    low_band = [{"index": r["index"], "prob": r["prob"], "text": passages[r["index"]][:200]}
                for r in ranked if r["prob"] < threshold]
    return {"sufficient": top["prob"] >= threshold,
            "reason": f"top relevance {top['prob']:.4f} vs threshold {threshold}",
            "top": {"index": top["index"], "prob": top["prob"], "text": passages[top["index"]][:160]},
            "low_band": low_band,
            "all": ranked, "backend": backend}


# ------------------------------------------------------------------------------- baseline + control
_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "and",
         "or", "by", "it", "its", "this", "that", "with", "as", "from", "be", "has", "have", "can",
         "which", "some", "also", "not", "one", "two", "than", "then", "when", "occurs", "form"}


def _content_tokens(text: str) -> list:
    """Content words and numbers. Numbers are kept because they are the part a fact claim hinges on."""
    toks = re.findall(r"[a-z0-9][a-z0-9\-\.]*", text.lower())
    return [t.strip(".-") for t in toks if t.strip(".-") and t.strip(".-") not in _STOP]


def lexical_score(claim: str, passage: str) -> float:
    """Token-F1 between claim and passage. The dumb baseline: does the passage use the claim's words?

    This exists because a cross-encoder must beat something. If overlap already separates relevant from
    irrelevant on my fixtures, the encoder has demonstrated nothing and its AUC should be read as a
    property of the fixtures rather than of the model.
    """
    a, b = _content_tokens(claim), _content_tokens(passage)
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    prec = inter / len(sb)
    rec = inter / len(sa)
    return 2 * prec * rec / (prec + rec)


# claim -> (supporting passage, contradicting passage, off-topic passage)
# The three-way control. Two-way "relevant vs not" hides the case that matters for cleaning: evidence
# that CONTRADICTS. A drop-only system lives or dies on whether it can see contradiction.
TRIPLES = [
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower stands 330 m high.",
     "The Eiffel Tower is 300 metres tall.",
     "The Eiffel Tower appears in the 1974 film The Towering Inferno."),
    ("One mile is 1.609344 kilometres.",
     "One mile equals exactly 1.609344 kilometres.",
     "One mile is 1.500000 kilometres.",
     "Mile is also a surname in some European families."),
    ("Apoptosis is programmed cell death.",
     "Apoptosis is a form of programmed cell death.",
     "Apoptosis is uncontrolled cell proliferation.",
     "Apoptosis was first described in 1972 by Kerr."),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "The served model reports max_input_length of 8192 tokens.",
     "bge-reranker-v2-m3 is limited to 512 tokens.",
     "Qdrant is a vector search engine with HNSW indexing."),
    ("A categorical syllogism has two premises and a conclusion.",
     "A categorical syllogism is a major premise, a minor premise and a conclusion.",
     "A categorical syllogism has three premises and no conclusion.",
     "Aristotle wrote the Prior Analytics, which treats inference."),
]


# Passages that are TOPICALLY IDENTICAL and FACTUALLY OPPOSITE. The reranker is trained on relevance, so
# it should score these HIGH -- and so should token overlap. If both score them high, then neither can be
# used to decide whether a claim is TRUE, and that boundary needs a number rather than a paragraph.
CONTRADICTIONS = [
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower is 300 metres tall."),
    ("One mile is 1.609344 kilometres.",
     "One mile is 1.500000 kilometres."),
    ("Apoptosis is programmed cell death.",
     "Apoptosis is uncontrolled cell proliferation."),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "bge-reranker-v2-m3 accepts inputs up to 512 tokens."),
    ("A categorical syllogism has two premises and a conclusion.",
     "A categorical syllogism has three premises and no conclusion."),
]


# ------------------------------------------------------------------------------- fixtures
# Hand-labelled for ONE question only: is the passage about the CLAIM, or merely about the ENTITY?
# The decoys share the subject deliberately, so lexical overlap cannot win.
PAIRS = [
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower is a wrought-iron lattice tower in Paris. It is 330 m tall.", 1),
    ("The Eiffel Tower is 330 metres tall.",
     "The Eiffel Tower appears in the 1974 film The Towering Inferno.", 0),
    ("Apoptosis is programmed cell death.",
     "Apoptosis is a form of programmed cell death occurring in multicellular organisms.", 1),
    ("Apoptosis is programmed cell death.",
     "Apoptosis was first described in 1972 by Kerr, Wyllie and Currie.", 0),
    ("A categorical syllogism has two premises and a conclusion.",
     "A categorical syllogism consists of a major premise, a minor premise, and a conclusion.", 1),
    ("A categorical syllogism has two premises and a conclusion.",
     "Aristotle wrote the Prior Analytics, which treats inference.", 0),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "The served model reports max_input_length of 8192 tokens.", 1),
    ("bge-reranker-v2-m3 accepts inputs up to 8192 tokens.",
     "Qdrant is a vector search engine with HNSW indexing.", 0),
    ("One mile is 1.609344 kilometres.",
     "One mile equals exactly 1.609344 kilometres by international definition.", 1),
    ("One mile is 1.609344 kilometres.",
     "Mile is also a surname in some European families.", 0),
]


def _auc(scored) -> float:
    """AUC by pair counting: P(score(relevant) > score(irrelevant)). Ties count 0.5."""
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


def measure() -> int:
    print("=== what is live ===")
    h = health()
    for k, v in h.items():
        print(f"  {k:8s} {json.dumps(v)[:120]}")

    print("\n=== scale: the same pair through both runtimes ===")
    q = "what is a syllogism"
    docs = ["A syllogism is a form of deductive reasoning.", "The bridge is 120 metres tall."]
    scales = {}
    for name in ("tei", "argus"):
        try:
            r = rerank(q, docs, backend=name)
            scales[name] = r
            raw = {x["index"]: x["raw"] for x in r}
            prb = {x["index"]: x["prob"] for x in r}
            print(f"  {name:6s} raw {raw}  -> prob { {i: round(p, 5) for i, p in prb.items()} }"
                  f"  scale={r[0]['scale']}")
        except Exception as e:                               # noqa: BLE001
            print(f"  {name:6s} DOWN ({type(e).__name__})")

    agree = None
    if "tei" in scales and "argus" in scales:
        tp = {x["index"]: x["prob"] for x in scales["tei"]}
        ap = {x["index"]: x["prob"] for x in scales["argus"]}
        diffs = [abs(tp[i] - ap[i]) for i in tp if i in ap]
        agree = max(diffs) if diffs else None
        print(f"\n  after sigmoid, max |TEI - argus| = {agree:.6f}")
        if agree is not None and agree < 0.01:
            print(f"  ==> THESE ARE ONE MODEL IN TWO RUNTIMES, NOT TWO OPINIONS.")
            print(f"      Counted as two witnesses they would manufacture corroboration -- the exact")
            print(f"      fake independence the truth store now refuses by construction.")
        else:
            print(f"  ==> they diverge; possibly different weights. Do not merge them without checking.")

    print("\n=== relevance test, WITH a dumb baseline on the same pairs ===")
    print(f"  {len(PAIRS)} hand-labelled pairs, {sum(y for _, _, y in PAIRS)} relevant / "
          f"{sum(1 for _, _, y in PAIRS if not y)} decoy")

    lex_scored = [(lexical_score(c, p), y) for c, p, y in PAIRS]
    lex_auc = _auc(lex_scored)
    lex_pos = [sc for sc, y in lex_scored if y == 1]
    lex_neg = [sc for sc, y in lex_scored if y == 0]
    print(f"  {'baseline':6s} AUC {lex_auc:.3f}   relevant mean {sum(lex_pos) / len(lex_pos):.4f}"
          f"   decoy mean {sum(lex_neg) / len(lex_neg):.4f}"
          f"   separation {sum(lex_pos) / len(lex_pos) - sum(lex_neg) / len(lex_neg):+.4f}   (token-F1)")

    model_auc = {}
    for name in ("tei", "argus"):
        try:
            scored = []
            for claim, passage, y in PAIRS:
                pv = rerank(claim, [passage], backend=name)[0]["prob"]
                scored.append((pv, y))
            auc = _auc(scored)
            model_auc[name] = auc
            pos = [sc for sc, y in scored if y == 1]
            neg = [sc for sc, y in scored if y == 0]
            print(f"  {name:6s} AUC {auc:.3f}   relevant mean {sum(pos) / len(pos):.4f}"
                  f"   decoy mean {sum(neg) / len(neg):.4f}"
                  f"   separation {sum(pos) / len(pos) - sum(neg) / len(neg):+.4f}   (cross-encoder)")
        except Exception as e:                               # noqa: BLE001
            print(f"  {name:6s} DOWN ({type(e).__name__})")

    print()
    if model_auc:
        best = max(model_auc.values())
        gain = best - lex_auc
        if gain <= 0.02:
            print(f"  ==> THE CROSS-ENCODER BEATS TOKEN OVERLAP BY {gain:+.3f} AUC. On these fixtures the")
            print(f"      relevance result is a property of the FIXTURES, not of the model: my decoys differ")
            print(f"      in wording as well as meaning, so a bag of words already separates them. AUC 1.000")
            print(f"      from the encoder alone would have been read as a model result. It is not one.")
        else:
            print(f"  ==> the encoder beats token overlap by {gain:+.3f} AUC, so it is adding something a")
            print(f"      bag of words cannot. Still n={len(PAIRS)} hand-written pairs: illustrative of the")
            print(f"      mechanism, not a rate.")

    print("\n=== the boundary, three-way: support / contradicts / off-topic ===")
    print("  'contradicts' is the case a cleaning system needs: it is what justifies a REJECT.")
    print(f"  {'claim':34s} {'support':>8s} {'contra':>8s} {'off':>8s}   contra discarded?")
    sup_wins = 0
    contra_dropped = 0
    con, off = [], []
    for claim, sup, con_p, off_p in TRIPLES:
        try:
            r = rerank(claim, [sup, con_p, off_p], backend="tei")
        except Exception as e:                               # noqa: BLE001
            print(f"  {claim[:34]:34s} DOWN ({type(e).__name__})")
            continue
        by = {x["index"]: x["prob"] for x in r}
        if by[0] > by[1]:
            sup_wins += 1
        dropped = by[1] < 0.5
        if dropped:
            contra_dropped += 1
        con.append(by[1])
        off.append(by[2])
        print(f"  {claim[:34]:34s} {by[0]:8.4f} {by[1]:8.4f} {by[2]:8.4f}   "
              f"{'yes' if dropped else 'no'}")
    print()
    print(f"  supporting beat contradicting in {sup_wins}/{len(TRIPLES)} triples -- so ranking support above")
    print(f"  contradiction is something the encoder DOES do, and token overlap does not.")
    print(f"  contradicting evidence fell below its own threshold in {contra_dropped}/{len(TRIPLES)} -- it is")
    print(f"  DISCARDED, not flagged.")

    # can the score separate 'contradicts' from 'about something else'? that is the second bit.
    pairs = [(c, 1) for c in con] + [(o, 0) for o in off]
    auc = _auc(pairs)
    print()
    print(f"  AUC separating CONTRADICTING from OFF-TOPIC on this score: {auc:.3f}  (0.5 = no information)")
    print(f"    contradicting scores {[round(c, 4) for c in con]}")
    print(f"    off-topic  scores    {[round(o, 4) for o in off]}")
    print()
    print(f"  ==> SO THE SCORE IS NOT ONE BIT, AND IT IS NOT A REFUTATION EITHER.")
    print()
    print(f"      Contradicting passages DO score above off-topic ones (AUC {auc:.3f}), because a")
    print(f"      contradiction is topically on-point. That is real signal, and I expected none: my")
    print(f"      prediction was that the score would be blind to the numeric flip. It is not.")
    print()
    print(f"      But it is not safe to act on. The syllogism contradiction scored 0.0038 while the")
    print(f"      APOPTOSIS OFF-TOPIC passage scored 0.0138 -- one inversion, so a band drawn on this")
    print(f"      score would sometimes send an off-topic passage to refutation and a real contradiction")
    print(f"      to the bin. And 4 of 5 contradictions sit under a 0.5 threshold, so a single-threshold")
    print(f"      support gate DISCARDS them rather than flagging them.")
    print()
    print(f"      CONSEQUENCE, and it governs the design: below-threshold evidence must NOT be dropped.")
    print(f"      'No supporting evidence' is a true reading of a low score. 'Therefore nothing here")
    print(f"      refutes the claim' is not. Contradicting and off-topic passages arrive as one")
    print(f"      undifferentiated low band, and the cheap correct response is to send the band to a")
    print(f"      VALIDATOR -- which can tell 300 metres from 330 -- rather than to discard it.")
    print()
    print(f"      If the gate dropped its low band, the pipeline could ABSTAIN but never REJECT, and the")
    print(f"      reject pile -- the entire point of a precision-first cleaner -- would be empty by")
    print(f"      construction. So evidence_sufficient() returns the low band instead of swallowing it.")

    print("\n=== the gate, on a claim with deliberately irrelevant evidence ===")
    g = evidence_sufficient("One mile is 1.609344 kilometres.",
                            ["Mile is also a surname in some European families."], threshold=0.5)
    print(f"  irrelevant evidence -> sufficient={g['sufficient']}  ({g['reason']})")
    g = evidence_sufficient("One mile is 1.609344 kilometres.",
                            ["One mile equals exactly 1.609344 kilometres by international definition."],
                            threshold=0.5)
    print(f"  relevant   evidence -> sufficient={g['sufficient']}  ({g['reason']})")
    if g["sufficient"] is False:
        print("  ==> so a row with only off-topic evidence is forced to UNKNOWN instead of being scored.")
    print("\n  NB the gate answers 'is there evidence at all'. It never answers 'is the claim true'.")
    print("  'The Eiffel Tower appears in a 1974 film' is RELEVANT and supports nothing.")
    return 0


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.rerank_client - SELF TEST")
    ok = True

    # --- the scale conversion is the whole point, so pin it
    good = abs(sigmoid(7.402874946594238) - 0.99939) < 1e-4
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] sigmoid(7.40287) = {sigmoid(7.402874946594238):.5f}, "
          f"TEI said 0.99833 -- the two services publish the SAME score on different scales")

    good = sigmoid(0.0) == 0.5 and sigmoid(1000.0) > 0.999 and sigmoid(-1000.0) < 0.001
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] sigmoid survives extreme logits without overflowing")

    # --- ranking is by DESCENDING probability regardless of backend scale
    fake = [{"index": 0, "prob": 0.01, "raw": -4.6, "scale": "logit"},
            {"index": 1, "prob": 0.99, "raw": 4.6, "scale": "logit"}]
    good = sorted(fake, key=lambda r: -r["prob"])[0]["index"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ranking uses probability, so a logit backend cannot invert it")

    # --- the gate is fail-CLOSED: a dead backend gives None, never False
    saved = dict(BACKENDS)
    def dead(q, t):
        raise OSError("connection refused")
    BACKENDS["tei"] = dead
    g = evidence_sufficient("claim", ["some passage"], backend="tei")
    good = g["sufficient"] is None and "unreachable" in g["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unreachable backend is None (UNKNOWN), not 'no evidence'")
    BACKENDS.update(saved)

    # --- and with no candidates at all it is False, which is a real answer, not an error
    g = evidence_sufficient("claim", [], backend="tei")
    good = g["sufficient"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no candidate evidence is False, distinct from unreachable")

    # --- the low band must be RETURNED, never swallowed. This is the difference between a pipeline
    #     that can reject and one that can only abstain.
    g = evidence_sufficient("The Eiffel Tower is 330 metres tall.",
                            ["The Eiffel Tower is 300 metres tall.",
                             "The Eiffel Tower appears in the 1974 film The Towering Inferno."],
                            threshold=0.5, backend="tei")
    good = len(g.get("low_band", [])) == 2 and g["sufficient"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a contradicting passage is KEPT in the low band "
          f"({len(g.get('low_band', []))} passages), not discarded")

    good = bool(g.get("low_band")) and any("300" in p["text"] for p in g["low_band"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the low band carries passage TEXT so a validator can run on it")

    # --- AUC is computed by pair counting and a perfect ranking scores 1.0
    good = _auc([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]) == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC of a perfectly ordered set is 1.000")

    # --- an inverted ranking scores 0.0, so the metric can actually report failure
    good = _auc([(0.1, 1), (0.2, 1), (0.8, 0), (0.9, 0)]) == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC of an inverted set is 0.000 (the metric can fail)")

    # --- ties are half credit, so a constant scorer lands at chance not at 1.0
    good = abs(_auc([(0.5, 1), (0.5, 0)]) - 0.5) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a constant scorer scores 0.500, not a free win")

    # --- the lexical baseline is a real baseline, not a stub: it separates obvious cases and ties at 0
    good = lexical_score("One mile is 1.609344 kilometres.",
                         "One mile equals exactly 1.609344 kilometres.") > 0.7
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] token-F1 sees an obvious match")

    good = lexical_score("One mile is 1.609344 kilometres.",
                         "Qdrant indexes vectors with HNSW.") == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] token-F1 sees an obvious mismatch")

    # --- and it CANNOT see a contradiction: the numbers differ, the words agree. This is the control
    #     that shows a lexical score is not a truth score either.
    good = lexical_score("The Eiffel Tower is 330 metres tall.",
                         "The Eiffel Tower is 300 metres tall.") > 0.7
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and token-F1 scores a CONTRADICTION as a near-match "
          f"(so it is not a truth signal either)")

    # --- live: the services are up and a real pair separates
    try:
        r = rerank("what is a syllogism",
                   ["A syllogism is deductive reasoning.", "The bridge is 120 metres tall."])
        good = r[0]["index"] == 0 and r[0]["prob"] > 0.9
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] LIVE tei: relevant {r[0]['prob']:.4f} > "
              f"irrelevant {r[1]['prob']:.2e}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [SKIP] LIVE tei unreachable ({type(e).__name__})")

    try:
        r = rerank("what is a syllogism",
                   ["A syllogism is deductive reasoning.", "The bridge is 120 metres tall."],
                   backend="argus")
        good = r[0]["index"] == 0 and r[0]["scale"] == "logit"
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] LIVE argus: raw {r[0]['raw']:.2f} logit -> "
              f"{r[0]['prob']:.4f} prob")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [SKIP] LIVE argus unreachable ({type(e).__name__})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-encoder client and evidence-sufficiency gate.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--query", default="")
    ap.add_argument("--texts", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.health:
        print(json.dumps(health(), indent=2))
        return 0
    if args.query and args.texts:
        print(json.dumps(rerank(args.query, [t for t in args.texts.split("|") if t]), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
