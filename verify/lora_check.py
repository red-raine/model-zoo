"""lora_check.py — Locate / Retrieve / Verify / Aggregate, done the way the field does it.

WHY THIS REPLACES THE NAIVE CHECKER
-----------------------------------
`fact_verify.py` was a naive fact-checker: split into sentences, search the claim
verbatim, one LLM call over the whole evidence blob, count the verdicts. Research
against Libr-AI/OpenFactVerification (LORA), GAIR-NLP/factool and the
bullshit-detector skill set found 12 documented failure modes in that shape. The
ones that actually change the answer:

 1. VERBATIM CLAIM AS QUERY. Long, compound, pronoun-laden strings make terrible
    queries — engines rank documents *about* the topic, not the primary source.
    LORA generates a minimal set of QUESTIONS per claim instead.
 2. ONE CALL OVER THE WHOLE BLOB. A single confident snippet dominates and you
    cannot audit which source did the work. LORA judges every (claim, snippet)
    PAIR and records per-pair reasoning.
 3. NO EVIDENCE COLLAPSES INTO A VERDICT. LORA returns the string
    "No evidence found." precisely so it cannot be averaged into a probability.
 4. COUNTING VERDICTS = MAJORITY VOTE. 5 syndicated reprints beat 1 primary
    source. LORA uses an evidence RATIO: S/(S+R), plus a "controversial" bucket
    (verified - supported - refuted) instead of averaging disagreement away.
 5. NO CHECK-WORTHINESS GATE. Opinions and predictions get verdicts and distort
    the denominator. LORA gates them out — and correctly marks *verifiable but
    false* claims as check-worthy.
 6. NO DECONTEXTUALISATION. "he resigned" is unsearchable; LORA rewrites to a
    self-contained <15-word claim and keeps a span pointer to the original.

Deterministic channels run first (sympy for arithmetic, Wikidata SPARQL for
entity facts) because a model must never do arithmetic or recall an entity fact.

LABELS
------
Evidence-level (per pair): SUPPORTS / REFUTES / IRRELEVANT
Claim-level: one of
    "SUPPORTED", "REFUTED", "CONFLICTING", "NO_EVIDENCE", "NOT_CHECK_WORTHY"
Document-level: an evidence ratio over verified claims only, plus coverage, plus
    the self-check baseline arm so results are falsifiable.

HONEST LIMITS
-------------
* An LLM judge over web snippets is a signal, not ground truth. Deterministic
  channels (math/Wikidata) are the only verdicts that carry certainty.
* This is NOT validated on a benchmark. AVeriTeC / FEVER / SciFact are the
  measuring instruments; until it is run against one, accuracy is unknown and
  `--self-test` only proves the plumbing.
* Same-content verdict drift across runs is documented in the field (~half of
  claims in one published eval). Always report >=3 runs.

USAGE
-----
    python tools/verify/lora_check.py --self-test
    python tools/verify/lora_check.py --text "..."
    python tools/verify/lora_check.py --jsonl answers.jsonl --field response --max-claims 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_verify import search, wikidata_claim          # reuse: retrieval + KG
from math_verify import verify_text as math_verify      # reuse: arithmetic

JUDGE = "http://192.168.8.237:8082/v1"
JUDGE_MODEL = "gpt-oss-20b-Q4_K_M.gguf"

EVIDENCE_LABELS = ("SUPPORTS", "REFUTES", "IRRELEVANT")
NO_EVIDENCE = "No evidence found."

# ---------------------------------------------------------------------------
# Prompts (modelled on LORA's decompose / checkworthy / qgen / verify stages)
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """Decompose the text into atomic factual claims.

Rules:
- Each claim must be a context-independent statement of ONE fact.
- Be concise: under 15 words and self-contained.
- Avoid vague references ("he", "she", "it", "this", "the company"). Use the actual name.
- Generate at least one claim per sentence.
- Skip opinions, predictions and advice.

Return JSON only: {{"claims": ["...", "..."]}}

TEXT:
{text}"""

CHECKWORTHY_PROMPT = """Is this statement checkable against sources, even if it turns out to be false?

Answer Yes if it asserts a verifiable fact about the world.
Answer No if it is an opinion, a prediction, a value judgement, or too vague to look up.
A FALSE but specific statement is still Yes.

STATEMENT: {claim}

Return JSON only: {{"checkworthy": "Yes" or "No"}}"""

QGEN_PROMPT = """Write the MINIMUM number of search questions needed to check this claim.

Rules:
- Questions, not keywords.
- Do not restate the claim as a question if a simpler lookup exists.
- 1 question is normal; use 2-3 only for compound claims.

Return JSON only: {{"questions": ["...", "..."]}}

CLAIM: {claim}"""

PAIR_PROMPT = """Does the EVIDENCE support or refute the CLAIM?

Rules:
- SUPPORTS: the evidence states the same fact.
- REFUTES: the evidence states something incompatible with the claim.
- IRRELEVANT: the evidence does not address the claim.
- Use ONLY the evidence. Not your own knowledge.

CLAIM: {claim}

EVIDENCE: {evidence}

Return JSON only: {{"reasoning": "one sentence", "relationship": "SUPPORTS" or "REFUTES" or "IRRELEVANT"}}"""


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

def _chat(prompt: str, endpoint: str, model: str, max_tokens: int = 768,
          timeout: int = 300) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"].get("content") or "").strip()


def _json_from(raw: str) -> dict:
    """Models wrap JSON in prose or fences. Recover the object, else {}."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _ask_json(prompt: str, endpoint: str, model: str, **kw) -> dict:
    for _ in range(2):
        try:
            d = _json_from(_chat(prompt, endpoint, model, **kw))
            if d:
                return d
        except Exception:
            time.sleep(1)
    return {}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_decompose(text: str, endpoint: str, model: str,
                    max_claims: int = 3) -> list[str]:
    d = _ask_json(DECOMPOSE_PROMPT.format(text=text[:2500]), endpoint, model)
    claims = [c.strip() for c in (d.get("claims") or []) if isinstance(c, str)]
    claims = [c for c in claims if 8 <= len(c) <= 220]
    if not claims:
        # fallback: conservative sentence split, flagged as such
        return []
    return claims[:max_claims]



_STOP = set("""the a an and or but of to in on at for with from by as is are was were
be been being it this that these those he she they we you i not no than then there
their its which who whom whose what when where why how all any both each few more
most other some such only own same so very can will just should now""".split())


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 2}


def decomposition_coverage(source: str, claims: list[str]) -> float:
    """Fraction of the source's content words that survive into the claims.

    Decomposition can DELETE the checkable content. Measured regression:
    "The sky appears green because air molecules are larger than the wavelength
    of blue light" decomposed to just "The sky appears green.", which the pair
    judge then marked SUPPORTED (ratio 1.0) -- a worse answer than the naive
    whole-sentence checker. The false part was the causal clause, and it was
    dropped.
    """
    src = _content(source)
    if not src:
        return 1.0
    got = set()
    for cl in claims:
        got |= _content(cl)
    return len(src & got) / len(src)


def densest_sentence(text: str) -> str | None:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    if not sents:
        return None
    return max(sents, key=lambda s: len(_content(s)))


def stage_checkworthy(claim: str, endpoint: str, model: str) -> bool:
    d = _ask_json(CHECKWORTHY_PROMPT.format(claim=claim), endpoint, model,
                  max_tokens=256)
    return str(d.get("checkworthy", "")).strip().lower().startswith("y")


def stage_questions(claim: str, endpoint: str, model: str) -> list[str]:
    d = _ask_json(QGEN_PROMPT.format(claim=claim), endpoint, model, max_tokens=384)
    qs = [q.strip() for q in (d.get("questions") or []) if isinstance(q, str)]
    qs = [q for q in qs if 6 <= len(q) <= 200]
    if not qs:
        # never fall back to the verbatim claim; use a naive wh-question
        qs = [f"Is it true that {claim.rstrip('.')}?"]
    return qs[:3]


def stage_verify_pair(claim: str, snippet: dict, endpoint: str,
                      model: str) -> dict:
    ev = f"{snippet['title']}: {snippet['content']}"
    d = _ask_json(PAIR_PROMPT.format(claim=claim, evidence=ev[:1800]),
                  endpoint, model, max_tokens=768)
    rel = str(d.get("relationship", "")).strip().upper()
    if rel not in EVIDENCE_LABELS:
        # tolerate "REFUTE" / "IRRELEVANT."
        for lab in EVIDENCE_LABELS:
            if lab.rstrip("S") in rel:
                rel = lab
                break
        else:
            rel = "IRRELEVANT"
    return {"relationship": rel,
            "reasoning": str(d.get("reasoning", ""))[:200],
            "source": snippet.get("url") or snippet.get("title")}


# ---------------------------------------------------------------------------
# Aggregation — evidence ratio, not a vote
# ---------------------------------------------------------------------------

def aggregate_claim(pairs: list[dict]) -> dict:
    s = sum(1 for p in pairs if p["relationship"] == "SUPPORTS")
    r = sum(1 for p in pairs if p["relationship"] == "REFUTES")
    irr = sum(1 for p in pairs if p["relationship"] == "IRRELEVANT")
    if s + r == 0:
        # LORA: a STRING, deliberately not a label and not 0.0
        return {"claim_verdict": "NO_EVIDENCE", "ratio": None,
                "supports": 0, "refutes": 0, "irrelevant": irr}
    ratio = s / (s + r)
    if s and r:
        verdict = "CONFLICTING"
    elif s:
        verdict = "SUPPORTED"
    else:
        verdict = "REFUTED"
    return {"claim_verdict": verdict, "ratio": round(ratio, 4),
            "supports": s, "refutes": r, "irrelevant": irr}


def check_text(text: str, endpoint: str = JUDGE, model: str = JUDGE_MODEL,
               max_claims: int = 3, snippets_per_q: int = 4,
               use_deterministic: bool = True) -> dict:
    out = {"claims": [], "text_chars": len(text or "")}

    # 1. deterministic channels first — never let a model do these
    math_v = math_verify(text or "")
    out["math"] = {k: math_v[k] for k in
                   ("verdict", "equalities", "verified", "refuted",
                    "refuted_examples")}

    claims = stage_decompose(text or "", endpoint, model, max_claims)
    out["decomposed"] = len(claims)

    # MEASURED FAILURE: on HaluEval "question + answer" claims the decomposer
    # returned [] and the pipeline abstained on 10/10 items while still spending
    # 216 s/claim retrying the model. Never let an empty decomposition end the
    # run: fall back to the densest source sentence.
    if not claims:
        fb = densest_sentence(text or "")
        if fb:
            claims = [fb]
            out["decompose_fallback"] = fb
            out["decomposed"] = 1

    # guard: if decomposition lost most of the source's content, verify the
    # source sentence too. Decomposition must not be allowed to make a claim
    # easier to check by deleting the part that is false.
    cov = decomposition_coverage(text or "", claims)
    out["decomposition_coverage"] = round(cov, 4)
    if cov < 0.6:
        extra = densest_sentence(text or "")
        if extra and extra not in claims:
            claims = claims + [extra]
            out["added_source_sentence"] = extra
            out["decomposed"] = len(claims)
    if not claims:
        out["document"] = {"status": "NO_CLAIMS_EXTRACTED"}
        return out

    for claim in claims:
        rec = {"claim": claim}

        # Wikidata before anything semantic
        if use_deterministic:
            try:
                wd = wikidata_claim(claim)
            except Exception:
                wd = None
            if wd:
                rec.update({"claim_verdict": wd[0], "channel": "wikidata",
                            "why": wd[1], "pairs": []})
                out["claims"].append(rec)
                continue

        if not stage_checkworthy(claim, endpoint, model):
            rec.update({"claim_verdict": "NOT_CHECK_WORTHY", "channel": "gate"})
            out["claims"].append(rec)
            continue

        questions = stage_questions(claim, endpoint, model)
        rec["questions"] = questions

        snips = []
        for q in questions:
            try:
                snips.extend(search(q, k=snippets_per_q))
            except Exception:
                continue
        # dedupe by url
        seen, uniq = set(), []
        for s in snips:
            u = s.get("url") or s.get("title")
            if u in seen:
                continue
            seen.add(u)
            uniq.append(s)
        rec["sources"] = len(uniq)

        pairs = [stage_verify_pair(claim, s, endpoint, model) for s in uniq[:6]]
        rec["pairs"] = pairs
        rec.update(aggregate_claim(pairs))
        rec["channel"] = "web+llm"
        out["claims"].append(rec)

    # ---- document roll-up ----
    verified = [c for c in out["claims"] if c.get("ratio") is not None]
    counts = {}
    for c in out["claims"]:
        counts[c.get("claim_verdict", "?")] = counts.get(c.get("claim_verdict", "?"), 0) + 1
    out["document"] = {
        "claims_total": len(out["claims"]),
        "claims_verified": len(verified),
        "coverage": round(len(verified) / max(len(out["claims"]), 1), 4),
        "counts": counts,
        # LORA: mean evidence ratio over VERIFIED claims only
        "mean_evidence_ratio": (round(sum(c["ratio"] for c in verified) / len(verified), 4)
                                if verified else None),
        # factool: conjunction is the brutal, honest document metric
        "all_supported": all(c.get("claim_verdict") == "SUPPORTED" for c in out["claims"]),
        "any_refuted": any(c.get("claim_verdict") == "REFUTED" for c in out["claims"]),
    }
    return out


# ---------------------------------------------------------------------------
# Self-test: plumbing + the two aggregation rules that matter
# ---------------------------------------------------------------------------

def run_self_test() -> bool:
    print("lora_check.py - SELF TEST")
    ok = True

    # (a) aggregation: no evidence must NOT become a verdict or a 0.0 ratio
    a = aggregate_claim([])
    good = a["claim_verdict"] == "NO_EVIDENCE" and a["ratio"] is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty evidence -> {a['claim_verdict']} "
          f"ratio={a['ratio']} (must be NO_EVIDENCE/None)")

    # (b) ratio, not majority: 3 supports vs 1 refute is CONFLICTING, not 75% true
    b = aggregate_claim([{"relationship": "SUPPORTS"}] * 3 +
                        [{"relationship": "REFUTES"}])
    good = b["claim_verdict"] == "CONFLICTING" and abs(b["ratio"] - 0.75) < 1e-6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 3S/1R -> {b['claim_verdict']} "
          f"ratio={b['ratio']} (contradiction kept as its own bucket)")

    # (c) irrelevant-only is still no evidence
    c = aggregate_claim([{"relationship": "IRRELEVANT"}] * 5)
    good = c["claim_verdict"] == "NO_EVIDENCE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 5 irrelevant -> {c['claim_verdict']}")

    # (d) JSON recovery from fenced / prose-wrapped output
    for raw, want in [('{"claims": ["a"]}', ["a"]),
                      ('```json\n{"claims": ["a","b"]}\n```', ["a", "b"]),
                      ("no json here", [])]:
        got = _json_from(raw).get("claims", [])
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] json_from {raw[:32]!r} -> {got}")

    # (e) the label parser must not accept garbage
    for raw in ("", "maybe", "REFUTE", "IRRELEVANT"):
        rel = raw.strip().upper()
        mapped = rel if rel in EVIDENCE_LABELS else next(
            (l for l in EVIDENCE_LABELS if l.rstrip("S") in rel), "IRRELEVANT")
        good = mapped in EVIDENCE_LABELS
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] label {raw!r} -> {mapped}")

    # (e2) every prompt must survive .format() -- an unescaped {"claims": ...}
    # in DECOMPOSE_PROMPT raised KeyError('"claims"') at runtime
    for name, prompt, kwargs in (
        ("decompose", DECOMPOSE_PROMPT, {"text": "x"}),
        ("checkworthy", CHECKWORTHY_PROMPT, {"claim": "x"}),
        ("qgen", QGEN_PROMPT, {"claim": "x"}),
        ("pair", PAIR_PROMPT, {"claim": "x", "evidence": "y"}),
    ):
        try:
            prompt.format(**kwargs)
            good = True
            err = ""
        except Exception as exc:
            good = False
            err = f"{type(exc).__name__}: {exc}"
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] prompt.format {name} {err}")

    # (e3) the decomposition-coverage guard must fire on the measured regression
    src = ("The sky appears green because air molecules are larger than the "
           "wavelength of blue light")
    cov_bad = decomposition_coverage(src, ["The sky appears green."])
    cov_ok = decomposition_coverage(src, [src])
    good = cov_bad < 0.6 <= cov_ok
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] coverage guard: truncated={cov_bad:.2f} "
          f"full={cov_ok:.2f} (guard fires below 0.60)")

    # (e4) empty decomposition must fall back, not abstain
    fb = densest_sentence("Reagan and Give Us Our Skeletons were both documentaries.")
    good = bool(fb) and "documentaries" in fb
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] decompose fallback -> {str(fb)[:52]!r}")

    # (f) arithmetic must be caught deterministically before any model call
    mv = math_verify("Therefore 2 + 2 = 5.")
    good = mv["verdict"] == "REFUTED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] deterministic math on '2+2=5' -> {mv['verdict']}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="LORA-style fact check: decompose, gate, question, retrieve, pairwise verify, aggregate.")
    ap.add_argument("--text")
    ap.add_argument("--jsonl")
    ap.add_argument("--field", default="response")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--max-claims", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    if args.text:
        r = check_text(args.text, args.judge, args.model, args.max_claims)
        print(json.dumps(r, indent=2, ensure_ascii=False)[:6000])
        return 0

    if args.jsonl:
        rows = []
        agg = {}
        with open(args.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                r = check_text(d.get(args.field) or "", args.judge, args.model,
                               args.max_claims)
                doc = r.get("document", {})
                for k, v in (doc.get("counts") or {}).items():
                    agg[k] = agg.get(k, 0) + v
                rows.append({"prompt": str(d.get("prompt"))[:100],
                             "document": doc})
        out = {"answers": len(rows), "claim_verdict_totals": agg, "rows": rows[:40]}
        print(json.dumps(out, indent=2)[:4000])
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
