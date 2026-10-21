"""fact_verify.py — check declarative claims against live web evidence.

WHY THIS FILE EXISTS
--------------------
TRACE gave "The sky appears green, therefore ... since air molecules are larger
than the wavelength of blue light" a score of 0.721 (grade B) because it counts
connectives and never looks at the claim. A correct explanation with no
connectives scored 0.133 (grade D).

Math is checkable deterministically (`math_verify.py`). Facts need evidence.
This module extracts atomic factual claims, retrieves sources from the local
SearXNG instance, and asks a local model to classify the claim against that
evidence.

HONEST LIMITS — read before trusting a number from this
-------------------------------------------------------
* An LLM judge over web snippets is a *signal*, not ground truth. It fails on
  contested claims, on claims needing specialist knowledge, and when retrieval
  returns nothing good.
* UNKNOWN is the most common honest answer and must not be reported as a pass.
* It is only as good as the retrieval. A claim with no good sources is
  unverifiable, not false.
* Agreement with a hand-labelled set must be measured before this can gate
  anything. `--agreement` exists for that and is not yet run.

USAGE
-----
    python tools/verify/fact_verify.py --self-test
    python tools/verify/fact_verify.py --text "The sky is green."
    python tools/verify/fact_verify.py --jsonl answers.jsonl --field response --max-claims 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

SEARX = "http://localhost:8888/search"
JUDGE = "http://192.168.8.237:8082/v1"      # nyx gpt-oss-20b (override with --judge)
JUDGE_MODEL = "gpt-oss-20b-Q4_K_M.gguf"

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")

# Sentences that are not checkable claims.
_SKIP = re.compile(
    r"^(?:i|we|you|let|here|this (?:is|means)|that (?:is|means)|note|"
    r"first|second|third|next|then|finally|step\s*\d+|in summary|overall)\b", re.I)

# Verb whitelist. Deliberately broad: an earlier version omitted "boils", so
# "Water boils at 100 C" was silently dropped and never checked.
_CLAIMY = re.compile(
    r"\b(is|are|was|were|has|have|had|does|do|did|can|cannot|will|would|"
    r"causes?|caused|makes?|made|equals?|contains?|consists?|produces?|"
    r"requires?|prevents?|increases?|decreases?|scatters?|absorbs?|emits?|"
    r"reflects?|boils?|freezes?|melts?|flows?|grows?|falls?|rises?|occurs?|"
    r"happens?|exists?|lies?|sits?|stands?|works?|fails?|runs?|needs?|uses?|"
    r"takes?|gives?|adds?|removes?|holds?|keeps?|turns?|becomes?|stays?|"
    r"remains?|depends?|belongs?|borders?|located|measures?|weighs?|"
    r"appears?|looks?|seems?)\b", re.I)


def extract_claims(text: str, max_claims: int = 3) -> list[str]:
    """Pick short, declarative, checkable sentences. Deliberately conservative."""
    if not text:
        return []
    text = re.sub(r"```.*?```", " ", text, flags=re.S)      # drop code blocks
    text = re.sub(r"`[^`]*`", " ", text)
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for s in sents:
        s = s.strip(" -•*\t")
        if not (14 <= len(s) <= 240):
            continue
        if s.endswith("?"):
            continue
        if _SKIP.match(s):
            continue
        if not _CLAIMY.search(s):
            continue
        if s.lower().startswith(("therefore", "however", "moreover", "thus", "so ")):
            continue
        out.append(s)
        if len(out) >= max_claims:
            break
    return out


_LEAD = re.compile(r"^(?:and|but|so|then|also|however|therefore|moreover|"
                   r"since|because|although|while|which|that)\b[\s,]*", re.I)
_CONJ = re.compile(r"\b(?:and|but|because|which|while|so that|although)\b", re.I)


_TOKEN = re.compile(r"[A-Za-z][\w'\-]*")
_STOP_Q = set("""the a an and or but of to in on at for with from by as is are was were
be been being it this that these those he she they we you i not no than then there
their its which who whom whose what when where why how all any both each few more
most other some such only own same very just now also into about over under again
were was been""".split())


def entities(claim: str) -> list[str]:
    """Capitalised phrases = the things worth looking up.

    Measured: searching the whole claim "Reagan and Give Us Our Skeletons were
    both documentaries" made Wikipedia return Michael Jackson, RoboCop and
    "Internment of Japanese Americans". The claim is a sentence; a search engine
    wants names. Splitting on "and" keeps two entities apart instead of gluing
    them into one unsearchable string.
    """
    out = []
    for part in re.split(r"\band\b|\bor\b|[,;]", claim):
        cur = []
        for tok in _TOKEN.findall(part):
            if tok[0].isupper():
                cur.append(tok)
            else:
                if cur:
                    out.append(" ".join(cur))
                    cur = []
        if cur:
            out.append(" ".join(cur))
    seen, uniq = set(), []
    for e in out:
        e = e.strip()
        if len(e) > 2 and e.lower() not in seen:
            seen.add(e.lower())
            uniq.append(e)
    return uniq


def predicate_terms(claim: str) -> list[str]:
    """Content words left after removing entities and stopwords."""
    ents = {w.lower() for e in entities(claim) for w in e.split()}
    toks = [t.lower() for t in _TOKEN.findall(claim)]
    return [t for t in toks if t not in _STOP_Q and t not in ents and len(t) > 2]


def _singular(word: str) -> str:
    w = word.strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def split_conjunction(claim: str) -> list[str]:
    """Split "A and B are both X" into one atomic claim per entity.

    Measured failure: "Reagan and Give Us Our Skeletons were both documentaries"
    was judged CONTRADICTED, because the evidence retrieved for the query
    "Reagan documentaries" is about the 2024 Reagan *biopic*, not the 2011
    documentary. Judging a two-entity claim against one entity's evidence is
    invalid no matter how good the judge is.

    Atomic claims are the standard fix (LORA/Loki decompose; Factool aggregates
    by conjunction). Returns [claim] unchanged when there is nothing to split.
    """
    ents = entities(claim)
    if len(ents) < 2:
        return [claim]
    preds = predicate_terms(claim)
    if not preds:
        return [claim]
    head = _singular(preds[0])
    out = []
    for e in ents:
        out.append(f"{e} is a {head}" if not head.endswith("s") else f"{e} is {head}")
    return out


def _query_for(claim: str) -> list[str]:
    """Progressive queries: entity+predicate, then entity alone, then the claim.

    Never the verbatim claim first: a sentence is not a query.
    """
    cands = []
    ents = entities(claim)
    preds = predicate_terms(claim)
    head = preds[0] if preds else ""
    for e in ents:
        if head:
            cands.append(f"{e} {head}")
    for e in ents:
        cands.append(e)
    if head:
        cands.append(head)
    cands.append(_LEAD.sub("", claim.strip().rstrip(".!?")))

    seen, out = set(), []
    for q in cands:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:6]



# ---------------------------------------------------------------------------
# Deterministic entity checks via Wikidata.
#
# Web search is the WRONG tool for "Delaware is in China": searching a false
# claim returns pages about the topic, not pages refuting it. Wikidata answers
# it exactly -- Delaware P17 (country) = United States.
#
# NOTE: this block was accidentally deleted once when replacing _query_for.
# Keep it BELOW _query_for and ABOVE search, and run the self-test after any
# edit to that region.
# ---------------------------------------------------------------------------

WD_API = "https://www.wikidata.org/w/api.php"
WD_SPARQL = "https://query.wikidata.org/sparql"
_WD_UA = {"User-Agent": "forge-factcheck/1.0 (research; local model zoo)"}

_IN_LOC = re.compile(
    r"^(?:the\s+)?(?P<x>[A-Z][\w'\- ]{1,40}?)\s+"
    r"(?:is located|is situated|is|lies|sits|belongs)\s+"
    r"(?:in|within|inside|part of|a state in|a province of)\s+"
    r"(?:the\s+)?(?P<y>[A-Z][\w'\- ]{1,40})", re.I)
_CAPITAL = re.compile(
    r"^(?:the\s+)?(?P<x>[A-Z][\w'\- ]{1,40}?)\s+is the capital of\s+"
    r"(?:the\s+)?(?P<y>[A-Z][\w'\- ]{1,40})", re.I)

_CONTAINMENT_PROPS = ("P17", "P131", "P30", "P361")

LAST_WD_ERROR = ""


def _wd_get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=_WD_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _wd_qid(name: str):
    url = (f"{WD_API}?action=wbsearchentities&format=json&language=en&limit=1"
           f"&search={urllib.parse.quote(name.strip())}")
    d = _wd_get(url, 20)
    s = d.get("search") or []
    return (s[0]["id"], s[0].get("label")) if s else (None, None)


def _wd_labels(qid: str, props: tuple) -> set:
    """Labels of the entities QID points at through PROPS.

    A first version selected ?label while binding ?v0..?vN, so the variable was
    never bound: Wikidata returned HTTP 400 and a bare `except: return None`
    hid it, silently disabling the whole KG channel.
    """
    global LAST_WD_ERROR
    values = " ".join(f"wdt:{pp}" for pp in props)
    q = ("SELECT ?label WHERE { VALUES ?p { " + values + " } "
         f"wd:{qid} ?p ?item . ?item rdfs:label ?label . "
         'FILTER(lang(?label) = "en") } LIMIT 30')
    url = f"{WD_SPARQL}?format=json&query={urllib.parse.quote(q)}"
    try:
        d = _wd_get(url, 30)
    except Exception as exc:
        LAST_WD_ERROR = f"{type(exc).__name__}: {str(exc)[:120]}"
        return set()
    return {b["label"]["value"].lower()
            for b in d["results"]["bindings"] if b.get("label", {}).get("value")}


def wikidata_claim(claim: str):
    """Deterministic verdict for entity-containment / capital claims."""
    m = _CAPITAL.search(claim.strip())
    if m:
        x, y = m.group("x"), m.group("y")
        try:
            xq, _ = _wd_qid(x)
            yq, _ = _wd_qid(y)
            if not (xq and yq):
                return None
            caps = _wd_labels(yq, ("P36",))
            if not caps:
                return None
            ok = any(x.lower() in c or c in x.lower() for c in caps)
            return ("SUPPORTED" if ok else "CONTRADICTED",
                    f"wikidata: capitals of {y} = {sorted(caps)[:4]}")
        except Exception:
            return None

    m = _IN_LOC.search(claim.strip())
    if m:
        x, y = m.group("x"), m.group("y")
        try:
            xq, xlab = _wd_qid(x)
            if not xq:
                return None
            rels = _wd_labels(xq, _CONTAINMENT_PROPS)
            if not rels:
                if LAST_WD_ERROR:
                    print(f"[wikidata] {LAST_WD_ERROR}", file=sys.stderr)
                return None
            yl = y.lower()
            ok = any(yl in rl or rl in yl for rl in rels)
            return ("SUPPORTED" if ok else "CONTRADICTED",
                    f"wikidata: {xlab} is in {sorted(rels)[:5]}")
        except Exception:
            return None
    return None


def search(query: str, k: int = 5, timeout: int = 15) -> list[dict]:
    """Retrieval via the provider ladder, not a single fragile backend.

    The original version called SearXNG directly. When every SearXNG engine got
    suspended (brave/google "too many requests", DuckDuckGo CAPTCHA, youtube 403)
    this returned [] for every query, the verifier abstained on 10/10 items, and
    the abstention looked honest. `retrieval.search` walks wikipedia -> wikidata
    -> searxng so one blocked provider degrades quality instead of deleting the
    capability.
    """
    try:
        from retrieval import search as ladder_search
    except Exception:
        ladder_search = None

    if ladder_search is not None:
        try:
            got = ladder_search(query, k=k)
            if got:
                return [{"title": (s.get("title") or "")[:160],
                         "url": s.get("url") or "",
                         "content": (s.get("content") or "")[:600]}
                        for s in got]
        except Exception:
            pass

    # last resort: the original single-provider path
    url = f"{SEARX}?q={urllib.parse.quote(query)}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "forge-factcheck/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    snips = []
    for item in (d.get("results") or [])[:k]:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        snips.append({"title": (item.get("title") or "")[:160],
                      "url": item.get("url") or "",
                      "content": content[:600]})
    return snips


_JUDGE_PROMPT = """You are a strict fact-checker. Decide whether the EVIDENCE supports or contradicts the CLAIM.

Rules:
- SUPPORTED: the evidence states the same fact, explicitly.
- CONTRADICTED: the evidence states something incompatible with the claim.
- UNKNOWN: the evidence does not address the claim, or is too vague.
- Do not use your own knowledge. Only the evidence counts.
- If the claim is partly right and partly wrong, answer CONTRADICTED.

CLAIM: {claim}

EVIDENCE:
{evidence}

Answer with exactly one word: SUPPORTED, CONTRADICTED, or UNKNOWN."""


def _judge_call(claim: str, ev: str, endpoint: str, model: str,
                max_tokens: int, timeout: int) -> tuple[str, str, str]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": _JUDGE_PROMPT.format(
            claim=claim, evidence=ev)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    return ((ch["message"].get("content") or "").strip(),
            ch.get("finish_reason") or "",
            str(len(ch["message"].get("reasoning_content") or "")))


def judge(claim: str, snips: list[dict], endpoint: str, model: str,
          timeout: int = 300, max_tokens: int = 512) -> tuple[str, str]:
    """Reasoning models burn the budget on reasoning and return empty content.

    gpt-oss-20b at max_tokens=12 returned content='' finish=length with 35
    reasoning tokens; at 300 it returned 'CONTRADICTED'. So: give a real budget,
    and if content is still empty, retry once with double.
    """
    if not snips:
        return "UNKNOWN", "no evidence retrieved"
    ev = "\n".join(f"[{i+1}] {s['title']}: {s['content']}"
                   for i, s in enumerate(snips))
    try:
        raw, finish, rlen = _judge_call(claim, ev[:3500], endpoint, model,
                                        max_tokens, timeout)
        if not raw and finish == "length":
            raw, finish, rlen = _judge_call(claim, ev[:3500], endpoint, model,
                                            max_tokens * 2, timeout)
    except Exception as exc:
        return "UNKNOWN", f"judge-error: {type(exc).__name__}"
    if not raw:
        return "UNKNOWN", f"empty-content(finish={finish},reasoning={rlen})"
    return parse_verdict(raw), raw[:80]


def parse_verdict(raw: str) -> str:
    """Strict parse. Ambiguity becomes UNKNOWN rather than a lucky match."""
    u = (raw or "").upper()
    hits = [v for v in VERDICTS if v in u]
    if len(hits) == 1:
        return hits[0]
    return "UNKNOWN"


def verify_text(text: str, endpoint: str, model: str, max_claims: int) -> dict:
    claims = extract_claims(text, max_claims)
    # atomic decomposition: a conjunction cannot be judged against one entity's evidence
    atomic: list[str] = []
    for cl in claims:
        atomic.extend(split_conjunction(cl))
    claims = atomic[:max(max_claims, len(atomic))] if atomic else claims
    results = []
    for c in claims:
        # deterministic first: Wikidata answers entity claims exactly
        wd = None
        try:
            wd = wikidata_claim(c)
        except Exception:
            wd = None
        if wd:
            results.append({"claim": c, "verdict": wd[0], "why": wd[1],
                            "sources": 1, "query": "(wikidata)", "top": "Wikidata"})
            continue

        snips, used_q = [], c
        try:
            for q in _query_for(c):
                for attempt in range(3):
                    try:
                        snips = search(q)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1.5 * (attempt + 1))
                used_q = q
                if snips:
                    break
        except Exception as exc:
            results.append({"claim": c, "verdict": "UNKNOWN",
                            "why": f"search-error: {type(exc).__name__}", "sources": 0})
            continue
        if not snips:
            # honest: unverifiable is not the same as false
            results.append({"claim": c, "verdict": "UNKNOWN",
                            "why": "no_sources", "sources": 0, "query": used_q})
            continue
        v, why = judge(c, snips, endpoint, model)
        results.append({"claim": c, "verdict": v, "why": why,
                        "sources": len(snips), "query": used_q,
                        "top": snips[0]["title"] if snips else None})
    counts = {k: sum(1 for r in results if r["verdict"] == k) for k in VERDICTS}
    return {
        "claims_found": len(claims),
        "counts": counts,
        "verdict": ("CONTRADICTED" if counts["CONTRADICTED"] else
                    "SUPPORTED" if counts["SUPPORTED"] and not counts["CONTRADICTED"] else
                    "UNKNOWN"),
        "results": results,
    }


# --- self-test: plumbing only, no network ----------------------------------
def run_self_test() -> bool:
    print("fact_verify.py - SELF TEST (plumbing; judge quality is measured separately)")
    ok = True

    # NOTE: an earlier version of this test compared len(got)==len(expect), which
    # passed while silently dropping "The sky is blue." -- the exact claim we care
    # about. Compare CONTENT now.
    cases = [
        ("The sky is blue. It scatters light because of air.",
         ["The sky is blue.", "It scatters light because of air."]),
        ("Step 1: inspect the repo. Then run the tests.", []),
        ("Is the sky blue?", []),
        ("```python\ndef f(): return 1\n```", []),
    ]
    for text, expect in cases:
        got = extract_claims(text, 3)
        good = got[:len(expect)] == expect and len(got) == len(expect)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] extract {text[:40]!r} -> {got}")

    vcases = [
        ("SUPPORTED", "SUPPORTED"),
        ("Contradicted.", "CONTRADICTED"),
        ("UNKNOWN", "UNKNOWN"),
        ("The evidence is SUPPORTED in part but CONTRADICTED elsewhere", "UNKNOWN"),
        ("", "UNKNOWN"),
        ("I think it is probably true", "UNKNOWN"),
    ]
    for raw, expect in vcases:
        got = parse_verdict(raw)
        good = got == expect
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] parse {raw[:44]!r} -> {got} (want {expect})")

    # no evidence must never become SUPPORTED
    v, _ = judge("x", [], "http://127.0.0.1:1/v1", "none")
    good = v == "UNKNOWN"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no-evidence -> {v} (must be UNKNOWN)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify factual claims against web evidence.")
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
        print(json.dumps(verify_text(args.text, args.judge, args.model,
                                     args.max_claims), indent=2, ensure_ascii=False))
        return 0

    if args.jsonl:
        agg = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}
        rows = []
        n = 0
        with open(args.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                r = verify_text(d.get(args.field) or "", args.judge, args.model,
                                args.max_claims)
                n += 1
                agg[r["verdict"]] += 1
                if r["verdict"] == "CONTRADICTED":
                    rows.append({"prompt": str(d.get("prompt"))[:100],
                                 "bad": [x for x in r["results"]
                                         if x["verdict"] == "CONTRADICTED"]})
        out = {"answers": n, "totals": agg, "contradicted": rows[:30]}
        print(json.dumps(out, indent=2, ensure_ascii=False)[:3000])
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
