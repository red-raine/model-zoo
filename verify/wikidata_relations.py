"""wikidata_relations.py — a general entity-relation checker, not two hardcoded patterns.

WHY THIS FILE EXISTS
--------------------
`fact_verify.wikidata_claim` only understood two claim shapes:

    X is (located) in Y      -> P17/P131/P30/P361
    X is the capital of Y    -> P36

Measured on AVeriTeC train it settled **2 of 3,067 claims (0.07%)**. That is not a
deterministic channel, it is a rounding error — and I had claimed "Boolean and Extractive
answers are what sympy and Wikidata can settle exactly", which was wrong.

The 2026 literature says where the points are: **gold vs retrieved evidence is worth
+14 to +22 points, and "retrieval remains the primary bottleneck"** (arXiv:2608.25934).
ClaimCheck reaches 76.4% on AVeriTeC with a Qwen3-4B *pipeline* (arXiv:2510.01226) —
the pipeline, not the model. So a wider deterministic channel is the highest-value work
available, and it costs $0 and generates no tokens.

WHAT IT DOES
------------
1. spaCy pulls the entities out of a claim (reusing retrieval.spacy_entities).
2. A relation lexicon maps the phrase BETWEEN two entities to a Wikidata property.
3. SPARQL asks: does SUBJECT --Pxxx--> OBJECT (or the inverse) hold?
4. Returns SUPPORTED / CONTRADICTED / UNKNOWN, with the actual property values, so the
   verdict is auditable rather than asserted.

HONEST LIMITS
-------------
* Wikidata is incomplete. A CONTRADICTED verdict means "Wikidata says otherwise", which is
  strong but not absolute — Wikidata can be wrong or stale.
* Only relations in RELATION_MAP are covered. An unknown relation returns UNKNOWN, never a
  guess. Refusing to guess is the whole point.
* Temporal claims ("was CEO in 2019") are not handled; Wikidata qualifiers would be needed.

USAGE
-----
    python tools/verify/wikidata_relations.py --self-test
    python tools/verify/wikidata_relations.py --claim "Donald Trump was born in New York City."
    python tools/verify/wikidata_relations.py --coverage B:/datasets/forge/verifier-av/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

WD_API = "https://www.wikidata.org/w/api.php"
WD_SPARQL = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "forge-logos/1.0 (research; local model zoo)"}

# --------------------------------------------------------------------------
# relation lexicon: phrase pattern -> (property, direction)
#   direction "fwd"  : SUBJECT --P--> OBJECT
#   direction "inv"  : OBJECT --P--> SUBJECT  (e.g. "founded by")
# --------------------------------------------------------------------------
RELATION_MAP: list[tuple[str, str, str]] = [
    # direction semantics: "fwd" = SUBJECT --P--> OBJECT
    #                      "inv" = OBJECT  --P--> SUBJECT
    # (The first version had six of these backwards. "X is the capital of Y" means Y has
    #  capital X, so P36 lives on the OBJECT. The self-test caught it.)
    # --- taxonomy / membership (property lives on the subject) ---
    # NOTE: these must require actual content. An earlier version was
    #   r"\bis (?:a|an|the)?\s*(?:kind of|type of)?\b"
    # where every group was optional, so it matched a bare " is " and beat every more
    # specific relation in the first-match race -- "Paris is the capital of France"
    # extracted as P31 instead of P36.
    (r"\bis (?:a|an|the)\s+(?:kind|type|form|sort) of\b", "P31", "fwd"),
    (r"\bis (?:a|an) (?:instance|example) of\b", "P31", "fwd"),
    (r"\bis an? \w", "P31", "fwd"),
    (r"\bis part of\b", "P361", "fwd"),
    (r"\bis a member of\b", "P463", "fwd"),
    (r"\bis a subsidiary of\b", "P749", "fwd"),
    (r"\bis affiliated with\b", "P1142", "fwd"),
    # --- geography ---
    (r"\bis (?:located )?in (?:the )?(?:country|nation) of\b", "P17", "fwd"),
    (r"\bis (?:located )?in\b", "P17", "fwd"),
    (r"\bshares a border with\b|\bborders\b", "P47", "fwd"),
    (r"\bis (?:situated|located) (?:at|on)\b", "P131", "fwd"),
    (r"\bis headquartered in\b", "P159", "fwd"),
    (r"\bis (?:the )?capital of\b", "P36", "inv"),          # Y --P36--> X
    (r"\bis the official language of\b", "P37", "inv"),      # Y --P37--> X
    # --- people (property lives on the person) ---
    (r"\bwas born in\b|\bis from\b", "P19", "fwd"),
    (r"\bdied in\b", "P20", "fwd"),
    (r"\bis married to\b|\bwas married to\b", "P26", "fwd"),
    (r"\bis (?:the )?(?:father|mother|parent) of\b", "P40", "fwd"),   # X's child is Y
    (r"\bis a child of\b", "P40", "fwd"),
    (r"\bwas educated at\b|\bstudied at\b", "P69", "fwd"),
    (r"\bworks? (?:for|at)\b", "P108", "fwd"),
    # --- organisations (property lives on the org, so inverse) ---
    (r"\bis the (?:founder|creator) of\b", "P112", "inv"),
    (r"\bfounded\b", "P112", "inv"),
    (r"\bis (?:the )?(?:president|head of state) of\b", "P35", "inv"),
    (r"\bis (?:the )?(?:CEO|chief executive) of\b", "P169", "inv"),
    (r"\bis owned by\b", "P127", "fwd"),                      # X --P127--> Y
    (r"\bowns\b", "P127", "inv"),                             # Y --P127--> X
    # --- works / media (property lives on the work) ---
    (r"\bwas (?:written|authored) by\b", "P50", "fwd"),
    (r"\bis the author of\b", "P50", "fwd"),
    (r"\bwas directed by\b", "P57", "fwd"),
    (r"\bwas produced by\b", "P162", "fwd"),
    (r"\bstars?\b|\bis (?:played|portrayed) by\b", "P161", "fwd"),
    (r"\bwas published (?:in|by)\b", "P123", "fwd"),
    # --- misc ---
    (r"\bwas named after\b", "P138", "fwd"),
    (r"\bis written in\b", "P277", "fwd"),
]

_COMPILED = [(re.compile(p, re.I), prop, d) for p, prop, d in RELATION_MAP]

# properties where the object is a quantity, not an entity -> handled separately
NUMERIC_PROPS = {"P1082": "population", "P2046": "area", "P2131": "nominal GDP"}


def _get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


_QID_CACHE: dict[str, str | None] = {}


def qid(name: str) -> str | None:
    key = name.strip().lower()
    if key in _QID_CACHE:
        return _QID_CACHE[key]
    out = None
    try:
        d = _get_json(f"{WD_API}?action=wbsearchentities&format=json&language=en"
                      f"&limit=1&search={urllib.parse.quote(name.strip())}", 20)
        s = d.get("search") or []
        out = s[0]["id"] if s else None
    except Exception:
        out = None
    _QID_CACHE[key] = out
    return out


def entity_labels(qid_: str, prop: str, limit: int = 40) -> set[str]:
    q = (f"SELECT ?label WHERE {{ wd:{qid_} wdt:{prop} ?item . "
         f"?item rdfs:label ?label . FILTER(lang(?label) = \"en\") }} LIMIT {limit}")
    try:
        d = _get_json(f"{WD_SPARQL}?format=json&query={urllib.parse.quote(q)}")
    except Exception:
        return set()
    return {b["label"]["value"].lower()
            for b in d["results"]["bindings"] if b.get("label", {}).get("value")}


def entity_ids(qid_: str, prop: str, limit: int = 40) -> set[str]:
    q = (f"SELECT ?item WHERE {{ wd:{qid_} wdt:{prop} ?item . }} LIMIT {limit}")
    try:
        d = _get_json(f"{WD_SPARQL}?format=json&query={urllib.parse.quote(q)}")
    except Exception:
        return set()
    return {b["item"]["value"].rsplit("/", 1)[-1]
            for b in d["results"]["bindings"] if b.get("item", {}).get("value")}


def extract_triples(claim: str) -> list[dict]:
    """Find (subject, relation, object) using spaCy entity spans, not capitalisation.

    Measured failure: the first version required a CAPITALISED run immediately after the
    relation phrase. On real AVeriTeC prose ("refused to appear in an apple commercial")
    that found nothing — coverage 0.0 on 50 claims. The bottleneck was entity extraction,
    not the relation lexicon.

    Now: spaCy finds the entities with character offsets, then for each adjacent pair we
    look at the text BETWEEN them and match that against the relation lexicon. The
    relation must be between the two entities, which is the actual syntactic claim.
    """
    triples = []
    ents = _spans(claim)
    if len(ents) < 2:
        return []
    for (s_txt, s_end), (o_txt, o_start) in zip(ents, ents[1:]):
        if o_start < s_end:
            continue
        between = claim[s_end:o_start]
        if len(between) > 120:          # too far apart to be one claim
            continue
        for rx, prop, direction in _COMPILED:
            if rx.search(between):
                triples.append({"subject": s_txt, "object": o_txt,
                                "property": prop, "direction": direction,
                                "phrase": between.strip()[:60]})
                break
    return triples


_SPACY = None


def _spans(text: str) -> list[tuple[str, int]]:
    """[(entity_text, end_offset)] for named entities, plus capitalised fallbacks."""
    global _SPACY
    out = []
    try:
        if _SPACY is None:
            import spacy
            _SPACY = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
        doc = _SPACY(text[:1200])
        keep = ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT",
                "PRODUCT", "FAC", "NORP", "LAW", "DATE")
        for e in doc.ents:
            if e.label_ in keep and len(e.text.strip()) > 1:
                out.append((e.text.strip(), e.end_char))
    except Exception:
        pass
    if not out:
        # fallback: capitalised runs with offsets
        for m in re.finditer(r"[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*", text):
            if len(m.group(0)) > 2:
                out.append((m.group(0).strip(), m.end()))
    out.sort(key=lambda x: x[1])
    return out


def check_triple(t: dict) -> dict:
    """Ask Wikidata whether the triple holds."""
    s_q = qid(t["subject"])
    o_q = qid(t["object"])
    if not s_q or not o_q:
        return {**t, "verdict": "UNKNOWN", "why": "entity not found in Wikidata"}

    if t["direction"] == "fwd":
        holder, target = s_q, o_q
    else:
        holder, target = o_q, s_q

    labels = entity_labels(holder, t["property"])
    ids = entity_ids(holder, t["property"])
    if not labels and not ids:
        return {**t, "verdict": "UNKNOWN",
                "why": f"Wikidata has no {t['property']} for {t['subject' if t['direction']=='fwd' else 'object']}"}

    hit = target in ids
    if not hit:
        ol = t["object"].lower() if t["direction"] == "fwd" else t["subject"].lower()
        hit = any(ol in lb or lb in ol for lb in labels)
    return {**t, "verdict": "SUPPORTED" if hit else "CONTRADICTED",
            "why": f"{t['property']} -> {sorted(labels)[:5]}"}


def verify_claim(claim: str) -> dict:
    triples = extract_triples(claim)
    checked = [check_triple(t) for t in triples]
    settled = [c for c in checked if c["verdict"] in ("SUPPORTED", "CONTRADICTED")]
    if any(c["verdict"] == "CONTRADICTED" for c in settled):
        v = "CONTRADICTED"
    elif settled:
        v = "SUPPORTED"
    else:
        v = "UNKNOWN"
    return {"claim": claim, "triples": checked, "verdict": v,
            "settled": len(settled)}


# --------------------------------------------------------------------------
def run_self_test() -> bool:
    print("wikidata_relations.py - SELF TEST")
    ok = True
    # triple extraction must find the relation and both endpoints
    cases = [
        ("Donald Trump was born in New York City.", "P19", "Donald Trump", "New York City"),
        ("Paris is the capital of France.", "P36", "Paris", "France"),
        ("Delaware is located in China.", "P17", "Delaware", "China"),
    ]
    for claim, prop, subj, obj in cases:
        ts = extract_triples(claim)
        good = any(t["property"] == prop for t in ts)
        ok = ok and good
        got = [(t["property"], t["subject"], t["object"]) for t in ts]
        print(f"  [{'PASS' if good else 'FAIL'}] extract {claim[:40]!r} -> {got[:2]}")

    # direction semantics: the property must live on the right endpoint
    dir_cases = [
        ("Paris is the capital of France.", "P36", "inv"),
        ("Delaware is located in China.", "P17", "fwd"),
        ("Donald Trump was born in New York City.", "P19", "fwd"),
        ("Steve Jobs founded Apple Inc.", "P112", "inv"),
    ]
    for claim, prop, want_dir in dir_cases:
        ts = [t for t in extract_triples(claim) if t["property"] == prop]
        good = bool(ts) and ts[0]["direction"] == want_dir
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] direction {claim[:34]!r} "
              f"{prop} -> {ts[0]['direction'] if ts else None} (want {want_dir})")

    # a claim with no known relation must yield nothing, not a guess
    ts = extract_triples("The quick brown fox jumps over the lazy dog.")
    good = ts == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unknown relation -> no triple ({len(ts)})")

    # live checks
    for claim, want in (("Paris is the capital of France.", "SUPPORTED"),
                        ("Delaware is located in China.", "CONTRADICTED")):
        try:
            r = verify_claim(claim)
            good = r["verdict"] == want
            ok = ok and good
            print(f"  [{'PASS' if good else 'FAIL'}] {claim[:34]!r} -> {r['verdict']} "
                  f"(settled {r['settled']})")
        except Exception as e:
            print(f"  [SKIP] live {claim[:30]!r} ({type(e).__name__})")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def coverage(path: str, limit: int = 0) -> dict:
    """How many claims can the deterministic channel actually settle?"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    import collections
    out = {"n": len(rows), "settled": 0, "verdicts": collections.Counter(),
           "agree_with_gold": 0, "props": collections.Counter()}
    for r in rows:
        claim = r.get("claim") or ""
        try:
            res = verify_claim(claim)
        except Exception:
            continue
        out["verdicts"][res["verdict"]] += 1
        for t in res["triples"]:
            out["props"][t["property"]] += 1
        if res["verdict"] in ("SUPPORTED", "CONTRADICTED"):
            out["settled"] += 1
            gold = (r.get("completion") or "")
            mapped = {"SUPPORTED": "SUPPORTED", "CONTRADICTED": "REFUTED"}.get(
                res["verdict"])
            if gold == mapped:
                out["agree_with_gold"] += 1
    out["coverage"] = round(out["settled"] / max(out["n"], 1), 4)
    out["agreement"] = round(out["agree_with_gold"] / max(out["settled"], 1), 4)
    out["verdicts"] = dict(out["verdicts"])
    out["props"] = dict(out["props"].most_common(10))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="General entity-relation fact checker.")
    ap.add_argument("--claim")
    ap.add_argument("--coverage")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if args.claim:
        print(json.dumps(verify_claim(args.claim), indent=2, ensure_ascii=False)[:2500])
        return 0
    if args.coverage:
        print(json.dumps(coverage(args.coverage, args.limit), indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
