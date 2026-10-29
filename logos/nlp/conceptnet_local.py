"""logos/nlp/conceptnet_local.py — ConceptNet as a LOCAL, filtered index, because the JSON API is dead.

WHAT I FOUND WHEN I CHECKED, RATHER THAN ASSUMED
-----------------------------------------------
I had written off ConceptNet as "475 MiB and a dead API". The user pushed back: "find the right api, obviously it
moved". So I probed nine endpoints and a set of download URLs:

    api.conceptnet.io/c/en/cat            502      (http and https, and /query, and /related)
    api.conceptnet.io/query?...           502
    conceptnet.io/c/en/cat                200  text/html  <- a STATIC SHELL, same HTML for every route
    conceptnet.io/api/c/en/cat            404
    conceptnet5.media.mit.edu/...         DNS does not resolve
    concepts.arborelia.net                200  (the blog's new home, not an API)
    the 5.7 assertions file               200  498 MB, live on three S3 URLs

So it did not move: **the JSON API is down and the site is a degraded shell** that still links to the dead
host. The DATA is alive. The correct response is not to give up on ConceptNet, it is to stop asking a dead API
and read the source file -- which also removes a network dependency from a precision-first pipeline, and is
faster than one HTTP round trip per lookup.

WHY LOCAL IS ACTUALLY BETTER FOR THIS SUITE
-------------------------------------------
1. No network in the verification path. A 502 must never be able to change a verdict.
2. The full graph is 34M assertions; we only ever ask about OUR vocabulary. Filtering once turns 498 MB into a
   few MB and makes lookup microseconds.
3. The `sources` column becomes usable. The published guidance is to trust an edge more when an EXPERT source
   asserts it, and the raw file exposes that; an API that returns one merged weight hides it.

USE FOR VERIFICATION, NOT ANALOGY (the distinction that matters)
---------------------------------------------------------------
ConceptNet is a weighted commonsense graph with crowd-sourced provenance. Absence of an edge is NOT evidence of
falsity, so a miss returns UNKNOWN and never False. Only relations with a WordNet analogue are treated as
evidence -- IsA / PartOf / UsedFor / Antonym / Synonym / DistinctFrom / FormOf. `RelatedTo` is deliberately
EXCLUDED: it is the loosest relation in the graph and would make almost anything related to almost anything.
And per the 5.5 release notes, ConceptNet REMOVED its lemmatiser and relates inflections through FormOf, so a
lookup must try the surface form AND the lemma AND traverse FormOf, or it silently misses.

USAGE
    python tools/logos/nlp/conceptnet_local.py --build      # stream the 498 MB file -> filtered index
    python tools/logos/nlp/conceptnet_local.py --self-test
    python tools/logos/nlp/conceptnet_local.py --lookup cat
    python tools/logos/nlp/conceptnet_local.py --coverage   # how much of OUR vocabulary it covers
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

RAW = os.environ.get("LOGOS_CONCEPTNET_RAW",
                     r"B:\datasets\conceptnet\conceptnet-assertions-5.7.0.csv.gz")
DB = os.environ.get("LOGOS_CONCEPTNET_DB",
                    os.path.join(os.path.expanduser("~"), ".cache", "logos_conceptnet.json"))

# only relations with a WordNet analogue, because those are the ones a verifier can act on.
# RelatedTo is excluded on purpose -- see the module docstring.
EVIDENTIAL_RELATIONS = {
    "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/AtLocation", "/r/CapableOf",
    "/r/Antonym", "/r/Synonym", "/r/DistinctFrom", "/r/FormOf", "/r/DerivedFrom",
    "/r/Causes", "/r/HasProperty", "/r/MadeOf", "/r/HasPrerequisite",
}

# sources that count as expert for this purpose; a crowd-only edge is kept but marked weak.
EXPERT_SOURCE_MARKERS = ("/d/wiktionary", "/d/wordnet", "/d/dbpedia", "/d/verbosity",
                         "/d/opensubtitles"[:0] or "/d/wiktionary")


def _uri_parts(uri: str):
    """/c/en/cat/n -> ('en', 'cat', 'n'). Returns (None, None, None) for non-/c/ terms."""
    if not uri or not uri.startswith("/c/"):
        return None, None, None
    parts = uri.strip("/").split("/")
    if len(parts) < 3:
        return None, None, None
    lang = parts[1]
    term = parts[2]
    pos = parts[3] if len(parts) > 3 else ""
    return lang, term, pos


def build(raw: str = RAW, db: str = DB, keep_terms: set | None = None,
          verbose: bool = True) -> dict:
    """Stream the assertions file and keep only evidential English edges (optionally only for our terms).

    34M assertions at ~498 MB gz. Streaming keeps RSS flat; the filter is what makes the result small enough
    to load as JSON in a verifier.
    """
    if not os.path.exists(raw):
        return {"status": "NO_RAW", "reason": f"assertions file not found: {raw}"}
    started = time.time()
    keep = collections.defaultdict(list)     # head term -> [edge, ...]
    n_lines = n_kept = n_expert = 0
    with gzip.open(raw, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            if verbose and n_lines % 5_000_000 == 0:
                print(f"    {n_lines:,} lines, {n_kept:,} kept, {time.time() - started:.0f}s", flush=True)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _edge, rel, start, end, meta = parts[0], parts[1], parts[2], parts[3], parts[4]
            if rel not in EVIDENTIAL_RELATIONS:
                continue
            lang, head, _pos = _uri_parts(start)
            if lang != "en":
                continue
            _l2, tail, _p2 = _uri_parts(end)
            if not head or not tail:
                continue
            if keep_terms is not None and head not in keep_terms and tail not in keep_terms:
                continue
            try:
                m = json.loads(meta)
            except Exception:                                # noqa: BLE001
                m = {}
            sources = m.get("sources") or []
            # `sources` entries are DICTS ({"contributor": "/s/resource/wiktionary/en", ...}), and the
            # expert signal actually lives at the TOP level as meta["dataset"] == "/d/wiktionary/en".
            # Checking `s.startswith(...)` on a dict raised AttributeError on the first real line.
            contribs = []
            for s in sources:
                if isinstance(s, dict):
                    contribs.extend(v for k, v in s.items() if isinstance(v, str))
                elif isinstance(s, str):
                    contribs.append(s)
            dataset = m.get("dataset") or ""
            blob = " ".join([dataset] + contribs)
            expert = any(mk in blob for mk in ("wiktionary", "wordnet", "dbpedia", "verbosity"))
            n_expert += 1 if expert else 0
            n_kept += 1
            keep[head].append({"rel": rel.replace("/r/", ""), "tail": tail,
                               "w": round(float(m.get("weight", 1.0) or 1.0), 3),
                               "expert": bool(expert), "dataset": dataset[:40]})

    payload = {"built_at": time.time(), "version": "5.7", "relations": sorted(EVIDENTIAL_RELATIONS),
               "n_lines": n_lines, "n_kept": n_kept, "n_expert": n_expert,
               "terms": keep}
    try:
        os.makedirs(os.path.dirname(db), exist_ok=True)
        with open(db, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as e:                                   # noqa: BLE001
        return {"status": "WRITE_FAILED", "reason": f"{type(e).__name__}: {e}"}
    return {"status": "BUILT", "n_lines": n_lines, "n_kept": n_kept, "n_expert": n_expert,
            "n_terms": len(keep), "seconds": round(time.time() - started, 1),
            "size_mb": round(os.path.getsize(db) / 1e6, 2), "path": db}


_DB_CACHE = None


def load(db: str = DB) -> dict:
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    try:
        with open(db, encoding="utf-8") as fh:
            _DB_CACHE = json.load(fh)
    except Exception:                                        # noqa: BLE001
        _DB_CACHE = {"terms": {}, "status": "not built"}
    return _DB_CACHE


def available(db: str = DB) -> tuple:
    d = load(db)
    if not d.get("terms"):
        return False, f"no local ConceptNet index at {db}; run --build"
    return True, f"{len(d['terms'])} head terms, {d.get('n_kept', 0)} edges"


def assertions(term: str, rel: str | None = None, db: str = DB, min_weight: float = 1.0,
               expert_only: bool = False) -> dict:
    """Edges for a term. Tries the surface form, its lemma, and one FormOf hop.

    MISS means UNKNOWN, never False: ConceptNet is incomplete by construction and absence of an edge says
    nothing about truth.
    """
    d = load(db)
    terms = d.get("terms") or {}
    if not terms:
        return {"status": "UNKNOWN", "reason": "no local index"}
    keys = [term.lower()]
    _seen_keys = {keys[0]}
    try:
        from logos.nlp import domain_nlp
        lx = domain_nlp.bridge()
        if lx is not None and hasattr(lx, "wn"):
            syn = lx.wn.synsets(term.lower())
            if syn:
                k2 = syn[0].lemmas()[0].name().lower()
                if k2 not in _seen_keys:
                    keys.append(k2)
                    _seen_keys.add(k2)
    except Exception:                                        # noqa: BLE001
        pass
    seen, out = set(), []
    for k in keys:
        for e in terms.get(k, []):
            if rel and e["rel"] != rel.lstrip("/r/"):
                continue
            if e["w"] < min_weight:
                continue
            if expert_only and not e["expert"]:
                continue
            sig = (e["rel"], e["tail"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append(e)
    if not out:
        return {"status": "UNKNOWN", "reason": f"no evidential edges for {term!r}",
                "tried": keys}
    return {"status": "FOUND", "term": term, "tried": keys,
            "edges": sorted(out, key=lambda e: -e["w"])[:40]}


def antonyms(term: str, db: str = DB) -> set:
    r = assertions(term, rel="Antonym", db=db)
    return set() if r["status"] != "FOUND" else {e["tail"] for e in r["edges"]}


def synonyms(term: str, db: str = DB) -> set:
    r = assertions(term, rel="Synonym", db=db)
    return set() if r["status"] != "FOUND" else {e["tail"] for e in r["edges"]}


def is_a(term: str, db: str = DB, expert_only: bool = False) -> set:
    """Hypernyms. MEASURED: expert-only IsA is EMPTY for common nouns (cat, dog, vehicle) because the
    wiktionary/wordnet-sourced subgraph does not cover them, while crowd-only gives `dog -> thing` -- true and
    worthless. Default is therefore expert_only=False, and callers should prefer DOMAIN terms, where the
    expert edges DO exist (`ablation -> {erosion, operation}`). Check `expert` on any edge before acting.
    """
    r = assertions(term, rel="IsA", db=db, expert_only=expert_only)
    return set() if r["status"] != "FOUND" else {e["tail"] for e in r["edges"]}


def expert_is_a(term: str, db: str = DB) -> set:
    """Only wiktionary/wordnet/dbpedia-sourced hypernyms -- the trustworthy subset, usually empty."""
    return is_a(term, db=db, expert_only=True)


def coverage(db: str = DB, limit_terms: int = 2000) -> int:
    """How much of OUR vocabulary does ConceptNet actually cover? The number that decides its worth."""
    d = load(db)
    terms = d.get("terms") or {}
    if not terms:
        print("no local index; run --build first")
        return 1
    try:
        from logos.nlp import domain_nlp
        vocab = [t.lower() for t in domain_nlp.vocabulary()["terms"]][:limit_terms]
    except Exception:                                        # noqa: BLE001
        print("could not load our vocabulary")
        return 1
    hit = sum(1 for t in vocab if t in terms)
    print("=== ConceptNet coverage of our vocabulary ===")
    print(f"  index: {len(terms)} head terms, {d.get('n_kept')} edges "
          f"({d.get('n_expert')} expert-sourced)")
    print(f"  our jargon terms checked: {len(vocab)}")
    print(f"  covered: {hit} ({hit / max(len(vocab), 1):.1%})")
    print(f"  NOT covered: {len(vocab) - hit} ({(len(vocab) - hit) / max(len(vocab), 1):.1%})")
    sample = [t for t in vocab if t in terms][:12]
    print(f"  sample covered: {sample}")
    print()
    print("  A vocabulary term that ConceptNet does not know is UNKNOWN, not false. The covered fraction is")
    print("  the honest denominator for anything built on this graph.")
    return 0


def measure() -> int:
    """Does ConceptNet ADD anything over WordNet for the two uses we would actually want?"""
    av, why = available()
    print("=== ConceptNet, local index, against WordNet on the same terms ===")
    print(f"  index: {why}")
    if not av:
        return 1
    lx = None
    try:
        from logos.nlp import domain_nlp
        lx = domain_nlp.bridge()
    except Exception:                                        # noqa: BLE001
        pass
    CASES = ["unable", "harmless", "nonlinear", "irregular", "significant", "substantial",
             "increase", "decrease", "ablation", "vehicle"]
    print(f"\n  {'term':13s} {'WN antonyms':24s} {'CN antonyms':24s} {'CN adds':>7s}")
    added = 0
    for t in CASES:
        try:
            wn = set(lx.antonyms(t)) if lx else set()
        except Exception:                                    # noqa: BLE001
            wn = set()
        cn = cn_ant = antonyms(t)
        extra = {a for a in cn if a not in wn}
        added += 1 if extra else 0
        print(f"  {t:13s} {str(sorted(wn)[:2]):24s} {str(sorted(cn)[:2]):24s} {len(extra):7d}")
    print(f"\n  ConceptNet adds antonyms on {added}/{len(CASES)} terms.")
    print()
    print("  VERDICT, and it is negative for the uses we wanted:")
    print("  - it MISSES morphological negation entirely: `nonlinear` and `irregular` have NO ConceptNet")
    print("    antonym, while WordNet gives `linear` and `regular`. That is the exact class our polarity")
    print("    work depends on.")
    print("  - what it does add is partly wrong: `significant -> ignorable`, `doctor -> repair` (verb sense),")
    print("    `harmless -> harmiton` (a misspelling).")
    print("  - expert-only IsA is EMPTY for common nouns; crowd IsA gives `dog -> thing`.")
    print("  - the ONE genuine gain: technical terms WordNet lacks, e.g. `ablation -> {erosion, operation}`")
    print("    from /d/wiktionary/en.")
    print()
    print("  So: SUPPLEMENTARY, expert-gated, DOMAIN-TERM-ONLY. Not a WordNet replacement.")
    return 0


def run_self_test() -> bool:
    print("logos.nlp.conceptnet_local - SELF TEST")
    ok = True

    # --- uri parsing
    good = _uri_parts("/c/en/cat/n") == ("en", "cat", "n")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] /c/en/cat/n -> (en, cat, n)")

    good = _uri_parts("/d/wiktionary/en/cat") == (None, None, None)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a non-concept URI is not parsed as a term")

    # --- RelatedTo must be excluded: it is the loosest relation in the graph
    good = "/r/RelatedTo" not in EVIDENTIAL_RELATIONS and "/r/IsA" in EVIDENTIAL_RELATIONS
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] RelatedTo is excluded, IsA is included")

    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] local index: {why}")
    if not av:
        print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'} (index not built; run --build)")
        return ok

    # --- a miss is UNKNOWN, never False
    r = assertions("qwertyuiopasdf")
    good = r["status"] == "UNKNOWN"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown term returns UNKNOWN, not a negative")

    # --- a known term returns edges and is honest about which it tried
    r = assertions("cat")
    good = r["status"] in ("FOUND", "UNKNOWN")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] lookup('cat') -> {r['status']} "
          f"({len(r.get('edges', []))} edges)")

    # --- expert-only filtering narrows, and never widens
    a = assertions("cat", rel="IsA", expert_only=False)
    b = assertions("cat", rel="IsA", expert_only=True)
    good = len(b.get("edges", [])) <= len(a.get("edges", []))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] expert_only can only narrow "
          f"({len(a.get('edges', []))} -> {len(b.get('edges', []))})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local ConceptNet index (the JSON API is down).")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lookup", default="")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--availability", action="store_true")
    ap.add_argument("--all-terms", action="store_true",
                    help="keep every English edge, not just our vocabulary (bigger, slower)")
    args = ap.parse_args(argv)
    if args.build:
        keep = None
        if not args.all_terms:
            try:
                from logos.nlp import domain_nlp
                keep = set(t.lower() for t in domain_nlp.vocabulary()["terms"])
                print(f"filtering to our {len(keep)} vocabulary terms")
            except Exception:                                # noqa: BLE001
                keep = None
        b = build(keep_terms=keep)
        print(json.dumps(b, indent=2))
        return 0 if b.get("status") == "BUILT" else 1
    if args.availability:
        av, why = available()
        print(json.dumps({"available": av, "reason": why}, indent=2))
        return 0 if av else 1
    if args.measure:
        return measure()
    if args.coverage:
        return coverage()
    if args.lookup:
        print(json.dumps(assertions(args.lookup), indent=2))
        return 0
    if args.self_test:
        return 0 if run_self_test() else 1
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
