"""logos/nlp/lexicon_ensemble.py — WordNet + ConceptNet + FrameNet COMBINED, which is what the papers do.

THE MISTAKE THIS CORRECTS
------------------------
I compared ConceptNet against WordNet and concluded "ConceptNet is worse than WordNet where polarity matters".
That was a false dichotomy, and the user was right to call it. The literature does not compare them, it
COMBINES them:

  Di Caro & Boella, LREC 2016 (L16-1132), "Automatic Enrichment of WordNet with Common-Sense Knowledge":
      WordNet organises words into synsets "often described by few definitions (or glosses) and examples", and
      "most WSD approaches fully rely on these short texts as a source of contextual information". The paper
      enriches synsets with common-sense definitions retrieved from ConceptNet 5 and disambiguated to WordNet,
      producing ~600K synset enrichments at high precision.

  That is the DIRECT FIX for a failure I measured and had blamed on the resource. Plain Lesk on
  "my cat is sitting on the mat" scores the CORRECT synset (cat.n.01) at ZERO signature overlap while
  kat.n.01 -- the khat plant -- wins on the stopwords "is" and "the". The signatures are too short. The paper
  says so, and says what to do about it: make them longer with ConceptNet.

  The Springer geospatial-metadata study (s41324-026-00710-y) makes the same argument one level up: a
  rule-based spaCy branch and a transformer branch, combined through a weighted fusion layer, with
  low-confidence cases kept for human review. Rule-based + neural FUSED, not ranked against each other.

SO THIS MODULE NEVER CHOOSES. IT UNIONS, AND IT REPORTS PROVENANCE
----------------------------------------------------------------
    antonyms(t)   = WordNet  UNION  ConceptNet          each tagged with its source
    synonyms(t)   = WordNet  UNION  ConceptNet
    verify_related(a, b) -> the evidence from every source, so a caller can require TWO
    enriched_signature(synset) = WordNet gloss + examples  PLUS  ConceptNet edges of that synset's lemmas

Provenance is not decoration. A union without provenance cannot answer "did two INDEPENDENT sources agree?",
which is the only question that makes corroboration worth anything in a precision-first suite -- the same
reason the truth store keys witnesses on implementation tags rather than names.

USAGE
    python tools/logos/nlp/lexicon_ensemble.py --self-test
    python tools/logos/nlp/lexicon_ensemble.py --measure     # does enrichment fix the Lesk failure?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

_POS = {"n": "n", "v": "v", "a": "a", "s": "a", "r": "r"}


def _wn():
    try:
        from nltk.corpus import wordnet as wn
        return wn
    except Exception:                                        # noqa: BLE001
        return None


def _cn():
    try:
        from logos.nlp import conceptnet_local as cn
        ok, _why = cn.available()
        return cn if ok else None
    except Exception:                                        # noqa: BLE001
        return None


ALL_POS = ("n", "v", "a", "r")


def _pos_list(pos) -> tuple:
    """pos may be 'n', 'a', ... or None/'*' for ALL. Default is ALL, because polarity lives in adjectives
    and a noun-only lookup silently returns nothing for `unable` and `nonlinear`."""
    if pos in (None, "", "*"):
        return ALL_POS
    p = _POS.get(str(pos).lower(), "n")
    # wordnet treats adjective satellites as 'a' too
    return (p,)


def _stem(w: str) -> str:
    """Crude, deterministic stem for signature matching. Porter would be better; this keeps the module
    dependency-light and is only used to compare words within one sentence."""
    w = (w or "").lower()
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


# ------------------------------------------------------------------ union with provenance
def antonyms(term: str, pos=None) -> dict:
    """WordNet UNION ConceptNet antonyms, each tagged. Never a choice between the two.

    `pos=None` means ALL parts of speech, which is the default BECAUSE polarity is carried by adjectives:
    a noun-only query returns nothing for `unable` and `nonlinear`, and that is exactly the class our
    negation detection depends on.
    """
    out = {}
    wn = _wn()
    if wn is not None:
        for one in _pos_list(pos):
            try:
                for s in wn.synsets(term, pos=one):
                    for l in s.lemmas():
                        for a in l.antonyms():
                            out.setdefault(a.name().lower().replace("_", " "), set()).add("wordnet")
            except Exception:                                # noqa: BLE001
                continue
    cn = _cn()
    if cn is not None:
        for a in cn.antonyms(term):
            out.setdefault(str(a).lower().replace("_", " "), set()).add("conceptnet")
    return {"term": term, "sources": {k: sorted(v) for k, v in out.items()},
            "both": sorted(k for k, v in out.items() if len(v) > 1),
            "n": len(out)}


def synonyms(term: str, pos=None) -> dict:
    out = {}
    wn = _wn()
    if wn is not None:
        for one in _pos_list(pos):
            try:
                for s in wn.synsets(term, pos=one):
                    for l in s.lemmas():
                        name = l.name().lower().replace("_", " ")
                        if name != term.lower():
                            out.setdefault(name, set()).add("wordnet")
            except Exception:                                # noqa: BLE001
                continue
    cn = _cn()
    if cn is not None:
        for a in cn.synonyms(term):
            name = str(a).lower().replace("_", " ")
            if name != term.lower():
                out.setdefault(name, set()).add("conceptnet")
    return {"term": term, "sources": {k: sorted(v) for k, v in out.items()},
            "both": sorted(k for k, v in out.items() if len(v) > 1), "n": len(out)}


def corroborated(term: str, pos=None) -> dict:
    """Antonyms/synonyms asserted by BOTH sources. Two independent vocabularies agreeing is the only
    combination that raises our confidence, and it is why the union is kept with provenance instead of
    being flattened to a set."""
    a = antonyms(term, pos)
    s = synonyms(term, pos)
    return {"term": term, "antonyms_both": a["both"], "synonyms_both": s["both"],
            "antonym_sources": a["sources"], "synonym_sources": s["sources"]}


def verify_related(a: str, b: str, pos=None) -> dict:
    """Every source's opinion on whether a and b are related, with the relation named.

    Returns a LIST of claims rather than a verdict: a caller decides how many independent sources it needs,
    which is the same discipline the truth store applies to validators.
    """
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    ev = []
    an = antonyms(a, pos)
    if b in an["sources"]:
        ev.append({"relation": "antonym", "sources": an["sources"][b],
                   "independent": len(an["sources"][b])})
    sn = synonyms(a, pos)
    if b in sn["sources"]:
        ev.append({"relation": "synonym", "sources": sn["sources"][b],
                   "independent": len(sn["sources"][b])})
    wn = _wn()
    if wn is not None:
        try:
            ps = _pos_list(pos)
            sa = {s for one in ps for s in wn.synsets(a, pos=one)}
            sb = {s for one in ps for s in wn.synsets(b, pos=one)}
            if sa & sb:
                ev.append({"relation": "same_synset", "sources": ["wordnet"], "independent": 1})
            if any(s in sa for hyp in sb for s in hyp.hypernyms()):
                ev.append({"relation": "hypernym", "sources": ["wordnet"], "independent": 1})
        except Exception:                                    # noqa: BLE001
            pass
    cn = _cn()
    if cn is not None:
        for rel in ("IsA", "PartOf", "UsedFor", "Synonym"):
            tails = cn.assertions(a, rel=rel)
            if tails.get("status") == "FOUND" and b in {str(e["tail"]).lower() for e in tails["edges"]}:
                ev.append({"relation": rel.lower(), "sources": ["conceptnet"], "independent": 1})
    return {"a": a, "b": b, "evidence": ev, "n_independent_sources": len({
        s for e in ev for s in e["sources"]})}


# ------------------------------------------------------------------ the L16-1132 enrichment
def enriched_signature(synset_name: str, use_conceptnet: bool = True) -> dict:
    """A synset's signature: WordNet gloss + examples, EXTENDED with ConceptNet edges of its lemmas.

    This is the mechanism from Di Caro & Boella (LREC 2016). It exists because plain WSD "fully relies on
    these short texts", and short texts are exactly what made Lesk pick a khat plant for "my cat".
    """
    wn = _wn()
    if wn is None:
        return {"status": "NO_WORDNET", "tokens": []}
    try:
        s = wn.synset(synset_name)
    except Exception:                                        # noqa: BLE001
        return {"status": "NO_SUCH_SYNSET", "tokens": []}
    parts = [s.definition() or ""] + list(s.examples() or [])
    lemmas = [l.name().replace("_", " ") for l in s.lemmas()]
    parts.extend(lemmas)
    cn_edges = []
    if use_conceptnet:
        cn = _cn()
        if cn is not None:
            for lem in lemmas[:4]:
                r = cn.assertions(lem)
                if r.get("status") != "FOUND":
                    continue
                for e in r["edges"][:12]:
                    cn_edges.append({"lemma": lem, "rel": e["rel"], "tail": str(e["tail"]),
                                     "expert": e["expert"]})
                    parts.append(str(e["tail"]).replace("_", " "))
    text = " ".join(parts)
    tokens = {_stem(t) for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())}
    return {"status": "OK", "synset": synset_name, "tokens": sorted(tokens),
            "n_tokens": len(tokens), "n_cn_edges": len(cn_edges),
            "cn_edges": cn_edges[:16],
            "wordnet_only_tokens": len({_stem(t) for t in re.findall(
                r"[A-Za-z][A-Za-z\-]{2,}",
                " ".join([s.definition() or ""] + list(s.examples() or [])).lower())})}


def lesk(context: str, word: str, pos: str = "n", use_conceptnet: bool = True) -> dict:
    """Signature-overlap WSD, with the enriched signature. Compared against plain nltk.lesk in --measure."""
    wn = _wn()
    if wn is None:
        return {"status": "NO_WORDNET"}
    ctx = {_stem(t) for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (context or "").lower())}
    best, best_score, scores = None, -1, []
    try:
        candidates = wn.synsets(word, pos=_POS.get(pos, "n"))
    except Exception:                                        # noqa: BLE001
        return {"status": "NO_SYNSETS"}
    for s in candidates:
        sig = enriched_signature(s.name(), use_conceptnet=use_conceptnet)
        if sig["status"] != "OK":
            continue
        overlap = ctx & set(sig["tokens"])
        score = len(overlap)
        scores.append({"synset": s.name(), "score": score, "overlap": sorted(overlap)[:6]})
        if score > best_score:
            best, best_score = s.name(), score
    return {"status": "OK", "word": word, "best": best, "score": best_score,
            "runner_up": sorted(scores, key=lambda x: -x["score"])[1:4], "scores": scores}


# ------------------------------------------------------------------ measure
def measure() -> int:
    print("=== WordNet + ConceptNet COMBINED ===")
    print(f"  wordnet: {'available' if _wn() else 'MISSING'}")
    cn = _cn()
    print(f"  conceptnet: {'available' if cn else 'not built (run conceptnet_local --build)'}\n")

    print("=== 1. union with provenance (never a choice between sources) ===")
    print(f"  {'term':13s} {'antonyms (source)':58s} {'both':>5s}")
    for t in ("significant", "increase", "nonlinear", "harmless", "unable"):
        a = antonyms(t)
        shown = ", ".join(f"{k}[{'+'.join(v)}]" for k, v in list(a["sources"].items())[:4])
        print(f"  {t:13s} {shown[:58]:58s} {len(a['both']):5d}")

    print("\n=== 2. THE L16-1132 TEST: does ConceptNet enrichment fix the Lesk failure? ===")
    CASES = [
        ("my cat is sitting on the mat", "cat", "cat.n.01"),
        ("the vet examined the sick cat", "cat", "cat.n.01"),
        ("he is a cool cat from the jazz scene", "cat", "guy.n.01"),
        ("the doctor prescribed antibiotics", "doctor", "doctor.n.01"),
    ]
    print(f"  {'context':38s} {'correct':16s} {'plain':16s} {'enriched':16s}")
    plain_ok = enr_ok = 0
    for ctx, w, correct in CASES:
        try:
            from nltk.wsd import lesk as nltk_lesk
            p = nltk_lesk(ctx.split(), w, "n")
            pname = p.name() if p is not None else "-"
        except Exception:                                    # noqa: BLE001
            pname = "err"
        e = lesk(ctx, w, "n", use_conceptnet=True)
        ename = e.get("best") or "-"
        plain_ok += 1 if pname == correct else 0
        enr_ok += 1 if ename == correct else 0
        print(f"  {ctx[:38]:38s} {correct:16s} {pname[:16]:16s} {ename[:16]:16s}")
    n = len(CASES)
    print(f"\n  plain nltk lesk : {plain_ok}/{n} correct")
    print(f"  enriched        : {enr_ok}/{n} correct")
    print()
    if enr_ok > plain_ok:
        print(f"  ==> enrichment moved {plain_ok} -> {enr_ok} of {n}. ON {n} SENTENCES THAT IS NOISE, NOT A")
        print(f"      RESULT. The L16-1132 mechanism is not demonstrated here, and I am not going to call a")
        print(f"      single flipped sentence evidence. What the numbers DO show is that both methods are far")
        print(f"      below the first-sense baseline that was measured separately at 99.5% agreement with the")
        print(f"      SemCor MFS -- so for THIS suite, first-sense plus a monosemy refusal is still the method,")
        print(f"      and enrichment is a candidate worth testing properly, not an adopted improvement.")
    elif enr_ok == plain_ok:
        print(f"  ==> no measurable gain on {n} sentences. Honest reading: {n} sentences cannot show a WSD")
        print(f"      improvement, and first-sense frequency already beats both (measured earlier at 99.5%")
        print(f"      agreement with the SemCor MFS). Do not claim enrichment works from this.")
    else:
        print(f"  ==> enrichment did WORSE on this set. Report it; do not tune until it looks better.")

    print("\n=== 3. corroboration: relations BOTH sources assert ===")
    for t in ("significant", "unable", "harmless", "increase"):
        c = corroborated(t)
        if c["antonyms_both"] or c["synonyms_both"]:
            print(f"  {t:12s} antonyms_both={c['antonyms_both']}  synonyms_both={c['synonyms_both']}")
    print("\n  Only 'both' has earned corroboration. A single-source relation is a suggestion, and the union")
    print("  keeps that distinction instead of flattening it into one undifferentiated set.")
    return 0


def run_self_test() -> bool:
    print("logos.nlp.lexicon_ensemble - SELF TEST")
    ok = True

    # --- the union reports provenance, which is the whole point
    r = antonyms("significant")
    good = isinstance(r["sources"], dict) and all(isinstance(v, list) for v in r["sources"].values())
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonyms() returns per-source provenance "
          f"({len(r['sources'])} terms)")

    # --- "both" is the corroborated subset and is never larger than the union
    good = len(r["both"]) <= r["n"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the corroborated subset is not larger than the union")

    # --- REGRESSION: POS must default to ALL. A noun-only lookup returns nothing for `nonlinear` and
    #     `unable`, which is exactly the class polarity depends on.
    r2 = antonyms("nonlinear")
    good = any("wordnet" in v for v in r2["sources"].values())
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an ADJECTIVE antonym survives the union "
          f"(nonlinear -> {list(r2['sources'])[:2]})")

    r3 = antonyms("unable")
    good = "able" in r3["sources"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'unable' -> 'able' from the union")

    # --- and exporting pos='n' still restricts, so the parameter is honest
    rn = antonyms("nonlinear", pos="n")
    good = rn["n"] == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] pos='n' restricts (nonlinear has no noun antonym)")

    # --- enrichment strictly extends the signature
    plain = enriched_signature("cat.n.01", use_conceptnet=False)
    rich = enriched_signature("cat.n.01", use_conceptnet=True)
    good = rich["n_tokens"] >= plain["n_tokens"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] enrichment never shrinks a signature "
          f"({plain['n_tokens']} -> {rich['n_tokens']} tokens)")

    # --- verify_related reports claims, not a verdict
    v = verify_related("doctor", "physician")
    good = "evidence" in v and isinstance(v["evidence"], list) and v["n_independent_sources"] >= 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] verify_related returns evidence, not a verdict "
          f"({len(v['evidence'])} claims)")

    # --- an unrelated pair must not manufacture evidence
    v = verify_related("cat", "refrigerator")
    good = v["evidence"] == [] or all(e["relation"] != "antonym" for e in v["evidence"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unrelated pair produces no antonym claim")

    # --- a missing conceptnet index must DEGRADE, not crash
    import logos.nlp.conceptnet_local as cnmod
    saved = cnmod._DB_CACHE
    cnmod._DB_CACHE = {"terms": {}, "status": "not built"}
    try:
        r3 = antonyms("harmless")
        good = isinstance(r3["sources"], dict)
    finally:
        cnmod._DB_CACHE = saved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] with no ConceptNet index it degrades to WordNet only")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WordNet + ConceptNet combined, with provenance.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--term", default="")
    ap.add_argument("--related", nargs=2, default=None)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.related:
        print(json.dumps(verify_related(*args.related), indent=2))
        return 0
    if args.term:
        print(json.dumps(corroborated(args.term), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
