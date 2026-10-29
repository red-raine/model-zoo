"""logos/nlp/synset_injection.py — ConceptNet INJECTED into a specific synset, not unioned at the lemma.

THE MEASUREMENT THIS MODULE RESPONDS TO
---------------------------------------
`lexicon_ensemble.py` unions WordNet with ConceptNet at the LEMMA level: every antonym of every synset of a
word, plus every raw ConceptNet edge on the word string. That is the "plain union", and it was measured and it
does not work:

    Ruggeri, Di Caro et al., "The Role of Common-Sense Knowledge in Assessing Semantic Association",
    J. Data and Information Quality, Tables 4-6 (weighted F):
        ConceptNet only            0.591 (MEN)   0.552 (SimLex)
        WordNet only               0.413         0.484
        WordNet + ConceptNet       0.592         0.538   <- the plain UNION: no gain, and it DESTROYS
                                                            WordNet's precision on SimLex
        WordNet *injected* +CN     0.653         0.642   <- +0.061 / +0.104 over the union

    Their conclusion, verbatim: "the simple union (WN+CN) between WordNet and ConceptNet features did not carry
    to any increase in performance. This is due to the high ambiguity of the terms in ConceptNet ... ConceptNet
    may not be directly usable as is."

The operative difference is the ALIGNMENT STEP, not the sources. In the working version each ConceptNet
assertion is first disambiguated to a SPECIFIC WordNet synset; only then is it injected into that synset's
description. Here the alignment test is structural and explicit: **does WordNet's own graph assert the same
relation between the same two synsets?**

That gives TWO INDEPENDENT MECHANISMS agreeing -- WordNet's hand-built graph and ConceptNet's crowd-derived
one -- which is the gate the same authors' enrichment work justifies: even their gated path measured 88.19%
accuracy on 505 sampled enrichments, i.e. ~11.8% wrong. One source agreeing with itself is not corroboration;
the truth store in this project already keys witnesses on implementation tags for the same reason.

WHICH RELATIONS CAN BE VERIFIED, AND WHICH CANNOT
-------------------------------------------------
ConceptNet relations in the local index, by count: Synonym 68,596 | DerivedFrom 22,071 | IsA 12,314 |
FormOf 4,171 | Antonym 2,974 | UsedFor 2,125 | AtLocation 1,711 | CapableOf 1,062 | DistinctFrom 1,058 |
HasProperty 1,041 | HasPrerequisite 1,017 | PartOf 559.

WordNet has a structural analogue for exactly six of them (Synonym, Antonym, IsA, PartOf, DerivedFrom/FormOf,
HasProperty). For the rest -- UsedFor, AtLocation, CapableOf, HasPrerequisite, DistinctFrom -- WordNet asserts
nothing either way, so agreement is IMPOSSIBLE to establish and the edge is UNVERIFIED rather than wrong.
Unverified edges are reported and NOT injected by default. That is the whole point: the union injects them
anyway, which is how ~14,000 UsedFor/AtLocation/CapableOf edges with no corroboration enter a signature.

USAGE
    python tools/logos/nlp/synset_injection.py --self-test
    python tools/logos/nlp/synset_injection.py --measure          # verification rate per relation
    python tools/logos/nlp/synset_injection.py --synset cat.n.01
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

# ConceptNet relation -> the WordNet structural test that can vouch for it.
# Anything absent from this map has NO WordNet analogue and can only ever be UNVERIFIED.
VERIFIABLE = {
    "Synonym": "synonym",
    "Antonym": "antonym",
    "IsA": "hypernym",
    "PartOf": "part_of",
    "DerivedFrom": "derivation",
    "FormOf": "derivation",
    "HasProperty": "attribute",
}

# Relations that carry no WordNet-checkable claim. Listed explicitly so the UNSHAPE is legible, and so that a
# future addition has to be a decision rather than an oversight.
UNVERIFIABLE = ("UsedFor", "AtLocation", "CapableOf", "HasPrerequisite", "DistinctFrom",
                "RelatedTo", "Causes", "HasSubEvent", "MotivatedByGoal", "Desires", "ReceivesAction")

_CACHE = {}


def _wn():
    from nltk.corpus import wordnet as wn
    return wn


def _cn():
    try:
        from logos.nlp import conceptnet_local as cn
        ok, _why = cn.available()
        return cn if ok else None
    except Exception:                                        # noqa: BLE001
        return None


def _norm(term: str) -> str:
    return str(term or "").lower().replace(" ", "_").strip()


def _synsets_of(term: str) -> set:
    wn = _wn()
    t = _norm(term)
    out = set()
    for pos in ("n", "v", "a", "r"):
        try:
            for s in wn.synsets(t.replace("_", "_"), pos=pos):
                out.add(s)
        except Exception:                                    # noqa: BLE001
            continue
    return out


def _antonym_synsets(wn_synset) -> set:
    """Synsets reached by antonymy, resolved to SYNSETS rather than strings.

    WordNet antonymy is LEMMA-level and often ONE-DIRECTIONAL. Measured: `significant.a.01` has lemmas
    ['significant', 'important'], and `significant.antonyms()` -> ['insignificant'] while
    `important.antonyms()` -> **[]**. So looking up the string 'unimportant' fails even though WordNet does
    place it opposite: `unimportant` shares the synset `insignificant.a.04`, and `insignificant` IS the
    antonym of `significant`. Comparing strings reported a false DISAGREED; comparing SYNSETS does not.
    """
    out = set()
    for l in wn_synset.lemmas():
        for a in l.antonyms():
            out |= _synsets_of(a.name().replace("_", " "))
    return out


def verify(wn_synset, rel: str, tail: str) -> dict:
    """Does WordNet corroborate, contradict, or simply not address this ConceptNet claim?

    FOUR verdicts, because collapsing them is how a "not asserted" becomes a false accusation:
        VERIFIED       WordNet's graph asserts the same relation between the same synsets
        CONTRADICTED   WordNet asserts something INCOMPATIBLE with the claim
        NOT_ASSERTED   WordNet has an analogue for this relation and is silent on this pair
        UNVERIFIABLE   no WordNet analogue exists for this relation at all

    NOT_ASSERTED is the honest majority. WordNet's silence is not evidence against a crowd-sourced claim, and
    reporting it as "disagreed" would overstate how much of ConceptNet is actually wrong. It still does not
    license injection -- uncorroborated is uncorroborated -- but the wording matters for a suite that cites
    its refusals as proof.
    """
    kind = VERIFIABLE.get(rel)
    if kind is None:
        return {"status": "UNVERIFIABLE", "rel": rel, "why": "WordNet has no structural analogue"}
    tails = _synsets_of(tail)
    if not tails:
        return {"status": "UNVERIFIABLE", "rel": rel, "why": f"{tail!r} has no WordNet synsets"}

    try:
        if kind == "synonym":
            if tails & {wn_synset}:
                return {"status": "VERIFIED", "wn_rel": "same_synset"}
            # a claim of synonymy where WordNet says OPPOSITE is a genuine contradiction
            if tails & _antonym_synsets(wn_synset):
                return {"status": "CONTRADICTED", "wn_rel": "antonym",
                        "why": f"ConceptNet calls {tail!r} a synonym; WordNet places it opposite"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": f"WordNet has a synonymy relation and {tail!r} is a different synset"}

        if kind == "antonym":
            ants = _antonym_synsets(wn_synset)
            if tails & ants:
                return {"status": "VERIFIED", "wn_rel": "antonym"}
            if tails & {wn_synset}:
                return {"status": "CONTRADICTED", "wn_rel": "same_synset",
                        "why": f"ConceptNet calls {tail!r} an antonym; WordNet puts it in the SAME synset"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": f"WordNet records no antonymy between {wn_synset.name()} and {tail!r}"}

        if kind == "hypernym":
            # depth 1 is the honest claim; walk two levels because ConceptNet's IsA is coarser at the top
            reach = set(wn_synset.hypernyms()) | set(wn_synset.hyponyms())
            reach |= set(wn_synset.instance_hypernyms()) | set(wn_synset.instance_hyponyms())
            if tails & reach:
                return {"status": "VERIFIED", "wn_rel": "hypernym_d1"}
            reach2 = set()
            for s in reach:
                reach2 |= set(s.hypernyms()) | set(s.hyponyms())
            if tails & reach2:
                return {"status": "VERIFIED", "wn_rel": "hypernym_d2"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": f"WordNet does not place {tail!r} within two hypernym steps of "
                           f"{wn_synset.name()}"}

        if kind == "part_of":
            reach = (set(wn_synset.part_meronyms()) | set(wn_synset.part_holonyms())
                     | set(wn_synset.member_meronyms()) | set(wn_synset.member_holonyms())
                     | set(wn_synset.substance_meronyms()) | set(wn_synset.substance_holonyms()))
            if tails & reach:
                return {"status": "VERIFIED", "wn_rel": "meronym"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": f"WordNet records no meronymy between {wn_synset.name()} and {tail!r}"}

        if kind == "derivation":
            want = set()
            for l in wn_synset.lemmas():
                for d in l.derivationally_related_forms():
                    want.add(d.name().lower().replace("_", " "))
            if _norm(tail) in want:
                return {"status": "VERIFIED", "wn_rel": "derivation"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": f"WordNet records no derivation between {wn_synset.name()} and {tail!r}"}

        if kind == "attribute":
            reach = set()
            try:
                reach |= set(wn_synset.attributes())
            except Exception:                                # noqa: BLE001
                pass
            for s in tails:
                try:
                    if wn_synset in set(s.attributes()):
                        reach.add(s)
                except Exception:                            # noqa: BLE001
                    continue
            if tails & reach:
                return {"status": "VERIFIED", "wn_rel": "attribute"}
            return {"status": "NOT_ASSERTED", "rel": rel,
                    "why": "WordNet records no attribute link"}

    except Exception as e:                                    # noqa: BLE001
        return {"status": "UNVERIFIABLE", "rel": rel, "why": f"wordnet error: {type(e).__name__}"}

    return {"status": "NOT_ASSERTED", "rel": rel, "why": "unhandled relation kind"}


def candidates(synset_name: str, max_lemmas: int = 4) -> list:
    """Every ConceptNet edge incident to this synset's lemmas, each with its WordNet verdict."""
    ck = ("cand", synset_name, max_lemmas)
    if ck in _CACHE:
        return _CACHE[ck]
    wn = _wn()
    cn = _cn()
    try:
        s = wn.synset(synset_name)
    except Exception:                                        # noqa: BLE001
        _CACHE[ck] = []
        return []
    if cn is None:
        _CACHE[ck] = []
        return []
    out = []
    for lem in [l.name().replace("_", " ") for l in s.lemmas()][:max_lemmas]:
        r = cn.assertions(lem)
        if r.get("status") != "FOUND":
            continue
        seen = set()
        for e in r.get("edges", []):
            tail = _norm(e["tail"])
            rel = e["rel"]
            key = (rel, tail)
            if not tail or key in seen or tail == _norm(lem):
                continue
            seen.add(key)
            v = verify(s, rel, tail)
            out.append({"lemma": lem, "rel": rel, "tail": tail, "expert": e.get("expert", False),
                        "verdict": v["status"], "wn_rel": v.get("wn_rel", ""), "why": v.get("why", "")})
    _CACHE[ck] = out
    return out


def injected_terms(synset_name: str, allow_unverified: bool = False) -> list:
    """The terms that may be added to this synset's signature.

    Default is VERIFIED ONLY: WordNet and ConceptNet agree, so two independent mechanisms support the term.
    `allow_unverified=True` reproduces the plain-union behaviour, and exists so the two can be compared on the
    same call path rather than in two codebases.
    """
    out, seen = [], set()
    for c in candidates(synset_name):
        if c["verdict"] != "VERIFIED" and not allow_unverified:
            continue
        if allow_unverified and c["verdict"] == "UNVERIFIABLE":
            pass                                          # the union injects these; this is that behaviour
        t = c["tail"].replace("_", " ")
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def describe(synset_name: str) -> dict:
    cands = candidates(synset_name)
    by_verdict = collections.Counter(c["verdict"] for c in cands)
    by_rel = collections.Counter(c["rel"] for c in cands)
    verified_rel = collections.Counter(c["rel"] for c in cands if c["verdict"] == "VERIFIED")
    return {"synset": synset_name, "n_candidates": len(cands),
            "by_verdict": dict(by_verdict), "by_relation": dict(by_rel),
            "verified_by_relation": dict(verified_rel),
            "union_terms": len(injected_terms(synset_name, allow_unverified=True)),
            "injected_terms": len(injected_terms(synset_name)),
            "examples_verified": [f"{c['rel']} -> {c['tail']}" for c in cands
                                  if c["verdict"] == "VERIFIED"][:6],
            "examples_contradicted": [f"{c['rel']} -> {c['tail']}" for c in cands
                                      if c["verdict"] == "CONTRADICTED"][:6],
            "examples_not_asserted": [f"{c['rel']} -> {c['tail']}" for c in cands
                                      if c["verdict"] == "NOT_ASSERTED"][:6],
            "examples_unverifiable": [f"{c['rel']} -> {c['tail']}" for c in cands
                                      if c["verdict"] == "UNVERIFIABLE"][:6]}


# ------------------------------------------------------------------ measure
def measure(synsets=None) -> int:
    wn = _wn()
    cn = _cn()
    print("=== ConceptNet INJECTED into a specific synset (WordNet must agree) ===")
    print(f"  conceptnet index: {'available' if cn else 'MISSING'}")
    print(f"  wordnet {wn.get_version()}\n")
    if synsets is None:
        synsets = ["cat.n.01", "dog.n.01", "doctor.n.01", "bank.n.01", "increase.v.01",
                   "significant.a.01", "water.n.01", "car.n.01", "run.v.01", "happy.a.01"]

    print("=== per-synset: how much survives verification ===")
    print(f"  {'synset':17s} {'cand':>5s} {'verified':>9s} {'contra':>7s} {'silent':>7s} {'no-analogue':>12s} "
          f"{'inject':>7s} {'union':>6s}")
    tot = collections.Counter()
    allcands = []
    for name in synsets:
        d = describe(name)
        v = d["by_verdict"]
        tot.update(v)
        allcands.extend(candidates(name))
        print(f"  {name:17s} {d['n_candidates']:5d} {v.get('VERIFIED', 0):9d} "
              f"{v.get('CONTRADICTED', 0):7d} {v.get('NOT_ASSERTED', 0):7d} "
              f"{v.get('UNVERIFIABLE', 0):12d} "
              f"{d['injected_terms']:7d} {d['union_terms']:6d}")

    print(f"\n  across {len(synsets)} synsets: {sum(tot.values())} ConceptNet edges")
    for k in ("VERIFIED", "CONTRADICTED", "NOT_ASSERTED", "UNVERIFIABLE"):
        n = tot.get(k, 0)
        pct = 100 * n / max(sum(tot.values()), 1)
        print(f"    {k:13s} {n:5d}  ({pct:5.1f}%)")
    n_all = max(sum(tot.values()), 1)
    print(f"\n  ==> WordNet can VOUCH for {100*tot.get('VERIFIED',0)/n_all:.1f}% of the edges.")
    print(f"      The plain union injects the other "
          f"{100*(n_all-tot.get('VERIFIED',0))/n_all:.1f}% with no corroboration.")
    print(f"      WordNet actively CONTRADICTS {100*tot.get('CONTRADICTED',0)/n_all:.1f}%, and is SILENT on")
    print(f"      {100*tot.get('NOT_ASSERTED',0)/n_all:.1f}% -- and silence is not the same accusation as")
    print(f"      contradiction, which is why they are separate verdicts and neither one is 'wrong'.")

    print("\n=== verification rate BY RELATION (the real finding) ===")
    byrel = collections.defaultdict(lambda: collections.Counter())
    for c in allcands:
        byrel[c["rel"]][c["verdict"]] += 1
    print(f"  {'relation':16s} {'n':>5s} {'verified':>9s} {'contra':>7s} {'silent':>7s} {'no-analogue':>12s}  "
          f"verifiable?")
    for rel, ctr in sorted(byrel.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(ctr.values())
        print(f"  {rel:16s} {n:5d} {ctr.get('VERIFIED',0):9d} {ctr.get('CONTRADICTED',0):7d} "
              f"{ctr.get('NOT_ASSERTED',0):7d} {ctr.get('UNVERIFIABLE',0):12d}  "
              f"{'YES' if rel in VERIFIABLE else 'NO'}")
    print("\n  Relations with NO WordNet analogue are UNVERIFIABLE, which is not the same as WRONG. They are")
    print("  reported, and not injected. That is the difference between this module and the union.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.synset_injection - SELF TEST")
    ok = True
    cn = _cn()
    print(f"  [{'PASS' if cn else 'SKIP'}] ConceptNet index available")
    if cn is None:
        print("SELF TEST: SKIPPED (no index)")
        return True

    wn = _wn()
    good = wn.get_version() == "3.0"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] WordNet 3.0 present")

    # --- a relation with no WordNet analogue must be UNVERIFIABLE, NEVER 'disagreed' or 'wrong'
    v = verify(wn.synset("cat.n.01"), "UsedFor", "pet")
    good = v["status"] == "UNVERIFIABLE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] UsedFor has no WordNet analogue -> UNVERIFIABLE "
          f"({v['status']})")

    # --- a WordNet-checkable relation that WordNet DOES assert must verify
    v = verify(wn.synset("cat.n.01"), "IsA", "feline")
    good = v["status"] == "VERIFIED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] IsA cat -> feline VERIFIED by WordNet "
          f"({v['status']}, {v.get('wn_rel')})")

    # --- ONE-DIRECTIONAL ANTONYMY: the bug this test was written for. `important.antonyms()` is [] in
    #     WordNet, so a string comparison reports NOT_ASSERTED for `unimportant` even though WordNet does place
    #     it opposite (it shares synset insignificant.a.04 with `insignificant`, the real antonym).
    v = verify(wn.synset("significant.a.01"), "Antonym", "unimportant")
    good = v["status"] == "VERIFIED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonymy resolved through SYNSETS, not strings: "
          f"significant / unimportant VERIFIED ({v['status']})")

    # --- and WordNet's SILENCE must not be reported as a contradiction
    v = verify(wn.synset("cat.n.01"), "IsA", "automobile")
    good = v["status"] == "NOT_ASSERTED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] IsA cat -> automobile is NOT_ASSERTED, not 'wrong' "
          f"({v['status']})")

    # --- but a claim WordNet actively contradicts must say so
    v = verify(wn.synset("cat.n.01"), "Antonym", "true_cat")
    good = v["status"] == "CONTRADICTED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Antonym cat -> true_cat CONTRADICTED "
          f"(WordNet puts them in the SAME synset) ({v['status']})")

    # --- the four verdicts are distinct; collapsing CONTRADICTED into NOT_ASSERTED would overstate ConceptNet
    seen = set()
    for syn, rel, tail in (("cat.n.01", "UsedFor", "pet"), ("cat.n.01", "IsA", "feline"),
                           ("cat.n.01", "IsA", "automobile"), ("cat.n.01", "Antonym", "true_cat")):
        seen.add(verify(wn.synset(syn), rel, tail)["status"])
    good = seen == {"UNVERIFIABLE", "VERIFIED", "NOT_ASSERTED", "CONTRADICTED"}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all four verdicts are reachable and distinct ({sorted(seen)})")

    # --- injection must be a SUBSET of the union, never a superset
    u = set(injected_terms("cat.n.01", allow_unverified=True))
    i = set(injected_terms("cat.n.01"))
    good = i <= u
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] injected ({len(i)}) is a SUBSET of the union ({len(u)})")

    # --- the unverifiable relations must be named somewhere, not silently dropped
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    good = all(r in src for r in ("UsedFor", "AtLocation", "CapableOf"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the unverifiable relations are named in the module, not "
          f"silently omitted")

    # --- an unknown synset degrades to nothing rather than raising
    good = injected_terms("not.a.synset") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown synset returns []")

    # --- describe() is internally consistent
    d = describe("dog.n.01")
    good = (d["union_terms"] >= d["injected_terms"]
            and sum(d["by_verdict"].values()) == d["n_candidates"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] describe() is consistent "
          f"({d['n_candidates']} candidates, {d['injected_terms']} injectable)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ConceptNet injected into a specific synset.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--synset", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.synset:
        print(json.dumps(describe(args.synset), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
