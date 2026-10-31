"""logos/nlp/attribute_relations.py â€” relations live on ATTRIBUTES, not on whole concepts.

THE ARGUMENT THIS IMPLEMENTS, AND WHERE IT COMES FROM
----------------------------------------------------
Dong Zhendong & Dong Qiang, *HowNet: a hybrid language and knowledge resource*, COLING 2010 system
demonstrations (C10-3014), read in full. HowNet is built on two nets, not one:

    CRN  Concept Relation Net     -- relations BETWEEN concepts
    ARN  Attribute Relation Net   -- relations among the ATTRIBUTES of concepts

And the ARN paragraph is the one that matters here, verbatim:

    "Fig. 2 reveals that it is not 'paper' as a whole that is related to 'write', but only one of its
     attributes, say 'color', is related to 'write' with 'contrast' as the condition. Therefore in a strict
     sense, 'paper' is not necessarily related to 'write'. We can sometimes even write on the sand with a twig
     or on the table with our wet finger. On the contrary, we cannot write on a piece of white paper with a
     chalk or on the blackboard in black ink. Therefore, for writing, what affects may not be the whole lot of
     the concept like 'paper', but some attributes of the concept."

WHY THIS DISSOLVES THE PROBLEM I HAVE BEEN STUCK ON
---------------------------------------------------
I have an edge `paper --UsedFor--> write` and I kept asking "which resource can CONFIRM this?" WordNet cannot,
frames cannot, and I reported 38.2% of ConceptNet edges as having no witness in the suite. The ARN argument says
the question was malformed: the relation does not hold between the CONCEPTS, it holds between an ATTRIBUTE of
one and an activity, under a CONDITION.

So `UsedFor(paper, write)` is not false, and it is not unverifiable -- it is UNDER-SPECIFIED. The well-formed
claim is something like

    UsedFor(paper, write)  REQUIRES  surface_writable(paper)
    surface_writable(paper)  HOLDS IF  colour_contrast(paper, writing_instrument)

and "we cannot write on white paper with chalk" is that qualifier FAILING, not the UsedFor edge being wrong.
That is why no concept-level witness could ever have confirmed it: the witness was being asked about the wrong
object.

WHAT THIS MODULE DOES
--------------------
    attributes(term)     the attributes WordNet exposes for a concept (attributes(), plus materials and parts)
    qualify(term, rel, tail)
                         QUALIFIED    -- the relation decomposes onto a named attribute + condition
                         UNDERSPECIFIED -- the relation is attribute-mediated but no attribute is available, so
                                          the honest answer is UNKNOWN and the edge must not be injected
                         NOT_ATTRIBUTE_MEDIATED -- the relation is fine at concept level (IsA, PartOf, Synonym)
    measure()            over real ConceptNet UsedFor / AtLocation / CapableOf edges, how many decompose

The qualifier is NOT a truth value. It is a precondition, which is exactly the shape the weighted MAX-SAT solver
already consumes: a conditional fact is one whose assertion depends on another atom. Only `constraint_solver`
decides; this module supplies the FORM.

Licence note repeated from earlier work: HowNet's data has no commercial-use grant that I could find, and its
Python client has had no release since 2021-12-16. Nothing here reads HowNet data. What is borrowed is the ARGUMENT,
which is a published idea, not an artefact.

USAGE
    python tools/logos/nlp/attribute_relations.py --self-test
    python tools/logos/nlp/attribute_relations.py --measure
    python tools/logos/nlp/attribute_relations.py --term paper --rel UsedFor --tail write
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

# Relations that hold between whole concepts: a witness exists and means what it says.
CONCEPT_LEVEL = ("Synonym", "Antonym", "IsA", "PartOf", "DistinctFrom", "DerivedFrom", "FormOf")

# Relations whose truth depends on WHICH ASPECT of the concept is in play. The ARN set.
ATTRIBUTE_MEDIATED = ("UsedFor", "CapableOf", "AtLocation", "HasProperty", "HasPrerequisite",
                      "ReceivesAction", "MotivatedByGoal", "Causes", "SubeventOf")

_CACHE = {}


def _wn():
    from nltk.corpus import wordnet as wn
    return wn


def available() -> tuple:
    try:
        v = _wn().get_version()
        return True, f"WordNet {v}"
    except Exception as e:                                   # noqa: BLE001
        return False, f"WordNet unavailable: {type(e).__name__}"


def _synsets(term: str, pos=None) -> list:
    wn = _wn()
    t = str(term or "").strip().replace(" ", "_")
    out = []
    for p in ([pos] if pos else ["n", "v", "a", "r"]):
        try:
            out.extend(wn.synsets(t, pos=p))
        except Exception:                                    # noqa: BLE001
            continue
    return out


def attributes(term: str) -> dict:
    """The attributes a concept exposes, from WordNet. These are the fillers an ARN relation would attach to.

    WordNet gives three usable channels:
        attributes()          noun -> adjective  (size.n.01 -> big/small)   the closest thing to ARN
        substance_meronyms()  what a thing is MADE OF (the "paper is like cloth or plastic" case)
        part_meronyms()       what a thing is composed of
    """
    ck = ("attrs", term)
    if ck in _CACHE:
        return _CACHE[ck]
    syns = _synsets(term)
    if not syns:
        _CACHE[ck] = {"status": "NO_SYNSETS", "term": term, "attributes": [], "materials": [],
                      "parts": []}
        return _CACHE[ck]
    attrs, mats, parts = [], [], []
    for s in syns[:4]:
        try:
            for a in s.attributes():
                attrs.append({"name": a.name(), "via": s.name(), "channel": "attribute"})
        except Exception:                                    # noqa: BLE001
            pass
        try:
            for m in s.substance_meronyms():
                mats.append({"name": m.name(), "via": s.name(), "channel": "material"})
        except Exception:                                    # noqa: BLE001
            pass
        try:
            for p in s.part_meronyms():
                parts.append({"name": p.name(), "via": s.name(), "channel": "part"})
        except Exception:                                    # noqa: BLE001
            pass
    out = {"status": "OK", "term": term, "n_synsets": len(syns),
           "attributes": attrs[:24], "materials": mats[:16], "parts": parts[:16],
           "n_attributes": len(attrs), "n_materials": len(mats), "n_parts": len(parts)}
    _CACHE[ck] = out
    return out


def attribute_available(term: str) -> bool:
    a = attributes(term)
    return bool(a["n_attributes"] or a["n_materials"] or a["n_parts"])


def qualify(term: str, rel: str, tail: str) -> dict:
    """Turn a concept-level edge into its well-formed conditional form, or say it cannot be decomposed.

    QUALIFIED         the relation is attribute-mediated AND an attribute exists to carry it, so the claim is
                      representable as `rel(term, tail) REQUIRES attr(term)`.
    UNDERSPECIFIED    the relation is attribute-mediated and NO attribute is available. This is the honest
                      verdict for the edge: not false, not verified, and NOT injectable. It is the category
                      that replaces my old "UNVERIFIABLE", which blamed the claim for the gap.
    NOT_ATTRIBUTE_MEDIATED
                      the relation is concept-level, so this module has nothing to add and a concept-level
                      witness should be consulted instead.
    """
    if rel in CONCEPT_LEVEL:
        return {"status": "NOT_ATTRIBUTE_MEDIATED", "rel": rel, "term": term, "tail": tail,
                "why": f"{rel} holds between whole concepts; consult a concept-level witness, not an "
                       f"attribute qualifier"}
    if rel not in ATTRIBUTE_MEDIATED:
        return {"status": "UNKNOWN_RELATION", "rel": rel, "term": term, "tail": tail}

    a = attributes(term)
    attrs = a["attributes"] + a["materials"]
    if not attrs:
        return {"status": "UNDERSPECIFIED", "rel": rel, "term": term, "tail": tail,
                "why": f"the relation depends on some aspect of {term!r}, and WordNet exposes no attribute "
                       f"or material for it, so the edge cannot be given a well-formed conditional form",
                "n_attributes": a.get("n_attributes", 0)}

    # The condition is the tail's own nature: what the activity or location requires of the term.
    ta = attributes(tail)
    return {"status": "QUALIFIED", "rel": rel, "term": term, "tail": tail,
            "requires": [{"atom": f"{term}|attribute|{x['name']}", "channel": x["channel"],
                          "via": x["via"]} for x in attrs[:4]],
            "condition": {"on": tail, "aspects": [x["name"] for x in (ta.get("attributes") or [])][:4]},
            "form": f"{rel}({term}, {tail}) REQUIRES attribute({term})",
            "why": "the relation holds between an ASPECT of the term and the tail, under a condition -- "
                   "not between the two concepts as wholes"}


def to_solver_input(edges, attested=None) -> dict:
    """Turn qualified edges into `constraint_solver` input, so a conditional claim is only asserted when its
    qualifier is ATTESTED BY SOMETHING.

    `edges`    iterable of (term, rel, tail).
    `attested` the set of qualifier atoms a source actually supports. Anything not in here is bound HARD false.

    Each QUALIFIED edge becomes
        fact   term|rel|tail        weight 1.0  (retractable)
        atom   term|attribute|<n>   the qualifier
        clause A -> P  ==  P or not A
    and a qualifier NOT in `attested` is added to `false_atoms`, so the clause forces the edge to be retracted.
    An UNDERSPECIFIED edge is never turned into a fact at all.
    """
    attested = set(attested or ())
    facts, weights, requires, skipped, false_atoms = [], {}, [], [], []
    for term, rel, tail in edges:
        q = qualify(term, rel, tail)
        if q["status"] != "QUALIFIED":
            skipped.append({"edge": f"{term} {rel} {tail}", "status": q["status"]})
            continue
        key = f"{term}|{rel}|{tail}"
        facts.append(key)
        weights[key] = 1.0
        pre = [x["atom"] for x in q["requires"]]
        requires.append((key, pre))
        for p in pre:
            facts.append(p)
            weights[p] = 1.0
            if p not in attested:
                false_atoms.append(p)
    return {"facts": facts, "weights": weights, "requires": requires, "skipped": skipped,
            "false_atoms": false_atoms, "n_attested": len(attested)}


def discharge(edges, attested=None) -> dict:
    """Assert the qualified edges and report which survived, with the retraction set as the proof.

    The result is the answer to "which of these attribute-mediated claims can this suite actually stand behind?"
    and it names the ones it cannot, rather than silently including them in a signature.
    """
    try:
        from logos.logic import constraint_solver as CS
    except Exception as e:                                   # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"{type(e).__name__}: {e}"}
    spec = to_solver_input(edges, attested=attested)
    if not spec["facts"]:
        return {"status": "NO_QUALIFIED_EDGES", "n_input": len(list(edges)) if hasattr(edges, "__len__") else 0,
                "skipped": spec["skipped"]}
    r = CS.solve(spec["facts"], requires=spec["requires"], weights=spec["weights"],
                 false_atoms=spec["false_atoms"])
    held = set(r.get("held") or [])
    edge_keys = [f for f, _ in spec["requires"]]
    kept = [k for k in edge_keys if k in held]
    lost = [k for k in edge_keys if k not in held]
    return {"status": r.get("status"), "n_edges": len(edge_keys), "n_kept": len(kept),
            "n_disqualified": len(lost), "kept": kept, "disqualified": lost,
            "skipped": spec["skipped"][:8], "cost": r.get("cost"), "unique": r.get("unique"),
            "why": "a QUALIFIED edge is asserted only if its attribute qualifier survives; an edge whose "
                   "condition cannot be established is retracted and NAMED, not silently included"}


# ------------------------------------------------------------------ measure
def measure() -> int:
    ok, why = available()
    print("=== attribute-mediated relations (the ARN argument, applied) ===")
    print(f"  {why}\n")
    if not ok:
        return 1

    print("=== the worked case from the paper: 'paper' and 'write' ===")
    a = attributes("paper")
    print(f"  paper: {a.get('n_attributes', 0)} attribute(s), {a.get('n_materials', 0)} material(s), "
          f"{a.get('n_parts', 0)} part(s)")
    for ch, key in (("attribute", "attributes"), ("material", "materials"), ("part", "parts")):
        names = [x["name"] for x in a.get(key, [])][:6]
        if names:
            print(f"    {ch:10s} {names}")
    q = qualify("paper", "UsedFor", "write")
    print(f"  UsedFor(paper, write) -> {q['status']}")
    if q["status"] == "QUALIFIED":
        print(f"    form      {q['form']}")
        print(f"    requires  {[x['atom'] for x in q['requires']]}")
    else:
        print(f"    {q.get('why')}")

    print("\n=== the classification, on relations I can name ===")
    u = qualify("water", "UsedFor", "drink")
    print(f"  UsedFor(water, drink)  -> {u['status']}")
    if u["status"] == "QUALIFIED":
        print(f"    form      {u['form']}")
    i = qualify("cat", "IsA", "feline")
    print(f"  IsA(cat, feline)       -> {i['status']}")

    print("\n=== over REAL ConceptNet edges: how many can be given a conditional form? ===")
    try:
        from logos.nlp import synset_injection as S
    except Exception as e:                                   # noqa: BLE001
        print(f"  (synset_injection unavailable: {type(e).__name__})")
        return 0
    terms = ["cat.n.01", "dog.n.01", "doctor.n.01", "bank.n.01", "increase.v.01",
             "significant.a.01", "water.n.01", "car.n.01", "run.v.01", "happy.a.01"]
    counts = collections.Counter()
    examples = {"QUALIFIED": [], "UNDERSPECIFIED": []}
    n = 0
    for name in terms:
        base = name.split(".")[0]
        for c in S.candidates(name):
            if c["rel"] not in ATTRIBUTE_MEDIATED:
                continue
            n += 1
            q = qualify(base, c["rel"], str(c["tail"]).replace("_", " "))
            counts[q["status"]] += 1
            if len(examples.get(q["status"], [])) < 5:
                examples.setdefault(q["status"], []).append(f"{c['rel']}({base}, {c['tail']})")
    print(f"  attribute-mediated ConceptNet edges examined: {n}")
    for k, v in counts.most_common():
        print(f"    {k:16s} {v:4d}  ({100*v/max(n,1):5.1f}%)")
    print()
    for k in ("QUALIFIED", "UNDERSPECIFIED"):
        if examples.get(k):
            print(f"  {k} examples: {examples[k]}")
    print()
    print("  READ THIS: this is the number that replaces my old claim that 38.2% of ConceptNet edges have 'no")
    print("  witness'. A QUALIFIED edge is not unverifiable -- it is a CONDITIONAL claim, and the condition is")
    print("  what a solver should evaluate. An UNDERSPECIFIED edge is the honest residue: the relation is real")
    print("  but this suite cannot state it well-formed yet, so it must not be injected into a signature.")

    # ---- and now actually DISCHARGE them through the solver, which is the point of having a conditional form
    print("\n=== discharging the qualified edges through weighted MAX-SAT ===")
    qedges = []
    for name in terms:
        base = name.split(".")[0]
        for c in S.candidates(name):
            if c["rel"] in ATTRIBUTE_MEDIATED:
                qedges.append((base, c["rel"], str(c["tail"]).replace("_", " ")))
    d = discharge(qedges)
    print(f"  input attribute-mediated edges: {len(qedges)}")
    print(f"  status            {d.get('status')}")
    print(f"  QUALIFIED edges   {d.get('n_edges', 0)}")
    print(f"  kept              {d.get('n_kept', 0)}")
    print(f"  disqualified      {d.get('n_disqualified', 0)}")
    print(f"  cost              {d.get('cost')}  unique: {d.get('unique')}")
    if d.get("kept"):
        print(f"  examples KEPT:        {d['kept'][:4]}")
    if d.get("disqualified"):
        print(f"  examples DISQUALIFIED: {d['disqualified'][:4]}")
    print()
    print("  AND THIS IS THE HONEST RESULT: an attribute qualifier has no support of its own, so the cheapest")
    print("  consistent assignment is to RETRACT the conditional edge. Almost every edge is disqualified --")
    print("  which is the correct outcome, not a failure. It says: without a source that attests WHICH ASPECT of")
    print("  the concept is in play, these claims cannot be stood behind. The ARN argument gives them a correct")
    print("  FORM; it does not by itself give them a WITNESS. That is what a resource like HowNet's ARN would")
    print("  supply, and it is not in this suite.")

    # ---- the POSITIVE direction, because "rejects everything" is indistinguishable from "broken"
    if d.get("disqualified"):
        target = d["disqualified"][0]
        q = qualify(*target.split("|"))
        one = [x["atom"] for x in q.get("requires", [])][:1]
        d2 = discharge(qedges, attested=one)
        print(f"\n=== and WITH a source attesting one qualifier ({one[0] if one else '-'}) ===")
        print(f"  n_kept {d2.get('n_kept')}  n_disqualified {d2.get('n_disqualified')}  cost {d2.get('cost')}")
        if d2.get("kept"):
            print(f"  NOW KEPT: {d2['kept'][:3]}")
        print("  So the mechanism is not a blanket rejection: attest the ASPECT and the edge is dischargeable.")
        print("  That is the measured difference between a conditional claim with a witness and one without.")
        print()
        print(f"  BUT READ THE GRANULARITY, because it is a real limitation: {d2.get('n_kept')} of 13 edges")
        print("  became dischargeable from ONE attested aspect. `qualify()` returns the attribute set of the")
        print("  CONCEPT, not of the relation, so a single attested aspect unlocks EVERY relation on that")
        print("  concept. That is more permissive than the ARN argument implies -- `water UsedFor cleaning`")
        print("  depends on water's solvent property, not on its oxygen content, and this module cannot tell")
        print("  those apart. Per-relation aspect selection is what a resource with ARN data would supply;")
        print("  the qualifier here is per-concept and is therefore only a coarse bound, not a judgement.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.attribute_relations - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] {why}")
    if not av:
        print("SELF TEST: SKIPPED (no WordNet)")
        return True

    # --- concept-level relations are NOT attribute-mediated, and the module says so instead of guessing
    q = qualify("cat", "IsA", "feline")
    good = q["status"] == "NOT_ATTRIBUTE_MEDIATED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] IsA is concept-level, so no attribute qualifier is invented "
          f"({q['status']})")

    # --- an attribute-mediated relation that CAN be decomposed yields a REQUIRES form
    q = qualify("water", "UsedFor", "drink")
    good = q["status"] == "QUALIFIED" and "REQUIRES" in q["form"] and q["requires"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] UsedFor(water, drink) decomposes to a conditional form "
          f"({q['status']})")

    # --- and one that cannot is UNDERSPECIFIED, which is NOT the same as false and NOT the same as unverifiable
    q = qualify("zzznosuchconcept", "UsedFor", "write")
    good = q["status"] == "UNDERSPECIFIED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an undecomposable edge is UNDERSPECIFIED, not false and not "
          f"'unverifiable' ({q['status']})")

    # --- the qualifier is a precondition atom, not a truth value: it must be assertable
    q = qualify("water", "UsedFor", "drink")
    good = all("|attribute|" in x["atom"] for x in q["requires"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] qualifiers are precondition ATOMS a solver can evaluate "
          f"({q['requires'][0]['atom'] if q['requires'] else '-'})")

    # --- WordNet's three attribute channels are surfaced distinctly, because they answer different questions
    a = attributes("water")
    good = a["status"] == "OK" and (a["n_attributes"] + a["n_materials"] + a["n_parts"]) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] attributes() reports attribute/material/part separately "
          f"({a['n_attributes']}/{a['n_materials']}/{a['n_parts']} for 'water')")

    # --- a missing concept degrades to NO_SYNSETS rather than raising
    a2 = attributes("zzznosuchconcept")
    good = a2["status"] == "NO_SYNSETS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown term returns NO_SYNSETS ({a2['status']})")

    # --- the two sets are disjoint, so a relation cannot be both concept-level and attribute-mediated
    good = not (set(CONCEPT_LEVEL) & set(ATTRIBUTE_MEDIATED))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the concept-level and attribute-mediated sets are disjoint")

    # --- and the qualifier feeds the MAX-SAT solver's `requires` shape without translation
    try:
        from logos.logic import constraint_solver as CS
        q = qualify("water", "UsedFor", "drink")
        atom = q["requires"][0]["atom"]
        f1 = CS.fact("water", "UsedFor", "drink")
        r = CS.solve([f1, atom], functional=())
        good = r["status"] == CS.CONSISTENT
    except Exception as e:                                   # noqa: BLE001
        good = False
        print(f"        (solver check failed: {type(e).__name__}: {e})")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a qualifier atom is consumable by constraint_solver as-is")

    # --- THE MODELLING ERROR, AS A REGRESSION TEST. A precondition that is free to assert is NOT a constraint:
    #     `A -> P` is satisfied by simply asserting P at zero cost, so 13 of 13 edges came back KEPT. The
    #     qualifier must be bound HARD FALSE unless something attests it.
    edges = [("water", "UsedFor", "cleaning")]
    free = discharge(edges)                                     # unattested -> qualifier false
    good = free["n_disqualified"] == 1 and free["n_kept"] == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an UNATTESTED precondition disqualifies the edge "
          f"(kept={free['n_kept']}, disqualified={free['n_disqualified']})")

    # --- and attesting the qualifier flips it, so the mechanism is not a blanket rejection
    q = qualify("water", "UsedFor", "cleaning")
    one = [x["atom"] for x in q.get("requires", [])][:1]
    if one:
        att = discharge(edges, attested=one)
        good = att["n_kept"] == 1 and att["n_disqualified"] == 0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] an ATTESTED precondition keeps the edge ({one[0]})")
    else:
        print("  [SKIP] no qualifier atom came back for water/UsedFor/cleaning")
        good = True

    # --- a precondition is a distinct atom, not a duplicate of the edge
    good = q["requires"][0]["atom"] != "water|UsedFor|cleaning"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the qualifier is its own atom, so it can carry its own provenance")

    # --- an UNDERSPECIFIED edge must never enter the solver as though it were assertable
    spec = to_solver_input([("zzznosuchconcept", "UsedFor", "write")])
    good = spec["facts"] == [] and spec["skipped"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an UNDERSPECIFIED edge contributes no fact to the solver")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attribute-mediated relations: the ARN argument.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--term", default="")
    ap.add_argument("--rel", default="UsedFor")
    ap.add_argument("--tail", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.term:
        print(json.dumps(qualify(args.term, args.rel, args.tail) if args.tail else attributes(args.term),
                         indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())

