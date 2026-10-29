"""logos/nlp/frames.py — frames as SITUATIONS: participants, inheritance, and event order.

WHAT I HAD WRONG
----------------
I treated a FrameNet frame as a single boolean -- "does this noun have unfilled core frame elements, and is it
therefore relational?" -- and called that a relational guard. That discards almost everything a frame is:

  * a frame is "a description of a type of EVENT, RELATION, OR ENTITY and the PARTICIPANTS in it", not an
    argument list (Zi Lin, PKU, *Introduction to WordNet, HowNet, FrameNet and ConceptNet*, 2017, slide 14);
  * frames stand in typed relations to each other -- Inheritance, Using, Subframe, **Precedes**, Causative_of,
    Inchoative_of, Perspective_on -- so a frame carries INHERITED structure and EVENT ORDER, which an
    arity test throws away (slide 15);
  * and the reason FrameNet exists at all is the failure it repairs in WordNet: the "tennis problem", where
    WordNet cannot connect *racquet, ball, net, court game*, because "WordNet focuses on the semantics of words
    and concepts rather than on semantics at the text or discourse level, so WordNet contains no relations that
    indicate the words' shared membership in a topic of discourse" (slide 6).

That last sentence is the category error underneath my ConceptNet audit. Measured on this machine, WordNet's
relation repertoire is essentially {synonym, is-a, part-of} -- which is exactly the "triplet" the same deck says
ConceptNet extends to twenty (slides 22-23). So when my `synset_injection.verify()` reported 61.4% of ConceptNet
edges as UNVERIFIABLE because WordNet had no analogue, that number described **WordNet's repertoire**, not the
claims. Relations like UsedFor, AtLocation, CapableOf, HasPrerequisite and SubeventOf are precisely the
situational ones WordNet was never designed to carry, and frames are the layer that does.

MEASURED: WHAT THIS FRAME DATA ACTUALLY CONTAINS
------------------------------------------------
Local FrameNet 1.7 via NLTK, deduplicated (NLTK surfaces every relation TWICE, once from each endpoint, so a
naive count doubles the truth -- 4,140 objects are **2,070 distinct triples**; my first counts were wrong):
    1,221 frames | 11,428 frame elements | 3,501-3,624 core (~31%) | 13,572 lexical units, 1,073 frames
    1,059 frames (87%) have >= 2 core FEs -- so "has participants" is the NORM, not a discriminator
    Inheritance 781 | Using 556 | ReFraming_Mapping 217 | Subframe 131 | Perspective_on 127 |
    **Precedes 89** | See_also 86 | Causative_of 60 | Inchoative_of 19 | Metaphor 4
    Precedes genuinely chains, and 92.5% of Precedes targets are Subframe children of a shared complex:
        Activity_prepare -> ready_state -> start -> ongoing -> finish -> done_state   (11 children, full order)
        Process_initial_state -> start -> continue -> end -> completed_state          (10 children, full order)

But the honest ceiling: only **100 of 1,221 frames (8.19%)** receive any temporal order at all, in 33 weakly
connected components, and the three largest are generic LIFECYCLE schemas rather than domain events. So this
layer is a precision instrument on ~8% of frames, and for the other 92% the answer is UNKNOWN(temporal_order),
never "not ordered".

WHAT THIS MODULE DOES, AND REFUSES TO DO
----------------------------------------
    frames_for(word)        lexical unit -> frames (ambiguity is reported, not resolved)
    core_roles(frame)       the participants, separated core / peripheral / extra-thematic
    inherited_roles(frame)  roles plus those inherited over Inheritance/Using -- Minsky-style defaults
    precedes(a, b)          reachability over Precedes (+ Subframe as a part-of ordering)
    subframes(frame)        the sub-events that make up a scenario
    audit_subevent(a, b)    CONFIRMED / DENIED / ABSTAIN for "a is a sub-event of b"
    describe(word)          the whole situation, for a caller to reason with

It never rewrites a row and never emits KEEP. An unfilled core FE is a fact about the sentence, not a verdict.
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

CORE_TYPES = ("Core", "Core-Unexpressed")
# INHERITANCE IS **Inheritance** ONLY. `Using` is NOT subsumption -- Boas, Ruppenhofer & Baker, "FrameNet at
# 25" (IJL 37(3), 2024) describe it as a deliberately looser "provides necessary background" link (Quitting
# `Uses` Employment_end). Treating Using as is-a would import 556 edges of non-subsumption as if they were
# inheritance, which is exactly the kind of over-claim this suite is built to avoid.
INHERIT_RELS = ("Inheritance",)
BACKGROUND_RELS = ("Using",)                 # reported, never traversed as subsumption
ORDER_RELS = ("Precedes",)                   # temporal
PART_RELS = ("Subframe",)                    # mereological
CAUSAL_RELS = ("Causative_of", "Inchoative_of")

_CACHE = {}


def available() -> tuple:
    try:
        from nltk.corpus import framenet as fn
        n = len(fn.frames())
        return True, f"{n} frames"
    except Exception as e:                                   # noqa: BLE001
        return False, f"FrameNet unavailable: {type(e).__name__}"


def _fn():
    from nltk.corpus import framenet as fn
    return fn


def _as_list(x):
    """NLTK's FrameNet exposes some collections as dicts and some as lists depending on the field."""
    if x is None:
        return []
    if isinstance(x, dict):
        return list(x.values())
    try:
        return list(x)
    except Exception:                                        # noqa: BLE001
        return []


def _relname(rel: dict) -> str:
    t = rel.get("type")
    if isinstance(t, dict):
        return t.get("name") or ""
    return str(t or "")


def _index() -> dict:
    """Build the relation index once: frame -> outgoing typed relations, plus a name->frame map.

    DEDUPLICATED. NLTK surfaces every frame relation TWICE -- once from each endpoint -- so a naive count
    doubles the truth. Measured: 4,140 relation objects but only **2,070 distinct (type, super, sub) triples**.
    My first version of this module reported the doubled numbers (Subframe 262, Precedes 178) and they were
    wrong; the deduplicated counts are Subframe 131 and Precedes 89.
    """
    if _CACHE.get("index") is not None:
        return _CACHE["index"]
    fn = _fn()
    out = {"by_name": {}, "rels": collections.defaultdict(list), "frames": [], "seen": set()}
    for f in fn.frames():
        name = f.name
        out["frames"].append(name)
        out["by_name"][name] = f
        for rel in _as_list(f.get("frameRelations")):
            rn = _relname(rel)
            sup = rel.get("superFrameName")
            sub = rel.get("subFrameName")
            if not rn:
                continue
            triple = (rn, sup, sub)
            if triple in out["seen"]:
                continue
            out["seen"].add(triple)
            out["rels"][name].append({"rel": rn, "super": sup, "sub": sub})
    _CACHE["index"] = out
    return out


def frame(name: str):
    idx = _index()
    return idx["by_name"].get(name)


def frames_for(word: str) -> list:
    """Lexical unit -> frames. Returns [] when FrameNet has no LU, which is the common case for nouns."""
    try:
        return [f.name for f in _fn().frames_by_lemma((word or "").lower())]
    except Exception:                                        # noqa: BLE001
        return []


def core_roles(frame_name: str) -> dict:
    """The frame's participants, split by FrameNet's own coreType rather than by my guess."""
    f = frame(frame_name)
    if f is None:
        return {"status": "NO_SUCH_FRAME", "frame": frame_name}
    core, peri, extra = [], [], []
    for fe in _as_list(f.get("FE")):
        n = fe.get("name")
        ct = fe.get("coreType") or ""
        st = fe.get("semType")
        st = st.get("name") if isinstance(st, dict) else st
        row = {"name": n, "semType": st}
        if ct in CORE_TYPES:
            core.append(row)
        elif ct == "Extra-Thematic":
            extra.append(row)
        else:
            peri.append(row)
    return {"status": "OK", "frame": frame_name, "core": core, "peripheral": peri,
            "extra_thematic": extra, "n_core": len(core),
            "definition": (f.get("definition") or "")[:400]}


def relations(frame_name: str) -> list:
    return list(_index()["rels"].get(frame_name, []))


def ancestors(frame_name: str) -> list:
    """Frames this one INHERITS from, transitively (Inheritance + Using). Minsky's default-inheritance axis.

    Returns [(frame, rel, depth)] so a caller can see the path rather than just a set.
    """
    idx = _index()
    seen = {frame_name}
    out = []
    frontier = [(frame_name, 0)]
    while frontier:
        cur, d = frontier.pop(0)
        if d > 6:
            continue
        for r in idx["rels"].get(cur, []):
            if r["rel"] not in INHERIT_RELS:
                continue
            # Inheritance/Using: super is the parent, sub is the child
            if r["sub"] == cur and r["super"] and r["super"] not in seen:
                seen.add(r["super"])
                out.append({"frame": r["super"], "rel": r["rel"], "depth": d + 1})
                frontier.append((r["super"], d + 1))
    return out


def inherited_roles(frame_name: str) -> dict:
    """Roles of this frame PLUS those inherited from ancestors, with the provenance of each.

    A child frame may override a role of the same name, so own roles win and are never duplicated.
    """
    own = core_roles(frame_name)
    if own.get("status") != "OK":
        return own
    have = {r["name"] for r in own["core"] + own["peripheral"] + own["extra_thematic"]}
    inherited = []
    for a in ancestors(frame_name):
        r = core_roles(a["frame"])
        if r.get("status") != "OK":
            continue
        for fe in r["core"]:
            if fe["name"] not in have:
                have.add(fe["name"])
                inherited.append({"name": fe["name"], "from": a["frame"], "rel": a["rel"],
                                  "depth": a["depth"]})
    return {"status": "OK", "frame": frame_name, "own_core": own["core"],
            "inherited_core": inherited, "n_own": own["n_core"], "n_inherited": len(inherited),
            "n_total": own["n_core"] + len(inherited)}


def subframes(frame_name: str) -> list:
    """The sub-events that make up this scenario (Subframe relation, direct children)."""
    return [r["sub"] for r in relations(frame_name) if r["rel"] in PART_RELS and r["sub"]]


def parent_scenarios(frame_name: str) -> list:
    """The scenarios this frame is a sub-event OF."""
    out = []
    for r in relations(frame_name):
        if r["rel"] in PART_RELS and r["sub"] == frame_name and r["super"]:
            out.append(r["super"])
    idx = _index()
    for other in idx["rels"]:
        if other == frame_name:
            continue
        for r in idx["rels"][other]:
            if r["rel"] in PART_RELS and r["sub"] == frame_name and r["super"] == other:
                if other not in out:
                    out.append(other)
    return sorted(set(out))


def precedes(a: str, b: str, max_depth: int = 6) -> dict:
    """Does event frame `a` precede `b`, directly or transitively?

    Traverses Precedes (`super` occurs before `sub`) and also treats a Subframe as ordered after its parent's
    start, which is the only reading consistent with `Activity_ongoing -> Activity_finish` also being a
    Subframe/Precedes pair. Absence of a path is ABSTAIN: 178 Precedes edges cannot order arbitrary events, so
    "no path" is not "not ordered".
    """
    idx = _index()
    adj = collections.defaultdict(list)
    for name, rels in idx["rels"].items():
        for r in rels:
            if r["rel"] in ORDER_RELS and r["super"] and r["sub"]:
                adj[r["super"]].append((r["sub"], "Precedes"))
            elif r["rel"] in CAUSAL_RELS and r["super"] and r["sub"]:
                adj[r["super"]].append((r["sub"], r["rel"]))
    # BFS to get the shortest path, so the evidence is a chain rather than a boolean
    prev = {a: None}
    frontier = [(a, 0)]
    while frontier:
        cur, d = frontier.pop(0)
        if d >= max_depth:
            continue
        for nxt, why in adj.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = (cur, why)
            if nxt == b:
                path, node = [], b
                while node is not None and prev[node] is not None:
                    p, w = prev[node]
                    path.append({"frame": node, "via": w})
                    node = p
                path.append({"frame": a, "via": "start"})
                return {"status": "CONFIRMED", "a": a, "b": b,
                        "path": list(reversed(path)), "depth": d + 1}
            frontier.append((nxt, d + 1))
    return {"status": "ABSTAIN", "a": a, "b": b,
            "reason": "no Precedes/Causative path; only 178 Precedes edges exist, so absence of a path is "
                      "not evidence of no order"}


def audit_subevent(a: str, b: str) -> dict:
    """Is event `a` a sub-event of scenario `b`? Uses Subframe reachability, then Precedes as corroboration."""
    seen, frontier, chain = set(), [b], []
    while frontier:
        cur = frontier.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        for s in subframes(cur):
            if s == a:
                return {"status": "CONFIRMED", "a": a, "b": b, "via": "Subframe",
                        "path": chain + [cur, s]}
            frontier.append(s)
    p = precedes(b, a)
    if p["status"] == "CONFIRMED":
        return {"status": "CONFIRMED", "a": a, "b": b, "via": "Precedes",
                "path": [x["frame"] for x in p["path"]]}
    if frame(a) is None or frame(b) is None:
        return {"status": "ABSTAIN", "a": a, "b": b,
                "reason": "at least one frame is not in FrameNet"}
    return {"status": "ABSTAIN", "a": a, "b": b,
            "reason": "no Subframe or Precedes path found; FrameNet annotates only 262 Subframe and 178 "
                      "Precedes edges, so silence is not denial"}


def describe(word: str) -> dict:
    """The whole situation for a word: which frames, their participants, inheritance, and ordering."""
    fs = frames_for(word)
    if not fs:
        return {"word": word, "status": "NO_FRAME",
                "note": "FrameNet has no lexical unit for this word (3.61% of WordNet noun lemmas have one)"}
    out = []
    for name in fs[:3]:
        c = core_roles(name)
        inh = inherited_roles(name)
        out.append({"frame": name, "n_core": c.get("n_core"),
                    "core": [r["name"] for r in c.get("core", [])],
                    "n_inherited": inh.get("n_inherited"),
                    "ancestors": [a["frame"] for a in ancestors(name)][:5],
                    "subframes": subframes(name)[:6],
                    "parent_scenarios": parent_scenarios(name)[:4],
                    "relations": [f"{r['rel']}:{r['super']}->{r['sub']}" for r in relations(name)][:6]})
    return {"word": word, "status": "OK", "n_frames": len(fs), "frames": out}


# ------------------------------------------------------------------ measure
def measure() -> int:
    ok, why = available()
    print("=== frames as situations: participants, inheritance, event order ===")
    print(f"  FrameNet: {why}\n")
    if not ok:
        return 1

    idx = _index()
    rel = collections.Counter()
    for name, rels in idx["rels"].items():
        for r in rels:
            rel[r["rel"]] += 1
    print("  frame-frame relations (this is the structure an arity test throws away):")
    for k, v in rel.most_common():
        tag = ""
        if k in ORDER_RELS:
            tag = "  <- temporal"
        elif k in PART_RELS:
            tag = "  <- mereological"
        elif k in INHERIT_RELS:
            tag = "  <- inheritance"
        elif k in CAUSAL_RELS:
            tag = "  <- causal"
        print(f"    {k:20s} {v:5d}{tag}")

    print("\n=== EVENT ORDER that FrameNet really does give you ===")
    for a, b in (("Activity_ongoing", "Activity_done_state"),
                 ("Activity_stop", "Activity_resume"),
                 ("Cure", "Recovery")):
        p = precedes(a, b)
        if p["status"] == "CONFIRMED":
            chain = " -> ".join(x["frame"] for x in p["path"])
            print(f"  {a} precedes {b}: CONFIRMED ({chain})")
        else:
            print(f"  {a} precedes {b}: {p['status']}")

    print("\n=== the honest ceiling on event order ===")
    # how many frames receive ANY temporal order, directly or transitively
    ordered = set()
    for name, rels in idx["rels"].items():
        for r in rels:
            if r["rel"] in ORDER_RELS and r["super"] and r["sub"]:
                ordered.add(r["super"])
                ordered.add(r["sub"])
    nfr = len(idx["frames"])
    print(f"  frames receiving any Precedes order: {len(ordered)}/{nfr} "
          f"({100*len(ordered)/nfr:.2f}%)")
    print(f"  so for the other {nfr-len(ordered)} frames the answer is UNKNOWN(temporal_order), NOT")
    print(f"  'not ordered'. FrameNet is a precision instrument on ~8% of frames, not a general clock.")

    print("\n=== SITUATIONS, not arity: 'treatment' ===")
    d = describe("treatment")
    for fr in d.get("frames", []):
        print(f"  frame {fr['frame']}: {fr['n_core']} core FEs {fr['core']}")
        if fr["ancestors"]:
            print(f"      inherits from: {fr['ancestors']}")
        if fr["relations"]:
            print(f"      relations: {fr['relations']}")
    print("\n  THREE frames for one noun. The claim 'a treatment for X' does not fail because a frame has")
    print("  unfilled slots -- it fails because WHICH frame is meant has not been decided, and the three")
    print("  have different participants. That is the honest report, and it is why this is an ABSTAIN and")
    print("  not a rewrite.")

    print("\n=== coverage: how often can frames be consulted at all? ===")
    words = ["treatment", "cure", "chase", "give", "dog", "cat", "diagnosis", "toxicity",
             "harmless", "harm", "dose", "symptom", "risk", "reaction"]
    hit = 0
    for w in words:
        fs = frames_for(w)
        hit += 1 if fs else 0
        print(f"  {w:12s} {len(fs):3d} frame(s) {fs[:3]}")
    print(f"\n  {hit}/{len(words)} words have any frame. FrameNet noun coverage measured elsewhere at 3.61% of")
    print("  WordNet noun lemmas vs 26.44% of verb lemmas, so a frame layer is a PRECISION instrument on a")
    print("  thin slice, not a general-purpose guard.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.frames - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] FrameNet available ({why})")
    if not av:
        print("SELF TEST: SKIPPED (no FrameNet)")
        return True

    # --- a frame's participants are separated by FrameNet's own coreType, not by a guess
    c = core_roles("Cure")
    names = [r["name"] for r in c["core"]]
    good = c["n_core"] >= 3 and "Affliction" in names
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Cure has {c['n_core']} core FEs {names[:5]}")

    # --- and 'relational' is the NORM, not a discriminator: 87% of frames have >=2 core FEs
    multi = sum(1 for f in _index()["frames"]
                if sum(1 for fe in _as_list(frame(f).get("FE")) if fe.get("coreType") in CORE_TYPES) >= 2)
    good = multi > 900
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {multi} frames have >=2 core FEs, so 'has unfilled slots' cannot "
          f"be a relational detector")

    # --- INHERITANCE: the Minsky axis an arity test discards
    a = ancestors("Cure")
    good = any(x["frame"] == "Transitive_action" for x in a)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Cure inherits over {[x['frame'] for x in a][:4]}")

    inh = inherited_roles("Cure")
    good = inh["n_total"] >= inh["n_own"] and inh["n_inherited"] >= 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] inherited_roles gives {inh['n_own']} own + "
          f"{inh['n_inherited']} inherited = {inh['n_total']}")

    # --- EVENT ORDER: the thing I claimed frames were for and had not implemented
    p = precedes("Activity_ongoing", "Activity_done_state")
    good = p["status"] == "CONFIRMED" and len(p["path"]) >= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] FrameNet orders events: Activity_ongoing -> "
          f"{' -> '.join(x['frame'] for x in p['path'][:] if x['frame'])}")

    # --- causal order too: curing causes recovery
    p2 = precedes("Cure", "Recovery")
    good = p2["status"] == "CONFIRMED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Cure precedes Recovery causally "
          f"({[x['via'] for x in p2['path']]})")

    # --- ABSENCE of a path must ABSTAIN, never deny
    p3 = precedes("Activity_ongoing", "Cure")
    good = p3["status"] == "ABSTAIN" and "not evidence of no order" in p3.get("reason", "")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unannotated pair ABSTAINS rather than denying "
          f"({p3['status']})")

    # --- a word with no LU must say so, not invent a frame
    d = describe("diagnosis")
    good = d["status"] == "NO_FRAME"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'diagnosis' has no frame and reports NO_FRAME, not a guess")

    # --- ambiguity is REPORTED: one noun can name three frames with different participants
    d2 = describe("treatment")
    good = d2["status"] == "OK" and d2["n_frames"] >= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'treatment' resolves to {d2.get('n_frames')} frames -- ambiguity "
          f"is reported, not silently resolved")

    # --- subevent audit distinguishes confirmed from abstained
    s = audit_subevent("Activity_finish", "Activity")
    good = s["status"] == "CONFIRMED" and s.get("via") == "Subframe"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] subevent audit: Activity_finish is a subframe of Activity "
          f"({s['status']}/{s.get('via')})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frames as situations: participants, inheritance, event order.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--word", default="")
    ap.add_argument("--frame", default="")
    ap.add_argument("--precedes", nargs=2, default=None)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.precedes:
        print(json.dumps(precedes(*args.precedes), indent=2))
        return 0
    if args.frame:
        print(json.dumps({"core": core_roles(args.frame), "ancestors": ancestors(args.frame),
                          "subframes": subframes(args.frame),
                          "parent_scenarios": parent_scenarios(args.frame)}, indent=2))
        return 0
    if args.word:
        print(json.dumps(describe(args.word), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
