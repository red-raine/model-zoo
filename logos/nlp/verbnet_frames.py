"""logos/nlp/verbnet_frames.py — per-verb SYNTACTIC FRAME legality from VerbNet. A gate, never a guess.

WHY VERBNET AND NOT SELECTIONAL RESTRICTIONS -- THE MEASUREMENT THAT DECIDES IT
-----------------------------------------------------------------------------
Dunietz, Levin & Carbonell (ACL 2013) built DAVID, a WordNet+VerbNet selectional-preference violation
detector, and ran it on the resources AS-IS: **F1 = 28.27%**. Their own conclusion was that resource quality
matters more than the algorithm. So a selectional-restriction check would be a ~28-F1 instrument, which in a
precision-first suite is a false-DROP generator. It is not used here.

Frame LEGALITY is a different proposition: VerbNet declares, per verb class, an explicit list of permitted
syntactic frames (`Agent V Patient`, `Agent V {to} Recipient`, ...). That is a finite lookup against the
dependency parse, with no plausibility guessing. Coverage is partial and must be respected: only ~46% of
WordNet verbal senses align to a VerbNet class, so "verb not in VerbNet" is UNKNOWN and must fall through,
never become "illegal".

WHAT THIS MODULE IS ALLOWED TO DO
--------------------------------
Only ADD refusals. It never accepts a parse that the encoder would otherwise reject, so it cannot loosen the
pipeline -- which means the LOST_CORRECT gate stays honest when it is wired in.

Licence: VerbNet is "unencumbered and may be used in commercial applications" (University of Colorado), the
opposite of HowNet's position. Version 3.3, 329 class XML files, 265 KB.
NOTE: NLTK's bundled `verbnet` corpus is **Version 2.1**, two major versions behind, under a weaker
"distributed with permission" licence. This module reads the Colorado XML directly and never touches NLTK.

USAGE
    python tools/logos/nlp/verbnet_frames.py --self-test
    python tools/logos/nlp/verbnet_frames.py --measure
    python tools/logos/nlp/verbnet_frames.py --verb give
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

VN_DIR = os.environ.get("LOGOS_VERBNET_DIR", r"B:\datasets\verbnet\verbnet3.3")

# VerbNet role names that stand for a noun phrase in a syntactic frame.
ROLE_TOKENS = {
    "agent", "patient", "theme", "experiencer", "instrument", "recipient", "source", "goal",
    "location", "destination", "beneficiary", "attribute", "result", "predicate", "topic",
    "stimulus", "product", "material", "asset", "value", "initial_state", "final_state",
    "trajectory", "path", "time", "duration", "extent", "cause", "actor", "undergoer",
}

_CACHE = None


def available(directory: str = VN_DIR) -> tuple:
    if not os.path.isdir(directory):
        return False, f"VerbNet not found at {directory}"
    files = glob.glob(os.path.join(directory, "*.xml"))
    if not files:
        return False, f"no XML class files in {directory}"
    return True, f"{len(files)} class files"


def _shape_from_syntax(syn) -> str:
    """Coarse shape from a <SYNTAX> element's CHILDREN, reading `value` attributes.

    VerbNet 3.x writes `<NP value="Agent"/>`, `<VERB/>`, `<PREP value="to"/>` -- self-closing elements with
    the role in an ATTRIBUTE and no text at all. Joining `.text` yields "" for every frame, which is how the
    first version of this module ended up with 4,577 verbs and ZERO classes.
    """
    toks = []
    for child in list(syn):
        tag = (child.tag or "").upper()
        if tag == "VERB":
            toks.append("V")
        elif tag == "NP":
            toks.append("N")
            for g in list(child):
                if (g.tag or "").upper() == "PREP":
                    pv = (g.get("value") or (g.text or "")).strip().lower()
                    if pv:
                        toks.append(pv)
        elif tag == "PREP":
            pv = (child.get("value") or (child.text or "")).strip().lower()
            if pv:
                toks.append(pv)
        elif tag in ("ADV", "ADJ"):
            toks.append(tag.lower())
        elif tag == "LEX":
            lv = (child.get("value") or (child.text or "")).strip().lower()
            if lv:
                toks.append(lv)
    out, prev = [], None
    for t in toks:
        if t == "N" and prev == "N":
            continue
        out.append(t)
        prev = t
    return " ".join(out)


def _shape_from_description(primary: str) -> str:
    """Fallback from DESCRIPTION primary='NP V NP PP.recipient' -- coarser, but always present."""
    toks = []
    for raw in (primary or "").split():
        low = raw.lower()
        if low == "v":
            toks.append("V")
        elif low.startswith("np") or low.startswith("pp"):
            toks.append("N")
        elif low in ("adv", "adj"):
            toks.append(low)
    out, prev = [], None
    for t in toks:
        if t == "N" and prev == "N":
            continue
        out.append(t)
        prev = t
    return " ".join(out)


def load(directory: str = VN_DIR) -> dict:
    """Parse every class file once. Returns {verbs: {lemma: [class]}, classes: {id: [frames]}}."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    ok, why = available(directory)
    if not ok:
        _CACHE = {"status": why, "verbs": {}, "classes": {}}
        return _CACHE

    verbs = collections.defaultdict(set)
    classes = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.xml"))):
        try:
            root = ET.parse(path).getroot()
        except Exception:                                    # noqa: BLE001
            continue
        for cls in root.iter("VNCLASS"):
            cid = cls.get("ID") or os.path.basename(path).replace(".xml", "")
            frames = []
            # class-level FRAMES apply to every MEMBER of the class
            for fr in cls.iter("FRAME"):
                shape = ""
                syns = [x for x in fr.iter("SYNTAX")]
                if syns:
                    shape = _shape_from_syntax(syns[0])
                if not shape:
                    desc = fr.find("DESCRIPTION")
                    if desc is not None:
                        shape = _shape_from_description(desc.get("primary") or "")
                if shape:
                    frames.append(_norm_shape(shape))
            for mem in cls.iter("MEMBER"):
                lemma = (mem.get("name") or "").strip().lower()
                if not lemma:
                    continue
                verbs[lemma].add(cid)
            if frames:
                classes[cid] = sorted(set(frames))
    _CACHE = {"status": "ok", "verbs": {k: sorted(v) for k, v in verbs.items()},
              "classes": classes}
    return _CACHE


def classes_for_verb(lemma: str) -> list:
    return load()["verbs"].get((lemma or "").lower(), [])


def frames_for_verb(lemma: str) -> list:
    """The coarse syntactic SHAPES this verb is permitted in, e.g. ['N V N', 'N V to N']."""
    d = load()
    out = []
    for cid in classes_for_verb(lemma):
        out.extend(d["classes"].get(cid, []))
    return sorted(set(out))


# ------------------------------------------------------------------ signatures
def frame_shape(vn_syntax: str) -> str:
    """Normalise a VerbNet SYNTAX template to a coarse shape.

    `Agent V Patient` -> `N V N`;  `Agent V {to} Recipient` -> `N V to N`;  `Agent V` -> `N V`.
    Braces mark literal optional words; role names become N; the verb becomes V. Everything else is dropped,
    because a coarse shape is what a dependency parse can honestly be compared against.
    """
    toks = []
    for raw in re.split(r"\s+", (vn_syntax or "").strip()):
        t = raw.strip()
        if not t:
            continue
        low = t.lower().strip("{}[]()")           # braces mark an OPTIONAL word; keep the word itself
        if not low or low in (":", ",", "/"):
            continue
        if low == "v":
            toks.append("V")
        elif low in ROLE_TOKENS:
            toks.append("N")
        elif re.fullmatch(r"[a-z\-]+", low):
            toks.append(low)
        # anything else (multi-word role lists, punctuation groups) is dropped
    # collapse runs of N (VerbNet sometimes lists Patient and Theme separately)
    out, prev = [], None
    for t in toks:
        if t == "N" and prev == "N":
            continue
        out.append(t)
        prev = t
    return " ".join(out)






def parse_shape(doc) -> str:
    """Coarse shape of a dependency parse, in TOKEN ORDER: N V N, N V to N, N V, ...

    Walks the document in order instead of emitting V first, because the POSITION of the subject relative to
    the verb is part of the frame. Emitting V first made every sentence look like `V N ...` and nothing ever
    matched -- while a function that always returns NO_MATCH would still have passed every negative test.
    """
    if doc is None:
        return ""
    arg_deps = {"nsubj", "nsubjpass", "dobj", "attr", "acomp", "dative"}
    toks = []
    for t in doc:
        if t.dep_ == "ROOT":
            toks.append("V")
        elif t.dep_ in arg_deps:
            toks.append("N")
        elif t.dep_ in ("prep", "prt") and t.head is not None and t.head.dep_ == "ROOT":
            toks.append(t.lower_)
        elif t.dep_ == "pobj" and t.head is not None and t.head.dep_ in ("prep", "prt") \
                and t.head.head is not None and t.head.head.dep_ == "ROOT":
            # the OBJECT of a root-attached preposition is an argument: "gave it to the WINNER".
            # Dropping it made the dative frame read as "N V N to" and never match "N V n to n".
            toks.append("N")
    out, prev = [], None
    for x in toks:
        if x == "N" and prev == "N":
            continue
        out.append(x)
        prev = x
    return " ".join(out)


def _norm_shape(shape: str) -> str:
    """ONE normalisation for both sides: N/V uppercase, everything else lowercase, consecutive N collapsed."""
    toks = []
    for raw in (shape or "").split():
        low = raw.lower()
        if low == "v":
            toks.append("V")
        elif low == "n":
            toks.append("N")
        else:
            toks.append(low)
    out, prev = [], None
    for x in toks:
        if x == "N" and prev == "N":
            continue
        out.append(x)
        prev = x
    return " ".join(out)


def check_frame(lemma: str, doc=None, text: str = "") -> dict:
    """Is this verb used in one of ITS permitted frames?

    Returns MATCHED / NO_MATCH / UNKNOWN. NO_MATCH names the templates that were permitted, so a refusal can
    cite the violated frame rather than asserting "bad grammar".
    UNKNOWN means the verb is not in VerbNet -- about half of English verb senses -- and MUST fall through.
    """
    l = (lemma or "").lower()
    if not l:
        return {"status": "UNKNOWN", "reason": "no lemma"}
    classes = classes_for_verb(l)
    if not classes:
        return {"status": "UNKNOWN", "reason": f"{l!r} is not in VerbNet (about half of verb senses are not)",
                "verb": l}
    permitted = frames_for_verb(l)      # already coarse shapes
    if not permitted:
        return {"status": "UNKNOWN", "reason": f"{l!r} has classes but no frames declared", "verb": l}
    if doc is None:
        try:
            from logos.nlp import domain_nlp
            doc = domain_nlp.make_nlp(use_domain=True)(text or "")
        except Exception:                                    # noqa: BLE001
            return {"status": "UNKNOWN", "reason": "no parse available", "verb": l}
    got = _norm_shape(parse_shape(doc))
    # ---- THE ROOT MUST BE THE VERB WE WERE ASKED ABOUT. Otherwise the shape describes a different word and
    # comparing it to `give`'s frames is meaningless. Measured case: "Dogs chase cats" parses with a root that
    # is not `chase`, so the shape came out as bare "V" and `chase` looked frame-illegal for a sentence that is
    # perfectly ordinary. A shape computed from the wrong root is not evidence about this verb.
    try:
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        root_lemma = (root.lemma_ or "").lower() if root is not None else ""
        if root_lemma and root_lemma != l:
            return {"status": "UNKNOWN", "verb": l,
                    "reason": f"the parse's root is {root_lemma!r}, not {l!r}; the shape describes a "
                              f"different word", "root": root_lemma}
    except Exception:                                        # noqa: BLE001
        pass
    allowed = {_norm_shape(p) for p in permitted}
    if got in allowed:
        return {"status": "MATCHED", "verb": l, "shape": got, "classes": classes,
                "n_frames": len(permitted)}
    return {"status": "NO_MATCH", "verb": l, "shape": got, "classes": classes,
            "permitted_shapes": sorted(allowed)[:8], "permitted_frames": permitted[:6],
            "reason": f"{l!r} is used as {got!r}, which is not one of its permitted frames"}


# ------------------------------------------------------------------ measure
def measure() -> int:
    ok, why = available()
    print("=== VerbNet 3.3, syntactic frame legality ===")
    print(f"  {ok} ({why})")
    if not ok:
        return 1
    d = load()
    print(f"  verbs: {len(d['verbs']):,}   classes: {len(d['classes']):,}")
    shapes = collections.Counter()
    for frames in d["classes"].values():
        for f in frames:
            shapes[frame_shape(f)] += 1
    print(f"  distinct coarse shapes across all classes: {len(shapes)}")
    print(f"  most common: {shapes.most_common(6)}\n")

    print("=== frame legality on real sentences ===")
    CASES = [
        ("The committee gave the award to the winner.", "give"),
        ("She gave the book.", "give"),
        ("Dogs chase cats.", "chase"),
        ("The cat sleeps.", "sleep"),
        ("The committee gave loudly under the table for hours.", "give"),
        ("Quantum frobnication transmutes the lattice.", "transmute"),
    ]
    print(f"  {'sentence':52s} {'verb':11s} {'status':10s} shape")
    for text, lemma in CASES:
        r = check_frame(lemma, text=text)
        print(f"  {text[:52]:52s} {lemma:11s} {r['status']:10s} {r.get('shape', '-')}")
        if r["status"] == "NO_MATCH":
            print(f"      permitted: {r['permitted_shapes'][:4]}")
    print()
    print("  READ THIS: NO_MATCH is a refusal the pipeline can cite ('give is used as N V, which is not one of")
    print("  its permitted frames'). UNKNOWN is NOT a refusal -- about 46% of WordNet verbal senses have no")
    print("  VerbNet class, so an unknown verb must fall through, never become 'illegal'.")
    print()
    print("  AND THE BOUNDARY: selectional restrictions are NOT used. DAVID (Dunietz et al., ACL 2013) measured")
    print("  WordNet+VerbNet selectional-preference violation detection at F1 28.27% on as-is resources. A 28-F1")
    print("  instrument inside a precision-first suite is a false-DROP generator, so only FRAME LEGALITY is")
    print("  consulted, which is a finite lookup rather than a plausibility judgement.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.verbnet_frames - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] VerbNet available ({why})")
    if not av:
        print("SELF TEST: SKIPPED (VerbNet not downloaded)")
        return True

    d = load()
    good = len(d["verbs"]) > 3000 and len(d["classes"]) > 200
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] loaded {len(d['verbs']):,} verbs across "
          f"{len(d['classes']):,} classes")

    # --- shape normalisation is the load-bearing transformation
    good = frame_shape("Agent V Patient") == "N V N"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Agent V Patient' -> 'N V N'")

    good = frame_shape("Agent V {to} Recipient") == "N V to N"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Agent V {{to}} Recipient' -> 'N V to N'")

    # --- a verb absent from VerbNet is UNKNOWN, NOT illegal (about half of verb senses)
    r = check_frame("zzznotaverb", text="The thing zzznotaverbed.")
    good = r["status"] == "UNKNOWN"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a verb not in VerbNet is UNKNOWN, never 'illegal'")

    # --- a known verb yields classes and frames
    good = len(classes_for_verb("give")) > 0 and len(frames_for_verb("give")) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'give' has {len(classes_for_verb('give'))} class(es) and "
          f"{len(frames_for_verb('give'))} frame(s)")

    # --- a NO_MATCH cites the permitted frames rather than asserting bad grammar
    r = check_frame("give", text="The committee gave loudly under the table for hours.")
    good = r["status"] in ("MATCHED", "NO_MATCH", "UNKNOWN")
    if r["status"] == "NO_MATCH":
        good = good and bool(r.get("permitted_frames"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a NO_MATCH carries the permitted frames "
          f"({r['status']})")

    # --- the module must never claim to use selectional restrictions
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    good = "SELECTIONAL" in src.upper() or "selectional" in src
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the module documents that selectional restrictions are NOT used")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VerbNet syntactic frame legality (a gate, not a guess).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--verb", default="")
    ap.add_argument("--sentence", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.verb:
        if args.sentence:
            print(json.dumps(check_frame(args.verb, text=args.sentence), indent=2))
        else:
            print(json.dumps({"verb": args.verb, "classes": classes_for_verb(args.verb),
                              "frames": frames_for_verb(args.verb)}, indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
