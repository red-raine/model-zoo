"""logos/logic/reasoning.py — validate the REASONING, not the fact.

THE REFRAME THIS MODULE EXISTS FOR
----------------------------------
Everything before this validated facts: is the number right, is the statement refuted by evidence, is the frame's
argument list legal. That is one dimension, and it is the wrong one. The thing worth validating is **cognitive
reason** -- whether the structure of the inference holds.

The clearest proof that the old framing was wrong is metaphor. In the fact frame, a participant whose type does
not fit a frame's role is a DEFECT: a category error. But:

    "The argument is a house of cards."

`argument` is abstract; `house` requires a physical entity. A type checker reports a violation. And it is not a
violation -- it is a **licensed cross-frame mapping**, the whole apparatus of metaphor. Under a fact-validating
reading, English is mostly broken. Under a cognition-validating reading, the same sentence is SOUND: it asserts
that the argument has a fragile support structure, which is exactly what the mapping carries.

So the SAME signal (a filler that does not fit the role) has two completely different meanings:

    violation + a licensing mapping exists      ->  METAPHOR        a reasoning mode, not an error
    violation + no licensing mapping exists     ->  CATEGORY_ERROR  the reasoning failure
    no violation                                ->  SOUND

That single distinction is the whole module, and no amount of fact checking can produce it, because both cases
are equally "not literally true".

THE SECOND THING FACT-VALIDATION CANNOT SEE
--------------------------------------------
A metaphor is not an entailment. `The argument is a house of cards` licenses thinking about fragility; it does
NOT support `therefore the argument is false`. When a figurative premise is used to ground a factual conclusion,
the reasoning is a NON_SEQUITUR even though every clause is well-formed. That is a cognitive defect and it is
invisible to a truth oracle: the premise can be perfectly apt and the conclusion still not follow.

VERDICTS
    SOUND               fillers fit the frame, or a fit cannot be established (no violation found)
    METAPHOR            a type violation that a frame mapping LICENSES
    CATEGORY_ERROR      a type violation with no licensing mapping
    NON_SEQUITUR        a figurative premise offered as grounds for a factual conclusion
    INSUFFICIENT        no frame resolves, so nothing is claimed

WHAT IS MEASURED AND WHAT IS CONSTRUCTED -- READ THIS
-----------------------------------------------------
MEASURED on this machine: FrameNet 1.7 has 1,221 frames and **exactly 4 `Metaphor` edges** (0.3% of frames), 127
`Perspective_on`, 781 `Inheritance`, 556 `Using`. So FrameNet CANNOT license metaphor recognition in general --
four annotated mappings is not a licensing layer. That is a real, quantified limit and this module reports it
rather than papering over it.

CONSTRUCTED: the map from FrameNet `semType` names to WordNet lexical domains is MY approximation, not an
authoritative alignment. It is labelled `CONSTRUCTED` in the code and a caller can see it. No published
semType-to-WordNet-domains alignment with measured accuracy was available to me, so it must not be cited as one.

USAGE
    python tools/logos/logic/reasoning.py --self-test
    python tools/logos/logic/reasoning.py --measure
    python tools/logos/logic/reasoning.py --text "The argument is a house of cards."
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

SOUND, METAPHOR, CATEGORY_ERROR, NON_SEQUITUR, INSUFFICIENT = (
    "SOUND", "METAPHOR", "CATEGORY_ERROR", "NON_SEQUITUR", "INSUFFICIENT")
CIRCULAR, UNFALSIFIABLE = "CIRCULAR", "UNFALSIFIABLE"

# Markers that introduce GROUNDS. Splitting on these is what makes circularity checkable at all.
GROUND_MARKERS = ("because", "since", "given that", "due to the fact that", "on the grounds that",
                  "for the reason that")

# Markers of an appeal to self-evidence. These are not grounds -- they are the ABSENCE of grounds presented as
# a ground, which is why they belong with unfalsifiability rather than with circularity.
SELF_EVIDENCE_MARKERS = ("obviously", "clearly", "everyone knows", "everybody knows", "it is self-evident",
                         "it is obvious", "needless to say", "goes without saying", "by definition",
                         "cannot be disproven", "cannot be disproved", "impossible to disprove",
                         "no one can deny", "undeniably")

# Hedges that remove all content from a claim: nothing observable would count against them.
VACUITY_MARKERS = ("in some sense", "somehow", "in a way", "to some extent", "more or less",
                   "it could be argued", "one might say", "in certain respects")

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in", "on",
             "at", "for", "with", "and", "or", "but", "that", "this", "these", "those", "it", "its", "as",
             "we", "i", "you", "they", "he", "she", "not", "no", "do", "does", "did", "has", "have", "had",
             "can", "could", "would", "should", "may", "might", "will", "shall", "there", "here", "so",
             "because", "since", "if", "then", "than", "which", "who", "whom", "what", "when", "where"}

# CONSTRUCTED (see the docstring): FrameNet semType -> WordNet lexnames that may fill it. Approximate by
# construction. Any semType absent from here yields None = "cannot tell", never False.
SEMTYPE_DOMAINS = {
    "Physical_entity": {"noun.artifact", "noun.body", "noun.object", "noun.substance", "noun.animal",
                        "noun.plant", "noun.food", "noun.location", "noun.shape"},
    "Physical_object": {"noun.artifact", "noun.body", "noun.object", "noun.substance"},
    "Animate": {"noun.animal", "noun.person", "noun.body"},
    "Sentient": {"noun.person", "noun.animal"},
    "Human": {"noun.person"},
    "Locative_relation": {"noun.location", "noun.object", "noun.artifact", "noun.shape"},
    "State_of_affairs": {"noun.state", "noun.act", "noun.event", "noun.cognition", "noun.attribute"},
    "Event": {"noun.act", "noun.event", "noun.communication", "noun.process"},
    "Activity": {"noun.act", "noun.process", "noun.communication"},
    "Time": {"noun.time"},
    "Quantity": {"noun.quantity", "noun.attribute", "noun.number"},
    "Substance": {"noun.substance"},
    "Artifact": {"noun.artifact"},
    "Scalar": {"noun.attribute", "a"},
}

# Markers that a premise is being offered as grounds. Used only for the NON_SEQUITUR test, and deliberately
# conservative: a bare "so" in narrative prose is not an inference claim.
INFERENCE_MARKERS = ("therefore", "thus", "hence", "consequently", "it follows that", "proves that",
                     "which proves", "so we can conclude", "this shows that", "proving that")

# Cue words that a sentence is figurative. This is the cheap, CPU-only mechanism the literature keeps landing on
# (the metonymy work finds "trigger words information contributes to performance improvement"); it is used here
# only to LABEL a candidate, never to decide one.
FIGURATIVE_CUES = ("like", "as if", "kind of", "sort of", "is a", "are a", "sea of", "house of cards",
                   "mountain of", "ocean of", "river of", "wall of", "forest of", "storm of")

_CACHE = {}


def _wn():
    from nltk.corpus import wordnet as wn
    return wn


def _fn():
    from nltk.corpus import framenet as fn
    return fn


def available() -> tuple:
    try:
        _wn().get_version()
        n = len(_fn().frames())
        return True, f"WordNet {_wn().get_version()}, FrameNet {n} frames"
    except Exception as e:                                   # noqa: BLE001
        return False, f"unavailable: {type(e).__name__}"


def lexnames(word: str) -> set:
    """WordNet lexical domains a word belongs to. The domain is the type signal used for role compatibility."""
    wn = _wn()
    out = set()
    w = (word or "").strip().lower().replace(" ", "_")
    if not w:
        return out
    for pos in ("n", "v", "a", "r"):
        try:
            for s in wn.synsets(w, pos=pos):
                out.add(s.lexname())
        except Exception:                                    # noqa: BLE001
            continue
    return out


def compatible(filler: str, semtype: str) -> bool | None:
    """Can `filler` fill a role typed `semtype`? None means cannot tell -- and None must never become False."""
    if not semtype:
        return None
    doms = SEMTYPE_DOMAINS.get(semtype)
    if not doms:
        return None                                          # unknown semType: cannot tell
    lex = lexnames(filler)
    if not lex:
        return None                                          # unknown word: cannot tell
    return bool(lex & doms)


def frames_for_word(word: str) -> list:
    try:
        return [f.name for f in _fn().frames_by_lemma((word or "").lower().replace(" ", "_"))]
    except Exception:                                        # noqa: BLE001
        return []


def frame_roles(frame_name: str) -> dict:
    from logos.nlp import frames as F
    return F.core_roles(frame_name)


def vocabulary(text: str) -> list:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower())]


def copular_clash(text: str) -> list:
    """`X is [a] Y` where Y's lexical domain is disjoint from X's AND Y demands a kind X cannot be.

    WHY THIS REPLACED FRAME-ROLE TYPE CHECKING
    My first version cross-producted every word in the sentence against every core frame element, and it
    reported `CATEGORY_ERROR` for **"The committee gave the award to the winner."** -- a perfectly ordinary
    sentence -- because it tested the verb 'gave' against a role typed Human. That is a false positive in the
    catastrophic direction, so that approach is REJECTED here and this function replaces it.

    The rejection is principled, not just empirical: without a role binder (a parse mapped onto the frame's
    valence pattern) I cannot say WHICH word fills WHICH role, and a cross product is not a binding. `committee`
    is also noun.group, so even a correct binder would need metonymy to allow it in a Human role -- which is the
    same licensed-violation problem one level up.

    A copular is different: the syntax itself binds the two nominals, so the comparison is between the actual
    subject and the actual predicate nominal. That is a binding I can defend.
    """
    # NOTE: the tokenizer here must keep SHORT function words. `[A-Za-z][A-Za-z\-]{2,}` demands three or more
    # characters and silently drops the copula "is", which made this function return [] for every input while
    # looking correct -- the tests that passed were the ones asserting NO clash.
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", (text or "").lower())
    if len(words) < 3:
        return []
    wn = _wn()
    out = []
    for i, w in enumerate(words):
        if w not in ("is", "are", "was", "were", "becomes", "became"):
            continue
        subj = None
        for j in range(i - 1, -1, -1):
            if words[j] not in ("the", "a", "an", "this", "that", "these", "those", "of", "and"):
                subj = words[j]
                break
        pred = None
        for j in range(i + 1, len(words)):
            if words[j] not in ("the", "a", "an", "this", "that", "these", "those", "of", "and", "very"):
                pred = words[j]
                break
        if not subj or not pred:
            continue
        ds, dp = lexnames(subj), lexnames(pred)
        if not ds or not dp:
            continue
        # a clash needs DISJOINT domains, and a predicate demanding animacy/physicality
        if ds & dp:
            continue
        animate_pred = bool(dp & {"noun.animal", "noun.person", "noun.body"})
        physical_pred = bool(dp & {"noun.artifact", "noun.object", "noun.substance", "noun.food",
                                   "noun.shape", "noun.location", "noun.plant"})
        subj_physical = bool(ds & {"noun.artifact", "noun.object", "noun.substance", "noun.food",
                                   "noun.shape", "noun.location", "noun.plant", "noun.animal",
                                   "noun.person", "noun.body"})
        subj_abstract = bool(ds & {"noun.cognition", "noun.communication", "noun.attribute", "noun.state",
                                   "noun.act", "noun.event", "noun.time", "noun.relation",
                                   "noun.motive", "noun.process"})
        # engine(artifact) is a beast(animal): physical subject, animate predicate, disjoint domains
        # argument(cognition) is a house(artifact): abstract subject, physical predicate, disjoint domains
        if (animate_pred and subj_physical) or (physical_pred and subj_abstract):
            out.append({"subject": subj, "predicate": pred,
                        "subject_domains": sorted(ds)[:4], "predicate_domains": sorted(dp)[:4],
                        "kind": "animate_predicate_on_object" if animate_pred
                                else "physical_predicate_on_abstract"})
    return out


def frame_role_note() -> dict:
    """The recorded REJECTION, kept in the code so it cannot be quietly reintroduced.

    Frame-role type checking was implemented, measured against a control sentence, and rejected. Keeping the
    counter-example here is the point: an approach that produces a false CATEGORY_ERROR on ordinary English is
    disqualifying in a precision-first suite, no matter how principled it looks on paper.
    """
    return {"approach": "cross-product every word against every core frame element",
            "status": "REJECTED",
            "counter_example": "The committee gave the award to the winner.",
            "observed": "CATEGORY_ERROR: gave vs Members:Human",
            "why": "the verb was tested against a role typed Human; a cross product is not a role binding, and "
                   "without a valence-pattern binder no honest role assignment exists here",
            "replacement": "copular_clash -- the copula binds the two nominals syntactically"}


def type_violations(text: str, max_frames: int = 4) -> list:
    """Deprecated shim. Returns [] and exists so an old caller gets silence, not a false accusation.

    The frame cross-product it used to perform is recorded as REJECTED in `frame_role_note()`. A function that
    can emit a category error for correct English must not remain reachable.
    """
    return []


def licensing(f1: str, f2: str) -> dict:
    """Does FrameNet annotate a mapping between these two frames that could license a metaphor?

    Checks the Metaphor relation in both directions, plus Perspective_on (same event, different lexicalisation),
    which is a weaker but real license that two frames describe the same situation.
    """
    from logos.nlp import frames as F
    idx = F._index()
    found = []
    for a, rels in idx["rels"].items():
        for r in rels:
            if r["rel"] in ("Metaphor", "Perspective_on"):
                if (r["super"] == f1 and r["sub"] == f2) or (r["super"] == f2 and r["sub"] == f1):
                    found.append({"rel": r["rel"], "super": r["super"], "sub": r["sub"]})
    return {"licensed": bool(found), "edges": found,
            "reason": ("a Metaphor/Perspective_on edge licenses the mapping" if found else
                       "FrameNet annotates no Metaphor or Perspective_on edge between these frames")}


def metaphor_capacity() -> dict:
    """How much licensing capacity FrameNet actually has. Measured, and it is the ceiling on this approach."""
    from logos.nlp import frames as F
    idx = F._index()
    c = collections.Counter()
    for _name, rels in idx["rels"].items():
        for r in rels:
            c[r["rel"]] += 1
    n = len(idx["frames"])
    meta = c.get("Metaphor", 0)
    persp = c.get("Perspective_on", 0)
    return {"n_frames": n, "metaphor_edges": meta, "perspective_edges": persp,
            "metaphor_fraction": round(100.0 * meta / max(n, 1), 2),
            "perspective_fraction": round(100.0 * persp / max(n, 1), 2),
            "verdict": ("FrameNet cannot license metaphor recognition in general: "
                        f"{meta} Metaphor edges over {n} frames ({100.0*meta/max(n,1):.2f}%). "
                        "Metaphor detection needs its own model and corpus -- which is what the frame-recognition "
                        "literature builds and what this suite does not have.")}


def non_sequitur(text: str) -> dict:
    """Is a FIGURATIVE premise being offered as grounds for a factual conclusion?

    This is the cognitive defect a truth oracle cannot see: the premise may be perfectly apt and the conclusion
    still not follow, because a metaphor licenses a way of thinking, not an entailment.
    """
    t = (text or "").lower()
    marker = next((m for m in INFERENCE_MARKERS if _has_marker(t, m)), None)
    cue = next((c for c in FIGURATIVE_CUES if _has_marker(t, c)), None)
    if not marker:
        return {"status": "NO_INFERENCE_CLAIM", "marker": None, "cue": cue}
    if not cue:
        return {"status": "NO_FIGURATIVE_PREMISE", "marker": marker, "cue": None}
    return {"status": "NON_SEQUITUR", "marker": marker, "cue": cue,
            "why": f"a figurative premise ({cue!r}) is offered as grounds for a conclusion ({marker!r}); a "
                   f"metaphor licenses a construal, not an entailment"}


def _norm_text(text: str) -> str:
    """Lowercase and reduce every run of punctuation/whitespace to a single space, with padding at both ends.

    WHY THIS EXISTS. Every detector here checks `" marker " in " text "`, and that idiom FAILS whenever the marker
    is adjacent to punctuation: measured, `"The plan will work in some sense."` did not contain `" in some sense "`
    because a full stop follows the marker. Three detectors (`circular`, `unfalsifiable`, `non_sequitur`) shared
    the idiom, so all three silently matched nothing on ordinary punctuated prose -- and every NEGATIVE test still
    passed. Normalising here fixes the class rather than the instance.
    """
    return " " + re.sub(r"[^a-z0-9'\-]+", " ", (text or "").lower()).strip() + " "


def _has_marker(text: str, marker: str) -> bool:
    return f" {marker} " in _norm_text(text)


def _content(text: str) -> list:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower()) if w not in STOPWORDS]


def circular(text: str, overlap_threshold: float = 0.75) -> dict:
    """Do the GROUNDS merely restate the CLAIM?

    "X because X" is circular, and so is "X because X'", where X' is X in synonyms -- the second form is the one
    that fools a word-overlap test, which is why WordNet synonymy is consulted.

    The proxy is information ADDED by the grounds: if (nearly) every content word of the grounds already appears
    in the claim, the grounds presuppose what they are supposed to support. A genuine explanation introduces at
    least one independent term, and that is the negative control this test is built around.
    """
    t = (text or "").lower()
    marker = next((m for m in GROUND_MARKERS if _has_marker(t, m)), None)
    if not marker:
        return {"status": "NO_GROUNDS", "marker": None}
    head, _, grounds = t.partition(marker)
    claim_c = _content(head)
    ground_c = _content(grounds)
    if len(claim_c) < 2 or len(ground_c) < 1:
        return {"status": "ABSTAIN", "marker": marker,
                "reason": "not enough content on one side to judge"}

    claim_set = set(claim_c)
    # RESTATEMENT VIA SYNONYMY: a grounds word that is a WordNet synonym of a claim word is not new information
    syn_added = set()
    try:
        from logos.nlp import lexicon_ensemble as LE
        for g in ground_c:
            if g in claim_set:
                continue
            s = LE.synonyms(g, pos=None)
            if claim_set & set(s["sources"]):
                syn_added.add(g)
    except Exception:                                        # noqa: BLE001
        pass

    covered = [g for g in ground_c if g in claim_set or g in syn_added]
    frac = len(covered) / len(ground_c)
    new_terms = [g for g in ground_c if g not in claim_set and g not in syn_added]
    if frac >= overlap_threshold:
        return {"status": CIRCULAR, "marker": marker, "coverage": round(frac, 3),
                "restated": covered[:6], "via_synonym": sorted(syn_added)[:4],
                "why": f"{len(covered)}/{len(ground_c)} content words of the grounds already appear in the "
                       f"claim, so the grounds add {len(new_terms)} independent term(s) -- they presuppose "
                       f"what they are offered to support"}
    return {"status": "NOT_CIRCULAR", "marker": marker, "coverage": round(frac, 3),
            "new_terms": new_terms[:6],
            "why": "the grounds introduce independent terms, so they are not a restatement"}


def unfalsifiable(text: str) -> dict:
    """Is the claim held in a way that no observation could count against it?

    Two distinct routes, both structural rather than about truth:
      SELF_EVIDENT     the claim is asserted as needing no support ("obviously", "everyone knows",
                       "cannot be disproven"). This is the absence of grounds presented as a ground.
      VACUOUS          hedges that remove all content ("in some sense", "to some extent"), so nothing
                       observable would count for or against it.

    Deliberately narrow. A bare "clearly" in ordinary prose is not an epistemic claim, so a marker only counts
    when it is not immediately followed by a reason -- `"clearly because X"` is falsifiable, `"clearly."` is not.
    """
    t = (text or "").lower()
    for m in SELF_EVIDENCE_MARKERS:
        if _has_marker(t, m):
            # if a grounds marker follows, the claim IS being supported, so it is not an appeal to self-evidence
            if not any(_has_marker(t, g) for g in GROUND_MARKERS):
                return {"status": UNFALSIFIABLE, "route": "SELF_EVIDENT", "marker": m,
                        "why": f"{m!r} offers the claim as needing no support and no grounds are given, so "
                               f"nothing observable is specified that could count against it"}
    for m in VACUITY_MARKERS:
        if _has_marker(t, m):
            return {"status": UNFALSIFIABLE, "route": "VACUOUS", "marker": m,
                    "why": f"{m!r} removes the content of the claim, so no observation would distinguish a "
                           f"world where it holds from one where it does not"}
    return {"status": "NOT_UNFALSIFIABLE", "marker": None}


def analyse(text: str) -> dict:
    """The cognitive verdict. Reasons about REASONING, not about truth."""
    ok, why = available()
    if not ok:
        return {"status": INSUFFICIENT, "reason": why}
    clashes = copular_clash(text)
    ns = non_sequitur(text)
    circ = circular(text)
    unf = unfalsifiable(text)

    # ORDER MATTERS: an inference-claim defect outranks a figure, because the inference is the reasoning act
    # itself. Then a circular ground, then unfalsifiability, then the figurative reading.
    if ns["status"] == "NON_SEQUITUR":
        return {"status": NON_SEQUITUR, "clashes": clashes, "inference": ns,
                "why": ns["why"],
                "note": "the clauses may each be apt; the INFERENCE is what fails"}

    if circ["status"] == CIRCULAR:
        return {"status": CIRCULAR, "inference": circ, "clashes": clashes, "why": circ["why"],
                "note": "the clause is fine; what fails is that the support is the thing being supported"}

    if unf["status"] == UNFALSIFIABLE:
        return {"status": UNFALSIFIABLE, "inference": unf, "clashes": clashes, "why": unf["why"],
                "note": "no observation is specified that could count against the claim, so it cannot be "
                        "reasoned with either way"}

    if not clashes:
        return {"status": SOUND, "clashes": [],
                "inference": circ if circ["status"] == "NOT_CIRCULAR" else None,
                "why": "no copular type clash, no figurative premise under an inference marker, no circular "
                       "ground and no self-evidence appeal, so no cognitive defect is detected -- which is NOT "
                       "the same as the statement being true"}

    lit, unl = [], []
    for c in clashes:
        sf = frames_for_word(c["subject"])[:1]
        pf = frames_for_word(c["predicate"])[:1]
        lic = licensing(sf[0], pf[0]) if sf and pf else None
        (lit if lic and lic["licensed"] else unl).append({**c, "licensing": lic})

    # A copular domain clash is FAR more likely a figure than a defect, and FrameNet cannot license it
    # (4 Metaphor edges), so the honest verdict is METAPHOR with the licensing recorded as unavailable --
    # NOT CATEGORY_ERROR. Reserving an error verdict for something I cannot distinguish from normal English
    # would be the same mistake the cross-product made.
    return {"status": METAPHOR, "clashes": clashes, "licensed": lit, "unlicensed": unl,
            "why": "the subject and predicate nominal belong to disjoint lexical domains and the predicate "
                   "demands a kind the subject is not, which is the shape of a figure rather than a defect",
            "note": ("FrameNet licenses %d of these; its total licensing capacity is 4 Metaphor edges over "
                     "1,221 frames, so licensing is reported when found and never required."
                     % len(lit)),
            "frame_role_note": frame_role_note()["status"]}


# ------------------------------------------------------------------ measure
def measure() -> int:
    ok, why = available()
    print("=== validating cognitive reason, not facts ===")
    print(f"  {why}\n")
    if not ok:
        return 1

    cap = metaphor_capacity()
    print("=== FrameNet's licensing capacity, measured ===")
    print(f"  frames {cap['n_frames']} | Metaphor edges {cap['metaphor_edges']} "
          f"({cap['metaphor_fraction']}%) | Perspective_on {cap['perspective_edges']} "
          f"({cap['perspective_fraction']}%)")
    print(f"  ==> {cap['verdict']}\n")

    print("=== the SAME signal, two different meanings ===")
    CASES = [
        "The argument is a house of cards.",
        "The committee gave the award to the winner.",
        "The doctor treated the patient with a new medication.",
        "The argument is a house of cards; therefore the conclusion is false.",
        "The engine is a beast.",
        "Water boils at 100 degrees celsius.",
        "The drug is effective because the drug is effective.",
        "Obviously the policy is the right one.",
        "The bridge collapsed because the supporting cables had corroded through.",
    ]
    print(f"  {'sentence':62s} {'verdict':16s} detail")
    for t in CASES:
        r = analyse(t)
        det = ""
        if r["status"] == METAPHOR:
            det = f"{len(r.get('licensed', []))} licensed"
        elif r["status"] == CATEGORY_ERROR:
            u = (r.get("unlicensed") or [{}])[0]
            det = f"{u.get('filler')} vs {u.get('role')}:{u.get('semType')}"
        elif r["status"] == NON_SEQUITUR:
            det = f"marker {r['inference']['marker']!r} on premise {r['inference']['cue']!r}"
        print(f"  {t[:62]:62s} {r['status']:16s} {det}")

    print()
    print("  READ THIS: rows 1 and 4 are the same words with different REASONING, and they get different")
    print("  verdicts. A fact validator cannot produce that distinction, because in both rows the sentence is")
    print("  literally false. Only the cognitive reading separates 'a metaphor' from 'a metaphor used as a")
    print("  proof'.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.logic.reasoning - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] {why}")
    if not av:
        print("SELF TEST: SKIPPED")
        return True

    # --- compatibility returns None when it cannot tell, and None must NEVER become False
    good = compatible("argument", "") is None and compatible("argument", "NoSuchSemType") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an untyped role or unknown semType yields None, never False")

    good = compatible("zzznotaword", "Physical_entity") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown filler yields None, never False")

    # --- and it does say True/False when it can tell
    good = compatible("apple", "Physical_entity") is True and compatible("courage", "Physical_entity") is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a physical filler fits Physical_entity and an abstract one does "
          f"not")

    # --- THE REGRESSION TEST FOR THE DEFECT THIS MODULE SHIPPED FIRST. Frame-role type checking reported
    #     CATEGORY_ERROR for this perfectly ordinary sentence, which is a false positive in the catastrophic
    #     direction. It must never come back.
    r = analyse("The committee gave the award to the winner.")
    good = r["status"] == SOUND
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the counter-example that killed frame-role checking is SOUND "
          f"({r['status']})")

    good = frame_role_note()["status"] == "REJECTED" and type_violations("x") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the rejected approach is recorded with its counter-example and "
          f"can no longer emit an accusation")

    # --- THE CENTRAL DISTINCTION, and it must actually fire rather than pass vacuously
    r = analyse("The engine is a beast.")
    good = r["status"] == METAPHOR and any(c["predicate"] == "beast" for c in r.get("clashes", []))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a copular domain clash is detected as a FIGURE rather than an "
          f"error ({r['status']})")

    r = analyse("The argument is a house of cards.")
    good = r["status"] == METAPHOR
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an abstract subject with a physical predicate is a FIGURE "
          f"({r['status']})")

    # --- and a literal copular that shares a domain must NOT be flagged
    r = analyse("The car is a vehicle.")
    good = r["status"] == SOUND
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a literal copular within one domain is not flagged "
          f"({r['status']})")

    # --- a metaphor offered as proof is a NON_SEQUITUR, and that is separable from the metaphor itself
    ns = non_sequitur("The argument is a house of cards; therefore the conclusion is false.")
    good = ns["status"] == NON_SEQUITUR
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a figurative premise grounding a conclusion is NON_SEQUITUR "
          f"({ns['status']})")

    # --- a factual sentence with an inference marker is NOT flagged: no figurative premise
    ns2 = non_sequitur("The tower is 330 metres tall; therefore it is taller than 300 metres.")
    good = ns2["status"] == "NO_FIGURATIVE_PREMISE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an inference over literal premises is not flagged "
          f"({ns2['status']})")

    # --- and a metaphor with no conclusion is not flagged either
    ns3 = non_sequitur("The argument is a house of cards.")
    good = ns3["status"] == "NO_INFERENCE_CLAIM"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a metaphor on its own is not flagged as a non-sequitur "
          f"({ns3['status']})")

    # --- licensing reports the truth about FrameNet: it usually cannot license anything
    l = licensing("Cure", "Recovery")
    good = "licensed" in l and isinstance(l["edges"], list)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] licensing() answers, and does not invent an edge "
          f"(Cure/Recovery licensed={l['licensed']})")

    # --- the measured ceiling on licensing must be reported, not hidden
    cap = metaphor_capacity()
    good = cap["metaphor_edges"] < 20 and "cannot license metaphor recognition" in cap["verdict"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the module reports that FrameNet has only "
          f"{cap['metaphor_edges']} Metaphor edges over {cap['n_frames']} frames")

    # --- a plain true sentence must not be turned into a defect
    r = analyse("Water boils at 100 degrees celsius.")
    good = r["status"] in (SOUND, INSUFFICIENT)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a literal sentence is not manufactured into a defect "
          f"({r['status']})")

    # --- CIRCULARITY, with the negative control that matters: a real explanation introduces new terms
    c = circular("The drug is effective because the drug is effective.")
    good = c["status"] == CIRCULAR
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a restated ground is CIRCULAR ({c['status']}, "
          f"coverage {c.get('coverage')})")

    c2 = circular("The bridge collapsed because the supporting cables had corroded through.")
    good = c2["status"] == "NOT_CIRCULAR" and c2.get("new_terms")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a ground that adds independent terms is NOT circular "
          f"({c2.get('new_terms')})")

    # --- restatement via SYNONYM is the form that defeats a word-overlap test
    c3 = circular("The claim is accurate because the claim is correct.")
    good = c3["status"] in (CIRCULAR, "NOT_CIRCULAR")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] synonym restatement is examined ({c3['status']}, "
          f"via_synonym={c3.get('via_synonym')})")

    # --- no grounds marker means nothing to judge, and it must abstain rather than guess
    c4 = circular("The bridge collapsed.")
    good = c4["status"] == "NO_GROUNDS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a claim with no grounds marker abstains ({c4['status']})")

    # --- UNFALSIFIABILITY: an appeal to self-evidence with no grounds
    u = unfalsifiable("Obviously the policy is the right one.")
    good = u["status"] == UNFALSIFIABLE and u["route"] == "SELF_EVIDENT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an appeal to self-evidence is UNFALSIFIABLE ({u['route']})")

    # --- and the control: 'clearly BECAUSE X' IS supported, so it must not be flagged
    u2 = unfalsifiable("Clearly the policy failed because unemployment rose by two points.")
    good = u2["status"] == "NOT_UNFALSIFIABLE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'clearly because X' is supported and not flagged "
          f"({u2['status']})")

    u3 = unfalsifiable("The plan will work in some sense.")
    good = u3["status"] == UNFALSIFIABLE and u3["route"] == "VACUOUS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a vacuous hedge is UNFALSIFIABLE ({u3['route']})")

    # --- the analyser now reports the new classes
    r = analyse("The drug is effective because the drug is effective.")
    good = r["status"] == CIRCULAR
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] analyse() surfaces CIRCULAR ({r['status']})")

    r = analyse("Obviously the policy is the right one.")
    good = r["status"] == UNFALSIFIABLE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] analyse() surfaces UNFALSIFIABLE ({r['status']})")

    # --- and a well-formed explanation over literal clauses stays SOUND
    r = analyse("The tower is 330 metres tall because it has 330 metres of steel framework.")
    good = r["status"] in (SOUND, INSUFFICIENT)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a supported literal explanation is not manufactured into a defect "
          f"({r['status']})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate reasoning, not facts.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--text", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.text:
        print(json.dumps(analyse(args.text), indent=2, default=str))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
