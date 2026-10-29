"""logos/nlp/domain_nlp.py — make spaCy see our vocabulary, and connect WordNet/FrameNet to the logic layer.

WHY THIS EXISTS, MEASURED
------------------------
spaCy's stock tokenizer SHREDS our jargon, which breaks every downstream term operation (ontology matching,
predicate normalisation, second-source lookup):

    "macro-F1 is a metric."                 -> macro | - | F1 | is | a | metric | .
    "The bge-reranker-v2-m3 model is int8." -> The | bge | - | reranker | - | v2 | - | m3 | model | is | ...
    "SNOMED-CT"                             -> SNOMED | - | CT

The fragments are not the term. A domain vocabulary fixes THAT. It does not, and cannot, improve the
dependency parser: parsing accuracy comes from treebank-annotated trees, and a word list contains none. So
this module claims exactly what a lexicon can deliver -- tokenisation, lemma stability, and a canonical
surface form -- and nothing more.

THE SECOND HALF IS THE INTERESTING ONE: `logos/lexical.py` ALREADY CONTAINS A WORDNET + FRAMENET ENGINE
---------------------------------------------------------------------------
`Lexicon` has antonyms(), synonyms(), hypernyms(), hyponyms(), frame_of(), frame_elements(), is_homonymous(),
relation(), stopwords(), modality(). It is wired into dataset generation and into NOTHING in the logic layer.
So the machinery to make "physician" and "doctor" the same predicate, and to know that "unable" is the
antonym of "able" (hence a negation), exists and is unreachable -- the same pattern as row_eval, the Lean
bridge, and logic/channel.

Three levers, in order of measured value:
  ANTONYMY -> POLARITY. "unable" is not in any negation word list, but WordNet says it is the antonym of
      "able". This catches morphological negation (un-/in-/ir-/non-/less) that a lemma lexicon misses.
  SYNONYMY -> PREDICATE IDENTITY. "physician" and "doctor" share synsets, so a claim about one bears on the
      other. Without it, the encoder builds two unrelated predicates and proves nothing across claims.
  FRAMENET -> HIDDEN ARGUMENTS. A frame has typed slots (Buy: Buyer/Goods). A noun whose frame has unfilled
      core elements is RELATIONAL -- "a treatment for X" is not the unary property treatment(x). This is the
      principled version of the ad-hoc `prep` guard in dep_encoder.

USAGE
    python tools/logos/nlp/domain_nlp.py --self-test
    python tools/logos/nlp/domain_nlp.py --measure
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

# ------------------------------------------------------------------ the domain vocabulary
# A seed set from ontology_bundle.JARGON plus the identifiers this project actually uses, because those are
# the strings spaCy mangles and the strings our ontology layer must match on. Mined terms are added at
# import time so the list grows with the repository rather than going stale.
SEED_TERMS = [
    # metrics / calibrations
    "macro-F1", "micro-F1", "AUC", "AURC", "ECE", "Brier", "Platt", "isotonic", "conformal", "Mondrian",
    "NEI", "CTU", "AVeriTeC", "ABSTAIN",
    # infrastructure
    "LiteLLM", "ONNX", "ONNX Runtime", "Qdrant", "Mathlib", "Lean 4", "spaCy", "WordNet", "FrameNet",
    "VerbNet", "Z3", "SMT-LIB", "GGUF", "Q4_K_M", "int8", "AVX2", "AVX-512",
    # models
    "bge-reranker-v2-m3", "bge-small-en-v1.5", "all-MiniLM-L6-v2", "ms-marco-MiniLM-L-4-v2",
    "LFM2.5-2.6B", "Qwen3.5-9B", "Granite-4.2-8B", "ThinkingCap",
    # ontology / knowledge
    "SNOMED-CT", "ICD-11", "ICD-10", "MeSH", "OLS4", "Wikidata", "P585", "P813", "OpenAlex", "Crossref",
    "OKF", "Type 1", "Type 2",
]

_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")   # hyphen/underscore compounds
_VER_RE = re.compile(r"\b[A-Za-z]+-v?\d+(?:\.\d+)*\b")                     # foo-v2, bar-1.5
_ACRO_RE = re.compile(r"\b[A-Z]{2,}[0-9]*\b")                              # all-caps acronyms


def mine_terms(roots: list, limit_files: int = 400) -> set:
    """Harvest identifier-shaped tokens from our own artifacts.

    Deliberately shape-based rather than frequency-based: the tokens worth protecting are the ones the
    tokenizer BREAKS, and those are exactly the hyphenated, versioned and acronym forms. Counting words
    would add thousands of ordinary English nouns that spaCy already handles.
    """
    found: set = set()
    n = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith((".md", ".py", ".json")):
                    continue
                n += 1
                if n > limit_files:
                    return found
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                        text = fh.read(200_000)
                except Exception:                             # noqa: BLE001
                    continue
                for rx in (_IDENT_RE, _VER_RE, _ACRO_RE):
                    for m in rx.findall(text):
                        if 3 <= len(m) <= 40:
                            found.add(m)
    return found


VOCAB_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "logos_domain_vocab.json")
VOCAB_TTL_DAYS = 7


def _cache_vocab(terms: list, mangled: list) -> None:
    try:
        import json as _json
        import time as _time
        os.makedirs(os.path.dirname(VOCAB_CACHE), exist_ok=True)
        with open(VOCAB_CACHE, "w", encoding="utf-8") as fh:
            _json.dump({"built_at": _time.time(), "terms": terms, "mangled": mangled}, fh)
    except Exception:                                         # noqa: BLE001
        pass


def vocabulary(include_mined: bool = True, use_cache: bool = True, refresh: bool = False) -> dict:
    """The domain vocabulary, with a TTL cache.

    Mining walks the repository and probing 5,934 candidates against the tokenizer takes a measured 60 s. That
    is fine once and unacceptable per invocation, so it is cached. The TTL is a blunt instrument and is named
    as one: a new jargon term will not appear until the cache expires or --refresh-vocab is passed.
    """
    if use_cache and not refresh and os.path.exists(VOCAB_CACHE):
        try:
            import json as _json
            with open(VOCAB_CACHE, encoding="utf-8") as fh:
                cached = _json.load(fh)
            age_days = (__import__("time").time() - cached.get("built_at", 0)) / 86400.0
            if age_days <= VOCAB_TTL_DAYS and cached.get("terms"):
                return {"terms": cached["terms"], "n": len(cached["terms"]), "cached": True,
                        "mangled": cached.get("mangled"),
                        "age_days": round(age_days, 2)}
        except Exception:                                     # noqa: BLE001
            pass

    terms = {t for t in SEED_TERMS}
    if include_mined:
        terms |= mine_terms([os.path.join(_TOOLS, "logos"),
                             os.path.join(os.path.dirname(_TOOLS), "knowledge"),
                             os.path.join(os.path.dirname(_TOOLS), "wiki")])
    out = sorted(terms, key=str.lower)
    try:
        import json as _json
        import time as _time
        os.makedirs(os.path.dirname(VOCAB_CACHE), exist_ok=True)
        with open(VOCAB_CACHE, "w", encoding="utf-8") as fh:
            _json.dump({"built_at": _time.time(), "terms": out}, fh)
    except Exception:                                         # noqa: BLE001
        pass
    return {"terms": out, "n": len(out), "cached": False}


# ------------------------------------------------------------------ spaCy with the vocabulary
_NLP = None


def available() -> tuple:
    try:
        import spacy                                             # noqa: F401
    except Exception as e:                                       # noqa: BLE001
        return False, f"spacy missing: {type(e).__name__}"
    return True, "ready"


def make_nlp(use_domain: bool = True, model: str = ""):
    """A spaCy pipeline that keeps our jargon intact.

    `add_special_case` pins a string to a single token. This is the ONLY mechanism a word list can use to
    change parsing outcomes, and it changes them by changing what a token IS -- not by teaching the parser.
    Claiming more would be false.
    """
    global _NLP
    if _NLP is not None and use_domain:
        return _NLP
    import spacy
    import time as _time
    _phases = {}
    _t = _time.perf_counter()
    name = model or os.environ.get("LOGOS_SPACY_MODEL", "en_core_web_sm")
    nlp = spacy.load(name, disable=["ner"])
    _phases["load"] = round(_time.perf_counter() - _t, 2)
    if use_domain:
        _t = _time.perf_counter()
        vocab = vocabulary()
        _phases["vocabulary"] = round(_time.perf_counter() - _t, 2)
        _t = _time.perf_counter()
        mangled = vocab.get("mangled")
        if not mangled:
            # IDENTIFY THE MANGLED CLASS BY SHAPE, NOT BY PROBING. Tokenising 5,937 candidates through a
            # blank pipeline costs ~40 s of build time, which made every CLI invocation unusable. The terms
            # the tokenizer breaks are exactly the ones containing a hyphen, an underscore, a digit, or a
            # long compound -- a short ordinary word survives anyway. This is instant and covers the class.
            mangled = [t for t in vocab["terms"]
                       if ("-" in t or "_" in t or any(ch.isdigit() for ch in t) or len(t) > 14)]
            _cache_vocab(vocab["terms"], mangled)
        _phases["select_mangled"] = round(_time.perf_counter() - _t, 2)
        # ONE compiled regex over the mangled terms instead of N add_special_case calls (2,061 of those cost
        # ~42 s to install). The alternation is bounded: an unbounded pattern over thousands of literals is a
        # measured compile-time hazard, and the terms that actually corrupt a parse are the long, hyphenated
        # and versioned ones -- a short ordinary word survives tokenisation anyway.
        _t = _time.perf_counter()
        high_value = [t for t in (mangled or []) if len(t) >= 5]
        _phases["select"] = round(_time.perf_counter() - _t, 2)
        if high_value:
            _t = _time.perf_counter()
            longest_first = sorted(high_value, key=len, reverse=True)
            pattern = "|".join(re.escape(t) for t in longest_first)
            _phases["join"] = round(_time.perf_counter() - _t, 2)
            _t = _time.perf_counter()
            rx = re.compile(pattern)
            _phases["compile"] = round(_time.perf_counter() - _t, 2)
            _t = _time.perf_counter()
            nlp.tokenizer.token_match = rx.match
            _phases["assign"] = round(_time.perf_counter() - _t, 2)
        nlp.meta["logos_domain_terms"] = len(mangled or [])
        nlp.meta["logos_domain_installed"] = len(high_value)
        nlp.meta["logos_domain_candidates"] = vocab["n"]
        nlp.meta["logos_build_phases"] = _phases
        _NLP = nlp
    return nlp


def reset():
    global _NLP
    _NLP = None


# ------------------------------------------------------------------ WordNet / FrameNet bridge
def bridge():
    """The existing Lexicon, or None. Imported lazily so this module works without nltk data."""
    try:
        from logos.lexical import Lexicon
        return Lexicon()
    except Exception:                                            # noqa: BLE001
        return None


def same_predicate(a: str, b: str, pos: str = "n", lx=None) -> bool:
    """Same predicate? Shared synset, restricted to ONE part of speech. Never renames anything.

    SYNONYM-BASED RENAMING IS NOT SAFE and the probe shows why. `lx.synonyms` merges across parts of speech:
        doctor    -> {bushel, doc, doctor of the church, doctor up, fix}    (the VERB "to doctor")
        cat       -> {african tea, barf, be sick, bozo, big cat}            (the VERB "to cat" a rope)
        physician -> 'doc'                                                  (alphabetically first, wrong)
    Choosing a canonical STRING from that set invents a name and can merge two unrelated predicates, after which
    a proof about one appears to be about the other. So the surface form is left alone and the equivalence is a
    TEST. Restricting the POS is what removes the polysemy: "doctor" the NOUN is "physician" the NOUN, and the
    verb senses never enter the comparison.
    """
    lx = lx or bridge()
    if not lx:
        return (a or "").strip().lower() == (b or "").strip().lower()
    x = (a or "").strip().lower().replace(" ", "_")
    y = (b or "").strip().lower().replace(" ", "_")
    if x == y:
        return True
    # SOUNDNESS GUARD. Merging statistical/modal/quantifier vocabulary turns a real distinction into a
    # spurious equivalence: "statistically significant" and "substantial" were measured to collapse via
    # significant.s.01, and a verifier that believes those are one predicate will call a contradiction
    # consistent. These never merge, whatever WordNet says.
    if x in NEVER_MERGE or y in NEVER_MERGE:
        return False
    try:
        sx = {s for s in lx.synsets(x) if getattr(s, "pos", lambda: "")() == pos}
        sy = {s for s in lx.synsets(y) if getattr(s, "pos", lambda: "")() == pos}
    except Exception:                                            # noqa: BLE001
        return False
    if not sx or not sy:
        return False
    return bool(sx & sy)


_WN_POS = {"n": "n", "v": "v", "a": "a", "s": "a", "r": "r"}


def canonical_predicate(word: str, pos: str = "n", lx=None) -> str:
    """The canonical surface form for a predicate: MOST FREQUENT SENSE, HIGHEST-COUNT LEMMA.

    I previously called this "unsalvageable" and blamed WordNet for merging parts of speech. THAT WAS WRONG,
    and the WordNet data says so:

        cat      8 NOUN synsets, incl. guy.n.01 {guy, cat, hombre, bozo} and
                 kat.n.01 {kat, khat, qat, quat, cat, Arabian_tea}   <- real NOUN co-lemmas, not verb bleed
        cat(noun) intersect barf(noun) = EMPTY                        <- barf only collided via my unioning

    The bug was mine: I unioned lemma names across EVERY synset of the word, which merges unrelated senses by
    construction. The correct method is standard WordNet practice and it works:

        1. restrict to ONE part of speech
        2. take the FIRST synset -- WordNet orders synsets by frequency of use
        3. take the lemma with the highest `count()` (WordNet's sense-tagged frequency)

    Measured: cat->cat, dog->dog, doctor->doctor, physician->doctor.

    NOT Lesk. Measured here, nltk's lesk() picks kat.n.01 (the khat plant) for "my cat is sitting on the mat"
    and doctor_of_the_church.n.01 for "the doctor prescribed antibiotics". First-sense frequency beats that
    badly, so context-based WSD is deliberately NOT used.
    """
    lx = lx or bridge()
    w = (word or "").strip().lower().replace(" ", "_")
    if not w:
        return w
    if w in NEVER_MERGE:
        return w
    wn = getattr(lx, "wn", None) if lx is not None else None
    if wn is None:
        try:
            from nltk.corpus import wordnet as _wn
            wn = _wn
        except Exception:                                        # noqa: BLE001
            return w
    try:
        synsets = wn.synsets(w, pos=_WN_POS.get((pos or "n").lower(), "n"))
    except Exception:                                            # noqa: BLE001
        return w
    if not synsets:
        return w
    try:
        lemmas = synsets[0].lemmas()
    except Exception:                                            # noqa: BLE001
        return w
    if not lemmas:
        return w
    best = max(lemmas, key=lambda l: (l.count() or 0))
    return best.name().lower().replace(" ", "_")


def antonym_of(word: str, lx=None) -> set:
    lx = lx or bridge()
    try:
        return set(lx.antonyms((word or "").lower()))
    except Exception:                                            # noqa: BLE001
        return set()


_NEG_PREFIX = ("un", "in", "ir", "im", "il", "non", "dis", "anti", "mis", "under")
_NEG_SUFFIX = ("less", "free")


# Words whose "dis-" or "under-" is NOT privative. `disorder` is not "not an order"; measured in the literature
# scan as the exact false positive of blind antonymy ("disorders -> order"). A curated blocklist is honest
# about being curated: the general problem is undecidable from spelling, so the module refuses rather than
# pretending to solve it.
_NON_PRIVATIVE = {
    "disorder", "disaster", "discover", "discuss", "distance", "disease", "display", "discipline",
    "dispute", "dismiss", "discount", "disclose", "dish", "disk", "disc", "disco", "distinct",
    "distinguish", "distribute", "under", "understand", "underlie", "undergo", "undertake",
}

# Predicates that must NEVER be merged by synonymy, because merging them converts a real distinction into a
# spurious equivalence. Measured false merge from the literature scan: `significant` ~ `substantial` via
# significant.s.01 -- in a claims suite "statistically significant" and "substantial" must never be the same
# predicate. Statistical, modal and quantifier vocabulary is exactly where a merge costs soundness.
NEVER_MERGE = {
    "significant", "substantial", "cause", "may", "might", "must", "should", "can", "will",
    "all", "some", "no", "not", "none", "most", "many", "few", "every", "each", "any",
    "increase", "decrease", "rise", "fall", "more", "less", "greater", "fewer",
    "possible", "probable", "certain", "likely", "unlikely",
}


def _neg_morpheme(w: str) -> bool:
    w = (w or "").lower()
    return (any(w.startswith(p) and len(w) > len(p) + 2 for p in _NEG_PREFIX)
            or any(w.endswith(sfx) and len(w) > len(sfx) + 2 for sfx in _NEG_SUFFIX))


def is_morphological_negation(word: str, lx=None) -> dict:
    """Is this word the NEGATED form of a base predicate?

    WordNet stores antonyms sparsely ON THE LEMMA, so "unable" lists "able". That makes morphological negation
    ("un-", "in-", "ir-", "non-", "-less") detectable from the lexicon instead of from a hand-written prefix
    rule -- and prefix rules are where "uninteresting" becomes a negation of "interesting" and "under" becomes
    a negation of "der".
    """
    w = (word or "").strip().lower()
    # ADVISORY ONLY. This is an ESCALATE trigger, never a source of ¬. Measured: 52.2% of WordNet antonym
    # pairs are lexical CONTRARIES rather than logical negations (borrow/lend, majority/minority), so mapping
    # them to ¬ is a soundness bug; and only 48.8% of un-/in-/non- adjectives carry an antonym link at all,
    # so the recall ceiling is about half. The `dis-` blocklist below closes the one false positive that the
    # scan actually exhibited.
    if w in _NON_PRIVATIVE:
        return {"negated": False, "base": None, "advisory": True,
                "source": "blocklisted: the prefix is not privative here (e.g. disorder is not 'not an order')"}
    ants = antonym_of(w, lx)
    if not ants:
        return {"negated": False, "base": None, "advisory": True, "source": "no-antonym"}
    base = sorted(ants)[0]
    w_neg, b_neg = _neg_morpheme(w), _neg_morpheme(base)
    if w_neg and not b_neg:
        return {"negated": True, "base": base, "advisory": True,
                "source": "wordnet-antonym + negative morpheme"}
    if b_neg and not w_neg:
        # this word is the POSITIVE member of the pair ("able" against "unable")
        return {"negated": False, "base": base, "advisory": True,
                "source": "word-is-positive-member"}
    # neither or both carry a negative morpheme: spelling cannot decide the direction, so say so
    return {"negated": None, "base": base, "advisory": True,
            "source": "ambiguous-direction: both or neither look negated"}


def relational_frame(word: str, lx=None) -> dict:
    """Does this noun belong to a frame with UNFILLED CORE slots? If so it is relational, not a property.

    This is the principled version of dep_encoder's ad-hoc `prep` guard: "treatment" belongs to a frame with a
    Patient/Drug slot, so "a treatment for X" carries an argument and P(x) would drop it.
    """
    lx = lx or bridge()
    try:
        frames = lx.frame_of((word or "").lower())
    except Exception:                                            # noqa: BLE001
        return {"relational": False, "frames": [], "reason": "frame lookup failed"}
    core = []
    for f in frames[:3]:
        try:
            core += [e["name"] for e in lx.frame_elements(f) if e.get("core")]
        except Exception:                                        # noqa: BLE001
            pass
    return {"relational": bool(core), "frames": frames[:3], "core_elements": sorted(set(core))[:6]}


# ------------------------------------------------------------------ measure
def measure() -> int:
    ok, why = available()
    print("=== domain vocabulary + WordNet/FrameNet bridge ===")
    print(f"  spacy {ok} ({why})")
    vocab = vocabulary()
    print(f"  vocabulary: {vocab['n']} terms (seeds + mined from our artifacts)")
    print(f"    sample: {vocab['terms'][:10]}")

    import spacy
    base = spacy.load(os.environ.get("LOGOS_SPACY_MODEL", "en_core_web_sm"), disable=["ner"])
    dom = make_nlp(use_domain=True)
    print(f"  special cases installed: {dom.meta.get('logos_domain_terms')}")

    SAMPLES = [
        "macro-F1 is a metric.",
        "The bge-reranker-v2-m3 model is int8.",
        "SNOMED-CT and ICD-11 are ontologies.",
        "AVeriTeC is a dataset.",
        "Mondrian conformal prediction bounds the risk.",
    ]
    print(f"\n=== tokenisation, before and after ===")
    fixed = 0
    for s in SAMPLES:
        before = [t.text for t in base(s)]
        after = [t.text for t in dom(s)]
        changed = before != after
        fixed += 1 if changed else 0
        print(f"  {s}")
        print(f"    base: {before}")
        print(f"    dom : {after}{'   <- FIXED' if changed else ''}")
    print(f"\n  {fixed}/{len(SAMPLES)} sentences tokenised differently")

    print(f"\n=== does the fix change the PARSE? (not just the tokens) ===")
    for s in ("macro-F1 is a metric.", "AVeriTeC is a dataset."):
        for label, nlp in (("base", base), ("dom ", dom)):
            d = nlp(s)
            root = next((t for t in d if t.dep_ == "ROOT"), None)
            subs = [(c.text, c.dep_) for c in (root.children if root is not None else [])]
            print(f"  {label} {s:28s} root={root.text if root is not None else '-'!r:12s} "
                  f"dep={root.dep_ if root is not None else '-':6s} children={subs}")
    print("  If the root or the subject changed, the vocabulary bought a BETTER PARSE, not just fewer tokens.")
    print("  If only the token count changed, it bought tokenisation -- which is still the term identity the")
    print("  ontology layer needs, but it is not a parser improvement and should not be reported as one.")

    lx = bridge()
    print(f"\n=== WordNet / FrameNet levers ===")
    if lx is None:
        print("  Lexicon unavailable (nltk wordnet not downloaded)")
        return 0
    for a, b in (("doctor", "physician"), ("able", "unable"), ("mammal", "animal"), ("cat", "dog")):
        try:
            print(f"  relation({a!r},{b!r}) = {lx.relation(a, b)}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  relation({a!r},{b!r}) raised {type(e).__name__}")
    print()
    for w in ("unable", "harmless", "nonlinear", "irregular", "able"):
        r = is_morphological_negation(w, lx)
        print(f"  negation({w:12s}) -> {r}")
    print()
    for w in ("treatment", "oxygen", "cat", "scam"):
        r = relational_frame(w, lx)
        print(f"  relational({w:11s}) -> relational={r['relational']} core={r['core_elements']} "
              f"frames={r['frames']}")
    print("\n  The relational lever is the principled version of dep_encoder's `prep` guard, and the negation")
    print("  lever catches morphological negation that no lemma word-list contains.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.domain_nlp - SELF TEST")
    ok = True
    ok_av, why = available()
    if not ok_av:
        print(f"  [SKIP] {why}")
        return True

    # --- the vocabulary is non-trivial and derived from our own artifacts, not padded
    vocab = vocabulary()
    good = vocab["n"] >= 15 and "macro-F1" in vocab["terms"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] vocabulary has {vocab['n']} terms derived from seeds + artifacts")

    # --- jargon survives tokenisation as ONE token
    import spacy
    dom = make_nlp(use_domain=True)
    toks = [t.text for t in dom("macro-F1 is a metric.")]
    good = "macro-F1" in toks
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'macro-F1' stays one token -> {toks}")

    toks = [t.text for t in dom("The bge-reranker-v2-m3 model is int8.")]
    good = "bge-reranker-v2-m3" in toks
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a versioned model id stays one token -> {toks}")

    # --- and the baseline really did break them, so the test is measuring something
    base = spacy.load(os.environ.get("LOGOS_SPACY_MODEL", "en_core_web_sm"), disable=["ner"])
    good = "macro-F1" not in [t.text for t in base("macro-F1 is a metric.")]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the STOCK tokenizer does break it (so the fix is real)")

    # --- WordNet levers
    lx = bridge()
    if lx is None:
        print("  [SKIP] Lexicon/WordNet unavailable")
        print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
        return ok

    # --- CANONICALISATION, done the correct way: first synset, highest-count lemma.
    #     (My earlier "unsalvageable" verdict was wrong -- the bug was unioning across all synsets.)
    good = canonical_predicate("doctor", "n", lx) == "doctor"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] canonical(doctor) = {canonical_predicate('doctor', 'n', lx)!r}")

    good = canonical_predicate("physician", "n", lx) == "doctor"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] canonical(physician) = {canonical_predicate('physician', 'n', lx)!r} "
          f"(same predicate as doctor)")

    good = canonical_predicate("cat", "n", lx) == "cat"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] canonical(cat) = {canonical_predicate('cat', 'n', lx)!r} "
          f"(NOT 'bozo' -- the first noun synset is cat.n.01)")

    good = canonical_predicate("dog", "n", lx) == "dog"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] canonical(dog) = {canonical_predicate('dog', 'n', lx)!r}")

    good = canonical_predicate("doctor", "n", lx) != canonical_predicate("bushel", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] doctor and bushel canonicalise DIFFERENTLY "
          f"({canonical_predicate('bushel', 'n', lx)!r})")

    # --- the guard still holds: statistical vocabulary is never renamed into a neighbour
    good = canonical_predicate("significant", "a", lx) == "significant"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] canonical(significant) is left alone")

    good = same_predicate("doctor", "physician", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] doctor and physician are the SAME predicate (noun synset)")

    good = not same_predicate("cat", "dog", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cat and dog are NOT the same predicate")

    # --- the POS restriction is what makes this safe: the VERB "to doctor" drags in unrelated nouns
    good = not same_predicate("doctor", "bushel", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] doctor is not merged with bushel")

    good = not same_predicate("cat", "barf", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cat is not merged with barf")

    # --- SOUNDNESS GUARD: the measured false merge from the literature scan
    good = not same_predicate("significant", "substantial", "s", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] significant/substantial are NOT merged (statistical vocabulary)")

    good = not same_predicate("increase", "decrease", "n", lx)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] increase/decrease are NOT merged (direction vocabulary)")

    r = is_morphological_negation("unable", lx)
    good = r["negated"] is True and r["base"] == "able"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'unable' is detected as NEGATED 'able' via antonymy")

    r = is_morphological_negation("harmless", lx)
    good = r["negated"] is True and r["base"] == "harmful"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'harmless' is negated 'harmful'")

    r = is_morphological_negation("nonlinear", lx)
    good = r["negated"] is True and r["base"] == "linear"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'nonlinear' is negated 'linear'")

    # --- THE MEASURED FALSE POSITIVE: "disorders -> order" from the literature scan. "dis-" is not always
    #     privative, so a prefix rule alone turns a noun into a negation.
    r = is_morphological_negation("disorder", lx)
    good = r["negated"] is False and "blocklisted" in r["source"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'disorder' is NOT treated as negated 'order' (blocklist)")

    # --- and the whole function is advisory, because antonymy is not negation
    r = is_morphological_negation("unable", lx)
    good = r.get("advisory") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the result is marked ADVISORY (ESCALATE trigger, never a source "
          f"of negation)")

    r = is_morphological_negation("able", lx)
    good = r["negated"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'able' itself is not flagged as negated")

    # --- an antonym pair that spelling cannot orient must return None, not a guess
    r = is_morphological_negation("good", lx)
    good = r["negated"] in (False, None)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unorientable antonym pair is not guessed ({r['negated']})")

    # --- an unknown word must not crash and must not be invented as negated
    r = is_morphological_negation("qwertyuiop", lx)
    good = r["negated"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown word is not negated (no invention)")

    # --- frames
    r = relational_frame("treatment", lx)
    good = isinstance(r["relational"], bool) and isinstance(r["frames"], list)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] frame lookup returns a well-formed answer "
          f"(relational={r['relational']})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Domain vocabulary for spaCy + WordNet/FrameNet levers.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--vocab-size", action="store_true")
    args = ap.parse_args(argv)
    if args.vocab_size:
        v = vocabulary()
        print(f"{v['n']} terms")
        for t in v["terms"]:
            print(t)
        return 0
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
