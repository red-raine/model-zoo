"""logos.lexical — WordNet + FrameNet as a deterministic semantic layer.

WHY A LEXICON AND NOT A MODEL
-----------------------------
spaCy rules are only as good as their word lists. `EntityRuler` and `DependencyMatcher` need to
know that "big" and "large" are interchangeable, that "hot" and "cold" are not, that "bank"
has two unrelated senses, and which words are too common to carry signal. Those four facts are
exactly WordNet's four relations, and they are cheap, offline, and auditable:

    synonym   -> an ADVERSARIAL pair must NOT change the verdict
    antonym   -> an ADVERSARIAL pair MUST flip the verdict
    hypernym  -> licenses taxonomic inference ("a sparrow is a bird")
    homonym   -> a trap: "bank" must be disambiguated before any relation check

That gives us a way to BUILD golden training data with derived labels instead of judged
labels, which is the only kind of synthetic data worth trusting (the verified-synthesis
pattern: construct the instance from the known answer, never ask a model for the answer).

FrameNet is the "frames" half: a predicate maps to a semantic frame with typed frame elements
(buy -> Commerce_buy with Buyer/Goods/Money). That is a Minsky-style frame with named slots,
generated from a curated lexicon rather than invented, and it is what lets the claim IR ask
"is the MONEY slot a currency?" instead of "does the sentence look right".

USAGE
-----
    python tools/logos/lexical.py --self-test
"""

from __future__ import annotations

import argparse
import functools
from typing import Literal

# Words that carry no verification signal. WordNet's own stopword list is too small for
# fact-checking: it misses the journalistic filler ("reportedly", "according", "allegedly")
# which is often the most important MODALITY marker, so those are kept in a separate list on
# purpose rather than being stripped.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without into over under about as is are was were be been being am do does did doing have
has had having will would shall should can could may might must it its it's they them
their there here where when who whom which what how why not no nor so such very too also
just only own same other some any all both each few more most much many several
""".split())

# Modality markers: NOT stopwords, deliberately retained. A claim that reports what someone
# else said is a different verification target from a claim asserting the same thing.
MODALITY = {
    "hedged": frozenset("""allegedly reportedly may might could possibly probably likely
        appears seems suggests claimed claims said says according""".split()),
    "reported": frozenset("""reportedly according said says told stated announced declared
        claimed tweeted posted wrote""".split()),
}


def _wn():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")           # force the corpus to load
        return wn
    except Exception:                 # noqa: BLE001
        return None


def _fn():
    try:
        from nltk.corpus import framenet as fn
        fn.frames()[0]               # force load
        return fn
    except Exception:                 # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def available() -> dict[str, bool]:
    return {"wordnet": _wn() is not None, "framenet": _fn() is not None}


class Lexicon:
    """Deterministic lexical relations. All lookups cached; no model, no network."""

    def __init__(self) -> None:
        self.wn = _wn()
        self.fn = _fn()
        self._cache: dict[tuple[str, str], frozenset[str]] = {}

    def _need_wn(self) -> bool:
        return self.wn is not None

    # ------------------------------------------------------------------ relations
    def synsets(self, word: str) -> list:
        if not self._need_wn():
            return []
        return self.wn.synsets(word.lower().replace(" ", "_"))

    def synonyms(self, word: str) -> frozenset[str]:
        """Lemma names sharing any synset with `word` (multi-word lemmas underscored)."""
        key = ("syn", word)
        if key in self._cache:
            return self._cache[key]
        out: set[str] = set()
        for s in self.synsets(word):
            for l in s.lemmas():
                n = l.name().replace("_", " ").lower()
                if n != word.lower():
                    out.add(n)
        res = frozenset(out)
        self._cache[key] = res
        return res

    def antonyms(self, word: str) -> frozenset[str]:
        """True lemma antonyms. WordNet stores them sparsely, on the lemma not the synset."""
        key = ("ant", word)
        if key in self._cache:
            return self._cache[key]
        out: set[str] = set()
        for s in self.synsets(word):
            for l in s.lemmas():
                for a in l.antonyms():
                    out.add(a.name().replace("_", " ").lower())
        res = frozenset(out)
        self._cache[key] = res
        return res

    def hypernyms(self, word: str, depth: int = 8) -> frozenset[str]:
        """TRANSITIVE hypernym closure. Is-a is transitive and WordNet only stores one level.

        A single-level lookup says sparrow -> oscine, and then fails the obvious question
        "is a sparrow a bird?". In a claim verifier that is not a cosmetic miss: taxonomic
        inference is one of the few relation checks we can do exactly, so the closure is the
        difference between the channel working and the channel settling nothing.
        """
        out: set[str] = set()
        frontier = list(self.synsets(word))
        seen: set[str] = set()
        for _ in range(depth):
            nxt = []
            for s in frontier:
                if s.name() in seen:
                    continue
                seen.add(s.name())
                for h in s.hypernyms():
                    for l in h.lemmas():
                        out.add(l.name().replace("_", " ").lower())
                    nxt.append(h)
            if not nxt:
                break
            frontier = nxt
        return frozenset(out)

    def hyponyms(self, word: str) -> frozenset[str]:
        out: set[str] = set()
        for s in self.synsets(word):
            for h in s.hyponyms():
                for l in h.lemmas():
                    out.add(l.name().replace("_", " ").lower())
        return frozenset(out)

    def homonym_groups(self, word: str) -> list[frozenset[str]]:
        """One group per SYNSET: because the same spelling with different POS or a distant
        meaning is a different word. 'bank' yields a financial group and a river group; any
        relation check that ignores this will happily 'prove' that a river pays interest.
        """
        groups = []
        for s in self.synsets(word):
            groups.append(frozenset(l.name().replace("_", " ").lower() for l in s.lemmas()))
        return groups

    def is_homonymous(self, word: str, min_groups: int = 2) -> bool:
        """Senses from different POS, or the same POS with unrelated hypernyms."""
        groups = self.homonym_groups(word)
        if len(groups) < min_groups:
            return False
        pos = {s.pos() for s in self.synsets(word)}
        if len(pos) > 1:
            return True
        hypers = [set(h.name() for h in s.hypernyms()) for s in self.synsets(word)]
        for i in range(len(hypers)):
            for j in range(i + 1, len(hypers)):
                if hypers[i] and hypers[j] and not (hypers[i] & hypers[j]):
                    return True
        return False

    def relation(self, a: str, b: str) -> Literal["synonym", "antonym", "hypernym",
                                                  "hyponym", "same", "unrelated"]:
        a, b = a.lower(), b.lower()
        if a == b:
            return "same"
        if b in self.synonyms(a) or a in self.synonyms(b):
            return "synonym"
        if b in self.antonyms(a) or a in self.antonyms(b):
            return "antonym"
        if b in self.hypernyms(a):
            return "hypernym"
        if a in self.hypernyms(b):
            return "hyponym"
        return "unrelated"

    def stopwords(self) -> frozenset[str]:
        return STOPWORDS

    def modality(self, token: str) -> str:
        t = token.lower()
        for name in ("reported", "hedged"):
            if t in MODALITY[name]:
                return name
        return "asserted"

    # -------------------------------------------------------------------- frames
    def frame_of(self, word: str) -> list[str]:
        """FrameNet frames triggered by a lemma. Frames ARE our typed logical forms."""
        if self.fn is None:
            return []
        out = []
        try:
            for lu in self.fn.lus(word.lower()):
                out.append(lu.frame.name)
        except Exception:                      # noqa: BLE001 - unregistered lemma
            return []
        seen, uniq = set(), []
        for f in out:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq

    def frame_elements(self, frame_name: str) -> list[dict]:
        """Typed slots of a frame: [{'name': 'Buyer', 'core': True, 'semantic_type': ...}].

        NLTK's `frame.FE` is a NAME-KEYED dict, not a list, so `fr.FE[0]` raises KeyError and
        iterating it yields names rather than objects. Both shapes are handled here because
        the accessor is the only place that should know about NLTK's container choices.
        """
        if self.fn is None:
            return []
        try:
            fr = self.fn.frame(frame_name)
            items = fr.FE.values() if hasattr(fr.FE, "values") else fr.FE
            out = []
            for fe in items:
                name = getattr(fe, "name", None)
                if name is None and hasattr(fe, "get"):
                    name = fe.get("name")
                core = False
                st = ""
                if hasattr(fe, "get"):
                    core = bool(fe.get("coreType"))
                    stv = fe.get("semanticType") or {}
                    st = stv.get("name", "") if isinstance(stv, dict) else str(stv)
                out.append({"name": str(name), "core": core, "semantic_type": st})
            return out
        except Exception:                      # noqa: BLE001 - frame shape varies by release
            return []

    @staticmethod
    def _words(text: str) -> list[str]:
        return [w.strip(".,;:!?()[]\"'").lower() for w in text.split()]

    def lexical_signature(self, text: str) -> frozenset[str]:
        """APPROXIMATE canonical bag of content words, for dedup only.

        Maps each word to its first (most frequent) WordNet sense. This is deliberately
        labelled approximate: an earlier version mapped each word to `min(synonyms)`, which
        is an arbitrary alphabetical representative, so "big" and "large" canonicalised to
        two DIFFERENT words and the synonym-invariance test failed. First-sense is the standard
        default and is good enough for near-duplicate detection; it is NOT good enough to
        decide whether two sentences assert the same thing.
        """
        bag: set[str] = set()
        for w in self._words(text):
            if not w or w in STOPWORDS:
                continue
            synsets = self.synsets(w)
            bag.add(synsets[0].name() if synsets else w)
        return frozenset(bag)

    def same_assertion(self, a: str, b: str) -> bool:
        """Aligned comparison: same length, and every aligned content-word pair is
        identical-or-synonymous. This is the property a synonym-substituted training pair
        must satisfy, and the one an antonym substitution must violate.
        """
        wa, wb = self._words(a), self._words(b)
        if len(wa) != len(wb):
            return False
        for x, y in zip(wa, wb):
            if x in STOPWORDS and y in STOPWORDS:
                continue
            rel = self.relation(x, y)
            if rel not in ("same", "synonym"):
                return False
        return True


# ------------------------------------------------------------------------------ tests
def run_self_test() -> bool:
    print("logos.lexicon - SELF TEST")
    ok = True
    print(f"  resources: wordnet={available()['wordnet']} framenet={available()['framenet']}")
    lx = Lexicon()

    good = lx.wn is not None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] WordNet loaded")
    if not good:
        print("SELF TEST: FAILURES (WordNet missing)")
        return False

    # --- synonyms: the relation that must NOT change a label
    good = "large" in lx.synonyms("big")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] synonyms(big) contains 'large' "
          f"(n={len(lx.synonyms('big'))})")

    # --- antonyms: the relation that MUST flip a label
    good = "cold" in lx.antonyms("hot")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonyms(hot) contains 'cold' "
          f"(got {sorted(lx.antonyms('hot'))[:4]})")

    good = "small" in lx.antonyms("large")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonyms(large) contains 'small'")

    # --- antisymmetry: if b is an antonym of a, a must be an antonym of b
    good = all("hot" in lx.antonyms(w) for w in ["cold"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonymy is symmetric (cold <-> hot)")

    # --- hypernym: taxonomic inference
    good = "bird" in lx.hypernyms("sparrow")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hypernyms(sparrow) contains 'bird'")

    # --- homonymy: 'bank' must be detected as ambiguous
    good = lx.is_homonymous("bank") and len(lx.homonym_groups("bank")) >= 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] bank is homonymous "
          f"({len(lx.homonym_groups('bank'))} groups)")

    # and 'hot' must NOT be flagged merely for having several senses of one POS
    groups_hot = len(lx.homonym_groups("hot"))
    print(f"  [INFO] homonym_groups(hot) = {groups_hot} (senses, not necessarily homonyms)")

    # --- the relation oracle must be correct on hand-checked pairs
    cases = [("big", "large", "synonym"), ("hot", "cold", "antonym"),
             ("sparrow", "bird", "hypernym"), ("big", "big", "same"),
             ("sparrow", "asphalt", "unrelated")]
    for a, b, want in cases:
        got = lx.relation(a, b)
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] relation({a},{b}) = {got} (want {want})")

    # --- modality markers must survive as content, not be stripped as stopwords
    good = lx.modality("reportedly") == "reported" and "reportedly" not in lx.stopwords()
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'reportedly' is a MODALITY marker, not a "
          f"stopword")

    # --- the label-invariance property that dataset building depends on
    good = lx.same_assertion("The big dog chased the cat", "The large dog chased the cat")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] synonym substitution preserves the assertion")

    good = not lx.same_assertion("The big dog chased the cat",
                                 "The small dog chased the cat")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonym substitution breaks the assertion "
          f"(this is what lets us build a flipped-label pair)")

    good = not lx.same_assertion("The dog chased the cat", "The dog chased the cat today")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] different token counts are not the same "
          f"assertion (no silent length-blind match)")

    # --- FrameNet: a frame is a typed logical form with named slots
    if lx.fn is not None:
        frames = lx.frame_of("buy")
        good = len(frames) >= 1
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] frame_of(buy) -> {frames[:2]}")
        if frames:
            fes = lx.frame_elements(frames[0])
            names = [f["name"] for f in fes]
            good = len(names) >= 2
            ok = ok and good
            print(f"  [{'PASS' if good else 'FAIL'}] frame '{frames[0]}' has typed slots "
                  f"{names[:5]}")
    else:
        print("  [SKIP] FrameNet unavailable")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WordNet + FrameNet lexical layer.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--word", default="")
    args = ap.parse_args(argv)
    if args.self_test or not args.word:
        return 0 if run_self_test() else 1
    lx = Lexicon()
    w = args.word
    print(f"{w}:")
    print("  synonyms :", sorted(lx.synonyms(w))[:12])
    print("  antonyms :", sorted(lx.antonyms(w))[:12])
    print("  hypernyms:", sorted(lx.hypernyms(w))[:8])
    print("  hyponyms :", sorted(lx.hyponyms(w))[:8])
    print("  homonymous:", lx.is_homonymous(w))
    print("  frames   :", lx.frame_of(w)[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
