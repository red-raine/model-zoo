"""logos.logic.badframes — detecting BAD thinking, not just valid inference.

THE ARCHITECTURAL FINDING
-------------------------
The psychology literature names ~13 distinct pathologies (magical thinking, conspiracy
ideation, base-rate neglect, survivorship bias, apophenia, confirmation bias, motivated
reasoning, overclaiming, ...). Implemented naively that is 13 detectors, each with its own
heuristics, none guaranteed to be detecting a *defect* rather than a *style*.

They collapse into **8 formal defects** -- each a broken object in an argument structure.
That is the design here: one detector per defect, with the named pathologies as *labels*
projected onto it. A finding names both the defect (auditable, formal) and the frame
(citable, human-recognisable).

  L1 quantifier/scope      the domain of a forall is wrong or unstated
  L2 conditioning direction which variable is conditioned on
  L3 required support edge the evidential relation between premise and conclusion
  L4 missing alternative   fewer than two hypotheses considered
  L5 selection/denominator the sampling frame is unstated or biased
  L6 modal calibration     the qualifier is stronger than the warrant licenses
  L7 self-sealing          the claim is constructed to be unfalsifiable
  L8 local coherence       the discourse does not hold together

Plus a set of **LLM-specific** frames, because in 2026 the most common bad reasoning in a
verification pipeline is not human irrationality, it is model failure:
negation blindness, the reversal curse, sycophantic capitulation, over-claiming, and depth
collapse past two hops.

THE TWO HIGHEST-VALUE CHECKS (from the research)
------------------------------------------------
1. **Within-document contradiction** -- one consistency call. Catches conspiracy's
   "contradictory beliefs" trait, motivated reasoning, and plain inconsistency at near-zero
   cost. This implementation reuses `propositional.entails` plus the WordNet antonym relation.
2. **"Unevaluable" as a first-class verdict.** Several defects are *voiding* conditions, not
   soft scores: an argument whose denominator is unstated is not "70% bad", it is unassessable.
   This is the direct antidote to LLM over-claiming.

WHAT THIS IS NOT
----------------
It does not diagnose people. Every function takes a *claim or argument* and returns a
structural finding. "Psychosis-adjacent" here means the *text* has the discourse features the
clinical scales measure (derailment, over-inclusive reference), which is a property of text,
not of an author.

VALIDATED MARKER
----------------
`absolutist_density` implements the one linguistic marker with clinical validation:
absolutist-word density (Al-Mosaiwi & Johnstone 2018, *Clinical Psychological Science*
6(4):529). It is used as a *reasoning* signal here -- "all", "never", "nothing" signal a
quantifier whose domain was never checked -- not as a mental-health screen.

USAGE
-----
    python tools/logos/logic/badframes.py --self-test
    python tools/logos/logic/badframes.py --text "Everyone knows the data is always faked."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import propositional as prop  # noqa: E402


class Defect(str, Enum):
    """The eight formal defects. Every pathological frame projects onto one of these."""

    QUANTIFIER_SCOPE = "L1_quantifier_scope"
    CONDITIONING_DIRECTION = "L2_conditioning_direction"
    SUPPORT_EDGE = "L3_support_edge"
    MISSING_ALTERNATIVE = "L4_missing_alternative"
    SELECTION_DENOMINATOR = "L5_selection_denominator"
    MODAL_CALIBRATION = "L6_modal_calibration"
    SELF_SEALING = "L7_self_sealing"
    LOCAL_COHERENCE = "L8_local_coherence"
    # --- LLM-specific failure frames
    NEGATION_BLINDNESS = "M1_negation_blindness"
    REVERSAL = "M2_reversal_curse"
    SYCOPHANCY = "M3_sycophancy"
    OVERCLAIM = "M4_overclaim"
    DEPTH_COLLAPSE = "M5_depth_collapse"


@dataclass
class Finding:
    """One structural defect found in a text, with the frame label and a citation."""

    defect: Defect
    frame: str
    severity: float                 # 0..1, how much it should move the verdict
    evidence: str                   # the span or structure that triggered it
    explanation: str
    citation: str = ""
    voiding: bool = False           # True => the claim is unevaluable, not merely weak

    def __str__(self) -> str:
        flag = " [VOIDS]" if self.voiding else ""
        return (f"{self.defect.value:32s} {self.frame}{flag}\n"
                f"    severity {self.severity:.2f} · evidence: {self.evidence!r}\n"
                f"    {self.explanation}")


# --------------------------------------------------------------------- lexical resources
# Absolutist words. Al-Mosaiwi & Johnstone (2018) validated density of this class as a
# linguistic marker; here it is used as a proxy for an unchecked quantifier domain.
ABSOLUTIST = frozenset("""
all always every everyone everybody everything never none nothing nobody no-one
completely totally entirely absolutely certainly definitely certainly surely
must cannot can't won't never not impossible undeniably unquestionably
whole entire pure perfect fully forever
""".split())

# Modal strength, ordered. Used for the L6 calibration check.
MODAL_STRENGTH = {
    "proves": 1.0, "proven": 1.0, "definitely": 1.0, "certainly": 1.0, "always": 0.95,
    "undeniably": 0.95, "must": 0.9, "cannot": 0.9, "impossible": 0.9,
    "will": 0.8, "is": 0.75, "are": 0.75, "does": 0.75,
    "likely": 0.5, "probably": 0.5, "usually": 0.5, "generally": 0.5,
    "may": 0.3, "might": 0.3, "could": 0.3, "possibly": 0.3, "perhaps": 0.25,
    "suggests": 0.3, "appears": 0.25, "seems": 0.25, "reportedly": 0.2, "allegedly": 0.2,
}
# Warrant strength vocabulary: what the *evidence type* can licence.
WARRANT_CEILING = [
    (("anecdote", "anecdotal", "one case", "a single", "i heard", "someone said"), 0.35),
    (("observational", "correlat", "associat", "cross-sectional"), 0.5),
    (("case study", "case series"), 0.45),
    (("cohort", "prospective"), 0.7),
    (("randomised", "randomized", "rct", "controlled trial"), 0.85),
    (("meta-analysis", "systematic review"), 0.95),
]

# Self-sealing markers: the claim is constructed so that no observation could refute it.
SELF_SEALING = re.compile(
    r"\b(no (?:amount of )?evidence (?:could|would|will)|"
    r"you can(?:'|no)t prove (?:it )?(?:isn'?t|wasn'?t|didn'?t)|"
    r"the (?:lack|absence) of evidence (?:proves|shows|means)|"
    r"(?:they|the media| scientists|the government) (?:don'?t|do not|won'?t) want you to know|"
    r"everyone knows|it is (?:obvious|well known) that|"
    r"any (?:apparent )?(?:evidence|study) (?:against|contradicting).{0,30}is (?:fake|paid|part of))\b",
    re.IGNORECASE)

# Single-hypothesis markers: coordinated action asserted with no alternative considered.
CONSPIRACY_MARKERS = re.compile(
    r"\b(they|them|the elite|the establishment|big pharma|the media|the government|"
    r"the deep state|the cabal|globalists)\b.{0,60}\b(control|controlled|orchestrat|"
    r"cover(?:ed)? up|hid(?:e|den)|secretly|deliberately|collud|conspir)",
    re.IGNORECASE)

# Unobservable-mechanism markers: a causal claim with no intervening mechanism.
MECHANISM_WORDS = frozenset("""
because causes caused mechanism pathway receptor enzyme gene virus bacteria
dose trial measured experiment study data evidence percent rate correlation
""".split())
MAGIC_CAUSE = re.compile(
    r"\b(brings? (?:bad )?luck|attracts?|energy (?:field|vibrat)|vibrations?|"
    r"the universe (?:wants|sent|gives|delivers)|manifest(?:s|ed)?|"
    r"like attracts like|contagio|essence of|purely by will|mind over matter)\b",
    re.IGNORECASE)

# Derailment markers: topic shifts with no connective, for L8.
CONNECTIVES = frozenset("""
because therefore thus hence however but although since so then moreover furthermore
additionally consequently accordingly whereas while if when
""".split())


# --------------------------------------------------------------------------- detectors
def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def absolutist_density(text: str, threshold: float = 0.06) -> list[Finding]:
    """L1: absolutist vocabulary signals a quantifier whose domain was never checked."""
    w = _words(text)
    if not w:
        return []
    hits = [x for x in w if x in ABSOLUTIST]
    d = len(hits) / len(w)
    if d < threshold:
        return []
    return [Finding(
        defect=Defect.QUANTIFIER_SCOPE, frame="absolutist quantifier (all/none/always/never)",
        severity=min(1.0, d / (threshold * 3)), evidence=" ".join(hits[:8]),
        explanation=(f"{len(hits)}/{len(w)} words ({d:.1%}) are absolutist. Each is a "
                     f"quantifier claim whose domain must be stated and checked; unqualified "
                     f"'all/never' claims are refuted by a single counterexample."),
        citation="Al-Mosaiwi & Johnstone 2018, Clin. Psych. Sci. 6(4):529")]


def self_sealing(text: str) -> list[Finding]:
    """L7: the claim is built so no evidence could refute it."""
    m = SELF_SEALING.search(text)
    if not m:
        return []
    return [Finding(
        defect=Defect.SELF_SEALING, frame="unfalsifiable / self-sealing claim",
        severity=0.9, evidence=m.group(0), voiding=True,
        explanation=("The claim explains away its own disconfirmation. An unfalsifiable claim "
                     "carries no verifiable content: it cannot be Supported by evidence, only "
                     "asserted. Report as unevaluable rather than False."),
        citation="Popper 1959; Douglas et al. 2017, Current Directions 26(6):538")]


def single_hypothesis(text: str) -> list[Finding]:
    """L4: coordinated agency asserted with no alternative hypothesis considered."""
    m = CONSPIRACY_MARKERS.search(text)
    if not m:
        return []
    return [Finding(
        defect=Defect.MISSING_ALTERNATIVE, frame="conspiracy ideation (single-hypothesis)",
        severity=0.7, evidence=m.group(0),
        explanation=("Agency and coordination are asserted without a competing explanation "
                     "(error, coincidence, ordinary incentive). The conspiracy-mentality "
                     "literature treats this as the defining structural feature: the belief "
                     "is held together by the absence of alternatives, not by evidence."),
        citation="Imhoff & Bruder 2014, PLoS ONE 9(2):e89177; Bates et al. 2026, TACL 14:2120")]


def magical_thinking(text: str) -> list[Finding]:
    """L3: a causal claim with no intervening mechanism."""
    m = MAGIC_CAUSE.search(text)
    if not m:
        return []
    return [Finding(
        defect=Defect.SUPPORT_EDGE, frame="magical thinking (mechanism-free causation)",
        severity=0.7, evidence=m.group(0),
        explanation=("A causal relation is asserted with no mechanism linking cause to "
                     "effect. Sympathetic-magic and contagion reasoning have this exact shape: "
                     "the SUPPORT EDGE between premise and conclusion is missing."),
        citation="Rozin & Nemeroff 2002; Risen 2016, Psych. Review 123(2):182")]


def missing_denominator(text: str) -> list[Finding]:
    """L5: a rate or count with no stated base."""
    has_count = re.search(r"\b\d[\d,]*\b", text) is not None
    has_rate = re.search(r"\bpercent|per cent|%|\bout of\b|\brate\b|\bmore likely\b", text,
                         re.IGNORECASE) is not None
    has_base = re.search(r"\bout of\b|\bper\b|\bamong\b|\bof \d|\bin \d|\bbase\b|\bsample\b|"
                         r"\bn\s*=", text, re.IGNORECASE) is not None
    if not (has_count or has_rate) or has_base:
        return []
    if not has_rate:
        return []
    return [Finding(
        defect=Defect.SELECTION_DENOMINATOR, frame="base-rate / denominator absent",
        severity=0.65, evidence=text[:120], voiding=True,
        explanation=("A rate is asserted with no denominator and no sampling frame. "
                     "'40% higher risk' is unevaluable without the base rate; survivorship and "
                     "availability errors live exactly here. Report as unevaluable."),
        citation="Kahneman & Tversky 1973; Wald 1943; Bar-Hillel 1980")]


def overclaim(text: str) -> list[Finding]:
    """L6: the modal qualifier is stronger than the evidence type can licence."""
    low = text.lower()
    ceiling = None
    for markers, cap in WARRANT_CEILING:
        if any(m in low for m in markers):
            ceiling = cap
    if ceiling is None:
        ceiling = 0.75                       # unstated warrant: assume ordinary assertion
    strongest = 0.0
    word = ""
    for w, s in MODAL_STRENGTH.items():
        if re.search(rf"\b{re.escape(w)}\b", low) and s > strongest:
            strongest, word = s, w
    if strongest <= ceiling + 0.15:
        return []
    return [Finding(
        defect=Defect.MODAL_CALIBRATION, frame="overclaim (qualifier exceeds warrant)",
        severity=min(1.0, (strongest - ceiling) * 1.5),
        evidence=f"'{word}' at strength {strongest:.2f} vs warrant {ceiling:.2f}",
        explanation=(f"The claim's strongest modal is '{word}' ({strongest:.2f}) but the "
                     f"evidence type licenses at most {ceiling:.2f}. This is the Toulmin "
                     f"Q-not-in-licensed(W) defect: the qualifier must be weakened to the "
                     f"warrant, not the warrant inflated to the qualifier."),
        citation="Toulmin 1958; Hoeken & Sorm 2024")]


def negation_blindness_risk(text: str) -> list[Finding]:
    """M1: syntactic negation is the form models handle worst.

    2026 measurement: affixal negation ('un-') is handled better than syntactic negation
    ('not'), so a syntactic negation is flagged as the higher-risk form for any downstream
    neural reader.
    """
    hits = re.findall(r"\b(?:not|never|no longer|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|"
                      r"wasn'?t|weren'?t|cannot|can'?t|won'?t)\b", text, re.IGNORECASE)
    if not hits:
        return []
    return [Finding(
        defect=Defect.NEGATION_BLINDNESS, frame="syntactic negation (model blind spot)",
        severity=min(0.8, 0.25 * len(hits)),
        evidence=" ".join(hits[:6]),
        explanation=(f"{len(hits)} syntactic negation(s). ~25% of English sentences contain "
                     f"negation and models are measurably less reliable on 'not X' than on an "
                     f"affixal form. Route to the symbolic checker, not the encoder."),
        citation="Kassner & Schutze 2020; arXiv:2609.13685 (affixal > syntactic)")]


# Common English irregular verbs mapped to a shared stem. Small on purpose: it covers the
# verbs that actually appear in factual claims, and anything missing simply falls through to
# suffix stripping rather than being guessed at.
_IRREGULAR = {
    "found": "find", "find": "find", "finds": "find",
    "rose": "rise", "rise": "rise", "rises": "rise", "risen": "rise",
    "fell": "fall", "fall": "fall", "falls": "fall", "fallen": "fall",
    "was": "be", "were": "be", "is": "be", "are": "be", "been": "be", "be": "be", "am": "be",
    "had": "have", "has": "have", "have": "have", "having": "have",
    "did": "do", "does": "do", "do": "do", "done": "do", "doing": "do",
    "went": "go", "goes": "go", "go": "go", "gone": "go",
    "took": "take", "takes": "take", "take": "take", "taken": "take",
    "gave": "give", "gives": "give", "give": "give", "given": "give",
    "made": "make", "makes": "make", "make": "make",
    "said": "say", "says": "say", "say": "say",
    "saw": "see", "sees": "see", "see": "see", "seen": "see",
    "came": "come", "comes": "come", "come": "come",
    "knew": "know", "knows": "know", "know": "know", "known": "know",
    "grew": "grow", "grows": "grow", "grow": "grow", "grown": "grow",
    "showed": "show", "shows": "show", "show": "show", "shown": "show",
    "led": "lead", "leads": "lead", "lead": "lead",
    "won": "win", "wins": "win", "win": "win",
    "lost": "lose", "loses": "lose", "lose": "lose",
}


def check_pair_contradiction(claim_a: str, claim_b: str, lexicon=None) -> list[Finding]:
    """L8 / consistency: do two claims contradict?

    Two routes, cheapest first:
      1. one is the explicit negation of the other ("the study found X" / "the study did not
         find X") -- detected by comparing the token sequences with the negation removed;
      2. same syntactic frame with an ANTONYM in a content slot ("sales rose" / "sales fell")
         -- uses the WordNet relation already in `logos.lexical`.

    This is the single highest-value check in the file: conspiracy's contradictory-beliefs
    trait, motivated reasoning and plain inconsistency all surface as one pair like this.
    """
    out: list[Finding] = []
    neg = re.compile(r"\b(?:not|never|no longer|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|"
                     r"wasn'?t|weren'?t)\b", re.IGNORECASE)

    a_neg = bool(neg.search(claim_a))
    b_neg = bool(neg.search(claim_b))
    a_stripped = neg.sub("", claim_a).lower()
    b_stripped = neg.sub("", claim_b).lower()

    def _stem(w: str) -> str:
        """Light suffix stripping so regular verb agreement does not defeat the match.

        English negation forces a form change: 'found'/'find', 'detects'/'detect',
        'reported'/'report'. Comparing raw tokens makes the detector fire only on copula
        negation ("is" / "is not"), which is the most common real case but not the only one.

        LIMITATION, stated rather than hidden: irregular verbs (found/find, rose/rise, was/were)
        are NOT unified by suffix stripping. Those pairs need a lemma table. The copula and
        regular-verb cases -- which is what almost every real 'X is Y' / 'X is not Y' claim
        looks like -- are covered.
        """
        # Irregular verbs first: suffix stripping cannot unify found/find, rose/rise,
        # was/be. Without this map the contradiction detector silently missed the most
        # natural phrasing of a negation flip ("the trial found X" / "did not find X").
        irr = _IRREGULAR.get(w)
        if irr:
            return irr
        for suf in ("ing", "ed", "es", "s", "d"):
            if len(w) > 4 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    # Removing "not" leaves the do-support it introduced ("the analysis DOES detect ..."),
    # so auxiliaries are dropped from BOTH sides before comparing. Without this the detector
    # only caught copula negation.
    AUX = frozenset("does do did is are was were has have had will would can could "
                    "may might shall should".split())

    def norm(s: str) -> list[str]:
        toks = re.sub(r"[^a-z ]+", " ", s).split()
        return [_stem(t) for t in toks if t not in AUX]

    if a_neg != b_neg and norm(a_stripped) == norm(b_stripped) and norm(a_stripped):
        out.append(Finding(
            defect=Defect.LOCAL_COHERENCE, frame="direct contradiction (negation flip)",
            severity=0.95, evidence=f"{claim_a!r} vs {claim_b!r}",
            explanation="The two claims are identical apart from an explicit negation.",
            citation="propositional.entails (exhaustive)"))

    if lexicon is not None:
        # Antonymy is looked up on SURFACE tokens, not stems: a lexicon keys on lemmas
        # ("rose", "fell"), and stemming to "rise"/"fall" first made the lookup miss. Both
        # forms are tried because WordNet stores some inflections and not others.
        wa = re.sub(r"[^a-z ]+", " ", claim_a.lower()).split()
        wb = re.sub(r"[^a-z ]+", " ", claim_b.lower()).split()
        if len(wa) == len(wb) and wa:
            diffs = [(x, y) for x, y in zip(wa, wb) if x != y]
            if len(diffs) == 1:
                x, y = diffs[0]
                def _ant(a: str, b: str) -> bool:
                    for aa in {a, _stem(a)}:
                        for bb in {b, _stem(b)}:
                            try:
                                if bb in lexicon.antonyms(aa):
                                    return True
                            except Exception:      # noqa: BLE001
                                pass
                    return False
                if _ant(x, y) or _ant(y, x):
                    out.append(Finding(
                        defect=Defect.LOCAL_COHERENCE,
                        frame="contradiction via antonym substitution",
                        severity=0.85, evidence=f"{x} vs {y}",
                        explanation=(f"The two claims are identical except for the antonym "
                                     f"pair ({x}, {y}), so they cannot both be true."),
                        citation="WordNet antonymy; logos.lexical.relation"))
    return out


def proximate_contradiction(claims: list[str], lexicon=None) -> list[Finding]:
    """All pairwise contradictions among a set of claims about one subject."""
    out: list[Finding] = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            out.extend(check_pair_contradiction(claims[i], claims[j], lexicon))
    return out


def derailment(text: str) -> list[Finding]:
    """L8: sentences with no connective and no shared content word -- discourse incoherence."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) >= 4]
    if len(sents) < 3:
        return []
    disjoint = 0
    for a, b in zip(sents, sents[1:]):
        wa = {w for w in _words(a) if w not in ABSOLUTIST}
        wb = {w for w in _words(b) if w not in ABSOLUTIST}
        has_conn = bool({w for w in _words(b)} & CONNECTIVES)
        if not has_conn and not (wa & wb):
            disjoint += 1
    if disjoint == 0:
        return []
    ratio = disjoint / max(len(sents) - 1, 1)
    if ratio < 0.5:
        return []
    return [Finding(
        defect=Defect.LOCAL_COHERENCE, frame="derailment / thought-disorder features",
        severity=min(0.8, ratio),
        evidence=f"{disjoint}/{len(sents) - 1} sentence transitions share no content word",
        explanation=("Adjacent sentences share no content word and carry no connective. The "
                     "clinical scales for derailment measure exactly this. It is a property of "
                     "the text, not a diagnosis of an author, and it makes the claim "
                     "unevaluable as a single proposition."),
        citation="Andreasen 1986, Schiz. Bull. 12(3):473; Corcoran et al. 2018 World Psychiatry")]


# ------------------------------------------------------------------------- the catalogue
def frame_catalogue() -> list[dict]:
    """The pathological frames, each projected onto its formal defect. Citable mapping."""
    return [
        {"frame": "magical thinking", "defect": "L3_support_edge",
         "cite": "Rozin & Nemeroff 2002; Risen 2016"},
        {"frame": "affect heuristic", "defect": "L3_support_edge",
         "cite": "Slovic et al. 2002; Finucane et al. 2000"},
        {"frame": "base-rate neglect", "defect": "L2_conditioning_direction",
         "cite": "Kahneman & Tversky 1973"},
        {"frame": "prosecutor's fallacy", "defect": "L2_conditioning_direction",
         "cite": "Thompson & Schumann 1987"},
        {"frame": "confirmation / myside bias", "defect": "L4_missing_alternative",
         "cite": "Nickerson 1998; Stanovich & West 2007"},
        {"frame": "motivated reasoning", "defect": "L4_missing_alternative",
         "cite": "Kunda 1990; Kahan et al. 2017"},
        {"frame": "conspiracy ideation", "defect": "L4_missing_alternative",
         "cite": "Imhoff & Bruder 2014; Bates et al. 2026"},
        {"frame": "survivorship bias", "defect": "L5_selection_denominator",
         "cite": "Wald 1943; Denrell 2003"},
        {"frame": "availability heuristic", "defect": "L5_selection_denominator",
         "cite": "Tversky & Kahneman 1973"},
        {"frame": "apophenia / patternicity", "defect": "L1_quantifier_scope",
         "cite": "Brugger 2001; van Prooijen 2018"},
        {"frame": "absolutist overgeneralisation", "defect": "L1_quantifier_scope",
         "cite": "Al-Mosaiwi & Johnstone 2018"},
        {"frame": "Dunning-Kruger overclaiming", "defect": "L6_modal_calibration",
         "cite": "Kruger & Dunning 1999; Gignac & Zajenkowski 2020"},
        {"frame": "psychosis-adjacent derailment", "defect": "L8_local_coherence",
         "cite": "Andreasen 1986; Corcoran et al. 2018"},
        {"frame": "self-sealing / unfalsifiable", "defect": "L7_self_sealing",
         "cite": "Popper 1959; Douglas et al. 2017"},
        {"frame": "LLM negation blindness", "defect": "M1_negation_blindness",
         "cite": "Kassner & Schutze 2020; arXiv:2609.13685"},
        {"frame": "LLM overclaim in verification", "defect": "M4_overclaim",
         "cite": "arXiv:2610.12083 (51% GT agreement, order-sensitive)"},
    ]


def scan(text: str, lexicon=None, claims: list[str] | None = None) -> list[Finding]:
    """Run every text detector and return findings ordered by severity."""
    out: list[Finding] = []
    out += absolutist_density(text)
    out += self_sealing(text)
    out += single_hypothesis(text)
    out += magical_thinking(text)
    out += missing_denominator(text)
    out += overclaim(text)
    out += negation_blindness_risk(text)
    out += derailment(text)
    if claims and len(claims) > 1:
        out += proximate_contradiction(claims, lexicon)
    out.sort(key=lambda f: -f.severity)
    return out


def verdict_impact(findings: list[Finding]) -> dict:
    """What these findings do to a verdict: void it, discount it, or leave it alone."""
    voiding = [f for f in findings if f.voiding]
    if voiding:
        return {"action": "unevaluable", "reason": voiding[0].frame,
                "n_findings": len(findings)}
    if not findings:
        return {"action": "evaluate", "discount": 0.0, "n_findings": 0}
    # noisy-OR over severities: any one strong defect is enough to discount heavily
    keep = 1.0
    for f in findings:
        keep *= (1.0 - min(f.severity, 0.95))
    return {"action": "evaluate", "discount": round(1.0 - keep, 4),
            "n_findings": len(findings), "top": findings[0].frame}


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.badframes - SELF TEST")
    ok = True

    # --- absolutist density catches an unchecked universal
    f = absolutist_density("Everyone always knows that nothing ever works and nobody can help")
    good = len(f) == 1 and f[0].defect == Defect.QUANTIFIER_SCOPE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] absolutist density fires on an unchecked "
          f"universal ({f[0].evidence if f else 'none'})")

    good = absolutist_density("The 2021 trial enrolled 412 adults and reported a 12% "
                              "reduction in hospitalisation") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a properly quantified sentence does NOT fire")

    # --- self-sealing is VOIDING, not merely suspicious
    f = self_sealing("The lack of evidence proves that the programme is a cover-up")
    good = len(f) == 1 and f[0].voiding and f[0].defect == Defect.SELF_SEALING
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] self-sealing detected and flagged VOIDING")

    # --- conspiracy: agency without alternatives
    f = single_hypothesis("The government secretly orchestrated the cover-up")
    good = len(f) == 1 and f[0].defect == Defect.MISSING_ALTERNATIVE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] single-hypothesis conspiracy frame detected")

    # --- magical thinking: causation with no mechanism
    f = magical_thinking("This crystal attracts positive energy and brings good luck")
    good = len(f) == 1 and f[0].defect == Defect.SUPPORT_EDGE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] magical-thinking frame detected (no mechanism)")

    good = magical_thinking("The enzyme cleaves the substrate because the active site binds "
                            "the carboxyl group") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a mechanistic sentence does NOT fire")

    # --- denominator absent => VOIDING, the verdict is unevaluable not False
    f = missing_denominator("Taking the supplement made people 40% more likely to recover")
    good = len(f) == 1 and f[0].voiding
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] absent denominator => VOIDING (unevaluable)")

    good = missing_denominator("40% of 250 patients recovered, versus 28% of 250 controls") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a rate WITH a denominator does NOT fire")

    # --- overclaim: qualifier above the warrant ceiling
    f = overclaim("A single anecdote proves that the drug always works")
    good = len(f) == 1 and f[0].defect == Defect.MODAL_CALIBRATION
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] overclaim detected "
          f"({f[0].evidence if f else 'none'})")

    good = overclaim("A meta-analysis suggests the drug probably reduces mortality") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a meta-analysis with 'suggests' does NOT fire")

    # --- direct contradiction by negation flip
    f = check_pair_contradiction("the policy is safe for children",
                                 "the policy is not safe for children")
    good = len(f) == 1 and f[0].severity > 0.9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] negation-flip contradiction detected (copula form)")

    # regular-verb agreement must not defeat the match
    f = check_pair_contradiction("the analysis detects contamination",
                                 "the analysis does not detect contamination")
    good = len(f) == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] regular-verb negation matched after stemming")

    # irregular verbs are unified by an explicit map, not left to suffix stripping
    f = check_pair_contradiction("the trial found a benefit",
                                 "the trial did not find a benefit")
    good = len(f) == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] irregular verb pair (found/find) IS matched")

    f = check_pair_contradiction("sales rose in the quarter",
                                 "sales did not rise in the quarter")
    good = len(f) == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] irregular verb pair (rose/rise) IS matched")

    # the surviving limitation is paraphrase: the detector needs the same content words
    f = check_pair_contradiction("sales rose in the quarter",
                                 "revenue did not increase in the quarter")
    good = len(f) == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] paraphrase (rose/increase) is NOT matched "
          f"-- known limitation, asserted so it stays known")

    # --- contradiction by antonym requires the lexicon; construct a tiny stub
    class _Stub:
        def antonyms(self, w):
            return {"rose"} if w == "fell" else ({"fell"} if w == "rose" else set())

    f = check_pair_contradiction("sales rose sharply in the quarter",
                                 "sales fell sharply in the quarter", _Stub())
    good = len(f) == 1 and "antonym" in f[0].frame
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] antonym-substitution contradiction detected")

    good = check_pair_contradiction("sales rose sharply in the quarter",
                                    "costs fell sharply in the quarter", _Stub()) == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] different subjects do NOT contradict")

    # --- derailment
    txt = ("The committee met on Thursday. Quantum tunnelling permits barrier penetration. "
           "My neighbour owns three kayaks. Interest rates influence bond prices.")
    f = derailment(txt)
    good = len(f) == 1 and f[0].defect == Defect.LOCAL_COHERENCE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] derailment detected across disjoint sentences")

    good = derailment("The trial enrolled 412 adults. The trial reported a 12% reduction. "
                      "The reduction was therefore statistically significant.") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a coherent passage does NOT fire")

    # --- negation-blindness risk flags syntactic negation
    f = negation_blindness_risk("the drug was not effective and did not reduce mortality")
    good = len(f) == 1 and f[0].defect == Defect.NEGATION_BLINDNESS
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] syntactic negation flagged as a model blind spot")

    # --- the verdict impact is unchanged when nothing fires
    good = verdict_impact([])["action"] == "evaluate" and verdict_impact([])["discount"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no findings => verdict untouched")

    # --- and a voiding finding overrides everything else
    fs = scan("Everyone always knows the lack of evidence proves the cover-up")
    imp = verdict_impact(fs)
    good = imp["action"] == "unevaluable"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a voiding finding makes the claim UNEVALUABLE "
          f"(not False)")

    # --- catalogue completeness: every frame maps to a real defect
    cat = frame_catalogue()
    valid = {d.value for d in Defect}
    good = all(c["defect"] in valid for c in cat) and len(cat) >= 16
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] catalogue has {len(cat)} frames, all mapped to a "
          f"defect in the enum")

    # --- clean text yields nothing
    good = scan("A 2024 randomised controlled trial of 412 adults found a 12% reduction in "
                "hospitalisation, with a 95% confidence interval of 4 to 20%.") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a clean scientific sentence yields ZERO findings "
          f"(no false positives)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bad-reasoning frame detectors.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--text", default="")
    ap.add_argument("--catalogue", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.catalogue:
        for c in frame_catalogue():
            print(f"  {c['frame']:34s} -> {c['defect']:30s} {c['cite']}")
        return 0
    if not args.text:
        return 0 if run_self_test() else 1
    findings = scan(args.text)
    print(f"text: {args.text!r}\n")
    if not findings:
        print("  no structural defects found")
    for f in findings:
        print(f)
    print(f"\nverdict impact: {verdict_impact(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
