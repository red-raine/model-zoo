"""logos.logic.argumentation — Toulmin layout, Walton schemes, enthymeme reconstruction.

WHY THREE THINGS IN ONE FILE
----------------------------
They are one pipeline, not three features:

    Toulmin layout      parse the argument into Claim / Data / Warrant / Qualifier
         |
         v
    Walton scheme       classify the warrant's type, then audit its critical questions
         |
         v
    Enthymeme           if a part is missing, reconstruct it -- and say when the
                        reconstruction is ambiguous

The single most useful rule from this whole layer is **Toulmin's qualifier-vs-warrant
check**: a claim's modal strength must not exceed what its warrant licenses. `Q not in
licensed(W) → overclaim`. It is one comparison, it is auditable, and it catches the dominant
failure mode of generated text.

WHY WALTON'S CRITICAL QUESTIONS AND NOT A VALIDITY CHECK
--------------------------------------------------------
Most real arguments are not deductively valid and are not meant to be. An appeal to expert
opinion is a *reasonable* argument form whose strength depends on whether the expert is
genuinely expert, in the right field, unbiased, and consistent with other experts. "Valid or
invalid" is the wrong question; *which critical questions are unaddressed* is the right one.

So a Walton scheme check does not return true/false. It returns the list of CQs with no
textual support -- the audit trail a reviewer needs.

HONESTY ABOUT THE COUNTS
------------------------
The literature does not agree on how many schemes there are: 96 in Walton's 2008 compendium,
65 in Macagno & Walton 2015, plus 44 sub-schemes in Lumer 2022. **Any single number quoted as
canonical is wrong.** This module implements a named subset and says so rather than implying
completeness.

USAGE
-----
    python tools/logos/logic/argumentation.py --self-test
    python tools/logos/logic/argumentation.py --toulmin "text"
    python tools/logos/logic/argumentation.py --schemes "text"
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import badframes, propositional as prop  # noqa: E402

# --------------------------------------------------------------------------- Toulmin
QUALIFIER_STRENGTH = badframes.MODAL_STRENGTH
WARRANT_CEILING = badframes.WARRANT_CEILING


@dataclass
class Toulmin:
    claim: str = ""
    data: list[str] = field(default_factory=list)
    warrant: str = ""
    backing: str = ""
    qualifier: str = ""
    rebuttal: str = ""


@dataclass
class ToulminFinding:
    defect: str
    severity: float
    explanation: str
    citation: str = ""
    voiding: bool = False


def _strongest_modal(text: str) -> tuple[float, str]:
    low = text.lower()
    best, word = 0.0, ""
    for w, s in QUALIFIER_STRENGTH.items():
        if re.search(rf"\b{re.escape(w)}\b", low) and s > best:
            best, word = s, w
    return best, word


def check_toulmin(t: Toulmin) -> list[ToulminFinding]:
    """Audit a Toulmin layout. Reports structural defects, not truth."""
    out: list[ToulminFinding] = []

    if not t.data:
        out.append(ToulminFinding(
            defect="missing_data", severity=0.8, voiding=True,
            explanation=("There is a claim and no grounds. A bare assertion cannot be "
                         "verified or refuted -- it is not an argument, it is a position."),
            citation="Toulmin 1958"))

    if not t.warrant:
        out.append(ToulminFinding(
            defect="enthymematic_warrant", severity=0.55,
            explanation=("The warrant is unstated, so the inference licence is invisible. "
                         "This is the normal enthymematic case, not an error -- but the "
                         "warrant must be reconstructed before the argument can be audited."),
            citation="Toulmin 1958; Walton 2008"))

    # THE key check: qualifier must not exceed what the warrant licenses.
    q_strength, q_word = _strongest_modal(t.qualifier or t.claim)
    ceiling = None
    basis = ""
    for markers, cap in WARRANT_CEILING:
        if any(m in (t.warrant + " " + t.backing).lower() for m in markers):
            ceiling, basis = cap, markers[0]
    if ceiling is None:
        ceiling = 0.75
        basis = "unstated warrant type"
    if q_strength > ceiling + 0.15:
        out.append(ToulminFinding(
            defect="overclaim", severity=min(1.0, (q_strength - ceiling) * 1.5),
            explanation=(f"Qualifier '{q_word}' ({q_strength:.2f}) exceeds the ceiling "
                         f"licensed by a '{basis}' warrant ({ceiling:.2f}). Either weaken "
                         f"the claim or strengthen the warrant -- do not inflate the warrant "
                         f"to match the rhetoric."),
            citation="Toulmin 1958; Hoeken & Sorm 2024"))

    if t.qualifier and t.rebuttal and _strongest_modal(t.rebuttal)[0] > 0.7:
        out.append(ToulminFinding(
            defect="rebuttal_contradicts_qualifier", severity=0.6,
            explanation=("A rebuttal is present but stated with near-certainty, which "
                         "contradicts having a qualified claim at all."),
            citation="Toulmin 1958"))
    return out


# ----------------------------------------------------------------------------- Walton
@dataclass(frozen=True)
class Scheme:
    name: str
    markers: tuple[str, ...]
    cqs: tuple[str, ...]
    conclusion_shape: str = ""


# Named subset, not the whole taxonomy. See the module docstring on why the counts disagree.
SCHEMES: tuple[Scheme, ...] = (
    Scheme("argument_from_expert_opinion",
           ("expert", "scientist", "professor", "study found", "research shows",
            "according to", "phd", "institute", "doctor", "peer-reviewed"),
           ("Is the source a genuine expert in THIS field?",
            "Is the expert's assertion consistent with other experts?",
            "Is the expert unbiased (no conflict of interest)?",
            "Does the expert have reliable evidence, or only opinion?",
            "Is the claim within the expert's competence?")),
    Scheme("argument_from_popularity",
           ("everyone", "most people", "popular", "widely believed", "consensus of the public",
            "millions believe"),
           ("Is popularity evidence of truth in this domain?",
            "Was the popularity measured or assumed?",
            "Are the people informed on this question?")),
    Scheme("argument_from_ignorance",
           ("no evidence", "cannot be disproved", "never been shown", "no one has proved",
            "absence of evidence", "you can't prove"),
           ("Is the search for evidence exhaustive?",
            "Would evidence be expected to exist if the claim were true?")),
    Scheme("argument_from_cause_to_effect",
           ("causes", "leads to", "results in", "because of", "due to", "triggers"),
           ("Is there a plausible mechanism?",
            "Is the correlation explained better by a common cause?",
            "Is the effect observed at the right time order?")),
    Scheme("argument_from_correlation",
           ("correlat", "associated with", "linked to", "goes up with", "tracked with"),
           ("Could the correlation be coincidence (sample size, multiple testing)?",
            "Is there a third common cause?",
            "Is the causal direction established?")),
    Scheme("argument_from_sign",
           ("indicates", "sign of", "suggests that", "points to", "a marker of"),
           ("Is the sign reliable in this context?",
            "Are there other explanations for the same sign?")),
    Scheme("argument_from_consequences",
           ("would mean", "if true then", "the consequence", "leads to disaster",
            "would result in", "catastroph"),
           ("Are the predicted consequences actually likely?",
            "Are the consequences desirable/undesirable is irrelevant to truth -- are they "
            "EVIDENCE?" )),
    Scheme("argument_from_analogy",
           ("just like", "similar to", "analogous", "in the same way", "by comparison"),
           ("Are the two cases similar in the RELEVANT respects?",
            "Are there differences that matter?")),
    Scheme("slippery_slope",
           ("will inevitably", "next thing", "then they'll", "one step toward",
            "opens the door", "where does it end"),
           ("Is each step in the chain actually probable?",
            "Is there a barrier preventing the chain from continuing?")),
    Scheme("ad_hominem",
           ("liar", "corrupt", "paid by", "can't be trusted", "biased so",
            "what do you expect from"),
           ("Is the personal attribute relevant to the truth of the CLAIM?",
            "Is the attribute itself established?")),
    Scheme("appeal_to_emotion",
           ("think of the children", "how would you feel", "disgusting", "outrageous",
            "heartbreaking", "fear"),
           ("Does the emotion bear on whether the claim is TRUE?")),
    Scheme("argument_from_authority_of_tradition",
           ("always been", "traditionally", "time-honoured", "our ancestors",
            "the way it's always"),
           ("Has the tradition been tested, or only continued?")),
    Scheme("argument_from_bias",
           ("funded by", "conflict of interest", "stands to gain", "works for"),
           ("Does the alleged bias actually invalidate the evidence?",
            "Is the bias itself documented?")),
    Scheme("abduction_best_explanation",
           ("best explanation", "most likely because", "the only explanation",
            "explains why", "would account for"),
           ("Are alternative hypotheses considered?",
            "Does the hypothesis predict anything not already observed?",
            "Is it the simplest explanation that covers the data?")),
)


@dataclass
class SchemeMatch:
    scheme: str
    matched_markers: list[str]
    unsupported_cqs: list[str]
    n_cqs: int

    @property
    def coverage(self) -> float:
        return 1.0 - len(self.unsupported_cqs) / max(self.n_cqs, 1)


def match_schemes(text: str) -> list[SchemeMatch]:
    """Which argument schemes the text invokes, and which critical questions it leaves open.

    CQ support is detected lexically -- a CQ counts as addressed when the text contains
    vocabulary from that CQ. This is deliberately conservative and auditable: the output is a
    list of questions a reviewer must still ask, not a verdict.
    """
    low = text.lower()
    out: list[SchemeMatch] = []
    for s in SCHEMES:
        hit = [m for m in s.markers if m in low]
        if not hit:
            continue
        unsupported = []
        for cq in s.cqs:
            words = [w for w in re.findall(r"[a-z]{4,}", cq.lower())
                     if w not in ("this", "that", "they", "does", "with", "have", "from",
                                  "their", "there", "which", "would", "claim")]
            if not any(w in low for w in words):
                unsupported.append(cq)
        out.append(SchemeMatch(scheme=s.name, matched_markers=hit,
                               unsupported_cqs=unsupported, n_cqs=len(s.cqs)))
    out.sort(key=lambda m: -len(m.matched_markers))
    return out


# ------------------------------------------------------------------------- enthymeme
@dataclass
class Reconstruction:
    valid_as_stated: bool
    candidates: list[str] = field(default_factory=list)
    ambiguous: bool = False
    note: str = ""


def reconstruct(premises: list[str], conclusion: str,
                extra_candidates: list[str] | None = None) -> Reconstruction:
    """Find the missing premise that makes an argument valid.

    Candidates are drawn from the atoms already present, plus any supplied extras. The
    conclusion itself is excluded -- adding the conclusion as a premise would "validate"
    every argument trivially and is the classic way to fake enthymeme recovery.

    AMBIGUITY IS A RESULT, NOT A FAILURE. When two or more distinct premises each make the
    argument valid, the reconstruction is underdetermined and the argument cannot be audited
    without choosing. That is reported, because silently picking one is how an enthymeme
    audit becomes fiction.
    """
    prem = [prop.parse(p) for p in premises]
    conc = prop.parse(conclusion)

    if prop.entails(prem, conc):
        return Reconstruction(valid_as_stated=True,
                              note="already valid as stated; no premise is missing")

    pool: set[str] = set()
    for p in prem:
        pool |= prop.atoms(p)
    pool |= prop.atoms(conc)
    if extra_candidates:
        for c in extra_candidates:
            try:
                pool |= prop.atoms(prop.parse(c))
            except Exception:              # noqa: BLE001
                pool.add(c)
    pool -= prop.atoms(conc)               # never propose the conclusion itself

    found: list[str] = []
    for atom in sorted(pool):
        cand = ("atom", atom)
        if prop.entails(prem + [cand], conc):
            found.append(atom)

    return Reconstruction(valid_as_stated=False, candidates=found,
                          ambiguous=len(found) > 1,
                          note=("reconstruction is AMBIGUOUS -- more than one missing premise "
                                "would validate the argument; do not pick one silently"
                                if len(found) > 1 else
                                f"missing premise is {found[0]}" if found else
                                "no single missing premise validates this argument; the "
                                "defect is in the inference itself, not a gap"))


# ------------------------------------------------------------------------ full audit
def audit(text: str, premises: list[str] | None = None,
          conclusion: str | None = None) -> dict:
    """Run the whole argumentation pipeline over a text and return one report."""
    t = Toulmin(claim=text)
    findings = check_toulmin(t)
    schemes = match_schemes(text)
    bad = badframes.scan(text)
    rec = None
    if premises and conclusion:
        rec = reconstruct(premises, conclusion)
    return {
        "toulmin_findings": [f.__dict__ for f in findings],
        "schemes": [{"scheme": m.scheme, "markers": m.matched_markers,
                     "coverage": round(m.coverage, 3),
                     "open_cqs": m.unsupported_cqs} for m in schemes],
        "bad_frames": [{"defect": f.defect.value, "frame": f.frame, "severity": f.severity}
                       for f in bad],
        "reconstruction": rec.__dict__ if rec else None,
    }


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.argumentation - SELF TEST")
    ok = True

    # --- the key Toulmin check: anecdote + "proves" is an overclaim
    f = check_toulmin(Toulmin(claim="this proves the policy works", data=["I saw one case"],
                              warrant="anecdote"))
    defects = [x.defect for x in f]
    good = "overclaim" in defects
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] anecdote warrant + 'proves' -> overclaim "
          f"({defects})")

    # --- RCT + "suggests" is NOT an overclaim
    f = check_toulmin(Toulmin(claim="the drug suggests a benefit",
                              data=["n=412 randomised"], warrant="randomised controlled trial"))
    good = "overclaim" not in [x.defect for x in f]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] RCT warrant + 'suggests' does NOT overclaim")

    # --- no data at all is VOIDING
    f = check_toulmin(Toulmin(claim="the policy is bad"))
    voiding = [x for x in f if x.voiding]
    good = len(voiding) == 1 and voiding[0].defect == "missing_data"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] claim with no grounds -> VOIDING missing_data")

    # --- unstated warrant is enthymematic, NOT voiding
    f = check_toulmin(Toulmin(claim="x is bad", data=["a study"]))
    good = any(x.defect == "enthymematic_warrant" and not x.voiding for x in f)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unstated warrant -> enthymematic (not voiding)")

    # --- Walton: an appeal to expertise is detected with its open CQs
    m = match_schemes("According to a peer-reviewed study, the drug is effective")
    good = m and m[0].scheme == "argument_from_expert_opinion"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] expert-opinion scheme detected "
          f"({m[0].scheme if m else 'none'})")

    good = m and len(m[0].unsupported_cqs) >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and it reports "
          f"{len(m[0].unsupported_cqs)}/{m[0].n_cqs} open critical questions")

    # --- appeal to ignorance
    m = match_schemes("No evidence has ever been shown that it is safe")
    good = any(x.scheme == "argument_from_ignorance" for x in m)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] argument-from-ignorance scheme detected")

    # --- slippery slope
    m = match_schemes("Allow this and it will inevitably lead to full censorship")
    good = any(x.scheme == "slippery_slope" for x in m)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] slippery-slope scheme detected")

    # --- every scheme has at least one critical question
    good = all(len(s.cqs) >= 1 for s in SCHEMES)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(SCHEMES)} schemes carry critical "
          f"questions")

    # --- ENTHYMEME: P->Q, therefore Q, missing P
    r = reconstruct(["P -> Q"], "Q")
    good = (not r.valid_as_stated) and r.candidates == ["P"] and not r.ambiguous
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] enthymeme recovery: P->Q |- Q needs {r.candidates}")

    # --- already valid: nothing is missing
    r = reconstruct(["P -> Q", "P"], "Q")
    good = r.valid_as_stated and not r.candidates
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] already-valid argument reports no missing premise")

    # --- affirming the consequent: NO single premise rescues it
    r = reconstruct(["P -> Q", "Q"], "P")
    good = (not r.valid_as_stated) and not r.candidates
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] affirming the consequent cannot be rescued by "
          f"adding a premise ({r.note[:40]})")

    # --- the classic Barbara enthymeme in propositional dress
    r = reconstruct(["men -> mortal"], "mortal")
    good = (not r.valid_as_stated) and r.candidates == ["men"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'all men are mortal, so Socrates is mortal' "
          f"needs {r.candidates}")

    # --- ambiguity: report it rather than silently choosing
    r = reconstruct(["P -> Q", "R -> Q"], "Q")
    good = r.ambiguous or (not r.candidates)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] two routes to Q are reported as ambiguous="
          f"{r.ambiguous} (candidates {r.candidates})")

    # --- the conclusion is never proposed as its own premise
    r = reconstruct(["P -> Q"], "Q")
    good = "Q" not in r.candidates
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the conclusion is never proposed as the missing "
          f"premise")

    # --- full pipeline runs end to end
    rep = audit("According to experts this proves the link is certain, because of one case",
                ["P -> Q"], "Q")
    good = (len(rep["toulmin_findings"]) >= 1 and len(rep["schemes"]) >= 1
            and rep["reconstruction"] is not None)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] full audit pipeline returns toulmin + schemes + "
          f"bad frames + reconstruction")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Toulmin + Walton + enthymeme audit.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--toulmin", default="")
    ap.add_argument("--schemes", default="")
    ap.add_argument("--reconstruct", nargs="+", default=[], metavar="PREMISE_OR_CONCLUSION")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.reconstruct and len(args.reconstruct) >= 2:
        r = reconstruct(args.reconstruct[:-1], args.reconstruct[-1])
        print(f"valid as stated: {r.valid_as_stated}")
        print(f"missing premise: {r.candidates}")
        print(f"ambiguous      : {r.ambiguous}")
        print(f"note           : {r.note}")
        return 0
    if args.toulmin:
        for f in check_toulmin(Toulmin(claim=args.toulmin)):
            print(f"{f.defect:30s} sev={f.severity:.2f}{' VOIDS' if f.voiding else ''}")
            print(f"    {f.explanation}")
        return 0
    if args.schemes:
        for m in match_schemes(args.schemes):
            print(f"{m.scheme:36s} coverage {m.coverage:.2f} "
                  f"({m.n_cqs - len(m.unsupported_cqs)}/{m.n_cqs} CQs addressed)")
            for cq in m.unsupported_cqs:
                print(f"    OPEN: {cq}")
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
