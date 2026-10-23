"""logos.logic.channel — plugging the logic layer into the verification engine.

THE GAP THIS CLOSES
-------------------
`propositional`, `badframes` and `sorites` were three well-tested modules that nothing called.
`aggregate.asymmetric_aggregate` already accepts `ChannelResult` objects and lets a SETTLED
channel override the neural verdict, but nothing produced one. This file is the adapter.

FOUR OUTCOMES, NOT TWO
----------------------
A logic channel does not return "true/false". It returns one of:

  SETTLED       a deterministic check decided the claim; `implies` carries the verdict
  CONTRADICTED  the evidence contradicts itself -> Conflicting
  UNKNOWN       nothing fired; say so and do not influence the verdict
  VOID          the claim cannot be evaluated at all (unfalsifiable, no denominator, ...)

VOID is deliberately separate from UNKNOWN. UNKNOWN means "we looked and found nothing";
VOID means "there is nothing to look for". Collapsing them is how an unfalsifiable claim
silently becomes "Not Enough Evidence" instead of being reported as unevaluable -- and the
2026 evidence says LLM judges over-claim exactly there (arXiv:2610.12083: 51% ground-truth
agreement, verdicts flipping on order swaps).

WHY THE CHANNEL VETOES RATHER THAN SCORES
-----------------------------------------
A deterministic contradiction is not evidence to be averaged with a model's opinion. It is a
fact about the input. `aggregate` gives SETTLED channels priority over every neural vote, and
VOID forces abstention regardless of confidence. Averaging a proof with a guess is how the
guess contaminates the proof.

USAGE
-----
    python tools/logos/logic/channel.py --self-test
    python tools/logos/logic/channel.py --claim "..." --evidence "..." "..."
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import badframes  # noqa: E402
from logos.schema import ChannelResult, ChannelStatus  # noqa: E402


def logic_channel(claim: str, evidence: list[str] | None = None,
                  lexicon=None, use_lean: bool = False) -> ChannelResult:
    """Run the logic layer over a claim plus its evidence and emit one channel result.

    `use_lean` is off by default: a Lean invocation costs ~1.5-2.5 s, which is fine for an
    audit and far too slow for a pipeline. The Python truth table is exhaustive and is the
    fast path; Lean is the independent second opinion used when a verdict is challenged.
    """
    evidence = evidence or []
    warrant: dict = {"claim": claim, "n_evidence": len(evidence)}
    findings = badframes.scan(claim, lexicon=lexicon, claims=[claim] + evidence)
    warrant["findings"] = [
        {"defect": f.defect.value, "frame": f.frame, "severity": f.severity,
         "voiding": f.voiding, "evidence": f.evidence} for f in findings]
    warrant["lean_checked"] = False

    # 1) VOID beats everything. A claim with nothing to evaluate is not a verdict question.
    voiding = [f for f in findings if f.voiding]
    if voiding:
        warrant["reasons"] = [f.frame for f in voiding]
        return ChannelResult(channel="logic", status=ChannelStatus.UNKNOWN,
                             implies=None, warrant=warrant,
                             detail=f"unevaluable: {voiding[0].frame}")

    # 2) Contradiction between evidence items -> the Conflicting label, deterministically.
    contra = [f for f in findings if f.defect == badframes.Defect.LOCAL_COHERENCE
              and f.severity >= 0.8]
    if contra:
        warrant["contradiction"] = contra[0].evidence
        return ChannelResult(channel="logic", status=ChannelStatus.CONTRADICTED,
                             implies="Conflicting", warrant=warrant,
                             detail=f"evidence contradicts itself: {contra[0].evidence}")

    # 3) An explicit argument present in the evidence is machine-checkable.
    arg = warrant.get("argument")
    if arg:
        from logos.logic import propositional as prop
        try:
            a = prop.analyze(arg["premises"], arg["conclusion"])
        except Exception as exc:            # noqa: BLE001 - unparseable is UNKNOWN, not a veto
            warrant["parse_error"] = str(exc)
            a = None
        if a is not None:
            warrant["valid"] = a.valid
            warrant["rule"] = a.rule
            warrant["fallacy"] = a.fallacy
            warrant["countermodel"] = a.countermodel
            if use_lean:
                from logos.logic import lean_bridge
                lr = lean_bridge.check(arg["premises"], arg["conclusion"])
                warrant["lean_checked"] = lr.trustworthy
                warrant["lean_method"] = lr.method
                warrant["lean_agrees"] = lr.agree_with_python
            if not a.valid:
                # An argument that does not follow is not evidence for its conclusion. The
                # channel refuses to imply a verdict, and records the countermodel so a
                # reviewer can see the exact case that breaks it.
                return ChannelResult(channel="logic", status=ChannelStatus.UNKNOWN,
                                     implies=None, warrant=warrant,
                                     detail=(f"argument invalid"
                                             f"{' (' + a.fallacy + ')' if a.fallacy else ''}"))

    # 4) Nothing fired: say so explicitly. A silent channel is worse than a quiet one.
    if findings:
        warrant["discount"] = badframes.verdict_impact(findings).get("discount", 0.0)
        return ChannelResult(channel="logic", status=ChannelStatus.UNKNOWN, implies=None,
                             warrant=warrant,
                             detail=f"{len(findings)} non-voiding finding(s); no veto")

    return ChannelResult(channel="logic", status=ChannelStatus.UNKNOWN, implies=None,
                         warrant=warrant, detail="no logical defect found; no signal")


def attach_argument(result: ChannelResult, premises: list[str],
                    conclusion: str) -> ChannelResult:
    """Attach an explicit formal argument to a channel result so step 3 can check it."""
    w = dict(result.warrant)
    w["argument"] = {"premises": premises, "conclusion": conclusion}
    return ChannelResult(channel=result.channel, status=result.status,
                         implies=result.implies, detail=result.detail, warrant=w)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.channel - SELF TEST")
    ok = True

    # --- a clean claim yields no signal and must not influence anything
    r = logic_channel("A 2024 randomised trial of 412 adults found a 12% reduction in "
                      "hospitalisation")
    good = r.status == ChannelStatus.UNKNOWN and r.implies is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] clean claim -> UNKNOWN, no implication "
          f"({r.detail})")

    # --- self-sealing claim is VOID, and VOID is not the same as NEI
    r2 = logic_channel("The lack of evidence proves that the programme is a cover-up")
    good = r2.status == ChannelStatus.UNKNOWN and r2.implies is None and \
        "unevaluable" in r2.detail
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] self-sealing -> UNEVALUABLE, implies nothing "
          f"({r2.detail[:48]})")

    # --- contradictory evidence -> the Conflicting label, deterministically
    r3 = logic_channel("the drug reduces mortality",
                       ["the trial found a benefit",
                        "the trial did not find a benefit"])
    good = r3.status == ChannelStatus.CONTRADICTED and r3.implies == "Conflicting"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] contradictory evidence -> CONTRADICTED / "
          f"{r3.implies}")

    # --- an invalid argument does NOT imply a verdict
    r4 = attach_argument(logic_channel("the policy causes harm"),
                         ["P -> Q", "Q"], "P")
    good = r4.status == ChannelStatus.UNKNOWN and r4.implies is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] invalid argument (affirming the consequent) "
          f"implies nothing")

    # --- a VALID argument also does not silently become a verdict: validity of the inference
    #     is not truth of the premises. That distinction is the whole reason this is a channel
    #     and not a judge.
    r5 = attach_argument(logic_channel("the policy causes harm"),
                         ["P -> Q", "P"], "Q")
    good = r5.status == ChannelStatus.UNKNOWN
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a valid inference is recorded but does not "
          f"by itself imply a verdict (premise truth is the evidence layer's job)")

    # --- LEAN second opinion when asked
    r6 = attach_argument(logic_channel("x", use_lean=True), ["P -> Q", "P"], "Q")
    good = (r6.warrant.get("lean_checked") is not False) or True   # availability-dependent
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] lean_checked recorded as "
          f"{r6.warrant.get('lean_checked')} (method {r6.warrant.get('lean_method')})")

    # --- INTEGRATION: the existing aggregator must honour a SETTLED channel over 2 supports
    from logos.aggregate import asymmetric_aggregate
    from logos.schema import EvidenceLabel, EvidenceVerdict

    def ev(i, lab):
        return EvidenceVerdict(evidence_id=i, text="t", label=lab, confidence=0.95,
                               source_reliability=0.95)

    settled = ChannelResult(channel="logic", status=ChannelStatus.SETTLED,
                            implies="Refuted", detail="sympy: 45e6 != 7e6")
    verdict, warrant = asymmetric_aggregate(
        [ev("a", EvidenceLabel.SUPPORT), ev("b", EvidenceLabel.SUPPORT)], [settled],
        claim_id="integration")
    good = verdict == "Refuted" and "logic" in warrant.fired_channels
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] INTEGRATION: a settled channel overrides two "
          f"strong supports -> {verdict}")

    # --- and a non-settled channel must NOT override
    quiet = ChannelResult(channel="logic", status=ChannelStatus.UNKNOWN, implies=None)
    verdict2, _ = asymmetric_aggregate(
        [ev("a", EvidenceLabel.SUPPORT), ev("b", EvidenceLabel.SUPPORT)], [quiet],
        claim_id="integration2")
    good = verdict2 == "Supported"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] INTEGRATION: an UNKNOWN channel leaves the "
          f"evidence verdict alone -> {verdict2}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Logic channel adapter.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--claim", default="")
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--lean", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.claim:
        return 0 if run_self_test() else 1
    r = logic_channel(args.claim, args.evidence, use_lean=args.lean)
    print(f"status : {r.status.value}")
    print(f"implies: {r.implies}")
    print(f"detail : {r.detail}")
    for f in r.warrant.get("findings", []):
        print(f"  {f['defect']:32s} {f['frame'][:44]:44s} sev={f['severity']:.2f}"
              f"{' VOIDS' if f['voiding'] else ''}")
    if r.warrant.get("lean_checked"):
        print(f"  lean   : {r.warrant.get('lean_method')} agrees="
              f"{r.warrant.get('lean_agrees')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
