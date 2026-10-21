"""calibrate_trace.py — does TRACE measure reasoning, or vocabulary?

THE QUESTION
------------
`tools/scoring/trace_eval.py` implements TRACE as a bag-of-phrases counter:
each dimension is `1.0 + min(6, distinct_marker_count * weight)` where the
markers are literal word lists ("because", "however", "for example", "moreover",
"likely", "therefore"...). Nothing in the scorer checks whether a claim is TRUE.

If that is right, then:
  * a WRONG answer stuffed with markers should score HIGH
  * a CORRECT answer without markers should score LOW

Those are falsifiable predictions. This script tests them.

Design notes / self-critique
----------------------------
* I previously claimed "a hand-written textbook Toulmin answer scores 0.281 (grade
  D) therefore the metric is strict". That was weak: I wrote the answer and I
  ASSERTED it was good. Asserting a fixture is high-quality is not evidence.
  The fixtures below are labelled by an OBJECTIVE property (true/false, marked/
  unmarked) rather than by my opinion of their quality.
* `distinct` counting is the key mechanism: repeating "however" 50 times does not
  raise rebuttal, but sprinkling 5 different connectives does. The
  `repeat_same_marker` fixture tests that directly.

USAGE
-----
    python tools/scoring/calibrate_trace.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_eval import trance_report  # noqa: E402

# --------------------------------------------------------------------------
# Fixtures. Each is labelled by an objective property, not by taste.
#   truth     : "true" | "false" | "nonsense"
#   markers   : "dense" | "none" | "single"
# --------------------------------------------------------------------------

FIXTURES = [
    dict(
        id="false_dense_markers",
        truth="false", markers="dense",
        note="every sentence is wrong; every Toulmin connective is present",
        text=(
            "The sky appears green, therefore the answer is wavelength absorption. "
            "For example, since air molecules are larger than the wavelength of blue "
            "light, the data shows they reflect green most strongly. Moreover, studies "
            "show that oxygen is a metal, and it is well-known that nitrogen glows "
            "green by definition. This is because photons lose energy in the "
            "troposphere. However, one might object that the sun is purple; "
            "nevertheless the evidence shows green dominates. In most cases it is "
            "likely, possibly, generally true that grass is sky-coloured."
        ),
    ),
    dict(
        id="false_dense_markers_v2",
        truth="false", markers="dense",
        note="same trick, different wrong domain (arithmetic)",
        text=(
            "Therefore, 2 + 2 = 5. For example, since 2 is approximately 2.5, the "
            "sum of 2.5 and 2.5 equals 5, when rounding. Moreover, studies show that "
            "in addition the difference is negligible, based on observed data. This is "
            "because the reason is that binary addition rounds upward, which means it "
            "follows that 2 + 2 = 5. However, one might object; nevertheless, in most "
            "cases it is generally, probably, likely correct."
        ),
    ),
    dict(
        id="correct_no_markers",
        truth="true", markers="none",
        note="a correct, complete answer using plain declarative prose",
        text=(
            "Air molecules are far smaller than visible wavelengths, so incident "
            "sunlight is scattered elastically off them. Scattering strength rises "
            "steeply as wavelength falls, close to the inverse fourth power, and this "
            "is why short blue wavelengths are redirected across the whole sky while "
            "red passes through. An observer looking away from the sun sees that "
            "scattered blue arriving from every direction. Near sunset the optical "
            "path is many times longer, so most blue is removed before the light "
            "reaches the eye and the residual red dominates."
        ),
    ),
    dict(
        id="correct_short_no_markers",
        truth="true", markers="none",
        note="a correct, terse answer",
        text="Rayleigh scattering: air scatters short wavelengths far more strongly than long ones.",
    ),
    dict(
        id="correct_toulmin_form",
        truth="true", markers="dense",
        note="a correct answer written in labelled Toulmin form",
        text=(
            "Claim: the sky is blue because of Rayleigh scattering. Evidence: the "
            "measured scattering cross-section varies with wavelength close to the "
            "inverse fourth power. Warrant: atmospheric molecules are much smaller "
            "than visible wavelengths, so scattering is elastic. Backing: this is "
            "supported by spectra measured across many sites. Qualifier: the effect "
            "is generally masked when dust or cloud is present. Rebuttal: however, at "
            "sunset the optical path lengthens so the transmitted colour shifts to red."
        ),
    ),
    dict(
        id="nonsense_dense_markers",
        truth="nonsense", markers="dense",
        note="grammatical word salad, zero information, all connectives",
        text=(
            "Therefore the flurble is because of the snazz. For example, since the "
            "wibble observes the data, we see that moreover the quux is likely. This "
            "is because the reason is that it follows that the glorp, which means the "
            "answer is glorp. However, one might object that the blim; nevertheless, "
            "in most cases studies show the blim is generally, probably true."
        ),
    ),
    dict(
        id="repeat_single_marker",
        truth="true", markers="single",
        note="correct content, one connective repeated many times",
        text=(
            "Air scatters blue light however . Rayleigh scattering applies however. "
            "Short wavelengths scatter more however. The effect goes as inverse "
            "fourth power however. Therefore sunset looks red however."
        ),
    ),
    dict(
        id="empty", truth="nonsense", markers="none", note="empty string", text="",
    ),
]


def main() -> int:
    rows = []
    print("=" * 92)
    print(f"{'fixture':<28}{'truth':<10}{'markers':<9}{'TRACE':>7}{'grade':>7}"
          f"{'claims':>8}{'evid':>6}{'warr':>6}{'back':>6}{'qual':>6}{'rebu':>6}")
    print("-" * 92)
    for f in FIXTURES:
        r = trance_report(f["text"])
        d = r["per_dim"]
        rows.append({**f, "trace": r["trace_score"], "grade": r["grade"], **d})
        print(f"{f['id']:<28}{f['truth']:<10}{f['markers']:<9}"
              f"{r['trace_score']:>7.3f}{r['grade']:>7}"
              f"{d['claims']:>8.1f}{d['evidence']:>6.1f}{d['warrant']:>6.1f}"
              f"{d['backing']:>6.1f}{d['qualifier']:>6.1f}{d['rebuttal']:>6.1f}")
    print("=" * 92)

    # ---- the two predictions -------------------------------------------------
    by = {r["id"]: r for r in rows}
    false_dense = [by["false_dense_markers"]["trace"], by["false_dense_markers_v2"]["trace"]]
    correct_plain = [by["correct_no_markers"]["trace"], by["correct_short_no_markers"]["trace"]]
    nonsense = by["nonsense_dense_markers"]["trace"]
    labelled = by["correct_toulmin_form"]["trace"]

    checks = []

    checks.append((
        "P1: WRONG answer with dense markers scores >= 0.55 (the 'gate')",
        all(x >= 0.55 for x in false_dense),
        f"false-answer scores {[round(x,3) for x in false_dense]}",
    ))
    checks.append((
        "P2: CORRECT answer without markers scores BELOW the false ones",
        max(correct_plain) < min(false_dense),
        f"correct {[round(x,3) for x in correct_plain]} vs false {[round(min(false_dense),3), round(max(false_dense),3)]}",
    ))
    checks.append((
        "P3: word salad with markers scores >= 0.55",
        nonsense >= 0.55,
        f"nonsense score {round(nonsense,3)}",
    ))
    checks.append((
        "P4: labelled-Toulmin correct answer scores highest",
        labelled >= max(r["trace"] for r in rows),
        f"labelled {round(labelled,3)} vs max other "
        f"{round(max(r['trace'] for r in rows if r['id'] != 'correct_toulmin_form'),3)}",
    ))
    checks.append((
        "P5: repeating ONE connective does not substitute for several",
        by["repeat_single_marker"]["trace"] < labelled,
        f"repeat-single {round(by['repeat_single_marker']['trace'],3)} vs labelled {round(labelled,3)}",
    ))

    print("\nPREDICTIONS")
    ok = 0
    for label, passed, detail in checks:
        print(f"  [{'HOLDS' if passed else 'FAILS'}] {label}")
        print(f"           {detail}")
        ok += 1 if passed else 0

    print(f"\n{ok}/{len(checks)} predictions hold")

    if checks[0][1] and checks[1][1]:
        print("\nCONCLUSION: TRACE is a lexical-marker counter, not a truth or")
        print("reasoning-quality metric. A confidently WRONG answer that uses Toulmin")
        print("connectives outscores a correct answer that does not. It cannot serve as")
        print("the model-quality gate; it can still serve as a *structural* diagnostic")
        print("and as a training-data filter IF its threshold is calibrated on data,")
        print("not on a guess.")
    else:
        print("\nCONCLUSION: the marker-counter hypothesis did NOT hold. My reading of")
        print("the scorer is wrong and the earlier claim needs withdrawing.")

    out = Path(r"B:\datasets\forge\eval\trace-calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fixtures": rows,
                               "predictions": [{"label": l, "holds": p, "detail": d}
                                               for l, p, d in checks]}, indent=2),
                   encoding="utf-8")
    print(f"\n[calibrate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
