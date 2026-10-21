"""syllogism.py — categorical syllogism validity, by the rules, not by vibes.

WHY THIS FILE EXISTS
--------------------
The engine in `dataset-builders/reasoning_engine.py` has a frame called
"Disjunctive Syllogism" whose entire detection signal was the substring "or".
It generated traces saying "The syllogistic form is: 'P or Q. Not P. Therefore, Q.'"
with an empty form, on texts about cron jobs.

Logical validity is not a lexical property. It is a property of ARGUMENT FORM, and
for the categorical syllogism it is completely decidable — 256 forms, 24 valid.
This module decides it properly.

WHAT IT DOES AND DOES NOT DO
----------------------------
DOES: decide whether a categorical syllogism (mood + figure) is valid, and say
      WHICH rule it violates when it is not. Deterministic, zero model calls.
DOES NOT: decide whether the premises are TRUE. That needs evidence, not logic.
      A valid syllogism with false premises gives a false conclusion; an invalid
      one can have a true conclusion by luck. Keeping these separate is the whole
      point of the honesty manifest.

MOODS
-----
    A  All S are P          (universal affirmative)
    E  No S are P           (universal negative)
    I  Some S are P         (particular affirmative)
    O  Some S are not P     (particular negative)

FIGURE (position of the middle term M)
    1   M-P / S-M / S-P
    2   P-M / S-M / S-P
    3   M-P / M-S / S-P
    4   P-M / M-S / S-P

USAGE
-----
    python tools/logos/logic/syllogism.py --self-test
    python tools/logos/logic/syllogism.py --mood AAA --figure 1
    python tools/logos/logic/syllogism.py --check "all men are mortal; all greeks are men; therefore all greeks are mortal"
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys

MOODS = ("A", "E", "I", "O")
FIGURES = (1, 2, 3, 4)

# The 24 valid forms (Aristotelian, including the 5 weakened ones).
# Anything not in here is invalid; the rules below must agree with this table.
VALID = {
    1: {"AAA", "EAE", "AII", "EIO", "AAI", "EAO"},
    2: {"EAE", "AEE", "EIO", "AOO", "EAO", "AEO"},
    3: {"AAI", "IAI", "AII", "EAO", "OAO", "EIO"},
    4: {"AAI", "AEE", "IAI", "EAO", "EIO", "AEO"},
}

NAMES = {
    ("AAA", 1): "Barbara", ("EAE", 1): "Celarent", ("AII", 1): "Darii",
    ("EIO", 1): "Ferio", ("AAI", 1): "Barbari", ("EAO", 1): "Celaront",
    ("EAE", 2): "Cesare", ("AEE", 2): "Camestres", ("EIO", 2): "Festino",
    ("AOO", 2): "Baroco", ("EAO", 2): "Cesaro", ("AEO", 2): "Camestrop",
    ("AAI", 3): "Darapti", ("IAI", 3): "Disamis", ("AII", 3): "Datisi",
    ("EAO", 3): "Felapton", ("OAO", 3): "Bocardo", ("EIO", 3): "Ferison",
    ("AAI", 4): "Bramantip", ("AEE", 4): "Camenes", ("IAI", 4): "Dimaris",
    ("EAO", 4): "Fesapo", ("EIO", 4): "Fresison", ("AEO", 4): "Camenop",
}


# --------------------------------------------------------------------------
# Distribution of terms
# --------------------------------------------------------------------------
# A term is DISTRIBUTED when the proposition says something about ALL of it.
#   A: All S are P      -> S distributed, P not
#   E: No S are P       -> both distributed
#   I: Some S are P     -> neither
#   O: Some S are not P -> P distributed, S not

def distributed(mood: str, position: str) -> set[str]:
    """Which of {'subject','predicate'} is distributed in this proposition."""
    if mood == "A":
        return {"subject"}
    if mood == "E":
        return {"subject", "predicate"}
    if mood == "I":
        return set()
    if mood == "O":
        return {"predicate"}
    raise ValueError(f"unknown mood {mood!r}")


def _term_positions(figure: int) -> list[tuple[str, str, str]]:
    """(major_subject, major_predicate, minor_subject, minor_predicate, concl_subject, concl_predicate)
    as symbolic names S/P/M for each figure."""
    return {
        1: [("M", "P"), ("S", "M"), ("S", "P")],
        2: [("P", "M"), ("S", "M"), ("S", "P")],
        3: [("M", "P"), ("M", "S"), ("S", "P")],
        4: [("P", "M"), ("M", "S"), ("S", "P")],
    }[figure]


def check_rules(mood: str, figure: int) -> tuple[bool, list[str]]:
    """Apply the six classical validity rules. Returns (valid, violated_rules)."""
    if len(mood) != 3 or any(m not in MOODS for m in mood):
        raise ValueError(f"mood must be 3 letters from AEIO, got {mood!r}")
    if figure not in FIGURES:
        raise ValueError(f"figure must be 1-4, got {figure!r}")

    m_mood, n_mood, c_mood = mood[0], mood[1], mood[2]
    (M_s, M_p), (N_s, N_p), (C_s, C_p) = _term_positions(figure)

    violations = []

    # distribution sets per premise/conclusion, keyed by term name
    dist: dict[str, int] = {"S": 0, "P": 0, "M": 0}
    for prop_mood, (subj, pred) in ((m_mood, (M_s, M_p)), (n_mood, (N_s, N_p))):
        d = distributed(prop_mood, "subject")
        if "subject" in d:
            dist[subj] += 1
        if "predicate" in d:
            dist[pred] += 1
    concl_dist: dict[str, int] = {"S": 0, "P": 0}
    dc = distributed(c_mood, "subject")
    if "subject" in dc:
        concl_dist[C_s] = 1
    if "predicate" in dc:
        concl_dist[C_p] = 1

    # R1: the middle term must be distributed at least once
    if dist["M"] == 0:
        violations.append("R1 undistributed middle term")

    # R2: no term may be distributed in the conclusion unless distributed in a premise
    for t in ("S", "P"):
        if concl_dist.get(t) and dist[t] == 0:
            violations.append(f"R2 illicit process of the {'major' if t == 'P' else 'minor'} term")

    # R3: two negative premises are not allowed
    neg = {"E", "O"}
    if m_mood in neg and n_mood in neg:
        violations.append("R3 two negative premises")

    # R4: a negative premise requires a negative conclusion, and vice versa
    if (m_mood in neg) != (n_mood in neg):
        if c_mood not in neg:
            violations.append("R4 negative premise with affirmative conclusion")
    if c_mood in neg and not (m_mood in neg or n_mood in neg):
        violations.append("R4 negative conclusion from affirmative premises")

    # R5: two particular premises are not allowed
    part = {"I", "O"}
    if m_mood in part and n_mood in part:
        violations.append("R5 two particular premises")

    # R6: a particular premise requires a particular conclusion
    if (m_mood in part or n_mood in part) and c_mood not in part:
        violations.append("R6 particular premise with universal conclusion")

    return (len(violations) == 0, violations)


def is_valid(mood: str, figure: int) -> bool:
    """Rule-derived validity (the table is only used by the self-test)."""
    ok, _ = check_rules(mood, figure)
    return ok


def explain(mood: str, figure: int) -> dict:
    ok, violations = check_rules(mood, figure)
    return {
        "mood": mood,
        "figure": figure,
        "valid": ok,
        "name": NAMES.get((mood, figure)),
        "violations": violations,
        "table_agrees": (mood in VALID[figure]),
        "note": ("valid by form. This says NOTHING about whether the premises "
                 "are true." if ok else
                 "invalid by form. A true conclusion would be a coincidence."),
    }


# --------------------------------------------------------------------------
# Natural-language front end (shallow, and honest about being shallow)
# --------------------------------------------------------------------------

_QUANT = [
    (r"^(?:all|every|each)\b", "A"),
    (r"^(?:no|none|not a single)\b", "E"),
    (r"^(?:some|at least one|a few)\b.*\bnot\b", "O"),
    (r"^(?:some|at least one|a few)\b", "I"),
]


def classify_proposition(text: str) -> str | None:
    t = text.strip().rstrip(".").lower()
    for pat, mood in _QUANT:
        if re.match(pat, t):
            return mood
    return None


def parse_argument(text: str) -> dict | None:
    """Very shallow parse of 'A; B; therefore C'. Returns mood+figure or None.

    Deliberately conservative: if the premises do not map onto a categorical
    shape it returns None rather than guessing. Guessing is how the substring
    'or' became a Disjunctive Syllogism.
    """
    parts = [p.strip() for p in re.split(r"[;\n]|\.\s+(?=[A-Z])", text) if p.strip()]
    concl = None
    premises = []
    for p in parts:
        low = p.lower()
        if low.startswith(("therefore", "thus", "hence", "so ", "consequently")):
            concl = re.sub(r"^(therefore|thus|hence|so|consequently)[,:]?\s*", "",
                           p, flags=re.I)
        else:
            premises.append(p)
    if concl is None or len(premises) != 2:
        return None

    moods = [classify_proposition(p) for p in premises] + [classify_proposition(concl)]
    if any(m is None for m in moods):
        return None

    # terms: crude "all X are Y" -> (X, Y)
    def terms(p):
        m = re.match(r"^(?:all|every|each|no|none|some|at least one|a few)\s+(.+?)\s+"
                     r"(?:are|is)\s+(?:not\s+)?(.+)$", p.strip().rstrip("."), re.I)
        return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else None

    ts = [terms(p) for p in premises] + [terms(concl)]
    if any(t is None for t in ts):
        return None
    # The MAJOR premise is the one containing the conclusion's PREDICATE, the
    # MINOR the one containing its SUBJECT -- not simply "the first line".
    # "All cats are mammals; All mammals are animals; Therefore all cats are
    # animals" is written minor-first and was being read as figure 4.
    (a_s, a_p), (b_s, b_p), (cs, cp) = ts
    a_has_P = cp in (a_s, a_p)
    b_has_P = cp in (b_s, b_p)
    if a_has_P and not b_has_P:
        (ms, mp), (ns, np_) = (a_s, a_p), (b_s, b_p)
        m_mood, n_mood = moods[0], moods[1]
    elif b_has_P and not a_has_P:
        (ms, mp), (ns, np_) = (b_s, b_p), (a_s, a_p)
        m_mood, n_mood = moods[1], moods[0]
    else:
        # ambiguous: both or neither contain P -> refuse rather than guess
        return None

    # find the middle term = the term shared by the two premises
    middles = ({ms, mp} & {ns, np_}) - {cs, cp}
    if len(middles) != 1:
        return None
    M = middles.pop()

    # Figure is where M sits in each premise:
    #   fig1 M-P / S-M  -> major subject M, minor predicate M
    #   fig2 P-M / S-M  -> major predicate M, minor predicate M
    #   fig3 M-P / M-S  -> major subject M, minor subject M
    #   fig4 P-M / M-S  -> major predicate M, minor subject M
    # (The first version had these four cases in the wrong order and parsed
    #  Barbara as figure 4.)
    if ms == M and np_ == M:
        figure = 1
    elif mp == M and np_ == M:
        figure = 2
    elif ms == M and ns == M:
        figure = 3
    elif mp == M and ns == M:
        figure = 4
    else:
        return None

    return {"mood": f"{m_mood}{n_mood}{moods[2]}", "figure": figure,
            "terms": {"S": cs, "P": cp, "M": M},
            "premises": premises, "conclusion": concl}


# --------------------------------------------------------------------------
# Self-test: the rules MUST reproduce the 24-form table exactly
# --------------------------------------------------------------------------

def run_self_test() -> bool:
    print("syllogism.py - SELF TEST")
    ok = True

    # 1. rules vs table over all 256 forms -- this is the real test
    mismatches = []
    for figure in FIGURES:
        for mood in ("".join(c) for c in itertools.product(MOODS, repeat=3)):
            rule = is_valid(mood, figure)
            table = mood in VALID[figure]
            if rule != table:
                mismatches.append((mood, figure, rule, table))
    good = not mismatches
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] rules reproduce the 24-form table over "
          f"all 256 forms (mismatches={len(mismatches)})")
    for m in mismatches[:6]:
        print(f"      MISMATCH mood={m[0]} fig={m[1]} rules={m[2]} table={m[3]}")

    # 2. total count
    total = sum(1 for f in FIGURES
                for m in ("".join(c) for c in itertools.product(MOODS, repeat=3))
                if is_valid(m, f))
    good = total == 24
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] exactly 24 valid forms (got {total})")

    # 3. named classics
    for mood, fig, name in (("AAA", 1, "Barbara"), ("EAE", 1, "Celarent"),
                            ("AII", 1, "Darii"), ("EIO", 1, "Ferio"),
                            ("AOO", 2, "Baroco"), ("OAO", 3, "Bocardo")):
        e = explain(mood, fig)
        good = e["valid"] and e["name"] == name
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<10} {mood}-{fig} valid={e['valid']}")

    # 4. famous fallacies must be caught WITH a reason
    for mood, fig, label in (("AAA", 2, "undistributed middle"),
                             ("AEE", 1, "illicit major"),
                             ("IIA", 3, "two particular premises")):
        e = explain(mood, fig)
        good = (not e["valid"]) and len(e["violations"]) > 0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<26} {mood}-{fig} "
              f"-> {e['violations'][:1]}")

    # 5. natural language: a valid Barbara
    p = parse_argument("All men are mortal; All greeks are men; Therefore all greeks are mortal.")
    good = p is not None and p["mood"] == "AAA" and p["figure"] == 1 and is_valid("AAA", 1)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] NL Barbara parsed -> "
          f"{p['mood'] + '-' + str(p['figure']) if p else None}")

    # 6. the cron-job text must NOT be forced into a syllogism
    p2 = parse_argument("Cron jobs on the same runner start the same reindex task "
                        "and corrupt the temp tables.")
    good = p2 is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] non-argument returns None (no forced form)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Categorical syllogism validity checker.")
    ap.add_argument("--mood")
    ap.add_argument("--figure", type=int)
    ap.add_argument("--check")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if args.mood and args.figure:
        import json
        print(json.dumps(explain(args.mood.upper(), args.figure), indent=2))
        return 0
    if args.check:
        import json
        p = parse_argument(args.check)
        if not p:
            print(json.dumps({"parsed": False,
                              "why": "not a categorical syllogism shape; refusing to guess"},
                             indent=2))
            return 2
        out = explain(p["mood"], p["figure"])
        out["terms"] = p["terms"]
        print(json.dumps(out, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
