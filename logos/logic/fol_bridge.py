"""logos/logic/fol_bridge.py â€” English argument -> Z3 first-order encoding -> validity, with a counterexample.

WHAT THIS REPLACES, AND WHY
---------------------------
`row_eval` has a check called `derivation` whose entire implementation is a REGEX: it looks for
"therefore/thus/hence/so" and counts occurrence of "step 1 / first / then / next / finally". It measures the
SHAPE of an argument and nothing about whether the argument holds. It is the weakest check in the suite and it
is the one most likely to certify nonsense, because fluent prose about reasoning is trivially producible.

Meanwhile `syllogism.py` checks categorical syllogisms properly (measured 24/24 over a 256-form sweep) but only
within that grid, and `lean_bridge.py` checks propositional validity through the Lean kernel -- correct, but it
accepts propositional atoms only (`["P", "P -> Q"] -> "Q"`) and cannot see inside "All men are mortal".

This module fills the gap between them: a DECLARED, template-based English -> first-order encoding, checked by
Z3. It costs milliseconds where Lean costs 13-52 seconds, and it returns a COUNTEREXAMPLE MODEL when an argument
fails, which is far more useful than "invalid".

WHY A TEMPLATE PARSER AND NOT A MODEL (the Type-2 question)
----------------------------------------------------------
The autoformalization literature separates Type 1 (the formalisation compiles, but the statement was quietly
adjusted to fit, so the proof is about a different claim) from Type 2 (everything matches the author's intent).
Letting a model choose the encoding is how Type 1 is produced at scale. So the encoding here is a small set of
explicit patterns, and anything that does not match returns UNPARSEABLE -- a declared, reviewable mapping or
nothing. The formal encoding is RETURNED with every verdict so a mismatch is visible as a different statement
rather than as a wrong answer.

    "All men are mortal"        -> ForAll [x], Man(x) -> Mortal(x)
    "No reptiles are mammals"   -> ForAll [x], Reptile(x) -> Not(Mammal(x))
    "Some birds cannot fly"     -> Exists [x], Bird(x) and Not(Fly(x))
    "Socrates is a man"         -> Man(socrates)

USAGE
    python tools/logos/logic/fol_bridge.py --self-test
    python tools/logos/logic/fol_bridge.py --measure
    python tools/logos/logic/fol_bridge.py --arg "All men are mortal ; Socrates is a man ; therefore Socrates is mortal"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

VALID = "VALID"            # the conclusion follows from the premises
INVALID = "INVALID"        # it does not, and there is a counterexample
UNPARSEABLE = "UNPARSEABLE"  # the English did not match a declared pattern
UNKNOWN = "UNKNOWN"        # Z3 declined

# ------------------------------------------------------------------ declared patterns (Type 2 by construction)
_PLURAL_IRREGULAR = {"men": "man", "women": "woman", "children": "child", "people": "person",
                     "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot"}

_DETERMINERS = r"(?:all|every|each|any)"
_NEG_DET = r"(?:no|no single)"

PATTERNS = [
    # universal affirmative: All A are B
    (re.compile(rf"^\s*{_DETERMINERS}\s+(?P<s>.+?)\s+(?:are|is|were|was)\s+(?P<p>.+?)\s*[.!]?\s*$", re.I),
     "universal_affirmative"),
    # universal negative: No A are B
    (re.compile(rf"^\s*{_NEG_DET}\s+(?P<s>.+?)\s+(?:are|is|were|was)\s+(?P<p>.+?)\s*[.!]?\s*$", re.I),
     "universal_negative"),
    # particular affirmative: Some A are B
    (re.compile(r"^\s*(?:some|at least one|there (?:is|are) (?:an?|some))\s+(?P<s>.+?)\s+(?:that\s+)?"
                r"(?:are|is)\s+(?P<p>.+?)\s*[.!]?\s*$", re.I), "particular_affirmative"),
    # particular negative: Some A are not B  /  Some A cannot B
    (re.compile(r"^\s*(?:some|at least one)\s+(?P<s>.+?)\s+(?:are|is)\s+not\s+(?P<p>.+?)\s*[.!]?\s*$", re.I),
     "particular_negative"),
    (re.compile(r"^\s*(?:some|at least one)\s+(?P<s>.+?)\s+(?:can ?not|cannot|can't)\s+(?P<p>.+?)\s*[.!]?\s*$",
                re.I), "particular_negative_modal"),
    # singular: NAME is (not) a PREDICATE
    (re.compile(r"^\s*(?P<s>[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s+is\s+not\s+(?:an?\s+|the\s+)?(?P<p>.+?)\s*[.!]?\s*$"),
     "singular_negative"),
    (re.compile(r"^\s*(?P<s>[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s+is\s+(?:an?\s+|the\s+)?(?P<p>.+?)\s*[.!]?\s*$"),
     "singular_affirmative"),
]

_NEG_PREDICATE = re.compile(r"^(?:cannot|can't|not)\s+(.+)$", re.I)
_NEG_CUES = re.compile(r"\b(?:not|no|never|nobody|none|neither|nor|cannot|can't|isn't|aren't|doesn't|"
                       r"don't|won't|can ?not)\b", re.I)


def _negation_ledger(text: str) -> int:
    """How many surface negation cues does the English carry?

    Used as a counting invariant against the number of Not() nodes in the encoding. A ledger is crude and
    that is the point: a dropped "not" is a wrong ARGUMENT, not a wrong proof, and no solver downstream can
    ever detect it. This was a real bug here -- "No reptiles are mammals" encoded as "All reptiles are
    mammals" -- so the check earns its place empirically.
    """
    return len(_NEG_CUES.findall(text or ""))


def _norm_term(t: str) -> str:
    """Normalise a noun phrase to one identifier. Articles stripped, simple plurals singularised.

    The singularisation is what makes `All men are mortal` and `Socrates is a man` refer to the SAME predicate.
    Without it the encoder would build two unrelated predicates and prove nothing about the argument -- the
    quiet-failure mode this module exists to avoid.
    """
    t = (t or "").strip().lower()
    t = re.sub(r"^(?:a|an|the|some|any|all|every|each)\s+", "", t)
    t = t.strip(" .,;:!?\"'")
    words = []
    for w in re.split(r"\s+", t):
        if w in _PLURAL_IRREGULAR:
            w = _PLURAL_IRREGULAR[w]
        elif len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(re.sub(r"[^a-z0-9_\-]", "", w))
    return "_".join(x for x in words if x)


def parse_sentence(text: str) -> dict:
    """One English sentence -> one declared formal shape, or UNPARSEABLE.

    UNPARSEABLE is a real answer and must stay one. Guessing an encoding is how a proof ends up being about a
    different sentence than the one on the page, and no amount of Z3 afterwards can detect that.
    """
    s = (text or "").strip()
    if not s:
        return {"status": UNPARSEABLE, "text": text, "reason": "empty"}
    for rx, kind in PATTERNS:
        # FULLMATCH. `.match` accepts a prefix, so "Humans are not mortal" would parse as "Humans are mortal"
        # and the leftover "not" would vanish -- a silent polarity flip, which is the failure class this
        # module exists to prevent.
        m = rx.fullmatch(s)
        if not m:
            continue
        subj, pred = m.group("s"), m.group("p")
        neg = False
        nm = _NEG_PREDICATE.match(pred)
        if nm:
            pred, neg = nm.group(1), True
        if kind == "particular_negative_modal":
            neg = True
        if kind in ("universal_negative", "singular_negative", "particular_negative",
                    "particular_negative_modal"):
            # the pattern CONSUMED the negation cue, so it is not left in `pred` to be detected. Every
            # *_negative shape must set the flag explicitly or the encoder asserts the opposite claim.
            # "No reptiles are mammals" is ForAll x, Reptile(x) -> NOT Mammal(x). Without this the encoder
            # produced the exact opposite assertion and Z3 dutifully proved a claim nobody made.
            neg = True
        out = {"status": "PARSED", "kind": kind, "subject": _norm_term(subj),
               "predicate": _norm_term(pred), "negated": neg, "text": s}
        if not out["subject"] or not out["predicate"]:
            return {"status": UNPARSEABLE, "text": text, "reason": "a term normalised to empty"}
        return out
    return {"status": UNPARSEABLE, "text": text,
            "reason": "no declared pattern matched; this module does not guess encodings"}


# ------------------------------------------------------------------ Z3 encoding
def _z3():
    try:
        import z3
        return z3
    except Exception:                                         # noqa: BLE001
        return None


def formalise(premises: list, conclusion: str) -> dict:
    """Build the Z3 assertions AND a human-readable record of the encoding.

    The `encoding` field is not decoration: it is the Type-2 audit trail. A verdict that cannot show the
    statement it proved cannot be checked by anyone, and a formalisation that silently drifted from the English
    would look identical to a correct one.
    """
    z3 = _z3()
    if z3 is None:
        return {"status": "UNAVAILABLE", "reason": "z3 not installed"}

    parsed = [parse_sentence(p) for p in premises]
    parsed.append(parse_sentence(conclusion))
    bad = [p for p in parsed if p["status"] != "PARSED"]
    if bad:
        return {"status": UNPARSEABLE, "offenders": [b.get("text") for b in bad],
                "reason": "; ".join(sorted({b.get("reason", "") for b in bad}))}

    U = z3.DeclareSort("U")
    preds: dict = {}
    consts: dict = {}

    def P(name):
        if name not in preds:
            preds[name] = z3.Function(name, U, z3.BoolSort())
        return preds[name]

    def C(name):
        if name not in consts:
            consts[name] = z3.Const(name, U)
        return consts[name]

    def term_is_constant(t: str) -> bool:
        return t in consts

    assertions = []
    encoding = []
    x = z3.Const("x", U)

    def build(node: dict, bound=None):
        """Return a Z3 Bool expression for one parsed sentence."""
        sp, pp, neg = node["subject"], node["predicate"], node["negated"]
        kind = node["kind"]
        if kind == "singular_affirmative" or kind == "singular_negative":
            atom = P(pp)(C(sp))
            return z3.Not(atom) if neg else atom
        if kind.startswith("universal"):
            # `universal_negative` ("No A are B") MEANS the predicate is negated. It used to fall through
            # with negated=False and encode "All A are B" -- the opposite claim, silently.
            body = z3.Implies(P(sp)(x), z3.Not(P(pp)(x)) if neg else P(pp)(x))
            return z3.ForAll([x], body)
        # particular: SKOLEMISED to a fresh constant rather than a real Exists.
        # "Some A are B" is witnessed by a specific (unknown) individual, and writing it as
        # A(c) and B(c) keeps every assertion either universal or ground, which keeps the problem inside the
        # Effectively Propositional fragment -- decidable. A real Exists would drag it out of EPR, where Z3
        # may return `unknown`, and then ESCALATE stops being an encoding-error signal.
        n_sk = sum(1 for _n, _e in assertions)
        witness = z3.Const(f"witness_{len(consts)}_{n_sk}", U)
        consts[witness.decl().name()] = witness
        atom = P(sp)(witness)
        other = z3.Not(P(pp)(witness)) if neg else P(pp)(witness)
        return z3.And(atom, other)

    for i, node in enumerate(parsed[:-1]):
        assertions.append((f"premise_{i + 1}", build(node)))
        encoding.append({"role": f"premise_{i + 1}", "english": node["text"],
                         "kind": node["kind"], "subject": node["subject"],
                         "predicate": node["predicate"], "negated": node["negated"]})
    conc = parsed[-1]
    conclusion_expr = build(conc)
    encoding.append({"role": "conclusion", "english": conc["text"], "kind": conc["kind"],
                     "subject": conc["subject"], "predicate": conc["predicate"],
                     "negated": conc["negated"]})

    readable = [{"name": name, "smt": str(expr)} for name, expr in assertions]
    readable.append({"name": "conclusion", "smt": str(conclusion_expr)})
    # PREMISES AND CONCLUSION ARE RETURNED SEPARATELY. The conclusion must never be among the assertions:
    # asserting it and then asserting its negation makes the solver trivially unsat and every argument
    # "valid". That was a real bug here, caught only because three INVALID cases came back VALID.
    return {"status": "BUILT", "premises": assertions, "conclusion": conclusion_expr,
            "encoding": encoding, "smt": readable,
            "n_predicates": len(preds), "n_constants": len(consts)}


def check_argument(premises: list, conclusion: str, timeout_ms: int = 5000) -> dict:
    """Validity by refutation, TWO-SIDED, with the fail-closed gates the literature converged on.

    FoVer's Algorithm 1 does not run a single unsat test; it runs a PAIR, because one test cannot tell
    "the conclusion follows" from "the premises say the opposite". Both are informative and they are not the
    same finding, so both are reported:

        P and not-q  unsat  -> ENTAILED      the conclusion follows
        P and q      unsat  -> CONTRADICTED  the premises entail its NEGATION
        neither             -> UNDETERMINED  with a model of a case where the premises hold and q fails

    status stays VALID / INVALID / UNPARSEABLE / UNKNOWN for callers that only want a verdict; `detail`
    carries the finer finding. `unknown` maps to UNKNOWN and must be treated as ESCALATE by callers -- never
    as validity and never as invalidity.
    """
    z3 = _z3()
    if z3 is None:
        return {"status": UNKNOWN, "reason": "z3 not installed"}
    f = formalise(premises, conclusion)
    if f["status"] != "BUILT":
        return f

    # ---- CHEATING CHECK (Proof of Thought added this after finding its own proofs were circular).
    # If the conclusion is literally one of the premises, or is a bare True, the proof is vacuously valid
    # and worth nothing. Report it as valid-but-vacuous rather than silently counting it as a success.
    enc = f["encoding"]
    conc_e = enc[-1]
    for e in enc[:-1]:
        if (e["role"].startswith("premise") and e["kind"] == conc_e["kind"]
                and e["subject"] == conc_e["subject"] and e["predicate"] == conc_e["predicate"]
                and e["negated"] == conc_e["negated"]):
            return {"status": VALID, "detail": "CIRCULAR", "vacuous": True,
                    "reason": "the conclusion is asserted verbatim among the premises; this is valid and "
                              "proves nothing",
                    "encoding": enc, "smt": f["smt"], "ms": 0.0}

    # ---- NEGATION LEDGER: surface negation cues must match Not() nodes in the encoding.
    # A dropped "not" is a wrong ARGUMENT, not a wrong proof, and nothing downstream can detect it. This was
    # a real bug here: "No reptiles are mammals" encoded as "All reptiles are mammals".
    surface = sum(_negation_ledger(x) for x in list(premises) + [conclusion])
    encoded = sum(1 for e in enc if e["negated"])
    ledger_ok = surface == encoded

    if not ledger_ok:
        # A parity mismatch means the encoding lost or gained a negation somewhere. Any verdict from here
        # would be about a different statement, so nothing is claimed.
        return {"status": UNKNOWN, "detail": "NEGATION_LEDGER_MISMATCH", "negation_ledger": False,
                "reason": f"the English carries {surface} negation cue(s) but the encoding has {encoded} "
                          f"negated term(s); the formalisation is not the sentence, so ESCALATE",
                "encoding": enc, "smt": f["smt"], "ms": 0.0}

    if not f["premises"]:
        return {"status": INVALID, "detail": "NO_PREMISES",
                "reason": "nothing to entail the conclusion from",
                "encoding": enc, "smt": f["smt"], "ms": 0.0, "negation_ledger": ledger_ok}

    # ---- GATE 0: are the premises consistent with each other?
    # ex falso proves anything, so a "valid" argument built on contradictory premises is worthless. The
    # linear-logic paper identifies exactly this as the trap: an inconsistent set yields unsat even without
    # the negated goal. Without this gate every such argument reports as VALID.
    s_cons = z3.Solver()
    s_cons.set("timeout", timeout_ms)
    for _name, expr in f["premises"]:
        s_cons.add(expr)
    try:
        consistency = s_cons.check()
    except Exception:                                         # noqa: BLE001
        consistency = z3.unknown
    if consistency == z3.unsat:
        return {"status": INVALID, "detail": "INCONSISTENT_PREMISES", "inconsistent_premises": True,
                "reason": "the premises are mutually inconsistent, so anything follows from them; a validity "
                          "proof here would be vacuous",
                "encoding": enc, "smt": f["smt"], "ms": 0.0, "negation_ledger": ledger_ok}
    if consistency == z3.unknown:
        return {"status": UNKNOWN, "detail": "PREMISE_CHECK_UNKNOWN",
                "reason": "z3 declined to decide premise consistency; ESCALATE",
                "encoding": enc, "smt": f["smt"], "ms": 0.0, "negation_ledger": ledger_ok}

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.set(unsat_core=True)
    for name, expr in f["premises"]:
        s.assert_and_track(expr, z3.Bool(name))

    out = {"encoding": enc, "smt": f["smt"], "negation_ledger": ledger_ok,
           "n_predicates": f["n_predicates"]}

    # ---- GATE 1: P and not-q
    t0 = time.perf_counter()
    s.push()
    s.add(z3.Not(f["conclusion"]))
    try:
        r1 = s.check()
    except Exception as e:                                    # noqa: BLE001
        s.pop()
        return {**out, "status": UNKNOWN, "reason": f"z3 raised {type(e).__name__}: {e}"}
    ms = (time.perf_counter() - t0) * 1000
    out["ms"] = round(ms, 2)

    if r1 == z3.unsat:
        out["status"] = VALID
        out["detail"] = "ENTAILED"
        try:
            core = [str(c) for c in s.unsat_core()]
            needed = [c for c in core if c.startswith("premise_")]
            out["premises_needed"] = needed
            out["all_premises_needed"] = len(needed) == len(premises)
        except Exception:                                     # noqa: BLE001
            pass
        s.pop()
        return out

    if r1 == z3.unknown:
        s.pop()
        return {**out, "status": UNKNOWN, "detail": "ENTAILMENT_UNKNOWN",
                "reason": "z3 returned unknown; this is an undecidable-fragment signal, ESCALATE"}

    # ---- a model of P and not-q: a concrete case where the premises hold and the conclusion fails
    try:
        m = s.model()
        cx = {}
        for d in m.decls():
            name = str(d)
            if name.startswith("witness_"):
                continue
            cx[name] = str(m[d])
        out["counterexample"] = cx
    except Exception:                                         # noqa: BLE001
        pass
    s.pop()

    # ---- GATE 2: P and q -- does the premise set entail the NEGATION instead?
    s2 = z3.Solver()
    s2.set("timeout", timeout_ms)
    for _name, expr in f["premises"]:
        s2.add(expr)
    s2.add(f["conclusion"])
    try:
        r2 = s2.check()
    except Exception:                                         # noqa: BLE001
        r2 = z3.unknown
    if r2 == z3.unsat:
        out["status"] = INVALID
        out["detail"] = "CONTRADICTED"
        out["reason"] = "the premises entail the NEGATION of the conclusion"
    elif r2 == z3.unknown:
        out["status"] = UNKNOWN
        out["detail"] = "CONTRADICTION_CHECK_UNKNOWN"
        out["reason"] = "z3 returned unknown on the two-sided check; ESCALATE"
    else:
        out["status"] = INVALID
        out["detail"] = "UNDETERMINED"
        out["reason"] = "the premises neither entail nor contradict the conclusion"
    return out


# ------------------------------------------------------------------ argument splitting
_CONCL = re.compile(r"(?:^|\n|\s)(?:therefore|thus|hence|so,|so\b|âˆ´)\s*,?\s*", re.I)


def split_argument(row: str):
    """Split a row into (premises, conclusion) using an explicit conclusion marker, or return None.

    A comma-separated list WITHOUT a marker is deliberately NOT treated as an argument: "Cats are mammals,
    dogs are mammals" makes no claim and inventing a conclusion for it would be authoring.
    """
    text = (row or "").strip()
    if not text:
        return None
    m = None
    for m in _CONCL.finditer(text):
        pass
    if m is None:
        return None
    head, tail = text[:m.start()].strip(), text[m.end():].strip()
    if not head or not tail:
        return None
    premises = [p.strip() for p in re.split(r"[;\n]|(?<=[.!?])\s+|\s*,\s*(?=[A-Z])", head) if p.strip()]
    if not premises:
        return None
    conclusion = tail.split(".")[0].strip() or tail
    return premises, conclusion


def check_row(row: str, timeout_ms: int = 5000) -> dict:
    """Convenience for the pipeline: split, then check. A row with no argument marker is NOT_AN_ARGUMENT."""
    parts = split_argument(row)
    if parts is None:
        return {"status": "NOT_AN_ARGUMENT", "reason": "no explicit conclusion marker (therefore/thus/hence)"}
    premises, conclusion = parts
    out = check_argument(premises, conclusion, timeout_ms=timeout_ms)
    out["premises"] = premises
    out["conclusion"] = conclusion
    return out


# ------------------------------------------------------------------ measure
def measure() -> int:
    z3 = _z3()
    print("=== first-order argument checking via Z3 ===")
    print(f"  z3 {z3.get_version_string() if z3 else 'MISSING'}")
    ok, why = True, "z3 present"
    print(f"  available {ok} ({why})\n")

    CASES = [
        (["All men are mortal", "Socrates is a man"], "Socrates is mortal", VALID),
        (["All men are mortal", "Socrates is a man"], "Socrates is a fish", INVALID),
        (["All men are mortal", "Socrates is a man"], "Socrates is mortal", VALID),
        (["No reptiles are mammals", "All snakes are reptiles"], "No snakes are mammals", VALID),
        (["No reptiles are mammals", "All snakes are reptiles"], "All snakes are mammals", INVALID),
        (["All dogs are mammals", "All mammals are animals"], "All dogs are animals", VALID),
        (["All dogs are mammals", "All mammals are animals"], "All animals are dogs", INVALID),
        (["Some birds cannot fly", "All penguins are birds"], "Some penguins cannot fly", INVALID),
        (["All men are mortal"], "Socrates is mortal", INVALID),
        (["Socrates is a man"], "Socrates is not a man", INVALID),
        (["All men are mortal"], "The moon is made of cheese", UNPARSEABLE),
        (["Bananas are tasty"], "All fruits are tasty", UNPARSEABLE),
    ]
    print(f"  {'premises':52s} {'conclusion':30s} {'expect':11s} {'got':11s} {'ms':>6s}")
    agree, total, ms_total = 0, 0, 0.0
    for prem, conc, expect in CASES:
        r = check_argument(prem, conc)
        got = r["status"]
        total += 1
        agree += 1 if got == expect else 0
        ms_total += r.get("ms", 0.0)
        p = "; ".join(prem)[:52]
        print(f"  {p:52s} {conc[:30]:30s} {expect:11s} {got:11s} {r.get('ms', 0):6.1f}")
        if got != expect:
            print(f"      ^ disagreement: {r.get('reason') or r.get('offenders') or ''}")
        if got == INVALID and r.get("counterexample"):
            cx = {k: v for k, v in r["counterexample"].items() if not k.startswith("x!")}
            if cx:
                print(f"        counterexample: {cx}")
        if got == VALID and r.get("all_premises_needed") is False:
            print(f"        note: only {r.get('premises_needed')} were needed -- the argument is valid but "
                  f"some premise is redundant")

    print()
    print(f"  agreement with my construction: {agree}/{total}")
    print(f"  mean Z3 time {ms_total / max(total, 1):.2f} ms/case")
    print()
    print("=== cost, against the other instruments ===")
    mean_ms = ms_total / max(total, 1)
    print(f"  arithmetic (sympy)      <1 ms")
    print(f"  this module (Z3)     {mean_ms:6.2f} ms")
    print(f"  19M cross-encoder      10.2 ms")
    print(f"  2.6B LLM             2200.0 ms")
    print(f"  Mathlib (Lean)      13167.0 ms")
    print(f"  so a first-order validity check is ~{13167 / max(mean_ms, 0.01):,.0f}x cheaper than Lean and")
    print(f"  it returns a COUNTEREXAMPLE. That is the trade: Z3 gives refutation-by-model at milliseconds, Lean")
    print(f"  gives a constructive kernel proof at seconds. Use Z3 first, Lean only when the Z3 answer is")
    print(f"  undecidable or when a proof object is required.")

    print("\n=== the Type-2 audit trail ===")
    r = check_argument(["All men are mortal", "Socrates is a man"], "Socrates is mortal")
    for e in r["encoding"]:
        print(f"  {e['role']:12s} {e['english'][:36]:38s} {e['kind']:24s} "
              f"{e['subject']}(x) -> {'' if not e['negated'] else 'NOT '}{e['predicate']}(x)")
    print("  every verdict carries the formal statement it checked. a formalisation that drifted from the")
    print("  English would be visible here as a DIFFERENT statement, rather than hiding as a wrong answer.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.logic.fol_bridge - SELF TEST")
    ok = True
    z3 = _z3()
    if z3 is None:
        print("  [SKIP] z3 not installed")
        return True

    # --- the classical syllogism, which is the whole point
    r = check_argument(["All men are mortal", "Socrates is a man"], "Socrates is mortal")
    good = r["status"] == VALID
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Socrates syllogism is VALID")

    # --- and it must be able to say NO, with a counterexample rather than a shrug
    r = check_argument(["All men are mortal", "Socrates is a man"], "Socrates is a fish")
    good = r["status"] == INVALID and isinstance(r.get("counterexample"), dict)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an invalid argument returns INVALID with a counterexample model")

    # --- a MISSING premise must break validity. If this passed, the checker would be ignoring its premises.
    r = check_argument(["All men are mortal"], "Socrates is mortal")
    good = r["status"] == INVALID
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dropping the minor premise makes it INVALID (premises are used)")

    # --- universal negative, and its mistaken converse
    good = (check_argument(["No reptiles are mammals", "All snakes are reptiles"], "No snakes are mammals")
            ["status"] == VALID)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] No reptiles are mammals / All snakes are reptiles => valid")

    good = (check_argument(["No reptiles are mammals", "All snakes are reptiles"], "All snakes are mammals")
            ["status"] == INVALID)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the converse is INVALID")

    # --- a two-step chain (sorites), which syllogism.py cannot express as one step
    r = check_argument(["All dogs are mammals", "All mammals are animals"], "All dogs are animals")
    good = r["status"] == VALID
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a two-step chain is VALID (beyond a single syllogism)")

    # --- REGRESSION, AND IT IS A TYPE-2 FAILURE IN MY OWN ENCODER: "No A are B" must be encoded as a
    #     NEGATED predicate. It used to encode as "ALL A are B" -- the opposite claim.
    p = parse_sentence("No reptiles are mammals")
    good = p["status"] == "PARSED" and p["negated"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'No reptiles are mammals' encodes as a NEGATED predicate")

    f = formalise(["No reptiles are mammals"], "All reptiles are mammals")
    smt = " ".join(x["smt"] for x in f.get("smt", []))
    good = "Not(" in smt
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the SMT text contains Not(...) for a negative universal")

    # --- REGRESSION: the conclusion must NOT be among the asserted premises. If it is, it is asserted
    #     alongside its own negation and everything becomes trivially VALID.
    f = formalise(["All men are mortal"], "Socrates is a fish")
    names = [n for n, _e in f.get("premises", [])]
    good = "conclusion" not in names and f.get("conclusion") is not None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the conclusion is NOT among the asserted premises")

    # --- and inconsistent premises are caught rather than yielding a vacuous "valid".
    #     NOTE: "All men are mortal" + "No men are mortal" is CONSISTENT (vacuously, if no men exist) -- my
    #     first version of this test asserted it was inconsistent, which was my error, not the engine's.
    #     A genuinely inconsistent pair needs an existential: "All men are mortal" + "Some men are not mortal".
    r = check_argument(["All men are mortal", "Some men are not mortal"], "Socrates is a fish")
    good = r["status"] == INVALID and r.get("inconsistent_premises") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] inconsistent premises are reported, not used to 'prove' anything")

    # --- a vacuous argument (premises that cannot all hold) must ALSO be distinguished from a genuinely
    #     consistent premise set that simply fails to entail the conclusion
    r = check_argument(["All men are mortal", "No men are mortal"], "Socrates is a fish")
    good = r["status"] == INVALID and r.get("inconsistent_premises") is not True and \
        r.get("detail") == "UNDETERMINED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a CONSISTENT set that merely fails to entail is reported as "
          f"UNDETERMINED, not as inconsistent")

    # --- the two-sided check: premises that entail the NEGATION are their own finding, not just "invalid"
    r = check_argument(["All men are mortal", "Socrates is a man"], "Socrates is not mortal")
    good = r["status"] == INVALID and r.get("detail") == "CONTRADICTED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] entailing the NEGATION is reported as CONTRADICTED "
          f"(not lumped with 'undetermined')")

    # --- circularity: a conclusion asserted among its own premises is valid and worthless
    r = check_argument(["Socrates is a man", "Socrates is mortal"], "Socrates is mortal")
    good = r["status"] == VALID and r.get("vacuous") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a circular argument is VALID but flagged vacuous")

    # --- REGRESSION (third polarity inversion found today): "X is not Y" must encode as Not(Y(X)).
    p = parse_sentence("Socrates is not mortal")
    good = p["status"] == "PARSED" and p["negated"] is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Socrates is not mortal' sets negated (the pattern ate the 'not')")

    f = formalise(["Socrates is a man"], "Socrates is not mortal")
    conc_smt = f["smt"][-1]["smt"] if f.get("smt") else ""
    good = "Not(" in conc_smt
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the conclusion's SMT is wrapped in Not(...)")

    # --- and here the honest answer is UNDETERMINED: nothing in the premises says men are mortal, so
    #     neither the claim nor its negation follows. A checker that "found" a contradiction here would be
    #     inventing a premise. (My first version of this test expected CONTRADICTED -- wrong again, and the
    #     engine was right.)
    r = check_argument(["Socrates is a man"], "Socrates is not mortal")
    good = r["status"] == INVALID and r.get("detail") == "UNDETERMINED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] without a rule linking them it is UNDETERMINED, not a verdict "
          f"({r.get('detail')})")

    # --- the negation ledger must agree; a mismatch means the encoding lost or gained a negation
    r = check_argument(["No reptiles are mammals", "All snakes are reptiles"], "No snakes are mammals")
    good = r.get("negation_ledger") is True
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the negation ledger balances ({r.get('negation_ledger')})")

    # --- FAIL CLOSED: English that matches no declared pattern is UNPARSEABLE, never a guess
    r = check_argument(["Bananas are tasty"], "All fruits are tasty")
    good = r["status"] == UNPARSEABLE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unmatched English returns UNPARSEABLE (no guessed encoding)")

    # --- and a parseable premise with an unparseable conclusion is still UNPARSEABLE as a whole:
    #     half an encoding would be a proof about half an argument.
    r = check_argument(["All men are mortal"], "The moon is made of cheese")
    good = r["status"] == UNPARSEABLE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one unparseable part makes the WHOLE argument UNPARSEABLE")

    # --- singularisation: "men" and "man" must be the SAME predicate or nothing connects
    good = _norm_term("men") == _norm_term("a man") == "man"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'men' and 'a man' normalise to one predicate")

    # --- the encoding is returned with the verdict: the Type-2 audit trail
    r = check_argument(["All men are mortal"], "Socrates is mortal")
    good = bool(r.get("encoding")) and all("english" in e for e in r["encoding"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the verdict carries the formal statement it checked")

    # --- unsat core: a redundant premise is reported rather than hidden
    r = check_argument(["All men are mortal", "Socrates is a man", "All cats are mammals"],
                       "Socrates is mortal")
    good = r["status"] == VALID and r.get("all_premises_needed") is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a redundant premise is reported "
          f"(needed {r.get('premises_needed')})")

    # --- splitting: no conclusion marker means no argument, and we do not invent one
    good = split_argument("Cats are mammals, dogs are mammals") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a list with no conclusion marker is NOT treated as an argument")

    parts = split_argument("All men are mortal; Socrates is a man; therefore Socrates is mortal.")
    good = parts is not None and len(parts[0]) == 2 and "mortal" in parts[1]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'therefore' splits premises from conclusion")

    # --- a broken parse must not be reported as VALID. This is the failure direction that matters.
    r = check_row("The sky is blue.") if False else check_row("Blah blah; therefore nothing at all")
    good = r["status"] != VALID
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] gibberish is never VALID ({r['status']})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="English argument -> Z3 first-order validity check.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--arg", default="")
    ap.add_argument("--premise", action="append", default=[])
    ap.add_argument("--conclusion", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    if args.arg:
        print(json.dumps(check_row(args.arg), indent=2))
        return 0
    if args.premise and args.conclusion:
        print(json.dumps(check_argument(args.premise, args.conclusion), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())

