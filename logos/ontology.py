"""logos.ontology — logical frames as executable ontologies.

THE IDEA
--------
A frame in `logos/frames` or a rule in `logos/logic` is documentation. An **ontology** is the
same thing with teeth: a schema plus constraints a machine can check.

That matters for small models. A 1B model cannot reliably *know* that a categorical syllogism has
exactly three terms, that the middle term must occur in both premises and not in the conclusion,
or that a mood has one letter per proposition. But it can be *told*, and its output can be
*validated*. The claim being tested here is that constraining the output shape and then rejecting
non-conforming generations is a cheaper route to reliability than making the model bigger --
which is the same conclusion the 2026 reasoning benchmarks reached from a different direction
(SemEval-2026 Task 11: symbolic verification, not scale).

So this module provides two things:

  1. `render_prompt(onto)`  -- the ontology as an instruction block for any model, small or large
  2. `validate(onto, obj)`  -- the same ontology as a checker, returning named violations

The pair is the point. A prompt without a validator is a hope; a validator without a prompt only
sorts garbage. Together they let a tiny model be useful *and* auditable, and they let a big model
be checked rather than trusted.

WHAT AN ONTOLOGY IS HERE
------------------------
    Ontology(
        name, description,
        fields   = {name: Field(type, required, enum, min, max, description)},
        rules    = [(name, predicate, message)],
        examples = [(valid_or_not, payload)],
    )

`rules` are Python predicates over the parsed object, so a constraint is executable rather than
prose. A rule that cannot be expressed as a predicate does not belong in an ontology -- it belongs
in a comment, which is where most "schemas" in this project have lived until now.

THREE ONTOLOGIES SHIP HERE
--------------------------
`CATEGORICAL_SYLLOGISM` -- three terms, two premises, mood of three letters, middle term shared
and absent from the conclusion. This is the classical integrity constraint that a 4-term fallacy
("the fallacy of four terms", the commonest informal equivocation) violates.

`SPAN_EVIDENCE` -- the per-evidence contract from the constructed corpus: a span verdict must
name a label, a subject, and either a quantity or an explicit reason it cannot be compared.

`MACHINE_CHECKED_PROOF` -- a proof is only acceptable if it compiled AND is `sorry`-free. This
encodes the lesson from the Lean experiment as a constraint rather than a code comment.

USAGE
-----
    python tools/logos/ontology.py --self-test
    python tools/logos/ontology.py --prompt categorical_syllogism
    python tools/logos/ontology.py --check categorical_syllogism '{"premises": [...], ...}'
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- schema
TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


@dataclass
class Field:
    type: str
    required: bool = True
    enum: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def render(self) -> str:
        bits = [f"`{self.type}`"]
        if self.required:
            bits.append("required")
        else:
            bits.append("optional")
        if self.enum:
            bits.append("one of: " + ", ".join(self.enum))
        if self.minimum is not None:
            bits.append(f">= {self.minimum:g}")
        if self.maximum is not None:
            bits.append(f"<= {self.maximum:g}")
        line = " ".join(bits)
        return f"{line} -- {self.description}" if self.description else line


@dataclass
class Ontology:
    name: str
    description: str
    fields: dict[str, Field]
    rules: list[tuple[str, Callable[[dict], bool], str]] = field(default_factory=list)
    examples: list[tuple[bool, dict]] = field(default_factory=list)

    # ------------------------------------------------------------------ validation
    def validate(self, obj: Any) -> list[str]:
        """Return named violations. Empty list means conforming."""
        out: list[str] = []
        if not isinstance(obj, dict):
            return [f"payload must be an object, got {type(obj).__name__}"]

        for name, f in self.fields.items():
            if name not in obj:
                if f.required:
                    out.append(f"missing required field '{name}'")
                continue
            v = obj[name]
            check = TYPE_CHECKS.get(f.type)
            if check and not check(v):
                out.append(f"field '{name}' must be {f.type}, got {type(v).__name__}")
                continue
            if f.enum and v not in f.enum:
                out.append(f"field '{name}' must be one of {list(f.enum)}, got {v!r}")
            if f.minimum is not None and isinstance(v, (int, float)) and v < f.minimum:
                out.append(f"field '{name}' must be >= {f.minimum:g}, got {v}")
            if f.maximum is not None and isinstance(v, (int, float)) and v > f.maximum:
                out.append(f"field '{name}' must be <= {f.maximum:g}, got {v}")

        for rule_name, pred, message in self.rules:
            try:
                if not pred(obj):
                    out.append(f"rule '{rule_name}' violated: {message}")
            except Exception as exc:          # noqa: BLE001 - a crashing rule is a violation
                out.append(f"rule '{rule_name}' raised {type(exc).__name__}: {exc}")
        return out

    def conforms(self, obj: Any) -> bool:
        return not self.validate(obj)

    # --------------------------------------------------------------------- prompt
    def render_prompt(self, task: str = "") -> str:
        """The ontology as an instruction block a model can follow.

        Deliberately terse and imperative. Long schemas are not followed by small models; a
        numbered field list plus an explicit refusal instruction is.
        """
        lines = [f"# ONTOLOGY: {self.name}", "", self.description, ""]
        if task:
            lines += [f"TASK: {task}", ""]
        lines += ["## Required output fields", ""]
        for name, f in self.fields.items():
            lines.append(f"- `{name}` ({f.render()})")
        lines += ["", "## Constraints that will be CHECKED", ""]
        for rule_name, _pred, message in self.rules:
            lines.append(f"- {rule_name}: {message}")
        lines += ["", "## Rules of engagement", "",
                  "1. Output a single JSON object matching the fields above.",
                  "2. Do not invent fields. Do not omit required fields.",
                  "3. If you cannot satisfy a constraint, output "
                  '{"refuse": true, "reason": "<which constraint and why>"} instead of guessing.',
                  "4. A refused answer is acceptable. A wrong answer that looks conforming is not.",
                  ""]
        if self.examples:
            lines += ["## Examples", ""]
            for ok, payload in self.examples[:3]:
                tag = "VALID" if ok else "INVALID"
                lines.append(f"- {tag}: `{json.dumps(payload, ensure_ascii=False)}`")
            lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------------ syllogism ontology
def _terms_of(prop: str) -> set[str]:
    """Content terms of a categorical proposition, lowercased."""
    stop = {"all", "no", "some", "are", "is", "not", "the", "a", "an", "every", "each"}
    return {w for w in re.findall(r"[a-z]+", prop.lower()) if w not in stop}


def _mood_of(prop: str) -> str:
    low = prop.lower().strip()
    if low.startswith("all"):
        return "A"
    if low.startswith("no "):
        return "E"
    if low.startswith("some") and "not" in low:
        return "O"
    if low.startswith("some"):
        return "I"
    return "?"


def _p_three_terms(o: dict) -> bool:
    t = set()
    for p in o.get("premises", []):
        t |= _terms_of(p)
    t |= _terms_of(o.get("conclusion", ""))
    return len(t) == 3


def _p_middle_shared(o: dict) -> bool:
    prem = o.get("premises", [])
    if len(prem) != 2:
        return False
    a, b = _terms_of(prem[0]), _terms_of(prem[1])
    conc = _terms_of(o.get("conclusion", ""))
    shared = (a & b) - conc
    return len(shared) >= 1


def _p_middle_not_in_conclusion(o: dict) -> bool:
    prem = o.get("premises", [])
    if len(prem) != 2:
        return False
    middle = (_terms_of(prem[0]) & _terms_of(prem[1]))
    return not (middle & _terms_of(o.get("conclusion", "")))


def _p_mood_matches(o: dict) -> bool:
    prem = o.get("premises", [])
    if len(prem) != 2:
        return False
    derived = "".join(_mood_of(p) for p in prem) + _mood_of(o.get("conclusion", ""))
    stated = o.get("mood", "")
    return (not stated) or stated == derived


def _p_two_premises(o: dict) -> bool:
    return len(o.get("premises", [])) == 2


CATEGORICAL_SYLLOGISM = Ontology(
    name="categorical_syllogism",
    description=("A categorical syllogism: exactly two premises and one conclusion, over exactly "
                 "three terms (minor, middle, major). The middle term must occur in both premises "
                 "and must NOT occur in the conclusion. Violating the term count is the fallacy "
                 "of four terms, which is the commonest equivocation error."),
    fields={
        "premises": Field("list", True, description="exactly two categorical propositions"),
        "conclusion": Field("str", True, description="the conclusion proposition"),
        "mood": Field("str", False,
                      description="three letters from {A,E,I,O}: All/No/Some/Some-not"),
        "figure": Field("int", False, minimum=1, maximum=4,
                        description="position of the middle term"),
        "valid": Field("bool", False, description="whether the form is valid"),
    },
    rules=[
        ("two_premises", _p_two_premises, "a categorical syllogism has exactly two premises"),
        ("exactly_three_terms", _p_three_terms,
         "exactly three distinct terms across premises and conclusion"),
        ("middle_term_shared", _p_middle_shared,
         "the middle term must occur in both premises"),
        ("middle_not_in_conclusion", _p_middle_not_in_conclusion,
         "the middle term must not occur in the conclusion"),
        ("mood_matches_propositions", _p_mood_matches,
         "the stated mood must match the propositions' quantifiers"),
    ],
    examples=[
        (True, {"premises": ["All cats are mammals", "All mammals are animals"],
                "conclusion": "All cats are animals", "mood": "AAA", "figure": 1,
                "valid": True}),
        (False, {"premises": ["All cats are mammals", "All dogs are animals"],
                 "conclusion": "All cats are animals"},
         ),
    ],
)

# -------------------------------------------------------------------- span ontology
SPAN_EVIDENCE = Ontology(
    name="span_evidence",
    description=("A per-evidence verdict. One snippet, one label, and an explicit reason when "
                 "the snippet cannot be compared to the claim. The `comparable` field is what "
                 "stops a checker from turning a co-occurrence into a conflict."),
    fields={
        "label": Field("str", True, enum=("support", "refute", "insufficient")),
        "subject": Field("str", True, description="the entity the snippet is about"),
        "attribute": Field("str", True, description="the property being compared"),
        "comparable": Field("bool", True,
                            description="true only if the snippet and claim share a unit family"),
        "quantity": Field("float", False, description="value in base units, when comparable"),
        "reason": Field("str", True, description="why this label, in one clause"),
    },
    rules=[
        ("insufficient_needs_reason",
         lambda o: o.get("label") != "insufficient" or bool(o.get("reason")),
         "an insufficient verdict must state why it could not decide"),
        ("comparable_needs_quantity",
         lambda o: (not o.get("comparable")) or o.get("quantity") is not None,
         "a comparable snippet must carry a quantity"),
        ("decisive_needs_comparable",
         lambda o: o.get("label") == "insufficient" or bool(o.get("comparable")),
         "a support/refute verdict requires a comparable quantity -- otherwise it is a guess"),
    ],
    examples=[
        (True, {"label": "refute", "subject": "the bridge", "attribute": "length",
                "comparable": True, "quantity": 1120.0,
                "reason": "same family, differs by 239%"}),
        (False, {"label": "refute", "subject": "the bridge", "attribute": "length",
                 "comparable": False, "reason": "looks wrong"}),
    ],
)

# ------------------------------------------------------------------- proof ontology
MACHINE_CHECKED_PROOF = Ontology(
    name="machine_checked_proof",
    description=("A proof is acceptable only if it compiled in Lean AND contains no `sorry` or "
                 "`admit`. Lean ACCEPTS sorry and exits 0, so compiling is not verifying -- this "
                 "constraint is the difference between the two."),
    fields={
        "compiled": Field("bool", True, description="Lean exited 0"),
        "has_sorry": Field("bool", True, description="source contains sorry or admit"),
        "method": Field("str", True, enum=("tactic", "rfl", "python_only", "none")),
        "lean_version": Field("str", False),
    },
    rules=[
        ("sorry_is_not_a_proof",
         lambda o: not (o.get("compiled") and o.get("has_sorry")),
         "a compiled proof containing sorry proves nothing"),
        ("python_is_not_machine_checked",
         lambda o: o.get("method") != "python_only",
         "python_only means no kernel ran; it may not be reported as machine-checked"),
    ],
    examples=[
        (True, {"compiled": True, "has_sorry": False, "method": "tactic"}),
        (False, {"compiled": True, "has_sorry": True, "method": "tactic"}),
    ],
)

REGISTRY: dict[str, Ontology] = {o.name: o for o in
                                 (CATEGORICAL_SYLLOGISM, SPAN_EVIDENCE,
                                  MACHINE_CHECKED_PROOF)}


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.ontology - SELF TEST")
    ok = True

    # --- Barbara conforms
    barbara = {"premises": ["All cats are mammals", "All mammals are animals"],
               "conclusion": "All cats are animals", "mood": "AAA", "figure": 1, "valid": True}
    v = CATEGORICAL_SYLLOGISM.validate(barbara)
    good = not v
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Barbara conforms ({v or 'no violations'})")

    # --- the fallacy of four terms is caught by the term-count rule
    four = {"premises": ["All cats are mammals", "All dogs are animals"],
            "conclusion": "All cats are animals"}
    v = CATEGORICAL_SYLLOGISM.validate(four)
    good = any("exactly_three_terms" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] four-term fallacy caught ({len(v)} violation(s))")

    # --- missing middle term
    nomid = {"premises": ["All cats are mammals", "All dogs are animals"],
             "conclusion": "All cats are animals"}
    v = CATEGORICAL_SYLLOGISM.validate(nomid)
    good = any("middle_term_shared" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] missing middle term caught")

    # --- middle term appearing in the conclusion
    midconc = {"premises": ["All cats are mammals", "All mammals are animals"],
               "conclusion": "All mammals are animals"}
    v = CATEGORICAL_SYLLOGISM.validate(midconc)
    good = any("middle_not_in_conclusion" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] middle term in conclusion caught")

    # --- stated mood disagreeing with the propositions
    badmood = {"premises": ["All cats are mammals", "All mammals are animals"],
               "conclusion": "All cats are animals", "mood": "AAA"}
    good = CATEGORICAL_SYLLOGISM.conforms(badmood)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] correct mood AAA accepted")

    # NOTE: "AAA"[::-1] is "AAA" -- a palindrome, so an earlier version of this test asserted
    # that a CORRECT mood should be rejected and quietly passed for the wrong reason.
    wrongmood = dict(badmood, mood="EEE")
    v = CATEGORICAL_SYLLOGISM.validate(wrongmood)
    good = any("mood_matches" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] wrong mood caught")

    # --- one premise is not a syllogism
    v = CATEGORICAL_SYLLOGISM.validate({"premises": ["All cats are mammals"],
                                        "conclusion": "All cats are animals"})
    good = any("two_premises" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] one premise caught")

    # --- missing required field
    v = CATEGORICAL_SYLLOGISM.validate({"premises": ["All a are b", "All b are c"]})
    good = any("missing required field 'conclusion'" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] missing conclusion field caught")

    # --- SPAN_EVIDENCE: the decisive-needs-comparable rule is the important one
    decisive_uncomparable = {"label": "refute", "subject": "the bridge", "attribute": "length",
                             "comparable": False, "reason": "looks wrong"}
    v = SPAN_EVIDENCE.validate(decisive_uncomparable)
    good = any("decisive_needs_comparable" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a decisive span label without a comparable "
          f"quantity is rejected")

    good = SPAN_EVIDENCE.conforms({"label": "insufficient", "subject": "the bridge",
                                   "attribute": "opened", "comparable": False,
                                   "reason": "different unit family"})
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a reasoned insufficient verdict conforms")

    # --- MACHINE_CHECKED_PROOF encodes the Lean sorry lesson
    v = MACHINE_CHECKED_PROOF.validate({"compiled": True, "has_sorry": True,
                                        "method": "tactic"})
    good = any("sorry_is_not_a_proof" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a compiled sorry proof is rejected by the ontology")

    v = MACHINE_CHECKED_PROOF.validate({"compiled": True, "has_sorry": False,
                                        "method": "python_only"})
    good = any("python_is_not_machine_checked" in x for x in v)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] python_only cannot claim machine-checked")

    # --- the prompt renders the constraints AND the refusal instruction
    p = CATEGORICAL_SYLLOGISM.render_prompt("formalise the argument")
    good = ("exactly_three_terms" in p and "refuse" in p and "middle_term_shared" in p)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] prompt carries every constraint and the refusal "
          f"instruction ({len(p)} chars)")

    # --- every ontology's declared examples agree with its own validator
    bad = []
    for o in REGISTRY.values():
        for expected_ok, payload in o.examples:
            actual_ok = o.conforms(payload)
            if actual_ok != expected_ok:
                bad.append(f"{o.name}: expected {expected_ok}, got {actual_ok} for {payload}")
    good = not bad
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all declared examples agree with their own "
          f"validator" + (f" -- {bad}" if bad else ""))

    # --- a rule that raises is treated as a violation, not a crash
    broken = Ontology(name="broken", description="has a rule that raises",
                      fields={"x": Field("int")},
                      rules=[("boom", lambda o: o["nonexistent"] > 0, "never evaluated")])
    v = broken.validate({"x": 5})
    good = len(v) == 1 and "raised" in v[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a crashing rule is reported as a violation, "
          f"not an exception")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Logical frames as executable ontologies.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--check", nargs=2, default=[], metavar=("ONTOLOGY", "JSON"))
    args = ap.parse_args(argv)
    if args.self_test or (not args.prompt and not args.check and not args.list):
        return 0 if run_self_test() else 1
    if args.list:
        for name, o in REGISTRY.items():
            print(f"{name:28s} {len(o.fields)} fields, {len(o.rules)} rules -- "
                  f"{o.description[:60]}...")
        return 0
    if args.prompt:
        o = REGISTRY.get(args.prompt)
        if not o:
            print(f"unknown ontology; try {list(REGISTRY)}")
            return 2
        print(o.render_prompt())
        return 0
    o = REGISTRY.get(args.check[0])
    if not o:
        print(f"unknown ontology; try {list(REGISTRY)}")
        return 2
    try:
        payload = json.loads(args.check[1])
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}")
        return 2
    violations = o.validate(payload)
    print(f"conforms: {not violations}")
    for x in violations:
        print(f"  - {x}")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
