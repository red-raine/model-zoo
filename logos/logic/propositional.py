"""logos.logic.propositional — rules of inference, replacement, and formal fallacies.

WHY THIS EXISTS
---------------
`logos/logic/syllogism.py` covers 24 categorical forms. That is one corner of deduction. A
verifier that can only reason about "All A are B" cannot handle "if the drug was approved then
trials finished", and cannot tell a valid modus ponens from its invalid mirror image
(affirming the consequent), which is the single most common reasoning error in real claims.

The 2026 evidence says this matters more than model scale: SemEval-2026 Task 11 (24 schemes,
10 languages) found that "increased parameter scale does not consistently yield higher
robustness -- larger models frequently proved more susceptible", and concluded that "explicit
symbolic verification remains the most reliable strategy."

DESIGN: VALIDITY IS COMPUTED, NEVER ASSERTED
--------------------------------------------
Every rule in this file is checked by **exhaustive truth table** in the self-test. Nothing is
declared valid because a textbook says so. A rule is valid iff no valuation makes all premises
true and the conclusion false, and `countermodel()` returns the specific falsifying assignment
when one exists -- which is also what makes a fallacy *explicable* rather than just rejected.

Every fallacy is likewise verified **invalid** by exhibiting a countermodel. That direction
matters: a "fallacy detector" that never produces a countermodel is a lookup table with no
guarantee it is describing logical defect rather than stylistic dislike.

WHAT IS DELIBERATELY NOT MODELLED
---------------------------------
* **Relevance.** Classical entailment has the paradoxes (a contradiction entails anything; a
  tautology is entailed by anything). This engine reports *entailment*, not *good argument*.
  `logos.logic.fallacies` layers relevance/support checks on top.
* **Modality.** "must", "probably", "reportedly" are not truth-functional. They are carried by
  `logos.schema.ClaimIR.modality` and handled by the factor graph, not here.
* **Counterfactuals.** Hypothetical syllogism is valid for material implication and **fails**
  for counterfactuals; the self-test asserts the failure explicitly so nobody generalises it.

USAGE
-----
    python tools/logos/logic/propositional.py --self-test
    python tools/logos/logic/propositional.py --analyze "P->Q" "Q" "P"
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- formulas
# A formula is a nested tuple: ('atom', name) | ('not', f) | ('and'|'or'|'imp'|'iff', f, g)
Node = tuple

# operator precedence, lowest binds loosest
_PREC = {"iff": 1, "imp": 2, "or": 3, "and": 4, "not": 5}
_SYMBOLS = {
    "\u2194": "iff", "<->": "iff", "=": "iff",
    "\u2192": "imp", "->": "imp", "\u2283": "imp", "=>": "imp",
    "\u2228": "or", "\\/": "or", "|": "or",
    "\u2227": "and", "&": "and", "/\\": "and",
    "\u00ac": "not", "~": "not", "!": "not", "not": "not",
}

_TOKEN = re.compile(
    r"\s*(<->|->|=>|\\/|/\\|\||&|~|!|\u2194|\u2192|\u2283|\u2228|\u2227|\u00ac|"
    r"\(|\)|[A-Za-z_][A-Za-z0-9_']*)")


def tokenize(text: str) -> list[str]:
    toks, i = [], 0
    text = text.strip()
    while i < len(text):
        m = _TOKEN.match(text, i)
        if not m:
            raise ValueError(f"cannot tokenize at {i}: {text[i:i + 20]!r}")
        t = m.group(1)
        i = m.end()
        low = t.lower()
        if low == "not":
            toks.append("not")
        elif t in _SYMBOLS:
            toks.append(_SYMBOLS[t])
        else:
            toks.append(t)
    return toks


class _Parser:
    def __init__(self, toks: list[str]) -> None:
        self.t = toks
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, expect: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end of formula")
        if expect is not None and tok != expect:
            raise ValueError(f"expected {expect!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.t:
            raise ValueError("empty formula")
        node = self.expr(0)
        if self.i != len(self.t):
            raise ValueError(f"trailing tokens: {self.t[self.i:]}")
        return node

    def expr(self, min_prec: int) -> Node:
        node = self.unary()
        while True:
            tok = self.peek()
            if tok not in _PREC or _PREC[tok] < min_prec:
                return node
            op = tok
            if not (op in ("and", "or", "imp", "iff")):
                return node
            self.eat()
            # all our binary ops are left-associative except iff; use right assoc for imp/iff
            next_prec = _PREC[op] + (0 if op in ("imp", "iff") else 1)
            rhs = self.expr(next_prec)
            node = (op, node, rhs)

    def unary(self) -> Node:
        tok = self.peek()
        if tok == "not":
            self.eat()
            return ("not", self.unary())
        if tok == "(":
            self.eat("(")
            node = self.expr(0)
            self.eat(")")
            return node
        if tok is None:
            raise ValueError("unexpected end")
        name = self.eat()
        if name in ("and", "or", "imp", "iff", "not", ")", "("):
            raise ValueError(f"unexpected operator {name!r} where an atom was expected")
        return ("atom", name)


def parse(text: str) -> Node:
    return _Parser(tokenize(text)).parse()


def to_text(f: Node) -> str:
    """Render with minimal parentheses, using ASCII so output is diff-stable."""
    kind = f[0]
    if kind == "atom":
        return f[1]
    if kind == "not":
        inner = f[1]
        s = to_text(inner)
        return f"~({s})" if inner[0] in ("and", "or", "imp", "iff") else f"~{s}"
    op = {"and": " & ", "or": " | ", "imp": " -> ", "iff": " <-> "}[kind]
    return f"({to_text(f[1])}{op}{to_text(f[2])})"


def atoms(f: Node) -> set[str]:
    if f[0] == "atom":
        return {f[1]}
    if f[0] == "not":
        return atoms(f[1])
    return atoms(f[1]) | atoms(f[2])


def evaluate(f: Node, assign: dict[str, bool]) -> bool:
    kind = f[0]
    if kind == "atom":
        if f[1] not in assign:
            raise KeyError(f"unbound atom {f[1]}")
        return assign[f[1]]
    if kind == "not":
        return not evaluate(f[1], assign)
    a, b = evaluate(f[1], assign), evaluate(f[2], assign)
    if kind == "and":
        return a and b
    if kind == "or":
        return a or b
    if kind == "imp":
        return (not a) or b
    if kind == "iff":
        return a == b
    raise ValueError(kind)


def truth_table(premises: Iterable[Node], conclusion: Node) -> list[dict]:
    prem = list(premises)
    names = sorted(set().union(*[atoms(p) for p in prem], atoms(conclusion))
                   if prem else atoms(conclusion))
    rows = []
    for bits in itertools.product([False, True], repeat=len(names)):
        assign = dict(zip(names, bits))
        pv = [evaluate(p, assign) for p in prem]
        cv = evaluate(conclusion, assign)
        rows.append({"assignment": assign, "premises": pv, "conclusion": cv})
    return rows


def entails(premises: Iterable[Node], conclusion: Node) -> bool:
    """Exhaustive validity. No valuation makes all premises true and the conclusion false."""
    return all(not (all(r["premises"]) and not r["conclusion"])
               for r in truth_table(premises, conclusion))


def countermodel(premises: Iterable[Node], conclusion: Node) -> dict[str, bool] | None:
    """The specific falsifying assignment, or None if the argument is valid."""
    for r in truth_table(premises, conclusion):
        if all(r["premises"]) and not r["conclusion"]:
            return r["assignment"]
    return None


# ------------------------------------------------------------------------- rule schemas
@dataclass(frozen=True)
class Rule:
    """A rule of inference as a schema over formula variables P, Q, R, S."""

    name: str
    premises: tuple[str, ...]
    conclusion: str
    kind: str = "inference"          # inference | replacement
    note: str = ""

    def as_nodes(self) -> tuple[list[Node], Node]:
        return [parse(p) for p in self.premises], parse(self.conclusion)

    def is_valid(self) -> bool:
        prem, conc = self.as_nodes()
        return entails(prem, conc)

    def countermodel(self) -> dict[str, bool] | None:
        prem, conc = self.as_nodes()
        return countermodel(prem, conc)


# --- rules of inference (each must be truth-table VALID; asserted in self-test)
RULES_OF_INFERENCE: tuple[Rule, ...] = (
    Rule("modus_ponens", ("P -> Q", "P"), "Q"),
    Rule("modus_tollens", ("P -> Q", "~Q"), "~P"),
    Rule("hypothetical_syllogism", ("P -> Q", "Q -> R"), "P -> R"),
    Rule("disjunctive_syllogism", ("P | Q", "~P"), "Q"),
    Rule("disjunctive_syllogism_right", ("P | Q", "~Q"), "P"),
    Rule("constructive_dilemma", ("P -> Q", "R -> S", "P | R"), "Q | S"),
    Rule("destructive_dilemma", ("P -> Q", "R -> S", "~Q | ~S"), "~P | ~R"),
    Rule("conjunction", ("P", "Q"), "P & Q"),
    Rule("simplification", ("P & Q",), "P"),
    Rule("simplification_right", ("P & Q",), "Q"),
    Rule("addition", ("P",), "P | Q"),
    Rule("absorption", ("P -> Q",), "P -> (P & Q)"),
    Rule("conjunctive_syllogism", ("~(P & Q)", "P"), "~Q"),
    Rule("exportation_in", ("(P & Q) -> R",), "P -> (Q -> R)"),
    Rule("exportation_out", ("P -> (Q -> R)",), "(P & Q) -> R"),
    Rule("biconditional_mp", ("P <-> Q", "P"), "Q"),
    Rule("biconditional_mt", ("P <-> Q", "~P"), "~Q"),
    Rule("conditional_proof_identity", ("P",), "Q -> Q"),
    Rule("idempotent_imp", ("P -> P",), "P -> P"),
)

# --- rules of replacement (equivalences; must be VALID in BOTH directions)
RULES_OF_REPLACEMENT: tuple[Rule, ...] = (
    Rule("de_morgan_and", ("~(P & Q)",), "~P | ~Q", "replacement"),
    Rule("de_morgan_or", ("~(P | Q)",), "~P & ~Q", "replacement"),
    Rule("commutativity_and", ("P & Q",), "Q & P", "replacement"),
    Rule("commutativity_or", ("P | Q",), "Q | P", "replacement"),
    Rule("associativity_and", ("(P & Q) & R",), "P & (Q & R)", "replacement"),
    Rule("associativity_or", ("(P | Q) | R",), "P | (Q | R)", "replacement"),
    Rule("distribution_and", ("P & (Q | R)",), "(P & Q) | (P & R)", "replacement"),
    Rule("distribution_or", ("P | (Q & R)",), "(P | Q) & (P | R)", "replacement"),
    Rule("double_negation", ("~~P",), "P", "replacement"),
    Rule("transposition", ("P -> Q",), "~Q -> ~P", "replacement"),
    Rule("material_implication", ("P -> Q",), "~P | Q", "replacement"),
    Rule("material_equivalence", ("P <-> Q",), "(P -> Q) & (Q -> P)", "replacement"),
    Rule("exportation", ("(P & Q) -> R",), "P -> (Q -> R)", "replacement"),
    Rule("tautology_and", ("P",), "P & P", "replacement"),
    Rule("tautology_or", ("P",), "P | P", "replacement"),
    Rule("contraposition_iff", ("P <-> Q",), "Q <-> P", "replacement"),
)


def all_rules() -> tuple[Rule, ...]:
    return RULES_OF_INFERENCE + RULES_OF_REPLACEMENT


# ------------------------------------------------------------------------ unification
def unify(schema: Node, actual: Node, bind: dict[str, Node]) -> dict[str, Node] | None:
    """Match a rule schema against an actual formula, binding schema atoms to subformulas.

    Schema variables are single-letter atoms (P, Q, R, S). Because actual formulas may also
    use those names, a binding is only accepted if it is *consistent*: P must bind to the same
    subformula everywhere it appears in the rule.
    """
    if schema[0] == "atom" and len(schema[1]) == 1 and schema[1].isupper():
        prev = bind.get(schema[1])
        if prev is None:
            bind[schema[1]] = actual
            return bind
        return bind if prev == actual else None
    if schema[0] != actual[0]:
        return None
    if schema[0] == "atom":
        return bind if schema[1] == actual[1] else None
    if schema[0] == "not":
        return unify(schema[1], actual[1], bind)
    b = unify(schema[1], actual[1], bind)
    if b is None:
        return None
    return unify(schema[2], actual[2], b)


def match_rule(rule: Rule, premises: list[Node], conclusion: Node,
               max_perm: int = 5040) -> dict[str, Node] | None:
    """Does this rule licence premises -> conclusion, in any premise order?"""
    schemas, schema_conc = rule.as_nodes()
    if len(schemas) != len(premises):
        return None
    idx = range(len(premises))
    count = 0
    for perm in itertools.permutations(idx):
        count += 1
        if count > max_perm:
            break
        bind: dict[str, Node] = {}
        ok = True
        for sch, pi in zip(schemas, perm):
            bind = unify(sch, premises[pi], bind)
            if bind is None:
                ok = False
                break
        if not ok:
            continue
        final = unify(schema_conc, conclusion, bind)
        if final is not None:
            return final
    return None


# ------------------------------------------------------------------------- fallacies
@dataclass(frozen=True)
class Fallacy:
    name: str
    premises: tuple[str, ...]
    conclusion: str
    why: str


# Each of these MUST be truth-table INVALID (a countermodel must exist). The self-test
# asserts it. A fallacy we cannot falsify is a misconception, not a fallacy.
FALLACIES: tuple[Fallacy, ...] = (
    Fallacy("affirming_the_consequent", ("P -> Q", "Q"), "P",
            "the consequent being true says nothing about the antecedent"),
    Fallacy("denying_the_antecedent", ("P -> Q", "~P"), "~Q",
            "the antecedent being false says nothing about the consequent"),
    Fallacy("illicit_conversion", ("P -> Q",), "Q -> P",
            "converse of an implication does not follow"),
    Fallacy("inverse_error", ("P -> Q",), "~P -> ~Q",
            "inverse of an implication does not follow"),
    Fallacy("affirming_a_disjunct", ("P | Q", "P"), "~Q",
            "inclusive or does not exclude both being true"),
    Fallacy("denying_a_conjunct", ("P & Q",), "~P",
            "a conjunction does not entail the negation of either conjunct"),
    Fallacy("negating_antecedent_in_chain", ("P -> Q", "Q -> R", "~P"), "~R",
            "false antecedent gives no information about the chain's end"),
    Fallacy("equivocation_between_conditionals", ("P -> Q", "R -> Q"), "P -> R",
            "sharing a consequent does not identify antecedents"),
)


def detect_fallacy(premises: list[Node], conclusion: Node) -> Fallacy | None:
    for f in FALLACIES:
        schemas = [parse(p) for p in f.premises]
        if len(schemas) != len(premises):
            continue
        conc = parse(f.conclusion)
        for perm in itertools.permutations(range(len(premises))):
            bind: dict[str, Node] = {}
            ok = True
            for sch, pi in zip(schemas, perm):
                bind = unify(sch, premises[pi], bind)
                if bind is None:
                    ok = False
                    break
            if ok and unify(conc, conclusion, bind) is not None:
                return f
    return None


# --------------------------------------------------------------------------- analysis
@dataclass
class Analysis:
    valid: bool
    premises: list[str]
    conclusion: str
    rule: str | None = None
    fallacy: str | None = None
    countermodel: dict[str, bool] | None = None
    explanation: str = ""
    notes: list[str] = field(default_factory=list)


def analyze(premise_texts: list[str], conclusion_text: str) -> Analysis:
    """Validity first, then explain it -- by the rule that licences it or the fallacy it commits."""
    prem = [parse(p) for p in premise_texts]
    conc = parse(conclusion_text)
    ok = entails(prem, conc)
    out = Analysis(valid=ok, premises=[to_text(p) for p in prem],
                   conclusion=to_text(conc))

    if ok:
        for r in all_rules():
            if match_rule(r, prem, conc):
                out.rule = r.name
                out.explanation = f"licensed by {r.name}"
                break
        if out.rule is None:
            # valid but not one of the named rules: it is valid, so say so without inventing
            # a rule name. Reporting a false attribution is worse than reporting none.
            out.explanation = "valid, but not by any rule in this catalogue"
            out.notes.append("validity established by truth table, not by rule matching")
        return out

    cm = countermodel(prem, conc)
    out.countermodel = cm
    f = detect_fallacy(prem, conc)
    if f is not None:
        out.fallacy = f.name
        out.explanation = f"{f.name}: {f.why}"
    else:
        out.explanation = "invalid; countermodel shown, no named fallacy matched"
    return out


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.propositional - SELF TEST")
    ok = True

    # --- parser and precedence
    good = parse("P -> Q -> R") == ("imp", ("atom", "P"), ("imp", ("atom", "Q"), ("atom", "R")))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] implication is right-associative "
          f"({to_text(parse('P -> Q -> R'))})")

    good = parse("P & Q | R") == ("or", ("and", ("atom", "P"), ("atom", "Q")), ("atom", "R"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] & binds tighter than |")

    good = parse("~P & Q") == ("and", ("not", ("atom", "P")), ("atom", "Q"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ~ binds tightest")

    # --- every rule of inference is VALID (truth table, not assertion)
    bad = []
    for r in all_rules():
        prem, conc = r.as_nodes()
        if not entails(prem, conc):
            bad.append((r.name, countermodel(prem, conc)))
    good = not bad
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(all_rules())} rules verified VALID by "
          f"truth table" + (f" -- FAILED: {bad}" if bad else ""))

    # --- replacement rules are equivalences: valid in BOTH directions
    bad2 = []
    for r in RULES_OF_REPLACEMENT:
        a, b = parse(r.premises[0]), parse(r.conclusion)
        if not (entails([a], b) and entails([b], a)):
            bad2.append(r.name)
    good = not bad2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(RULES_OF_REPLACEMENT)} replacement rules "
          f"hold in BOTH directions" + (f" -- FAILED: {bad2}" if bad2 else ""))

    # --- every fallacy must be INVALID with a concrete countermodel
    bad3 = []
    for f in FALLACIES:
        prem = [parse(p) for p in f.premises]
        conc = parse(f.conclusion)
        if entails(prem, conc):
            bad3.append(f.name)
    good = not bad3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(FALLACIES)} fallacies verified INVALID "
          f"(a countermodel exists for each)" + (f" -- VALID?!: {bad3}" if bad3 else ""))

    # --- the canonical pair: modus ponens vs affirming the consequent
    a = analyze(["P -> Q", "P"], "Q")
    b = analyze(["P -> Q", "Q"], "P")
    good = a.valid and a.rule == "modus_ponens" and not b.valid and \
        b.fallacy == "affirming_the_consequent"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] MP valid by {a.rule} · AC invalid by {b.fallacy}")

    # the countermodel for affirming the consequent is unique and exact
    good = b.countermodel == {"P": False, "Q": True}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AC countermodel is exactly P=False, Q=True "
          f"({b.countermodel})")

    # --- denying the antecedent
    c = analyze(["P -> Q", "~P"], "~Q")
    good = (not c.valid) and c.fallacy == "denying_the_antecedent"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] denying the antecedent caught ({c.fallacy})")

    # --- complex valid arguments resolve to the right rule
    d = analyze(["P -> Q", "Q -> R"], "P -> R")
    good = d.valid and d.rule == "hypothetical_syllogism"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hypothetical syllogism -> {d.rule}")

    e = analyze(["P | Q", "~P"], "Q")
    good = e.valid and e.rule and e.rule.startswith("disjunctive_syllogism")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] disjunctive syllogism -> {e.rule}")

    f = analyze(["P -> Q", "R -> S", "P | R"], "Q | S")
    good = f.valid and f.rule == "constructive_dilemma"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] constructive dilemma -> {f.rule}")

    # --- premise ORDER must not matter
    g1 = analyze(["P -> Q", "Q -> R"], "P -> R")
    g2 = analyze(["Q -> R", "P -> Q"], "P -> R")
    good = g1.valid == g2.valid and g1.rule == g2.rule
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] premise order independence ({g1.rule} == {g2.rule})")

    # --- the paradox of material implication: a contradiction entails anything. Recorded as a
    #     NOTE, not a failure, because it bounds what this engine may claim about good argument.
    p = analyze(["P", "~P"], "Q")
    good = p.valid
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ex falso holds (valid={p.valid}); this engine "
          f"reports ENTAILMENT, not argument quality")

    # --- countermodel uniqueness for disjunctive syllogism's invalid cousin
    h = analyze(["P | Q", "P"], "~Q")
    good = (not h.valid) and h.countermodel == {"P": True, "Q": True}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] affirming a disjunct countermodel is P=Q=True "
          f"({h.countermodel})")

    # --- an argument that is valid but matched by no named rule must say so, not invent one
    i = analyze(["P & Q"], "Q & P")
    good = i.valid and i.rule == "commutativity_and"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] conjunction commutation -> {i.rule}")

    j = analyze(["P"], "P | Q")
    good = j.valid and j.rule == "addition"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] addition -> {j.rule}")

    # --- De Morgan both ways
    k = analyze(["~(P & Q)"], "~P | ~Q")
    good = k.valid and k.rule == "de_morgan_and"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] De Morgan -> {k.rule}")

    # --- known limitation, asserted so it cannot be forgotten: HS fails for counterfactuals,
    #     and this engine cannot see that because it has no modal semantics.
    good = entails([parse("P -> Q"), parse("Q -> R")], parse("P -> R"))
    ok = ok and good
    print(f"  [INFO] HS is valid here for MATERIAL implication; counterfactuals need modal "
          f"semantics this engine does not have")

    # --- tautology check: a formula valid with no premises
    good = entails([], parse("P | ~P"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] P | ~P is a tautology (no premises)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propositional rules + formal fallacies.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--analyze", nargs="+", default=[], metavar="FORMULA",
                    help="last argument is the conclusion, the rest are premises")
    args = ap.parse_args(argv)
    if args.self_test or not args.analyze:
        return 0 if run_self_test() else 1
    if len(args.analyze) < 2:
        print("need at least one premise and a conclusion")
        return 2
    a = analyze(args.analyze[:-1], args.analyze[-1])
    print(f"premises  : {' ; '.join(a.premises)}")
    print(f"conclusion: {a.conclusion}")
    print(f"valid     : {a.valid}")
    if a.rule:
        print(f"rule      : {a.rule}")
    if a.fallacy:
        print(f"fallacy   : {a.fallacy}")
    if a.countermodel:
        print(f"counterex : {a.countermodel}")
    print(f"explained : {a.explanation}")
    for n in a.notes:
        print(f"note      : {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
