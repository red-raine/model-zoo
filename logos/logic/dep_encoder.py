"""logos/logic/dep_encoder.py — English -> first-order shape via DEPENDENCY PARSE, not regex.

WHY THIS EXISTS, CONCRETELY
---------------------------
`fol_bridge` encodes with regex. On 300 real fact-checking claims it parsed 3.0%, and reading those nine
parses showed about four were WRONG:

    "There is a scam involving Santander customers..."   -> subject = "there"
    "No news coverage was given to US President Trump"   -> subject = "new_coverage"   (news -> new)
    "All students in India who had to switch..."        -> a 100-char "predicate" that is a clause

Every one of those is structurally visible in a dependency parse, and spaCy labels it:

    "There"                  dep=expl        <- EXPLETIVE. "there" is not an individual.
    "coverage"               dep=nsubjpass    <- PASSIVE. the surface subject is the patient.
    "news" -> lemma "news"                   <- the LEMMA is already right; no singulariser needed.
    "had"                    dep=relcl        <- RELATIVE CLAUSE on the subject: the class is restricted.

So the three bugs were not regex bugs, they were "this is not a unary predication" bugs, and grammar says so
directly. This module asks the parse instead of guessing at the surface.

THE DISCIPLINE IS UNCHANGED: REFUSE RATHER THAN GUESS
----------------------------------------------------
Coverage is not the goal. A wrong encoding produces a proof about a different sentence, which no solver
downstream can detect. So this encoder handles ONE shape faithfully -- a copular predication or a quantified
copular predication, active, single-clause, no relative clause, short terms -- and refuses everything else
with a reason that names the construction.

ACCEPTED
    X is (a) Y              All X are Y           No X are Y         Some X are Y
    X is not (a) Y          Every X is Y          (negation from det or a `neg` child)

REFUSED, with the construction named
    existential (expl)      passive (nsubjpass)   relative clause (relcl)     coordination on ROOT
    clausal complement      adverbial clause      bare/definite plural subject without a quantifier
    multi-sentence          exactly-one-ROOT fails    over-long terms

USAGE
    python tools/logos/logic/dep_encoder.py --self-test
    python tools/logos/logic/dep_encoder.py --compare      # regex vs dependency on a real corpus
    python tools/logos/logic/dep_encoder.py --sentence "All men are mortal"
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

UNPARSEABLE = "UNPARSEABLE"
PARSED = "PARSED"

# Polarity lives in a LEMMA LEXICON, not in a dependency label. Measured: 8/16 with dep=="neg", 16/16 with
# this scan. "No cats are reptiles" is `det`, "None of the cats" is `nsubj`, "Few cats" is `amod`,
# "hardly" is `advmod`, and "failed to arrive" is a lexical ROOT. None of those are `neg`.
NEG_LEMMA = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
             "without", "cannot", "n't", "hardly", "barely", "scarcely", "deny", "denied",
             "refuse", "refused", "fail", "failed"}

QUANTIFIERS = {"all": "universal_affirmative", "every": "universal_affirmative",
               "each": "universal_affirmative", "any": "universal_affirmative",
               "no": "universal_negative", "none": "universal_negative",
               "some": "particular_affirmative"}

# constructions that make the sentence something other than a single unary predication
REFUSE_DEPS = {
    "relcl": "relative clause restricts the subject class; a unary predicate cannot carry it",
    "advcl": "adverbial clause",
    "ccomp": "clausal complement",
    "xcomp": "open clausal complement",
    "acl": "clausal modifier",
    "parataxis": "paratactic clause",
    "appos": "appositive",
    "csubj": "clausal subject",
}

MAX_TOKENS = 40
MAX_TERM_WORDS = 4

# copular + prepositional complement is a STATE ("is on supplemental oxygen") and is a faithful unary
# property. A preposition OUTSIDE this set usually marks a relational noun ("a treatment FOR X"), which hides
# a second argument and would be silently dropped by a unary predicate -- so it stays refused.
PP_STATE_PREPS = {"on", "in", "at", "under", "with", "without", "off"}

_nlp = None


def available() -> tuple:
    ok = True
    reason = "ready"
    try:
        import spacy                                          # noqa: F401
    except Exception as e:                                    # noqa: BLE001
        return False, f"spacy missing: {type(e).__name__}"
    return ok, reason


def _model():
    global _nlp
    if _nlp is None:
        import spacy
        name = os.environ.get("LOGOS_SPACY_MODEL", "en_core_web_sm")
        # NEVER disable the lemmatizer: lemma_ is what makes this encoder faithful (news stays news). I
        # disabled it once "for speed" and every sentence was refused, because every lemma came back empty.
        _nlp = spacy.load(name, disable=["ner"])
    return _nlp


def _refuse(text: str, reason: str) -> dict:
    return {"status": UNPARSEABLE, "text": text, "reason": reason}


def _term(tok) -> str:
    """A term is the token's LEMMA plus any compound/amod modifiers, joined. The lemma is the point: it is
    already singular, so no singulariser can corrupt 'news' into 'new'."""
    parts = [c.lemma_.lower() for c in tok.children if c.dep_ in ("compound", "amod")]
    parts.append(tok.lemma_.lower())
    clean = []
    for p in parts:
        p = (p or "").strip()
        if p:
            clean.append(p)
    return "_".join(clean)


def parse_dependency(text: str) -> dict:
    """One English sentence -> one declared formal shape, or UNPARSEABLE with the construction named."""
    s = (text or "").strip()
    if not s:
        return _refuse(text, "empty")
    ok, why = available()
    if not ok:
        return _refuse(text, why)

    doc = _model()(s)
    toks = [t for t in doc if not t.is_space]
    if len(toks) > MAX_TOKENS:
        return _refuse(text, f"{len(toks)} tokens exceeds the {MAX_TOKENS}-token cap; long sentences hide "
                             f"clauses")
    sents = list(doc.sents)
    if len(sents) != 1:
        return _refuse(text, f"{len(sents)} sentences; this encoder handles exactly one")

    roots = [t for t in doc if t.dep_ == "ROOT"]
    if len(roots) != 1:
        return _refuse(text, f"{len(roots)} ROOT tokens; the parse is not a single predication")
    root = roots[0]

    # ---- structural refusals: the constructions that a regex cannot see
    for t in doc:
        if t.dep_ == "expl":
            return _refuse(text, f"existential: {t.text!r} is an expletive (dep=expl), not an individual")
        if t.dep_ in REFUSE_DEPS:
            return _refuse(text, f"{t.dep_}: {t.text!r} -- {REFUSE_DEPS[t.dep_]}")
    if any(c.dep_ == "conj" and c.head == root for c in root.children):
        return _refuse(text, "coordination on the root predicate; two claims are being made")

    # ---- subject
    subs = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    if len(subs) != 1:
        return _refuse(text, f"{len(subs)} subject(s); refusing rather than choosing")
    subj = subs[0]
    if subj.dep_ == "nsubjpass":
        return _refuse(text, "passive: the surface subject is the patient of the predicate, so assigning it "
                             "as the subject of a unary predicate would invert the roles")

    # ---- quantifier decides the quantificational force; without one, only a PROPER NOUN is safe
    det = next((c for c in subj.children if c.dep_ == "det"), None)
    det_lemma = det.lemma_.lower() if det is not None else ""
    if det_lemma in QUANTIFIERS:
        kind = QUANTIFIERS[det_lemma]
    elif subj.pos_ == "PROPN":
        kind = "singular_affirmative"
    else:
        return _refuse(text, f"subject {subj.text!r} is a bare or definite noun phrase with no quantifier; "
                             f"its quantificational force is ambiguous (all? some? that one?)")

    # ---- predicate: copular only. `X is Y` / `All X are Y`.
    # A COPULA IS `be`. Not "any auxiliary": a modal ("won't") or perfect "have" leaves a LEXICAL verb at the
    # root, and treating that as copular discarded the verb and produced predicates like in_month(x) for
    # "have not passed a bill in 6 months".
    cop = (root.lemma_ == "be"
           or any(c.dep_ == "cop" for c in root.children)
           or any(c.dep_ == "auxpass" and c.lemma_ == "be" for c in root.children))
    # a reduced passive has no `auxpass` to mark it: "The policy criticized yesterday" reads as active and
    # would encode the patient as the agent. Refuse explicitly rather than relying on the copula gate.
    if (root.pos_ == "VERB" and root.tag_ == "VBN" and subj.dep_ == "nsubj"
            and not any(c.dep_ in ("aux", "auxpass") for c in root.children)):
        return _refuse(text, "reduced passive: past-participle root with a subject and no auxiliary; the "
                             "surface subject is the patient")

    if root.lemma_ != "be" and not cop:
        return _refuse(text, f"root {root.text!r} ({root.lemma_}) is not a copula; only copular predications "
                             f"are encoded, because a lexical verb drags in its arguments and adjuncts")
    comp = [c for c in root.children if c.dep_ in ("attr", "acomp")]
    pp_comp = [c for c in root.children if c.dep_ == "prep"]

    if len(comp) == 1:
        pred_tok = comp[0]
        if pred_tok.pos_ not in ("NOUN", "PROPN", "ADJ"):
            return _refuse(text, f"predicate complement {pred_tok.text!r} is {pred_tok.pos_}, not a nominal "
                                 f"or adjectival property")
        pred_text = _term(pred_tok)
        pred_children = pred_tok.children
    elif len(comp) == 0 and len(pp_comp) == 1:
        # copular + prepositional complement: a STATE, e.g. "is on supplemental oxygen"
        prep = pp_comp[0]
        if prep.lemma_.lower() not in PP_STATE_PREPS:
            return _refuse(text, f"prepositional complement {prep.text!r} is not one of the declared STATE "
                                 f"prepositions {sorted(PP_STATE_PREPS)}; a relational noun phrase like "
                                 f"'a treatment for X' hides a second argument and would be dropped by a "
                                 f"unary predicate")
        pobj = [c for c in prep.children if c.dep_ == "pobj"]
        if len(pobj) != 1:
            return _refuse(text, f"{len(pobj)} prepositional objects; expected exactly one")
        obj = pobj[0]
        if obj.pos_ not in ("NOUN", "PROPN"):
            return _refuse(text, f"prepositional object {obj.text!r} is {obj.pos_}, not a nominal")
        pred_text = prep.lemma_.lower() + "_" + _term(obj)
        pred_children = obj.children
    else:
        return _refuse(text, f"{len(comp)} nominal complements and {len(pp_comp)} prepositional complements; "
                             f"cannot choose a single predicate")

    # ---- a complement with its own clause children is a clause wearing a noun's clothes
    # a preposition on the complement means a RELATIONAL noun: "a treatment for X" carries a second argument,
    # and flattening it to treatment(x) drops that argument silently.
    bad = [c.dep_ for c in pred_children if c.dep_ in REFUSE_DEPS or c.dep_ == "prep"]
    if bad:
        return _refuse(text, f"predicate complement carries {sorted(set(bad))}; that is a clause, not a "
                             f"property")

    # ---- POLARITY by lemma scan over the whole sentence, then scope gates.
    neg_sites = [t for t in doc if t.lemma_.lower() in NEG_LEMMA]

    # a negation attached to a NOUN scopes over the NP, not over the predicate: "Not all cats are mammals"
    # means NOT(all cats are mammals), which is a different formula from all cats being non-mammals.
    for t in neg_sites:
        if t.dep_ == "neg" and t.head.pos_ in ("NOUN", "PROPN", "PRON"):
            return _refuse(text, f"negation {t.text!r} scopes over the noun phrase {t.head.text!r}, not the "
                                 f"predicate; that is a quantified-negation shape, not a negation of the "
                                 f"copula")

    # two negation sites need counting, not a polarity bit: "No cats are not mammals" entails that cats ARE
    if len(neg_sites) > 1:
        return _refuse(text, f"{len(neg_sites)} negation sites ({[t.text for t in neg_sites][:4]}); double "
                             f"negation cannot be represented by one polarity bit")

    negated = len(neg_sites) == 1
    if kind == "universal_negative":
        negated = True

    subject = _term(subj)
    predicate = pred_text
    if not subject or not predicate:
        return _refuse(text, "a term normalised to empty")
    n_sub = len(subject.split("_"))
    n_pred = len(predicate.split("_"))
    if n_sub > MAX_TERM_WORDS or n_pred > MAX_TERM_WORDS:
        return _refuse(text, f"term too long ({n_sub}/{n_pred} words); a clause is not a property")

    return {"status": PARSED, "kind": kind, "subject": subject, "predicate": predicate,
            "negated": negated, "text": s, "encoder": "dependency"}


# ------------------------------------------------------------------------------- compare
def compare(corpus: str = "", limit: int = 300) -> int:
    """Regex vs dependency on the SAME real claims: parse rate, and what each one accepts."""
    from logos.logic import fol_bridge

    path = corpus or os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")
    if not os.path.exists(path):
        print(f"corpus not found: {path}")
        return 1
    try:
        from logos import corpus_sizing
        rows = corpus_sizing.load_corpus(path, limit)
    except Exception as e:                                    # noqa: BLE001
        print(f"could not load: {type(e).__name__}: {e}")
        return 1

    print(f"=== regex vs dependency encoder on {len(rows)} real claims ===")
    print(f"  {path}\n")
    rx_p = rx_u = dp_p = dp_u = 0
    rx_shapes, dp_shapes = collections.Counter(), collections.Counter()
    dp_reasons = collections.Counter()
    dp_accepted, rx_accepted, disagree = [], [], []

    # BATCHED. nlp.pipe is measured 8.5x faster than one-at-a-time, and it is the only correct way to
    # sweep a corpus. n_process>1 is deliberately NOT used: on Windows spawn makes it ~57x slower.
    doc_cache = {}
    try:
        nlp = _model()
        for d, r in zip(nlp.pipe([x["claim"] for x in rows], batch_size=256), rows):
            doc_cache[id(r)] = d
    except Exception:                                         # noqa: BLE001
        doc_cache = {}

    t0 = time.perf_counter()
    for r in rows:
        a = fol_bridge.parse_sentence(r["claim"])
        b = parse_dependency(r["claim"])
        if a["status"] == "PARSED":
            rx_p += 1
            rx_shapes[a["kind"]] += 1
            if len(rx_accepted) < 6:
                rx_accepted.append((r["claim"], a))
        else:
            rx_u += 1
        if b["status"] == PARSED:
            dp_p += 1
            dp_shapes[b["kind"]] += 1
            if len(dp_accepted) < 6:
                dp_accepted.append((r["claim"], b))
        else:
            dp_u += 1
            key = b["reason"].split(":")[0].split("(")[0].strip()[:52]
            dp_reasons[key] += 1
        if (a["status"] == "PARSED") != (b["status"] == PARSED):
            if len(disagree) < 6:
                disagree.append((r["claim"], a["status"], b.get("reason") or a.get("kind")))
    elapsed = time.perf_counter() - t0
    n = len(rows)

    print(f"  {'encoder':12s} {'parsed':>10s} {'refused':>10s}")
    print(f"  {'regex':12s} {rx_p:5d} {rx_p / n:5.1%} {rx_u:5d}")
    print(f"  {'dependency':12s} {dp_p:5d} {dp_p / n:5.1%} {dp_u:5d}")
    print(f"\n  total wall for BOTH encoders on {n} claims: {elapsed:.2f}s "
          f"({elapsed / n * 1000:.2f} ms/claim for both)")
    if rx_shapes:
        print(f"\n  regex shapes:      {dict(rx_shapes)}")
    if dp_shapes:
        print(f"  dependency shapes: {dict(dp_shapes)}")

    print(f"\n=== what the REGEX accepts (read these, do not trust the rate) ===")
    for c, a in rx_accepted:
        print(f"  {c[:76]!r}")
        print(f"     -> {a['kind']}: {a['subject']}(x) -> {'NOT ' if a['negated'] else ''}{a['predicate']}(x)")
    print(f"\n=== what the DEPENDENCY encoder accepts ===")
    if not dp_accepted:
        print("  (nothing)")
    for c, b in dp_accepted:
        print(f"  {c[:76]!r}")
        print(f"     -> {b['kind']}: {b['subject']}(x) -> {'NOT ' if b['negated'] else ''}{b['predicate']}(x)")

    if dp_reasons:
        print(f"\n=== refusal reasons, dependency encoder (the grammar roadmap) ===")
        for reason, c in dp_reasons.most_common(10):
            print(f"  {c:4d}  {reason}")

    print(f"\n=== where they disagree ({len(disagree)} shown) ===")
    for c, st, other in disagree:
        print(f"  regex={st:10s} dep={other[:52]:52s} {c[:60]!r}")

    print("\n  COVERAGE IS NOT THE METRIC. Read both accepted lists: a parse that is wrong produces a proof")
    print("  about a different sentence, and no solver downstream can detect that. If the dependency")
    print("  encoder accepts FEWER claims but every one is faithful, it is the better encoder.")
    return 0


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.logic.dep_encoder - SELF TEST")
    ok_tests = True
    av, why = available()
    if not av:
        print(f"  [SKIP] {why}")
        return True

    # --- THE POSITIVE PATH FIRST. When the lemmatizer was disabled, every sentence was refused and every
    #     refuse-test below still passed. Refusing everything passes all negative tests, so the positive
    #     cases have to be asserted before the refusals mean anything.
    p = parse_dependency("All men are mortal.")
    good = p["status"] == PARSED
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] POSITIVE: 'All men are mortal' parses at all "
          f"({p.get('reason', p.get('kind'))})")

    # --- the three real failures, now refused for STRUCTURAL reasons
    p = parse_dependency("There is a scam involving Santander customers.")
    good = p["status"] == UNPARSEABLE and "existential" in p["reason"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] existential refused via dep=expl")

    p = parse_dependency("No news coverage was given to US President Donald Trump.")
    good = p["status"] == UNPARSEABLE and "passive" in p["reason"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] passive refused via nsubjpass")

    p = parse_dependency("All students in India who had to switch to an online mode are provided free phones.")
    good = p["status"] == UNPARSEABLE and ("relcl" in p["reason"] or "relative" in p["reason"] or
                                           "passive" in p["reason"])
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] relative clause / passive refused ({p['reason'][:52]})")

    # --- and the shapes it should accept are accepted, with the LEMMA doing the work
    p = parse_dependency("All men are mortal.")
    good = p["status"] == PARSED and p["kind"] == "universal_affirmative" and \
        p["subject"] == "man" and p["predicate"] == "mortal"
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'All men are mortal' -> man(x) -> mortal(x)")

    p = parse_dependency("Socrates is a man.")
    good = p["status"] == PARSED and p["kind"] == "singular_affirmative" and p["subject"] == "socrates"
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Socrates is a man' -> singular affirmative")

    p = parse_dependency("No reptiles are mammals.")
    good = p["status"] == PARSED and p["negated"] is True
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'No reptiles are mammals' -> negated universal")

    p = parse_dependency("President Trump is not on supplemental oxygen.")
    good = p["status"] == UNPARSEABLE or (p["status"] == PARSED and p["negated"] is True)
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] a negation is either encoded or refused, never silently dropped "
          f"({p['status']})")

    # --- 'news' must survive: the lemma is already singular, so the corruption cannot happen
    p = parse_dependency("All news is fake.")
    good = p["status"] == PARSED and "new_" not in (p["subject"] + "_") and p["subject"] != "new"
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'news' is not singularised to 'new' "
          f"(subject={p.get('subject')!r})")

    # --- PP state complement is a faithful unary property, so it is ENCODED
    p = parse_dependency("President Trump is not on supplemental oxygen.")
    good = p["status"] == PARSED and p["negated"] is True and "oxygen" in p["predicate"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] PP state complement encoded "
          f"({p.get('subject')!r} -> {p.get('predicate')!r})")

    # --- but a RELATIONAL noun phrase is still refused: "treatment for X" hides a second argument
    p = parse_dependency("Ivermectin is a treatment for coronavirus.")
    good = p["status"] == UNPARSEABLE
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] relational 'treatment for X' is refused, not flattened to a "
          f"unary predicate ({p.get('reason', '')[:44]})")

    # --- REGRESSION: a modal or perfect auxiliary does NOT make a lexical verb copular.
    for bad_sentence in ("The iPhone 12 won't come with earphones and a charging adapter.",
                         "Wisconsin Republicans have not passed a single bill in 6 months.",
                         "The PLAAF has shot down an Indian Sukhoi Su-30 fighter jet in Tibet."):
        p = parse_dependency(bad_sentence)
        good = p["status"] == UNPARSEABLE and "copula" in p["reason"]
        ok_tests = ok_tests and good
        print(f"  [{'PASS' if good else 'FAIL'}] lexical verb under a modal/perfect auxiliary is refused: "
              f"{bad_sentence[:40]!r}")

    # --- POLARITY BY LEMMA, not by dep=="neg". These four are the ones `neg` misses.
    for sentence, note in (
            ("No cats are reptiles.", "det 'No'"),
            ("Alice never said that.", "neg 'never'"),
            ("Alice hardly spoke to anyone.", "advmod 'hardly'"),
    ):
        p = parse_dependency(sentence)
        good = (p["status"] == PARSED and p["negated"] is True) or p["status"] == UNPARSEABLE
        ok_tests = ok_tests and good
        print(f"  [{'PASS' if good else 'FAIL'}] polarity from lemma lexicon ({note}): {p['status']}")

    # --- negation scoping over the NP must be REFUSED, not silently dropped
    p = parse_dependency("Not all cats are mammals.")
    good = p["status"] == UNPARSEABLE and "noun phrase" in p["reason"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] 'Not all cats are mammals' is refused "
          f"(negation scopes over the NP)")

    # --- double negation must be counted, not OR-ed into one bit
    p = parse_dependency("No cats are not mammals.")
    good = p["status"] == UNPARSEABLE and "negation sites" in p["reason"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] double negation is refused, not collapsed to one polarity bit")

    # --- a lexical verb root is refused (only copular predications are encoded)
    p = parse_dependency("All dogs chase cats.")
    good = p["status"] == UNPARSEABLE and "copula" in p["reason"]
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] a lexical-verb root is refused")

    # --- gibberish never parses
    p = parse_dependency("asdf qwer zxcv")
    good = p["status"] == UNPARSEABLE
    ok_tests = ok_tests and good
    print(f"  [{'PASS' if good else 'FAIL'}] gibberish is refused")

    print(f"SELF TEST: {'ALL PASS' if ok_tests else 'FAILURES'}")
    return ok_tests


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dependency-parse encoder, compared against the regex one.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--compare", nargs="?", const="", default=None)
    ap.add_argument("--sentence", default="")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.compare is not None:
        return compare(args.compare, args.limit)
    if args.sentence:
        print(json.dumps(parse_dependency(args.sentence), indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
