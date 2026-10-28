"""logos/corpus_sizing.py — run the pipeline over a REAL labelled corpus and report what it can decide.

WHY THIS, AND WHY NOW
--------------------
Every module in this suite has been measured on rows I wrote. That is the circularity the project keeps
naming: constructed labels can show that the code does what I designed, never that it is right. The
verifier-av corpus is the first real labelled data available here -- 500 AVeriTeC claims with human gold
verdicts and the actual gathered evidence embedded in the prompt.

    REFUTED 305 | SUPPORTED 122 | NOT_ENOUGH_EVIDENCE 35 | CONFLICTING 38

THE CORRECT READING, WHICH IS NOT "HOW ACCURATE ARE WE"
------------------------------------------------------
Our disposition is about ROW QUALITY, not claim truth. These 500 rows are curated benchmark data, so the
expectation is **KEEP on all of them** and any DROP is a FALSE POSITIVE. The claim's truth is the gold
label and is a different question. So this measures three things that cannot be faked:

    PRECISION   how many curated rows does the pipeline wrongly drop? (want 0)
    COVERAGE    how many can it decide at all? (want high; reality will be low for prose)
    GAP         the gold corpus needs ~93% of claims decided; if we decide ~0%, that is the real size of
                the work left, stated as a number instead of a paragraph

AND IT IS THE FIRST REAL TEST OF numeric_conflict
-------------------------------------------------
`numeric_conflict` was validated on five triples I wrote and scored 4/5 with 0/10 false flags. Five pairs
is not evidence of precision on prose. Run against 500 real claim/evidence pairs, its false-positive rate
is finally measurable -- and a contradiction detector that fires on real evidence in the wrong direction
would quietly poison the reject pile, which is the one failure this project calls catastrophic.

USAGE
    python tools/logos/corpus_sizing.py --self-test
    python tools/logos/corpus_sizing.py --measure [--limit 500] [--corpus <path>]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

CORPUS = os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")


def parse_verifier_av(prompt: str):
    """Pull CLAIM and EVIDENCE out of the AVeriTeC prompt. Returns (claim, evidence).

    Parsed rather than assumed: the corpus uses exactly this wording, and silently reading the wrong slice
    would produce a coverage number about the header text.
    """
    claim = ""
    evidence = ""
    # prefer the span up to EVIDENCE: -- it handles both a single-line claim and one that wraps, and the
    # single-line regex alone silently truncated wrapped claims to their first line.
    m = re.search(r"CLAIM:\s*(.+?)\s*EVIDENCE:", prompt or "", re.S)
    if m:
        claim = m.group(1).strip()
    if not claim:
        m = re.search(r"CLAIM:\s*(.+?)(?:\r?\n)", prompt or "", re.S)
        if m:
            claim = m.group(1).strip()
    m = re.search(r"EVIDENCE:\s*(.+?)(?:\r?\n\s*Answer with|\Z)", prompt or "", re.S)
    if m:
        evidence = m.group(1).strip()
    return claim, evidence


def load_corpus(path: str, limit: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            claim, evidence = parse_verifier_av(o.get("prompt", ""))
            if not claim:
                claim = (o.get("claim") or "").strip()
            rows.append({"claim": claim, "evidence": evidence,
                         "gold": (o.get("completion") or "").strip()})
    return rows


def measure(limit: int = 500, corpus: str = CORPUS) -> int:
    from logos import numeric_conflict, pipeline
    from logos import rerank_client

    if not os.path.exists(corpus):
        print(f"corpus not found: {corpus}")
        return 1
    rows = load_corpus(corpus, limit)
    print(f"=== the pipeline on a REAL labelled corpus ({len(rows)} rows) ===")
    print(f"  {corpus}")
    gold_counts = collections.Counter(r["gold"] for r in rows)
    n_gold_decided = sum(1 for r in rows if r["gold"] in ("SUPPORTED", "REFUTED", "CONFLICTING"))
    print(f"  gold: {dict(gold_counts)}")
    print(f"  the corpus answers {n_gold_decided}/{len(rows)} "
          f"({n_gold_decided / len(rows):.1%}); the rest are genuinely undecidable")
    with_ev = sum(1 for r in rows if r["evidence"])
    print(f"  rows carrying evidence text: {with_ev}/{len(rows)}")
    print()

    disp = collections.Counter()
    drops, drop_examples = 0, []
    fired, fire_examples = 0, []
    gate_sufficient = 0

    for r in rows:
        ev = [r["evidence"]] if r["evidence"] else None
        out = pipeline.evaluate_row(r["claim"], kind="text", evidence=ev)
        disp[out["disposition"]] += 1
        if out["disposition"] == "DROP":
            drops += 1
            if len(drop_examples) < 5:
                drop_examples.append((r["claim"][:96], out["refuted_by"], r["gold"]))
        if out.get("gate_sufficient") is True:
            gate_sufficient += 1
        # numeric_conflict independently, on every real pair, to measure its precision at scale
        if r["evidence"]:
            nc = numeric_conflict.conflicts(r["claim"], r["evidence"])
            if nc["conflict"]:
                fired += 1
                if len(fire_examples) < 8:
                    d = nc["disagreements"][0]
                    fire_examples.append((r["claim"][:70], d["claim_raw"], d["passage_raw"], r["gold"]))

    n = len(rows)
    print("=== what the pipeline could decide ===")
    for d in ("KEEP", "REPAIR", "ESCALATE", "DROP"):
        print(f"  {d:9s} {disp.get(d, 0):5d}  {disp.get(d, 0) / n:6.1%}")
    print()
    print(f"  decided (KEEP+REPAIR+DROP) {n - disp.get('ESCALATE', 0)}/{n} "
          f"= {(n - disp.get('ESCALATE', 0)) / n:.1%}")
    print(f"  escalated                  {disp.get('ESCALATE', 0)}/{n} "
          f"= {disp.get('ESCALATE', 0) / n:.1%}")
    print(f"  gate found sufficient evidence on {gate_sufficient}/{n} rows")

    print("\n=== PRECISION: did it drop curated data? ===")
    print(f"  DROP count {drops}/{n} = {drops / n:.2%}   (these rows are curated; each one is a false positive)")
    for c, why, g in drop_examples:
        print(f"    DROPPED {c!r}  refuter={why!r}  gold={g}")

    print("\n=== numeric_conflict on REAL evidence: its first precision test at scale ===")
    print(f"  fired on {fired}/{with_ev} pairs = {fired / max(with_ev, 1):.2%}")
    for c, a, b, g in fire_examples:
        print(f"    {c[:66]!r}")
        print(f"      claim says {a!r} vs evidence says {b!r}   gold={g}")

    print("\n=== the honest reading ===")
    cov = (n - disp.get("ESCALATE", 0)) / n
    print(f"  COVERAGE: {cov:.1%} decided. The corpus needs {n_gold_decided / n:.1%} decided. So the gap")
    print(f"  between what a validator-only pipeline can do and what the task requires is roughly")
    print(f"  {(n_gold_decided / n - cov) * 100:.0f} percentage points of a real corpus.")
    print(f"  That gap is the actual work remaining, and it is a FACT-VERIFICATION gap, not a routing one:")
    print(f"  nothing in this suite decides whether 'Sean Connery refused an Apple commercial' is true.")
    if drops == 0:
        print(f"\n  PRECISION: zero false drops on {n} curated rows. That is the property that had to hold,")
        print(f"  and it holds on real data rather than on my fixtures.")
    else:
        print(f"\n  PRECISION: {drops} false drops on curated data. Every one is a bug, not a tuning knob.")
    if fired == 0:
        print(f"  numeric_conflict never fired on this evidence -- it is mostly prose Q/A with few")
        print(f"  comparable quantities, which is consistent with 'it is narrow' and says nothing about")
        print(f"  whether it is RIGHT when it does fire.")
    else:
        print(f"  numeric_conflict fired {fired} times. This is the number to inspect by hand: it was")
        print(f"  validated on five triples I wrote, and five pairs is not evidence of precision on prose.")
    print("\n  AND THE LIMIT: this corpus measures claim TRUTH. Our dispositions measure ROW QUALITY.")
    print("  They are different questions, and conflating them would make the DROP column look like a")
    print("  verdict column. What transfers is precision and coverage, not accuracy.")
    return 0


def run_self_test() -> bool:
    print("logos.corpus_sizing - SELF TEST")
    ok = True

    sample = ("You are a strict fact-checker.\n\nCLAIM: One mile is 1.5 kilometres.\n\n"
              "EVIDENCE:\nQ: What is a mile A: 1.609344 km\n\n"
              "Answer with exactly one word from: SUPPORTED, REFUTED.")
    claim, ev = parse_verifier_av(sample)
    good = claim == "One mile is 1.5 kilometres."
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the CLAIM parses out of the prompt: {claim!r}")

    good = "1.609344" in ev and "Answer with" not in ev
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the EVIDENCE parses and stops before the instruction line")

    # --- a prompt with no CLAIM must not silently return the header as the claim
    c2, e2 = parse_verifier_av("no markers here at all")
    good = c2 == "" and e2 == ""
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a prompt without markers returns empty, not the header text")

    # --- a wrapped claim still parses
    c3, _ = parse_verifier_av("CLAIM: a claim that\nspans lines\nEVIDENCE:\nx\n\nAnswer with one word")
    good = "spans lines" in c3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a claim spanning lines is captured whole")

    if not os.path.exists(CORPUS):
        print(f"  [SKIP] corpus not present: {CORPUS}")
    else:
        rows = load_corpus(CORPUS, 5)
        good = len(rows) == 5 and all(r["claim"] for r in rows)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] loads real rows with non-empty claims "
              f"({len(rows)} read)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline coverage and precision on a real labelled corpus.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure(args.limit, args.corpus)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
