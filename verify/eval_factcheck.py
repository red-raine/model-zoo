"""eval_factcheck.py — measure a fact-checker against a labelled set, with a baseline.

WHY THIS FILE EXISTS
--------------------
Everything built in `tools/verify/` so far was validated on 8–16 hand-picked cases.
That is not a measurement, and the field's own literature is blunt about it: a
published eval found ~half of claims changed verdict across three runs of identical
content, and factool ships a self-check baseline specifically so a pipeline has to
beat "just ask the model".

This harness does three things the anecdote could not:
  1. runs on a LABELLED set (HaluEval qa: right_answer vs hallucinated_answer),
  2. runs the SELF-CHECK BASELINE on the same items,
  3. reports accuracy, coverage, abstention and the DANGEROUS error rate
     (a hallucinated answer accepted as TRUE), with >=3 runs and mean+/-std.

The dangerous error is the headline number. A checker that abstains on everything
looks safe and is useless; a checker that accepts hallucinations is actively harmful.
Reporting only accuracy hides both.

LABELS
------
A verifier verdict maps to a prediction:
    SUPPORTED / SUPPORTED (wikidata)  -> TRUE
    REFUTED / CONTRADICTED            -> FALSE
    NO_EVIDENCE / UNKNOWN / CONFLICTING / NOT_CHECK_WORTHY -> ABSTAIN

Abstentions are NOT counted as correct. Coverage is reported separately so
"we only answered the easy half" cannot look like accuracy.

USAGE
-----
    python tools/verify/eval_factcheck.py --self-test
    python tools/verify/eval_factcheck.py --n 20 --runs 1
    python tools/verify/eval_factcheck.py --n 100 --runs 3 --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_check
import fact_verify
from lora_check import check_text, _chat, JUDGE, JUDGE_MODEL

HALUEVAL = r"B:\datasets\forge\downloads\halueval\qa.jsonl"

# verdict -> predicted class (None = abstain)
PRED = {
    "SUPPORTED": True,
    "CONTRADICTED": False,
    "REFUTED": False,
    "NO_EVIDENCE": None,
    "CONFLICTING": None,
    "NOT_CHECK_WORTHY": None,
    "UNKNOWN": None,
}


def load_items(path: str, n: int, seed: int = 42) -> list[dict]:
    """Balanced TRUE/FALSE items built from HaluEval qa.

    Each row gives a question with a `right_answer` and a `hallucinated_answer`.
    We turn both into standalone statements so the checker sees a claim, not a
    Q/A pair it can pattern-match.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    half = max(n // 2, 1)
    picked = rows[:half]

    items = []
    for r in picked:
        q = (r.get("question") or "").strip()
        ra = (r.get("right_answer") or "").strip()
        ha = (r.get("hallucinated_answer") or "").strip()
        if not (q and ra and ha):
            continue
        know = (r.get("knowledge") or "").strip()
        items.append({"claim": f"{q} {ra}".strip(), "gold": True,
                      "kind": "right", "question": q, "knowledge": know})
        items.append({"claim": f"{q} {ha}".strip(), "gold": False,
                      "kind": "hallucinated", "question": q, "knowledge": know})
    rng.shuffle(items)
    return items[:n]


BASELINE_PROMPT = """Is the following answer factually correct?

Answer with exactly one word: TRUE or FALSE.

ANSWER: {claim}"""


def baseline_predict(claim: str, endpoint: str, model: str) -> bool | None:
    raw = _chat(BASELINE_PROMPT.format(claim=claim[:1500]), endpoint, model,
                max_tokens=512)
    u = (raw or "").upper()
    t, f = "TRUE" in u, "FALSE" in u
    if t and not f:
        return True
    if f and not t:
        return False
    return None


def naive_predict(claim: str, endpoint: str, model: str) -> tuple[bool | None, str]:
    """The pre-LORA checker: sentence split -> search the claim -> one judge call."""
    r = fact_verify.verify_text(claim, endpoint, model, 1)
    res = (r.get("results") or [])
    if not res:
        return None, "no_claim"
    v = res[0].get("verdict", "UNKNOWN")
    return PRED.get(v, None), v


def verifier_predict(claim: str, endpoint: str, model: str) -> tuple[bool | None, str]:
    r = check_text(claim, endpoint, model, max_claims=1)
    claims = r.get("claims") or []
    if not claims:
        return None, "no_claim"
    c = claims[0]
    v = c.get("claim_verdict", "NO_EVIDENCE")
    # a deterministic channel is authoritative when present
    return PRED.get(v, None), v


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Needed because the honest answer to "accuracy 0.000" depends entirely on n.
    At n=2 that is [0.00, 0.84]; at n=40 it is [0.00, 0.09]. Reporting the point
    estimate without the interval is how a 2-decision result gets read as a finding.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def score(pairs: list[tuple[bool, bool | None]]) -> dict:
    """pairs = (gold, predicted_or_None)"""
    answered = [(g, p) for g, p in pairs if p is not None]
    correct = sum(1 for g, p in answered if g == p)
    halluc = [g for g, p in answered if not g]              # gold FALSE
    halluc_accepted = sum(1 for g, p in answered if (not g) and p is True)
    true_items = [g for g, p in answered if g]
    true_accepted = sum(1 for g, p in answered if g and p is True)
    return {
        "n": len(pairs),
        "answered": len(answered),
        "coverage": round(len(answered) / max(len(pairs), 1), 4),
        "accuracy_on_answered": round(correct / len(answered), 4) if answered else None,
        "accuracy_ci95": list(wilson_ci(correct, len(answered))) if answered else None,
        "abstained": len(pairs) - len(answered),
        # the two numbers that actually matter
        "hallucinations_accepted": halluc_accepted,
        "hallucinations_total": len(halluc),
        "dangerous_rate": round(halluc_accepted / len(halluc), 4) if halluc else None,
        "dangerous_ci95": list(wilson_ci(halluc_accepted, len(halluc))) if halluc else None,
        "true_accepted": true_accepted,
        "true_total": len(true_items),
        "recall_true": round(true_accepted / len(true_items), 4) if true_items else None,
    }


def knowledge_predict(item: dict, endpoint: str, model: str) -> tuple[bool | None, str]:
    """Judge the answer against the row's OWN supporting passage.

    HaluEval qa is question+answer, not a declarative claim, so the open-web
    extractor returns no_claim on most rows (measured: 14/20). That made the
    coverage-0.000 run an ADAPTER artifact rather than a verifier result.

    HaluEval ships a `knowledge` passage per row, so the honest, cheap test of
    the weak link -- the judge -- is: given this evidence, is this answer right?
    No retrieval, no decomposition, no open-web variance. Just the judge.
    """
    claim = item.get("claim", "")
    ev = item.get("knowledge", "")
    if not (claim and ev):
        return None, "no_knowledge"
    snip = {"title": "given knowledge", "content": ev[:1800], "url": ""}
    import fact_verify
    v = fact_verify.judge_pair(claim, snip, endpoint, model)
    return PRED.get(v, None), v


def run_once(items: list[dict], which: str, endpoint: str, model: str) -> dict:
    pairs = []
    detail = []
    for it in items:
        if which == "baseline":
            p = baseline_predict(it["claim"], endpoint, model)
            v = "TRUE" if p else ("FALSE" if p is False else "ABSTAIN")
        elif which == "knowledge":
            p, v = knowledge_predict(it, endpoint, model)
        elif which == "naive":
            p, v = naive_predict(it["claim"], endpoint, model)
        else:
            p, v = verifier_predict(it["claim"], endpoint, model)
        pairs.append((it["gold"], p))
        detail.append({"gold": it["gold"], "pred": p, "verdict": v,
                       "kind": it["kind"], "claim": it["claim"][:110]})
    s = score(pairs)
    s["detail"] = detail
    return s


def aggregate(runs: list[dict]) -> dict:
    def ms(key):
        xs = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if not xs:
            return None, None
        return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4)
                                               if len(xs) > 1 else 0.0)
    out = {}
    for k in ("coverage", "accuracy_on_answered", "dangerous_rate", "recall_true"):
        m, s = ms(k)
        out[k] = {"mean": m, "std": s}
    out["runs"] = len(runs)
    return out


# --- self-test -------------------------------------------------------------
def run_self_test() -> bool:
    print("eval_factcheck.py - SELF TEST")
    ok = True

    # scoring must not reward abstention
    pairs = [(True, None), (False, None), (True, True), (False, False)]
    s = score(pairs)
    good = s["accuracy_on_answered"] == 1.0 and s["coverage"] == 0.5 and s["abstained"] == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] abstention not counted as correct: "
          f"acc={s['accuracy_on_answered']} cov={s['coverage']}")

    # dangerous error must be surfaced, and accuracy must not hide it
    pairs = [(False, True), (True, True)]
    s = score(pairs)
    good = s["dangerous_rate"] == 1.0 and s["accuracy_on_answered"] == 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hallucination accepted flagged: "
          f"dangerous={s['dangerous_rate']} acc={s['accuracy_on_answered']}")

    # all-abstain must give coverage 0, not accuracy 0
    s = score([(True, None), (False, None)])
    good = s["coverage"] == 0.0 and s["accuracy_on_answered"] is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all-abstain -> coverage=0, acc=None")

    # Wilson CI: the whole point is that n=2 tells you almost nothing
    lo2, hi2 = wilson_ci(0, 2)
    lo40, hi40 = wilson_ci(0, 40)
    # Wilson(0,2) upper bound is 0.658 (verified by hand); Wilson(0,40) is 0.088
    good = lo2 == 0.0 and 0.60 < hi2 < 0.70 and hi40 < 0.15
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Wilson CI shows the power problem: "
          f"0/2 -> [{lo2}, {hi2}], 0/40 -> [{lo40}, {hi40}]")

    # verdict mapping
    for v, want in [("SUPPORTED", True), ("REFUTED", False), ("CONTRADICTED", False),
                    ("NO_EVIDENCE", None), ("CONFLICTING", None)]:
        good = PRED.get(v) == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] map {v} -> {PRED.get(v)}")

    # item builder must balance
    items = load_items(HALUEVAL, 10)
    gs = [i["gold"] for i in items]
    good = len(items) == 10 and gs.count(True) == gs.count(False)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] balanced items: {len(items)} "
          f"true={gs.count(True)} false={gs.count(False)}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure fact-checkers against a labelled set.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--out")
    ap.add_argument("--systems", nargs="+",
                    default=["baseline", "knowledge", "naive", "verifier"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    items = load_items(HALUEVAL, args.n)
    print(f"[eval] {len(items)} items "
          f"({sum(1 for i in items if i['gold'])} true / "
          f"{sum(1 for i in items if not i['gold'])} false) x {args.runs} runs")

    results = {"items": len(items), "runs": args.runs,
               "judge": args.judge, "model": args.model, "by_system": {}}

    for which in args.systems:
        runs = []
        for i in range(args.runs):
            t0 = time.time()
            r = run_once(items, which, args.judge, args.model)
            r["seconds"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"  {which} run{i+1}: cov={r['coverage']} "
                  f"acc={r['accuracy_on_answered']} "
                  f"dangerous={r['dangerous_rate']} ({r['seconds']}s)", flush=True)
        results["by_system"][which] = {"aggregate": aggregate(runs), "runs": runs}

    print("\n" + "=" * 78)
    print(f"{'system':<12}{'coverage':>22}{'accuracy':>22}{'dangerous':>22}")
    print("-" * 78)
    for which, d in results["by_system"].items():
        a = d["aggregate"]

        def fmt(k):
            v = a.get(k, {})
            if v.get("mean") is None:
                return "-"
            return f"{v['mean']:.3f} +/- {v['std']:.3f}"
        print(f"{which:<12}{fmt('coverage'):>22}{fmt('accuracy_on_answered'):>22}"
              f"{fmt('dangerous_rate'):>22}")
        # per-run CIs make the power problem visible instead of hiding it
        for r in d["runs"]:
            if r.get("accuracy_ci95"):
                print(f"    run: answered {r['answered']}/{r['n']}  "
                      f"acc CI95 {r['accuracy_ci95']}  "
                      f"dangerous CI95 {r.get('dangerous_ci95')}")
    print("=" * 78)
    print("dangerous = a HALLUCINATED answer accepted as TRUE (lower is better)")
    print("coverage  = fraction where a verdict was given at all")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\n[eval] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
