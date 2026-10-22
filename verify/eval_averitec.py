"""eval_averitec.py — 4-way evaluation on AVeriTeC dev, with the baselines attached.

WHY THE BASELINES ARE IN HERE
-----------------------------
On HaluEval a two-line substring rule scored 97.5% and beat a 20B judge, because the
dataset was 95.5% extractive. The lesson was not "HaluEval is bad", it was: **run the
dumbest possible baseline first**, or a meaningless number looks like a result.

So this harness always reports, alongside the model:
    majority   — always predict the most common dev label
    substring  — predict SUPPORTED iff the claim appears in the evidence
    random     — uniform over the 4 labels

A model that does not beat `majority` has learned nothing about verification.

4-WAY LABELS
------------
SUPPORTED / REFUTED / NOT_ENOUGH_EVIDENCE / CONFLICTING.
Abstention is a *prediction*, not a non-answer, here: NEI is a real class. That is
different from the 2-way harness, where abstaining had to be scored separately.

USAGE
-----
    python tools/verify/eval_averitec.py --self-test
    python tools/verify/eval_averitec.py --adapter B:/.../adapter --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factcheck import wilson_ci  # noqa: E402

DEV = r"B:\datasets\forge\verifier-av\dev.jsonl"
BASE = "LiquidAI/LFM2.5-350M"
LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE", "CONFLICTING")


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def load(path: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def baselines(rows: list[dict]) -> dict:
    n = len(rows)
    gold = [r["completion"] for r in rows]
    dist = collections.Counter(gold)
    maj_label, maj_n = dist.most_common(1)[0]

    sub = 0
    for r in rows:
        ev = norm(r["prompt"].split("EVIDENCE:")[-1])
        cl = norm(r.get("claim"))
        pred = "SUPPORTED" if (cl and cl in ev) else "REFUTED"
        if pred == r["completion"]:
            sub += 1

    rng = random.Random(42)
    rnd = sum(1 for g in gold if rng.choice(LABELS) == g)

    return {
        "n": n,
        "label_dist": dict(dist),
        "majority": {"label": maj_label, "accuracy": round(maj_n / n, 4),
                     "ci95": list(wilson_ci(maj_n, n))},
        "substring": {"accuracy": round(sub / n, 4), "ci95": list(wilson_ci(sub, n))},
        "random": {"accuracy": round(rnd / n, 4), "ci95": list(wilson_ci(rnd, n))},
    }


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = set(gold) | set(pred)
    scores = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        scores.append(f1)
    return round(sum(scores) / max(len(scores), 1), 4)


def parse_label(raw: str) -> str | None:
    """Longest-label-first so NOT_ENOUGH_EVIDENCE is not eaten by a shorter match."""
    u = (raw or "").upper()
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab in u:
            return lab
    if "NOT ENOUGH" in u or "NEI" in u:
        return "NOT_ENOUGH_EVIDENCE"
    return None


def eval_model(rows: list[dict], adapter: str, limit: int = 0) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True,
                                             dtype=torch.float32)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()

    gold, pred, unknown = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            # MUST match training: the model's native chat template.
            text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512)
            out = m.generate(**ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            p = parse_label(gen)
            if p is None:
                unknown += 1
                p = "REFUTED"          # neutral fallback, counted in accuracy honestly
            gold.append(r["completion"])
            pred.append(p)
            if i % 50 == 0:
                acc = sum(1 for g, q in zip(gold, pred) if g == q) / len(gold)
                print(f"    {i}/{len(rows)} acc={acc:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "ci95": list(wilson_ci(correct, len(rows))),
        "macro_f1": macro_f1(gold, pred),
        "unparsed": unknown,
        "seconds": round(time.time() - t0, 1),
        "ms_per_claim": round(1000 * (time.time() - t0) / len(rows), 1),
        "confusion": {g: dict(collections.Counter(
            p for gg, p in zip(gold, pred) if gg == g)) for g in sorted(set(gold))},
    }


def run_self_test() -> bool:
    print("eval_averitec.py - SELF TEST")
    ok = True
    cases = [("SUPPORTED", "SUPPORTED"), ("REFUTED.", "REFUTED"),
             ("NOT_ENOUGH_EVIDENCE", "NOT_ENOUGH_EVIDENCE"),
             ("CONFLICTING", "CONFLICTING"),
             ("not enough evidence here", "NOT_ENOUGH_EVIDENCE"),
             ("", None), ("banana", None)]
    for raw, want in cases:
        got = parse_label(raw)
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] parse {raw[:26]!r} -> {got}")
    # NOT_ENOUGH_EVIDENCE must not be truncated to a shorter label
    good = parse_label("NOT_ENOUGH_EVIDENCE") == "NOT_ENOUGH_EVIDENCE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] longest-label-first ordering")
    rows = load(DEV, 20)
    b = baselines(rows)
    good = b["n"] == 20 and 0 < b["majority"]["accuracy"] <= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] baselines computed: "
          f"majority={b['majority']['accuracy']} substring={b['substring']['accuracy']}")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="4-way AVeriTeC evaluation with baselines.")
    ap.add_argument("--dev", default=DEV)
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    rows = load(args.dev, args.limit)
    b = baselines(rows)
    print(f"[eval] dev n={b['n']} labels={b['label_dist']}")
    print(f"[eval] BASELINES")
    print(f"    majority  {b['majority']['accuracy']:.4f}  {b['majority']['ci95']} "
          f"(always '{b['majority']['label']}')")
    print(f"    substring {b['substring']['accuracy']:.4f}  {b['substring']['ci95']}")
    print(f"    random    {b['random']['accuracy']:.4f}  {b['random']['ci95']}")

    out = {"baselines": b}
    if args.adapter:
        print(f"[eval] model: {args.adapter}")
        m = eval_model(rows, args.adapter, args.limit)
        out["model"] = m
        print(f"[eval] MODEL accuracy {m['accuracy']:.4f}  {m['ci95']}  "
              f"macroF1 {m['macro_f1']}  {m['ms_per_claim']} ms/claim  "
              f"unparsed {m['unparsed']}")
        print(f"[eval] confusion: {json.dumps(m['confusion'])}")
        beat = m["accuracy"] - b["majority"]["accuracy"]
        print(f"[eval] vs majority: {beat:+.4f}  "
              f"=> {'BEATS' if beat > 0 else 'DOES NOT BEAT'} the baseline")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"[eval] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
