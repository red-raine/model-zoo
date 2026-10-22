"""build_averitec.py — AVeriTeC -> a 4-way verifier training set, deterministic-first.

WHY AVeriTeC AND NOT HaluEval
-----------------------------
HaluEval qa is 95.5% extractive: the right answer is a verbatim substring of the
knowledge passage, so a two-line string rule scores 97.5% and beats a 20B judge. It
measures matching, not verification.

AVeriTeC is 0.0% extractive (dev) / 0.1% (train): the claim never appears in its own
evidence, and the evidence answers are gathered from the open web with a declared
`answer_type`. It is the instrument that can actually measure verification.

LABEL SCHEME — the full 4-way, deliberately
-------------------------------------------
    Supported                        evidence establishes the claim
    Refuted                          evidence contradicts the claim
    Not Enough Evidence              nothing found that settles it
    Conflicting Evidence/Cherrypicking  sources disagree, or were cherry-picked

Keeping NEI and Conflicting is the point. A 2-way scheme has to fake them, and a
verifier that cannot abstain is dangerous — it will accept a hallucination because
"SUPPORTED" was the only positive class available.

DETERMINISTIC CHANNELS FIRST
----------------------------
Boolean answers and numeric claims can be settled exactly, for $0, with no model:
  * `math_verify` (sympy) on arithmetic in the claim
  * Wikidata SPARQL on entity-containment / capital-of claims
Only the residue needs a model. This module records which channel settled each claim so
the model's workload is measurable rather than assumed.

USAGE
-----
    python tools/verify/build_averitec.py --stats
    python tools/verify/build_averitec.py --out B:/datasets/forge/verifier-av
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_verify import verify_text as math_verify  # noqa: E402

SRC = r"B:\datasets\forge\downloads\averitec"
OUT_DIR = r"B:\datasets\forge\verifier-av"

LABELS = ("Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking")
SHORT = {"Supported": "SUPPORTED", "Refuted": "REFUTED",
         "Not Enough Evidence": "NOT_ENOUGH_EVIDENCE",
         "Conflicting Evidence/Cherrypicking": "CONFLICTING"}

PROMPT = """You are a strict fact-checker. Given the CLAIM and the EVIDENCE gathered about it, assign exactly one verdict.

Verdicts:
- SUPPORTED: the evidence establishes the claim.
- REFUTED: the evidence contradicts the claim.
- NOT_ENOUGH_EVIDENCE: nothing in the evidence settles it either way.
- CONFLICTING: the evidence disagrees with itself, or supports only a cherry-picked part.

Use ONLY the evidence. Do not use your own knowledge.

CLAIM: {claim}

EVIDENCE:
{evidence}

Answer with exactly one word from: SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE, CONFLICTING."""

_WS = re.compile(r"\s+")


def norm(s, cap: int = 1600) -> str:
    return _WS.sub(" ", str(s or "").strip())[:cap]


def questions_of(row: dict) -> list[dict]:
    qs = row.get("questions")
    if isinstance(qs, str):
        try:
            qs = ast.literal_eval(qs)
        except Exception:
            return []
    return qs if isinstance(qs, list) else []


def evidence_text(row: dict, max_answers: int = 8) -> tuple[str, dict]:
    """Flatten the evidence answers, and count the answer types."""
    parts, types = [], {}
    for q in questions_of(row):
        qt = norm(q.get("question"), 200)
        for a in (q.get("answers") or []):
            at = str(a.get("answer_type") or "?")
            types[at] = types.get(at, 0) + 1
            ans = norm(a.get("answer"), 400)
            if ans:
                parts.append(f"Q: {qt} A: {ans}")
            if len(parts) >= max_answers:
                break
        if len(parts) >= max_answers:
            break
    return " | ".join(parts), types


def deterministic_channel(claim: str) -> tuple[str | None, str]:
    """Settle what can be settled exactly. Returns (verdict or None, channel)."""
    mv = math_verify(claim)
    if mv.get("verdict") == "REFUTED":
        return "REFUTED", "sympy"
    try:
        from fact_verify import wikidata_claim
        wd = wikidata_claim(claim)
        if wd:
            v = "SUPPORTED" if wd[0] == "SUPPORTED" else "REFUTED"
            return v, "wikidata"
    except Exception:
        pass
    return None, ""


def build(split: str, limit: int = 0) -> tuple[list[dict], dict]:
    path = os.path.join(SRC, f"{split}.json")
    rows = json.load(open(path, encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    out, stats = [], {"split": split, "rows": len(rows), "kept": 0,
                      "labels": {}, "answer_types": {}, "channels": {},
                      "deterministic": 0, "agree_with_gold": 0}
    for r in rows:
        claim = norm(r.get("claim"), 500)
        gold = r.get("label")
        if not claim or gold not in SHORT:
            continue
        ev, types = evidence_text(r)
        if not ev:
            continue
        for k, v in types.items():
            stats["answer_types"][k] = stats["answer_types"].get(k, 0) + v
        rec = {
            "prompt": PROMPT.format(claim=claim, evidence=ev),
            "completion": SHORT[gold],
            "claim": claim,
            "gold_long": gold,
            "evidence_chars": len(ev),
            "claim_date": r.get("claim_date"),
            "speaker": r.get("speaker"),
            "source": f"averitec/{split}",
        }
        # deterministic pass, recorded but NOT used to overwrite the gold label
        v, ch = deterministic_channel(claim)
        if v:
            stats["deterministic"] += 1
            stats["channels"][ch] = stats["channels"].get(ch, 0) + 1
            if v == SHORT[gold]:
                stats["agree_with_gold"] += 1
            rec["deterministic_verdict"] = v
            rec["deterministic_channel"] = ch
            rec["deterministic_agrees"] = (v == SHORT[gold])
        stats["labels"][SHORT[gold]] = stats["labels"].get(SHORT[gold], 0) + 1
        out.append(rec)
    stats["kept"] = len(out)
    return out, stats


def report(stats: dict) -> None:
    print(f"  rows={stats['rows']} kept={stats['kept']}")
    print(f"  labels: {stats['labels']}")
    print(f"  answer_types: {stats['answer_types']}")
    if stats["deterministic"]:
        pct = 100 * stats["agree_with_gold"] / stats["deterministic"]
        print(f"  deterministic channel settled {stats['deterministic']} claims "
              f"{stats['channels']} — agreed with gold {stats['agree_with_gold']} ({pct:.1f}%)")
    else:
        print("  deterministic channel settled 0 claims on this split")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build AVeriTeC 4-way verifier data.")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    print("=== AVeriTeC train ===")
    train, ts = build("train", args.limit)
    report(ts)
    print("=== AVeriTeC dev ===")
    dev, ds = build("dev", args.limit)
    report(ds)

    if args.stats:
        return 0

    os.makedirs(args.out, exist_ok=True)
    for name, part in (("train", train), ("dev", dev)):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p} ({len(part)} rows)")

    manifest = {
        "source": "AVeriTeC (MichSchli/AVeriTeC, GitHub raw)",
        "licence": "CC BY-NC 4.0 — non-commercial. Operator: MIT/Stanford PhD research, not a commercial product.",
        "label_scheme": "4-way: SUPPORTED / REFUTED / NOT_ENOUGH_EVIDENCE / CONFLICTING",
        "extractiveness": "dev 0.0%, train 0.1% (claim never appears in its own evidence)",
        "train_stats": ts, "dev_stats": ds,
        "note": ("Deterministic verdicts are recorded as metadata and are NOT used to "
                 "overwrite the gold label. They are a measurement of how much the free "
                 "channels can settle, not a source of training labels."),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"manifest -> {os.path.join(args.out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
