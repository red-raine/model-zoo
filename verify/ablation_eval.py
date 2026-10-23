"""ablation_eval.py — the guardrail test: does the model read the evidence, or recite a prior?

THE TEST (REAL, arXiv:2609.08943)
---------------------------------
LLM verifiers "rely more on their parametric knowledge than on the evidence provided". The
countermeasure is counterfactual evidence supervision, and the diagnostic is brutally simple:

    run the model normally
    run it again with the evidence DELETED
    if the verdict does not change, the model was never using the evidence

A flip rate near zero means we have a prior with a citation attached, not a verifier, and every
accuracy number above it means something other than what it looks like.

A SECOND CONTROL THAT MOST VERSIONS OF THIS TEST MISS
-----------------------------------------------------
A model that flips on *any* perturbation also fails — it would rather be sensitive than
grounded. So this runs three conditions:

    full      claim [SEP] evidence            (the real input)
    ablated   claim only                      (REAL: evidence deleted)
    placebo   claim [SEP] unrelated sentence  (a distractor that says nothing about the claim)

Evidence-grounding requires **full != ablated** and ideally **full ~ placebo-with-irrelevant**,
but the decisive comparison is ablated. The placebo column exists to separate "reads evidence"
from "flips when text changes".

USAGE (on Sol, inside the unsloth-intel image)
    python tools/verify/ablation_eval.py --model /tmp/data/nli-post2 --out /out
"""

from __future__ import annotations

import argparse
import collections
import json
import os

def _default_data() -> str:
    """Pick the data root that actually exists, so the same file runs locally and in the
    container. A hard-coded '/data/verifier-av' default made the local self-test fail with a
    mangled '/data/verifier-av\\dev.jsonl' path on Windows."""
    cands = ["/data/verifier-av", r"B:\datasets\forge\verifier-av"]
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


DATA = os.environ.get("VERIFIER_DATA", _default_data())
SHORT = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 2, "CONFLICTING": 3}
LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting"]
MAX_LEN = 512
PLACEBO = ("The local council met on Tuesday to discuss routine road maintenance and "
           "confirmed that bin collection schedules would be unchanged this year.")


def build(split: str = "dev") -> tuple[list[str], list[str], list[str], list[int]]:
    """Return full, claim-only, and placebo texts plus labels."""
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"{split}.jsonl"),
                                        encoding="utf-8") if l.strip()]
    full, abl, plb, gold = [], [], [], []
    for r in rows:
        g = SHORT.get(r.get("completion") or "")
        claim = (r.get("claim") or "").strip()
        body = r.get("prompt") or ""
        ev = body.split("EVIDENCE:")[-1] if "EVIDENCE:" in body else ""
        ev = ev.split("Answer with exactly one word")[0].strip()
        if g is None or not claim or not ev:
            continue
        full.append(f"{claim} [SEP] {ev}")
        abl.append(claim)
        plb.append(f"{claim} [SEP] {PLACEBO}")
        gold.append(g)
    return full, abl, plb, gold


def run_self_test() -> bool:
    print("ablation_eval.py - SELF TEST")
    ok = True
    full, abl, plb, gold = build("dev")
    good = len(full) == len(abl) == len(plb) == len(gold) and len(full) > 400
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] built {len(full)} rows in 3 conditions")
    good = all("[SEP]" in t for t in full) and all("[SEP]" not in t for t in abl)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ablated condition has no [SEP] / no evidence")
    good = all("[SEP]" in t and PLACEBO[:20] in t for t in plb[:5])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] placebo carries the distractor, not the evidence")
    good = all(0 <= g < 4 for g in gold)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] labels in range")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evidence-ablation flip test.")
    ap.add_argument("--model", required=False, default="")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    full, abl, plb, gold = build("dev")
    if args.limit:
        full, abl, plb, gold = (x[: args.limit] for x in (full, abl, plb, gold))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval().float()

    def predict(texts: list[str]) -> list[int]:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), args.bs):
                enc = tok(texts[i:i + args.bs], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt")
                out.extend(model(**enc).logits.argmax(-1).tolist())
        return [int(x) for x in out]

    p_full = predict(full)
    p_abl = predict(abl)
    p_plb = predict(plb)

    def acc(p):
        return sum(1 for a, b in zip(p, gold) if a == b) / len(gold)

    flip_real = [int(a != b) for a, b in zip(p_full, p_abl)]
    flip_plc = [int(a != b) for a, b in zip(p_full, p_plb)]
    rate_real = sum(flip_real) / len(flip_real)
    rate_plc = sum(flip_plc) / len(flip_plc)

    print(f"[abl] model {args.model} on {len(gold)} dev rows")
    print(f"[abl] accuracy   full {acc(p_full):.4f} · ablated {acc(p_abl):.4f} · "
          f"placebo {acc(p_plb):.4f}")
    print(f"[abl] FLIP RATE  evidence-deleted {rate_real:.4f} "
          f"({sum(flip_real)} rows) · placebo {rate_plc:.4f} ({sum(flip_plc)} rows)")
    if rate_real < 0.10:
        print("[abl] >>> RED FLAG: deleting the evidence barely changes the verdict. "
              "This is a prior model, not a verifier.")
    elif rate_real > 0.90:
        print("[abl] >>> WARNING: almost every verdict changes when evidence is removed; "
              "check the model is not just reading surface length.")
    else:
        print("[abl] flip rate is in a plausible evidence-dependent range")

    # per-class breakdown: which classes depend on evidence most?
    per = collections.Counter()
    per_n = collections.Counter()
    for i, g in enumerate(gold):
        per_n[LABELS[g]] += 1
        per[LABELS[g]] += flip_real[i]
    print("[abl] flips by gold label:")
    for lab in LABELS:
        if per_n[lab]:
            print(f"[abl]   {lab:26s} {per[lab]:3d}/{per_n[lab]:3d} "
                  f"({per[lab]/per_n[lab]:.3f})")

    # what does the ablated model collapse to? the classic 0.244-style fingerprint check
    dist = collections.Counter(p_abl)
    print(f"[abl] ablated prediction distribution: "
          f"{ {LABELS[k]: v for k, v in sorted(dist.items())} }")
    if max(dist.values()) / len(p_abl) > 0.85:
        print("[abl] >>> the ablated model has collapsed to one class, which is expected "
              "without evidence; the useful signal is the flip rate above")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.json"), "w") as fh:
        json.dump({"model": args.model, "n": len(gold),
                   "acc_full": acc(p_full), "acc_ablated": acc(p_abl),
                   "acc_placebo": acc(p_plb),
                   "flip_rate_evidence": rate_real, "flip_rate_placebo": rate_plc,
                   "flips_by_label": {LABELS[k]: per[k] for k in range(4) if per_n[k]},
                   "n_by_label": {LABELS[k]: per_n[k] for k in range(4) if per_n[k]},
                   "preds_full": p_full, "preds_ablated": p_abl, "preds_placebo": p_plb,
                   "gold": gold}, fh, indent=2)
    print(f"[abl] wrote {args.out}/ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
