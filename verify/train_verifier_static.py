"""train_verifier_static.py — model2vec static-embedding 4-way classifier.

WHY
---
`train_verifier_cls.py` trains deberta-v3-base (86M, torch) and got 0.780 acc / 0.613
macro-F1 on AVeriTeC dev. That is good but heavy: torch, GPU, ~20 min.

model2vec distils a sentence transformer into **static token embeddings** — up to 50x
smaller and 500x faster, CPU-only, numpy in the base package. If a static classifier
approaches DeBERTa's numbers it becomes the fast tier of the cascade and a natural fit
for the ONNX/llama.cpp deployment lane in GOLDEN-MASTER-PLAN.md tier 2.

The comparison is the point. A static model that gets 0.70 at 500x the speed is a better
engineering answer than 0.78 in torch, for a pipeline that has to judge thousands of rows.

USAGE
-----
    python tools/verify/train_verifier_static.py --self-test
    python tools/verify/train_verifier_static.py --encoder minishlab/potion-base-32M
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")
LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
SHORT = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 2, "CONFLICTING": 3}

CLAIM_ONLY = os.environ.get("CLAIM_ONLY") == "1"


def load(split: str, limit: int = 0) -> tuple[list[str], list[int], int]:
    p = os.path.join(DATA, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    xs, ys, skipped = [], [], 0
    for r in rows:
        gold = SHORT.get(r.get("completion") or "")
        claim = (r.get("claim") or "").strip()
        body = r.get("prompt") or ""
        ev = body.split("EVIDENCE:")[-1] if "EVIDENCE:" in body else ""
        ev = ev.split("Answer with exactly one word")[0].strip()
        if gold is None or not claim or not ev:
            skipped += 1
            continue
        xs.append(claim if CLAIM_ONLY else f"{claim} [SEP] {ev}")
        ys.append(gold)
    return xs, ys, skipped


def macro_f1(gold: list[int], pred: list[int]) -> tuple[float, list[float], list[list[int]]]:
    from sklearn.metrics import f1_score, confusion_matrix
    per = f1_score(gold, pred, average=None, labels=list(range(4)),
                   zero_division=0).round(3).tolist()
    mac = float(f1_score(gold, pred, average="macro", labels=list(range(4)),
                         zero_division=0))
    cm = confusion_matrix(gold, pred, labels=list(range(4))).tolist()
    return mac, per, cm


def run_self_test() -> bool:
    print("train_verifier_static.py - SELF TEST")
    ok = True
    xs, ys, sk = load("train", 200)
    print(f"  built {len(xs)} train examples (skipped {sk})")
    good = len(xs) > 150 and len(xs) == len(ys)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] example build")
    good = len(set(ys)) >= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] label diversity "
          f"{dict(sorted(collections.Counter(ys).items()))}")
    try:
        from model2vec.train import StaticModelForClassification
        good = True
    except Exception as e:
        good = False
        print(f"    import error: {e}")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] model2vec StaticModelForClassification importable")
    mac, per, cm = macro_f1([0, 1, 2, 3], [0, 1, 2, 3])
    good = mac == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] macro_f1 helper ({mac})")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="model2vec static 4-way verifier.")
    ap.add_argument("--encoder", default="minishlab/potion-base-32M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=r"B:\datasets\forge\verifier-av\static-model")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    from model2vec.train import StaticModelForClassification

    tr_x, tr_y, sk_tr = load("train", args.limit)
    dv_x, dv_y, sk_dv = load("dev", args.limit)
    print(f"[static] train {len(tr_x)} (skip {sk_tr}) · dev {len(dv_x)} (skip {sk_dv})")
    print(f"[static] encoder {args.encoder}")
    print(f"[static] train labels {dict(sorted(collections.Counter(tr_y).items()))}")
    print(f"[static] dev   labels {dict(sorted(collections.Counter(dv_y).items()))}")

    t0 = time.time()
    clf = StaticModelForClassification.from_pretrained(path=args.encoder)
    print(f"[static] loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    clf.fit(tr_x, tr_y)
    train_s = time.time() - t0

    t0 = time.time()
    pred = list(clf.predict(dv_x))
    infer_s = time.time() - t0

    acc = sum(1 for g, p in zip(dv_y, pred) if g == p) / max(len(dv_y), 1)
    mac, per, cm = macro_f1(dv_y, [int(p) for p in pred])
    print(f"\n[static] fit {train_s:.1f}s · predict {infer_s:.2f}s for {len(dv_x)} "
          f"({1000*infer_s/max(len(dv_x),1):.2f} ms/claim)")
    print(f"[static] DEV acc {acc:.4f}  macro_f1 {mac:.4f}")
    print(f"[static] per-class F1 {per}  ({['Sup','Ref','NEI','Con']})")
    print(f"[static] confusion {cm}")
    print(f"\n[static] COMPARISON")
    print(f"    majority baseline            0.6100 acc")
    print(f"    deberta-v3-base + Linear(4)  0.7800 acc / 0.6130 macro-F1  (~20 min, torch)")
    print(f"    model2vec static             {acc:.4f} acc / {mac:.4f} macro-F1  "
          f"({train_s:.0f}s fit, no torch)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump({"encoder": args.encoder, "acc": acc, "macro_f1": mac,
                   "per_class_f1": per, "confusion": cm,
                   "fit_s": round(train_s, 1), "predict_s": round(infer_s, 2),
                   "claim_only": CLAIM_ONLY}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
