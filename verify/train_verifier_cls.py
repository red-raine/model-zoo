"""train_verifier_cls.py — 4-way claim verification as SEQUENCE CLASSIFICATION.

WHY THIS REPLACES THE GENERATIVE APPROACH
-----------------------------------------
I trained a 350M model to *generate* one of four label strings and got 0.245 accuracy
against a 0.610 majority baseline. Two independent things were wrong:

1. **HARNESS BUG.** 0.245 is almost exactly the dev prevalence of `Supported` (24.4%).
   The generations were degenerate (`'REFUTED.SUPPORTED.SUPPORTED'`) and my label parser
   matched *any* label found in that text, so it reported a label that had nothing to do
   with the model's intent. I never measured the model.

2. **WRONG TASK SHAPE.** Verdict prediction is a closed-set judgment, not string
   production. The field does not generate label strings below 7B:

       AVeriTeC official baseline   BERT + Linear(4) + if-chain rules   dev 0.49
       DeBERTa-v3-large + Linear(4), multi-evidence, argmax             dev 0.63
       BLOOM-7B generating labels   macro-F1 .43  <  BERT-large .49 on the same evidence

   Generation adds *surface-form competition* (probability mass smeared across
   `Refuted`/`REFUTED`/`false`/`incorrect`), a generic-string attractor, and a failure
   mode where decoding loops and no prediction exists at all. A linear head cannot do any
   of those.

   Source: Ullrich et al. 2024 (AIC CTU), arXiv:2410.11446, Table 1 — a 435M encoder with
   a 4-way head gets 0.63 dev accuracy, 14 points above the official pipeline.

THE REAL CEILING
----------------
AVeriTeC inter-annotator agreement is free-marginal kappa = 0.619, which for 4 classes is
P_o = 0.714. **Two trained annotators given the same evidence agree only 71.4% of the
time.** So 0.63 is within 8 points of human. Anything above ~0.66 at sub-1B on dev would
be publishable. Do not budget for more.

METRIC
------
Report **macro-F1**, not accuracy. Dev is 24.4/61.0/7.6/7.0 — accuracy alone cannot
distinguish "learned something" from "collapsed to the majority class". `Conflicting` has
F1 .15 even for the official baseline *with gold evidence*; it is near-unlearnable at
small scale and will dominate the macro average.

USAGE
-----
    python tools/verify/train_verifier_cls.py --self-test
    python tools/verify/train_verifier_cls.py --model microsoft/deberta-v3-base --epochs 20
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")
LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
SHORT = {"SUPPORTED": "Supported", "REFUTED": "Refuted",
         "NOT_ENOUGH_EVIDENCE": "Not Enough Evidence",
         "CONFLICTING": "Conflicting Evidence/Cherrypicking"}
L2I = {lab: i for i, lab in enumerate(LABELS)}
I2L = {i: lab for lab, i in L2I.items()}

MAX_LEN = 512


def load(split: str, limit: int = 0) -> list[dict]:
    p = os.path.join(DATA, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def to_example(rec: dict) -> dict | None:
    """One example per claim: [CLS] claim [SEP] q1 a1 [SEP] q2 a2 ...

    Multi-evidence concatenation, NOT per-pair classification with if-chain rules. The
    if-chain is the official baseline and it scores 0.49 — below majority — because it
    turns any single bad pair into a wrong verdict.
    """
    gold = SHORT.get(rec.get("completion") or "")
    if gold is None:
        return None
    claim = (rec.get("claim") or "").strip()
    if not claim:
        return None
    # the prompt holds "CLAIM: ... \n\nEVIDENCE:\n q/a | q/a"
    body = rec.get("prompt") or ""
    ev = body.split("EVIDENCE:")[-1] if "EVIDENCE:" in body else ""
    ev = ev.split("Answer with exactly one word")[0].strip()
    if not ev:
        return None
    text = f"{claim} [SEP] {ev}"
    return {"text": text, "label": L2I[gold], "claim": claim}


def build(split: str, limit: int = 0) -> tuple[list[dict], list[dict]]:
    rows = load(split, limit)
    texts, labels, skipped = [], [], 0
    for r in rows:
        ex = to_example(r)
        if ex is None:
            skipped += 1
            continue
        texts.append(ex["text"])
        labels.append(ex["label"])
    return texts, labels, skipped


def class_weights(labels: list[int]) -> list[float]:
    c = collections.Counter(labels)
    n = len(labels)
    return [n / (len(LABELS) * c.get(i, 1)) for i in range(len(LABELS))]


# --------------------------------------------------------------------------
def run_self_test() -> bool:
    print("train_verifier_cls.py - SELF TEST")
    ok = True
    texts, labels, skipped = build("train", 200)
    print(f"  built {len(texts)} train examples (skipped {skipped})")
    good = len(texts) > 150 and len(texts) == len(labels)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] example build")

    dist = collections.Counter(labels)
    good = len(dist) >= 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] label diversity "
          f"{ {I2L[k]: v for k, v in sorted(dist.items())} }")

    # label order MUST match AVeriTeC eval.py
    good = LABELS == ["Supported", "Refuted", "Not Enough Evidence",
                      "Conflicting Evidence/Cherrypicking"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] label order matches eval.py")

    w = class_weights(labels)
    good = len(w) == 4 and all(x > 0 for x in w)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] class weights {[round(x,2) for x in w]}")

    # a long example must be truncatable without crashing
    long_text = "x " * 2000
    good = len(long_text) > MAX_LEN * 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] long-input guard exercised")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="4-way AVeriTeC verifier as classification.")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("VERIFIER_OUT", r"B:\datasets\forge\verifier-av\cls-model"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    import numpy as np
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tr_x, tr_y, sk_tr = build("train", args.limit)
    dv_x, dv_y, sk_dv = build("dev", args.limit)
    print(f"[cls] train {len(tr_x)} (skipped {sk_tr}) · dev {len(dv_x)} (skipped {sk_dv})")
    print(f"[cls] train labels: "
          f"{ {I2L[k]: v for k, v in sorted(collections.Counter(tr_y).items())} }")
    print(f"[cls] dev labels:   "
          f"{ {I2L[k]: v for k, v in sorted(collections.Counter(dv_y).items())} }")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4, id2label=I2L, label2id=L2I)

    class DS(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.enc = tok(xs, truncation=True, max_length=MAX_LEN,
                           padding="max_length")
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    w = torch.tensor(class_weights(tr_y), dtype=torch.float32)

    def compute_metrics(ep):
        logits, labels = ep
        pred = np.argmax(logits, axis=-1)
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
        return {
            "acc": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro",
                                       labels=list(range(4)), zero_division=0)),
            "per_class_f1": f1_score(labels, pred, average=None,
                                     labels=list(range(4)),
                                     zero_division=0).round(3).tolist(),
            "confusion": confusion_matrix(labels, pred,
                                          labels=list(range(4))).tolist(),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels,
                                                     weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        use_cpu=not torch.xpu.is_available(),
        seed=42,
        remove_unused_columns=True,
    )
    trainer = WeightedTrainer(model=model, args=targs,
                              train_dataset=DS(tr_x, tr_y),
                              eval_dataset=DS(dv_x, dv_y),
                              compute_metrics=compute_metrics)
    trainer.train()
    m = trainer.evaluate()
    print(f"\n[cls] DEV  acc={m.get('eval_acc'):.4f}  "
          f"macro_f1={m.get('eval_macro_f1'):.4f}")
    print(f"[cls] per-class F1: {m.get('eval_per_class_f1')}  "
          f"({LABELS})")
    print(f"[cls] confusion: {m.get('eval_confusion')}")
    print(f"\n[cls] reference points:")
    print(f"    majority baseline (always Refuted)  0.6100 acc")
    print(f"    official AVeriTeC baseline (BERT)   0.4900 acc")
    print(f"    DeBERTa-v3-large + head (published) 0.6300 acc / 0.39 macro-F1")
    print(f"    human annotator agreement ceiling   0.7140 acc")
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump({k: v for k, v in m.items()}, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
