"""train_verifier_nli.py — post-train the CTU AIC 4-way AVeriTeC NLI encoder.

WHY THIS MODEL
--------------
`ctu-aic/deberta-v3-large-AVeriTeC-nli` is DeBERTa-v3-large fine-tuned by the CTU AIC team
on the SAME task (config `_name_or_path` ends in `nli_4concat`, i.e. 4-way classification
over concatenated evidence) and it is published under MIT. Two consequences:

1. **Zero pretraining FLOPs.** We inherit AVeriTeC-specific training instead of starting from
   an NLI/generic checkpoint and hoping 8 epochs on 2.5k rows is enough.
2. **Zero-shot it is a real baseline**, so this file measures it BEFORE training. That is the
   control. The published reference for this family is 0.63 dev acc / 0.39 macro-F1.

THE TRAP THIS FILE EXISTS TO AVOID
----------------------------------
The checkpoint's head is `LABEL_0..LABEL_3` with no label names. Our order is
`[Supported, Refuted, Not Enough Evidence, Conflicting]` and there is **no guarantee** they
match. Reading a zero-shot number without checking the mapping is exactly how three earlier
results collapsed to 0.244 — the `Supported` prevalence. So zero-shot is scored twice: raw,
and under the best label permutation found by the Hungarian algorithm on the confusion
matrix. If the permutation is not the identity, the raw number is meaningless and we say so.

USAGE
-----
    python tools/verify/train_verifier_nli.py --self-test
    python tools/verify/train_verifier_nli.py --eval-only --out /out          # zero-shot
    python tools/verify/train_verifier_nli.py --calib 500 --dump-logits --out /out
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")
MODEL = os.environ.get("VERIFIER_MODEL", "ctu-aic/deberta-v3-large-AVeriTeC-nli")
LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
SHORT = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 2, "CONFLICTING": 3}
L2I = {lab: i for i, lab in enumerate(LABELS)}
I2L = {i: lab for lab, i in L2I.items()}
MAX_LEN = 512
CLAIM_ONLY = os.environ.get("CLAIM_ONLY") == "1"


def load(split: str, limit: int = 0) -> list[dict]:
    p = os.path.join(DATA, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def build(split: str, limit: int = 0) -> tuple[list[str], list[int], int]:
    """Same input format as train_verifier_cls.py: `claim [SEP] evidence`.

    Kept byte-identical on purpose -- the whole point is a controlled comparison against the
    deberta-v3-bedbase number, so the only variable that may change is the backbone.
    """
    texts, labels, skipped = [], [], 0
    for r in load(split, limit):
        gold = SHORT.get(r.get("completion") or "")
        claim = (r.get("claim") or "").strip()
        body = r.get("prompt") or ""
        ev = body.split("EVIDENCE:")[-1] if "EVIDENCE:" in body else ""
        ev = ev.split("Answer with exactly one word")[0].strip()
        if gold is None or not claim or not ev:
            skipped += 1
            continue
        texts.append(f"{claim} [SEP] {ev}" if not CLAIM_ONLY else claim)
        labels.append(gold)
    return texts, labels, skipped


def class_weights(labels: list[int]) -> list[float]:
    c = collections.Counter(labels)
    n = len(labels)
    return [n / (len(LABELS) * c.get(i, 1)) for i in range(len(LABELS))]


def metrics(gold: list[int], pred: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return {
        "acc": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro",
                                   labels=list(range(4)), zero_division=0)),
        "per_class_f1": f1_score(gold, pred, average=None, labels=list(range(4)),
                                 zero_division=0).round(3).tolist(),
        "confusion": confusion_matrix(gold, pred, labels=list(range(4))).tolist(),
    }


def best_permutation(cm: list[list[int]]) -> tuple[list[int], int]:
    """Hungarian on the confusion matrix: which model label should map to which gold label.

    Returns (perm, hits) where perm[model_label] = gold_label. If `perm` is not the identity,
    the checkpoint's LABEL_n order does not match ours and every raw number about it is noise.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a = np.asarray(cm, dtype=float)
    r, c = linear_sum_assignment(-a)          # maximise the trace
    perm = [0] * 4
    for i, j in zip(r, c):
        perm[int(i)] = int(j)
    return perm, int(sum(a[i, j] for i, j in zip(r, c)))


def apply_perm(pred, perm):
    return [perm[int(p)] for p in pred]


def run_self_test() -> bool:
    print("train_verifier_nli.py - SELF TEST")
    ok = True
    texts, labels, skipped = build("dev", 100)
    good = len(texts) == 100 and len(labels) == len(texts)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dev build {len(texts)} (skip {skipped})")
    good = all("[SEP]" in t for t in texts)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] claim [SEP] evidence format preserved")

    cm = [[100, 5, 0, 0], [7, 200, 1, 0], [0, 2, 40, 3], [1, 0, 4, 30]]
    perm, hits = best_permutation(cm)
    good = perm == [0, 1, 2, 3] and hits == 370
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] identity perm on diagonal matrix {perm}")

    # a rotated confusion matrix must produce a non-identity permutation
    cm2 = [[5, 100, 0, 0], [200, 7, 1, 0], [2, 0, 30, 3], [0, 1, 4, 30]]
    perm2, _ = best_permutation(cm2)
    good = perm2 != [0, 1, 2, 3] and perm2[0] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] detects rotated label order {perm2}")

    good = MODEL.count("/") == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] model id {MODEL}")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Post-train the CTU AIC AVeriTeC encoder.")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=2, help="gradient accumulation (effective bs)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--calib", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.environ.get("VERIFIER_OUT", "/out"))
    ap.add_argument("--eval-only", action="store_true", help="zero-shot control, no training")
    ap.add_argument("--dump-logits", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    if not os.path.isdir(args.out):
        try:
            os.makedirs(args.out, exist_ok=True)
        except OSError:
            pass

    import numpy as np
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tr_x, tr_y, sk_tr = build("train", args.limit)
    dv_x, dv_y, sk_dv = build("dev", args.limit)
    cal_x, cal_y = [], []
    if args.calib and args.calib > 0:
        order = list(range(len(tr_x)))
        random.Random(args.seed).shuffle(order)
        held = set(order[: args.calib])
        keep = [i for i in range(len(tr_x)) if i not in held]
        cal_x = [tr_x[i] for i in sorted(held)]
        cal_y = [tr_y[i] for i in sorted(held)]
        tr_x = [tr_x[i] for i in keep]
        tr_y = [tr_y[i] for i in keep]
    print(f"[nli] model {args.model}")
    print(f"[nli] fit {len(tr_x)} · calib {len(cal_x)} · dev {len(dv_x)}")
    print(f"[nli] dev labels "
          f"{ {I2L[k]: v for k, v in sorted(collections.Counter(dv_y).items())} }")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4, id2label=I2L, label2id=L2I,
        ignore_mismatched_sizes=True).float()

    class DS(torch.utils.data.Dataset):
        def __init__(self, xs, ys, pad=False):
            self.enc = tok(xs, truncation=True, max_length=MAX_LEN,
                           padding="max_length" if pad else True)
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            it = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            it["labels"] = torch.tensor(self.ys[i])
            return it

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to=[], use_cpu=not torch.xpu.is_available(),
        # fp16 is what produced NaN eval_loss and an all-class-0 collapse on deberta-v3-base;
        # a 435M model is more exposed to it, not less.
        fp16=False, bf16=False, seed=args.seed, remove_unused_columns=True)

    w = torch.tensor(class_weights(tr_y), dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            wl = w.to(device=out.logits.device, dtype=out.logits.dtype)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=wl)
            return (loss, out) if return_outputs else loss

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=DS(tr_x, tr_y),
                              eval_dataset=DS(dv_x, dv_y))

    # ------------------------------------------------------- CONTROL: zero-shot on dev
    zs = trainer.predict(DS(dv_x, dv_y))
    zlog = np.asarray(zs.predictions)
    zpred = zlog.argmax(-1).tolist()
    zs_m = metrics(dv_y, zpred)
    perm, hits = best_permutation(zs_m["confusion"])
    perm_pred = apply_perm(zpred, perm)
    perm_m = metrics(dv_y, perm_pred)
    print(f"\n[nli] === ZERO-SHOT CONTROL (no training at all) ===")
    print(f"[nli] raw      acc {zs_m['acc']:.4f}  macro_f1 {zs_m['macro_f1']:.4f}")
    print(f"[nli] raw per-class F1 {zs_m['per_class_f1']}")
    print(f"[nli] raw confusion {zs_m['confusion']}")
    print(f"[nli] label permutation model->ours {perm}   identity={perm == [0,1,2,3]}")
    print(f"[nli] permuted acc {perm_m['acc']:.4f}  macro_f1 {perm_m['macro_f1']:.4f}")
    print(f"[nli] permuted per-class F1 {perm_m['per_class_f1']}")
    if perm != [0, 1, 2, 3]:
        print("[nli] WARNING: the checkpoint's LABEL_n order is NOT ours. Only the permuted "
              "number means anything.")
    out = {"model": args.model, "zero_shot": zs_m, "zero_shot_permuted": perm_m,
           "permutation": perm, "permutation_is_identity": perm == [0, 1, 2, 3]}

    if args.eval_only:
        with open(os.path.join(args.out, "zero_shot.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        # Dump per-row zero-shot predictions too: the error-finder wants a SECOND independent
        # model's opinion per row, and the zero-shot checkpoint is a genuinely independent
        # one (we never trained on it).
        if args.dump_logits:
            with open(os.path.join(args.out, "logits.jsonl"), "w", encoding="utf-8") as fh:
                for x, y, p, row in zip(dv_x, dv_y, zpred, zlog.tolist()):
                    fh.write(json.dumps({"split": "dev", "label": int(y), "pred": int(p),
                                         "logits": [round(v, 6) for v in row],
                                         "text": x}, ensure_ascii=False) + "\n")
            print(f"[nli] zero-shot per-row predictions -> {args.out}/logits.jsonl")
        print(f"\n[nli] eval-only: wrote {args.out}/zero_shot.json")
        return 0

    # ---------------------------------------------------------------------- post-training
    print(f"\n[nli] post-training on {len(tr_x)} rows, {args.epochs} epochs, "
          f"bs {args.bs} x ga {args.ga}")
    trainer.train()
    m = trainer.evaluate()
    print(f"\n[nli] POST-TRAINED dev acc {m.get('eval_loss', 0):.4f} loss")
    pt = trainer.predict(DS(dv_x, dv_y))
    plog = np.asarray(pt.predictions)
    ppred = plog.argmax(-1).tolist()
    pt_m = metrics(dv_y, ppred)
    print(f"[nli] dev acc {pt_m['acc']:.4f}  macro_f1 {pt_m['macro_f1']:.4f}")
    print(f"[nli] per-class F1 {pt_m['per_class_f1']}")
    print(f"[nli] confusion {pt_m['confusion']}")
    out["post_trained"] = pt_m
    out["fit_n"] = len(tr_x)

    # ---------------------------------------------------- the comparison that matters
    print(f"\n[nli] === COMPARISON (same data, same 500-row calib carve, same dev) ===")
    print(f"    majority baseline                 0.6100 acc")
    print(f"    published DeBERTa-v3-large + head 0.6300 acc / 0.3900 macro-F1")
    print(f"    CTU AVeriTeC-nli ZERO-SHOT        {perm_m['acc']:.4f} acc / "
          f"{perm_m['macro_f1']:.4f} macro-F1   <- control")
    print(f"    deberta-v3-base post-trained      0.7440 acc / 0.5876 macro-F1  (our line)")
    print(f"    CTU AVeriTeC-nli POST-TRAINED     {pt_m['acc']:.4f} acc / "
          f"{pt_m['macro_f1']:.4f} macro-F1")

    try:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"[nli] checkpoint saved -> {args.out}")
    except Exception as e:  # noqa: BLE001
        print(f"[nli] WARN save failed: {e}")

    if args.dump_logits:
        rows = []
        for name, xs, ys in (("fit", tr_x, tr_y), ("calib", cal_x, cal_y),
                             ("dev", dv_x, dv_y)):
            if not xs:
                continue
            pr = trainer.predict(DS(xs, ys))
            lg = np.asarray(pr.predictions)
            pred = lg.argmax(-1)
            print(f"[nli] logits[{name}] n={len(xs)} "
                  f"acc={float((pred == np.asarray(ys)).mean()):.4f}")
            for x, y, p, row in zip(xs, ys, pred, lg.tolist()):
                rows.append({"split": name, "label": int(y), "pred": int(p),
                             "logits": [round(v, 6) for v in row], "text": x})
        lp = os.path.join(args.out, "logits.jsonl")
        with open(lp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[nli] logits dumped -> {lp} ({len(rows)} rows)")

    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
