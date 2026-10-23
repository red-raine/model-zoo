"""embed_frozen.py — cache frozen-backbone embeddings so heads can be cross-fitted cheaply.

WHY EMBEDDINGS
--------------
The label-error experiment needs to train MANY small heads on DISJOINT data subsets. Doing
that end-to-end through a transformer would cost ~30 min per head. Freezing the backbone and
caching its pooled output turns each head into a few seconds of numpy, which is what makes a
proper cross-fitted design affordable at all.

WHICH BACKBONE, AND WHY IT MATTERS
----------------------------------
`microsoft/deberta-v3-base` **as released** — never fine-tuned on AVeriTeC labels by us.
That is the whole point: if the encoder has already seen the labels, its features encode
them, and any head we train on top inherits that. An earlier experiment in this project used
two models both fine-tuned on AVeriTeC train and then treated their agreement as independent
evidence; it was not, and the correlation was a property of the setup, not of ensembles.

Output: .npz with cls + mean pooled vectors and the labels/texts, one file per split.

USAGE
-----
    python tools/verify/embed_frozen.py --self-test
    python tools/verify/embed_frozen.py --max-len 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DATA = os.environ.get("VERIFIER_DATA", r"B:\datasets\forge\verifier-av")
SHORT = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 2, "CONFLICTING": 3}
NUM_LABELS = 4


def build(split: str, limit: int = 0) -> tuple[list[str], list[int]]:
    p = os.path.join(DATA, f"{split}.jsonl")
    xs, ys = [], []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        gold = SHORT.get(r.get("completion") or "")
        claim = (r.get("claim") or "").strip()
        body = r.get("prompt") or ""
        ev = body.split("EVIDENCE:")[-1] if "EVIDENCE:" in body else ""
        ev = ev.split("Answer with exactly one word")[0].strip()
        if gold is None or not claim or not ev:
            continue
        xs.append(f"{claim} [SEP] {ev}")
        ys.append(gold)
        if limit and len(xs) >= limit:
            break
    return xs, ys


def run_self_test() -> bool:
    print("embed_frozen.py - SELF TEST")
    ok = True
    # 40 rows, not 5: the first five dev rows happen to all be `Refuted`, so a
    # label-diversity assertion on 5 rows fails on data order rather than on code.
    xs, ys = build("dev", 40)
    good = len(xs) == 40 and all("[SEP]" in t for t in xs) and len(set(ys)) >= 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] dev build {len(xs)} rows, {len(set(ys))} labels")
    good = all(0 <= y < NUM_LABELS for y in ys)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] labels in range")
    try:
        import torch
        import transformers
        good = True
    except Exception as e:  # noqa: BLE001
        good = False
        print(f"    import error: {e}")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] torch + transformers importable")
    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cache frozen backbone embeddings.")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "emb-frozen.npz"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().float()
    print(f"[emb] {args.model} max_len {args.max_len} (frozen, no head, no labels seen)")

    out: dict[str, np.ndarray] = {}
    meta: dict[str, list[str]] = {}
    for split in ("train", "dev"):
        xs, ys = build(split, args.limit)
        cls_feats, mean_feats = [], []
        with torch.no_grad():
            for i in range(0, len(xs), args.bs):
                batch = xs[i:i + args.bs]
                enc = tok(batch, truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt")
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                cls_feats.append(h[:, 0, :].numpy())
                mean_feats.append(((h * mask).sum(1) / mask.sum(1)).numpy())
                if i % (args.bs * 40) == 0:
                    print(f"[emb] {split} {i}/{len(xs)}", flush=True)
        out[f"X_{split}_cls"] = np.vstack(cls_feats).astype(np.float32)
        out[f"X_{split}_mean"] = np.vstack(mean_feats).astype(np.float32)
        out[f"y_{split}"] = np.asarray(ys, dtype=np.int64)
        meta[split] = xs
        print(f"[emb] {split}: {out[f'X_{split}_cls'].shape}")

    out["text_train"] = np.array(meta["train"], dtype=object)
    out["text_dev"] = np.array(meta["dev"], dtype=object)
    np.savez_compressed(args.out, **out)
    print(f"[emb] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
