"""build_verifier_data.py — turn HaluEval into an NLI training set for a small judge.

WHY THIS FILE EXISTS
--------------------
Measured state of the judge (HaluEval qa, n=40, evidence given):
    gpt-oss-20b Q4   accuracy 0.886 [0.741, 0.955]   dangerous 0.105   ~25 s/claim

That is a *general* 20B model doing a narrow classification task at 25 s/claim.
HaluEval ships 10,000 rows, each with a question, a `right_answer`, a
`hallucinated_answer`, and the `knowledge` passage that settles it. That is
**20,000 labelled (evidence, claim, label) triples** — enough to fine-tune a small
model to do this one job.

The bet: a 350M-1.2B model trained on 20k labelled pairs beats a 20B generalist on
this specific task, runs ~500x faster, and costs nothing. If it does not, that is
also a result worth having — it would mean the task genuinely needs scale.

LABEL SCHEME
------------
Two-way NLI, deliberately not three-way:
    knowledge + right_answer        -> SUPPORTED
    knowledge + hallucinated_answer -> CONTRADICTED

No NEUTRAL class, because HaluEval never supplies a passage that is merely
unrelated. The deployed verifier still needs an abstain path; that is handled by
the *threshold* on SUPPORTED probability at inference, not by a training class.
Adding a fake NEUTRAL class here would teach the model to hedge on everything.

USAGE
-----
    python tools/verify/build_verifier_data.py --stats
    python tools/verify/build_verifier_data.py --out B:/datasets/forge/verifier
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

HALUEVAL = r"B:\datasets\forge\downloads\halueval\qa.jsonl"
OUT_DIR = r"B:\datasets\forge\verifier"

PROMPT = """You are a strict fact-checker. Decide whether the PASSAGE supports or contradicts the ANSWER.

Rules:
- SUPPORTED: the passage states the same fact as the answer.
- CONTRADICTED: the passage states something incompatible with the answer.
- Use ONLY the passage. Not your own knowledge.
- If the passage is silent on the point, answer CONTRADICTED only when it states something incompatible; otherwise SUPPORTED if it agrees.

PASSAGE: {passage}

ANSWER: {answer}

Answer with exactly one word: SUPPORTED or CONTRADICTED."""

_WS = re.compile(r"\s+")


def norm(s: str, cap: int = 1800) -> str:
    return _WS.sub(" ", (s or "").strip())[:cap]


def load_rows(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def build(rows: list[dict], val_frac: float = 0.1, seed: int = 42) -> tuple[list, list]:
    pairs = []
    for r in rows:
        know = norm(r.get("knowledge"))
        q = norm(r.get("question"), 400)
        ra = norm(r.get("right_answer"), 400)
        ha = norm(r.get("hallucinated_answer"), 400)
        if not (know and ra and ha):
            continue
        # The answer alone can be terse ("Yes"). Prefixing the question makes it a
        # self-contained claim, which is what the deployed verifier receives.
        pairs.append({"prompt": PROMPT.format(passage=know, answer=f"{q} {ra}".strip()),
                      "completion": "SUPPORTED"})
        pairs.append({"prompt": PROMPT.format(passage=know, answer=f"{q} {ha}".strip()),
                      "completion": "CONTRADICTED"})
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]


def stats(pairs: list[dict]) -> dict:
    import collections
    labels = collections.Counter(p["completion"] for p in pairs)
    lens = sorted(len(p["prompt"]) + len(p["completion"]) for p in pairs)
    return {
        "n": len(pairs),
        "labels": dict(labels),
        "prompt_chars_p50": lens[len(lens) // 2] if lens else 0,
        "prompt_chars_p95": lens[int(len(lens) * 0.95)] if lens else 0,
        "prompt_chars_max": lens[-1] if lens else 0,
        "over_512_tokens_est": sum(1 for x in lens if x > 2000),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build NLI training data for a small verifier.")
    ap.add_argument("--src", default=HALUEVAL)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.src):
        print(f"MISSING {args.src}")
        return 1

    rows = load_rows(args.src)
    train, val = build(rows, args.val_frac)
    print(f"source rows      {len(rows)}")
    print(f"train            {len(train)}")
    print(f"val              {len(val)}")
    for name, part in (("train", train), ("val", val)):
        s = stats(part)
        print(f"  {name}: labels={s['labels']} "
              f"chars p50={s['prompt_chars_p50']} p95={s['prompt_chars_p95']} "
              f"max={s['prompt_chars_max']} >512tok={s['over_512_tokens_est']}")

    if args.stats:
        return 0

    os.makedirs(args.out, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p}")

    manifest = {
        "source": args.src,
        "task": "2-way NLI (SUPPORTED / CONTRADICTED)",
        "train": len(train), "val": len(val),
        "stats": {"train": stats(train), "val": stats(val)},
        "note": ("Derived from HaluEval qa (pminervini/HaluEval). Training data, not "
                 "evaluation data: the n=40 judge measurement used a different slice."),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {os.path.join(args.out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
