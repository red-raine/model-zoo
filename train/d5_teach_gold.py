"""d5_teach_gold.py — D5 gate: nyx thinkingcap-as-teacher gold generation.

Uses nyx-thinkingcap-v2 (via LiteLLM) as the TEACHER to re-generate rich,
structured traces on the golden prompt set (the distill-to-tiny input we've
been building toward). Writes a DVC-able gold file + TRACE-validates samples.

Usage:
    python d5_teach_gold.py --prompts 20 --out <path>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

SCORING = Path(r"E:\vibe_coding\dev\bitnet runner\tools\scoring")
sys.path.insert(0, str(SCORING))
from trace_eval import trance_report  # noqa: E402

GOLDEN = Path(r"B:\datasets\factory\truthmatrix\golden\prompts-tune.jsonl")
DEFAULT_OUT = Path(r"B:\datasets\local\model-zoo-d5-teach.jsonl")
LITELLM = "http://localhost:4000/v1/chat/completions"
MODEL = "nyx-thinkingcap-v2"
RPM = 30  # nyx local = no provider 429s; keep it modest


def _key() -> str:
    # CONSTITUTIONAL LAW: per-app key via key_store, NEVER master key.
    from key_store import get_key
    try:
        return get_key("pool-model-zoo")
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc


def teach(prompt: str, model: str, key: str, retries: int = 2) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": prompt + "\n\nReason step-by-step with evidence; end with answer: <X>."}],
        "max_tokens": 2048, "temperature": 0.3,
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            LITELLM, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                d = json.loads(r.read().decode())
            return {"prompt": prompt, "response": d["choices"][0]["message"].get("content") or "",
                    "model": model, "finish": d["choices"][0].get("finish_reason"),
                    "tokens": d.get("usage", {}).get("completion_tokens", 0)}
        except Exception as exc:
            if attempt == retries - 1:
                return {"prompt": prompt, "response": f"<teach-error: {str(exc)[:60]}>",
                        "model": model, "finish": "error", "tokens": 0}
            time.sleep(5 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    key = _key()
    prompts = []
    with open(GOLDEN, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = (d.get("prompt") or "").strip()
            if p:
                prompts.append(p[:500])
    prompts = prompts[: args.prompts]
    print(f"[d5] teacher={args.model} prompts={len(prompts)} out={args.out}", flush=True)

    rows, start = [], time.time()
    for i, p in enumerate(prompts):
        row = teach(p, args.model, key)
        rows.append(row)
        if i and i % 5 == 0:
            print(f"[d5] {i}/{len(prompts)}  ({time.time()-start:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # TRACE-validate the non-empty answers
    scores = [float(trance_report(r["response"])["trace_score"])
              for r in rows if r["response"] and not r["response"].startswith("<teach")]
    print(f"[d5] done in {time.time()-start:.0f}s | rows={len(rows)} | "
          f"TRACE mean={statistics.mean(scores):.3f} (n={len(scores)}) | -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())