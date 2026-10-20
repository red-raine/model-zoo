"""d7_research_task.py — D7 E2E: run ONE real research task through the lab.

Uses the working free lanes (pool-model-zoo key) to research a genuinely
useful question, then writes a structured artifact. This is the "bang for
buck" proof: the model zoo doing real work.

Usage:
    python d7_research_task.py --question "..." --out <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"E:\vibe_coding\dev\bitnet runner\tools\tm")
from key_store import get_key  # noqa: E402

GW = "http://localhost:4000/v1/chat/completions"
APP = "pool-model-zoo"

# Free-first lane order (verified working with the pool-model-zoo key)
LANES = ["free-flash-pool", "nyx-qwen25-1.5b", "google-free"]

DEFAULT_Q = (
    "What are the highest-impact, lowest-cost techniques for distilling a "
    "small (300M-1B) language model from a larger teacher when you have very "
    "limited fine-tuning data (under 1,000 examples) and only CPU inference? "
    "Cover: data selection/augmentation, loss choices (KD vs SFT vs DPO), "
    "LoRA hyperparameters, and the most common failure modes. Be concrete and "
    "cite mechanisms, not opinions."
)


def ask(lane: str, question: str, key: str, max_tokens: int = 1200,
        retries: int = 2) -> tuple[str, str]:
    body = json.dumps({
        "model": lane,
        "messages": [
            {"role": "system", "content": "You are a precise ML research assistant. "
                                          "Answer concretely with mechanisms, tradeoffs, and "
                                          "concrete numbers where known. No filler."},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens, "temperature": 0.3,
    }).encode()
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            GW, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode())
            return lane, (d["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            last = f"<{lane} error: {str(exc)[:60]}>"
            time.sleep(3 * (attempt + 1))
    return lane, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--out", default=r"B:\datasets\local\d7-research-result.md")
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()

    key = get_key(APP)
    print(f"[d7] app={APP} question={args.question[:70]}...", flush=True)

    results = []
    for lane in LANES:
        t0 = time.time()
        lane, text = ask(lane, args.question, key, args.max_tokens)
        ok = not text.startswith("<")
        print(f"[d7] {lane:22} {'OK' if ok else 'FAIL'} {time.time()-t0:6.1f}s len={len(text)}", flush=True)
        results.append({"lane": lane, "ok": ok, "text": text})

    good = [r for r in results if r["ok"] and len(r["text"]) > 100]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# D7 E2E research task\n",
          f"**Question:** {args.question}\n",
          f"**Lanes:** {', '.join(r['lane'] for r in results)} "
          f"({len(good)}/{len(results)} answered)\n"]
    for r in results:
        md.append(f"\n## {r['lane']}\n")
        md.append(r["text"] or "(no answer)")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[d7] wrote {out} ({len(good)}/{len(results)} lanes answered)", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())