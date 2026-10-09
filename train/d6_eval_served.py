"""d6_eval_served.py — D6 gate: TRACE-eval a *served* local model.

Scores whatever llama.cpp :8081 (or any OpenAI-compat endpoint) serves, on a
held-out TRACE probe set — the final quality gate before "ship". Free, local.

Usage:
    python d6_eval_served.py --endpoint http://localhost:8081/v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

SCORING = Path(r"E:\vibe_coding\dev\bitnet runner\tools\scoring")
sys.path.insert(0, str(SCORING))
from trace_eval import trance_report  # noqa: E402

PROBES = [
    "Explain why the sky is blue, with evidence.",
    "What are the 3 main causes of the French Revolution? Justify each.",
    "Compare Bayesian and frequentist statistics, with warrants.",
    "Why does water freeze at 0C? Give evidence and one counterexample.",
    "Explain the pigeonhole principle with a backing example.",
    "How does a transformer attention head work? Qualify your claims.",
]


def call(endpoint: str, prompt: str, tmo: int = 120) -> str:
    body = json.dumps({"model": "", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 512, "temperature": 0.2}).encode()
    req = urllib.request.Request(endpoint + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=tmo) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"].get("content") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8081/v1")
    ap.add_argument("--out", default=r"B:\datasets\local\model-zoo-d6-eval.json")
    args = ap.parse_args()

    print(f"[d6] endpoint={args.endpoint} probes={len(PROBES)}", flush=True)
    rows, scores = [], []
    for p in PROBES:
        try:
            r = call(args.endpoint, p)
        except Exception as exc:
            r = f"<served-error: {str(exc)[:70]}>"
        rows.append({"prompt": p, "response": r})
        if not r.startswith("<"):
            scores.append(float(trance_report(r)["trace_score"]))
            print(f"[d6] TRACE={scores[-1]:.3f} len={len(r)}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({"endpoint": args.endpoint, "n": len(rows),
                               "trace_mean": round(statistics.mean(scores), 3) if scores else None,
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"[d6] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())