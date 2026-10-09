"""live_crosscheck.py — TRACE-reward cross-check on REAL RUNG-0 v6 traces.

Master-plan Next item: "a small TRACE-reward cross-check on real traces via
posttrain.py with injected live hooks". GRPO consumes the actual v6 corpus
(via sample_fn) and scores each with the real TRACE Toulmin metric. Runs
offline (no proxy calls) so it never interferes with the Google-paced sweep.

Output: per-round reward stats + how many would pass the trace_filter gate.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

TM = Path(r"E:\vibe_coding\dev\bitnet runner\tools\tm")
SCORING = Path(r"E:\vibe_coding\dev\bitnet runner\tools\scoring")
sys.path.insert(0, str(TM))
sys.path.insert(0, str(SCORING))

from trace_eval import trance_report  # noqa: E402


def load_v6_traces(limit: int = 500) -> dict:
    """Load real RUNG-0 v6 traces grouped by model + prompt."""
    base = Path(r"E:\vibe_coding\dev\bitnet runner\truthmatrix\state\rung0_v2\rung0_v6")
    prompts: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(base / "**" / "traces_*.jsonl"), recursive=True)):
        for line in Path(f).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            model = d.get("model", "?")
            pid = d.get("prompt_id") or d.get("id") or "?"
            text = (d.get("response") or "") + "\n" + (d.get("reasoning") or "")
            key = f"{model}::{pid}"
            prompts.setdefault(key, []).append(text)
            if sum(len(v) for v in prompts.values()) >= limit:
                return prompts
    return prompts


def real_reward(text: str, prompt: str) -> float:
    """Real TRACE Toulmin reward (grades evidence structure)."""
    try:
        return float(trance_report(text)["trace_score"])
    except Exception:
        return 0.0


def real_sample(prompt: str, round_idx: int) -> list[str]:
    """Deterministic real-corpus sampler: best-effort key match, else any."""
    pool = CORPUS.get(prompt)
    if not pool:
        # no exact key — grab a deterministic sample across all groups
        keys = sorted(CORPUS.keys())
        if not keys:
            return ["<no real trace>"]
        pick = keys[(round_idx + len(prompt)) % len(keys)]
        pool = CORPUS[pick]
    size = max(1, len(pool))
    return [pool[(round_idx + len(prompt) + i) % len(pool)] for i in range(size)]


CORPUS: dict[str, list[str]] = {}


def main():
    global CORPUS
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    CORPUS = load_v6_traces(limit=n)
    print(f"[live] loaded {len(CORPUS)} real (model::prompt) trace groups", flush=True)

    from grpo import GRPOConfig, GRPOLoop

    cfg = GRPOConfig(rounds=2, prompts_per_round=20, group_size=3)
    loop = GRPOLoop(config=cfg, reward_fn=real_reward, sample_fn=real_sample)
    s = loop.run()

    print(f"[live] GRPO rounds={s.get('rounds')} examples={s.get('examples_total')}", flush=True)
    rewards = [ex.chosen_reward for ex in loop.examples]
    if rewards:
        print(f"[live] reward mean={statistics.mean(rewards):.3f} "
              f"p25={sorted(rewards)[int(0.25*len(rewards))]:.3f} "
              f"p90={sorted(rewards)[int(0.9*len(rewards))-1]:.3f}", flush=True)
        above = [r for r in rewards if r >= 0.55]
        print(f"[live] >= trace_filter(0.55): {len(above)}/{len(rewards)} "
              f"({100*len(above)/len(rewards):.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())