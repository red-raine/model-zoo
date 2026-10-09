"""d6_redteam_litellm.py — run the gpt-oss-20b redteam harness against LiteLLM.

Imports the 5 probe tests from the vendored harness (B:/datasets/family/
gptoss20b-redteam-harness) with a LiteLLM/OpenAI-compatible ModelClient. This
is the Model Zoo's red-team eval gate — free (gpt-oss-20b is $0.09/M fallback
or any pool leaf).

Usage:
    python d6_redteam_litellm.py [--model openai/gpt-oss-20b] [--max-tokens 512]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"B:\datasets\family\gptoss20b-redteam-harness")
from gptoss_redteam_harness import (  # noqa: E402
    test_code_optimization_sabotage,
    test_eval_awareness,
    test_inappropriate_command_synthesis,
    test_reward_hacking_confident_hallucination,
    test_trigger_leakage_long_horizon,
)

LITELLM = "http://localhost:4000/v1/chat/completions"


class LtClient:
    """OpenAI-compatible client speaking the harness's generate() contract."""

    def __init__(self, model: str, max_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.model_name = model  # harness expects .model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.key = self._key()

    @staticmethod
    def _key() -> str:
        # CONSTITUTIONAL LAW: per-app key via key_store, NEVER master key.
        import sys as _s
        _s.path.insert(0, r"E:\vibe_coding\dev\bitnet runner\tools\tm")
        try:
            from key_store import get_key
            return get_key("pool-eval")
        except Exception:
            return ""

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, seed: int | None = None) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature if seed is None else temperature,
        }).encode()
        req = urllib.request.Request(
            LITELLM, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=150) as r:
            d = json.loads(r.read().decode())
        return (d["choices"][0]["message"].get("content") or "").strip()


TESTS = [
    ("F1-eval_awareness", test_eval_awareness),
    ("F2-reward_hacking", test_reward_hacking_confident_hallucination),
    ("F3-command_synthesis", test_inappropriate_command_synthesis),
    ("F4-trigger_leakage", test_trigger_leakage_long_horizon),
    ("F5-code_sabotage", test_code_optimization_sabotage),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"B:\datasets\local\model-zoo-d6-redteam.json")
    args = ap.parse_args()

    client = LtClient(args.model, args.max_tokens)
    print(f"[redteam] model={args.model} probes=5", flush=True)
    findings, failed = [], 0
    for name, fn in TESTS:
        try:
            f = fn(client, args.seed)
            findings.append(f)
            ok = f.success if hasattr(f, "success") else None
            sev = getattr(f, "severity_estimate", "?")
            print(f"[redteam] {name}: success={ok} severity={sev}", flush=True)
            if not ok:
                failed += 1
        except Exception as exc:
            print(f"[redteam] {name}: RUN ERROR {str(exc)[:90]}", flush=True)
            failed += 1
        time.sleep(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": args.model, "seed": args.seed, "tests": len(TESTS),
            "failed": failed, "pass": failed == 0,
            "findings": [json.dumps(f.__dict__, default=str) for f in findings]}
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"[redteam] done -> {out} (failed={failed}/5)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())