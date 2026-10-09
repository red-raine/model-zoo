# Model Zoo reserved pool — for the WASM app + assembler + redteam gate

**User ask:** reserve free models for the model-zoo app with a rotating pool +
paid fallbacks when quota is exhausted. This is the reservation spec.

## The pool (free, rotating — in priority order)

| Lane | Models (all verified working 2026-10-07/08) | Notes |
|------|---------------------------------------------|-------|
| **nyx local (free, no quota)** | `nyx-thinkingcap-v2`, `nyx-qwen25-1.5b`, `nyx-qwen2.5-7b`, `nyx-gpt-oss-20b`, `nyx-phi-4-mini`, `nyx-granite-4.1-8b`, `nyx-gemma-4-e4b` | our own GPU box; **preferred lane** (user: "use nyx more"); thinkingcap = teacher |
| **Google tier-2 (free)** | `google-gemini-2-5-flash`, `google-gemini-3-1-flash-lite`, `google-gemini-3-5-flash-lite`, `google-gemini-3-6-flash`, `google-gemini-3-7-flash` | 150K/Unlimited RPD, ~2-4 RPM — slow steady lane |
| **AIHubMix free** | `coding-glm-5.1-free`, `coding-minimax-m2.7-free`, `wayward-coding`, `agnes-flash` | 20 RPM/50 RPD each; rapid lane |
| **Pools** | `free-flash-pool`, `wayward-minion` (rotate internally, $0) | proven 100% pass |

## Paid fallbacks (only if all free exhausted — $0.03-0.09/M)

| Model | $/M | Role |
|-------|-----|------|
| `or-mistralai-mistral-nemo` | 0.030 | general/summary/judge (registered alias) |
| `or-inclusionai-ling-3.0-flash` | 0.063 | super-cheap flash |
| `openai/gpt-oss-20b` | 0.090 | **the redteam target** + coding (also `gpt-oss-20b-free`, `nyx-gpt-oss-20b`, `ocloud-gpt-oss-20b`) |

> Note: last check `openai/gpt-oss-20b` returned 402 (combined budget
> exhausted) — so the free-first order matters; paid only after.

## Works for the whole app
- **WASM assembler LLM plan**: `nyx-qwen25-1.5b` (fast, free, verified)
- **Teacher distillation**: `nyx-thinkingcap-v2` (D5 corpus built with it)
- **Red-team eval gate**: `nyx-qwen25-1.5b` (F4 leakage detected) — swap
  `--model openai/gpt-oss-20b` when quota clears
- **Sweep/corpus**: `google-gemini-3-6-flash` (0.562 TRACE star) +
  `free-flash-pool`

## LiteLLM reservation (warden action)
Create a virtual key for the app named `pool-model-zoo`:

```bash
lite keys generate --alias pool-model-zoo \
  --models "nyx-thinkingcap-v2,nyx-qwen25-1.5b,google-gemini-3-6-flash,google-gemini-3-1-flash-lite,free-flash-pool,wayward-minion,coding-glm-5.1-free,or-mistralai-mistral-nemo,openai/gpt-oss-20b" \
  --max-budget 1.00 --budget-duration 30d \
  --model-max-budget 0.10:daily --model-max-budget 0.50:weekly --model-max-budget 1.00:monthly
# fallbacks (cheap paid if free quota exhausted):
rest_key <key> --budget-fallbacks "or-mistralai-mistral-nemo,openai/gpt-oss-20b"
```

Rotation: LiteLLM `cost-based-routing` already on → zero-cost leaves rotate
first, paid fires only past the free bucket. New app pools follow the existing
`alice1-poolside` pattern at the key level.