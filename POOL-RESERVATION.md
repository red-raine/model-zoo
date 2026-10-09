# Model Zoo — per-app key reservation (2026-10-08)

> **Per stack-master policy:** one key PER APP, assigned models, revoke old
> keys. This file is the reservation manifest the stack master uses to create
> keys + checkout/checkin quota via the new n8n pipelines.

## Key currently in use (to be replaced)
- **My tools read `LITELLM_MASTER_KEY`** from `B:\ai-stack\.env` (env-based,
  passed to detached runs). That's the shared `opencoder` master key —
  **should be revoked once app keys exist.**

## The per-app keys I want (3 apps)

### 1. `pool-model-zoo` — the WASM app + assembler + frontend
| Model | Lane | Why |
|-------|------|-----|
| `nyx-qwen25-1.5b` | nyx (free/our GPU) | assembler LLM plan (fast, verified) |
| `nyx-thinkingcap-v2` | nyx | teacher distillation |
| `google-gemini-3-6-flash` | Google free | star quality (TRACE 0.562) |
| `google-gemini-3-1-flash-lite` | Google free | 2nd lane |
| `free-flash-pool` | pool | rotating rapid lane |
| `wayward-minion` | pool | rotating rapid lane |
| `or-mistralai-mistral-nemo` | paid $0.03 | fallback |
| `openai/gpt-oss-20b` | paid $0.09 | fallback + red-team target |

### 2. `pool-sweep` — corpus/bench (long-running sweeps)
| Model | Why |
|-------|-----|
| `google-gemini-2-5-flash`, `3-5-flash-lite`, `3-7-flash` | full-budget corpus (verified 100%/372) |
| `free-flash-pool`, `wayward-minion`, `agnes-flash` | pools |
| `nyx-qwen2.5-7b` | our GPU big lane |
| fallback: `or-mistralai-mistral-nemo`, `openai/gpt-oss-20b` | |

### 3. `pool-eval` — red-team + TRACE evals (short, frequent)
| Model | Why |
|-------|-----|
| `nyx-qwen25-1.5b` | red-team target (F4 detected here) |
| `openai/gpt-oss-20b` | the gptoss harness's target |
| `google-gemini-3-6-flash` | TRACE eval reference |
| fallback: `or-mistralai-mistral-nemo` | |

## Budget per key (tiered, $1/mo)
```
0.10 : daily   → 0.50 : weekly  → 1.00 : monthly   (cheap-paid fallback kicks in past the free bucket)
```

## Warden commands (for stack master — exact)
```bash
# app: model-zoo
lite keys generate --alias pool-model-zoo \
  --models "nyx-qwen25-1.5b,nyx-thinkingcap-v2,google-gemini-3-6-flash,google-gemini-3-1-flash-lite,free-flash-pool,wayward-minion,or-mistralai-mistral-nemo,openai/gpt-oss-20b" \
  --max-budget 1.00 --budget-duration 30d \
  --model-max-budget 0.10:daily --model-max-budget 0.50:weekly --model-max-budget 1.00:monthly
rest_key <key> --budget-fallbacks "or-mistralai-mistral-nemo,openai/gpt-oss-20b"

# app: sweep
lite keys generate --alias pool-sweep \
  --models "google-gemini-2-5-flash,google-gemini-3-5-flash-lite,google-gemini-3-7-flash,free-flash-pool,wayward-minion,agnes-flash,nyx-qwen2.5-7b,or-mistralai-mistral-nemo" \
  --max-budget 1.00 --budget-duration 30d --model-max-budget 0.10:daily

# app: eval
lite keys generate --alias pool-eval \
  --models "nyx-qwen25-1.5b,openai/gpt-oss-20b,google-gemini-3-6-flash,or-mistralai-mistral-nemo" \
  --max-budget 1.00 --budget-duration 30d --model-max-budget 0.10:daily
```

## Checkout/checkin (n8n quota pipeline)
The n8n pipeline checks out a key+quota before a run and checks in after:
- **checkout**: verify key `/v1/models` 200 + key budget remaining ≥ needed; mark
  in-use in a ledger.
- **run**: my tools set the key via env (`LITELLM_API_KEY=<app key>`), not master.
- **checkin**: budget/spend snapshot → ledger, mark idle.
- TODO: implement `tools/tm/key_lease.py` + n8n workflow (checkout/checkin).

## Revoke list (after migration)
`opencoder` (master, all-team-models) — the currently-shared key my tools read.
Keep `alice1-poolside` (owned by alice-bot lane), `model-bench-*` (bench).