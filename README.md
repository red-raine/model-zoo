# model-zoo — the Model Zoo Builder

The unified family vision: **describe a model → get a built, traced, distilled,
evaluated, served artifact.** The agentic software factory glue.

```
WASM front-end (describe a model) → AI chat (LiteLLM/Open WebUI)
  → model-zoo ASSEMBLER (this repo)
  → data (traces/balance/version) → train (TRL/MLflow) → distill → eval → serve
```

## What's here

| Path | Purpose |
|------|---------|
| `assembler/assembler.py` | **the brain**: model spec → ordered pipeline plan (stages with dependency checks) |
| `assembler/tests/` | plan-assembly + dependency tests (4 green) |
| `n8n/` | zoo-builder n8n workflow exports (wiring n8n → LiteLLM → Langfuse) |
| `frontend-wasm/` | thin WASM chat (describe model → run pipeline) — D7 |

## Usage (CLI)

```bash
python assembler/assembler.py plan "build a 3b ternary coder distilled from deepseek traces"
python assembler/assembler.py plan --json "a small qwen chat model"
python assembler/assembler.py recipes
```

## The pipeline stages (schema)

`spec → research → recipe → data → version(dvc) → train(trl/mlflow) →
distill → eval(trace_eval/gates) → serve(llama.cpp/Ollama)` — each stage
declares inputs/outputs; the assembler warns on unproduced inputs.

## Integration points

- **n8n** runs the stages as workflows (HTTP → LiteLLM `:4000`, Langfuse-traced).
- **Recipe Engine** (`bitnet runner/tools/recipe_engine.py`) executes recipes.
- **TRL** (SFT/DPO/GRPO/KTO) does the post-training; **MLflow** tracks runs.
- **DVC** versions the data artifacts between stages.
- **trace-miner + Langfuse** provide the observability spine (every stage traced).

## 2026 backing
TRL v1 (post-train, HF), MLflow 3.x (AI platform, OTel/MCP), n8n (orchestrator),
DVC (reproducible pipelines + SkyPilot), TraceCompiler / TRACE metrics.

---
Master plan: `E:\vibe_coding\glm\master control\wiki\family\FORGE-MASTER.md`