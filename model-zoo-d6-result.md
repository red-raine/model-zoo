# Model Zoo D6 — serve + eval the distilled student (2026-10-20)

## What ran
Full distill→serve→eval loop closed:

1. **Teacher-gold** (D5): 124 prompt→response pairs distilled from
   `nyx-thinkingcap-v2` (median 1.7K chars), DVC-versioned.
2. **SFT** (D5B): `LiquidAI/LFM2.5-350M` + LoRA (r8/α16, q_proj/v_proj) on 16
   teacher rows, 3 steps. **train_loss 1.899**. Adapter saved (691KB).
3. **Merge** → full model saved (`model-zoo-d5b-merged`).
4. **GGUF convert** (f16, 709MB) → dropped into llama.cpp `models-ext`.
5. **Serve**: llama.cpp router loads it as `model-zoo-d5b-student-f16`
   (status: loaded, context 128K).
6. **D6 eval**: 6 TRACE probes against the served endpoint.

## D6 result (honest)

| Probe | TRACE |
|---|---|
| why sky blue | 0.067 |
| 3 causes French Revolution | 0.118 |
| Bayesian vs frequentist | 0.033 |
| water freezes at 0C | **0.384** |
| pigeonhole principle | 0.100 |
| transformer attention | (timed out — CPU ~50s/probe) |

**mean ≈ 0.14** (5 of 6 scored).

## Reading
The 350M student distilled for **3 steps on 16 rows** is genuinely weak — it
produces ~280-340 char answers with limited evidential structure. That's the
correct, honest outcome for a proof-of-pipeline run, not a quality run.

**What IS proven end-to-end:**
- spec → assembler → teacher-gold → TRL SFT → PEFT adapter → GGUF → served
  endpoint → TRACE eval, all at $0 and fully automated.
- The eval gate works (model-name bug fixed in `d6_eval_served.py`).
- CPU inference on this box = ~50s/probe for a 350M f16; use Q4_K_M or nyx GPU
  for real evals.

## What a real quality run needs
- **More data**: 124 rows → 2-5K rows (teacher-gold at scale, or the v6 corpus).
- **More steps**: 3 → 300-1000 (GPU, i.e. nyx).
- **Bigger student**: 350M → 1.2B (LFM2.5-1.2B-Instruct cached) with the same
  pipeline.
- **Q4_K_M quant** instead of f16 (4x smaller, ~2-3x faster CPU inference).

## Bugs fixed this round
- `d6_eval_served.py` sent `"model": ""` → fixed to pass `args.model`
  (the eval was returning HTTP 400 for every probe).
- Added `--max-tokens` / `--timeout` flags for slow CPU inference.

## Files
- `tools/tm/d6_eval_served.py` (fixed)
- `B:\datasets\local\model-zoo-d5b-student\adapter\` (LoRA)
- `B:\datasets\local\model-zoo-d5b-merged\` (merged HF)
- `B:\datasets\local\model-zoo-d5b-student-f16.gguf` (709MB)
- `B:\ai-stack\models\llama\models.ini` (+ our preset entry)