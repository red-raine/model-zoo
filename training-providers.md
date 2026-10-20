# Training providers — free-first ladder (2026-10-20)

> Where the Model Zoo's train lane actually runs. Ordered by cost, best first.
> Rule: **free tiers first, paid only with explicit sign-off** (constitutional
> law). All of these run the same artifacts our pipeline emits (HF dataset +
> TRL/Unsloth script + adapter out).

## Tier 0 — free, local (owned hardware)

| Host | Accelerator | Use | Status |
|---|---|---|---|
| **nyx** | **Intel Arc GPU + NPU** | **primary trainer** once XPU torch + unsloth land | GPU live for inference (SYCL, 24x); training pending torch-xpu |
| CE | UHD 630 (no Arc) | smoke tests only | torch 2.11+cpu |
| lyre | — | hub/media box, not a trainer | — |
| sol | ? | auth blocked | needs creds |

## Tier 1 — free cloud GPU (sign-up, quota-limited)

| Provider | Free GPU | Limits | Fit |
|---|---|---|---|
| **Kaggle** | T4 x2 / P100 | ~30 GPU-h/week, 12h max/session, 20GB notebook disk | **best free option for a 350M-1.2B LoRA run.** Already have account. Emit a Kaggle bundle from `tools/kaggle_push.py`. |
| **Google Colab** | T4 (free) | session caps, GPU not guaranteed | quick experiments, notebook-driven |
| **Modal** | serverless GPU, free starter credits | credit-metered | good for **bursty** jobs (train N minutes, stop) — no 12h cap like Kaggle |
| **Paperspace (DigitalOcean)** | **Free CPU (C4)** + **Free GPU (M4000)** on the free plan | **5-hour session limit**; Pro/Growth adds **P4000, P5000, RTX4000, RTX5000, A4000** | see below — the upgrade is the interesting one |

### Paperspace detail (user-provided, worth the upgrade)
```
Free plan      : Free CPU (C4), Free GPU (M4000)
Pro / Growth   : Free P4000, Free P5000, Free RTX4000, Free RTX5000, Free A4000
Constraint     : 5-hour session limit
```
A **free M4000 (8GB)** already beats anything on CE/nyx CPU for LoRA training.
The **A4000 (16GB)** / **RTX5000 (16GB)** tier would comfortably train
350M-1.2B LoRA and even soft-logit KD. The 5-hour cap fits a LoRA run:
350M × 500 rows × ~1 epoch is well under an hour on an A4000.
→ **Recommendation: upgrade to Pro for the A4000/RTX5000 tier, train on it,
keep inference on nyx Arc.** Docs: `docs.digitalocean.com/products/paperspace/`

## Tier 2 — paid / credits (only with sign-off)

| Provider | What | Note |
|---|---|---|
| **Google Cloud** | GPUs by the hour (T4/L4/A100) | use credits if any; watch spend |
| **SiliconFlow** | model **inference** API (not training) | useful for cheap inference, not SFT |
| **OpenRouter** | model **inference** API (1,000/day w/ top-up) | inference only |
| Kaggle paid / Colab Pro | longer sessions, better GPUs | optional |

## The routing rule for the train lane

```
spec → assembler → data (DVC) → TRAIN:
   1. nyx Arc (free, local)        -- once XPU torch + unsloth
   2. Kaggle T4x2 (free weekly)    -- for bigger runs, 12h sessions
   3. Modal (free credits)         -- bursty, no session cap
   4. Paperspace A4000/RTX5000     -- if upgraded; best sustained free GPU
   5. Google Cloud                 -- credits/paid only
→ adapter → GGUF → serve on nyx Arc (:8082) → TRACE eval (d6) → gate
```

## What each provider needs from us (all already produced by the pipeline)

| Artifact | Where |
|---|---|
| SFT/KD dataset (jsonl, prompt/completion or messages) | `B:\datasets\local\model-zoo-d5*.jsonl` |
| Train script | `tools/tm/d5b_distill_350m.py` (SFT), `d5e_softlabel_kd.py` (soft-KD) |
| Kaggle bundle | `tools/kaggle_push.py` (scaffold + validate) |
| Eval | `tools/tm/d6_eval_served.py` + `d6_redteam_litellm.py` |
| Tracking | MLflow `:5000` (local); remote runs log metrics back via the same URI |

## Ranked recommendation (what I'd do next)

1. **Paperspace Pro** (A4000/RTX5000, free with subscription) → run the **real**
   350M LoRA on 2-5K rows, ~30-60 min, inside the 5h cap. Biggest quality jump
   per effort.
2. **Kaggle** as the zero-cost fallback for the same run (30h/week T4x2).
3. **nyx Arc XPU** for local iteration + soft-logit KD once torch-xpu + unsloth
   are installed (no quota, no upload).
4. Keep **inference** on nyx Arc (:8082) regardless — it's already 24x and free.