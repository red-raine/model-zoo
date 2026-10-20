# sol = the training box (Intel Arc Pro B50 + Unsloth) — 240x faster

> **This closes the training blocker.** `sol` already had an `unsloth-intel:local`
> image; it just needed the GPU device passed through. Real LoRA training now
> runs on a **discrete Intel Arc Pro B50 (16 GB)**.

## sol hardware (verified)

| Item | Value |
|---|---|
| CPU | **Intel Core Ultra 9 285HX** |
| RAM | **125 GB** (111 GB available) |
| GPU 0 (integrated) | Arrow Lake-U Intel Graphics — `PCI 8086:7D67`, driver `xe` |
| **GPU 1 (discrete)** | **Intel Arc Pro B50 Graphics — `PCI 8086:E212`, 16,304 MB VRAM, 128 EUs** |
| Disk | 1.8 TB NVMe, 1.4 TB free |
| Device nodes | `/dev/dri/card0`, `card1`, `renderD128`, `renderD129` |

Already running (containers):
```
vllm-gptoss  :8020  vllm/vllm-openai-xpu:latest     ← GPU LLM serving (torch 2.14+xpu)
ovms         :8000  openvino/model_server:2026.4.1-gpu
mcp-server   :8002
dockhand     :3050
unsloth-intel:local  (11.9 GB image, present but unused)
```

## The unsloth stack (verified inside `unsloth-intel:local`)

```
torch 2.9.0+xpu        xpu_avail=True   device=Intel(R) Arc(TM) Pro B50 Graphics
unsloth 2026.10.5      trl 1.15.0       peft 0.21.2      transformers 5.17.0
```

Run it with the GPU device:
```bash
docker run --rm --device /dev/dri \
  -v /data:/data -v /out:/out -v /train.py:/train.py:ro \
  unsloth-intel:local python /train.py
```
(Without `--device /dev/dri` unsloth raises
`NotImplementedError: Unsloth cannot find any torch accelerator`.)

## Measured training speed (same job, same data)

| Host | Per step | Steps/min | Relative |
|---|---|---|---|
| **sol · Arc Pro B50 (XPU)** | **7.64 s** | **7.8** | **240×** |
| CE · CPU (i9, torch+cpu) | 1,838 s | 0.033 | 1× |

Proof run: `LFM2.5-350M` + LoRA r16, 200 v6 rows, seq 512, **10 steps in 77 s**,
`train_loss 2.091`, `mean_token_accuracy 0.55→0.61`, adapter 1.97 MB saved.

## What this unlocks
- **Real SFT runs**: 150 steps ≈ 20 min (vs ~3.8 days on CE).
- **Soft-logit KD** (`d5e_softlabel_kd.py`): the teacher top-k harvest already
  comes from a GPU; now the student trains on GPU too.
- **Bigger students**: 16 GB VRAM fits 1.2B with LoRA comfortably.

## Tooling added
- `tools/tm/train_arc.py` — Unsloth + TRL SFT for XPU (fixed for the TRL 1.15
  `dataset_text_field`-in-`SFTConfig` API change).
- `tools/setup_key_auth.py` — paramiko key installer (used to finish `sol`).

## Host role summary (final)
| Host | Role |
|---|---|
| **sol** | **trainer** — Arc Pro B50 16 GB + unsloth + vLLM XPU + 125 GB RAM |
| **nyx** | **inference** — Arc iGPU SYCL llama.cpp :8082 (24× vs CPU) + OpenVINO GPU/NPU |
| **CE** | orchestration — MLflow :5000, LiteLLM :4000, Langfuse :3002, n8n :5678, all repos |
| **lyre** | hub/backup — Gitea :3005 + media stack |
## FULL LOOP CLOSED — real training → serve → eval (2026-10-20)

### Training runs on sol (Arc Pro B50, XPU, unsloth)

| Run | Data | Steps | Time | Throughput | Final loss | Token acc |
|---|---|---|---|---|---|---|
| smoke | 200 rows | 10 | 77 s | 7.8 steps/min | 2.091 | 0.55 → 0.61 |
| quality | 200 rows | 150 | 108 s | 83 steps/min | **1.652** | 0.70 |
| **real** | **2,000 rows** | **600** | **261 s** | **137.8 steps/min** | **1.635** | — |

600 steps × 1,838 s/step on CE CPU would be **~306 hours**. On sol: **4.3 minutes**.

### How many unique prompts we actually have
The v6 corpus is 2,794 traces but only **87-124 unique prompts** (models × samples).
So 2,000 rows = the same prompts with different model responses. That still helps
(style/answer diversity) but the next quality lever is **prompt diversity**, not
row count.

### Serve + eval on nyx Arc (the trained model)
GGUF f16 (679 MB) → nyx Arc GPU on **:8083** (second container, warden's :8082 untouched).

| Model | TRACE mean | Answer length |
|---|---|---|
| CPU-trained (3 steps / 16 rows) | **0.140** | ~300 chars |
| **GPU-trained (600 steps / 2,000 rows)** | **0.253** | **~1,200 chars** |

**+81% TRACE and 4x longer answers** from real training. Still under the 0.55
gate — the model needs prompt diversity and more steps — but the trend is real
and measurable.

### Containers added on nyx (non-destructive)
`
llama-arc-modelzoo  :8083  our trained student (Arc, -ngl 99)
`
(the warden's llama-cpp-gpu :8082 with gpt-oss-20b is untouched)

### Artifacts on CE
- B:\datasets\local\model-zoo-arc-adapter\ — LoRA r16 from the 600-step run
- B:\datasets\local\model-zoo-arc-merged\ — merged HF model
- B:\datasets\local\model-zoo-arc-f16.gguf — 679 MB f16 GGUF
- B:\datasets\local\model-zoo-d6-arc-trained.json — eval (TRACE 0.253)
