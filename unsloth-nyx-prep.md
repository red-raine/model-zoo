# Unsloth on nyx (Intel Arc) — preparation (2026-10-20)

> **Status: NOT installed anywhere yet.** Verified on the boxes directly.

## What each host has today

| Host | GPU | torch | unsloth | Verdict |
|---|---|---|---|---|
| **nyx** (192.168.8.237) | **Intel Arc (Arrow Lake-P) + Meteor Lake NPU** | `2.14.1+cpu` ← **CPU-only build** | ❌ | **the training target** — needs XPU torch |
| **lyre** (192.168.8.217) | none useful | ❌ no torch | ❌ | media/hub box (Gitea, *arr stack) — not a trainer |
| **CE** (crystal_eyes) | UHD 630 (no Arc) | `2.11.0+cpu` | ❌ | smoke tests only |
| **sol** (192.168.8.152) | ? | ? | ? | **auth blocked** (see below) |

## What nyx already runs (useful)

```
llama-cpp-gpu        :8082  ghcr.io/ggml-org/llama.cpp:server-intel-b9144  ← SYCL Arc ✓
chat-llamacpp-stock  :8081  llama.cpp (CPU)
sd-openvino-api      :8005  openvino/model_server:latest-gpu     ← OpenVINO on GPU/NPU ✓
comfyui              :8188  openvino-comfyui
qdrant               :6333
piper-tts            :10200
whisper-ov-stt       :8003
dockhand             :3000   dockhand-mcp-server :3012
llama-swap           (router)
```
So the **inference** side is already GPU/NPU-accelerated (OpenVINO for vision/speech,
SYCL for LLMs). Only **training** is unaccelerated.

## Unsloth + Intel: current state (2026 research)

- **Unsloth officially supports Intel GPUs** — docs page
  `unsloth.ai/docs/get-started/install/intel` (Feb 2026).
- **No IPEX needed anymore:** PyTorch **2.7.1+xpu and newer** ship *native* Intel
  GPU support (`torch.xpu`). IPEX is being deprecated for this path.
- Unsloth merged the Intel-device `torch.compile` fix (unsloth issue #6105, Jun 2026).

## Prep steps (to run on nyx when the warden is ready)

```bash
# 1) replace the CPU-only torch with the XPU build (native Intel GPU)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/xpu

# 2) verify the Arc is visible to torch
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'n/a')"

# 3) install unsloth (+ it pulls trl/peft/transformers)
pip install unsloth

# 4) smoke: 1-step LoRA on a tiny model
python - <<'PY'
import torch
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained("LiquidAI/LFM2.5-350M", max_seq_length=512, load_in_4bit=False)
m = FastLanguageModel.get_peft_model(m, r=16, target_modules=["q_proj","v_proj","k_proj","o_proj"])
print("device:", torch.xpu.current_device() if torch.xpu.is_available() else "cpu")
PY
```

**Prereqs on the host:** the Intel GPU runtime (Level Zero / oneAPI runtime libs —
already present, since `libze_loader` + `intel_gpu_top` are installed and SYCL
llama.cpp works). If the XPU wheel complains, install the Intel compute runtime
package for Ubuntu.

**Container option (cleaner, avoids touching the host python):**
```bash
docker run --rm -it --device /dev/dri --device /dev/accel \
  -v /models:/models -v $HOME/work:/work \
  intel/intel-extension-for-pytorch:latest-xpu bash
```
(then `pip install unsloth` inside). Keeps nyx's host env pristine.

## Why this matters
With XPU torch, the two things that are impossible today become cheap:
1. **Real SFT/KD runs** — CE does 1,551 s/step on CPU; Arc should be 10-50x faster.
2. **Soft-logit KD** — `d5e_softlabel_kd.py` currently harvests teacher top-k from
   the GPU but trains the student on CPU. With XPU it trains on the GPU too.

## Blocker: `sol` auth
`sol.lan` = `192.168.8.152` (DNS confirmed), SSH reachable, but password
`ch4ng3m3` is **rejected for every user tried** (`zcoder`, `zcode`,
`pool-crysal-eyes`, `red`, `myste`). nyx + lyre work with **key auth**
(`~/.ssh/id_ed25519`). Need either the correct sol credentials or a key copied
over. Helper ready: `tools/setup_key_auth.py --host 192.168.8.152 --user <u>
--password <pw>`.