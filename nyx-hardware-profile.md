# nyx hardware profile — CORRECTED (2026-10-20)

> **Correction:** I earlier wrote "nyx has no GPU (CPU-only)". **That was wrong.**
> nyx is an **Intel Core Ultra 9 285H** with an **Intel Arc GPU** and a
> **Meteor Lake NPU**. Verified directly on the box.

## Verified hardware

| Component | Evidence |
|---|---|
| **CPU** | `Intel(R) Core(TM) Ultra 9 285H` (`/proc/cpuinfo`) |
| **GPU** | `00:02.0 VGA compatible controller: Intel Corporation Arrow Lake-P [Intel Graphics] (rev 03)` |
| **NPU** | `00:0b.0 Processing accelerators: Intel Corporation Meteor Lake NPU (rev 05)` |
| GPU device nodes | `/dev/dri/card0`, `/dev/dri/renderD128` |
| NPU device node | `/dev/accel/accel0` |
| Tools present | `intel_gpu_top`, `vulkaninfo`, `libze_loader` (Level Zero) |
| RAM | 58 GB (50 GB available) |

## Acceleration status (what's wired vs what isn't)

| Path | Status |
|---|---|
| **Level Zero / SYCL** | `libze_loader` present (12 vulkan/level0 libs) → **SYCL is viable** |
| **Vulkan** | ⚠️ falls back to `llvmpipe` (software rasterizer) — the Vulkan path does NOT hit the Arc GPU as configured |
| **NPU** | `/dev/accel/accel0` present; `onnxruntime 1.31.0` installed → **OpenVINO EP viable**, but OpenVINO itself is NOT installed |
| **llama.cpp** | all running `llama-server` instances are **CPU-only** (no `--n-gpu-layers` / `-ngl 0`) |
| PyTorch | **not installed** (`transformers 5.19.0` only) |

## So the real opportunity

nyx is not a CPU brick — it's an **unused Arc GPU + unused NPU**. Two concrete
wins:

1. **GPU inference** — build/run llama.cpp with `GGML_SYCL=1` (Level Zero is
   already present) or install IPEX-LLM. That turns the model-zoo eval from
   ~50 s/probe (CPU) into a few seconds, and makes the distill→serve→eval loop
   practical at scale.
2. **NPU inference** — `pip install openvino` + OpenVINO EP on the existing
   onnxruntime → low-power always-on inference (good for the trace-miner /
   classifier legs).

Training: no CUDA; PyTorch XPU (Intel GPU) is possible but needs
`torch` + IPEX + the Arc driver stack. CPU training on 58 GB RAM remains the
fallback (it's what CE does today).

## Plan items added

- **D6+** nyx GPU lane: SYCL llama.cpp build → serve the distilled student at
  GPU speed; re-run the D6 TRACE eval (expect 5-10x faster probes).
- **D6+** nyx NPU lane: OpenVINO EP for the classifier/judge legs.
- Update FORGE-MASTER's host table: CE = trainer, nyx = **accelerated
  inference** (GPU/NPU), not "CPU-only".

## Files
- Verified via `ssh pool-crysal-eyes@192.168.8.237` (2026-10-20).
- Prior wrong claim appeared in `knowledge/model-zoo-d6-result.md` context and
  the D7 run notes — this file supersedes it.
## Concrete GPU enablement recipe (2026 research)

**Official prebuilt Intel SYCL image exists** — no source build needed:
```
ghcr.io/ggml-org/llama.cpp:server-intel-b9144
```

Run it with the Arc GPU device nodes passed through and full offload:
```bash
docker run -d --name llama-arc \
  --device /dev/dri \
  --device /dev/accel \
  -v /models:/models:ro \
  -p 8082:8080 \
  ghcr.io/ggml-org/llama.cpp:server-intel-b9144 \
  --model /models/<model>.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 --jinja
```

Verify GPU is actually used: docker exec llama-arc sycl-ls (should list the Arc
device), then intel_gpu_top on the host during a request should show engine
busy (not 0%).

Notes from the 2026 sources:
- SYCL needs Intel MKL's SYCL BLAS → the **oneAPI Toolkit** (hence the official
  Intel image; plain ubuntu + dpclang does not work).
- oneAPI **2026.1** improves SYCL build perf ~3.1x (prompt 434 → 1331 t/s on Arc).
- Pin the image tag (avoid latest; known regressions exist per model).
- For iGPU (Arc 140T on Ultra 9 285H) use --device /dev/dri + -ngl 99;
  check ReBAR is enabled for best throughput.

**Who should do this:** nyx runs a shared stack (other users' llama-server
instances). Adding a GPU container is a stack-warden action — do NOT replace the
existing CPU servers; add llama-arc alongside on a new port (8082) and route
model-zoo evals to it.
