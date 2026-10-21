"""train_arc.py — real LoRA training on Intel Arc (sol) via Unsloth.

Runs INSIDE the `unsloth-intel:local` image on sol with `--device /dev/dri`.

    docker run --rm --device /dev/dri \
      -v /tmp/v6-sft.jsonl:/data/sft.jsonl:ro \
      -v /tmp/arc-out:/out \
      -v /tmp/train_arc.py:/train.py:ro \
      unsloth-intel:local python /train.py --steps 10

Why this file: proves the D4/D5 train lane on a real GPU (Arc Pro B50 16GB)
instead of a CPU crawler (1,551 s/step on crystal_eyes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Make `tools/data/gate.py` importable regardless of CWD.
_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/sft.jsonl")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    args = ap.parse_args()

    # --- FORGE GATE (chokepoint) ------------------------------------------
    # Refuse to train on a corpus that has no CLEAN gate file. This lives in
    # the trainer, not a wrapper, so bypassing the wrapper cannot bypass this.
    from data.gate import guard as forge_gate_guard, GateError
    try:
        forge_gate_guard(args.data)
    except GateError as exc:
        print(f"\n[arc] TRAINING REFUSED\n{exc}\n", flush=True)
        return 2

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"[arc] device={dev} model={args.model} steps={args.steps} "
          f"seed={args.seed} max_len={args.max_len} rows={args.rows}", flush=True)
    torch.manual_seed(args.seed)
    if dev == "xpu":
        print(f"[arc] GPU: {torch.xpu.get_device_name(0)}", flush=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_len,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.select(range(min(len(ds), args.rows)))

    def fmt(rec):
        return {"text": f"<|user|>\n{rec['prompt']}\n<|assistant|>\n{rec['completion']}"}

    ds = ds.map(fmt)

    targs = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=args.steps,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=max(args.steps, 1),
        report_to=[],
        fp16=False,
        bf16=(dev == "xpu"),
        use_cpu=(dev == "cpu"),
        optim="adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_len,
    )
    trainer = SFTTrainer(
        model=model, args=targs, train_dataset=ds,
        processing_class=tok,
    )
    t0 = time.time()
    res = trainer.train()
    dur = time.time() - t0
    metrics = getattr(res, "metrics", {}) or {}
    loss = metrics.get("train_loss")
    sps = args.steps / dur if dur else 0
    print(f"[arc] TRAINED loss={loss} in {dur:.1f}s "
          f"({sps*60:.1f} steps/min)", flush=True)
    out = os.path.join(args.out, "adapter")
    trainer.save_model(out)
    tok.save_pretrained(out)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump({"loss": loss, "seconds": round(dur, 1),
                   "steps_per_min": round(sps * 60, 2),
                   "device": dev, "model": args.model,
                   "rows": len(ds), "steps": args.steps,
                   "seed": args.seed, "max_len": args.max_len}, fh, indent=1)
    print(f"[arc] adapter -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())