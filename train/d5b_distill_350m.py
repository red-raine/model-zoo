"""d5b_distill_350m.py — D5B: distill LFM2.5-350M student on teacher-gold.

The CPU-feasible distill leg (350M ≈ 4x faster than the 1.2B attempt). Student
= LiquidAI/LFM2.5-350M (cached); teacher labels = nyx-thinkingcap-v2 outputs
(model-zoo-d5-teach-full.jsonl). SFT with LoRA, short steps, logged to MLflow.

Usage:
    python d5b_distill_350m.py --steps 4 --rows 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TM = Path(r"E:\vibe_coding\dev\bitnet runner\tools\tm")
sys.path.insert(0, str(TM))

GOLD = Path(r"B:\datasets\local\model-zoo-d5-sft-ready.jsonl")
BASE = "LiquidAI/LFM2.5-350M"
OUT = Path(r"B:\datasets\local\model-zoo-d5b-student")
MLFLOW_URI = "http://127.0.0.1:5000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rows", type=int, default=24)
    args = ap.parse_args()

    import mlflow
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("model-zoo-d5b-distill")

    print(f"[d5b] student={BASE} teacher_gold={GOLD} steps={args.steps} rows={args.rows}", flush=True)

    ds = load_dataset("json", data_files=str(GOLD), split="train")
    ds = ds.select(range(min(len(ds), args.rows)))
    print(f"[d5b] sliced to {len(ds)} rows (cols: {ds.column_names})", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True)

    cfg = SFTConfig(
        output_dir=str(OUT),
        per_device_train_batch_size=1,
        max_steps=args.steps,
        learning_rate=3e-4,
        max_length=768,
        logging_steps=1,
        save_steps=args.steps,
        report_to=[],
        fp16=False,
        use_cpu=True,
    )
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    with mlflow.start_run(run_name=f"distill-350m-{args.steps}s-{args.rows}r"):
        mlflow.log_param("student", BASE)
        mlflow.log_param("teacher", "nyx-thinkingcap-v2")
        mlflow.log_param("steps", args.steps)
        res = trainer.train()
        metrics = getattr(res, "metrics", {}) or {}
        mlflow.log_metric("train_loss", float(metrics.get("train_loss", -1)))
        print(f"[d5b] TRAINED loss={metrics.get('train_loss', '?')}", flush=True)
        trainer.save_model(str(OUT / "adapter"))
        mlflow.log_artifact(str(OUT), artifact_path="adapter")
        print(f"[d5b] adapter -> {OUT / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())