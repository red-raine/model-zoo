r"""d5c_kd_distill.py — proper knowledge distillation (KD) per D7 research.

Built on the D7 E2E research finding (B:\datasets\local\d7-research-result.md):
  - Use HYBRID KD loss (soft teacher targets), not plain hard-label SFT.
  - Teacher and student must share a tokenizer (else logit matching breaks).
  - Data mix: mostly teacher/real outputs, small gold anchor.

Design:
  teacher = LiquidAI/LFM2.5-1.2B-Instruct   (cached, same tokenizer family)
  student = LiquidAI/LFM2.5-350M            (cached) + LoRA
  trainer = trl.GKDTrainer  (generalized KD: KL on softened logits + SFT term)
  data    = v6 corpus traces (real full-budget free-model answers)

Usage:
    python d5c_kd_distill.py --steps 10 --rows 48
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TM = Path(r"E:\vibe_coding\dev\bitnet runner\tools\tm")
sys.path.insert(0, str(TM))

V6_DIR = Path(r"E:\vibe_coding\dev\bitnet runner\truthmatrix\state\rung0_v2\rung0_v6")
TEACHER = "LiquidAI/LFM2.5-1.2B-Instruct"
STUDENT = "LiquidAI/LFM2.5-350M"
OUT = Path(r"B:\datasets\local\model-zoo-d5c-kd")
DATA = Path(r"B:\datasets\local\model-zoo-d5c-kd-data.jsonl")
MLFLOW_URI = "http://127.0.0.1:5000"


def build_dataset(rows_wanted: int) -> int:
    """Extract prompt/response pairs from the v6 corpus (real full-budget answers)."""
    import glob

    pairs = []
    for f in sorted(glob.glob(str(V6_DIR / "**" / "traces_*.jsonl"), recursive=True)):
        try:
            fh = open(f, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (d.get("prompt") or "").strip()
                resp = (d.get("response") or "").strip()
                if len(prompt) > 20 and len(resp) > 150:
                    pairs.append({"messages": [
                        {"role": "user", "content": prompt[:800]},
                        {"role": "assistant", "content": resp[:1600]},
                    ]})
                if len(pairs) >= rows_wanted * 2:
                    break
        if len(pairs) >= rows_wanted * 2:
            break

    # prefer longer (richer) answers, then cap
    pairs.sort(key=lambda p: -len(p["messages"][1]["content"]))
    pairs = pairs[:rows_wanted]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    import mlflow
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GKDConfig, GKDTrainer

    OUT.mkdir(parents=True, exist_ok=True)
    n = build_dataset(args.rows)
    print(f"[d5c] KD dataset: {n} real v6 pairs -> {DATA}", flush=True)

    ds = load_dataset("json", data_files=str(DATA), split="train")

    tok_s = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok_s.pad_token is None:
        tok_s.pad_token = tok_s.eos_token
    tok_t = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)

    print(f"[d5c] teacher={TEACHER} student={STUDENT} steps={args.steps}", flush=True)
    student = AutoModelForCausalLM.from_pretrained(STUDENT, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, trust_remote_code=True)

    cfg = GKDConfig(
        output_dir=str(OUT),
        per_device_train_batch_size=args.batch,
        max_steps=args.steps,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=1,
        save_steps=args.steps,
        report_to=[],
        fp16=False,
        bf16=False,
        use_cpu=True,
        lmbda=0.5,          # 50/50 JSD (KD) + SFT — the hybrid D7 recommends
        beta=0.5,           # temperature for softening teacher logits
        temperature=0.9,
        seq_kd=False,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("model-zoo-d5c-kd")
    with mlflow.start_run(run_name=f"kd-gkd-{args.steps}s-{n}r"):
        mlflow.log_param("teacher", TEACHER)
        mlflow.log_param("student", STUDENT)
        mlflow.log_param("steps", args.steps)
        mlflow.log_param("rows", n)
        mlflow.log_param("lmbda", 0.5)
        mlflow.log_param("lora_r", 16)
        trainer = GKDTrainer(model=student, teacher_model=teacher, args=cfg,
                             train_dataset=ds, peft_config=lora,
                             processing_class=tok_s)
        res = trainer.train()
        m = getattr(res, "metrics", {}) or {}
        loss = float(m.get("train_loss", -1))
        mlflow.log_metric("train_loss", loss)
        print(f"[d5c] KD TRAINED loss={loss}", flush=True)
        trainer.save_model(str(OUT / "adapter"))
        mlflow.log_artifact(str(OUT / "adapter"), artifact_path="adapter")
    print(f"[d5c] adapter -> {OUT / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())