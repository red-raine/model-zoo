r"""d5e_softlabel_kd.py — soft-label KD from a GGUF teacher's top-k logprobs.

Novel path enabled by the nyx Arc GPU (2026-10-20):

  TRL's GKDTrainer needs a PyTorch teacher -> impossible with a GGUF teacher,
  and CPU JSD pegs a core forever. But llama.cpp's OpenAI endpoint exposes
  `logprobs` + `top_logprobs`, so we can do real soft-label KD:

    1. TEACHER (GPU, GGUF) generates a token sequence WITH top-k logprobs.
    2. STUDENT (CPU, 350M) is trained with a soft cross-entropy over those
       k candidates instead of one-hot -> transfers "dark knowledge".

  Only the teacher needs the GPU. The student is a forward pass on CPU.

Modes:
    python d5e_softlabel_kd.py gen      # harvest teacher soft labels (GPU)
    python d5e_softlabel_kd.py train    # train student on them (CPU)

Env / defaults:
    teacher endpoint = http://192.168.8.237:8082/v1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

TEACHER_EP = "http://192.168.8.237:8082/v1/chat/completions"
TEACHER_MODEL = "gpt-oss-20b-Q4_K_M.gguf"
TOPK = 8
OUT_DIR = Path(r"B:\datasets\local\model-zoo-d5e-soft")
SOFT_DATA = OUT_DIR / "teacher-soft.jsonl"
STUDENT = "LiquidAI/LFM2.5-350M"
V6 = Path(r"E:\vibe_coding\dev\bitnet runner\truthmatrix\state\rung0_v2\rung0_v6")


# ---------------------------------------------------------------- harvest ----
def _prompts(n: int) -> list[str]:
    seen, out = set(), []
    for f in sorted(glob.glob(str(V6 / "**" / "traces_*.jsonl"), recursive=True)):
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
                p = (d.get("prompt") or "").strip()
                if len(p) > 40 and p not in seen:
                    seen.add(p)
                    out.append(p[:600])
                if len(out) >= n:
                    return out
    return out


def harvest(n_prompts: int, max_tokens: int) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = _prompts(n_prompts)
    print(f"[d5e] harvesting {len(prompts)} prompts from GPU teacher "
          f"({TEACHER_MODEL})", flush=True)
    rows = 0
    with SOFT_DATA.open("w", encoding="utf-8") as fh:
        for i, p in enumerate(prompts):
            body = json.dumps({
                "model": TEACHER_MODEL,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "logprobs": True,
                "top_logprobs": TOPK,
            }).encode()
            req = urllib.request.Request(
                TEACHER_EP, data=body, method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read().decode())
            except Exception as exc:
                print(f"[d5e] {i} ERR {str(exc)[:60]}", flush=True)
                continue
            ch = d["choices"][0]
            lp = ch.get("logprobs") or {}
            content = lp.get("content") or []
            text = (ch["message"].get("content") or "")
            if not content or not text.strip():
                continue
            # keep per-position token + top-k {token: prob}
            seq = []
            for t in content:
                top = t.get("top_logprobs") or []
                probs = {}
                for alt in top:
                    tok = alt.get("token")
                    lg = alt.get("logprob")
                    if tok is None or lg is None:
                        continue
                    probs[tok] = round(math.exp(lg), 6)
                if not probs:
                    continue
                seq.append({"tok": t.get("token", ""), "top": probs})
            if len(seq) < 8:
                continue
            fh.write(json.dumps({"prompt": p, "text": text[:1600],
                                 "seq": seq[:512]}, ensure_ascii=False) + "\n")
            rows += 1
            if i and i % 5 == 0:
                print(f"[d5e] {i}/{len(prompts)} rows={rows}", flush=True)
    print(f"[d5e] harvested {rows} soft-label rows -> {SOFT_DATA}", flush=True)
    return rows


# ----------------------------------------------------------------- train ----
def train(steps: int, rows: int, lr: float, max_len: int) -> int:
    import mlflow
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = []
    with SOFT_DATA.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs = recs[:rows]
    if not recs:
        print("[d5e] no soft data — run `gen` first", flush=True)
        return 1
    print(f"[d5e] training student={STUDENT} on {len(recs)} soft rows, "
          f"{steps} steps", flush=True)

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(STUDENT, trust_remote_code=True)
    model.train()

    # only LoRA-ish: freeze all, train the last block + lm_head for CPU sanity
    for p in model.parameters():
        p.requires_grad = False

    def unfreeze_last():
        blk = None
        for name, mod in model.named_modules():
            if "layers" in name and name.count(".") == 2:
                blk = mod
        n = 0
        if blk is not None:
            for p in blk.parameters():
                p.requires_grad = True
                n += p.numel()
        for p in model.lm_head.parameters():
            p.requires_grad = True
            n += p.numel()
        return n

    n_train = unfreeze_last()
    print(f"[d5e] trainable params: {n_train:,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("model-zoo-d5e-softkd")
    with mlflow.start_run(run_name=f"softkd-{steps}s-{len(recs)}r"):
        mlflow.log_param("teacher", TEACHER_MODEL)
        mlflow.log_param("student", STUDENT)
        mlflow.log_param("topk", TOPK)
        mlflow.log_param("trainable_params", n_train)
        losses = []
        for step in range(steps):
            rec = recs[step % len(recs)]
            prompt_ids = tok(rec["prompt"], return_tensors="pt").input_ids
            # build target sequence from teacher tokens
            toks, targets = [], []
            for pos in rec["seq"][:max_len]:
                toks.append(pos["tok"])
                targets.append(pos["top"])
            if not toks:
                continue
            full_ids = tok(rec["prompt"] + "".join(toks), return_tensors="pt",
                           truncation=True, max_length=prompt_ids.shape[1] + max_len).input_ids
            if full_ids.shape[1] <= prompt_ids.shape[1] + 1:
                continue
            logits = model(full_ids).logits[0]                 # [seq, vocab]
            plen = prompt_ids.shape[1]
            tgt_logits = logits[plen - 1: plen - 1 + len(targets)]
            if tgt_logits.shape[0] != len(targets):
                continue
            # soft cross-entropy over the teacher's top-k
            loss = 0.0
            used = 0
            for j, probs in enumerate(targets):
                ids, ps = [], []
                for tstr, pr in probs.items():
                    tid = tok(tstr, add_special_tokens=False).input_ids
                    if not tid:
                        continue
                    ids.append(tid[0])
                    ps.append(pr)
                if len(ids) < 2:
                    continue
                p_t = torch.tensor(ps, dtype=torch.float32)
                p_t = p_t / p_t.sum()
                logp = F.log_softmax(tgt_logits[j][ids], dim=-1)
                loss = loss + (-(p_t * logp).sum())
                used += 1
            if used == 0:
                continue
            loss = loss / used
            opt.zero_grad()
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            losses.append(lv)
            mlflow.log_metric("soft_ce_loss", lv, step=step)
            print(f"[d5e] step {step+1}/{steps} soft_ce={lv:.4f} positions={used}",
                  flush=True)
        if losses:
            mlflow.log_metric("final_loss", losses[-1])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUT_DIR / "student-softkd"))
        tok.save_pretrained(str(OUT_DIR / "student-softkd"))
        mlflow.log_artifact(str(OUT_DIR / "student-softkd"), artifact_path="student")
    print(f"[d5e] saved -> {OUT_DIR / 'student-softkd'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--prompts", type=int, default=60)
    g.add_argument("--max-tokens", type=int, default=220)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=12)
    t.add_argument("--rows", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()
    if args.cmd == "gen":
        n = harvest(args.prompts, args.max_tokens)
        return 0 if n else 1
    return train(args.steps, args.rows, args.lr, args.max_len)


if __name__ == "__main__":
    raise SystemExit(main())