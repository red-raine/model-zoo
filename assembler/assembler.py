"""assembler — the model-zoo brain: model spec → recipe sequence → pipeline plan.

Reads the Recipe Catalog (bitsnet recipes/index.md + family recipes) and turns
a user "I want to build X" spec into an ordered, dependency-checked pipeline
plan that n8n (or Open WebUI / the CLI) can execute stage-by-stage.

2026 backing: n8n as orchestrator + LiteLLM OpenAI-compat + TRL/MLflow/DVC +
the family trace spine. This is the glue that makes "describe a model, get a
built+traced+distilled artifact" real.

Usage:
    python assembler.py plan "3b ternary coder distilled from deepseek traces"
    python assembler.py plan --json "describe a small qwen coder"
    python assembler.py recipes                  # list catalog
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECIPES_ROOT = Path(r"E:\vibe_coding\dev\bitnet runner\recipes")
FAMILY_ROOT = Path(r"E:\vibe_coding\dev\family")


@dataclass
class Stage:
    name: str
    tool: str            # tool/entrypoint that runs it
    inputs: list[str]    # artifact names it consumes
    outputs: list[str]   # artifact names it produces
    cost: str = "$0"
    notes: str = ""


@dataclass
class ZooPlan:
    spec: str
    stages: list[Stage] = field(default_factory=list)
    recipe_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec, "recipe_refs": self.recipe_refs,
                "warnings": self.warnings,
                "stages": [s.__dict__ for s in self.stages]}


# --- intent keywords → stage chains (the assembly ruleset) -----------------

MODEL_INTENTS = {
    "code|coder|programming": "code",
    "chat|assistant|instruct": "chat",
    "reason|think|math|r1": "reasoning",
    "classif|judge|gate|verifier": "judge",
    "ternary|1.5|1.58|bitnet|bit": "ternary",
    "distill|small|tiny|compress|kd": "distill",
}

STAGE_CATALOG: dict[str, list[Stage]] = {
    "default": [
        Stage("spec", "model-zoo/assembler.py", [], ["spec.md"], notes="normalize user intent -> spec"),
        Stage("research", "nanoresearch mcp / research-stage", ["spec.md"], ["research_notes.md"]),
        Stage("recipe", "recipe_engine.py run", ["spec.md", "research_notes.md"], ["recipe_plan.yaml"]),
        Stage("data", "dataset_pipeline + dlane + tm runner", ["recipe_plan.yaml"], ["traces.jsonl", "balanced.jsonl"]),
        Stage("version", "dvc add/commit/push", ["traces.jsonl", "balanced.jsonl"], ["dataset.dvc"]),
        Stage("train", "trl sft/dpo (or tools/tm posttrain)", ["balanced.jsonl", "dataset.dvc"], ["model.adapter", "run in mlflow"]),
        Stage("distill", "tiny ladder (LFM2/Qwen-0.5B) on nyx/CPU", ["model.adapter"], ["distilled.gguf"]),
        Stage("eval", "trace_eval + fingerprint_v0 + Gates 1-5", ["distilled.gguf"], ["eval_report.md"]),
        Stage("serve", "llama.cpp / Ollama; n8n exposes OpenAI endpoint", ["distilled.gguf", "eval_report.md"], ["served:port"]),
    ],
    "code": [
        Stage("spec", "assembler", [], ["spec_code.md"]),
        Stage("recipes", "recipe_engine.py run recipes/kaggle.yaml-style", ["spec_code.md"], ["recipe_code.yaml"]),
        Stage("data", "coding corpus (HumanEval/MBPP + mined code traces) -> balance", ["recipe_code.yaml"], ["code_traces.jsonl"]),
        Stage("train", "trl sft --model Qwen3-0.6B --dataset code_traces", ["code_traces.jsonl"], ["code_adapter"]),
        Stage("distill", "distill to tiny coder (functiongemma/lfm2 class)", ["code_adapter"], ["code_tiny.gguf"]),
        Stage("eval", "pass@1 on held-out + trace_miner score", ["code_tiny.gguf"], ["code_eval.md"]),
        Stage("serve", "Ollama/llama.cpp + n8n code-agent endpoint", ["code_tiny.gguf"], ["code_served"]),
    ],
}


def classify(spec: str) -> str:
    s = spec.lower()
    for pattern, intent in MODEL_INTENTS.items():
        if re.search(pattern, s):
            return intent
    return "default"


def plan(spec: str) -> ZooPlan:
    intent = classify(spec)
    zp = ZooPlan(spec=spec)
    stages = STAGE_CATALOG.get(intent, STAGE_CATALOG["default"])
    zp.stages = list(stages)
    # recipe refs we know exist
    if intent in ("code", "chat"):
        zp.recipe_refs.extend(["recipes/kaggle.yaml", "recipes/default.json"])
    elif intent == "distill":
        zp.recipe_refs.append("recipes/kaggle.yaml")
    # dependency sanity: every stage input must be produced earlier (or be a spec)
    produced: set[str] = {"spec.md", "spec_code.md"}
    for st in zp.stages:
        missing = [i for i in st.inputs if i not in produced]
        if missing:
            zp.warnings.append(f"{st.name}: inputs not produced upstream {missing}")
        produced.update(st.outputs)
    return zp


def list_recipes() -> dict[str, list[str]]:
    out = {}
    for f in sorted(RECIPES_ROOT.glob("*")):
        if f.is_file():
            out.setdefault(f.suffix or "file", []).append(f.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="assembler", description="model-zoo assembler")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan", help="assemble a pipeline plan from a model spec")
    p.add_argument("spec", nargs="+")
    p.add_argument("--json", action="store_true")
    p.add_argument("--intent", help="force intent (code/chat/reasoning/judge/ternary/distill/default)")
    r = sub.add_parser("recipes", help="list recipe catalog")
    args = ap.parse_args(argv)
    if args.cmd == "recipes":
        print(json.dumps(list_recipes(), indent=1))
        return 0
    if args.cmd == "plan":
        spec = " ".join(args.spec)
        zp = plan(spec)
        if args.intent:
            zp.intent = args.intent  # type: ignore[attr-defined]
        if args.json:
            print(json.dumps(zp.to_dict(), indent=1))
        else:
            print(f"[assembler] intent={classify(spec)} spec={spec!r}")
            for i, st in enumerate(zp.stages, 1):
                print(f"  {i}. {st.name:<10} [{st.tool}]  {st.cost}")
                if st.notes:
                    print(f"      {st.notes}")
            if zp.recipe_refs:
                print("  recipes:", ", ".join(zp.recipe_refs))
            for w in zp.warnings:
                print(f"  WARN: {w}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())