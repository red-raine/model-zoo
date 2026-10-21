"""refs_vault.py — build a frozen, hashed decontamination reference vault.

WHY THIS FILE
-------------
`recipe.py` declares `decontam_refs: ["mmlu","arc","hellaswag","winogrande"]`
and has ZERO consumers — dead config. The only decontamination that exists in
the family targets HumanEval + MBPP, i.e. CODE ONLY. Applied to the prose spine
it reports "CLEAN, 0 flagged" without ever looking, because
`decontaminate.extract_assistant_code()` falls back to treating prose as Python
and layer 4 is skipped when there is no ChatML.

This module builds the missing thing: a single frozen vault covering both code
and prose benchmarks, with a manifest that the gate can pin by hash.

USAGE
-----
    python tools/data/refs_vault.py build                 # download + build
    python tools/data/refs_vault.py build --sets gsm8k,humaneval,mbpp
    python tools/data/refs_vault.py verify                # re-hash, check manifest
    python tools/data/refs_vault.py stats

OUTPUT
------
    B:\\datasets\\forge\\refs\\vault.jsonl          one JSON per reference item
    B:\\datasets\\forge\\refs\\vault-manifest.json  counts + sha256 + built_at

Vault row schema:
    {source, task_id, prompt_text, reference_text, code, kind}

`reference_text` is what must not appear in a training corpus.
`prompt_text` is the question — kept separately so prompt-leak can be detected
even when the answer never leaked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

VAULT_ROOT = r"B:\datasets\forge\refs"
DL_ROOT = r"B:\datasets\forge\downloads"
VAULT = os.path.join(VAULT_ROOT, "vault.jsonl")
MANIFEST = os.path.join(VAULT_ROOT, "vault-manifest.json")

ALL_SETS = [
    "gsm8k", "humaneval", "mbpp",
    "mmlu", "arc", "hellaswag", "winogrande", "truthfulqa",
]

# Existing local code reference set (HumanEval 164 + MBPP 500 + mbpp_sanitized 374)
LEGACY_REFS = r"B:\datasets\bitqwen\data\decontam\reference_sets.json"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(s) -> str:
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = "\n".join(str(x) for x in s)
    return " ".join(str(s).split())


# --------------------------------------------------------------------------
# Per-dataset extractors.  Each yields vault rows.
# --------------------------------------------------------------------------

def ex_gsm8k(name):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    for split in ds:
        for i, r in enumerate(ds[split]):
            q, a = _norm(r["question"]), _norm(r["answer"])
            yield dict(source=name, task_id=f"gsm8k/{split}/{i}",
                       prompt_text=q, reference_text=a, code=None, kind="math")


def ex_humaneval(name):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    for r in ds:
        yield dict(source=name, task_id=str(r["task_id"]),
                   prompt_text=_norm(r["prompt"]),
                   reference_text=_norm(r["canonical_solution"]),
                   code=_norm(r["canonical_solution"]), kind="code")


def ex_mbpp(name):
    from datasets import load_dataset
    for cfg in ("full", "sanitized"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", cfg, split="test")
        except Exception:
            continue
        for r in ds:
            yield dict(source=f"mbpp_{cfg}", task_id=f"mbpp_{cfg}/{r.get('task_id')}",
                       prompt_text=_norm(r.get("text") or r.get("prompt")),
                       reference_text=_norm(r.get("code")),
                       code=_norm(r.get("code")), kind="code")


def ex_mmlu(name):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    for i, r in enumerate(ds):
        ch = r.get("choices") or []
        q = _norm(r.get("question"))
        opts = "\n".join(_norm(c) for c in ch)
        yield dict(source=name, task_id=f"mmlu/test/{i}",
                   prompt_text=q + "\n" + opts,
                   reference_text=opts + "\n" + _norm(r.get("answer")),
                   code=None, kind="prose")


def ex_arc(name):
    from datasets import load_dataset
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        try:
            ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        except Exception:
            continue
        for i, r in enumerate(ds):
            ch = r.get("choices") or {}
            opts = "\n".join(_norm(t) for t in (ch.get("text") or []))
            yield dict(source=f"arc_{cfg}", task_id=f"{cfg}/test/{i}",
                       prompt_text=_norm(r.get("question")) + "\n" + opts,
                       reference_text=opts + "\n" + _norm(r.get("answerKey")),
                       code=None, kind="prose")


def ex_hellaswag(name):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    for i, r in enumerate(ds):
        endings = "\n".join(_norm(e) for e in (r.get("endings") or []))
        yield dict(source=name, task_id=f"hellaswag/val/{i}",
                   prompt_text=_norm(r.get("ctx")),
                   reference_text=endings, code=None, kind="prose")


def ex_winogrande(name):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for i, r in enumerate(ds):
        yield dict(source=name, task_id=f"winogrande/val/{i}",
                   prompt_text=_norm(r.get("sentence")),
                   reference_text=_norm(r.get("option1")) + "\n" + _norm(r.get("option2")),
                   code=None, kind="prose")


def ex_truthfulqa(name):
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    for i, r in enumerate(ds):
        yield dict(source=name, task_id=f"truthfulqa/val/{i}",
                   prompt_text=_norm(r.get("question")),
                   reference_text=_norm(r.get("best_answer")),
                   code=None, kind="prose")


EXTRACTORS = {
    "gsm8k": ex_gsm8k, "humaneval": ex_humaneval, "mbpp": ex_mbpp,
    "mmlu": ex_mmlu, "arc": ex_arc, "hellaswag": ex_hellaswag,
    "winogrande": ex_winogrande, "truthfulqa": ex_truthfulqa,
}


def load_legacy() -> list[dict]:
    """Fold in the existing 1,038-entry code reference set, if present."""
    if not os.path.exists(LEGACY_REFS):
        print(f"  [skip] legacy refs not found: {LEGACY_REFS}")
        return []
    rows = []
    with open(LEGACY_REFS, encoding="utf-8") as fh:
        for r in json.load(fh):
            rows.append(dict(
                source=str(r.get("source", "legacy")),
                task_id=str(r.get("task_id", "")),
                prompt_text=_norm(r.get("prompt")),
                reference_text=_norm(r.get("code")),
                code=_norm(r.get("code")),
                kind="code",
            ))
    print(f"  [ok] legacy refs: {len(rows)}")
    return rows


def cmd_build(args) -> int:
    os.makedirs(VAULT_ROOT, exist_ok=True)
    os.makedirs(DL_ROOT, exist_ok=True)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in sets if s not in EXTRACTORS]
    if unknown:
        print(f"unknown sets: {unknown}\navailable: {ALL_SETS}")
        return 1

    rows: list[dict] = []
    per_source: dict[str, int] = {}
    failures: list[str] = []

    print("[vault] legacy code refs (humaneval/mbpp, already local)")
    for r in load_legacy():
        rows.append(r)
        per_source[r["source"]] = per_source.get(r["source"], 0) + 1

    for name in sets:
        print(f"[vault] downloading {name} ...", flush=True)
        t0 = time.time()
        try:
            n = 0
            for r in EXTRACTORS[name](name):
                if not r["prompt_text"] and not r["reference_text"]:
                    continue
                rows.append(r)
                n += 1
            per_source[name] = per_source.get(name, 0) + n
            print(f"  [ok] {name}: {n} rows in {time.time()-t0:.1f}s", flush=True)
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failures.append(name)

    # dedupe by (source, task_id)
    seen, uniq = set(), []
    for r in rows:
        k = (r["source"], r["task_id"], r["prompt_text"][:80])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    with open(VAULT, "w", encoding="utf-8") as fh:
        for r in uniq:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "vault": VAULT,
        "sha256": _sha256_file(VAULT),
        "bytes": os.path.getsize(VAULT),
        "total_entries": len(uniq),
        "per_source": dict(sorted(per_source.items())),
        "failures": failures,
        "sets_requested": sets,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "tools/data/refs_vault.py",
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[vault] wrote {len(uniq)} entries -> {VAULT}")
    print(f"[vault] sha256 {manifest['sha256'][:32]}...")
    for k, v in manifest["per_source"].items():
        print(f"  {k:22s} {v}")
    if failures:
        print(f"[vault] FAILED sets: {failures}")
    return 0


def cmd_verify(args) -> int:
    if not os.path.exists(MANIFEST):
        print("no manifest — run `build` first")
        return 1
    m = json.load(open(MANIFEST, encoding="utf-8"))
    actual = _sha256_file(VAULT)
    n = sum(1 for _ in open(VAULT, encoding="utf-8"))
    ok_hash = actual == m["sha256"]
    ok_rows = n == m["total_entries"]
    print(f"vault    : {VAULT}")
    print(f"sha256   : {actual[:32]}... match={ok_hash}")
    print(f"rows     : {n} (manifest {m['total_entries']}) match={ok_rows}")
    print(f"built_at : {m['built_at']}")
    print(f"VERDICT  : {'OK' if ok_hash and ok_rows else 'MISMATCH'}")
    return 0 if (ok_hash and ok_rows) else 2


def cmd_stats(args) -> int:
    if not os.path.exists(VAULT):
        print("no vault — run `build` first")
        return 1
    m = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else {}
    print(json.dumps(m, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build/verify the decontamination reference vault.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sets", default="gsm8k,humaneval,mbpp,mmlu,arc,hellaswag,winogrande,truthfulqa")
    b.set_defaults(func=cmd_build)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
