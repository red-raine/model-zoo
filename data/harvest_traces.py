"""harvest_traces.py — turn agent rollouts into (task, verified-report) SFT pairs.

WHY THIS FILE EXISTS
--------------------
v6-sft was 2,682 rows of which **1,824 were exact duplicates** (68%). The honest
unique count is 858. The binding constraint on TRACE is prompt diversity, not row
count, and generating more of the same eight prompts does not help.

`B:\\datasets\\distilling\\raw` holds ~6 GB of agent rollouts that were never
mined. Two are directly usable and verifier-tagged:

    greghavens_gpt56sol_partial                 11,410 rows /  918 tasks /  899 unique prompts
    greghavens_fable-5-coding-and-debugging     12,490 rows / 2,380 tasks / 3,062 unique prompts

Each row is a *snapshot at assistant_step k* of a longer trajectory. Group by
`source_trajectory_id`, keep the deepest step, and you get a clean pair:

    prompt     = the user's task (a real engineering request with constraints)
    completion = the final assistant message ("Fixed - all 32 regression checks pass...")

That is the same shape as v6 (task -> structured plan/report), but with 3,298
distinct tasks instead of 87, and every one of them verifier-tagged.

USAGE
-----
    python tools/data/harvest_traces.py --stats
    python tools/data/harvest_traces.py --out B:/datasets/forge/canon/harvested.jsonl

The output is NOT trainable yet - run it through the gate cascade:
    decontam_text.py -> license_gate.py -> pii_gate.py -> gate.py assemble
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

SOURCES = {
    "gpt56sol": r"B:\datasets\distilling\raw\greghavens_gpt56sol_partial\traces_chunk1.jsonl",
    "fable5":   r"B:\datasets\distilling\raw\greghavens_fable-5-coding-and-debugging-traces\traces.jsonl",
}

MIN_PROMPT = 120
MIN_COMPLETION = 200
MAX_TOTAL = 8000          # prompt+completion char budget (~2k tokens)

# `seed-authoring` rows are about authoring seed data, not solving a user task.
EXCLUDE_LANGS = {"seed-authoring", ""}


def norm_key(s: str) -> str:
    return hashlib.sha256(" ".join(s.lower().split()).encode()).hexdigest()


def truncate_at_sentence(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    cut = text[:budget]
    for sep in ("\n\n", ". ", "\n", " "):
        i = cut.rfind(sep)
        if i > budget * 0.6:
            return cut[: i + len(sep)].rstrip()
    return cut.rstrip()


def deepest_per_trajectory(path: str, limit: int | None = None) -> dict[str, dict]:
    """The row with the highest assistant_step for each trajectory."""
    best: dict[str, dict] = {}
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            if limit and n > limit:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = d.get("source_trajectory_id") or d.get("task")
            if not tid:
                continue
            step = d.get("assistant_step") or 0
            cur = best.get(tid)
            if cur is None or step > (cur.get("assistant_step") or 0):
                best[tid] = d
    return best


def extract_pair(d: dict) -> dict | None:
    msgs = [m for m in (d.get("messages") or []) if isinstance(m, dict)]
    users = [str(m.get("content") or "").strip() for m in msgs if m.get("role") == "user"]
    assts = [str(m.get("content") or "").strip() for m in msgs if m.get("role") == "assistant"]
    users = [u for u in users if u]
    assts = [a for a in assts if a]
    if not users or not assts:
        return None

    prompt = users[-1]
    completion = assts[-1]
    if len(prompt) < MIN_PROMPT or len(completion) < MIN_COMPLETION:
        return None
    if len(prompt) + len(completion) > MAX_TOTAL:
        completion = truncate_at_sentence(completion, MAX_TOTAL - len(prompt))
        if len(completion) < MIN_COMPLETION:
            return None
    return {"prompt": prompt, "completion": completion}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Harvest agent rollouts into SFT pairs.")
    ap.add_argument("--out")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--sources", default="gpt56sol,fable5")
    ap.add_argument("--exclude-seed-authoring", action="store_true", default=True)
    ap.add_argument("--keep-seed-authoring", dest="exclude_seed_authoring",
                    action="store_false")
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows: list[dict] = []
    per_source = collections.Counter()
    dropped = collections.Counter()
    traj_total = 0

    for name in names:
        path = SOURCES.get(name)
        if not path or not os.path.exists(path):
            print(f"[harvest] MISSING source {name}: {path}")
            continue
        t0 = time.time()
        best = deepest_per_trajectory(path)
        traj_total += len(best)
        print(f"[harvest] {name}: {len(best)} trajectories in {time.time()-t0:.0f}s",
              flush=True)

        for tid, d in best.items():
            lang = str(d.get("lang") or "")
            if args.exclude_seed_authoring and lang in EXCLUDE_LANGS:
                dropped["seed_authoring_or_no_lang"] += 1
                continue
            pair = extract_pair(d)
            if pair is None:
                dropped["too_short_or_long"] += 1
                continue
            pair.update({
                "source": f"greghavens/{name}",
                "task": d.get("task"),
                "lang": lang,
                "category": d.get("category"),
                "domain": d.get("domain"),
                "verifier": d.get("verifier"),
                "teacher_model": d.get("teacher_model") or d.get("observed_models"),
                "reasoning_effort": d.get("reasoning_effort"),
                "assistant_steps": d.get("assistant_step"),
                "trajectory_id": tid,
            })
            rows.append(pair)
            per_source[name] += 1

    # --- dedup: one example per unique prompt, keep the longest completion ---
    by_prompt: dict[str, dict] = {}
    exact_pairs = set()
    dup_prompts = 0
    dup_pairs = 0
    for r in rows:
        pk = norm_key(r["prompt"])
        ck = norm_key(r["prompt"] + "\x00" + r["completion"])
        if ck in exact_pairs:
            dup_pairs += 1
            continue
        exact_pairs.add(ck)
        prev = by_prompt.get(pk)
        if prev is None:
            by_prompt[pk] = r
        else:
            dup_prompts += 1
            if len(r["completion"]) > len(prev["completion"]):
                by_prompt[pk] = r

    final = sorted(by_prompt.values(), key=lambda r: -len(r["completion"]))

    print()
    print(f"[harvest] trajectories parsed      {traj_total}")
    print(f"[harvest] candidate pairs          {len(rows)}")
    print(f"[harvest]   exact duplicate pairs  {dup_pairs}")
    print(f"[harvest]   repeated prompts       {dup_prompts}")
    print(f"[harvest]   dropped (filters)      {dict(dropped)}")
    print(f"[harvest] UNIQUE PROMPTS           {len(final)}")
    if final:
        plens = sorted(len(r["prompt"]) for r in final)
        clens = sorted(len(r["completion"]) for r in final)
        print(f"[harvest] prompt chars   p50={plens[len(plens)//2]} "
              f"p90={plens[int(len(plens)*0.9)]} max={plens[-1]}")
        print(f"[harvest] completion chars p50={clens[len(clens)//2]} "
              f"p90={clens[int(len(clens)*0.9)]} max={clens[-1]}")
        print(f"[harvest] per source    {dict(per_source)}")
        print(f"[harvest] langs         {collections.Counter(r['lang'] for r in final).most_common(12)}")
        print(f"[harvest] categories    {collections.Counter(str(r['category']) for r in final).most_common(12)}")
        print(f"[harvest] verifiers     {collections.Counter(str(r['verifier']) for r in final).most_common()}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in final:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[harvest] wrote {len(final)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
