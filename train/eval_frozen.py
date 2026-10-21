"""eval_frozen.py — score a served model against the FROZEN eval set, with variance.

WHY THIS FILE EXISTS
--------------------
`d6_eval_served.py` scores 6 hard-coded probes, one run, and reports a single
number. That is exactly the PewDiePie failure:

    "His Aider Polyglot scores ranged 17% to 40% across runs with the same
     model. He got a 'god run' of 39.1% and reported that."

A 6-probe single-run number cannot support a mean +/- std, and the 6 probes are
generic science questions while the model is trained on devops scenarios.

This harness reads the FROZEN golden eval split (48 unique probes: coding,
reasoning, eq, agentic - including honeypot traps), scores each with the TRACE
Toulmin evaluator, and reports mean, std, min, max and a bootstrap CI so two
arms can be compared honestly.

Also supports --arms: evaluate several endpoints in one pass and emit a
comparison table.

USAGE
-----
    python tools/tm/eval_frozen.py --endpoint http://192.168.8.237:8083/v1 \
        --model model.gguf --out B:/datasets/forge/eval/armB-s42.json

    python tools/tm/eval_frozen.py --compare B:/datasets/forge/eval/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

SCORING = Path(__file__).resolve().parent.parent / "scoring"
sys.path.insert(0, str(SCORING))
from trace_eval import trance_report  # noqa: E402

FROZEN = r"B:\datasets\forge\refs\eval-frozen-20261021.jsonl"


def load_probes(path: str) -> list[dict]:
    out, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            p = (d.get("prompt") or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append({"prompt": p, "prompt_id": d.get("prompt_id"),
                        "purpose": d.get("purpose"), "topic": d.get("topic")})
    return out


def call(endpoint: str, model: str, prompt: str, max_tokens: int,
         timeout: int, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"].get("content") or "").strip()


def bootstrap_ci(xs: list[float], n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(xs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(42)
    means = []
    for _ in range(n):
        means.append(statistics.mean(rng.choices(xs, k=len(xs))))
    means.sort()
    lo = means[int(n * alpha / 2)]
    hi = means[int(n * (1 - alpha / 2)) - 1]
    return (round(lo, 4), round(hi, 4))


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    lo, hi = bootstrap_ci(scores)
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "ci95": [lo, hi],
    }


def run_one(args) -> int:
    probes = load_probes(args.eval)
    print(f"[eval] endpoint={args.endpoint} model={args.model} "
          f"probes={len(probes)} temp={args.temperature}", flush=True)

    rows, scores = [], []
    by_purpose: dict[str, list[float]] = {}
    t0 = time.time()
    for i, p in enumerate(probes):
        try:
            r = call(args.endpoint, args.model, p["prompt"], args.max_tokens,
                     args.timeout, args.temperature)
            err = None
        except Exception as exc:
            r, err = "", f"{type(exc).__name__}: {str(exc)[:90]}"
        sc = None
        if r:
            try:
                sc = float(trance_report(r)["trace_score"])
                scores.append(sc)
                by_purpose.setdefault(str(p.get("purpose")), []).append(sc)
            except Exception as exc:
                err = f"score-error: {str(exc)[:60]}"
        rows.append({**p, "response": r, "chars": len(r), "trace": sc, "error": err})
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(probes)} "
                  f"(mean so far {statistics.mean(scores) if scores else 0:.3f})", flush=True)

    summary = summarize(scores)
    out = {
        "eval_set": args.eval,
        "endpoint": args.endpoint,
        "model": args.model,
        "arm": args.arm,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seconds": round(time.time() - t0, 1),
        "trace": summary,
        "trace_by_purpose": {k: summarize(v) for k, v in sorted(by_purpose.items())},
        "mean_chars": round(statistics.mean([r["chars"] for r in rows]), 1) if rows else 0,
        "errors": sum(1 for r in rows if r["error"]),
        "rows": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[eval] -> {args.out}")
    print(f"[eval] TRACE mean={summary.get('mean')} std={summary.get('std')} "
          f"n={summary.get('n')} ci95={summary.get('ci95')} "
          f"chars={out['mean_chars']} errors={out['errors']}")
    return 0


def compare(paths: list[str]) -> int:
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    runs = []
    for f in files:
        try:
            runs.append((f, json.loads(Path(f).read_text(encoding="utf-8"))))
        except Exception as e:
            print(f"skip {f}: {e}")

    print("=" * 88)
    print(f"{'arm':<18}{'seed':>6}{'n':>5}{'mean':>8}{'std':>8}{'ci95':>20}{'chars':>8}")
    print("-" * 88)
    grouped: dict[str, list] = {}
    for f, r in runs:
        t = r.get("trace", {})
        arm = r.get("arm") or Path(f).stem
        grouped.setdefault(arm, []).append(t.get("mean"))
        ci = t.get("ci95", ["-", "-"])
        print(f"{arm:<18}{str(r.get('seed')):>6}{t.get('n', 0):>5}"
              f"{t.get('mean', 0):>8.4f}{t.get('std', 0):>8.4f}"
              f"{f'[{ci[0]}, {ci[1]}]':>20}{r.get('mean_chars', 0):>8.1f}")
    print("-" * 88)

    print("\nARM SUMMARY (mean of seeds +/- std across seeds)")
    results = []
    for arm, means in grouped.items():
        ms = [m for m in means if isinstance(m, (int, float))]
        if not ms:
            continue
        m = statistics.mean(ms)
        s = statistics.stdev(ms) if len(ms) > 1 else 0.0
        results.append((arm, m, s, len(ms)))
        print(f"  {arm:<18} mean={m:.4f}  std={s:.4f}  seeds={len(ms)}")

    results.sort(key=lambda x: -x[1])
    if len(results) >= 2:
        best, second = results[0], results[1]
        gap = best[1] - second[1]
        pooled = max(best[2], second[2], 1e-9)
        print(f"\n  {best[0]} beats {second[0]} by {gap:+.4f} "
              f"({gap/pooled:.2f}x the larger seed-std)")
        print(f"  VERDICT: {'SEPARATED' if abs(gap) > 2*pooled else 'NOT SEPARATED ' + chr(45) * 2 + ' more seeds or steps needed'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frozen-set TRACE eval with variance.")
    ap.add_argument("--endpoint", default="http://192.168.8.237:8083/v1")
    ap.add_argument("--model", default="model.gguf")
    ap.add_argument("--eval", default=FROZEN)
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--seed")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare)
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
