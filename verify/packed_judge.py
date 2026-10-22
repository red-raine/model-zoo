"""packed_judge.py — judge many items per API call, because free lanes bill per CALL.

THE COST MODEL THAT MATTERS
---------------------------
On a paid lane you pay per token, so you minimise tokens: engineer the prompt down to
the shortest thing that still works.

On a free lane you pay per **call**. Tokens are nearly free — the saved quota table
(`PROVIDER-RATE-LIMITS.md`, authoritative 2026-10-06) says:

    Gemini 3.1 Flash Lite   4 RPM   4M TPM   150K RPD
    Gemini 3.5 Flash Lite   2 RPM   4M TPM   150K RPD

The binding constraint is **RPM**, not RPD. 4 RPM x 1,440 min = **5,760 calls/day**, so
the 150K RPD ceiling is unreachable — you run out of *minutes*, not requests.

Therefore one-item-per-call is the expensive design. Packing 40 items into one call
turns 22,442 calls (~4 days at 4 RPM) into ~561 calls (~2.3 hours). The input side is
essentially free at 4M TPM; only the OUTPUT is worth keeping compact.

So the optimisation is inverted per lane:
    free lane   -> maximise items per call, keep output terse
    paid lane   -> minimise total tokens, keep output terse

`packed_judge` implements the free-lane side. It also implements the fix-then-validate
pattern: one model rewrites a rejected row, a *different* model checks the rewrite, so a
single model's blind spot cannot pass as agreement.

USAGE
-----
    python tools/verify/packed_judge.py --self-test
    python tools/verify/packed_judge.py --input X.jsonl --field text --pack 40 --limit 200
    python tools/verify/packed_judge.py --input X.jsonl --fix-then-validate --limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request

LITELLM = "http://localhost:4000/v1"
KEY_FILE = r"C:\Users\myste\.config\opencode\keys\pool-model-zoo.key"

# The 150K-RPD / 4M-TPM lanes. Two different models so fix-then-validate is a real
# cross-check rather than a model agreeing with itself.
LANE_JUDGE = "gemini-3.5-flash-lite-free"
LANE_FIXER = "gemini-3.5-flash-lite-free"
LANE_VALIDATOR = "gemini-3.6-flash-free"

JUDGE_PROMPT = """Judge each numbered item: is it GOOD training data (clear, coherent, on-topic, not boilerplate)?

Reply with ONLY the verdicts on one line, like: 1:y 2:n 3:y
No explanations. No other text.

{items}"""

FIX_PROMPT = """Rewrite each numbered text so it is clear, complete and self-contained.
Keep the meaning. Do not add commentary.

Reply as JSON only: {{"1":"rewritten text","2":"rewritten text"}}

{items}"""

VALIDATE_PROMPT = """For each numbered pair, does the REWRITE preserve the meaning of the ORIGINAL and read better?
Answer only: 1:y 2:n 3:y

{pairs}"""



# ---------------------------------------------------------------------------
# Call governor — operator constraints (2026-10-21)
#
#   * ONE request at a time. Never concurrent.
#   * 2-3 s between calls.
#   * Never spend the full daily RPD. Cap at ~50K calls/day even though the
#     lane advertises 150K, because the lane is shared with everything else and
#     a burned quota is a burned day.
#
# At 2.5 s spacing that is ~24 calls/min = ~1,440/hour, so the 50K cap is itself
# unreachable in one day: time is the binding constraint, not the quota. Packing
# is what makes that acceptable — 40 items/call x 1,440 calls/h = 57,600 items/h.
# ---------------------------------------------------------------------------

GOVERNOR_STATE = r"B:\datasets\forge\cache\judge_calls.json"


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class CallGovernor:
    """Sequential, spaced, daily-capped. Persists across processes."""

    def __init__(self, delay: float = 2.5, max_calls_per_day: int = 50000,
                 state_path: str = GOVERNOR_STATE):
        self.delay = max(0.0, delay)
        self.max_calls = max_calls_per_day
        self.path = state_path
        self.calls_today = 0
        self._last = 0.0
        self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("date") == _today():
                self.calls_today = int(d.get("calls", 0))
        except Exception:
            pass

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"date": _today(), "calls": self.calls_today}, fh)
        except Exception:
            pass

    def can_call(self) -> bool:
        return self.calls_today < self.max_calls

    def wait(self):
        """Block until it is polite to make the next call."""
        if self.delay:
            gap = time.time() - self._last
            if gap < self.delay:
                time.sleep(self.delay - gap)

    def record(self):
        self.calls_today += 1
        self._last = time.time()
        self._save()

    def status(self) -> str:
        return (f"{self.calls_today}/{self.max_calls} calls today, "
                f"delay {self.delay}s, sequential")


def _key() -> str:
    try:
        with open(KEY_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")


_GOV: "CallGovernor | None" = None


def chat(prompt: str, model: str, max_tokens: int = 4096,
         endpoint: str = LITELLM, timeout: int = 300, temperature: float = 0.0) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,          # never truncate a free lane
        "temperature": temperature,
    }).encode()
    if _GOV is not None:
        if not _GOV.can_call():
            raise RuntimeError(f"daily call cap reached ({_GOV.status()})")
        _GOV.wait()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    if _GOV is not None:
        _GOV.record()
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "").strip()


def _parse_verdicts(raw: str, n: int) -> dict[int, bool]:
    """Parse '1:y 2:n' in any spacing/separator form. Missing keys are absent."""
    out: dict[int, bool] = {}
    for m in re.finditer(r"(\d{1,4})\s*[:=.)\-]\s*([ynYNsS])", raw or ""):
        idx = int(m.group(1))
        if 1 <= idx <= n:
            out[idx] = m.group(2).lower() in ("y", "s")
    return out


def _parse_json_map(raw: str) -> dict[int, str]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(str(k).strip())] = str(v).strip()
        except (TypeError, ValueError):
            continue
    return out


def judge_packed(rows: list[dict], field: str = "text", pack: int = 40,
                 model: str = LANE_JUDGE, max_chars: int = 900,
                 log=print) -> tuple[list[dict], dict]:
    """Return (rows_with_verdict, stats). Packs `pack` items per call.

    On parse failure the batch is re-split in half and retried, down to single items,
    so one malformed reply cannot lose 40 rows.
    """
    stats = {"calls": 0, "items": 0, "retries": 0, "failed": 0, "seconds": 0.0}
    t0 = time.time()
    out: list[dict] = []

    def run_batch(batch: list[dict], pack_size: int):
        if not batch:
            return
        lines = []
        for i, r in enumerate(batch, 1):
            txt = " ".join(str(r.get(field) or "").split())[:max_chars]
            lines.append(f"{i}. {txt}")
        prompt = JUDGE_PROMPT.format(items="\n".join(lines))
        try:
            raw = chat(prompt, model)
            stats["calls"] += 1
        except Exception as exc:
            stats["failed"] += len(batch)
            log(f"    call failed ({type(exc).__name__}); dropping {len(batch)}")
            return
        v = _parse_verdicts(raw, len(batch))
        if len(v) >= max(1, int(len(batch) * 0.8)):
            for i, r in enumerate(batch, 1):
                out.append({**r, "keep": v.get(i), "judge": model})
            stats["items"] += len(batch)
            return
        # parse failure: split and retry
        stats["retries"] += 1
        if len(batch) == 1:
            stats["failed"] += 1
            out.append({**batch[0], "keep": None, "judge": model})
            return
        half = max(1, len(batch) // 2)
        run_batch(batch[:half], half)
        run_batch(batch[half:], half)

    for i in range(0, len(rows), pack):
        run_batch(rows[i:i + pack], pack)
        log(f"    {min(i+pack, len(rows))}/{len(rows)} rows, "
            f"{stats['calls']} calls, {time.time()-t0:.0f}s")
    stats["seconds"] = round(time.time() - t0, 1)
    stats["items_per_call"] = round(stats["items"] / max(stats["calls"], 1), 1)
    return out, stats


def fix_then_validate(rows: list[dict], field: str = "text", pack: int = 10,
                      fixer: str = LANE_FIXER, validator: str = LANE_VALIDATOR,
                      log=print) -> tuple[list[dict], dict]:
    """One model rewrites, a DIFFERENT model checks the rewrite preserved meaning.

    Same-model self-check is worthless — it agrees with itself. Cross-model
    disagreement is the signal worth escalating.
    """
    stats = {"fix_calls": 0, "val_calls": 0, "accepted": 0, "rejected": 0,
             "unparsed": 0, "seconds": 0.0}
    t0 = time.time()
    out = []
    for i in range(0, len(rows), pack):
        batch = rows[i:i + pack]
        lines = "\n".join(f"{j}. {' '.join(str(r.get(field) or '').split())[:1200]}"
                          for j, r in enumerate(batch, 1))
        try:
            raw = chat(FIX_PROMPT.format(items=lines), fixer)
            stats["fix_calls"] += 1
        except Exception:
            stats["unparsed"] += len(batch)
            continue
        fixed = _parse_json_map(raw)
        pairs = []
        for j, r in enumerate(batch, 1):
            new = fixed.get(j)
            if not new:
                continue
            orig = " ".join(str(r.get(field) or "").split())[:800]
            pairs.append((j, r, orig, new))
        if not pairs:
            stats["unparsed"] += len(batch)
            continue
        plines = "\n".join(f"{k}. ORIGINAL: {o}\n   REWRITE: {nw[:800]}"
                           for k, (_j, _r, o, nw) in enumerate(pairs, 1))
        try:
            vraw = chat(VALIDATE_PROMPT.format(pairs=plines), validator)
            stats["val_calls"] += 1
            v = _parse_verdicts(vraw, len(pairs))
        except Exception:
            v = {}
        for k, (_j, r, orig, new) in enumerate(pairs, 1):
            ok = v.get(k)
            if ok is True:
                stats["accepted"] += 1
            elif ok is False:
                stats["rejected"] += 1
            else:
                stats["unparsed"] += 1
            out.append({**r, "original": orig, "rewritten": new,
                        "validated": ok, "fixer": fixer, "validator": validator})
        log(f"    {min(i+pack, len(rows))}/{len(rows)} rows, "
            f"{stats['fix_calls']}fix/{stats['val_calls']}val, {time.time()-t0:.0f}s")
    stats["seconds"] = round(time.time() - t0, 1)
    return out, stats


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def run_self_test() -> bool:
    print("packed_judge.py - SELF TEST")
    ok = True

    cases = [
        ("1:y 2:n 3:y", 3, {1: True, 2: False, 3: True}),
        ("1:Y\n2:N\n3:Y", 3, {1: True, 2: False, 3: True}),
        ("1. y, 2. n, 3. y", 3, {1: True, 2: False, 3: True}),
        ("1: yes 2: no", 2, {1: True, 2: False}),
        ("garbage", 3, {}),
        ("", 2, {}),
    ]
    for raw, n, want in cases:
        got = _parse_verdicts(raw, n)
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] parse {raw[:26]!r} -> {got}")

    jm = _parse_json_map('{"1":"a b","2":"c"}')
    good = jm == {1: "a b", 2: "c"}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] json map -> {jm}")

    # packing arithmetic: the whole point is calls, not tokens
    n_rows, pack = 22442, 40
    calls = math.ceil(n_rows / pack)
    good = calls == 562 and calls * 4 > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {n_rows} rows / pack {pack} = {calls} calls "
          f"({calls/4/60:.1f} h at 4 RPM) vs {n_rows} calls one-per-row "
          f"({n_rows/4/60:.0f} h)")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack many items per call on free lanes.")
    ap.add_argument("--input")
    ap.add_argument("--field", default="text")
    ap.add_argument("--pack", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=LANE_JUDGE)
    ap.add_argument("--fix-then-validate", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--delay", type=float, default=2.5,
                    help="seconds between calls (default 2.5; sequential only)")
    ap.add_argument("--max-calls-per-day", type=int, default=50000,
                    help="hard daily call cap (default 50000, NOT the 150K RPD)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if not args.input:
        ap.print_help()
        return 0

    global _GOV
    _GOV = CallGovernor(delay=args.delay, max_calls_per_day=args.max_calls_per_day)
    print(f"[packed] governor: {_GOV.status()}")

    rows = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    print(f"[packed] {len(rows)} rows, pack={args.pack}, model={args.model}")
    if args.fix_then_validate:
        out, stats = fix_then_validate(rows, args.field, min(args.pack, 10))
    else:
        out, stats = judge_packed(rows, args.field, args.pack, args.model)

    print(f"\n[packed] stats: {json.dumps(stats)}")
    if not args.fix_then_validate:
        kept = sum(1 for r in out if r.get("keep") is True)
        dropped = sum(1 for r in out if r.get("keep") is False)
        print(f"[packed] keep={kept} drop={dropped} "
              f"unknown={len(out)-kept-dropped}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[packed] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
