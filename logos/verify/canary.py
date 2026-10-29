"""logos/verify/canary.py — a frozen held-out split, so a change that improves tuning cannot hide a regression.

WHY THIS EXISTS, AND WHY IT IS THE HONEST VERSION OF WHAT IS POSSIBLE
--------------------------------------------------------------------
This project has no human-labelled held-out set. `regression.py` already ratchets the VERDICTS, but a verdict
ratchet cannot see the failure the literature reports most often: **validation-gain reversal** -- a change that
improves the metric you were watching while degrading the one you were not. With no held-out data that failure
is undetectable, and it has been observed with otherwise-identical protocols.

What IS possible without new labels:
  1. A HOLD-OUT BY INDEX. Every measurement in this project used dev.jsonl rows 1..300. Rows 440..500 have
     never been read by any experiment, so they are held out with respect to MY work -- not with respect to
     AVeriTeC's own history, and that distinction is stated rather than blurred.
  2. FROZEN METRICS. On those rows we freeze the quantities a change can silently damage, all of which are
     measurable WITHOUT labels:
         false_drop_rate     curated rows must not be dropped          (want 0.0)
         coverage            fraction of rows reaching a certificate   (want not to fall)
         escalate_rate       abstention rate                           (want stable)
         scope_ambiguous     fraction flagged for scope                (want stable)
         encoder_refusal     fraction dep_encoder refuses               (want stable)
  3. A HASH LOCK. The row ids and the frozen metric values live in one file with a digest. If the row set
     changes, the canary REFUSES rather than silently measuring a different population.

The canary is deliberately SMALL and never used for tuning. A canary you look at while developing is not a
canary; it is a second training set.

USAGE
    python tools/logos/verify/canary.py --build     # freeze the split and its metrics
    python tools/logos/verify/canary.py --check     # re-measure and compare, nonzero exit on regression
    python tools/logos/verify/canary.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

CANARY = os.environ.get("LOGOS_CANARY", os.path.join(_TOOLS, "logos", "verify", "canary.json"))
CORPUS = os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")

# rows 1..300 were used by every measurement in this project; the tail has never been read.
CANARY_START = int(os.environ.get("LOGOS_CANARY_START", "440"))
CANARY_END = int(os.environ.get("LOGOS_CANARY_END", "500"))

TOLERANCE = {
    # a canary metric may drift by this much before it counts as a regression
    "false_drop_rate": 0.0,      # ANY false drop on curated data is a failure
    "coverage": 0.05,
    "escalate_rate": 0.05,
    "scope_ambiguous": 0.10,
    "encoder_refusal": 0.10,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _digest(rows: list, metrics: dict) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(r["sha"].encode())
        h.update(b"\x00")
    h.update(json.dumps(metrics, sort_keys=True).encode())
    return h.hexdigest()[:32]


def load_rows(start: int = CANARY_START, end: int = CANARY_END, corpus: str = CORPUS) -> list:
    """The held-out slice, by INDEX. Read here and nowhere else."""
    if not os.path.exists(corpus):
        return []
    try:
        from logos import corpus_sizing
        all_rows = corpus_sizing.load_corpus(corpus, end)
    except Exception:                                        # noqa: BLE001
        return []
    out = []
    for i, r in enumerate(all_rows):
        if i < start:
            continue
        text = r.get("claim") or ""
        if not text.strip():
            continue
        out.append({"idx": i, "text": text, "sha": _sha(text), "gold": r.get("gold", ""),
                    "evidence": r.get("evidence", "")})
    return out


def measure(rows: list) -> dict:
    """The label-free metrics a change can silently damage."""
    from logos import pipeline
    from logos.logic import dep_encoder, scope_triage

    n = len(rows)
    if not n:
        return {"n": 0}
    disp_counts = {}
    drops = 0
    certificates = 0
    scope_amb = 0
    refused = 0
    for r in rows:
        ev = [r["evidence"]] if r.get("evidence") else None
        out = pipeline.evaluate_row(r["text"], kind="text", evidence=ev)
        d = out["disposition"]
        disp_counts[d] = disp_counts.get(d, 0) + 1
        if d == "DROP":
            drops += 1
        if out.get("refuted_by") or d in ("KEEP", "REPAIR"):
            certificates += 1
        try:
            if scope_triage.triage(r["text"]).get("status") == "SCOPE_AMBIGUOUS":
                scope_amb += 1
        except Exception:                                    # noqa: BLE001
            pass
        try:
            if dep_encoder.parse_dependency(r["text"])["status"] != "PARSED":
                refused += 1
        except Exception:                                    # noqa: BLE001
            pass
    return {
        "n": n,
        "false_drop_rate": round(drops / n, 4),
        "coverage": round(certificates / n, 4),
        "escalate_rate": round(disp_counts.get("ESCALATE", 0) / n, 4),
        "scope_ambiguous": round(scope_amb / n, 4),
        "encoder_refusal": round(refused / n, 4),
        "dispositions": disp_counts,
    }


def build(path: str = CANARY) -> dict:
    rows = load_rows()
    if not rows:
        return {"status": "NO_ROWS",
                "reason": f"no rows in the held-out range {CANARY_START}..{CANARY_END} of {CORPUS}"}
    metrics = measure(rows)
    payload = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "range": [CANARY_START, CANARY_END],
               "corpus": CORPUS, "rows": rows, "metrics": metrics,
               "digest": _digest(rows, metrics)}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
    except Exception as e:                                   # noqa: BLE001
        return {"status": "WRITE_FAILED", "reason": f"{type(e).__name__}: {e}"}
    return {"status": "BUILT", "n": len(rows), "metrics": metrics, "digest": payload["digest"],
            "path": path, "size_kb": round(os.path.getsize(path) / 1024, 1)}


def check(path: str = CANARY) -> dict:
    """Re-measure the frozen split. FAIL if the row set changed, or any metric regressed beyond tolerance."""
    if not os.path.exists(path):
        return {"status": "NO_CANARY", "passed": False,
                "reason": f"no canary at {path}; run --build"}
    try:
        with open(path, encoding="utf-8") as fh:
            frozen = json.load(fh)
    except Exception as e:                                   # noqa: BLE001
        return {"status": "UNREADABLE", "passed": False, "reason": f"{type(e).__name__}: {e}"}

    rows = frozen.get("rows") or []
    if not rows:
        return {"status": "EMPTY", "passed": False, "reason": "canary has no rows"}

    # ---- HASH LOCK: the canary must still be the same population
    live = {r["idx"]: _sha(r["text"]) for r in load_rows()}
    drift = [r["idx"] for r in rows if live.get(r["idx"]) != r["sha"]]
    if drift:
        return {"status": "ROW_SET_CHANGED", "passed": False,
                "reason": f"{len(drift)} canary row(s) no longer match their frozen hash "
                          f"(e.g. {drift[:5]}); the split is not the one that was frozen",
                "drift": drift[:20]}

    now = measure(rows)
    before = frozen.get("metrics") or {}
    regressions, improvements, stable = [], [], []
    for key, tol in TOLERANCE.items():
        if key not in before or key not in now:
            continue
        b, a = float(before[key]), float(now[key])
        delta = a - b
        # coverage going UP is good; every other metric going up is bad
        worse = (delta > tol) if key != "coverage" else (delta < -tol)
        if worse:
            regressions.append({"metric": key, "before": b, "after": a, "delta": round(delta, 4),
                                "tolerance": tol})
        elif key == "coverage" and delta > tol:
            improvements.append({"metric": key, "before": b, "after": a, "delta": round(delta, 4)})
        else:
            stable.append(key)
    passed = not regressions
    return {"status": "PASSED" if passed else "FAILED", "passed": passed, "n": len(rows),
            "before": before, "after": now, "regressions": regressions,
            "improvements": improvements, "stable": stable,
            "digest_ok": _digest(rows, before) == frozen.get("digest")}


def report(res: dict) -> None:
    print("=== HELD-OUT CANARY ===")
    st = res.get("status")
    print(f"  status {st}")
    if st in ("NO_CANARY", "NO_ROWS", "UNREADABLE", "EMPTY", "ROW_SET_CHANGED"):
        print(f"  {res.get('reason')}")
        return
    print(f"  rows {res['n']}  (held out by INDEX, never used for any tuning in this project)")
    if not res.get("digest_ok"):
        print("  NOTE: the stored digest does not match its own contents -- the canary file was edited")
    print(f"\n  {'metric':20s} {'frozen':>9s} {'now':>9s} {'delta':>9s}  verdict")
    for key in TOLERANCE:
        b = (res["before"] or {}).get(key)
        a = (res["after"] or {}).get(key)
        if b is None or a is None:
            continue
        d = round(a - b, 4)
        reg = any(r["metric"] == key for r in res["regressions"])
        imp = any(r["metric"] == key for r in res["improvements"])
        tag = "REGRESSION" if reg else ("improvement" if imp else "stable")
        print(f"  {key:20s} {b:9.4f} {a:9.4f} {d:+9.4f}  {tag}")
    print()
    if res["passed"]:
        print("  CANARY PASSED: no metric regressed beyond tolerance on rows never seen during development.")
    else:
        print("  CANARY FAILED:")
        for r in res["regressions"]:
            print(f"    - {r['metric']} {r['before']:.4f} -> {r['after']:.4f} "
                  f"(tolerance {r['tolerance']})")
        print("  This is the failure a verdict ratchet cannot see: the tuning set can improve while held-out")
        print("  degrades. Do not tune to make the canary pass.")


def run_self_test() -> bool:
    print("logos.verify.canary - SELF TEST")
    ok = True
    import tempfile

    # --- the hold-out range must NOT overlap the rows every experiment used
    good = CANARY_START >= 300
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the canary starts at index {CANARY_START}, "
          f"clear of the 1..300 used for tuning")

    rows = load_rows()
    good = len(rows) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] loaded {len(rows)} held-out rows")

    m = measure(rows) if rows else {}
    good = m.get("n") == len(rows) and 0.0 <= m.get("false_drop_rate", -1) <= 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] metrics computed: false_drop={m.get('false_drop_rate')} "
          f"coverage={m.get('coverage')} escalate={m.get('escalate_rate')}")

    # --- a frozen canary passes when nothing changed
    tmp = tempfile.mkdtemp(prefix="canary_")
    path = os.path.join(tmp, "c.json")
    b = build(path)
    good = b["status"] == "BUILT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] built a canary with {b.get('n')} rows ({b.get('size_kb')} KB)")

    c = check(path)
    good = c["status"] == "PASSED"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unchanged canary PASSES")

    # --- a tampered row set is REFUSED, not silently re-measured
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    d["rows"][0]["sha"] = "0" * 32
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    c2 = check(path)
    good = c2["status"] == "ROW_SET_CHANGED" and c2["passed"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a changed row set is REFUSED (ROW_SET_CHANGED)")

    # --- a metric regression is DETECTED. Freezing a HIGHER coverage than reality is a regression
    #     (coverage falling is the bad direction), which is the opposite of what an "improvement" freeze
    #     would test -- my first version froze a lower value and correctly passed.
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    d["rows"][0]["sha"] = rows[0]["sha"]
    d["metrics"]["coverage"] = round(m.get("coverage", 0) + 0.5, 4)
    d["digest"] = _digest(d["rows"], d["metrics"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    c3 = check(path)
    good = c3["passed"] is False and any(r["metric"] == "coverage" for r in c3["regressions"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an injected coverage REGRESSION is detected "
          f"({c3['status']}, {[r['metric'] for r in c3.get('regressions', [])]})")

    # --- and a missing canary does not pass
    c4 = check(os.path.join(tmp, "nope.json"))
    good = c4["status"] == "NO_CANARY" and c4["passed"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a missing canary reports NO_CANARY and does not pass")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frozen held-out canary with label-free metrics.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--path", default=CANARY)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.build:
        b = build(args.path)
        print(json.dumps({k: v for k, v in b.items() if k != "metrics"}, indent=2))
        if b.get("metrics"):
            print(json.dumps(b["metrics"], indent=2))
        return 0 if b["status"] == "BUILT" else 1
    if args.check:
        r = check(args.path)
        report(r)
        return 0 if r["passed"] else 1
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
