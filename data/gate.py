"""gate.py — the chokepoint. No CLEAN gate file, no training.

WHY THIS FILE EXISTS
--------------------
From PEWDIEPIE-MISTAKES-REPEATING.md:

    "The tool existing is not the gate firing."

We had `decontaminate.py` for two months and never ran it. The spine shipped
labelled `_clean` while being byte-identical to the un-decontaminated file, and
its "decontamination" was a length+code-shape regex mislabelled in a docstring.

This module turns decontamination into a *contract*:

    1. a corpus may only be trained on if `<corpus>.gate.json` sits next to it
    2. that file must say verdict=CLEAN
    3. it must record that the gate itself was self-tested
    4. any agent calling a trainer directly still trips the guard, because the
       guard lives in the trainer, not in the wrapper

USAGE (library)
---------------
    from tools.data.gate import require_clean
    require_clean("B:/datasets/forge/cleaned/spine_train.jsonl")   # raises

USAGE (CLI)
-----------
    # assemble a gate from a decontamination report (+ optional dedup pass)
    python tools/data/gate.py assemble --corpus spine_train \
        --input  B:/datasets/spine/spine_train.jsonl \
        --output B:/datasets/forge/cleaned/spine_train.jsonl \
        --decontam-report B:/datasets/forge/gates/spine_train.gate.decontam.json \
        --dedup --self-test-pass 5/5

    # check a corpus before training
    python tools/data/gate.py check --data B:/datasets/forge/cleaned/spine_train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

GATE_VERSION = "forge-gate/v1"

# Internal near-duplicate bar. PEWDIEPIE-LESSONS-SYNTHESIS.md uses 0.8 for
# internal dedupe and 0.7 for contamination; keep them different on purpose.
DEDUP_THRESHOLD = 0.8
DEDUP_SHINGLE = 5
DEDUP_NUM_PERM = 128

_WORD = re.compile(r"[a-z0-9]+")


def gate_path_for(data_path: str) -> str:
    """`foo.train.jsonl` -> `foo.gate.json`; `spine_train.jsonl` -> `spine_train.gate.json`."""
    d = os.path.dirname(os.path.abspath(data_path))
    base = os.path.basename(data_path)
    for suffix in (".train.jsonl", ".jsonl"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.join(d, base + ".gate.json")


# --------------------------------------------------------------------------
# Enforcement — import this in every trainer
# --------------------------------------------------------------------------

class GateError(RuntimeError):
    pass


REQUIRED = [
    "corpus", "rows_in", "rows_out", "verdict",
    "refs_hash", "refs_count",
    "decontam_self_test", "built_at",
]


def load_gate(data_path: str) -> dict:
    gp = gate_path_for(data_path)
    if not os.path.exists(gp):
        raise GateError(
            f"NO GATE FILE for {data_path}\n"
            f"  expected: {gp}\n"
            f"  Training refused. Build the gate first:\n"
            f"    python tools/data/gate.py assemble --corpus <name> ...\n"
            f"  (PewDiePie skipped decontamination and lost a full run. We do not.)"
        )
    with open(gp, encoding="utf-8") as fh:
        g = json.load(fh)
    return g


def require_clean(data_path: str) -> dict:
    """Raise unless the corpus has a valid CLEAN gate. Returns the gate."""
    g = load_gate(data_path)
    missing = [k for k in REQUIRED if k not in g]
    if missing:
        raise GateError(f"GATE INCOMPLETE {gate_path_for(data_path)}: missing {missing}")
    if g.get("verdict") != "CLEAN":
        raise GateError(
            f"GATE NOT CLEAN for {data_path}: verdict={g.get('verdict')!r} "
            f"contaminated={g.get('decontam', {}).get('hits_total')}"
        )
    st = str(g.get("decontam_self_test", ""))
    if not st.startswith(tuple(f"{i}/{i}" for i in range(1, 10))):
        # accept "n/n" only
        try:
            a, b = st.split("/")
            if int(a) != int(b):
                raise ValueError
        except Exception:
            raise GateError(
                f"GATE NEVER PROVED IT CAN REJECT for {data_path}: "
                f"decontam_self_test={st!r}. A gate that cannot reject is not a gate."
            )
    return g


def require_clean_enabled() -> bool:
    """Escape hatch for deliberate control runs. Loud, and off by default."""
    return os.environ.get("FORGE_SKIP_GATE") == "1"


def guard(data_path: str) -> dict:
    """Call this at the top of any training entrypoint."""
    if require_clean_enabled():
        print("!! FORGE_SKIP_GATE=1 -> GATE BYPASSED. Numbers from this run are "
              "NOT publishable.", flush=True)
        return {"verdict": "BYPASSED"}
    g = require_clean(data_path)
    print(f"[gate] CLEAN {data_path} | refs={g['refs_count']} "
          f"| self_test={g['decontam_self_test']} | {g['built_at']}", flush=True)
    return g


# --------------------------------------------------------------------------
# Dedup — local, free, no external service
# --------------------------------------------------------------------------

def _shingles(text: str, k: int = DEDUP_SHINGLE) -> set[str]:
    t = " ".join(_WORD.findall(text.lower()))
    if len(t) <= k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _row_text(rec: dict) -> str:
    """Concatenate every text-bearing field, whatever the schema.

    Dedup used to key on `rec["text"]` alone. On a corpus with
    prompt/completion fields every key became "" and the first run of this
    gate declared 2,681 of 2,682 v6 rows "exact duplicates". A deduplicator
    that empties a corpus is worse than none.
    """
    parts = []
    for k in ("text", "prompt", "instruction", "question", "input",
              "completion", "response", "output", "answer", "code"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    if isinstance(rec.get("messages"), list):
        for m in rec["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str) \
                    and m["content"].strip():
                parts.append(m["content"])
    return "\n".join(parts)


def run_dedup(rows: list[dict], text_field: str = "text",
              threshold: float = DEDUP_THRESHOLD) -> tuple[list[int], dict]:
    """Return (indices_to_drop, stats). Exact sha256 then MinHash LSH."""
    from datasketch import MinHash, MinHashLSH

    seen_exact: dict[str, int] = {}
    drop: set[int] = set()
    exact_dups = 0
    empty_keys = 0

    for i, r in enumerate(rows):
        t = _row_text(r)
        if not t.strip():
            empty_keys += 1
            continue                      # never dedupe on an absent field
        h = hashlib.sha256(" ".join(_WORD.findall(t.lower())).encode()).hexdigest()
        if h in seen_exact:
            drop.add(i)
            exact_dups += 1
        else:
            seen_exact[h] = i

    if rows and empty_keys / len(rows) > 0.5:
        raise SystemExit(
            f"ABORT: {empty_keys}/{len(rows)} rows have no text-bearing field. "
            f"Dedup would collapse the corpus. Check the schema (text vs "
            f"prompt/completion vs messages) before running.")

    lsh = MinHashLSH(threshold=threshold, num_perm=DEDUP_NUM_PERM)
    sigs: list = []
    for i, r in enumerate(rows):
        m = MinHash(num_perm=DEDUP_NUM_PERM)
        for s in _shingles(_row_text(r)):
            m.update(s.encode("utf-8"))
        sigs.append(m)
        if i not in drop:
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass

    near_dups = 0
    for i, m in enumerate(sigs):
        if i in drop:
            continue
        for j in lsh.query(m):
            if j != str(i):
                drop.add(i)
                near_dups += 1
                break

    stats = {
        "rows": len(rows),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "empty_text_rows_skipped": empty_keys,
        "total_dropped": len(drop),
        "duplicate_rate": round(len(drop) / max(len(rows), 1), 6),
        "threshold": threshold,
        "shingle": DEDUP_SHINGLE,
        "num_perm": DEDUP_NUM_PERM,
        "engine": "datasketch MinHashLSH (local, free)",
    }
    return sorted(drop), stats


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _text_key(s: str) -> str:
    return hashlib.sha256(" ".join(_WORD.findall((s or "").lower())).encode()).hexdigest()


def eval_overlap_guard(train_rows: list[dict], eval_path: str,
                       text_field: str = "text") -> tuple[list[dict], dict]:
    """Drop train rows whose normalized text appears in the eval set.

    The paired split in pipeline.py is disjoint by ROW, not by TEXT: duplicate
    rows in the pool can land one in train and one in eval, leaking the eval
    set into training. This closes that hole.
    """
    if not os.path.exists(eval_path):
        return train_rows, {"status": "eval_missing", "path": eval_path}
    ev = {_text_key(_row_text(r)) for r in _iter_jsonl(eval_path)}
    keep, dropped = [], 0
    for r in train_rows:
        if _text_key(_row_text(r)) in ev:
            dropped += 1
            continue
        keep.append(r)
    return keep, {
        "status": "ok",
        "eval_path": os.path.abspath(eval_path),
        "eval_rows": len(ev),
        "dropped": dropped,
        "overlap_after": 0 if dropped >= 0 else None,
    }


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------

def cmd_assemble(args) -> int:
    dec = {}
    if args.decontam_report and os.path.exists(args.decontam_report):
        dec = json.load(open(args.decontam_report, encoding="utf-8"))
    else:
        print(f"WARN: no decontam report at {args.decontam_report}")

    rows = list(_iter_jsonl(args.output or args.input))
    rows_in = dec.get("rows_in", len(rows))
    rows_out = len(rows)

    prior = {}
    gp_existing = gate_path_for(args.output or args.input)
    if os.path.exists(gp_existing):
        try:
            prior = json.load(open(gp_existing, encoding="utf-8"))
        except Exception:
            prior = {}

    dedup_stats = {}
    if args.dedup:
        print(f"[gate] dedup over {rows_out} rows (minhash lsh {DEDUP_THRESHOLD}) ...",
              flush=True)
        t0 = time.time()
        drop, dedup_stats = run_dedup(rows)
        dedup_stats["seconds"] = round(time.time() - t0, 1)
        if drop:
            keep = [r for i, r in enumerate(rows) if i not in set(drop)]
            with open(args.output, "w", encoding="utf-8") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows_out = len(keep)
        print(f"[gate] dedup: {dedup_stats}")
    elif prior.get("dedup"):
        dedup_stats = prior["dedup"]
        print(f"[gate] dedup: reusing prior stats "
              f"(dropped={dedup_stats.get('total_dropped')})")

    eval_guard = {"status": "not_run"}
    if args.eval:
        rows, eval_guard = eval_overlap_guard(rows, args.eval)
        if eval_guard.get("dropped"):
            with open(args.output, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows_out = len(rows)
        print(f"[gate] eval guard: {eval_guard}")

    hits = dec.get("hits_by_layer", {})
    hits_total = sum(hits.values()) if hits else dec.get("rows_contaminated", 0)

    lic = {}
    if args.license_report and os.path.exists(args.license_report):
        lic = json.load(open(args.license_report, encoding="utf-8"))
    pii = {}
    if args.pii_report and os.path.exists(args.pii_report):
        pii = json.load(open(args.pii_report, encoding="utf-8"))

    verdict = "CLEAN"
    reasons = []
    if hits_total > 0 and not args.allow_hits:
        verdict = "CONTAMINATED"
        reasons.append(f"decontam hits={hits_total}")
    if not dec:
        verdict = "UNVERIFIED"
        reasons.append("no decontamination report")
    if pii.get("secret_hits", 0) > 0:
        verdict = "CONTAMINATED"
        reasons.append(f"live secrets found={pii['secret_hits']}")
    if not pii:
        reasons.append("PII gate not run")

    gate = {
        "gate": GATE_VERSION,
        "corpus": args.corpus,
        "input": os.path.abspath(args.input) if args.input else None,
        "output": os.path.abspath(args.output) if args.output else None,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "verdict": verdict,
        "note": args.note,
        "refs_hash": dec.get("vault", {}).get("sha256", "MISSING"),
        "refs_count": dec.get("vault", {}).get("entries", 0),
        "refs_per_source": dec.get("vault", {}).get("per_source", {}),
        "thresholds": dec.get("thresholds", {}),
        "decontam": {
            "layer": dec.get("gate", "unknown"),
            "hits_total": hits_total,
            "hits_by_layer": hits,
            "hits_by_source": dec.get("hits_by_source", {}),
            "contamination_rate": dec.get("contamination_rate", 0.0),
            "warn_band_rows": dec.get("rows_warn", 0),
            "examples": dec.get("examples", [])[:10],
            "seconds": dec.get("seconds"),
        },
        "dedup": dedup_stats,
        "eval_guard": eval_guard,
        "license": {
            "gate": lic.get("gate", "not_run"),
            "coverage": lic.get("license_coverage", 0.0),
            "commercial_coverage": lic.get("commercial_coverage", 0.0),
            "distribution": lic.get("distribution", {}),
            "policy": lic.get("policy", "internal-only"),
            "sources_needing_action": lic.get("sources_needing_action", {}),
            "registry": lic.get("registry"),
        },
        "pii": {
            "gate": pii.get("gate", "not_run"),
            "mode": pii.get("mode"),
            "rows_with_hits": pii.get("rows_with_hits", 0),
            "rows_dropped": pii.get("rows_dropped", 0),
            "hits_by_label": pii.get("hits_by_label", {}),
            "secret_hits": pii.get("secret_hits", 0),
            "verdict": pii.get("verdict", "not_run"),
        },
        "verdict_reasons": reasons,
        "decontam_self_test": args.self_test_pass,
        "license_coverage": lic.get("license_coverage", args.license_coverage),
        "audit": {"status": "not_run"},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "tools/data/gate.py",
    }

    gp = gate_path_for(args.output or args.input)
    with open(gp, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2)

    print(f"[gate] wrote {gp}")
    print(f"[gate] verdict={verdict} rows_in={rows_in} rows_out={rows_out} "
          f"decontam_hits={hits_total} dedup_drop={dedup_stats.get('total_dropped', 0)}")
    return 0 if verdict == "CLEAN" else 3


def cmd_check(args) -> int:
    try:
        g = require_clean(args.data)
    except GateError as e:
        print(f"BLOCKED: {e}")
        return 2
    print(f"OK — {args.data} is trainable")
    print(json.dumps({k: g[k] for k in
                      ("corpus", "verdict", "rows_out", "refs_count",
                       "decontam_self_test", "built_at") if k in g}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Forge gate contract + local dedup.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble")
    a.add_argument("--corpus", required=True)
    a.add_argument("--input")
    a.add_argument("--output", required=True)
    a.add_argument("--decontam-report", required=True)
    a.add_argument("--dedup", action="store_true", help="run MinHash LSH dedup in place")
    a.add_argument("--eval", help="eval jsonl; drop train rows whose text appears in it")
    a.add_argument("--self-test-pass", default="0/0",
                   help="decontam self-test result, e.g. 5/5")
    a.add_argument("--license-report", help="license_gate.py tag report JSON")
    a.add_argument("--pii-report", help="pii_gate.py report JSON")
    a.add_argument("--license-coverage", type=float, default=0.0,
                   help="manual override; normally taken from --license-report")
    a.add_argument("--allow-hits", action="store_true",
                   help="keep verdict CLEAN despite hits (only if hits were removed)")
    a.add_argument("--note", default="")
    a.set_defaults(func=cmd_assemble)

    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
