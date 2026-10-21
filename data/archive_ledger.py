"""archive_ledger.py — move junk out of B:\\datasets with a verifiable ledger.

WHY THIS FILE EXISTS
--------------------
A prior audit claimed ~30 GB of `B:\\datasets` is junk: 3.1M identical empty
rows, a 1M-row shard with 63 distinct values, ~10 GB of superseded bitqwen
versions, 8.5 GB of duplicate classifier checkpoints, and two Python venvs.

Those numbers came from a report, not from a measurement. Moving 30 GB because
a document said so is how you delete something load-bearing. So this tool
VERIFIES each claim before it moves anything, and records a ledger that makes
every move reversible.

Two-phase on purpose:
    scan   -> measure and report. Changes nothing.
    apply  -> move, writing ARCHIVE-LEDGER.json (sha256 tree fingerprint + rows).

    restore <ledger-id>  -> put it back.

Operator decision (2026-10-20): archive, do not delete.

USAGE
-----
    python tools/data/archive_ledger.py scan
    python tools/data/archive_ledger.py apply --only empty_shards
    python tools/data/archive_ledger.py apply
    python tools/data/archive_ledger.py list
    python tools/data/archive_ledger.py restore <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time

BASE = r"B:\datasets"
ARCHIVE = r"B:\datasets\_archive"
LEDGER = os.path.join(ARCHIVE, "ARCHIVE-LEDGER.json")

# --------------------------------------------------------------------------
# Candidates. `verify` says HOW to prove the claim; never trust the label.
# --------------------------------------------------------------------------

CANDIDATES = [
    # --- manusagents: empty shards -------------------------------------
    dict(id="empty_shards/fable5_2m", group="empty_shards",
         path=r"bitqwen\raw_fable\manusagents-distill\data\coding\fable5_2m",
         claim="101 shards x 20,000 = 2.02M identical EMPTY rows",
         verify="empty_rows", owner="zcode"),
    dict(id="empty_shards/deepseek_v4_pro_agent", group="empty_shards",
         path=r"bitqwen\raw_fable\manusagents-distill\data\coding\deepseek_v4_pro_agent",
         claim="5 x 20,000 empty rows", verify="empty_rows", owner="zcode"),
    dict(id="empty_shards/fenrir_v2_1", group="empty_shards",
         path=r"bitqwen\raw_fable\manusagents-distill\data\cybersecurity\fenrir_v2_1",
         claim="6 x 99,870 empty rows  [REFUTED 2026-10-20: 99,870 DISTINCT real "
               "cybersecurity rows; do not archive]", verify="empty_rows", owner="zcode"),

    # --- manusagents: degenerate shards ---------------------------------
    dict(id="degenerate/royal_ghost_1m", group="degenerate",
         path=r"bitqwen\raw_fable\manusagents-distill\data\coding\royal_ghost_1m",
         claim="1M rows but only 63 distinct values", verify="degenerate", owner="zcode"),
    dict(id="degenerate/autonomy", group="degenerate",
         path=r"bitqwen\raw_fable\manusagents-distill\data\coding\autonomy",
         claim="10,000 rows / 19 distinct", verify="degenerate", owner="zcode"),

    # --- venvs (never training data) ------------------------------------
    dict(id="venv/n8n", group="venvs",
         path=r"distilling\.venv-n8n",
         claim="7,022 files / 377 MB python venv inside a data dir",
         verify="venv", owner="zcode"),
    dict(id="venv/py-n8n", group="venvs",
         path=r"distilling\.py-n8n",
         claim="1,463 files / 38 MB python venv inside a data dir",
         verify="venv", owner="zcode"),
]

# Superseded bitqwen versions: keep only the load-bearing ones.
KEEP_BITQWEN = {"bitqwen_v11_train_fixed.jsonl", "bitqwen_v11_eval.jsonl"}
BITQWEN_CLEANED = r"bitqwen\cleaned"


def tree_fingerprint(path: str) -> tuple[str, int, int]:
    """sha256 over (relpath, size) pairs. Cheap and sufficient for a ledger."""
    h = hashlib.sha256()
    files = nbytes = 0
    for root, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            fp = os.path.join(root, name)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, path).replace("\\", "/")
            h.update(f"{rel}\0{sz}\n".encode())
            files += 1
            nbytes += sz
    return h.hexdigest(), files, nbytes


def dir_size(path: str) -> tuple[int, int]:
    files = nbytes = 0
    for root, _d, names in os.walk(path):
        for name in names:
            try:
                nbytes += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return files, nbytes


def _row_is_empty(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if len(line) < 4:
        return True
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return False
    for k in ("text", "prompt", "instruction", "question", "content"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return False
    for k in ("messages", "conversations"):
        v = d.get(k)
        if isinstance(v, list) and v:
            for m in v:
                if isinstance(m, dict) and str(m.get("content", "")).strip():
                    return False
            return True
    # no recognised text field carrying content => unusable.
    # (previous version returned `not d`, so a dict full of empty strings
    #  scored as non-empty and 2M empty rows passed verification.)
    return True


def verify_empty_rows(path: str, cap_rows: int = 200_000) -> dict:
    total = empty = 0
    _seen: set[str] = set()
    for root, _d, names in os.walk(path):
        for name in sorted(names):
            if not name.endswith((".jsonl", ".json")):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        total += 1
                        _seen.add(hashlib.sha256(line.strip().encode()).hexdigest())
                        if _row_is_empty(line):
                            empty += 1
                        if total >= cap_rows:
                            break
            except OSError:
                continue
            if total >= cap_rows:
                break
    rate = empty / total if total else 0.0
    # An "empty shard" is empty AND degenerate. fenrir_v2_1 turned out to be
    # 99,870 distinct real cybersecurity rows that a prior audit mislabelled.
    distinct = len(_seen)
    dratio = distinct / total if total else 0.0
    return {"rows_sampled": total, "rows_empty": empty, "empty_rate": round(rate, 6),
            "distinct_rows": distinct, "distinct_ratio": round(dratio, 8),
            "claim_holds": total > 0 and rate > 0.98 and dratio < 0.05}


def verify_degenerate(path: str, cap_rows: int = 200_000) -> dict:
    seen: set[str] = set()
    total = 0
    for root, _d, names in os.walk(path):
        for name in sorted(names):
            if not name.endswith((".jsonl", ".json")):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        total += 1
                        seen.add(hashlib.sha256(line.strip().encode()).hexdigest())
                        if total >= cap_rows:
                            break
            except OSError:
                continue
            if total >= cap_rows:
                break
    distinct = len(seen)
    ratio = distinct / total if total else 0.0
    return {"rows_sampled": total, "distinct_rows": distinct,
            "distinct_ratio": round(ratio, 8),
            "claim_holds": total > 0 and ratio < 0.05}


def verify_venv(path: str) -> dict:
    markers = ("pyvenv.cfg", "site-packages", "Scripts", "bin")
    found = [m for m in markers
             if os.path.exists(os.path.join(path, m))
             or any(os.path.exists(os.path.join(r, m)) for r, _d, _f in os.walk(path))]
    return {"venv_markers": found, "claim_holds": bool(found)}


def verify_duplicate_checkpoints() -> list[dict]:
    """Find identical .safetensors under classifier/ and report duplicates."""
    root = os.path.join(BASE, "classifier")
    if not os.path.isdir(root):
        return []
    by_size: dict[int, list[str]] = {}
    for r, _d, names in os.walk(root):
        for n in names:
            if not n.endswith(".safetensors"):
                continue
            fp = os.path.join(r, n)
            by_size.setdefault(os.path.getsize(fp), []).append(fp)
    out = []
    for size, paths in by_size.items():
        if len(paths) > 1:
            hashes = {}
            for p in paths:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                hashes.setdefault(h.hexdigest(), []).append(p)
            for hh, same in hashes.items():
                if len(same) > 1:
                    out.append({"hash": hh[:16], "size": size,
                                "copies": len(same), "keep": same[0],
                                "redundant": same[1:]})
    return out


# A version-stack artifact, strictly. `bitqwen_v7_topup`, `merged_*`,
# `bitqwen_train_v6`, `bitqwen_v11_train` (superseded by _fixed).
# NOT matched: cosmopedia, book_rag_chunks, docker_*, codebert_* — those are
# unrelated datasets that merely live in the same folder.
SUPERSEDED_RE = re.compile(
    r"^(?:bitqwen_(?:v\d+_(?:train|input|rated|topup|topup_rated|topup_capped|"
    r"topup_capped_rated|decontaminated|audit_clean|bug_injection\w*)|"
    r"train_(?:v\d+|final|platinum\w*))|merged_\w+)\.jsonl$", re.I)


def verify_superseded_bitqwen() -> list[dict]:
    """Report superseded version-stack artifacts.

    auto=False: this is zcode's corpus. We report exact numbers and let the
    owner decide — a 35 GB name-matched sweep is how you delete something
    load-bearing (the first run of this scan tried to include cosmopedia).
    """
    d = os.path.join(BASE, BITQWEN_CLEANED)
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        fp = os.path.join(d, n)
        if not os.path.isfile(fp) or not n.endswith(".jsonl"):
            continue
        if n in KEEP_BITQWEN or "eval" in n.lower():
            continue
        if not SUPERSEDED_RE.match(n):
            continue
        out.append({"path": os.path.relpath(fp, BASE),
                    "bytes": os.path.getsize(fp),
                    "reason": "superseded bitqwen version-stack artifact; "
                              "only bitqwen_v11_train_fixed + v11_eval are load-bearing",
                    "auto": False, "owner": "zcode"})
    return out


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def load_ledger() -> dict:
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER, encoding="utf-8"))
    return {"created_at": None, "entries": []}


def save_ledger(led: dict) -> None:
    os.makedirs(ARCHIVE, exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as fh:
        json.dump(led, fh, indent=2)


def cmd_scan(args) -> int:
    print("=" * 74)
    print("ARCHIVE SCAN — nothing is moved in this mode")
    print("=" * 74)
    total_bytes = 0
    verdicts = []

    for c in CANDIDATES:
        fp = os.path.join(BASE, c["path"])
        if not os.path.exists(fp):
            print(f"\n[MISSING] {c['id']}\n  {fp}")
            verdicts.append({**c, "exists": False, "claim_holds": False})
            continue
        files, nbytes = dir_size(fp)
        sha, nfiles, nbytes2 = tree_fingerprint(fp)
        print(f"\n[{'SCAN'}] {c['id']}")
        print(f"  path   {c['path']}")
        print(f"  files  {files:,}   size {nbytes/1e9:.2f} GB")
        print(f"  claim  {c['claim']}")
        if c["verify"] == "empty_rows":
            v = verify_empty_rows(fp)
        elif c["verify"] == "degenerate":
            v = verify_degenerate(fp)
        elif c["verify"] == "venv":
            v = verify_venv(fp)
        else:
            v = {"claim_holds": False}
        print(f"  verify {json.dumps(v)}")
        print(f"  => claim_holds={v['claim_holds']}")
        if v["claim_holds"]:
            total_bytes += nbytes
        verdicts.append({**c, "exists": True, "fingerprint": sha, "files": files,
                         "bytes": nbytes, "verification": v,
                         "claim_holds": v["claim_holds"]})

    print("\n" + "=" * 74)
    print("DUPLICATE MODEL CHECKPOINTS (classifier/)")
    print("=" * 74)
    dups = verify_duplicate_checkpoints()
    dup_bytes = 0
    for d in dups:
        redundant_bytes = d["size"] * len(d["redundant"])
        dup_bytes += redundant_bytes
        print(f"  {d['copies']} identical copies of {d['size']/1e6:.1f} MB "
              f"(sha {d['hash']}) -> {redundant_bytes/1e9:.2f} GB reclaimable")
        print(f"    keep: {os.path.relpath(d['keep'], BASE)}")
    if not dups:
        print("  none found")
    total_bytes += dup_bytes

    print("\n" + "=" * 74)
    print("SUPERSEDED BITQWEN VERSIONS")
    print("=" * 74)
    sup = verify_superseded_bitqwen()
    sup_bytes = sum(s["bytes"] for s in sup)
    print(f"  {len(sup)} files, {sup_bytes/1e9:.2f} GB")
    for s in sup[:40]:
        print(f"    {s['bytes']/1e6:8.1f} MB  {os.path.basename(s['path'])}")
    if len(sup) > 40:
        print(f"    ... and {len(sup)-40} more")
    total_bytes += sup_bytes

    print("\n" + "=" * 74)
    print(f"TOTAL RECLAIMABLE (verified claims only): {total_bytes/1e9:.2f} GB")
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "candidates": verdicts, "duplicate_checkpoints": dups,
                       "superseded_bitqwen": sup,
                       "reclaimable_bytes": total_bytes}, fh, indent=2)
        print(f"scan written to {args.json}")
    return 0


def cmd_apply(args) -> int:
    led = load_ledger()
    if led.get("created_at") is None:
        led["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(ARCHIVE, exist_ok=True)

    moved_bytes = 0
    moved = 0

    for c in CANDIDATES:
        if args.only and c["group"] not in args.only:
            continue
        fp = os.path.join(BASE, c["path"])
        if not os.path.exists(fp):
            continue
        if c["verify"] == "empty_rows":
            v = verify_empty_rows(fp)
        elif c["verify"] == "degenerate":
            v = verify_degenerate(fp)
        elif c["verify"] == "venv":
            v = verify_venv(fp)
        else:
            v = {"claim_holds": False}
        if not v["claim_holds"]:
            print(f"[SKIP] {c['id']} — claim did not hold: {v}")
            continue
        sha, files, nbytes = tree_fingerprint(fp)
        dest = os.path.join(ARCHIVE, c["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            print(f"[SKIP] {c['id']} — destination exists")
            continue
        shutil.move(fp, dest)
        led["entries"].append({
            "id": c["id"], "group": c["group"], "moved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "from": os.path.join(BASE, c["path"]), "to": dest,
            "files": files, "bytes": nbytes, "fingerprint": sha,
            "claim": c["claim"], "verification": v, "owner": c["owner"],
        })
        moved += 1
        moved_bytes += nbytes
        print(f"[MOVED] {c['id']}  {nbytes/1e9:.2f} GB -> {dest}")
        save_ledger(led)

    if args.only and "superseded" in args.only:
        for s in verify_superseded_bitqwen():
            if not s.get("auto"):
                continue
            src = os.path.join(BASE, s["path"])
            if not os.path.exists(src):
                continue
            dest = os.path.join(ARCHIVE, s["path"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                continue
            sha, files, nbytes = tree_fingerprint(os.path.dirname(src)) if False else (
                hashlib.sha256(open(src, "rb").read(0) or b"").hexdigest(), 1, s["bytes"])
            h = hashlib.sha256()
            with open(src, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            shutil.move(src, dest)
            led["entries"].append({
                "id": f"superseded/{os.path.basename(src)}",
                "group": "superseded",
                "moved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "from": src, "to": dest, "files": 1, "bytes": s["bytes"],
                "fingerprint": h.hexdigest(), "claim": s["reason"],
                "verification": {"claim_holds": True}, "owner": "zcode",
            })
            moved += 1
            moved_bytes += s["bytes"]
            print(f"[MOVED] superseded/{os.path.basename(src)}  {s['bytes']/1e6:.1f} MB")
            save_ledger(led)

    if not args.only or "duplicates" in args.only:
        for d in verify_duplicate_checkpoints():
            for src in d["redundant"]:
                if not os.path.exists(src):
                    continue
                rel = os.path.relpath(src, BASE)
                dest = os.path.join(ARCHIVE, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.exists(dest):
                    continue
                h = hashlib.sha256()
                with open(src, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                shutil.move(src, dest)
                led["entries"].append({
                    "id": f"duplicate/{rel}",
                    "group": "duplicates",
                    "moved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "from": src, "to": dest, "files": 1, "bytes": d["size"],
                    "fingerprint": h.hexdigest(),
                    "claim": f"byte-identical duplicate (sha {d['hash']}); keeper retained at "
                             f"{os.path.relpath(d['keep'], BASE)}",
                    "verification": {"claim_holds": True}, "owner": "zcode",
                })
                moved += 1
                moved_bytes += d["size"]
                print(f"[MOVED] duplicate/{rel}  {d['size']/1e6:.1f} MB")
                save_ledger(led)

    save_ledger(led)
    print(f"\n[ledger] {moved} items, {moved_bytes/1e9:.2f} GB archived -> {ARCHIVE}")
    print(f"[ledger] {LEDGER}")
    return 0


def cmd_list(args) -> int:
    led = load_ledger()
    print(f"ledger: {LEDGER}")
    print(f"created: {led.get('created_at')}  entries: {len(led['entries'])}")
    for e in led["entries"]:
        print(f"  {e['moved_at']}  {e['bytes']/1e6:9.1f} MB  {e['id']}")
    print(f"total: {sum(e['bytes'] for e in led['entries'])/1e9:.2f} GB")
    return 0


def cmd_restore(args) -> int:
    led = load_ledger()
    # newest entry wins: an item may have been archived, restored and re-archived
    for e in reversed(led["entries"]):
        if e["id"] == args.id:
            if os.path.exists(e["from"]):
                print(f"destination already exists: {e['from']}")
                return 2
            os.makedirs(os.path.dirname(e["from"]), exist_ok=True)
            shutil.move(e["to"], e["from"])
            print(f"restored {e['id']} -> {e['from']}")
            return 0
    print(f"no ledger entry with id {args.id!r}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verified archive ledger for B:\\datasets junk.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--json"); s.set_defaults(func=cmd_scan)
    a = sub.add_parser("apply"); a.add_argument("--only", nargs="*"); a.set_defaults(func=cmd_apply)
    sub.add_parser("list").set_defaults(func=cmd_list)
    r = sub.add_parser("restore"); r.add_argument("id"); r.set_defaults(func=cmd_restore)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
