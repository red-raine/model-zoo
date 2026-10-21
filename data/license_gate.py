"""license_gate.py — G-LIC. Tag every row with its license and commercial status.

WHY THIS FILE EXISTS
--------------------
`sources_ext.py` records a `license` per source and a `flags.needs_deid`, and
NOTHING READS THEM. Meanwhile:

  * EI-50K          -> "helpingai" custom license
  * Amod/mental...  -> RAIL-D  (LICENSE-RAIL-D.txt is sitting in B:\\datasets\\raw)
  * EmpatheticDialogues -> CC-BY-NC-4.0  (non-commercial)

all of those rows already flow into `B:\\datasets\\optimized\\optimized_training.jsonl`.
An untagged corpus is a corpus you cannot legally ship and cannot audit.

This module makes the license a first-class row field and refuses to pretend.

Policy (from operator, 2026-10-20): **internal-only**. Unknown licenses are ALLOWED
for research lanes but are tagged `commercial=unknown`. Nothing is silently dropped.

USAGE
-----
    python tools/data/license_gate.py resolve          # build/refresh the registry
    python tools/data/license_gate.py resolve --offline
    python tools/data/license_gate.py tag --input A.jsonl --output B.jsonl --report R.json
    python tools/data/license_gate.py stats --registry

Resolution order for each source:
    1. explicit override in tools/data/license_overrides.json
    2. subprojects/dataset_pipeline/sources.py  (+ sources_ext.py)
    3. local LICENSE / README card file in B:\\datasets\\raw\\<dir>
    4. HuggingFace API   https://huggingface.co/api/datasets/<repo>  (free, cached)
    5. -> "unknown"  (tagged, counted, never hidden)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

REGISTRY = r"B:\datasets\forge\refs\license-registry.json"
OVERRIDES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "license_overrides.json")
RAW_DIR = r"B:\datasets\raw"
SOURCES_PY = r"E:\vibe_coding\dev\bitnet runner\subprojects\dataset_pipeline\sources.py"

# --------------------------------------------------------------------------
# License normalisation
# --------------------------------------------------------------------------

# canonical -> (commercial, family, note)
LICENSE_MAP = {
    "mit":                      ("yes", "permissive", ""),
    "apache-2.0":               ("yes", "permissive", ""),
    "apache2.0":                ("yes", "permissive", ""),
    "bsd-2-clause":             ("yes", "permissive", ""),
    "bsd-3-clause":             ("yes", "permissive", ""),
    "cc0-1.0":                  ("yes", "public-domain", ""),
    "public-domain":            ("yes", "public-domain", ""),
    "public-domain-us":         ("yes", "public-domain", "US public domain"),
    "cc-by-4.0":                ("yes", "attribution", "must attribute"),
    "cc-by-sa-4.0":             ("yes", "attribution-sharealike",
                                 "share-alike: derivative corpus inherits the license"),
    "cdla-permissive-2.0":      ("yes", "permissive", ""),
    "cc-by-nc-4.0":             ("no",  "non-commercial", "NC: research only"),
    "cc-by-nc-sa-4.0":          ("no",  "non-commercial", "NC + share-alike"),
    "cc-by-nc-nd-4.0":          ("no",  "non-commercial", "NC + no-derivatives"),
    "rail-d":                   ("restricted", "responsible-ai",
                                 "RAIL-D: use restrictions apply — review before shipping"),
    "openrail":                 ("restricted", "responsible-ai", "use restrictions"),
    "openrail++":               ("restricted", "responsible-ai", "use restrictions"),
    "gated":                    ("restricted", "gated",
                                 "HF repo is gated/401 — terms must be accepted, licence unread"),
    "helpingai":                ("unknown", "custom", "custom HelpingAI license — needs legal read"),
    "other":                    ("unknown", "unstated", "upstream says 'other'; unclear"),
    "n/s":                      ("unknown", "unstated", "not stated by upstream"),
    "unknown":                  ("unknown", "unstated", ""),
    "":                         ("unknown", "unstated", ""),
}


def classify(raw: str | None) -> dict:
    key = (raw or "").strip().lower()
    key = key.replace(" ", "-").replace("_", "-")
    entry = LICENSE_MAP.get(key)
    if entry is None:
        for k, v in LICENSE_MAP.items():
            if k and (k in key or key in k):
                entry = v
                break
    if entry is None:
        entry = ("unknown", "unrecognised", f"unrecognised license string {raw!r}")
    commercial, family, note = entry
    return {"license": (raw or "unknown"), "canonical": key,
            "commercial": commercial, "family": family, "note": note}


# --------------------------------------------------------------------------
# Resolution sources
# --------------------------------------------------------------------------

def from_sources_py() -> dict[str, dict]:
    out = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_src", SOURCES_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, v in getattr(mod, "SOURCES", {}).items():
            out[k] = {"license": v.get("license"), "category": v.get("category"),
                      "origin": "sources.py"}
    except Exception as e:
        print(f"  warn: sources.py not loaded ({e})")
    ext = SOURCES_PY.replace("sources.py", "sources_ext.py")
    if os.path.exists(ext):
        try:
            spec = importlib.util.spec_from_file_location("_ext", ext)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(SOURCES_PY))
            spec.loader.exec_module(mod)
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, dict) and name.isupper():
                    for k, v in obj.items():
                        if isinstance(v, dict) and "license" in v:
                            out.setdefault(k, {"license": v.get("license"),
                                               "category": v.get("category"),
                                               "origin": f"sources_ext.{name}"})
        except Exception as e:
            print(f"  warn: sources_ext.py not loaded ({e})")
    return out


_INFER = re.compile(
    r"(?:license|licence)\s*[:\-]\s*\[?([a-z0-9\.\-\+ ]{2,40})\]?", re.I)


def from_local_files() -> dict[str, dict]:
    """Scan B:\\datasets\\raw\\<dir> for LICENSE / README with a license line."""
    out = {}
    if not os.path.isdir(RAW_DIR):
        return out
    for d in os.listdir(RAW_DIR):
        p = os.path.join(RAW_DIR, d)
        if not os.path.isdir(p):
            continue
        found = None
        for root, _dirs, files in os.walk(p):
            if "/.cache" in root.replace("\\", "/"):
                continue
            for f in files:
                low = f.lower()
                if not (low.startswith("license") or low.startswith("licence")
                        or low.startswith("readme")):
                    continue
                fp = os.path.join(root, f)
                if os.path.getsize(fp) > 200_000:
                    continue
                try:
                    t = open(fp, encoding="utf-8", errors="replace").read(4000)
                except OSError:
                    continue
                if low.startswith(("license", "licence")):
                    name = f.upper()
                    for tok in ("RAIL-D", "MIT", "APACHE", "CC-BY-NC", "CC-BY-SA",
                                "CC-BY", "BSD", "CC0", "GPL", "HELPINGAI"):
                        if tok in name:
                            found = tok.lower().replace("cc-by-", "CC-BY-")
                            break
                if not found:
                    m = _INFER.search(t)
                    if m:
                        found = m.group(1).strip()
                if found:
                    break
            if found:
                break
        if found:
            out[d] = {"license": found, "origin": "local-file"}
    return out


def from_hf_api(names: list[str], cache: dict) -> dict[str, dict]:
    out = {}
    for n in names:
        if n in cache:
            continue
        # try the name as given, then common short forms
        cands = [n]
        if "/" not in n:
            continue
        for c in cands:
            url = f"https://huggingface.co/api/datasets/{c}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "forge/1.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read().decode("utf-8"))
                lic = (data.get("cardData") or {}).get("license")
                if lic:
                    out[n] = {"license": lic, "origin": "hf-api"}
                    break
                out[n] = {"license": "unknown", "origin": "hf-api:no-license"}
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    out[n] = {"license": "unknown", "origin": "hf-api:404"}
                    break
                if e.code in (401, 403):
                    # Gated / access-controlled repo: a real license signal.
                    out[n] = {"license": "gated", "origin": f"hf-api:{e.code}"}
                    break
                continue
            except Exception:
                out[n] = {"license": "unknown", "origin": "hf-api:error"}
                break
        time.sleep(0.05)
    return out


def load_overrides() -> tuple[dict[str, dict], dict[str, str]]:
    if not os.path.exists(OVERRIDES):
        return {}, {}
    raw = json.load(open(OVERRIDES, encoding="utf-8"))
    aliases = raw.pop("_aliases", {}) or {}
    aliases = {k: v for k, v in aliases.items() if not k.startswith("_")}
    ov = {k: v for k, v in raw.items() if not k.startswith("_")}
    return ov, aliases


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def corpus_sources(paths: list[str]) -> collections.Counter:
    c = collections.Counter()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c[str(d.get("source", "?"))] += 1
    return c


def cmd_resolve(args) -> int:
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    print("[lic] reading sources.py / sources_ext.py")
    reg = from_sources_py()
    print(f"  {len(reg)} entries")

    print("[lic] reading local LICENSE / README files")
    loc = from_local_files()
    print(f"  {len(loc)} entries: {sorted(loc)[:8]}")

    print("[lic] scanning corpora for source names")
    cnt = corpus_sources(args.corpus)
    print(f"  {len(cnt)} distinct sources across {len(args.corpus)} corpora")

    ov, aliases = load_overrides()
    print(f"[lic] overrides: {len(ov)} | aliases: {len(aliases)}")

    unresolved = [s for s in cnt if s not in ov and s not in reg]
    if args.offline:
        print(f"[lic] offline: skipping HF API for {len(unresolved)} sources")
        hf = {}
    else:
        print(f"[lic] querying HF API for {len(unresolved)} sources ...")
        # resolve short labels through the alias map before asking HF
        looked_up = {}
        for s in unresolved:
            looked_up[aliases.get(s, s)] = s
        raw_hf = from_hf_api(list(looked_up), {})
        hf = {looked_up[k]: v for k, v in raw_hf.items()}

    merged = {}
    for s, n in cnt.most_common():
        rec = {"rows": n}
        if s in ov:
            rec.update(ov[s]); rec["resolved_by"] = "override"
        elif s in reg:
            rec.update(reg[s]); rec["resolved_by"] = "sources.py"
        elif s in hf:
            rec.update(hf[s]); rec["resolved_by"] = hf[s]["origin"]
        elif s in loc:
            rec.update(loc[s]); rec["resolved_by"] = "local-file"
        elif s in loc:
            rec.update(loc[s]); rec["resolved_by"] = "local-file"
        else:
            rec.update({"license": "unknown", "resolved_by": "none"})
        rec.update(classify(rec.get("license")))
        merged[s] = rec

    # attach local-file hits whose names differ from HF ids
    for d, v in loc.items():
        for s in merged:
            if s.split("/")[-1].lower().startswith(d.lower().split("-")[0].lower()) \
                    and merged[s]["resolved_by"] == "none":
                merged[s].update(v)
                merged[s]["resolved_by"] = "local-file"

    out = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "internal-only (unknown allowed, tagged, never hidden)",
        "corpora_scanned": [os.path.abspath(p) for p in args.corpus],
        "sources": merged,
        "summary": {
            "sources": len(merged),
            "commercial_yes": sum(1 for v in merged.values() if v["commercial"] == "yes"),
            "commercial_no": sum(1 for v in merged.values() if v["commercial"] == "no"),
            "commercial_restricted": sum(1 for v in merged.values() if v["commercial"] == "restricted"),
            "commercial_unknown": sum(1 for v in merged.values() if v["commercial"] == "unknown"),
            "rows_unknown": sum(v["rows"] for v in merged.values() if v["commercial"] == "unknown"),
            "rows_total": sum(v["rows"] for v in merged.values()),
        },
    }
    out["summary"]["coverage"] = round(
        1 - out["summary"]["rows_unknown"] / max(out["summary"]["rows_total"], 1), 4)

    with open(REGISTRY, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n[lic] wrote {REGISTRY}")
    for k, v in out["summary"].items():
        print(f"  {k:24s} {v}")
    print("\n[lic] unresolved sources (need overrides):")
    for s, v in merged.items():
        if v["resolved_by"] == "none":
            print(f"  {v['rows']:6d}  {s}")
    return 0


def cmd_tag(args) -> int:
    if not os.path.exists(REGISTRY):
        print("no registry — run `resolve` first")
        return 1
    reg = json.load(open(REGISTRY, encoding="utf-8"))["sources"]

    total = tagged = 0
    dist = collections.Counter()
    unknown_rows = 0
    offenders = collections.Counter()

    fin = open(args.input, encoding="utf-8")
    fout = open(args.output, "w", encoding="utf-8")
    try:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            src = str(d.get("source", "?"))
            r = reg.get(src) or {}
            lic = r.get("license", "unknown")
            com = r.get("commercial", "unknown")
            d["license"] = lic
            d["commercial"] = com
            tagged += 1
            dist[f"{lic}|{com}"] += 1
            if com in ("unknown", "no", "restricted"):
                offenders[src] += 1
                if com == "unknown":
                    unknown_rows += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    finally:
        fin.close()
        fout.close()

    report = {
        "gate": "license/v1",
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "registry": REGISTRY,
        "policy": "internal-only",
        "rows_total": total,
        "rows_tagged": tagged,
        "license_coverage": round(tagged / max(total, 1), 6),
        "rows_commercial_unknown": unknown_rows,
        "commercial_coverage": round(1 - unknown_rows / max(total, 1), 6),
        "distribution": dict(dist.most_common()),
        "sources_needing_action": dict(offenders.most_common(30)),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(f"[lic] {total} rows | tagged {tagged} | "
          f"license_coverage={report['license_coverage']} "
          f"commercial_coverage={report['commercial_coverage']}")
    for k, v in report["distribution"].items():
        print(f"  {k:44s} {v}")
    if offenders:
        print("[lic] sources needing action:")
        for s, n in offenders.most_common(15):
            print(f"  {n:6d}  {s}")
    return 0


def cmd_stats(args) -> int:
    if not os.path.exists(REGISTRY):
        print("no registry")
        return 1
    d = json.load(open(REGISTRY, encoding="utf-8"))
    print(json.dumps(d["summary"], indent=2))
    print(f"\n{'rows':>7}  {'commercial':>11}  {'license':24s} source")
    for s, v in sorted(d["sources"].items(), key=lambda kv: -kv[1]["rows"]):
        print(f"{v['rows']:7d}  {v['commercial']:>11}  {str(v['license'])[:24]:24s} {s}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G-LIC: license tagging gate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve")
    r.add_argument("--corpus", nargs="+", default=[
        r"B:\datasets\forge\cleaned\spine_train.jsonl",
    ])
    r.add_argument("--offline", action="store_true")
    r.set_defaults(func=cmd_resolve)

    t = sub.add_parser("tag")
    t.add_argument("--input", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--report")
    t.set_defaults(func=cmd_tag)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
