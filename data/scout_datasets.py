"""scout_datasets.py — probe a list of HF datasets and rank them for verification use.

WHY THIS FILE EXISTS
--------------------
A list of 25 candidate fact datasets is not a plan. Before downloading anything I need
to know, per dataset: does it exist, how big, what schema, what licence, and — the one
that actually matters — **is it extractive?**

Extractive means the answer appears verbatim in the evidence, which makes verification
collapse into string matching. Measured on HaluEval: 95.5% extractive, and a two-line
substring rule scored 97.5%, beating a 20B judge. So extractiveness is the first filter,
not a footnote.

This tool answers all of that from the HF API without downloading the data, then ranks
the survivors so we only spend bandwidth on datasets that can measure something.

USAGE
-----
    python tools/data/scout_datasets.py --list            # the built-in candidate list
    python tools/data/scout_datasets.py --ids a/b c/d
    python tools/data/scout_datasets.py --list --out B:/datasets/forge/eval/scout.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "forge-scout/1.0 (research; local model zoo)"}
HF_API = "https://huggingface.co/api/datasets"
ROWS_API = "https://datasets-server.huggingface.co"

# --------------------------------------------------------------------------
# The candidate list (operator-supplied, 2026-10-21)
# --------------------------------------------------------------------------
CANDIDATES = [
    # fact QA / factuality
    "rubenroy/GammaCorpus-Fact-QA-450k",
    "ImperialCollegeLondon/health_fact",
    "Mutonix/RefGPT-Fact-v2",
    "Mutonix/RefGPT-Fact",
    "Polyglot-or-Not/Fact-Completion",
    "wenhu/tab_fact",
    "amanrangapur/Fin-Fact",
    "sardinelab/MF2-movie-facts-and-fibs",
    "factckbr/factckbr",
    "HC-85/open-food-facts",
    "datacommonsorg/datacommons_factcheck",
    "florath/coq-facts-props-proofs-gen0-v1",
    "siddharthl1293/engineering_design_facts",
    "Svngoku/The-Africa-Factbook",
    "mesolitica/mixtral-factual-QA",
    "vllm-sr/fact-check-classification-dataset",
    # FactNet family (the Wikipedia one)
    "openbmb/factnet_factstatements",
    "openbmb/factnet_factsense",
    "openbmb/factnet_factsynset",
    # evidence sufficiency / factuality scoring
    "copenlu/sufficient_facts",
    "google-research-datasets/xsum_factuality",
    "L1Fthrasir/Facts-true-false",
    "facebook/factual_reasoning",
]

# column names that indicate a (evidence, claim, label) shape
EVIDENCE_COLS = ("evidence", "context", "passage", "knowledge", "premise", "document",
                 "source", "sources", "article", "text_evidence", "evidence_text",
                 "supporting", "wiki_sentence", "sentences")
CLAIM_COLS = ("claim", "hypothesis", "statement", "question", "query", "input",
              "text", "prompt", "utterance")
LABEL_COLS = ("label", "labels", "verdict", "gold", "answer", "target", "factuality",
              "is_factual", "class", "y", "supported", "label_text")


def _get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def probe(ds_id: str) -> dict:
    """Metadata + schema + a small sample. No full download."""
    out = {"id": ds_id, "exists": False, "error": None}
    try:
        d = _get(f"{HF_API}/{urllib.parse.quote(ds_id)}")
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}"
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:60]}"
        return out

    out["exists"] = True
    out["private"] = d.get("private")
    out["gated"] = d.get("gated")
    card = d.get("cardData") or {}
    lic = card.get("license") or "unknown"
    if isinstance(lic, (list, tuple)):
        lic = ",".join(str(x) for x in lic)
    out["license"] = str(lic)
    out["downloads"] = d.get("downloads")
    out["likes"] = d.get("likes")
    out["tags"] = [t for t in (d.get("tags") or []) if t.startswith(("task_categories", "size_categories", "language"))][:6]

    # schema + row count + sample via the datasets-server (no download)
    try:
        splits = _get(f"{ROWS_API}/splits?dataset={urllib.parse.quote(ds_id)}")
        sl = (splits.get("splits") or [])
        out["splits"] = [f"{s['config']}:{s['split']}" for s in sl][:6]
        if sl:
            cfg, split = sl[0]["config"], sl[0]["split"]
            info = _get(f"{ROWS_API}/size?dataset={urllib.parse.quote(ds_id)}"
                        f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}")
            si = info.get("size") or {}
            out["rows"] = si.get("num_rows")
            out["bytes"] = si.get("num_bytes_original_files")
            first = _get(f"{ROWS_API}/first-rows?dataset={urllib.parse.quote(ds_id)}"
                         f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}")
            feats = [f["name"] for f in (first.get("features") or [])]
            out["columns"] = feats
            rows = (first.get("rows") or [])
            out["sample"] = rows[:2]
            out["_cfg_split"] = (cfg, split)
    except urllib.error.HTTPError as e:
        # 501 = the dataset ships a loading SCRIPT, which the viewer cannot run and
        # which `datasets 4.8` refuses to load at all. That is a real access cost,
        # not a data-quality verdict, so record it as such instead of silently
        # scoring it as "no data".
        if e.code == 501:
            out["script_based"] = True
            out["schema_error"] = "501 script-based (viewer unsupported)"
        else:
            out["schema_error"] = f"HTTP {e.code}"
    except Exception as e:
        out["schema_error"] = f"{type(e).__name__}: {str(e)[:60]}"
    return out


def _pick(cols: list[str], wanted: tuple) -> str | None:
    low = {c.lower(): c for c in cols or []}
    for w in wanted:
        if w in low:
            return low[w]
    for c in cols or []:
        cl = c.lower()
        if any(w in cl for w in wanted):
            return c
    return None


def extractiveness(info: dict) -> dict:
    """Is the claim verbatim inside the evidence? Fetches up to 100 rows."""
    cols = info.get("columns") or []
    ev = _pick(cols, EVIDENCE_COLS)
    cl = _pick(cols, CLAIM_COLS)
    lb = _pick(cols, LABEL_COLS)
    res = {"evidence_col": ev, "claim_col": cl, "label_col": lb}
    if not (ev and cl) or "_cfg_split" not in info:
        res["verdict"] = "no evidence+claim pair"
        return res

    def norm(s):
        if isinstance(s, (list, tuple)):
            s = " ".join(str(x) for x in s)
        elif isinstance(s, dict):
            s = json.dumps(s)
        return re.sub(r"\s+", " ", str(s or "").lower()).strip()

    cfg, split = info["_cfg_split"]
    try:
        d = _get(f"{ROWS_API}/rows?dataset={urllib.parse.quote(info['id'])}"
                 f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}"
                 f"&offset=0&length=100")
    except Exception as e:
        res["verdict"] = f"rows fetch failed ({type(e).__name__})"
        return res

    n = sub = 0
    labels = {}
    for r in (d.get("rows") or []):
        row = r.get("row") or {}
        e, c = norm(row.get(ev)), norm(row.get(cl))
        if not (e and c):
            continue
        n += 1
        if c in e:
            sub += 1
        if lb:
            labels[str(row.get(lb))] = labels.get(str(row.get(lb)), 0) + 1
    res["sampled"] = n
    res["extractive"] = sub
    res["extractive_rate"] = round(sub / n, 4) if n else None
    res["labels"] = dict(sorted(labels.items(), key=lambda kv: -kv[1])[:8])
    if not n:
        res["verdict"] = "no usable rows"
    elif sub / n > 0.5:
        res["verdict"] = "EXTRACTIVE (string matching)"
    else:
        res["verdict"] = "NON-EXTRACTIVE (usable)"
    return res


def score(info: dict) -> tuple[int, list[str]]:
    """Rank for our purpose: non-extractive + labelled + big + licensed."""
    pts, why = 0, []
    ex = info.get("extract") or {}
    v = ex.get("verdict") or ""
    if "NON-EXTRACTIVE" in v:
        pts += 50
        why.append("non-extractive")
    elif "EXTRACTIVE" in v:
        pts -= 30
        why.append("extractive")
    if ex.get("label_col"):
        pts += 20
        why.append(f"labelled({ex['label_col']})")
    if ex.get("evidence_col") and ex.get("claim_col"):
        pts += 10
        why.append("evidence+claim")
    rows = info.get("rows") or 0
    if rows >= 10000:
        pts += 15
        why.append(f"{rows:,} rows")
    elif rows >= 1000:
        pts += 8
        why.append(f"{rows:,} rows")
    lic = str(info.get("license") or "unknown").lower()
    if any(x in lic for x in ("mit", "apache", "cc0", "cc-by-4", "cc-by-3", "public")):
        pts += 10
        why.append(f"licence {lic}")
    elif "nc" in lic:
        pts -= 5
        why.append(f"licence {lic} (NC)")
    if info.get("gated"):
        pts -= 10
        why.append("gated")
    if info.get("script_based"):
        pts -= 8
        why.append("script-based (needs parquet/manual)")
    return pts, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scout HF fact datasets for verification use.")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    ids = args.ids or (CANDIDATES if args.list else CANDIDATES)
    print(f"[scout] probing {len(ids)} datasets\n")
    results = []
    for i, ds in enumerate(ids, 1):
        info = probe(ds)
        if info.get("exists") and not args.no_extract:
            try:
                info["extract"] = extractiveness(info)
            except Exception as e:
                info["extract"] = {"verdict": f"error {type(e).__name__}"}
        pts, why = score(info)
        info["score"] = pts
        info["why"] = why
        results.append(info)
        ex = (info.get("extract") or {}).get("verdict", "-")
        rows = info.get("rows")
        print(f"[{i}/{len(ids)}] {ds}")
        print(f"    exists={info.get('exists')} rows={rows if rows is not None else '?'} "
              f"lic={info.get('license')} gated={info.get('gated')}")
        if info.get("columns"):
            print(f"    cols={info['columns'][:8]}")
        if info.get("script_based"):
            print(f"    ACCESS: script-based (datasets 4.8 cannot load; needs parquet/manual)")
        print(f"    extract: {ex}")
        if info.get("extract", {}).get("labels"):
            print(f"    labels: {info['extract']['labels']}")
        print(f"    score={pts} {why}\n")
        time.sleep(0.2)

    results.sort(key=lambda r: -r["score"])
    print("=" * 78)
    print(f"{'score':>6}  {'rows':>9}  {'lic':<12} {'verdict':<28} id")
    print("-" * 78)
    for r in results:
        if not r.get("exists"):
            print(f"{r['score']:>6}  {'-':>9}  {'-':<12} {'MISSING/'+str(r.get('error'))[:26]:<28} {r['id']}")
            continue
        ex = (r.get("extract") or {}).get("verdict", "-")[:26]
        print(f"{r['score']:>6}  {r.get('rows') or 0:>9,}  {str(r.get('license'))[:12]:<12} "
              f"{ex:<28} {r['id']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, default=str)
        print(f"\n[scout] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
