"""decontam_text.py — 13-gram + MinHash decontamination that actually works on PROSE.

WHY THIS FILE EXISTS
--------------------
The family's only decontamination (`build models/scripts/decontaminate.py`) is
code-only by construction:

    assistant = turns.get("assistant", "") or text      # prose falls through
    user_text = turns.get("user", "")                    # "" for plain {text} rows
    if user_text:                                        # <-- layer 4 SKIPPED

On a wiki/philosophy/Gutenberg corpus it therefore returns
`verdict: CLEAN, flagged_total: 0` WITHOUT EVER COMPARING ANYTHING. That is the
PewDiePie failure mode wearing a green checkmark.

This module is schema-agnostic and works on plain text:
  * Layer 1  exact   — sha256 of normalized text vs vault reference/prompt
  * Layer 2  13-gram — word 13-gram overlap (GPT-3 / standard practice).
                       ANY shared 13-gram is a hit. This is the primary gate.
  * Layer 3   8-gram — reported as a WARNING band (higher recall, more noise)
  * Layer 4  minhash — Jaccard >= 0.7 on char 5-grams (paraphrase of a solution)
  * Layer 5  prompt  — prompt-side leak of the vault `prompt_text`

Anything long enough to swallow a benchmark question verbatim (e.g. a 400 KB
Gutenberg chunk) is handled by layer 2 sliding over the whole row.

USAGE
-----
    python tools/data/decontam_text.py --self-test
    python tools/data/decontam_text.py --input <corpus.jsonl> \
        --output <clean.jsonl> --report <gate.json> [--quarantine <bad.jsonl>]

Exit codes: 0 CLEAN · 2 CONTAMINATED · 3 CONTAMINATED (but wrote output) · 1 error
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

VAULT = r"B:\datasets\forge\refs\vault.jsonl"
MANIFEST = r"B:\datasets\forge\refs\vault-manifest.json"

# --- thresholds (kept explicit so the gate is auditable) -------------------
NGRAM_STRICT = 13          # hit -> CONTAMINATED  (standard benchmark-decontam width)
NGRAM_WARN = 8             # hit -> WARN band only
MINHASH_THRESHOLD = 0.7    # matches PEWDIEPIE-LESSONS-SYNTHESIS.md Lesson 1
MINHASH_SHINGLE = 5
MASK64 = (1 << 64) - 1
BASE = 1000003

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def sha(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Rolling n-gram hashing (fast enough for 400 KB rows)
# --------------------------------------------------------------------------

class NgramIndex:
    """Word-level n-gram hash index over the reference vault."""

    def __init__(self, n: int):
        self.n = n
        self.hashes: set[int] = set()
        self.owner: dict[int, str] = {}     # hash -> source (for reporting)
        self._vocab: dict[str, int] = {}
        self._next = 1
        self.pow = pow(BASE, n - 1, 1 << 64)

    def _ids(self, toks: list[str]) -> list[int]:
        v, out = self._vocab, []
        nxt = self._next
        for t in toks:
            i = v.get(t)
            if i is None:
                i = nxt
                v[t] = i
                nxt += 1
            out.append(i)
        self._next = nxt
        return out

    def _iter_hashes(self, ids: list[int]):
        """Yield the rolling hash of every length-n window.

        Order matters: the oldest term must be REMOVED BEFORE the multiply,
        otherwise the accumulated state carries the whole prefix and only
        verbatim copies match (a mid-document slice would not).
        """
        n, pow_, base, mask = self.n, self.pow, BASE, MASK64
        h = 0
        for i, wid in enumerate(ids):
            if i >= n:
                h = (h - ids[i - n] * pow_) & mask
            h = (h * base + wid) & mask
            if i >= n - 1:
                yield h

    def add(self, text: str, source: str) -> int:
        ids = self._ids(tokenize(text))
        if len(ids) < self.n:
            return 0
        added = 0
        for h in self._iter_hashes(ids):
            if h not in self.hashes:
                self.hashes.add(h)
                self.owner[h] = source
                added += 1
        return added

    def first_hit(self, text: str) -> str | None:
        """Return the vault source of the first shared n-gram, else None."""
        ids = self._ids(tokenize(text))
        if len(ids) < self.n:
            return None
        for h in self._iter_hashes(ids):
            if h in self.hashes:
                return self.owner.get(h, "?")
        return None

    def count_hits(self, text: str, cap: int = 1000) -> int:
        ids = self._ids(tokenize(text))
        if len(ids) < self.n:
            return 0
        c = 0
        for h in self._iter_hashes(ids):
            if h in self.hashes:
                c += 1
                if c >= cap:
                    return c
        return c


def char_shingles(text: str, k: int = MINHASH_SHINGLE) -> set[str]:
    t = normalize_text(text)
    if len(t) <= k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------------
# Field extraction — works across every schema we own
# --------------------------------------------------------------------------

def row_texts(rec: dict) -> dict[str, str]:
    """Return {field_label: text} for every text-bearing field in a row."""
    out = {}
    if isinstance(rec.get("text"), str):
        out["text"] = rec["text"]
    if isinstance(rec.get("messages"), list):
        parts = []
        for m in rec["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        if parts:
            out["messages"] = "\n".join(parts)
    for k in ("prompt", "instruction", "question", "input"):
        if isinstance(rec.get(k), str):
            out[k] = rec[k]
    for k in ("completion", "response", "output", "answer"):
        if isinstance(rec.get(k), str):
            out[k] = rec[k]
    if isinstance(rec.get("code"), str):
        out["code"] = rec["code"]
    return out


# --------------------------------------------------------------------------
# Vault loading
# --------------------------------------------------------------------------

class Vault:
    def __init__(self, path: str = VAULT):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"reference vault missing: {path}\n"
                f"run: python tools/data/refs_vault.py build")
        self.path = path
        self.sha256 = _sha256(path)
        self.count = 0
        self.per_source = collections.Counter()
        self.exact: set[str] = set()
        self.n13 = NgramIndex(NGRAM_STRICT)
        self.n8 = NgramIndex(NGRAM_WARN)
        self.minhash_index: list[tuple[str, set[str]]] = []
        self._load()

    def _load(self):
        t0 = time.time()
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                src = r.get("source", "?")
                self.count += 1
                self.per_source[src] += 1
                for field in ("reference_text", "prompt_text"):
                    t = r.get(field) or ""
                    if not t:
                        continue
                    self.exact.add(sha(t))
                    self.n13.add(t, src)
                    self.n8.add(t, src)
                    if len(t) < 4000:            # cap minhash memory
                        self.minhash_index.append((src, char_shingles(t)))
        self.load_seconds = time.time() - t0

    def stats(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "entries": self.count,
            "per_source": dict(sorted(self.per_source.items())),
            "n13_grams": len(self.n13.hashes),
            "n8_grams": len(self.n8.hashes),
            "minhash_sigs": len(self.minhash_index),
            "load_seconds": round(self.load_seconds, 1),
        }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Core check
# --------------------------------------------------------------------------

def check_row(rec: dict, vault: Vault, minhash: bool = True) -> dict:
    """Return {verdict, layer, match_source, detail}."""
    texts = row_texts(rec)
    if not texts:
        return dict(verdict="SKIP", layer=None, match_source=None,
                    detail="no text-bearing field")

    warn_hit = None
    for label, text in texts.items():
        if not text.strip():
            continue

        # Layer 1 — exact
        if sha(text) in vault.exact:
            return dict(verdict="CONTAMINATED", layer="exact", match_source=None,
                        detail=f"exact normalized match in field '{label}'")

        # Layer 2 — 13-gram (primary)
        src = vault.n13.first_hit(text)
        if src is not None:
            n = vault.n13.count_hits(text)
            return dict(verdict="CONTAMINATED", layer="13gram", match_source=src,
                        detail=f"{n} shared 13-gram(s) in field '{label}'")

        # Layer 3 — 8-gram (warn band)
        if warn_hit is None:
            ws = vault.n8.first_hit(text)
            if ws is not None:
                warn_hit = (label, ws)

    # Layer 4 — minhash near-duplicate (paraphrase of a solution)
    if minhash and vault.minhash_index:
        for label, text in texts.items():
            if len(text) > 20000:            # skip huge rows for speed
                continue
            sh = char_shingles(text)
            if not sh:
                continue
            for src, ref in vault.minhash_index:
                if jaccard(sh, ref) >= MINHASH_THRESHOLD:
                    return dict(verdict="CONTAMINATED", layer="minhash",
                                match_source=src,
                                detail=f"jaccard>={MINHASH_THRESHOLD} vs {src}")

    if warn_hit:
        return dict(verdict="WARN", layer="8gram", match_source=warn_hit[1],
                    detail=f"8-gram overlap (field '{warn_hit[0]}') below 13-gram bar")

    return dict(verdict="CLEAN", layer=None, match_source=None, detail="")


# --------------------------------------------------------------------------
# Self-test — proves the gate can actually reject
# --------------------------------------------------------------------------

_SELFTEST_FIXTURES = [
    # (label, row, expected_verdict)
    ("clean_prose",
     {"text": "The committee adjourned after a long debate about irrigation rights "
              "in the northern valley, and nobody expected to reconvene before autumn."},
     "CLEAN"),
    ("clean_short", {"text": "It rained."}, "CLEAN"),
    ("gsm8k_verbatim", None, "CONTAMINATED"),        # filled from vault at runtime
    ("humaneval_solution", None, "CONTAMINATED"),    # filled from vault at runtime
]


def run_self_test() -> bool:
    print("decontam_text.py — SELF TEST")
    try:
        vault = Vault()
    except FileNotFoundError as e:
        print(f"  FAIL: {e}")
        return False
    print(f"  vault: {vault.count} entries, {len(vault.n13.hashes)} 13-grams, "
          f"sha {vault.sha256[:16]}...")

    # Pull real contaminating text out of the vault itself.
    gsm = hl = None
    with open(vault.path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if gsm is None and r["source"] == "gsm8k":
                gsm = r
            if hl is None and r["source"] == "humaneval" and r.get("reference_text"):
                hl = r
            if gsm and hl:
                break

    checks = []
    for label, row, expected in _SELFTEST_FIXTURES:
        if label == "gsm8k_verbatim":
            row = {"text": gsm["prompt_text"] + " " + gsm["reference_text"]}
        elif label == "humaneval_solution":
            row = {"text": hl["prompt_text"] + "\n" + hl["reference_text"]}
        res = check_row(row, vault)
        ok = res["verdict"] == expected
        checks.append((label, expected, res["verdict"], res.get("layer"), ok))
        if not ok:
            print(f"  detail: {res}")

    # A row literally containing a 13-word slice must be caught
    slice_row = {"text": "prefix words here " + " ".join(gsm["prompt_text"].split()[:20]) + " suffix"}
    res = check_row(slice_row, vault)
    checks.append(("gsm8k_20word_slice", "CONTAMINATED", res["verdict"], res.get("layer"),
                   res["verdict"] == "CONTAMINATED"))

    ok_all = True
    for label, exp, got, layer, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:26s} expected={exp:13s} got={got:13s} layer={layer}")
        ok_all = ok_all and ok

    print(f"SELF TEST: {'ALL PASS' if ok_all else 'FAILURES'}")
    return ok_all


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    vault = Vault(args.vault)
    vstats = vault.stats()
    print(f"[gate] vault {vstats['entries']} entries / {vstats['n13_grams']} 13-grams "
          f"(sha {vstats['sha256'][:16]})")

    t0 = time.time()
    total = kept = clean = warn = contam = skip = 0
    by_layer = collections.Counter()
    by_source = collections.Counter()
    examples = []

    fout = open(args.output, "w", encoding="utf-8")
    fq = open(args.quarantine, "w", encoding="utf-8") if args.quarantine else None
    try:
        with open(args.input, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skip += 1
                    continue
                res = check_row(rec, vault, minhash=not args.no_minhash)
                v = res["verdict"]
                if v == "CONTAMINATED":
                    contam += 1
                    by_layer[res["layer"]] += 1
                    by_source[res["match_source"] or "?"] += 1
                    if fq:
                        fq.write(json.dumps({**rec, "_gate": res}, ensure_ascii=False) + "\n")
                    if len(examples) < 25:
                        examples.append({k: res[k] for k in ("layer", "match_source", "detail")})
                    continue
                if v == "WARN":
                    warn += 1
                if v in ("CLEAN", "WARN"):
                    clean += 1
                if fq is None and v == "CONTAMINATED":
                    continue
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                if total % 2000 == 0:
                    print(f"  ... {total} rows, {contam} contaminated "
                          f"({100*contam/max(total,1):.3f}%)", flush=True)
    finally:
        fout.close()
        if fq:
            fq.close()

    dur = time.time() - t0
    verdict = "CONTAMINATED" if contam else "CLEAN"
    report = {
        "gate": "decontam_text/v1",
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "quarantine": os.path.abspath(args.quarantine) if args.quarantine else None,
        "vault": vstats,
        "thresholds": {
            "ngram_strict": NGRAM_STRICT,
            "ngram_warn": NGRAM_WARN,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_shingle": MINHASH_SHINGLE,
        },
        "rows_in": total,
        "rows_kept": kept,
        "rows_contaminated": contam,
        "rows_warn": warn,
        "rows_parse_skip": skip,
        "contamination_rate": round(contam / max(total, 1), 8),
        "hits_by_layer": dict(by_layer),
        "hits_by_source": dict(by_source.most_common(20)),
        "examples": examples,
        "seconds": round(dur, 1),
        "verdict": verdict,
        "self_test": "run --self-test separately and record the result",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print(f"[gate] rows_in={total} kept={kept} contaminated={contam} warn={warn} "
          f"skip={skip} in {dur:.1f}s")
    print(f"[gate] hits_by_layer={dict(by_layer)}")
    if by_source:
        print(f"[gate] hits_by_source={dict(by_source.most_common(10))}")
    print(f"[gate] VERDICT={verdict}")
    return 0 if verdict == "CLEAN" else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="13-gram + MinHash decontamination for prose AND code.")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--quarantine", help="write contaminated rows here")
    ap.add_argument("--report", help="write gate JSON here")
    ap.add_argument("--vault", default=VAULT)
    ap.add_argument("--no-minhash", action="store_true", help="skip layer 4 (faster)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --self-test)")
    try:
        return cmd_run(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
