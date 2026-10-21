# `data/` — the corpus gate

Everything here exists because the family shipped a corpus labelled
`spine_train_clean.jsonl` that had **never been compared against a single
benchmark**, and a `decontaminate.py` that returns `CLEAN, 0 flagged` on prose
without evaluating anything.

## The contract

```python
from data.gate import guard
guard(data_path)      # raises unless a sibling *.gate.json says verdict=CLEAN
```

`tools/tm/train_arc.py` calls this **before `import torch`**, so the guard lives in
the trainer, not in a wrapper. No gate file → training refuses.

Escape hatch for deliberate control runs: `FORGE_SKIP_GATE=1` (prints a loud
warning; numbers from that run are not publishable).

## Tools

| file | role | self-test |
|---|---|---|
| `refs_vault.py` | build/verify the frozen benchmark reference vault | `verify` (hash + row count) |
| `decontam_text.py` | **prose-capable** decontamination: 13-gram primary, 8-gram warn band, MinHash, exact, prompt-leak | `--self-test` → **5/5** |
| `gate.py` | the gate contract + local MinHashLSH dedup + eval-overlap guard | positive/negative via `check` |
| `license_gate.py` | **G-LIC** — resolve + tag licence and commercial status on every row | `resolve` / `stats` |
| `pii_gate.py` | **G-PII** — credentials, PEM, JWT, cards, phone, email, IP, IBAN, TCKN; redact by default | `--self-test` → **24/24** |
| `archive_ledger.py` | two-phase verified archive with sha256 ledger + `restore` | `scan` then `apply` |
| `license_overrides.json` | manual licence overrides; every entry needs an `evidence` field | — |

## Quick start

```powershell
python data/refs_vault.py build          # ~40k reference entries, sha256-pinned
python data/refs_vault.py verify

python data/decontam_text.py --self-test
python data/license_gate.py resolve --corpus <corpus.jsonl>
python data/pii_gate.py --self-test

python data/decontam_text.py --input A --output B --report gates/A.decontam.json
python data/license_gate.py  tag --input B --output C --report gates/A.license.json
python data/pii_gate.py      --input C --output D --report gates/A.pii.json
python data/gate.py assemble --corpus A --input A --output D `
    --decontam-report gates/A.decontam.json `
    --license-report gates/A.license.json `
    --pii-report gates/A.pii.json `
    --eval <eval.jsonl> --dedup --self-test-pass 5/5
python data/gate.py check --data D
```

## Gate record

`*.gate.json` carries: verdict, rows in/out, reference-vault hash + count,
per-layer decontamination hits with examples, dedup stats, eval-overlap stats,
licence distribution, PII stats, and the gate's own self-test result.

`require_clean()` demands all of: gate file exists · `verdict == CLEAN` ·
required fields present · `decontam_self_test` is `n/n` (the gate must prove it
can **reject**).

## Bugs these self-tests caught (in this code, on real data)

| bug | symptom | fixed by |
|---|---|---|
| rolling hash subtracted after multiplying | a 20-word mid-document slice scored CLEAN | subtract-before-multiply |
| `_row_is_empty` returned `not d` | 2M empty rows passed verification as data | return `True` when no text field |
| dedup keyed on `text` only | **2,681 of 2,682 v6 rows declared duplicates** | schema-agnostic `_row_text()` + abort guard |
| phone regex with optional separators | `1000000007` (prime modulus) and `2026040510` (timestamp) rewritten to `[PHONE]` inside SQL code | separators required + `_phone_ok()` structure check |
| card regex `(?:\d[ \-]?){13,16}` | citation range `Berleant [1985 … 2013]` became a Luhn-valid card | bare-or-grouped only |
| placeholder suppression too broad | the entropy floor stopped flagging **every phone number** in a therapy corpus | severity-aware: floor for secrets, not PII |

Every one of those has a permanent fixture so it cannot regress.

## Refuted claims

`archive_ledger.py scan` disproved two items from a prior audit:

1. **`fenrir_v2_1` is real data** — 199,740 distinct rows, 0.68 GB of
   cybersecurity content, mislabelled "6 × 99,870 empty rows". Empty-shard
   verification now requires *empty AND degenerate*.
2. **The superseded-bitqwen sweep tried to include `cosmopedia.jsonl`,
   `book_rag_chunks.jsonl` and `docker_recovered.jsonl`** — unrelated datasets in
   the same folder. Strict name pattern applied; the remaining 22 version-stack
   artifacts are reported with `auto: false` because they belong to zcode.

> Verify the claim, not the report.

## Status

| corpus | rows | verdict | decontam | dedup | secrets | licence |
|---|---|---|---|---|---|---|
| `spine_train` | 19,592 → 19,303 | CLEAN | 77 (gsm8k 72, truthfulqa 5) | 210 | 0 | 1.0 |
| `v6-sft` | 2,682 → **858** | CLEAN | 0 | **1,824** | 0 | 1.0 |

`v6-sft` collapsing by 68% is the single most important number here: the corpus
we trained 600 steps on was **68% exact duplicates**. Prompt diversity, not row
count, is the binding constraint.
