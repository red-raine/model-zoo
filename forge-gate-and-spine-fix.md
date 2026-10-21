# Forge gate + spine fix — "the tool existing is not the gate firing"

**Date:** 2026-10-20 · **Owner:** opencode · **Status:** Wave 7/8 groundwork landed

---

## The bug

`B:\datasets\spine\simple_decontam.py` said, in its own docstring:

> `1. Exact matches to HumanEval/MBPP problem statements`

It never loaded HumanEval. It never loaded MBPP. It had no reference set and no hashing.
It did three things: drop empty rows, drop rows under 50 characters, and drop rows
matching four code-shape regexes.

Then its output was copied **over** its own input, so `spine_train.jsonl` and
`spine_train_clean.jsonl` became byte-identical (`E2947078…`) — one of them named
"clean" — and the 19,592-row source was destroyed.

Everyone downstream believed the spine was decontaminated. **It had never been compared
against a single benchmark.**

This is verbatim PewDiePie mistake #1, and the family already had the lesson written down
in `E:\vibe_coding\glm\build models\research\PEWDIEPIE-MISTAKES-REPEATING.md`:

> The tool existing is not the gate firing.

---

## Why the existing `decontaminate.py` could not have saved us

`E:\vibe_coding\glm\build models\scripts\decontaminate.py` is a competent 4-layer gate —
for **code**. On prose it does this:

```python
assistant = turns.get("assistant", "") or text   # prose falls through to "the text"
user_text = turns.get("user", "")                # "" for plain {"text": ...} rows
if user_text:                                    # <-- layer 4 SKIPPED ENTIRELY
```

Consequences on a Gutenberg/wiki/philosophy corpus:

| Layer | Behaviour on prose |
|---|---|
| 1 exact | normalizes prose, hashes it, never equals a HumanEval solution hash → miss |
| 2 name+body | prose has no `def` → no candidates → miss |
| 3 minhash | char 5-gram Jaccard vs HumanEval/MBPP solutions → practically never ≥0.7 → miss |
| 4 prompt-leak | `if user_text:` is False → **skipped** |

So it returns `verdict: "CLEAN", flagged_total: 0` **without evaluating anything**.
A vacuous pass. And `--eval-sets` only accepts `humaneval,mbpp`.

Two more dead gates found in the same sweep:

* `subprojects/dataset_pipeline/validate_passes.py:395` — P0 decontamination is a literal `pass`:
  ```python
  removed = 0
  if decon and hasattr(decon, "MINHASH"):
      pass  # decontaminate is imported; heavy dedup runs later in pipeline
  ```
  It is a double no-op: the body does nothing **and** `hasattr(decon, "MINHASH")` is always
  False (the constant is `MINHASH_THRESHOLD`).
* `subprojects/dataset_pipeline/recipe.py:34` —
  `"decontam_refs": ["mmlu", "arc", "hellaswag", "winogrande"]` has **zero consumers**.
  Dead config. Those reference sets never existed anywhere in the family.

---

## What was built

### 1. `tools/data/refs_vault.py` — the frozen reference vault

```powershell
python tools/data/refs_vault.py build      # download + build + hash
python tools/data/refs_vault.py verify     # re-hash, compare manifest
```

`B:\datasets\forge\refs\vault.jsonl` + `vault-manifest.json`

**40,303 entries** across 9 benchmarks (was 1,038 code-only):

| source | entries |
|---|---|
| mmlu | 14,042 |
| hellaswag | 10,042 |
| gsm8k | 8,792 |
| arc (Challenge + Easy) | 3,548 |
| winogrande | 1,267 |
| truthfulqa | 817 |
| mbpp + mbpp_full + mbpp_sanitized | 1,631 |
| humaneval | 164 |

Vault sha256 `a02530625c52fb0639931f0fd922ecf8…`. The manifest pins it, so a gate can
prove *which* vault it was checked against.

### 2. `tools/data/decontam_text.py` — prose-capable decontamination

```powershell
python tools/data/decontam_text.py --self-test
python tools/data/decontam_text.py --input <corpus.jsonl> --output <clean.jsonl> \
    --quarantine <bad.jsonl> --report <gate.json>
```

Five layers, schema-agnostic (handles `text`, `messages`, `prompt`/`completion`,
`instruction`, `code`):

| Layer | Method | Fires on |
|---|---|---|
| 1 | exact | sha256 of normalized text vs vault prompt+reference |
| 2 | **13-gram** | any shared word 13-gram → CONTAMINATED (the primary gate, GPT-3-style) |
| 3 | 8-gram | WARN band only (higher recall, noisier) |
| 4 | MinHash ≥ 0.7 | paraphrase of a solution |
| 5 | prompt-leak | vault `prompt_text` appearing in the corpus |

Long rows are handled: the 13-gram window slides over the whole document, so a 400 KB
Gutenberg chunk containing a benchmark question verbatim is still caught.

### 3. `tools/data/gate.py` — the chokepoint

```powershell
python tools/data/gate.py assemble --corpus X --input A --output B \
    --decontam-report R --eval <eval.jsonl> [--dedup] --self-test-pass 5/5
python tools/data/gate.py check --data B
```

Library form, for every training entrypoint:

```python
from data.gate import guard
guard(data_path)   # raises GateError unless a sibling *.gate.json says CLEAN
```

`require_clean()` demands all of: a gate file exists · `verdict == CLEAN` · the required
fields are present · `decontam_self_test` is `n/n` (the gate must prove it can **reject**).

`tools/tm/train_arc.py` now calls it **before importing torch**, so the wrapper is not the
chokepoint — the trainer is. Proven:

```
[arc] TRAINING REFUSED
NO GATE FILE for B:\datasets\local\model-zoo-v6-sft-all.jsonl
  expected: B:\datasets\local\model-zoo-v6-sft-all.gate.json
```

### 4. Local dedup, free, no external service

`gate.py --dedup` uses `datasketch` MinHashLSH (128 perm, char 5-shingles, threshold 0.8 —
internal dedupe is 0.8, contamination 0.7, deliberately different), plus exact sha256.
19,515 rows in 789 s on CPU.

For **semantic** dedupe we already have a free local stack that needs no service call
budget: `text-embeddings-inference` on `localhost:8082` (BAAI/bge-small-en-v1.5, 384-d,
live) + `qdrant` on `:6333`. Not needed for the spine; available when paraphrase-level
dedupe across corpora matters.

---

## The spine, fixed

1. Backed up the surviving 18,447-row file to `B:\datasets\forge\backup\spine-20261020\`
   (necessary — `pipeline.py balance` opens its outputs with `"w"`).
2. **Regenerated the true pre-clean corpus exactly**:
   ```powershell
   python subprojects\dataset_pipeline\pipeline.py balance --target-rows 20000 --eval-frac 0.02
   ```
   → 19,592 train + 408 eval. `spine_eval.jsonl` came back **byte-identical**
   (`4B3A1737F141BE17`), which proves the regeneration is faithful and not a guess.
3. Ran the real gate.
4. Assembled the gate contract.

| | |
|---|---|
| pre-clean | 19,592 |
| benchmark-contaminated | **77 (0.393%)** — gsm8k 72, truthfulqa 5 |
| duplicates | 210 (126 exact + 84 MinHash ≥0.8) |
| train/eval text overlap | 2 → 0 |
| **final** | **19,303 rows, verdict CLEAN** |

The 77 contaminated rows were **verbatim GSM8K word problems with worked solutions**,
12–139 shared 13-grams each, sitting in the `reasoning_decision_systems` bucket. Every
reasoning number we ever reported from the spine was inflated by them.

### Quarantined

`B:\datasets\forge\quarantine\legacy-spine-20261020\` — `spine_train_clean.jsonl`,
`spine_train_contaminated.jsonl` (0 bytes), `simple_decontam.py`, `decontam_report.json`,
plus `WHY-QUARANTINED.md`.

---

## The lesson worth keeping

The self-test caught a bug in **my own** gate on the first run. The rolling hash subtracted
the outgoing term *after* multiplying instead of before, so the accumulator carried the
entire prefix history rather than just the sliding window. Verbatim copies still matched —
so a naive test would have passed. Only the **20-word mid-document slice** fixture exposed
it:

```
[FAIL] gsm8k_20word_slice  expected=CONTAMINATED  got=CLEAN
```

After the fix: `SELF TEST: ALL PASS` (5/5).

> A gate you have not tried to break is a gate you have not tested.

---

## Files

| Path | Role |
|---|---|
| `tools/data/refs_vault.py` | build/verify the frozen reference vault |
| `tools/data/decontam_text.py` | 13-gram + MinHash decontamination for prose and code |
| `tools/data/gate.py` | gate contract, enforcement, local dedup, eval-overlap guard |
| `tools/tm/train_arc.py` | now calls `guard()` before importing torch |
| `B:\datasets\forge\refs\` | vault + manifest |
| `B:\datasets\forge\cleaned\` | gated corpora |
| `B:\datasets\forge\gates\` | raw decontamination reports |
| `B:\datasets\forge\quarantine\` | evidence of what was wrong |
| `B:\datasets\forge\downloads\` | benchmark downloads (HF cache lives here too) |
| `B:\datasets\spine\README.md` | the corrected true state of the spine dir |


---

# Part 2 — G-LIC, G-PII, and the archive ledger (same session)

## G-LIC — `tools/data/license_gate.py`

`sources_ext.py` records a `license` per source and nothing read it. Meanwhile
EmpatheticDialogues (**CC-BY-NC**), Amod (**RAIL-D**, the LICENSE file is sitting in
`B:\datasets\raw`), and EI-50K (custom **helpingai**) rows already flow into
`B:\datasets\optimized\optimized_training.jsonl`.

Resolution order: overrides -> `sources.py`/`sources_ext.py` -> local LICENSE/README files ->
HuggingFace API -> `unknown`.

Commands: `resolve` (build the registry), `tag` (stamp every row), `stats`.

Registry: `B:\datasets\forge\refs\license-registry.json`. Overrides live in
`tools/data/license_overrides.json` and require an `evidence` field.

**Spine result (19,303 rows, coverage 1.0):**

| commercial | rows | licenses |
|---|---|---|
| yes | 9,591 | public-domain-US, CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, Apache-2.0, MIT |
| restricted | 354 | RAIL-D 52, gated 302 |
| unknown | 9,358 | `other` (SEP) 7,576, `unknown` 1,608, `NONE` 174 |

Three findings worth keeping:

1. **HF returns 401/403 for gated repos.** That is a licence signal, not a lookup failure —
   recorded as `license: gated` → `commercial: restricted` (302 rows).
2. **Three sources have no licence card at all on HF** (`samhog/psychology-10k`,
   `aizenSosuke/hope_therapy_conversation_transcripts`, `michelleyunun/therapydata`).
   Tagged `unknown`, counted, never hidden.
3. **1,608 rows carry `source: "cleaned"`** — a generic bucket label the EQ scripts wrote.
   No upstream repo id survives. That is a provenance failure, recorded as such.

## G-PII — `tools/data/pii_gate.py`

No presidio (not installed). A dependency-free scanner is better here anyway: reproducible,
auditable, no model download.

Detects: Anthropic/OpenAI/HF/GitHub/GitLab/AWS/Google/Slack/SendGrid/Stripe keys, JWTs, PEM
private keys, `key=value` secrets, URL credentials, emails, phones, SSNs, IPv4, IBANs,
cards (Luhn), Turkish TCKN (checksum). Modes: `redact` (default), `flag`, `drop`.

**Spine result:** 75 rows touched — 114 phones, 52 emails, 20 IPs, 3 cards. **0 secrets.**

### The false-positive war (three rounds, all found empirically)

**Round 1 — 3 "secrets" found, all placeholders:**
```
client_id='your_client_id'          -> generic_secret
string apiKey = "your_api_key"      -> generic_secret
--aws-access-key AKIAIOSFODNN7EXAMPLE -> aws_key_id  (the canonical AWS docs example)
```
Added `is_placeholder()`: placeholder tokens, known example values, test card numbers, an
entropy floor. A gate that cries wolf gets ignored — and then misses a real key.

**Round 2 — over-suppression.** The entropy floor is correct for credentials and **wrong for
PII**: a phone number is inherently low-entropy, so the floor silently stopped flagging *every
phone number in a therapy corpus*. Made suppression severity-aware: secrets get the floor,
PII does not.

**Round 3 — the card regex was a paper shredder.** `(?:\d[ \-]?){13,16}` permits a space
between *every* digit, so `Berleant [1985 … 2013]` parsed as a Luhn-valid 16-digit card.
Tightened to bare 13-19 digits or properly grouped `4-4-4-4` / `4-6-5`.

Self-test: 21 fixtures, including 6 false-positive controls that exist **because** the gate
got them wrong on real data.

## Archive ledger — `tools/data/archive_ledger.py`

Two phases on purpose: `scan` measures and changes nothing; `apply` moves and writes
`B:\datasets\_archive\ARCHIVE-LEDGER.json` (sha256 tree fingerprint + rows + claim + the
verification result). `restore <id>` puts it back; `list` shows everything.

### Two bugs and one refuted claim

**Bug 1 — my verifier.** `_row_is_empty()` ended with `return not d`, so a dict full of empty
strings scored as *non-empty*. 2M empty rows passed verification as clean data.

**Bug 2 — the claim was still wrong.** After fixing bug 1, `fenrir_v2_1` measured
**199,740 rows, 199,740 distinct**, 50% empty / 50% real — **0.68 GB of genuine cybersecurity
data** that the prior audit had labelled "6 x 99,870 empty rows". We were one `apply` away
from archiving it. Empty-shard verification now requires *empty AND degenerate*, so a
half-empty-but-real directory can never be swept.

**Also caught:** the first strict-name pass for superseded bitqwen versions tried to include
`cosmopedia.jsonl`, `book_rag_chunks.jsonl` and `docker_recovered.jsonl` — unrelated datasets
that merely live in the same folder. A 35 GB name-matched sweep is how you delete something
load-bearing. Restricted the pattern and set `auto: false`; that decision belongs to zcode.

### Archived (3.32 GB, all reversible)

| item | verification |
|---|---|
| `manusagents/.../fable5_2m` | 200,000/200,000 rows empty, **1 distinct row** |
| `manusagents/.../deepseek_v4_pro_agent` | 96,597/96,597 empty, **1 distinct row** |
| `manusagents/.../royal_ghost_1m` | 63 distinct / 200,000 |
| `manusagents/.../autonomy` | 19 distinct / 10,000 |
| `distilling/.venv-n8n` + `.py-n8n` | venvs inside a data dir |
| `classifier/*/model.safetensors` x4 | byte-identical duplicates, keeper retained |

Round-trip tested: `restore degenerate/autonomy` → present → re-archive → absent.

## Lessons

1. **A gate you have not tried to break is a gate you have not tested.** My own gate had a
   broken rolling hash on the first run. Only the deliberately adversarial fixture found it.
2. **A gate that cries wolf is worse than no gate** — it trains humans to ignore it. Six of
   21 PII fixtures exist solely to keep it quiet on placeholders.
3. **Suppression must be severity-aware.** The correct filter for a credential is the wrong
   filter for a phone number.
4. **Verify the claim, not the report.** Two of the three "empty" claims were wrong; one would
   have destroyed 0.68 GB of real data.
5. **Ownership is a technical constraint.** zcode's 29 GB version stack is reported with exact
   numbers and `auto: false` rather than swept, because a name match is not evidence.
