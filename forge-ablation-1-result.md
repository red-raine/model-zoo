# FORGE ablation #1 — the data was fine, the metric is the problem

**Date:** 2026-10-20/21 · **Owner:** opencode · **Verdict: NULL RESULT (and that is the finding)**

---

## The question

v6-sft, the corpus behind TRACE 0.253, turned out to be **68% exact duplicates**:
2,682 rows → **858 unique**. The obvious next move was to add real data. So:

> *Can we get another 2,000 rows of good data, and does it help?*

**Yes to the first. No, measurably, to the second.**

---

## The harvest — 2,281 unique rows from our own assets, $0

`B:\datasets\distilling\raw` held ~6 GB of agent rollouts that had never been mined.
Two are verifier-tagged. Each row is a *snapshot at assistant_step k*; grouping by
`source_trajectory_id` and keeping the deepest step yields a clean pair:

```
prompt     = the user's real engineering request
completion = the final assistant report ("Fixed - all 32 regression checks pass...")
```

| corpus | rows | unique prompts | unique answers | tasks |
|---|---|---|---|---|
| `greghavens_gpt56sol_partial` | 11,410 | 899 | 3,955 | 918 |
| `greghavens_fable-5-coding-and-debugging` | 12,490 | 3,062 | 4,845 | 2,380 |

`tools/data/harvest_traces.py` → **3,298 trajectories → 2,323 unique prompts**, then
dedup → **2,281 rows**, CLEAN through the full gate:

```
harvested    2,323 -> 2,281   CLEAN   decontam 0 · dedup 42 · PII 21 rows · secrets 0
harvested2k  1,013            CLEAN   (total <= 2000 chars, for 512-token training)
```

Diversity: 12 languages (en 640, python 247, non-code 240, go 209, ts 202, powershell 120,
java 92, csharp 86, bash 72, rust 67, c 57), 12 categories, 572 rows with an explicit
verifier (`acceptance-tests+quality-review` etc.).

**Zero downloads. Zero generation. Zero dollars.**

---

## The ablation that killed the hypothesis

Pre-registered: 2 arms × 3 seeds (42/1337/2026), same compute (600 steps), same seq (512),
same frozen eval (48 probes), `lr 2e-4`, LFM2.5-350M + LoRA r16, on sol's Arc Pro B50.

| arm | unique rows | TRACE mean | std | seeds |
|---|---|---|---|---|
| `v6-sft` | 858 | **0.1142** | 0.0100 | 3 |
| `harvested2k` | 1,013 | **0.1118** | 0.0052 | 3 |

```
GAP = +0.0065     larger seed-std = 0.0100     ratio = 0.65x
VERDICT: NOT SEPARATED
```

Per-run, the spread is the story:

```
v6-sft-s2026    0.1254      <- would have been reported as "the result"
v6-sft-s1337    0.1111
v6-sft-s42      0.1062
harvested2k-s42     0.1167
harvested2k-s2026   0.1125
harvested2k-s1337   0.1063
```

**Seed variance (0.106–0.125, a 19 pp band) is larger than the arm difference.**
Picking the best run — PewDiePie's exact mistake — would have "proven" that v6 beats
harvested by 0.019, which is noise. The single-run number we reported last week
(0.253) sits inside a distribution we had never measured.

---

## Why: the metric, not the data

Two measurements reframe everything.

**1. The eval set changed the scale.** 6 generic science probes gave **0.253**; the
frozen 48-probe set gives **~0.11**. Same class of model, same pipeline. The earlier
number was not comparable to anything.

**2. TRACE rewards Toulmin *form*, not answer quality.** A hand-written, deliberately
well-structured answer:

> *"The sky is blue because Rayleigh scattering favours short wavelengths. Evidence: the
> 1/lambda^4 dependence. Warrant: atmospheric molecules are much smaller than the
> wavelength. Backing: measured spectra. Qualifier: unless there is dust. Rebuttal: at
> sunset the path lengthens so red dominates."*

scores **0.281** — with per-dimension `claims 2.0, evidence 4.1, warrant 2.8, backing 1.0,
qualifier 3.0, rebuttal 2.5`. A human-written textbook answer gets a **grade D**.

For reference, `gpt-oss-20b-Q4` on the same 48 probes: **0.1324** (n=14 of 48 — its
content was short/empty because reasoning tokens consumed the budget, so treat the
number as indicative only). Our 350M students at 0.11 are not far from it.

**Therefore the 0.55 "gate" is not a quality bar. It is a formatting bar that no model
in this house reaches, including a 20B.** We were never failing a reasoning gate; we
were failing a "write in Toulmin structure" gate, and we never measured what a good
answer scores before declaring 0.55 the target.

---

## Conclusions

1. **Data is not the bottleneck at this scale.** +18% unique rows (858 → 1,013) moved
   TRACE by +0.0065, inside the noise. Stop adding rows at 350M / 600 steps.
2. **The eval is the bottleneck.** Fix the metric before buying more compute. The
   honest protocol now exists (`tools/tm/eval_frozen.py`: frozen set, mean ± std,
   bootstrap CI, separation test) — but the metric it wraps is miscalibrated.
3. **Everything is now gated and reproducible.** Three corpora with `gate.json`,
   40,303-entry reference vault, 3-seed matrix, 7 GGUFs, results under
   `B:\datasets\forge\eval\`.

## What to do next (in order)

1. **Calibrate TRACE against a proper reference** — score 5 tiers on the frozen set
   (base 350M untrained, our student, LFM2.5-1.2B, gpt-oss-20b at full budget, a
   frontier model), and set the gate at a measured percentile rather than a guess.
2. **Add a task-grounded metric** for the domains we actually train: code correctness
   (execute the answer), and a judge-based usefulness score. TRACE stays as a
   structural diagnostic, not as *the* gate.
3. **Then** re-open the data question — with a metric that can see a difference, and
   at a capacity where one exists (1.2B, or 350M with 4× the steps).
4. `harvested` (2,281 rows) needs 2048 seq, which costs 5.1 steps/min (2 h/run).
   Worth it only once the metric can discriminate.

## Artifacts

| path | what |
|---|---|
| `B:\datasets\forge\cleaned\harvested.jsonl` | 2,281 unique agent-task rows, CLEAN |
| `B:\datasets\forge\cleaned\harvested2k.jsonl` | 1,013 rows, 512-token slice, CLEAN |
| `B:\datasets\forge\cleaned\v6-sft.jsonl` | 858 unique (was 2,682 with 68% dupes) |
| `B:\datasets\forge\refs\eval-frozen-20261021.jsonl` | 48 frozen probes, sha `6D652A4B...` |
| `B:\datasets\forge\eval\*.json` | 7 arm×seed results + calibration |
| `B:\datasets\forge\gguf\` | 7 merged f16 GGUFs (all distinct sha256) |
| `tools/data/harvest_traces.py` | the miner |
| `tools/tm/eval_frozen.py` | frozen-set eval with variance + separation test |
| `tools/tm/run_ablation.sh`, `matrix.sh` | the training cells |
| `tools/tm/build_gguf.py`, `serve_eval_matrix.py` | merge/convert/serve/eval pipeline |

## The lesson

We set a target of 0.55, trained toward it for weeks, and never checked what a good
answer scores. **A gate you have not calibrated is as dangerous as a gate you have not
tested** — and this one would have told us a human-written textbook answer was a D.
