# model-zoo

**What this repo is:** the verification program — a reasoning/fact-checking SDK (`logos`), the
verifier training scripts, the dataset gates, and the experiment results behind them.

**Start here if you want the code:** [`logos/`](logos/README.md) — 29 modules, one command to verify:

```bash
python logos/run_all.py --fast
```

---

## Where the code lives

| directory | files | what it is | how to check it |
|---|---|---|---|
| **[`logos/`](logos/README.md)** | 36 py | **the reasoning + fact-checking SDK** — logic, logic verifiers, PGM, decision theory, conformal, provenance, source routing | `python logos/run_all.py` (29/29 pass) |
| `logos/logic/` | 10 py | **the logic verifier**: syllogistic validity (24/24 over 256 forms), 35 inference + 16 replacement rules, 8 fallacies with countermodels, Lean 4 bridge (51/51 lemmas proved, sorry-free) | `python logos/logic/syllogism.py --self-test` |
| `verify/` | 22 py | verifier training + evaluation: math, Wikidata, NLI, static tier, calibration, AVeriTeC | `python verify/math_verify.py --self-test` |
| `train/` | 22 files | teaching/eval runs: gold teaching, served eval, live crosscheck, redteam harnesses | — |
| `data/` | 11 files | dataset gates: decontamination, licence, PII, reference vault | `python data/gate.py` |
| `assembler/` | 7 files | token / trace assembler + its tests | `pytest assembler/tests` |
| `frontend-wasm/` | 6 files | Go→WASM demo (`asm.wasm`) for in-browser assembly | open `index.html` |
| `n8n/` | 1 json | zoo-builder pipeline plan | — |
| `datasets/` | 1 jsonl | golden set | — |

## Where the findings live

Root-level markdown is the lab notebook, in rough chronological order. The load-bearing ones:

**Measured limits — read these before trusting anything**
- `arbiter-and-what-ontologies-are-worth.md` — **the current architecture.** A learned,
  capability-gated arbiter over fixed-order / classifier / ontology / router, with four outcomes:
  keep, escalate, **unknowable → remove or rewrite**. First run asserted 15 dates of which 7 did not
  settle the claim; the relevance gate took it to **13/13**. Also the answer on free ontologies
- `two-memory-ablation-memo-wins-bandit-does-not.md` — ablated the two memories against a
  *fixed-order memo* baseline (not the parallel quorum): the memo alone halves requests (68 → 31)
  and loses nothing, the bandit alone costs **more** requests for no gain, and **naive bandit+memo
  scores 12/24 while no memory scores 14/24**. A cache in front of a learner removes the feedback
  the learner needs
- `redundancy-3x3-and-its-ceiling.md` — 3³ access paths built; **most facts have exactly one
  independent witness**, and that ceiling is a property of the web, not of the code
- `egress-two-paths-and-the-squid-correction.md` — the proxy is **not** a universal fallback: it
  breaks wayback and swap.stanford while recovering archive.is. Also a correction: the Squid
  container was running for two days while I reported it as unconfirmed
- `agentic-router-and-the-reward-that-decides.md` — the bandit learns which sources are worth
  asking, and **the reward decides what it learns**: an availability reward scored 96% coverage of
  which 12 of 23 dates were unusable
- `label-errors-and-the-independence-trap.md` — why the 0.714 agreement figure is not a ceiling
- `constructed-span-labels-work.md` / `per-evidence-architecture-test.md` — constructed labels beat
  derived ones; derived span labels have zero skill over the prior

**Provenance and time**
- `real-facts-provenance-ontology.md`, `provenance-gap-closed-to-35pct.md`,
  `sources-registry-197-and-reproducibility.md` (now 217 sources / 42 categories)

**Logic and proof**
- `phase-a-propositional-badframes-lean.md`, `three-experiments-slot-aggregate-lean.md`,
  `frames-engine-verified.md`

**Planning and budget**
- `TODO-verification-program.md`, `capability-map-and-engine-assignment.md`,
  `BUDGET-REQUEST-cascade-and-verifier.md`, `POOL-RESERVATION.md`,
  `absorb-later-catalogue-2026-10-25.md` (tools considered, with a trigger for each)

---

## The invariant this repo is organised around

> Every layer declares **what it proves and what it cannot**, machine-readably.

[`logos/honesty.py`](logos/honesty.py) is that declaration, and its self-test fails if a layer
claims to be built while its module is missing from disk. This exists because a builder once emitted
a trace claiming *"the syllogistic form is: 'P or Q. Not P. Therefore, Q.'"* — derived from the
substring `or` inside the word `corrupt`.

**What no part of this repo will claim:**

- that a valid syllogism is a true one (validity is form; truth is the world)
- that a hypernym chain proves a proposition (it licenses a substitution)
- that an LLM judge is ground truth
- that ConceptNet weight is a probability
- that a single archive capture is redundancy (three routes into one crawl are one witness)
