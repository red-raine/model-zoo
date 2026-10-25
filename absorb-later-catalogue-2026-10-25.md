# Absorb-later catalogue — every tool sent on 2026-10-25, with a verdict and a trigger

Not a link dump. Each entry says what it actually is (read from its own README), what it would do
for us, and **the condition that should make us pick it up**. Entries marked *NOT OURS* are here so
nobody re-researches them.

---

## USE NOW — verified keyless, fits a module that already exists

### `lc/gau` — the four-index idea (pattern, not code)
Fetches known URLs from **AlienVault OTX, Wayback, Common Crawl, URLScan**. We had one of four.
I probed the other three; **all keyless and answering**:

```
archive.org/wayback/available  200  246 B   closest snapshot in ONE call (we used CDX only)
urlscan.io/api/v1/search       200  5.8 KB  a genuinely independent URL index, not archive.org
otx.alienvault.com/api/v1/...  200         keyless host intel (empty for a wiki page: security-oriented)
```

**URLScan is the find** — archive redundancy was always capped by archive.org's coverage, and a
second *crawler* samples a different population of URLs rather than a second window on the same one.
→ *Add `wayback_available` as a tier-0 probe and `urlscan` as a fourth route in `redundancy.py`.*

### `xnl-h4ck3r/waymore` — the incompleteness principle
Two things worth stealing. It **downloads archived responses** (not just URLs), and its stated
differentiator is that other tools *"do not currently deal with the rate limiting now in place by
the sources, and will often just stop with incomplete results and not let you know they are
incomplete."*

That last clause is a design rule we should adopt everywhere: **never return a partial result
silently.** It is the same principle as `strength` / `conflicted` in `redundancy.py`, and waymore
arrived at it from bug-bounty practice rather than from statistics.
→ *Every collector reports `complete: bool` and what it could not reach.*

### `adrida/tracer` — TRACER: this is our two-tier cascade, already built by someone else
> *"Distill your LLM traces into fast ML System 1 models… ML students return labels locally for
> repetitive requests. **Calibrated deferral and an out-of-distribution (OOD) guard keep the teacher
> in the loop** for requests the ML path declines. No LLM fine-tuning. No GPU required."*

That is the model2vec-static → DeBERTa cascade plus Chow's rule, described independently. **The
architecture decision is corroborated by an outside implementation.** Two things to borrow:
- **calibrated deferral** as the routing criterion, rather than a hand-set threshold
- an **OOD guard** — we have no protection against a claim unlike anything in training, and TRACER
  treats that as a first-class decline reason

→ *Use as the reference design when the static tier is wired; take the OOD guard now.*

### `Memento-Teams/Memento` — memory-based continual learning without touching weights
> *"Fine-tuning LLM Agents **without fine-tuning LLMs** — a memory-based, continual-learning
> framework… Planner–Executor Architecture • **Case-Based Reasoning** • MCP Tooling •
> Memory-Augmented Learning."*

This is exactly what `router.py` + `Memo` does, and it is published with **continual-learning curves
on GAIA**. Two things we do not have:
- **case-based reasoning over stored cases** — we cache key→value; they retrieve *similar* past
  cases and reuse the reasoning, which generalises where a hash cache cannot
- **a continual-learning curve as the evaluation protocol** — we should report accuracy vs number
  of tasks seen, not one final number

→ *Adopt the CL-curve evaluation now (free). Consider CBR retrieval when the memo stops hitting.*

### `lightpanda-io/browser` — for the search stack, and it is real
> *"The headless browser built from scratch for AI agents… written in Zig. **16x lighter and 9x
> faster than Chromium.** Memory 123 MB vs 2 GB for 100 pages; 5 s vs 46 s."*

CDP-compatible, so it can replace Chromium behind Playwright for most fetches. Directly answers
"try lightpanda, see if it makes it in the search stack": **yes, as the default fetch engine for the
JS-heavy minority**, keeping Chromium for the pages that need full fidelity. Note for Windows: it
targets Linux/macOS first, so the practical path is the Docker image or WSL — which is fine, since
the search stack is containerised anyway.

→ *Hand to the search-stack agent as: "add Lightpanda as a Playwright CDP endpoint; benchmark
against Chromium on our 15 archive endpoints and on JS-heavy reference pages."*

---

## USE SOON — one decision away

### `NandhaKishorM/laya` — 33 ms typed decisions, trained on proper scoring rules
> *"Multilingual, non-autoregressive System 1 decision engine. Typed decisions over 100+ languages
> in a single forward pass — 33 ms — trained with RL against **strictly proper scoring rules
> (RLCD)**, with a router that picks the right checkpoint per request."*

Proper scoring rules are the same mathematics as our Brier/log-loss work, and 33 ms is faster than
our DeBERTa tier by 3×. The multilingual claim matters because our sources are not English-only
(arquivo.pt is Portuguese, and the archive corpus is full of non-English pages).
→ *Candidate replacement for the DeBERTa tier once we have a labelled verdict set to compare on.*

### `ollaya-dev/ollaya` — the serving layer, and it goes in Docker on nyx
> *"Run open decision models locally, the way Ollama runs LLMs. System 1 for AI agents: fast, typed,
> calibrated decisions… returns calibrated probabilities in a single forward pass. It never
> generates text. Ollaya pulls these models by name and serves them from a local daemon behind a
> self-hosted, **Jev-compatible** API."*

This is the piece that was missing: a local daemon serving decision models by name. It makes
laya/jev-style models a **network service instead of a Python dependency**, which is what we need
for the router to call from any tool.
→ *Deploy on nyx as `ollaya:PORT`, then point the static tier at it. Keyless, self-hosted, no vendor.*

### `shcherbak-ai/contextgem` — provenance per extracted fact, as a library
> *"Describe, don't prompt. **Every extracted fact is traced back to its source.**… decision models
> answer closed questions (yes/no, one of your labels, a level of a scale) with the probability of
> each possible answer… A model group sends each aspect and concept to the model that suits it,
> and any model can fall back."*

"Every extracted fact is traced back to its source" is our per-evidence requirement, implemented.
The model-group-with-fallback is our bandit in a different framing.
→ *Read the source-tracing implementation before writing our own document-level extraction.*

### `arxiv 2610.19376` — DeLoop: the loop problem, formalised
> *"Language-model agents that run long tasks often redo work they have already done: they re-issue
> a tool call they made earlier and get back exactly the same observation, a step that adds no new
> information. We identify this failure as **a loop**… Loops appear natively in Qwen, Llama, and
> Gemini agents and become **far more common once the agent's history is compressed**."*

This is our memo cache problem, named and published. Two consequences:
- **Our memo is the inference-time fix; DeLoop is the training-time fix.** They use the looping
  action as a *negative example* and the first later non-looping action as the positive.
- **Loops get worse after compaction**, which means our long sessions are exactly the failure regime.

→ *Add a loop detector to trace capture: an identical (tool, args) repeated with an identical
observation is a labelled negative. We have the traces; this is a cheap label.*

### `reticlehq/reticle` — three-way verdicts on "the agent said done"
> *"Your AI agent says 'done.' Reticle finds out if that is true. It drives your real, running app
> and reads what happened. Then it returns a verdict, **pass · fail · cannot tell**, with the
> file:line to fix."*

`pass · fail · cannot tell` is our Supported / Refuted / NEI plus abstain, arrived at independently
in a software-QA setting. The useful part is that it **drives the real app** rather than reading the
agent's claim — the same discipline as our "measured, not asserted" rule.
→ *Borrow the verdict shape and the "cannot tell" honesty for pipeline self-checks.*

---

## USE WHEN WE TRAIN A STUDENT — right tool, wrong time

### `GMvandeVen/continual-learning` (Mammoth)
PyTorch implementation of *"Three types of incremental learning"* (Nature Machine Intelligence,
2022). **Honest assessment: we do not need it yet, and our bandit does not need it at all** — a Beta
posterior is monotone and cannot catastrophically forget. Mammoth matters the moment the router is
replaced by a **neural** ranker, because that is when forgetting becomes real. What we should take
now, for free, is the **taxonomy**: task-incremental / domain-incremental / class-incremental. Our
contexts (`date_resolution`, `numeric_check`, …) are **task-incremental**, which is the easiest
case — worth stating plainly in any writeup rather than implying we solved the hard one.

### `yoshitomo-matsubara/torchdistill`
Configuration-driven distillation (YAML, not code), including intermediate-representation
extraction without reimplementing models. The right tool the day we distil a student from traces.
→ *Trigger: first attempt at a trained source-ranker.*

### `arxiv 2610.21601` — Waypoint-OPD
> *"Policy-shift distillation reuses the difference between a teacher's log probabilities before and
> after RL… the same endpoint shift can result from early and late shifts that reinforce **or
> partially cancel** each other. Waypoint-OPD uses an intermediate teacher checkpoint to compare
> early and late shifts."*

Relevant only when we distil from a teacher we can checkpoint mid-training. Also a useful warning:
**an endpoint delta can hide two opposite movements.** That is the same failure as reporting a
single F1 when the confusion matrix moved in two directions.
→ *Trigger: any distillation plan that only looks at start and end weights.*

---

## NOT OURS — recorded so nobody researches them twice

### `mliu98/awesome-human-distillation`
A Chinese list for distilling real people (colleagues, mentors, exes) into Claude Code skills. It is
partly a joke — the README includes an anti-AI-centrism rant and an "anti-distillation" skill. The
**pattern** is real and the user's instinct is right: an opinionated persona traced from real
material, per subject. But this list is not a research artefact and should not be cited as one.
→ *If we want opinionated per-subject traces, build them from our own traces with the DeLoop
labelling, and say so honestly.*

### `AntonioCoppe/jev-harness` + `yibie/awesome-jev`
Jev is TypeSafe's System-One typed-decision model; the harness adds policy, a confidence gate,
shadow mode and an offline eval CLI. **Requires a TypeSafe API key**, which violates the keyless-
first and one-key-per-app rules. The *pattern* (policy → confidence gate → shadow mode → offline
eval) is already what our `decision.py` / `abstain.py` do. → *Skip unless a key is explicitly
approved and budgeted.*

### `r0075h3ll/Oralyzer`
Open-redirect and CRLF fuzzer, with Common Crawl URL harvesting and proxy support. Security tool.
→ *Only the CC harvesting overlaps, and Common Crawl's index cluster is currently 502.*

### `koala73/worldmonitor`
Real-time news/geopolitics dashboard (web, Tauri desktop, terminal). Situational awareness UI, not
a verification component. → *Nothing for us.*

### `richardg867/WaybackProxy`
A retro-friendly HTTP proxy that serves pages from Wayback instead of the origin, stripping
toolbars for old browsers. → *The "transparent archive fallback proxy" idea is interesting for
egress, but this implementation is for vintage browsers. Our `egress.py` covers the need.*

---

## Videos sent (not watchable here)

`youtu.be/K0_Q5Hz-HOg`, `STRtnhlf9CI`, `M1nYgz4TUoE` — I cannot transcribe video. If they matter,
send the titles or the papers they cover and they will get a real entry above.

---

## The Wayback-alternatives list, scored against what we measured

Of the 25: **9 answer us** (archive.today, ghostarchive, conifer, commoncrawl-collinfo, arquivo.pt,
perma.cc, loc, ukwa, swap.stanford), **4 are dead** (memento time travel, bibalex, internet memory
foundation, webcite), and the rest are either proprietary SaaS (Stillio, Pagefreezer, MirrorWeb,
Visualping, Ahrefs, ScreenshotAPI, Capturly) or **software we would have to run ourselves**
(ArchiveBox, Browsertrix, OpenWayback).

Two of those self-hosted ones deserve a real look, because they invert the problem:
**ArchiveBox** and **Browsertrix** let us *become* an archive instead of begging one. That is the
only path to genuine redundancy on obscure facts — if only one crawler ever saw a page, the fix is
to have been the second crawler. It is expensive, and it is the honest answer.

---

## Sequencing

| when | what | why then |
|---|---|---|
| **now** | wayback availability (tier-0), urlscan route, `complete: bool` on collectors | keyless, verified, small |
| **now** | Lightpanda → search-stack agent | unblocks JS-heavy fetch at 1/16 the memory |
| **now** | DeLoop loop-detector on traces | we already have the traces to label |
| **soon** | ollaya on nyx, then point the static tier at it | turns models into a service |
| **soon** | Memento's CL-curve evaluation | changes how we report, costs nothing |
| **when training** | laya, torchdistill, Mammoth, Waypoint-OPD | correct tools, premature now |
| **strategic** | ArchiveBox / Browsertrix self-run | the only real fix for single-witness facts |
