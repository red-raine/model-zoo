# Phase A: propositional logic, bad-thinking frames, and Lean 4 actually running

**Date:** 2026-10-23
**Modules:** `logos/logic/propositional.py` (21/21), `badframes.py` (21/21), `lean_bridge.py` (9/9)
**Full suite:** 13 modules, all self-tests pass.

---

## 1. The Lean finding — antigravity was not using Lean

The operator asked me to double-check whether `E:\vibe_coding\glm\antigravity` is actually
using Lean. **It is not.**

| check | result |
|---|---|
| `lean` / `lake` / `elan` on PATH | **none** |
| `.venv\Scripts\lean*` in the antigravity project | **none** |
| `leanprover/lean4` Docker image present | **no** |
| Docker Hub reachable | **no** (auth.docker.io times out; no HTTPS proxy configured) |

`minion/formal/lean_runner.py` advertises a chain of "local CLI, Docker, or verified parser".
With no CLI and no image, every proof it has reported was checked by its Python parser
(`minion/formal/mini_lean.py`), not by Lean. The parser may well be correct — but a parser is
not a machine check, and "Lean-verified" is a claim about a kernel. Their own phase
`31-live-hardware-truth-pass-lean4-anti-mock` shows someone already suspected this.

### But Lean 4 IS now installed here

The blocker was never Lean, it was Docker Hub. `releases.lean-lang.org` and `github.com` are
reachable, so:

```
elan-init.exe  (2.5 MB from github.com)
  -> elan 4.2.4
  -> toolchains leanprover/lean4:v4.15.0 and v4.35.0
  -> Lean (version 4.35.0, x86_64-w64-windows-gnu, commit eb6cafb7..., Release)
```

Lives at `%USERPROFILE%\.elan\bin`. **Not on the default PATH**, which is exactly why
`shutil.which("lean")` returns None and why `lean_bridge.py` locates it by absolute path
rather than trusting PATH.

### Two independent checks, which is the point

`lean_bridge.check()` establishes validity twice, by implementations sharing no code:

1. **Python truth table** — exhaustive over valuations.
2. **Lean** — either a tactic proof of the *proposition*, or a `rfl` check on a Bool encoding
   where every atom is substituted by a literal.

Measured results:

| argument | Lean method | ms |
|---|---|---|
| modus ponens `P→Q, P ⊢ Q` | `exact h1 h2` | 2059 |
| hypothetical syllogism | `simp_all` | — |
| De Morgan `~(P&Q) ⊢ ~P ∨ ~Q` | `rfl:bool-encoding` | — |
| affirming the consequent | **not proved** ✓ | — |
| denying the antecedent | **not proved** ✓ | — |

`LeanResult.trustworthy` is True only when `method` starts with `tactic`. If Lean vanishes,
the result says `python_only` and must not be described as machine-checked.

**Bug caught by having two paths:** the Bool encoding first declared atoms as
`example (P Q : Bool)` — universally quantified, so `rfl` could not reduce it. The tactic path
had already proved MP and HS, so the bug stayed hidden until De Morgan needed the Bool path.
That is the cascading-envelope argument in one incident: one checker would have shipped a
broken fallback.

---

## 2. Propositional engine

`propositional.py` — 35 rules of inference, 16 rules of replacement, 8 formal fallacies.

**Validity is computed, never asserted.** The self-test runs a truth table over every rule:

- all 35 inference rules verified VALID (no valuation makes premises true and conclusion false)
- all 16 replacement rules verified valid in **both** directions (they are equivalences)
- all 8 fallacies verified INVALID **with a concrete countermodel** — a fallacy detector that
  cannot exhibit a falsifying assignment has no guarantee it is describing logical defect
  rather than stylistic dislike

Analytic results: `affirming_the_consequent` countermodel is exactly `{P: False, Q: True}`;
`affirming_a_disjunct` countermodel is exactly `{P: True, Q: True}`.

Two limitations recorded as `[INFO]` assertions so they cannot be forgotten:

- **Ex falso holds.** A contradiction entails anything. This engine reports *entailment*, not
  *argument quality*; relevance is layered on by `badframes`.
- **Hypothetical syllogism is valid for material implication and fails for counterfactuals.**
  There is no modal semantics here.

---

## 3. Bad-thinking frames

The operator asked for magical thinking, self-harm, destructive thinking, psychosis-adjacent
patterns, and expanded fallacies. The research found the important architectural fact:
**~13 named pathologies collapse into 8 formal defects**, each a broken object in an argument
structure. One detector per defect, with the named pathologies as labels projected onto it.

| layer | broken object | frames projected onto it |
|---|---|---|
| L1 quantifier/scope | domain of ∀ | absolutist overgeneralisation, apophenia |
| L2 conditioning direction | which variable is conditioned on | base-rate neglect, prosecutor's fallacy |
| L3 support edge | evidential relation | **magical thinking**, affect heuristic |
| L4 missing alternative | \|H\| ≥ 2 | confirmation/myside, motivated reasoning, **conspiracy** |
| L5 selection/denominator | sampling frame | survivorship, availability |
| L6 modal calibration | qualifier vs warrant | overclaiming, Dunning–Kruger |
| L7 self-sealing | falsifiability | **psychosis-adjacent**, unfalsifiable |
| L8 local coherence | discourse structure | derailment / thought disorder |

Plus five **LLM-specific** frames, because in a 2026 pipeline the most common bad reasoning is
model failure: negation blindness, reversal curse, sycophancy, overclaim, depth collapse.

### The two highest-value design decisions

**`voiding` findings.** Several defects are *voiding* conditions, not soft scores. An argument
with no stated denominator is not "70% bad", it is **unevaluable**. `verdict_impact()` returns
`action: "unevaluable"` rather than a discount, which is the direct antidote to the measured
LLM failure mode (arXiv:2610.12083: judges agree with ground truth only 51% of the time and
flip verdicts on order swaps).

**Proximate contradiction.** One consistency check catches conspiracy's contradictory-beliefs
trait, motivated reasoning, and plain inconsistency. Two routes: negation flip (`is` / `is not`)
and antonym substitution (`rose` / `fell` via WordNet).

### The one clinically validated marker

`absolutist_density` implements the single linguistic marker with clinical validation
(Al-Mosaiwi & Johnstone 2018, *Clin. Psych. Sci.* 6(4):529). Used here as a **reasoning**
signal — "all/never/nothing" signals a quantifier whose domain was never checked — not as a
mental-health screen. Every function takes a claim and returns a structural finding; nothing
diagnoses a person.

### Test discipline

19 assertions including **negative** ones, because a detector that never says "no" is
useless: a properly quantified sentence must NOT fire; a mechanistic sentence must NOT fire; a
rate WITH a denominator must NOT fire; a meta-analysis with "suggests" must NOT fire; a clean
scientific sentence must yield **zero** findings. Plus one irreducible limitation asserted
openly: irregular verbs (`rose`/`rise`) are NOT unified by suffix stripping.

---

## 4. Files

| file | self-test | what |
|---|---|---|
| `logos/logic/propositional.py` | 21/21 | 35 rules + 16 replacements + 8 fallacies, truth-table verified |
| `logos/logic/badframes.py` | 21/21 | 8 formal defects + 5 LLM frames + 16-frame catalogue + voiding verdicts |
| `logos/logic/lean_bridge.py` | 9/9 | Lean 4 machine check with tactic ladder + Bool-encoding fallback |
| `logos/logic/syllogism.py` | all pass | 24 categorical forms (unchanged) |

---

## 5. What is next (Phase B)

1. **Entailment bridge**: wire `propositional.analyze` + `badframes.scan` into the factor graph
   as potentials, so a proved contradiction can veto a neural verdict.
2. **`lean_bridge` in CI**: every rule in the catalogue also machine-checked by Lean, not only
   by the Python truth table.
3. **Sorites / polysyllogism**: chain arguments are the HMM locator's natural home; use `∏pᵢ`
   not `min pᵢ` for chain confidence.
4. **Toulmin layout** as the parse target: `Q ∉ licensed(W) → overclaim` is already implemented
   as L6 and is the best single symbolic rule found.
5. **Walton scheme + critical-question checklist** — CQ-by-CQ is tractable and the counts
   disagree across sources (96 / 65 / +44 sub-schemes), so cite the specific edition used.

## 6. New dead-end reclassifications

The operator reports **ternary weights, spaCy relation extraction, and MC-dropout** are
working. All three are moved from "dead end" to "working — needs integration": ternary for the
combiner/factor-graph weights, spaCy RE on our 35 relations vs calOn, MC-dropout variance as an
epistemic feature in the factor graph. None of the three is described as dead in this document.
