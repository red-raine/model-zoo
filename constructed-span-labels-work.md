# Constructed span labels work: +0.333 over majority, where derived labels gave 0.000

**Date:** 2026-10-23
**Modules:** `logos/per_evidence_synth.py` (11/11), `logos/logic/quantity.py` (13/13, +context slot)
**Data:** 60 constructed rows at `B:\datasets\forge\per-evidence-synth\per-evidence-synth.jsonl`

---

## The turn

Three experiments had established that span labels **derived from AVeriTeC's `answer_type`**
have zero skill over the class prior: support never survived measurement, so the aggregator
could only emit `Refuted` or `NEI`, and a one-class predictor cannot beat a 61%-Refuted majority.

The MiniCheck insight is that you should not *judge* a label you can **construct**. So this round
built the corpus instead: pick a fact, build the evidence from it, and the label follows from the
construction.

## Five evidence kinds, each demanding something different

For a fact `(entity, attribute, value, unit)`:

| kind | construction | label | what it tests |
|---|---|---|---|
| **K1** easy_support | one sentence states the value | SUPPORT | the trivial case |
| **K2** split_support | fact split across two sentences (survey + number) | **each alone INSUFFICIENT**, pair SUPPORT | composition — a presence detector cannot pass |
| **K3** refute | different value, same attribute | REFUTE | value comparison, not presence |
| **K4** wrong_attribute | right entity, different attribute | INSUFFICIENT | the R5/R6 failure mode |
| **K5** wrong_entity | right value, different entity | INSUFFICIENT | binding the value to its subject |

The kinds are labelled because **per-kind accuracy is the diagnostic** and the aggregate number is
not. A model that scores well on K1 and badly on K4/K5 has learned nothing about verification.

## The measurement

No training. Two reference "models" and the majority baseline:

| model | accuracy | margin over majority |
|---|---|---|
| majority (`Not Enough Evidence`, 30/60) | 0.500 | — |
| **presence detector** (claim's number appears in evidence) | **0.483** | **−0.017** |
| **slot-aligned compare** (`quantity.compare` per sentence) | **0.833** | **+0.333** |

Per-kind, which is the part that matters:

| kind | presence detector | slot-aligned |
|---|---|---|
| K1 easy support | 1.00 | 1.00 |
| K2 split support | 0.50 | **0.50** |
| K3 refute | 0.90 | 1.00 |
| **K4 wrong attribute** | **0.00** | **1.00** |
| **K5 wrong entity** | **0.00** | **1.00** |

**The presence detector scores 0.00 on both trap kinds** — the corpus is diagnostic, exactly as
designed. It is also *below the majority baseline* overall, which is the cleanest way to show the
corpus cannot be passed by presence matching.

**K2 at 0.50 for both is correct behaviour, not a failure.** The pair alone must be NEI and the
pair-plus-review must be Supported; neither reference model composes across sentences, so it gets
one of the two variants wrong. That gap is precisely what a learned span model would be for — and
it is now a measured, bounded target rather than a hope.

## Why this fixes the one-class collapse

| corpus | class balance | majority | best rule margin |
|---|---|---|---|
| AVeriTeC derived labels | **61% Refuted** | 0.61 | **+0.000** |
| constructed corpus | Supported 20 / NEI 30 / Refuted 10 | 0.50 | **+0.333** |

The derived corpus is 61% one class, so a constant predictor wins and any refute-only rule layer
ties it exactly. The constructed corpus has **no class above 60%**, so a constant predictor
*fails* — and that structural property, not a better model, is what makes learning meaningful.

## Two real bugs caught building it

1. **`_value_phrase` dropped the unit.** "330 m" rendered as "330", the quantity layer could not
   assign a family, and every K1/K3 row came back INSUFFICIENT. The unit is not decoration — it is
   what makes a number comparable at all.
2. **The noun anchor was positionally fragile.** In "the length of the bridge **is** 330" the
   nearest noun is `bridge`; in "the bridge **has** a length of 330" it is `length` — the same
   statement scored a slot mismatch. Replaced with **Jaccard overlap over the sentence's content
   words**, which is stable under that reordering while still separating "the bridge" from
   "the tunnel" (overlap 1/3, rejected).

## Honest limitations

- **The fact table is illustrative, not a knowledge base.** It exists to exercise the reasoning
  pattern. Training a verifier for real claims needs facts from a checkable source with provenance,
  and the temporal gate must run.
- **60 rows is a design specimen, not a dataset.** Enough to prove the construction is diagnostic;
  not enough to train on. Scaling it means sourcing facts, not writing more templates.
- **K2 composition remains unsolved.** It is now a measured gap (0.50) with a defined target.

## Files

| file | self-test | what |
|---|---|---|
| `logos/per_evidence_synth.py` | 11/11 | five-kind constructor, labels verified independently by `quantity.compare` |
| `logos/logic/quantity.py` | 13/13 | + context-overlap slot alignment (replaces the fragile noun anchor) |

## Next

1. **Scale the construction** by sourcing facts from Wikidata with provenance, keeping the five
   kinds and the balance constraint.
2. **Close K2** — a composition step (two-sentence entailment) is now the only kind the reference
   checker fails, and that is the natural place for the first learned component.
3. **Fit aggregate weights to measured per-rule consistency** instead of the hand-set
   `strong_weight=0.55`, since the earlier experiment showed the threshold decided the answer.
