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


---

# Addendum — composition closes K2, and one label in the corpus was wrong

## The corpus had a label error, and the checker was right

The first version labelled the K2 pair ("the bridge was surveyed in 2019" + "the survey reported
a length of 330 m") as **Not Enough Evidence**, on the theory that composition needed something
extra beyond the pair.

That was wrong. **Two sentences which jointly establish the fact do support it.** The correct
hard negative is the value-bearing sentence **alone** — where the subject is only implied by
"the survey" — not the joined pair. Fixed:

| row | evidence | correct verdict |
|---|---|---|
| K2a | `"The survey reported a length of 330 m."` alone | **NEI** — subject only implied |
| K2b | survey sentence **+** value sentence | **Supported** — jointly established |

The lesson is the same one this project keeps relearning: when a checker disagrees with a label,
check the label first.

## `compose()` — three routes, one refusal

`logos.logic.quantity.compose(claim, evidence)` decides a claim against a **set** of sentences:

1. **single sentence** — any one sentence that compares cleanly wins outright
2. **composition** — no single sentence carries both subject and value, but the union of the
   evidence's content words covers the claim's context while one sentence supplies a matching value
3. **refute** — a single sentence with a comparable quantity that disagrees

**Refutation deliberately does not compose.** Two sentences that each contribute half of a
contradiction are not a contradiction, and the asymmetric aggregator already treats one confident
refute as decisive. Manufacturing conflicts from partial negative evidence is how a verifier
starts inventing them.

## Strict coverage replaced two looser rules, both of which failed

The slot test went through three designs, each broken by a specific case:

| rule | broke on |
|---|---|
| compare the two anchor nouns | `"the length of the bridge **is** 330"` vs `"the bridge **has** a length of 330"` — anchors `bridge` / `length`, same statement, false mismatch |
| Jaccard ≥ 0.5 over content words | `"city has 45 million people"` vs `"city has 45,000,000 residents"` — only one of two content words overlaps |
| anchor match **or** subset | `"the survey reported a length of 330 m"` alone supported a claim about the bridge, which the evidence never mentions |

**Strict coverage** — every content word of the claim must appear on the evidence side, allowing
WordNet synonymy — is the property that actually matters: the evidence must be *about* the
claim's subject, not merely contain one of its words. All 21 tests pass under it.

Documented limitation: `people` / `residents` are not WordNet synonyms, so that pair comes back
`insufficient` rather than `support`. Reported openly rather than papered over with a looser rule.

## The result

| model | accuracy | margin over majority |
|---|---|---|
| majority (`NEI`, 30/60) | 0.500 | — |
| **presence detector** | **0.483** | **−0.017** |
| **slot-aligned + compose** | **1.000** | **+0.500** |

Per-kind:

| kind | presence | slot-aligned + compose |
|---|---|---|
| K1 easy support | 1.00 | **1.00** |
| K2 split support | 0.50 | **1.00** |
| K3 refute | 0.90 | **1.00** |
| K4 wrong attribute | 0.00 | **1.00** |
| K5 wrong entity | 0.00 | **1.00** |

Every kind at 1.00, on a corpus where the presence baseline is *below* the majority and scores
**0.00 on both trap kinds**.

## What this is and is not

**Is:** a deterministic span-level verifier that reaches 1.000 on a constructed, balanced corpus
with five distinct reasoning demands, and whose honest failure mode is `insufficient` rather than
a guess.

**Is not:** evidence that this generalises to real claims. The corpus is 60 template rows over
illustrative facts; the construction is a *design specimen* that proves the architecture and the
label scheme work. Scaling it means sourcing facts with provenance, not writing more templates —
and the temporal gate must run before any of it is used for training.

The realistic next step is unchanged: source facts, keep the five kinds and the balance
constraint, and treat any drop below 1.000 on real facts as the measure of how much the template
phrasing was doing the work.
