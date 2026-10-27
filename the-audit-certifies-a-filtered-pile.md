# The audit certifies a filtered pile, not a model

**Date:** 2026-10-27
**New:** `logos/audit.py` (12/12 self-tests), `B:/datasets/forge/verifier-av/audit-sheet.dev.jsonl`
**One finding worth keeping:** an audit is a statement about the population you *audit*, and auditing an
unfiltered model measures the rows you were going to throw away.

---

## 1. The two ways precision audits are usually botched

**Not stratifying.** Precision is not uniform. Measured on 500 AVeriTeC dev rows by confidence band:

| band | confidence | rows | true precision |
|---|---|---|---|
| 0 | [0.32, 0.83] | 125 | **0.504** |
| 1 | [0.83, 0.86] | 125 | 0.840 |
| 2 | [0.86, 0.87] | 125 | 0.928 |
| 3 | [0.87, 0.87] | 125 | **0.960** |

**The bottom quartile of the model is at chance (50.4%).** A uniform sample measures 80.8% and tells you
nothing about the top band. Stratifying is not a refinement, it is the difference between measuring a
model and measuring an average.

**Stratifying and then reporting the raw audit rate.** If you over-sample the bad band because that is
where the defects are, the raw defect rate of the sample is *not* the population defect rate. Each stratum
must be weighted by its **share of the population**, not its share of the audit. A test pins this: the
re-weighted estimate recovers 0.914 while the naive audit average gives 0.730 — **wrong by 0.184**.

## 2. Validity, done conservatively

Per-stratum bounds are exact Clopper–Pearson. To combine them with joint confidence `1-alpha`, each
stratum is bounded at `alpha/H` (Bonferroni over `H` strata) and the weighted sum is used. A weighted sum
of individual lower bounds is conservative, so the reported number is never optimistic. That is the right
direction: an understated precision costs samples, an overstated one costs the model.

**AMBIGUOUS counts as NOT correct.** A row a human cannot adjudicate is not evidence of quality, and
treating it as correct would inflate precision using the rows we understand least.

## 3. The finding: the same 190/200 gives three different answers

Identical audit data, three targets, because a **bound is not a point estimate**:

| target | verdict | why |
|---|---|---|
| **90%** | **CERTIFIED** | LB95 = 0.9167 clears it |
| 94% | CONTINUE | between the LB and the point estimate; 1,325 more audits would settle it |
| **95%** | **FUTILE** | the point estimate *is* the target, so the bound can never reach it |

Futility matters as much as certification. If the observed precision is at or below the target, **no
sample size can certify it** and the filter must change — the audit cannot fix a filter. Reporting that
saves the entire remaining audit budget.

## 4. And on real data: the audit correctly refuses

Auditing the **whole** dev set at any budget:

```
per stratum  audits   point    LB95   verdict @0.90
         10      40  0.7750  0.4543        futile
         50     200  0.8300  0.7051        futile
        124     496  0.8065  0.7262        futile
```

Every budget is futile, and **that is the correct answer**: the population is 80.8% precise, so nothing
sampled from it can certify 90%.

Now audit the **keep pile** — the top band only, 125 rows, 5 defective:

```
point precision 0.9600   LB95 0.9177   ->  CERTIFIED at a 90% target
target 93% -> continue (needs 96 more; the pile has 125)
target 95% -> continue (needs 1,223 more; the pile has 125)
```

> **The audit did not fail. It refused to certify a population that is 80.8% precise, and certified one
> that is 96.0% precise. An audit certifies the FILTERED pile — so filtering is what makes a precision
> claim possible at all, and auditing the unfiltered model measures the rows you were going to discard.**

That reframes the whole pipeline: **filter first, then audit what you kept.** Precision is a property of
the keep pile, and the keep pile is a design decision made before any auditing happens.

## 5. The hard ceiling on this corpus

The population is 500 rows. Certifying 99% at zero defects needs 299 audits; 99.9% needs 2,995 — **six
times the entire dev set**. And the ceiling is structural:

**THE AUDIT CANNOT BE LARGER THAN THE KEEP PILE.**

So on AVeriTeC this module can only demonstrate the design. It is built for the code refinery, where the
keep pile is measured in millions. This is the same conclusion the scarcity arithmetic reached from the
other direction, arrived at independently.

## 6. Honest limits

- **The labels used in the run are the dataset's GOLD labels, not a human adjudicating our filter's
  errors.** The run demonstrates the harness and the arithmetic; it certifies nothing. Certification
  requires a human on the sheet.
- **`AMBIGUOUS` was never exercised** — no human has labelled anything yet, so the band that matters most
  for the reviewer's "genuinely ambiguous" pile is untested.
- **Bonferroni is conservative**, so the reported LB is slightly pessimistic. With 4 strata that is a small
  price; with many it would be worth a sharper simultaneous bound.
- **Band edges come from quantiles of the sample**, so an audit designed on dev cannot be reused unchanged
  on a different confidence distribution.

## 7. Next

1. **Label the sheet.** 100 rows at `audit-sheet.dev.jsonl` — the only way this stops being arithmetic.
2. **Move the harness to the code refinery**, where the keep pile is large enough for certification.
3. Adopt the principle in `triage.py`: **audit is the last stage, applied to KEEP, never to the raw stream.**
