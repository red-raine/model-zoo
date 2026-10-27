# The ruler, the held-out set, and where Z3 actually fits

**Date:** 2026-10-27
**Answers two questions:** how big must the eval set be, and must the ruler itself be better than 90%?
**Plus:** Z3 5.1.0 is installed, and here is what it does and does not add.

---

## 1. "Is 3,000 audits our held-out eval set?"

**No — it is a third thing, and conflating the three is how the project loses the ability to prove
anything.** They answer different questions and must not be the same rows:

| set | purpose | state |
|---|---|---|
| **DEV / TUNING** | fit `T = 4.226`, `tau`, weight modes, review band, strata edges | we have (AVeriTeC dev, 500) |
| **HELD-OUT TEST** | the only source of an *unbiased* accuracy number | **we do not have one** |
| **AUDIT SHEET** | human labels on the KEEP pile — the only source of a *certified* precision | built, drawn from dev |

### The disclosure that matters more than any number

**AVeriTeC dev is set 1, and the audit sheet was drawn from it.** `T` was fitted by cross-fitting inside
those rows, the probability-routing table was chosen by looking at them, and the confidence bands are
their own quantiles. Cross-fitting protects the **scalar**; it does not undo **model selection** made on
the same rows.

So every number from that sheet is a **development estimate**. It demonstrates the machinery. It does not
establish that the system is 96% precise.

And `calib` cannot rescue this: it is 92.6% accurate where dev is 80.8%, so it was seen by the model.

**Cheapest fix:** reserve a slice now and stop looking at it. **Correct fix:** obtain the AVeriTeC
test-split labels. Until one of those happens, no precision claim is certified.

## 2. "Do we need 3k, and must the ruler be better than 90%?"

**Both intuitions are right about the mechanism and need one correction in magnitude.**

### Audit size depends entirely on the target

| target | zero defects | 0.1% bad | 1% bad | **ruler must be** |
|---|---|---|---|---|
| 90.0% | **29** | 29 | 29 | 95.00% |
| 95.0% | 59 | 59 | 93 | 97.50% |
| 98.0% | 149 | 149 | 590 | 99.00% |
| 99.0% | 299 | 299 | **unachievable** | 99.50% |
| 99.5% | 598 | 947 | unachievable | 99.75% |
| 99.9% | **2,995** | **unachievable** | unachievable | 99.95% |

**3,000 audits is the budget for 99.9%, not for 90%.** A 90% target needs ~29 audits at zero defects. The
two differ by 100×, and neither is the right number until you say which target and which defect rate.

Note the diagonal: **at 0.1% defects, 99.9% is unachievable** — you cannot certify a target equal to your
own defect rate, for any sample size. That is not a budget problem, it is arithmetic.

### The ruler ceiling — the intuition made exact

Two error sources, and only one shrinks with sample size:

1. **Sampling error** — the audit samples the keep pile. Handled by Clopper–Pearson, shrinks with `n`.
2. **Label error** — the human labelling the sheet is sometimes wrong. **Does not shrink with `n`.** It is
   a bias, and no audit budget fixes it.

With symmetric label noise `eps`:

```
p_measured = eps + (1 - 2*eps) * p_true
p_true     = (p_measured - eps) / (1 - 2*eps)
```

so measured precision **saturates**, and the highest true precision distinguishable from 1.0 is:

```
ruler_limited_precision(eps) = 1 - eps
```

| labeller accuracy | highest certifiable precision |
|---|---|
| 99.99% | 99.99% |
| 99.00% | 99.00% |
| 95.00% | 95.00% |
| 90.00% | 90.00% |

**So yes: a 90%-accurate labeller cannot certify anything above 90%.** That is the precise form of the
intuition. It is not "we can never exceed the test set's accuracy" in general — it is that the ruler's
error rate is a hard cap on the target you can certify, and it must be **estimated**, not assumed.

Required ruler accuracy to certify `p`: roughly `eps <= (1-p)/2`. To certify 90% you need a labeller
better than 95%; to certify 99.9% you need better than 99.95%.

**How to turn `eps` from an assumption into a measurement:** double-label a random subset and report the
disagreement rate. That is the single cheapest upgrade to the whole audit, and it is now a prerequisite
rather than a nicety.

## 3. Formal logic is solved; the interface is not — and I largely agree

The reviewer is right and the distinction is worth keeping clean:

- **Solved since the 1970s–90s:** validity, consistency, entailment, contradiction, rule application —
  first-order logic, resolution, Prolog, description logics, rule engines.
- **Our own `logic/syllogism.py` already does this**: it reproduces all 24 valid forms over all 256
  mood/figure combinations with zero mismatches, plus 35 inference and 16 replacement rules truth-table
  verified, plus 51/51 lemmas machine-proved in real Lean 4. **No Z3 needed, no LLM needed.**
- **The unsolved part is exactly what they say**: turning messy natural language into the symbolic
  representation reliably. That is where NLP and LLMs earn their keep, and it is where the extraction
  layer's measured failures already are (derived span labels have zero skill over the prior).

So: **do not rebuild what is solved.** The symbolic layer's job is to be a *target* for extraction and a
*checker* for what was extracted.

## 4. What Z3 5.1.0 actually adds (it is installed; kanren/pydatalog/hy are not)

| available | role |
|---|---|
| **z3 5.1.0** | SMT: satisfiability, contradiction, numeric+logical constraints combined |
| sympy 1.14.0 | arithmetic already in use (`logic/quantity`) |
| networkx 3.6.1 | graph reachability over extracted triples |
| ast, symtable | the code refinery's structural + semantic layers |

**Z3 is genuinely additive for three things the syllogism engine cannot do:**

1. **Consistency of a set of extracted triples** — `KB` unsatisfiable means the sample contradicts itself,
   with no truth required. That is a Semantic-router check at zero model cost.
2. **Entailment as unsat** — `KB ∧ ¬claim` unsatisfiable. Covers far more than categorical syllogisms.
3. **Mixed constraints** — dates, quantities and logic in one solver. Our `quantity.py` compares numbers;
   Z3 can decide whether a *set* of numeric claims is simultaneously satisfiable, which is exactly the
   "internally consistent?" question the Semantic router needs.

**Where Z3 is NOT the answer:** categorical syllogism validity (already solved in 6 rules), and anything
requiring the NL→symbolic step, which remains the bottleneck. Putting Z3 in does not solve extraction.

**Proposed scope for the first Z3 module:** a `logic/smt.py` that takes a small set of triples plus
optional numeric constraints and answers CONSISTENT / INCONSISTENT / UNKNOWN. That plugs directly into
`triage.py`'s semantic router as a real deterministic check, and it is testable against hand-built
inconsistent sets before any extraction is involved.

## 5. Honest limits

- **`eps` is currently assumed, not measured.** Every ruler number above is conditional on a noise model
  nobody has estimated. Double-labelling is the prerequisite.
- **The noise model is symmetric.** Real annotators are biased (they miss subtle errors more than
  obvious ones), which makes measured precision optimistic in a way symmetric `eps` understates.
- **`ruler_limited_precision = 1 - eps` is the practical ceiling, not a theorem.** The formal statement
  needs a confidence level and a sample design; the version here is the operational bound.
- **Z3 has not been used yet.** Its fit is argued from the interface, not measured.
