# Objective correction: this is ore refining, not theorem proving

**Date:** 2026-10-27
**Trigger:** external review flagging that the work had drifted from the goal
**Verdict:** the review's core correction is right, one of its targets is unmeasurable, and one of its
proposed stages does not do what it claims.

---

## 1. The drift, accepted

The work drifted to: **"is this confidence value a calibrated probability?"**

The goal is: **"cheaply extract the highest-quality reasoning traces from billions of noisy tokens."**

Those are different optimisation targets. Calibration is an *instrument*, and instruments matter — but I
made it the headline. The honest read of the last several turns is that I built a very careful measuring
device and largely lost track of what it was for.

**What is kept rather than abandoned:** calibration is what lets us *certify* precision. In a
precision-first pipeline, "we measured 96.5%" and "we can prove 93.5%" are different claims, and only the
second survives an audit. So the instrument stays; it stops being the subject.

## 2. Asymmetric costs, adopted — and this changes the optimum

A production QA system wants balanced coverage. A refiner wants precision and does not care about recall:

| error | cost | why |
|---|---|---|
| keeping a bad sample (FP) | **catastrophic** | poisons training, invisible later |
| dropping a good sample (FN) | ~free | there are millions more |

Chow's rule gives `t* = 1 - C_d/C_w`. The earlier default (1/4 → **0.75**) was a balanced-classifier
setting. A refinery sets `C_d/C_w → 0` so **t\* → 1**. Implemented in `logos/refinery.py` with
KEEP / REVIEW / DROP, the ternary the review asked for — which `arbiter.py` already had as
KEEP / ESCALATE / UNKNOWABLE.

## 3. The blind spot: 99.9% precision is unmeasurable at this scale

The review says *"99.9% precision at 10% recall is probably far more valuable than chasing balanced
coverage."* It never checks whether the claim can be **certified**. It cannot, and the arithmetic is
short:

With zero defects observed, certifying precision ≥ p at 95% confidence needs `n ≥ ln(0.05)/ln(p)`:

```
certify 99.0%  @95%, zero defects :     299 audits
certify 99.5%  @95%, zero defects :     598 audits
certify 99.9%  @95%, zero defects :   2,995 audits
certify 99.99% @95%, zero defects :  29,956 audits
```

And if the pipeline is not defect-free it gets worse — or impossible:

```
certify 99.9% if 0.01% are bad :  2,995 audits
certify 99.9% if 0.10% are bad :  UNACHIEVABLE AT ANY n
certify 99.9% if 1.00% are bad :  UNACHIEVABLE AT ANY n
```

**You cannot certify a target equal to your own defect rate.** At 0.1% defects the expected precision *is*
99.9%, so the lower confidence bound sits below it forever. Headroom is mandatory.

**Consequence for the plan: the refinery's first milestone is keep-pile VOLUME, not 99.9% precision.**
An audit set must be larger than the keep pile before a precision claim means anything.

## 4. Measured on real data: the ceiling is 96.5%, and 93.5% is what can be certified

Ternary routing over 500 AVeriTeC dev rows at refinery costs:

| C_d/C_w | t\* | KEEP | precision on KEEP |
|---|---|---|---|
| 0.25 | 0.75 | 435 | 86.7% |
| 0.20 | 0.80 | 404 | 89.1% |
| 0.15 | 0.85 | 270 | 92.6% |
| 0.13 | 0.87 | 159 | **96.2%** |
| 0.12 | 0.88 | **0** | — |

Best certifiable operating point, scanning coverage:

| confidence | keep frac | n | precision | **LB95** |
|---|---|---|---|---|
| **raw** | 40% | 200 | 96.50% | **93.53%** |
| calibrated | 40% | 200 | 95.50% | 92.28% |

**99.9% is unreachable (ceiling 96.5%) and unverifiable (needs 2,995 audits against a 200-row keep
pile).** So the target in the review is not a stretch goal, it is an unmeasurable one on this data.

## 5. A conflict my own calibration work created, now measured

```
max confidence, RAW logits          : 1.0000
max confidence, CALIBRATED (T=4.23) : 0.8743
```

Temperature scaling took ECE from 0.098 → 0.031 and **compressed the tail**. Any refinery threshold
above 0.87 keeps zero rows. So:

> **ECE-optimal calibration is not selectivity-optimal calibration.**

This is the third independent confirmation that `ranking → raw` in the probability-routing table is
correct: AURC said it (0.0743 raw vs 0.0797 calibrated), the routing table encoded it, and now the
refinery measures it as certifiable precision (93.53% vs 92.28%). A tail-preserving calibration would be
the right tool for *both* jobs and is not what temperature scaling is.

## 6. Where the review is wrong, and this matters

**6a. FrameNet / VerbNet / WordNet cannot supply "semantic consistency."** These are lexical resources.
`honesty.py` has said since the beginning that they supply *premises and type constraints*, never
validity and never truth. Using them as a Stage-2 validator asks them to do something they cannot do.
Extracting frames from arbitrary CoT text additionally needs a trained frame-semantic parser, whose
accuracy on mixed code/prose is unknown — that is a research project presented as a pipeline stage.

**6b. The review is internally inconsistent.** It says most CoT data is *"planning, coding,
classification, explanation"* and then proposes state-transition consistency (`GIVE`/`own`), which covers
state transitions. Plans and explanations do not have clean state transitions. The proposed semantic
layer covers a minority of the corpus it is aimed at.

**6c. "Stack-scale" conflates two corpora.** Stack v2 is code — compilers, tests, AST repair all work.
Stack Exchange is prose — none of it works. **Adopting "ore refining" requires choosing a corpus**, and
that choice is the real decision the review skips. Our corpus today is AVeriTeC + Wikidata facts, which
is neither.

**6d. "The LLM has wandered into calibration" is half right.** The 0.224 macro-F1 recovery from
commensurable confidences is load-bearing, and it is exactly the kind of finding that stops us reporting
96.5% when we can only prove 93.5%. Right that it became the headline; wrong that it was a detour.

## 7. What is genuinely worth building next, in order

1. **Choose the corpus.** Code (compiler available) or prose (no compiler). Everything downstream
   depends on it, and the deterministic validator stack is completely different.
2. **Build the precision audit harness before the pipeline.** A stratified sampler over the keep pile, a
   hand-labelling format, and a running Clopper-Pearson bound. Without it, precision claims are rhetoric.
3. **Deterministic validators for code** — compile, execute, test-pass, type consistency, variable
   consistency. This is the cheapest precision available anywhere and needs no model.
4. **AST repair loop** — parse → fail → repair candidate → reparse → re-execute → accept/reject. The
   review's best concrete idea and cheaper than any LLM call.
5. **Only then** the semantic layer, scoped to whatever the corpus actually contains — and scoped
   honestly, with a declared capability for what it can and cannot judge (`honesty.py` style).
6. **LLM last**, on the REVIEW band only, which the measurements size at 1–5%.

## 8. What does not change

- `honesty.py`'s rule: **validity ≠ truth ≠ confidence ≠ calibrated probability.**
- Log-linear pooling, never attention.
- Cross-fitting or it did not happen.
- Report the definition next to the number.
- **The typed-confidence idea from the attached note is right and is the next code change**: make
  `RawScore` and `CalibratedProbability` distinct frozen types so an uncalibrated value cannot be
  constructed into a `Candidate` at all. That is the correct enforcement of a gap I had already named and
  only half-closed with a boolean flag.
