# The arbiter, and what ontologies are actually worth to us

**Date:** 2026-10-26
**New modules:** `logos/arbiter.py` (16/16), `logos/cascade.py`
**Correction accepted:** a decision model does **not** replace the router. It arbitrates over it.

---

## 1. The design error, stated plainly

I wrote that a decision model might *replace* the router. That is wrong in a way that matters. A
router is one opinion among several, and replacing opinions loses information. What was missing is a
layer that takes **the fixed order, the classifier, the ontology check, the router's pick and the
archive routes — all of them** — scores which is most likely right, and is permitted to answer:

> *"all of these look suspect — go search, or ask an LLM"*

and also permitted to answer:

> *"the truth cannot be known here. Remove this fact or rewrite it."*

Those four outcomes — **keep, escalate, unknowable, and the fact is not worth keeping** — are the
whole point. Abstaining is a decision about cost; unknowability is a property of the claim.

## 2. Three structural rules, each of which fixes a measured failure

**Rule 1 — a capability gate, not a vote.** Each method declares which claim kinds it can address,
reusing `honesty.py`. A method that cannot address this claim kind gets **weight 0**, which in
log-linear pooling is a multiplicative identity: the *absence* of an opinion, not a neutral vote.
Test: five incapable methods at 0.99 confidence **do not** outvote one capable method at 0.60.
A plain majority vote gets this wrong, and getting it wrong is indistinguishable from sabotage.

**Rule 2 — log-linear pooling, never attention.** Already established in this project: a softmax
gate over the same evidence scored 0.900 and collapsed to 0.633, because it violates coordinate-wise
monotonicity. Test: raising support for a verdict from 0.6 to 0.9 raises its fused mass 0.551 →
0.750, never lowers it.

**Rule 3 — weights learned, not set.** `aggregate` was decided by a hand-set `strong_weight=0.55`,
which is how a pipeline quietly becomes a hard-coded opinion. Every `(method, claim_kind)` pair now
carries a Beta posterior over "is this method right about this kind of claim", starting at 0.50 and
moving only from outcomes. Capability is per claim kind, so a method good at arithmetic is not
credited for dates (measured: 0.95 on `numeric`, 0.05 on `date_precedence`, same method).

## 3. The bug the cascade exposed on its first run — and it is the important one

First working version, 24 real facts:

```
keep=15   escalate=8   unknowable=1   ... but only 8 of the 15 kept dates settled the claim
```

**Fifteen confident assertions, seven of them wrong.** Why: all the archive routes returned the same
*post-claim* capture date, the methods agreed, and the fusion read unanimity as certainty. They were
all answering a different question — "when was this page captured?" — from the one asked — "did this
exist before 2022-01-01?"

So `Candidate` gained a field distinct from capability:

| field | question | failure mode |
|---|---|---|
| `claims` | *can* this method judge this kind of claim? | capability failure |
| `answers` | does this method's answer *address* the claim posed? | **relevance failure** |

A fusion that cannot tell "we all agree" from "we all answered the wrong question" will assert the
wrong question's answer with high confidence. That is the most dangerous failure a verifier can have,
because it looks exactly like success.

After the relevance gate, same 24 facts, same fixture:

```
keep=13   escalate=2   unknowable=9   ... and 13 of 13 kept dates settle the claim
```

**Precision on asserted dates: 8/15 (53%) → 13/13 (100%)**, paid for by 9 facts being flagged
instead of guessed. Those 9 are exactly the "remove from the set or rewrite" population.

## 4. What this does not prove

The learning pass uses the temporal gate as its label, which is **circular**: the arbiter learns
*which route produces gate-passing dates* — useful for route selection — and that is **not** learning
truth. Both the cold and learned passes are printed together, and neither may be quoted alone.

The learned numbers are nonetheless informative:

```
wayback_cdx         0.607   n=138
arquivo_pt_cdx      0.219   n=30
http_last_modified  0.050   n=18
```

`http_last_modified` collapsing to 0.05 is the server-render-date problem, learned automatically
rather than by my writing it down.

## 5. Ontologies: what is actually free, and what is actually useful

Researched each repo's own README and probed the endpoints. Verdicts, not a link list.

### Free ontologies — yes, and the good ones are the boring ones

| what | free? | note |
|---|---|---|
| **schema.org** | **yes, verified 200** | This *is* "Google's ontology" — schema.org is Google + Microsoft + Yahoo + Yandex, CC BY-SA. The single most useful vocabulary for describing sources and pages |
| **Wikidata** | **yes, CC0, verified** | already our fact backbone |
| **OpenAlex** | **yes, CC0, verified 200** | ~250M scholarly works. The successor to Microsoft Academic Graph |
| Google Knowledge Graph API | **no, paid** | Enterprise Knowledge Graph was discontinued |
| Microsoft Academic Graph | **retired 2022** | do not plan around it; OpenAlex replaced it |
| OBO Foundry / BFO | yes, free | biomedical + upper ontology; well maintained |
| Microsoft Ontology-Playground | free **tool**, not an ontology | a static web editor for *learning* about ontologies and exporting RDF/XML |

So the honest answer to "can we use Google's or Microsoft's ontologies for free": **yes, but not the
ones with those names on the API pricing page.** schema.org, Wikidata and OpenAlex are free and are
the ones worth using. Their *proprietary* graphs are not.

### The repos

| repo | what it actually is | verdict |
|---|---|---|
| **monarch-initiative/ontogpt** | LLM extraction with **ontology-based grounding** | **USE.** Its stated purpose is precisely our failure: *"Ask an LLM for an ontology identifier and it will often invent one, or attach a real identifier to the wrong thing."* That is our derived-label problem, and it is a solved-elsewhere problem |
| **aws/context-ontology-accelerator** | Scan → Model → Serve: induce ontologies, build a semantic graph, **determine correct actions** | **COPY THE PATTERN, NOT THE STACK** (needs Neptune + Bedrock). "Determine correct actions" is the arbiter's job, and their three-stage naming is the shape our pipeline already has |
| **open-metadata/OpenMetadata** | data catalogue: lineage, glossaries, classifications, data contracts, ownership | **LATER, as governance.** Not a reasoner. When the corpus is large enough that "which of 217 sources fed this fact" stops fitting in my head, this is the tool |
| **opensemanticsearch/open-semantic-search** | Solr + ETL + NER + OCR + knowledge-graph search, self-hosted | **CANDIDATE for corpus search**, competes with our orchestrator. Heavy. Not now |
| **microsoft/Ontology-Playground** | static teaching tool, exports RDF/XML | **NOT OURS.** We do not need a visual editor. Only the lesson stands: ship ontologies in standard serialisations (owl/ttl/nt/jsonld), which we already took from `ontime-ontology` |

### Do ontologies help us build datasets? Yes, in three specific ways

1. **A closed label space.** Labels cannot drift if they must resolve to an ontology ID. Our
   `ontology.py` currently validates against hand-written schemas; grounding to real IDs closes the
   remaining hole.
2. **Structural negatives for free.** A type constraint makes some labels *impossible*, which yields
   cheap, non-judged negative examples — exactly the "constructed over judged labels" discipline.
3. **A consistency gate that does not need truth.** Ontology violation is checkable without knowing
   whether the claim is true. That is a third independent axis next to our temporal gate and
   corroboration.

### And can it make thinkingcap-distill a subject expert?

Yes, and the mechanism is narrower than it sounds. Three layers:

- **Ontology as the schema of the subject.** The domain's entities, relations and admissible values
  become the generation grammar. `ontology.py` already renders schemas into prompts; the upgrade is
  grounding those schemas in real IDs so a hallucinated entity is *rejected*, not merely unlikely.
- **Ontology as the verifier.** A distilled model's output is scored by whether it instantiates the
  schema. This is the deterministic channel we already have for syllogisms, extended to a domain.
- **Ontology as the curriculum.** Grounded extractions give (text → typed triple) pairs at scale,
  which is a training set whose labels cost nothing to check. That is how a small model becomes
  narrow and confident instead of broad and vague.

**For actions specifically:** the unit is `(state, action, ontology-typed effect)`. The ontology types
the *effect*, and a typed effect is checkable — which is what makes an action label verifiable rather
than narrated. Without the type, "the cup moved right" is prose; with it, it is a constraint that can
fail.

## 6. Next, in order

1. **Wire the arbiter into `dating.py`** so real harvests get dispositions, and route `escalate`
   cases to the search orchestrator (`:8022`) — that is what "go search" means concretely.
2. **Make `unknowable` actionable**: emit a rewrite queue. A fact nobody can settle becomes a
   narrower claim that somebody can.
3. **Add OntoGPT-style grounding to the extractor** so ontology IDs are verified, not invented.
4. **Ablate the arbiter against the best single method** — the discipline that just demoted the
   router. Until the arbiter beats "always trust wayback_cdx" on gate-passing dates at equal cost,
   it is unproven.
5. **Add a second capable method per claim kind.** Right now `date_precedence` is judged by archive
   routes only, which all fail together when a page was never archived. A real second opinion needs
   a different evidence type — that is what the ontology tier and the classifier tier are for.

## 7. Honest limits

- **n = 24**, one claim kind. The 100% precision is 13 of 13 on facts chosen because they had
  references; it is not a general estimate.
- **The learning is circular** (gate as its own label) and is labelled as such wherever it appears.
- **`unknowable` is currently a verdict with no consumer.** Nine facts were flagged and nothing
  happened to them yet. Until the rewrite queue exists, that is a label, not a fix.
- **No second evidence type yet**, so the arbiter is arbitrating agreement among siblings rather than
  genuinely independent opinions.
