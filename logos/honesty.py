"""honesty.py - the capability manifest.

A consumer must not have to guess what a layer proves. This module returns, for
every layer, what it proves, what it cannot prove, how, and what it costs.

Exists because the dataset-builders engine emitted a trace claiming
"The syllogistic form is: 'P or Q. Not P. Therefore, Q.'" derived from the
substring "or" inside the word "corrupt". Over-trusting a layer is the failure
mode this file prevents.
"""

from __future__ import annotations

CAPABILITIES = {
    "logic.syllogism": {
        "proves": "validity of a categorical syllogism's FORM",
        "cannot": ["truth of the premises", "non-categorical argument forms",
                   "quantifier scope beyond AEIO", "modal or probabilistic reasoning"],
        "method": "six classical rules; verified to reproduce all 24 valid forms "
                  "across all 256 mood/figure combinations",
        "cost": "0 tokens, deterministic",
        "module": "logos.logic.syllogism",
    },
    "logic.types": {
        "proves": "a selectional-restriction violation (VerbNet)",
        "cannot": ["truth", "that a violation means false rather than figurative"],
        "method": "VerbNet role restrictions (Agent[+animate], Patient[+concrete])",
        "cost": "0 tokens, deterministic",
        "module": "logos.logic.types",
        "status": "not built",
    },
    "logic.contradiction": {
        "proves": "that two IsA assertions conflict across KBs",
        "cannot": ["which KB is right", "truth"],
        "method": "hypernym-chain intersection across WordNet and ConceptNet",
        "cost": "0 tokens, deterministic",
        "module": "logos.logic.contradiction",
        "status": "not built",
    },
    "evidence.math": {
        "proves": "arithmetic, percentages, grouped expressions",
        "cannot": ["word-problem modelling", "unit semantics", "anything requiring "
                   "choosing the right operation"],
        "method": "sympy on a strict character whitelist",
        "cost": "0 tokens, deterministic",
        "module": "tools.verify.math_verify",
    },
    "evidence.wikidata": {
        "proves": "entity containment (P17/P131/P30/P361) and capital-of (P36)",
        "cannot": ["contested facts", "temporal facts", "anything outside those relations"],
        "method": "Wikidata SPARQL with a language-filtered label service",
        "cost": "0 tokens, network",
        "module": "tools.verify.fact_verify",
    },
    "evidence.search_llm": {
        "proves": "NOTHING on its own",
        "cannot": ["be treated as ground truth"],
        "method": "SearXNG snippets judged pairwise by a local LLM",
        "cost": "1+ LLM calls",
        "module": "tools.verify.lora_check",
        "measured": "coverage 0.00 on HaluEval n=10; retrieval is the blocker",
    },
}


def capabilities(layer: str | None = None) -> dict:
    if layer:
        return CAPABILITIES.get(layer, {"error": f"unknown layer {layer!r}"})
    return CAPABILITIES


def can_prove(layer: str, claim_kind: str) -> bool:
    cap = CAPABILITIES.get(layer) or {}
    if cap.get("status") == "not built":
        return False
    return claim_kind not in (cap.get("cannot") or [])
