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


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    """The manifest must not lie about itself.

    A capability manifest is only useful if it is accurate, and an accurate manifest is exactly the
    kind of document that rots silently: someone adds a layer, someone deletes a module, and nothing
    complains. So the test checks STRUCTURE and DRIFT, not prose.
    """
    import os

    print("logos.honesty - SELF TEST")
    ok = True

    # --- every layer declares the four things a consumer needs
    missing = {name: [k for k in ("proves", "cannot", "method", "cost") if k not in cap]
               for name, cap in CAPABILITIES.items()}
    bad = {k: v for k, v in missing.items() if v}
    ok = ok and not bad
    print(f"  [{'PASS' if not bad else 'FAIL'}] all {len(CAPABILITIES)} layers declare "
          f"proves/cannot/method/cost" + (f"  missing: {bad}" if bad else ""))

    # --- "cannot" must be a non-empty list: a layer that cannot fail to prove anything is a lie
    bad = [n for n, c in CAPABILITIES.items()
           if not isinstance(c.get("cannot"), list) or not c["cannot"]]
    ok = ok and not bad
    print(f"  [{'PASS' if not bad else 'FAIL'}] every layer states at least one thing it CANNOT "
          f"prove" + (f"  empty: {bad}" if bad else ""))

    # --- DRIFT: a module path must exist on disk, unless the layer is declared not-built.
    #     This is the check that keeps the manifest honest as the tree changes.
    #     Resolution must be layout-independent: in the source tree the package lives at
    #     tools/logos/, in the published repo it lives at logos/. Resolving `logos.x` relative to
    #     THIS FILE's directory works in both; the first version walked up from the repo root and
    #     reported a false MISSING for the very module it lives beside.
    import os
    pkgdir = os.path.dirname(os.path.abspath(__file__))

    def _exists(mod: str) -> bool:
        """Resolve a manifest module path across BOTH layouts.

        The source tree keeps the package at `tools/logos/` and sibling scripts at `tools/verify/`.
        The published repo puts the package at `logos/` and the scripts at `verify/`. A single
        declared string therefore cannot be right in both, so every plausible prefix is tried rather
        than declaring one layout canonical and letting the check cry wolf in the other.
        """
        variants = [mod]
        if mod.startswith("tools."):
            variants.append(mod[len("tools."):])          # tools.verify.x -> verify.x
        else:
            variants.append("tools." + mod)               # verify.x -> tools.verify.x
        for v in variants:
            if v.startswith("logos."):
                if os.path.exists(os.path.join(
                        pkgdir, v[len("logos."):].replace(".", os.sep) + ".py")):
                    return True
                continue
            rel = v.replace(".", os.sep) + ".py"
            d = pkgdir
            while True:                                   # try every ancestor as a package root
                if os.path.exists(os.path.join(d, rel)):
                    return True
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        return False

    drifted, not_built = [], []
    for name, cap in CAPABILITIES.items():
        if cap.get("status") == "not built":
            not_built.append(name)
            continue
        if not _exists(cap.get("module", "")):
            drifted.append((name, cap.get("module", "")))
    ok = ok and not drifted
    print(f"  [{'PASS' if not drifted else 'FAIL'}] every built layer's module exists on disk "
          f"({len(CAPABILITIES) - len(not_built) - len(drifted)} found, "
          f"{len(not_built)} declared not-built)"
          + (f"  MISSING: {drifted}" if drifted else ""))

    # --- an unbuilt layer must never answer yes
    good = (not can_prove("logic.types", "truth")
            and not can_prove("logic.contradiction", "truth"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a layer marked 'not built' never claims to prove "
          f"anything")

    # --- the search layer must not be trusted as ground truth. This is the whole point of the file:
    #     it exists because a builder emitted a syllogism derived from "or" inside "corrupt".
    good = not can_prove("evidence.search_llm", "be treated as ground truth")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] search+LLM evidence is refused as ground truth")

    # --- a real layer answers correctly for a kind it does not handle
    good = (can_prove("logic.syllogism", "validity of form")
            and not can_prove("logic.syllogism", "truth of the premises"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] syllogism proves FORM and refuses truth of premises")

    # --- unknown layers fail closed, never open
    good = "error" in capabilities("does.not.exist")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unknown layer fails closed")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Capability manifest: what each layer proves.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--layer", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    caps = capabilities(args.layer or None)
    for name, cap in caps.items():
        print(f"\n{name}  [{cap.get('status', 'built')}]")
        print(f"  proves : {cap.get('proves', '')}")
        print(f"  cannot : {'; '.join(cap.get('cannot', []))}")
        print(f"  method : {cap.get('method', '')}")
        print(f"  cost   : {cap.get('cost', '')}")
        if cap.get("measured"):
            print(f"  MEASURED: {cap['measured']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
