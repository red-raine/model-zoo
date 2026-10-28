"""logos/ontology_bundle.py — an OKF-native domain jargon layer that builds itself, with a curation gate.

THE PROBLEM THE USER NAMED, AND IT IS EXACTLY RIGHT
---------------------------------------------------
`context_pack.normalize_terms` lemmatises with WordNet. WordNet is a general-English lexical database: it
knows `bridge` and `mouse`, and it knows nothing about `P585`, `macro-F1`, `Mondrian`, `commensurable` or
`CTU` in the senses this project uses them. Worse than knowing nothing, it will happily return an
*English sense that is wrong for the domain* -- which is the same failure mode as a real identifier attached
to the wrong thing.

**That is not a bug to patch in the lemmatiser. It is the gap an ontology fills**, and it is why the
ontology layer has to be per-domain rather than per-language.

WHY OKF, AND WHAT IT DOES NOT SOLVE
-----------------------------------
OKF gives us the right shape for free: a version-controlled directory tree of markdown files with YAML
frontmatter, an auto-generated `index.md` per directory for progressive disclosure, and a `status` field. Our
own `tools/okf_index.py` already validates and indexes that profile (required keys
`type/title/status`; status in `current|stale|archived|draft`).

**`draft` maps exactly onto "proposed but not yet curated."** That is the anti-poisoning gate expressed in
the format itself rather than bolted on:

    a term met in the corpus  ->  written as `status: draft`   (proposal, NOT served)
    human/curator approves    ->  `status: current`            (served as ground truth)
    definition goes wrong     ->  `status: stale`              (not served; re-curate)

The paper that motivated this (arXiv 2607.02609) is honest about the boundary and so is this module: **a
format standardises representation and interchange; it does not by itself give you provenance, quality or
contradiction management.** Those are ours, and the store's promotion gate is where they live.

HOW THE BUNDLE BUILDS ON ITSELF
-------------------------------
Each row processed proposes the jargon terms it used. Proposals accumulate as drafts; a curator promotes the
ones that matter. Every promotion makes the next corpus pass resolve more terms, so the bundle compounds.
The gate is what stops it compounding *poison*.

USAGE
    python tools/logos/ontology_bundle.py --self-test
    python tools/logos/ontology_bundle.py --measure     # WordNet vs bundle on real project jargon
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.context_pack import normalize_terms  # noqa: E402

DRAFT, CURRENT, STALE, ARCHIVED = "draft", "current", "stale", "archived"
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-") or "term"


@dataclass
class Term:
    term: str
    domain: str
    definition: str = ""
    aliases: tuple[str, ...] = ()
    units: str = ""
    status: str = DRAFT
    curator: str = ""
    provenance: tuple[str, ...] = ()
    notes: str = ""

    def keys(self) -> set:
        out = {self.term.lower()} | {a.lower() for a in self.aliases}
        for k in list(out):
            out |= set(normalize_terms(k))
        return {k for k in out if k}

    def to_okf(self) -> str:
        """Emit an OKF document. `status: draft` is what keeps an uncurated term out of service."""
        aliases = ", ".join(self.aliases)
        prov = "; ".join(self.provenance)
        return (
            "---\n"
            f"type: term\n"
            f"title: {self.term}\n"
            f"status: {self.status}\n"
            f"domain: {self.domain}\n"
            f"units: {self.units}\n"
            f"aliases: {aliases}\n"
            f"curator: {self.curator}\n"
            f"provenance: {prov}\n"
            "---\n\n"
            f"# {self.term}\n\n"
            f"{self.definition}\n"
            + (f"\n## Notes\n\n{self.notes}\n" if self.notes else "")
        )

    @classmethod
    def from_okf(cls, text: str) -> "Term | None":
        m = FRONTMATTER_RE.match(text)
        if not m:
            return None
        meta, body = m.group(1), m.group(2)
        kv = {}
        for line in meta.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        if kv.get("type") != "term":
            return None
        body_lines = [ln for ln in body.strip().splitlines() if not ln.startswith("#")]
        definition = " ".join(body_lines).strip()
        return cls(
            term=kv.get("title", ""), domain=kv.get("domain", ""), definition=definition,
            aliases=tuple(a.strip() for a in kv.get("aliases", "").split(",") if a.strip()),
            units=kv.get("units", ""), status=kv.get("status", DRAFT),
            curator=kv.get("curator", ""),
            provenance=tuple(p.strip() for p in kv.get("provenance", "").split(";") if p.strip()),
        )


class Bundle:
    """A directory tree of OKF term documents, one per domain."""

    def __init__(self, root: str):
        self.root = root
        self.terms: dict[str, Term] = {}
        self.load()

    # ------------------------------------------------------------------ io
    def path_for(self, term: Term) -> str:
        return os.path.join(self.root, _slug(term.domain), _slug(term.term) + ".md")

    def load(self) -> None:
        if not os.path.isdir(self.root):
            return
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for fn in filenames:
                if fn == "index.md" or not fn.endswith(".md"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                        t = Term.from_okf(fh.read())
                except OSError:
                    continue
                if t and t.term:
                    self.terms[f"{t.domain}|{t.term.lower()}"] = t

    def save(self, term: Term) -> str:
        p = self.path_for(term)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(term.to_okf())
        self.terms[f"{term.domain}|{term.term.lower()}"] = term
        return p

    def write_stub_indexes(self) -> list[str]:
        """Per-domain `index.md` with OKF frontmatter, so the bundle is discoverable by progressive
        disclosure the way the profile intends."""
        written = []
        by_domain: dict[str, list[Term]] = {}
        for t in self.terms.values():
            by_domain.setdefault(t.domain, []).append(t)
        for domain, terms in by_domain.items():
            p = os.path.join(self.root, _slug(domain), "index.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            lines = ["---", "type: index", f"title: {domain} terminology",
                     "status: current", f"count: {len(terms)}", "---", "",
                     f"# {domain} terminology", "",
                     "| term | status | definition |", "|---|---|---|"]
            for t in sorted(terms, key=lambda x: x.term.lower()):
                d = t.definition.replace("|", "/")[:110]
                lines.append(f"| {t.term} | {t.status} | {d} |")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            written.append(p)
        return written

    # ------------------------------------------------------------------ service
    def serve(self, query: str, domain: str | None = None) -> dict:
        """Resolve a term for USE. Only `current` terms are served: a draft is a proposal, not ground truth.

        This is the same promotion logic as the truth store, applied to definitions instead of facts -- a
        definition that has not been curated must not silently become the ontology's answer.
        """
        keys = {query.lower()} | set(normalize_terms(query))
        hits = []
        for t in self.terms.values():
            if domain and t.domain != domain:
                continue
            if keys & t.keys():
                hits.append(t)
        if not hits:
            return {"status": "MISS", "term": None, "reason": "no term or alias matches"}
        current = [t for t in hits if t.status == CURRENT]
        if current:
            return {"status": "HIT", "term": current[0], "definition": current[0].definition,
                    "units": current[0].units}
        return {"status": "DRAFT_ONLY", "term": hits[0],
                "reason": f"matched {hits[0].term!r} but its status is {hits[0].status!r}, not "
                          f"'current' -- a proposal is not ground truth"}

    def propose(self, term: str, domain: str, seen_in: str = "", definition: str = "",
                aliases: tuple[str, ...] = ()) -> Term:
        """Record a jargon term the corpus used but the bundle does not serve. This is the compounding step:
        proposals accumulate so a curator has a work queue, and the bundle grows without silently growing
        wrong."""
        existing = self.serve(term, domain)
        if existing["status"] == "HIT":
            return existing["term"]
        t = Term(term=term, domain=domain, definition=definition, aliases=aliases,
                 status=DRAFT, provenance=(seen_in,) if seen_in else ())
        self.save(t)
        return t

    def curate(self, term: str, domain: str, definition: str, curator: str,
               units: str = "", aliases: tuple[str, ...] = ()) -> Term:
        """Promote a proposal to served ground truth. Human act, recorded with attribution."""
        t = self.terms.get(f"{domain}|{term.lower()}")
        if t is None:
            t = Term(term=term, domain=domain)
        t.definition = definition
        t.curator = curator
        t.status = CURRENT
        if units:
            t.units = units
        if aliases:
            t.aliases = tuple(aliases)
        self.save(t)
        return t

    def mark_stale(self, term: str, domain: str) -> bool:
        t = self.terms.get(f"{domain}|{term.lower()}")
        if t is None:
            return False
        t.status = STALE
        self.save(t)
        return True

    # ------------------------------------------------------------------ stats
    def coverage(self, queries: list[tuple[str, str]]) -> dict:
        served = draft = miss = 0
        reasons = []
        for q, dom in queries:
            r = self.serve(q, dom)
            if r["status"] == "HIT":
                served += 1
            elif r["status"] == "DRAFT_ONLY":
                draft += 1
                reasons.append((q, "draft"))
            else:
                miss += 1
                reasons.append((q, "miss"))
        n = max(len(queries), 1)
        return {"n": len(queries), "served": served, "served_rate": served / n,
                "draft_only": draft, "miss": miss, "reasons": reasons[:10]}


# ------------------------------------------------------------------ measurement
# Real jargon from THIS project's own documents. Chosen because we know the correct domain sense, which is
# the only way to say whether a lexicon got it right.
JARGON = [
    ("macro-F1", "metrics"), ("ECE", "metrics"), ("Brier", "metrics"),
    ("Mondrian", "conformal"), ("isotonic", "calibration"), ("Platt", "calibration"),
    ("commensurable", "epistemics"), ("conformal", "conformal"), ("AURC", "metrics"),
    ("NEI", "averttec"), ("CTU", "averttec"), ("ABSTAIN", "epistemics"),
    ("P585", "wikidata"), ("P813", "wikidata"), ("cantilever", "engineering"),
]


def measure(verbose: bool = True) -> int:
    import tempfile
    from logos.context_pack import normalize_terms as lem

    # --- baseline 1: does WordNet know the term AT ALL?
    try:
        from nltk.corpus import wordnet as wn
        has_wn = True
    except Exception:                                        # noqa: BLE001
        has_wn = False

    tmp = tempfile.mkdtemp()
    b = Bundle(os.path.join(tmp, "ontology"))

    wn_known = 0
    for term, _dom in JARGON:
        if has_wn:
            lemmed = (lem(term) or [term.lower()])[0]
            if wn.synsets(lemmed) or wn.synsets(term.lower()):
                wn_known += 1

    # --- stage 1: cold bundle -- everything is a miss, which is honest
    cold = b.coverage(JARGON)

    # --- stage 2: the corpus PROPOSES the terms it met. This is the self-building step.
    for term, dom in JARGON:
        b.propose(term, dom, seen_in="project docs")

    # --- stage 3: a curator promotes them, with real definitions
    DEFS = {
        "macro-F1": ("Unweighted mean of per-class F1; used because class prevalence is skewed 61/24/8/7.",
                     "metrics", ""),
        "ECE": ("Expected calibration error: weighted |mean prediction - observed frequency| per bin.",
                "metrics", ""),
        "Brier": ("Mean squared error of predicted probabilities; strictly proper.", "metrics", ""),
        "Mondrian": ("Class-conditional conformal prediction: a separate quantile per class. Under-covers "
                     "rare classes if done marginally instead.", "conformal", ""),
        "isotonic": ("Monotone piecewise-constant calibration fitted by pool-adjacent-violators.",
                     "calibration", ""),
        "Platt": ("Sigmoid calibration of a scalar score; one slope and one intercept.", "calibration", ""),
        "commensurable": ("On the same scale, so pooling is meaningful. Two confidences that look alike but "
                          "mean different things are NOT commensurable.", "epistemics", ""),
        "conformal": ("Distribution-free set prediction with a finite-sample coverage guarantee.",
                      "conformal", ""),
        "AURC": ("Area under the risk-coverage curve; selective-prediction loss. Lower is better.",
                 "metrics", ""),
        "NEI": ("Not Enough Evidence -- an AVeriTeC verdict about the world, NOT an abstention decision.",
                "averttec", ""),
        "CTU": ("Conflicting-To-Unsupported label swap used in this project; measured, not assumed.",
                "averttec", ""),
        "ABSTAIN": ("A cost-sensitive decision NOT to answer. Distinct from NEI, which is a verdict.",
                    "epistemics", ""),
        "P585": ("Wikidata point-in-time qualifier: when the VALUE was measured.", "wikidata", ""),
        "P813": ("Wikidata retrieved-at: when an EDITOR checked the source -- an upper bound, not the "
                 "fact's date.", "wikidata", ""),
        "cantilever": ("A beam rigidly supported at one end only; the structural form of most long bridges.",
                       "engineering", ""),
    }
    for term, dom in JARGON:
        d, dom2, units = DEFS.get(term, ("", dom, ""))
        b.curate(term, dom2, d, curator="opencode", units=units)

    warm = b.coverage(JARGON)
    written = b.write_stub_indexes()

    if verbose:
        print("=== does a general lexicon know our jargon? ===")
        print(f"  terms tested                       : {len(JARGON)}")
        print(f"  known to WordNet                   : {wn_known}/{len(JARGON)} "
              f"({wn_known / len(JARGON):.0%})")
        print(f"  -> the other {len(JARGON) - wn_known} are invisible to a general-English lexicon, which is")
        print(f"     exactly the gap an ontology fills. And 'known' is not 'correct': WordNet's `bridge`")
        print(f"     sense says nothing about a cantilever span.")
        print()
        print("=== the bundle, before and after curation ===")
        print(f"  COLD  served {cold['served']}/{cold['n']} ({cold['served_rate']:.0%})  "
              f"miss {cold['miss']}")
        print(f"  WARM  served {warm['served']}/{warm['n']} ({warm['served_rate']:.0%})  "
              f"miss {warm['miss']}")
        print()
        print("=== and the gate is visible in the format ===")
        b.propose("sawtooth", "engineering", seen_in="corpus row 42")
        r = b.serve("sawtooth", "engineering")
        print(f"  a term the corpus used but nobody curated -> {r['status']}")
        print(f"    {r['reason']}")
        b.mark_stale("CTU", "averttec")
        r = b.serve("CTU", "averttec")
        print(f"  a term marked stale                       -> {r['status']}  ({r['reason'][:60]})")
        print()
        print(f"=== OKF files written ({len(written)} domain index(es)) ===")
        for p in sorted(written)[:4]:
            print(f"  {os.path.relpath(p, os.path.join(tmp, 'ontology'))}")
        print(f"  each term is one markdown file with frontmatter type/title/status -- the profile")
        print(f"  tools/okf_index.py already validates, so the bundle is greppable, diffable and")
        print(f"  reviewable in a pull request like any other source file.")
    return 0


# ------------------------------------------------------------------ tests
def run_self_test() -> bool:
    print("logos.ontology_bundle - SELF TEST")
    ok = True
    import tempfile

    tmp = tempfile.mkdtemp()
    b = Bundle(os.path.join(tmp, "ont"))

    # --- a cold bundle misses, and says so
    good = b.serve("ECE", "metrics")["status"] == "MISS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cold bundle MISSes rather than inventing a definition")

    # --- a proposal is NOT served. This is the anti-poisoning gate in the format itself.
    b.propose("ECE", "metrics", seen_in="row 7")
    r = b.serve("ECE", "metrics")
    good = r["status"] == "DRAFT_ONLY"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a proposed term is DRAFT_ONLY, not served as ground truth")

    # --- curation promotes it
    b.curate("ECE", "metrics", "Expected calibration error.", curator="tester")
    r = b.serve("ECE", "metrics")
    good = r["status"] == "HIT" and "calibration" in r["definition"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] curation promotes to HIT with the definition")

    # --- a stale term stops being served
    b.mark_stale("ECE", "metrics")
    good = b.serve("ECE", "metrics")["status"] == "DRAFT_ONLY"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a stale term is no longer served")

    # --- aliases resolve, including lemmatised forms
    b.curate("Mondrian conformal", "conformal", "Class-conditional conformal.", curator="t",
             aliases=("Mondrian",))
    good = b.serve("Mondrian", "conformal")["status"] == "HIT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an alias resolves to the curated term")

    # --- domain scoping: the same word in another domain does not match
    good = b.serve("Mondrian", "wikidata")["status"] == "MISS"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] domain scoping prevents a cross-domain false hit")

    # --- round-trip through OKF markdown
    b2 = Bundle(os.path.join(tmp, "ont"))
    good = b2.serve("Mondrian conformal", "conformal")["status"] == "HIT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the bundle reloads from OKF markdown with status intact")

    # --- OKF frontmatter carries the required keys
    p = b.path_for(b.terms["conformal|mondrian conformal"])
    text = open(p, encoding="utf-8").read()
    good = all(k in text for k in ("type: term", "title:", "status:"))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] emitted docs carry the OKF required keys (type/title/status)")

    # --- index generation is per-domain and counts the terms
    written = b.write_stub_indexes()
    txt = open(written[0], encoding="utf-8").read()
    good = "type: index" in txt and "| term |" in txt
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] per-domain index.md is generated with a term table")

    # --- and the compounding property: proposing raises the draft count, curating raises served
    b3 = Bundle(os.path.join(tmp, "ont3"))
    q = [("CTU", "averttec"), ("P585", "wikidata")]
    cold = b3.coverage(q)
    for term, dom in q:
        b3.propose(term, dom)
    mid = b3.coverage(q)
    for term, dom in q:
        b3.curate(term, dom, f"definition of {term}", curator="t")
    warm = b3.coverage(q)
    good = cold["served"] == 0 and mid["draft_only"] == 2 and warm["served"] == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] coverage compounds: cold {cold['served']} -> "
          f"drafted {mid['draft_only']} -> served {warm['served']}")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OKF-native domain jargon bundle with a curation gate.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
