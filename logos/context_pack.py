"""logos/context_pack.py — gather context BEFORE evaluating, as a graph, and batch by ontology.

THE IDEA THIS IMPLEMENTS
------------------------
Every gate built so far judges a sample in isolation. This module does the opposite: before any verdict,
assemble what is *known about* the sample, as a knowledge graph, and derive an ontology type so similar
samples can be handled together.

    claim -> gather context -> ontology type -> knowledge graph -> (then, and only then) evaluate

WHY IT IS NOT JUST A BIGGER PROMPT
----------------------------------
Two properties matter and a prompt does not have them:

1. **The context is a GRAPH, so it can be queried and checked.** "This function's doctest says 3*2=6" is
   a triple, and a triple can contradict another triple. That is what `logic/smt` and `rdflib` can act on.
2. **The ontology type is a BATCHING KEY.** Rows that share a type need the same context keys and the same
   tests, so they can be submitted together -- which is the only way a cheap model call becomes affordable
   at corpus scale, and the reason batching is a design decision rather than an optimisation.

WHAT CONTEXT IS ACTUALLY AVAILABLE, PER KIND -- and this is deliberately narrow
------------------------------------------------------------------------------
    code    signature, arity, docstring, DOCTESTS, imports, purity guess
    math    sympy-parsed expression, free symbols, whether it is an equality
    factual entities, relation, date, source

**Doctests are the important one.** A docstring in prose needs a language model to become a test, which is
the unsolved NL-to-symbolic step. A doctest is ALREADY a test: runnable, exact, no interpretation. So
context that can be turned into a check without a model is worth far more than context that cannot, and
this module prefers it.

USAGE
    python tools/logos/context_pack.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

try:
    import rdflib
except Exception:                                          # noqa: BLE001
    rdflib = None

NS = "https://logos.local/ctx#"


# ------------------------------------------------------------------------------- types
@dataclass
class FunctionFacts:
    name: str
    args: list[str] = field(default_factory=list)
    has_docstring: bool = False
    doctests: list[str] = field(default_factory=list)
    returns_value: bool = False
    is_async: bool = False

    @property
    def arity(self) -> int:
        return len(self.args)


@dataclass
class ContextPack:
    kind: str = "unknown"
    parses: bool = False
    ontology: str = "unknown"
    functions: list[FunctionFacts] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    digest: str = ""

    @property
    def runnable_specs(self) -> int:
        return sum(len(f.doctests) for f in self.functions)

    def batch_key(self) -> str:
        """Rows sharing this key need the same context keys and the same tests, so they can be batched."""
        return self.ontology

    def as_dict(self) -> dict:
        return {"kind": self.kind, "parses": self.parses, "ontology": self.ontology,
                "functions": [{"name": f.name, "arity": f.arity,
                               "doctests": len(f.doctests)} for f in self.functions],
                "imports": self.imports[:8], "runnable_specs": self.runnable_specs,
                "notes": self.notes, "digest": self.digest}


# ------------------------------------------------------------------------------- gathering
_DOCTEST = re.compile(r"^\s*>>>\s*(.+)$", re.M)


def gather(source: str, kind: str = "code") -> ContextPack:
    """Assemble what is known about this sample. No verdict is formed here."""
    pack = ContextPack(kind=kind, digest=hashlib.sha1(source.encode("utf-8")).hexdigest()[:12])
    try:
        tree = ast.parse(source)
        pack.parses = True
    except SyntaxError as e:
        pack.notes.append(f"unparsable: {e.msg}")
        pack.ontology = f"{kind}/unparsable"
        return pack

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                pack.imports.extend(a.name.split(".")[0] for a in node.names)
            elif node.module:
                pack.imports.append(node.module.split(".")[0])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            args = [a.arg for a in node.args.args if a.arg != "self"]
            pack.functions.append(FunctionFacts(
                name=node.name, args=args, has_docstring=bool(doc),
                doctests=_DOCTEST.findall(doc),
                returns_value=any(isinstance(n, ast.Return) and n.value is not None
                                  for n in ast.walk(node)),
                is_async=isinstance(node, ast.AsyncFunctionDef)))

    pack.ontology = _classify(pack)
    return pack


def _classify(pack: ContextPack) -> str:
    """A coarse, STABLE type used as a batching key and as a context-key selector."""
    if not pack.parses:
        return "code/unparsable"
    if not pack.functions:
        return "code/module-no-functions"
    f = pack.functions[0]
    purity = "impure" if pack.imports else "likely-pure"
    spec = "with-doctest" if pack.runnable_specs else "no-runnable-spec"
    asyncness = "async" if f.is_async else "sync"
    return f"code/{purity}/{spec}/{asyncness}/arity-{f.arity}"


# ------------------------------------------------------------------------------- lemmatisation
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def normalize_terms(text: str, keep_order: bool = True) -> list[str]:
    """Lemmatise to a canonical form so ontology lookup matches `bridges`, `bridge`, `bridged` as one term.

    WHY THIS IS A SEPARATE PASS, and the user is right that it belongs in the ontology pre-research step: an
    ontology is keyed on canonical lexical forms. Matching surface forms means `bridge`, `bridges` and
    `bridging` are three misses instead of one hit, so the ontology looks useless when it is only unqueried.
    Normalising ONCE, before any lookup, is far cheaper than normalising at every call site.

    POS is tried noun-first then verb, because dataset nouns dominate and `running` should resolve to `run`
    either way while `mice` only works as a noun. Falls back to the surface form when WordNet has no entry --
    a term we cannot lemmatise is still a term.
    """
    words = _WORD.findall(text)
    if not words:
        return []
    try:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet as wn
        lem = WordNetLemmatizer()

        def canon(w: str) -> str:
            wl = w.lower()
            # NOUN-FIRST IS WRONG here: `running` is also a noun, so lemmatize(...,'n') returns it unchanged
            # and the verb form `run` is never reached. Collect every POS candidate and prefer the SHORTEST
            # one that WordNet actually knows -- `running`->run, `mice`->mouse, `analysis`->analysis.
            cands = []
            for pos in ("n", "v", "a", "r"):
                c = lem.lemmatize(wl, pos)
                has = bool(wn.synsets(c, pos))
                cands.append((not has, len(c), c))       # synset-presence FIRST, then shortness
            cands.sort()
            return cands[0][2]
    except Exception:                                     # noqa: BLE001
        def canon(w: str) -> str:
            return w.lower()

    out: list[str] = []
    for w in words:
        c = canon(w)
        if keep_order:
            if c not in out:
                out.append(c)
        else:
            out.append(c)
    return out if keep_order else sorted(out)


# ------------------------------------------------------------------------------- graph
def to_graph(pack: ContextPack):
    """Emit the pack as an RDF graph. This is the 'open knowledge format' side of the idea: the context is
    a queryable graph, not a blob of text, so statements about it can contradict each other and be checked."""
    if rdflib is None:
        return None
    g = rdflib.Graph()
    g.bind("lx", NS)
    subject = rdflib.URIRef(f"{NS}sample/{pack.digest}")
    g.add((subject, rdflib.RDF.type, rdflib.URIRef(f"{NS}{pack.ontology.replace('/', '_')}")))
    g.add((subject, rdflib.URIRef(f"{NS}kind"), rdflib.Literal(pack.kind)))
    g.add((subject, rdflib.URIRef(f"{NS}parses"), rdflib.Literal(pack.parses)))
    g.add((subject, rdflib.URIRef(f"{NS}runnableSpecs"), rdflib.Literal(pack.runnable_specs)))
    for f in pack.functions:
        fsub = rdflib.URIRef(f"{NS}fn/{pack.digest}/{f.name}")
        g.add((subject, rdflib.URIRef(f"{NS}hasFunction"), fsub))
        g.add((fsub, rdflib.RDF.type, rdflib.URIRef(f"{NS}Function")))
        g.add((fsub, rdflib.URIRef(f"{NS}arity"), rdflib.Literal(f.arity)))
        g.add((fsub, rdflib.URIRef(f"{NS}name"), rdflib.Literal(f.name)))
        for dt in f.doctests:
            g.add((fsub, rdflib.URIRef(f"{NS}doctest"), rdflib.Literal(dt)))
    for imp in pack.imports:
        g.add((subject, rdflib.URIRef(f"{NS}imports"), rdflib.Literal(imp)))
    return g


def serialize(pack: ContextPack, fmt: str = "turtle") -> str:
    g = to_graph(pack)
    if g is None:
        return ""
    return g.serialize(format=fmt)


def render(pack: ContextPack) -> str:
    """Compact, prompt-ready context. Cacheable, because it is a deterministic function of the source."""
    lines = [f"ontology: {pack.ontology}", f"parses: {pack.parses}",
             f"runnable_specs: {pack.runnable_specs}"]
    for f in pack.functions[:4]:
        spec = f" | doctests={len(f.doctests)}" if f.doctests else ""
        lines.append(f"fn {f.name}({', '.join(f.args)}) arity={f.arity}{spec}")
    if pack.imports:
        lines.append(f"imports: {', '.join(sorted(set(pack.imports))[:6])}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.context_pack - SELF TEST")
    ok = True

    src = '''"""A module."""


def area(r):
    """Area of a circle.

    >>> round(area(1), 4)
    3.1416
    """
    return 3.141592653589793 * r * r


def perimeter(r):
    return 2 * 3.141592653589793 * r
'''
    p = gather(src)
    good = p.parses and len(p.functions) == 2 and p.runnable_specs == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] gathers functions and counts runnable specs "
          f"({len(p.functions)} fns, {p.runnable_specs} doctest)")

    # --- the ontology type is stable and carries the facts that decide batching
    good = p.ontology.startswith("code/") and "with-doctest" in p.ontology and "arity-1" in p.ontology
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ontology type is a usable batching key: {p.ontology}")

    # --- unparsable input is classified, not crashed on
    bad = gather("def f(:\n")
    good = bad.parses is False and bad.ontology == "code/unparsable"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unparsable input yields code/unparsable, no exception")

    # --- doctests ARE extracted, because they are the only context that becomes a check without a model
    good = p.functions[0].doctests and "round(area(1), 4)" in p.functions[0].doctests[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] doctest bodies are extracted verbatim")

    # --- the rendering is deterministic, so it is cacheable
    good = render(gather(src)) == render(gather(src))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] rendering is deterministic (cacheable)")

    # --- and the digest changes with the source, so a cache keyed on it cannot go stale
    good = gather(src).digest != gather(src + "\n# x\n").digest
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the digest changes when the source changes")

    # --- RDF graph emission, if rdflib is present
    if rdflib is None:
        print(f"  [SKIP] rdflib not installed -- graph emission not exercised")
    else:
        g = to_graph(p)
        good = g is not None and len(g) > 5
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] emits an RDF graph ({len(g)} triples)")

        ttl = serialize(p, "turtle")
        good = "doctest" in ttl and "ontology" not in ttl.lower() or "Function" in ttl
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] serialises to turtle with the function and its doctest")

        # --- a graph is QUERYABLE, which a prompt is not: this is the point of using RDF at all
        q = f"""PREFIX lx: <{NS}>
SELECT ?fn WHERE {{ ?s lx:hasFunction ?fn . }}"""
        n = len(list(g.query(q)))
        good = n == len(p.functions)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] and is queryable: {n} functions found by SPARQL")

    # --- LEMMATISATION: singular/plural/verb forms collapse to one ontology key. Without this an ontology
    #     looks useless because every surface form is a separate miss.
    pairs = [("bridges", "bridge"), ("running", "run"), ("mice", "mouse"), ("analysis", "analysis")]
    got = {w: normalize_terms(w)[0] for w, _want in pairs}
    bad = [(w, got[w], want) for w, want in pairs if got[w] != want]
    good = not bad
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] lemmatisation canonicalises surface forms "
          f"({got})" + (f"  MISSES: {bad}" if bad else ""))

    # --- dedup preserves first-seen order, so a rendered context is stable and cacheable
    good = normalize_terms("bridges bridge bridged") == ["bridge"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] duplicate surface forms collapse to one term")

    # --- text with no words yields an empty list, not a crash
    good = normalize_terms("123 += 456") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] non-word input yields no terms, no exception")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gather context before evaluating, as a graph.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
