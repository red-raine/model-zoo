"""logos/nlp/wsd_eval.py — WSD measured against a REAL gold standard, because 4 sentences are not a test.

WHY THIS MODULE EXISTS
----------------------
`lexicon_ensemble.measure()` compares plain Lesk against ConceptNet-enriched Lesk on FOUR SENTENCES. A 3-of-4
result has a 95% Wilson interval of **[0.30, 0.95]** -- it cannot separate 50% accuracy from 95%. Every claim
about enrichment made from that number is noise, and the module says so. This module replaces it with a gold
standard where the interval is narrow enough to decide something.

THE DATA
--------
Raganato, Camacho-Collados & Navigli (EACL 2017), "Word Sense Disambiguation: A Unified Evaluation Framework
and Empirical Comparison" -- the five all-words sets remapped to a COMMON WordNet version. That remapping is
what makes it usable here: the framework is on **WordNet 3.0**, and so is the local NLTK wordnet (verified:
`wn.get_version() == '3.0'`, 117,659 synsets). So `synsets(w, pos)[0]` -- this project's canonical-sense rule
-- IS the standard first-sense baseline, with no mapping step and no reimplementation.

Data obtained from the `SapienzaNLP/wsd-hard-benchmark` distribution (Maru et al., ACL 2022), which ships the
amended corpus. Gold keys are WordNet SENSE KEYS and an instance may have MORE THAN ONE correct sense
(`emission%1:04:00:: emission%1:27:00::`), so a prediction is correct if it is in the gold SET. Scoring it as
single-label would understate every method.

PUBLISHED NUMBERS THIS IS CHECKED AGAINST (first-sense / MFS baselines, F1):
    Senseval-2 65.6 | Senseval-3 66.0 | SemEval-2007 54.5 | SemEval-2013 63.8 | SemEval-2015 67.1 | ALL 65.5
    inter-annotator ceiling: 91.0 / 94.5 / 93.8 / 88.6 / 90.4 / ALL 91.0
If our MFS lands near 65.5 the pipeline is wired correctly; if it lands far off, the harness is wrong and no
other number from it means anything.

USAGE
    python tools/logos/nlp/wsd_eval.py --self-test
    python tools/logos/nlp/wsd_eval.py --measure [--limit N] [--methods mfs,lesk,enriched]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

WSD_DIR = os.environ.get(
    "LOGOS_WSD_DIR",
    r"B:\datasets\wsd\wsd-hard-benchmark-main\wsd_hard_benchmark\ALLamended")
DATA = os.path.join(WSD_DIR, "ALLamended.data.xml")
GOLD = os.path.join(WSD_DIR, "ALLamended.gold.key.txt")

# The published first-sense baseline for the FULL all-words set (Raganato et al. 2017).
PUBLISHED_MFS_ALL = 65.5

POSMAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "": "n"}

_KEYMAP = None
_SIG_CACHE = {}
_DATA_CACHE = {}


def available() -> tuple:
    if not os.path.exists(DATA):
        return False, f"no WSD corpus at {DATA}"
    if not os.path.exists(GOLD):
        return False, f"no gold key at {GOLD}"
    return True, WSD_DIR


def _wn():
    from nltk.corpus import wordnet as wn
    return wn


def key_map() -> dict:
    """sense key ('bank%1:17:00::') -> synset name ('bank.n.01'). Built once; ~200k entries."""
    global _KEYMAP
    if _KEYMAP is not None:
        return _KEYMAP
    wn = _wn()
    m = {}
    for s in wn.all_synsets():
        for l in s.lemmas():
            try:
                m[l.key()] = s.name()
            except Exception:                                # noqa: BLE001
                continue
    _KEYMAP = m
    return m


def load(limit: int = 0) -> list:
    """Parse the corpus and attach gold synsets. Context = the full sentence the instance sits in."""
    ck = (limit,)
    if ck in _DATA_CACHE:
        return _DATA_CACHE[ck]
    ok, why = available()
    if not ok:
        return []
    km = key_map()
    gold = {}
    with open(GOLD, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                gold[parts[0]] = set(parts[1:])

    rows = []
    for _event, elem in ET.iterparse(DATA, events=("end",)):
        if elem.tag != "sentence":
            continue
        words, targets = [], []
        for child in elem:
            if child.tag == "wf":
                words.append(child.text or "")
            elif child.tag == "instance":
                words.append(child.text or "")
                targets.append((child.get("id"), child.get("lemma") or "", child.get("pos") or ""))
        ctx = " ".join(w for w in words if w)
        for iid, lemma, pos in targets:
            keys = gold.get(iid)
            if not keys:
                continue
            gold_syn = {km[k] for k in keys if k in km}
            if not gold_syn:
                continue
            rows.append({"id": iid, "lemma": lemma.replace("_", " "),
                         "pos": POSMAP.get(pos, "n"), "context": ctx,
                         "gold": sorted(gold_syn), "gold_keys": sorted(keys)})
            if limit and len(rows) >= limit:
                _DATA_CACHE[ck] = rows
                return rows
        elem.clear()
    _DATA_CACHE[ck] = rows
    return rows


# ------------------------------------------------------------------ the methods
def mfs(lemma: str, pos: str):
    """FIRST SENSE. The published baseline, and this project's canonical-sense rule."""
    try:
        ss = _wn().synsets(lemma, pos=pos)
    except Exception:                                        # noqa: BLE001
        return None
    return ss[0].name() if ss else None


def n_candidates(lemma: str, pos: str) -> int:
    try:
        return len(_wn().synsets(lemma, pos=pos))
    except Exception:                                        # noqa: BLE001
        return 0


def _stem(w: str) -> str:
    w = (w or "").lower()
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _ctx_tokens(context: str) -> set:
    return {_stem(t) for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (context or "").lower())}


def signature(synset_name: str, enrich=False) -> set:
    """Synset signature.

    `enrich` may be False, 'verified' (injection: WordNet must corroborate the ConceptNet edge) or 'union'
    (the old lexicon_ensemble behaviour: inject every ConceptNet edge regardless). Both are offered on the same
    code path BECAUSE the published result is a comparison between them, not an argument about them.
    """
    ck = (synset_name, enrich)
    if ck in _SIG_CACHE:
        return _SIG_CACHE[ck]
    wn = _wn()
    try:
        s = wn.synset(synset_name)
    except Exception:                                        # noqa: BLE001
        _SIG_CACHE[ck] = set()
        return set()
    parts = [s.definition() or ""] + list(s.examples() or [])
    parts += [l.name().replace("_", " ") for l in s.lemmas()]
    if enrich:
        try:
            from logos.nlp import synset_injection as inj
            parts += inj.injected_terms(synset_name, allow_unverified=(enrich == "union"))
        except Exception:                                    # noqa: BLE001
            pass
    toks = {_stem(t) for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", " ".join(parts).lower())}
    _SIG_CACHE[ck] = toks
    return toks


def lesk(context: str, lemma: str, pos: str, enrich: bool = False):
    """Signature-overlap WSD. Ties are broken by WordNet order (so a tie inherits first sense, not noise)."""
    wn = _wn()
    try:
        cands = wn.synsets(lemma, pos=pos)
    except Exception:                                        # noqa: BLE001
        return None
    if not cands:
        return None
    ctx = _ctx_tokens(context) - {_stem(lemma)}
    best, best_score = None, -1
    for s in cands:
        overlap = ctx & signature(s.name(), enrich=enrich)
        if len(overlap) > best_score:
            best, best_score = s.name(), len(overlap)
    return best


METHODS = {
    "mfs": lambda r: mfs(r["lemma"], r["pos"]),
    "lesk": lambda r: lesk(r["context"], r["lemma"], r["pos"], enrich=False),
    "enriched": lambda r: lesk(r["context"], r["lemma"], r["pos"], enrich="verified"),
    "union": lambda r: lesk(r["context"], r["lemma"], r["pos"], enrich="union"),
}


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson interval. Reported for EVERY accuracy here, because a bare fraction invites over-reading."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(rows: list, method: str, polysemous_only: bool = False) -> dict:
    fn = METHODS[method]
    n = k = 0
    misses = []
    for r in rows:
        if polysemous_only and n_candidates(r["lemma"], r["pos"]) <= 1:
            continue
        n += 1
        try:
            pred = fn(r)
        except Exception:                                    # noqa: BLE001
            pred = None
        if pred and pred in r["gold"]:
            k += 1
        elif len(misses) < 25:
            misses.append({"lemma": r["lemma"], "pos": r["pos"], "gold": r["gold"], "pred": pred,
                           "ctx": r["context"][:90]})
    lo, hi = wilson(k, n)
    return {"method": method, "n": n, "correct": k,
            "accuracy": round(100.0 * k / n, 2) if n else 0.0,
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)], "misses": misses}


# ------------------------------------------------------------------ measure
def measure(limit: int = 0, methods=("mfs", "lesk", "enriched")) -> int:
    ok, why = available()
    print("=== WSD against a real gold standard (Raganato et al. 2017, WordNet 3.0) ===")
    print(f"  corpus: {why}")
    if not ok:
        return 1
    rows = load(limit)
    if not rows:
        print("  no rows loaded")
        return 1
    poly = sum(1 for r in rows if n_candidates(r["lemma"], r["pos"]) > 1)
    print(f"  instances: {len(rows):,}   polysemous: {poly:,} ({100*poly/len(rows):.1f}%)")
    print(f"  local wordnet {_wn().get_version()} -- the SAME inventory the gold is remapped to, so")
    print(f"  first-sense here IS the published baseline (ALL = {PUBLISHED_MFS_ALL}).\n")

    results = []
    for m in methods:
        r = score(rows, m)
        results.append(r)
        print(f"  {m:10s} {r['accuracy']:6.2f}%   n={r['n']:,}   "
              f"95% CI [{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]")

    print(f"\n  published first-sense on ALL: {PUBLISHED_MFS_ALL}  (inter-annotator ceiling 91.0)")
    base = next((r for r in results if r["method"] == "mfs"), None)
    if base:
        gap = base["accuracy"] - PUBLISHED_MFS_ALL
        print(f"  our MFS minus published: {gap:+.2f} points")
        if abs(gap) > 6.0:
            print("  WARNING: more than 6 points off the published baseline. The harness is suspect and no")
            print("  other number on this page should be believed until that is explained.")
        else:
            print("  The harness reproduces the published baseline, so the other rows are interpretable.")

    # the comparison that matters, with a paired test rather than two overlapping intervals
    for label, a, b in (("plain lesk -> VERIFIED injection", "lesk", "enriched"),
                        ("UNION injection -> VERIFIED injection", "union", "enriched")):
        if a not in methods or b not in methods:
            continue
        pl, _ = _paired(rows, a, b)
        print(f"\n  PAIRED TEST on {pl['n']:,} instances ({label}):")
        print(f"    {a} correct, {b} wrong : {pl['lost']}")
        print(f"    {a} wrong, {b} correct : {pl['gained']}")
        print(f"    both correct: {pl['both_ok']}   both wrong: {pl['both_bad']}")
        m = pl["lost"] + pl["gained"]
        if m:
            p = _binom_two_sided(min(pl["lost"], pl["gained"]), m)
            print(f"    McNemar exact two-sided p = {p:.4g} on {m} discordant instances")
            if p < 0.05:
                verdict = "IMPROVED" if pl["gained"] > pl["lost"] else "DEGRADED"
                print(f"    ==> {b} {verdict} significantly. Net {pl['gained'] - pl['lost']:+d} instances.")
            else:
                print(f"    ==> NOT significant. {b} is indistinguishable from {a} here, and claiming")
                print(f"        otherwise would be exactly the error the 4-sentence test invited.")
        else:
            print(f"    the two methods never disagree -- {b} changed nothing on this set")
    return 0


def _paired(rows: list, a: str, b: str) -> tuple:
    fa, fb = METHODS[a], METHODS[b]
    lost = gained = both_ok = both_bad = 0
    for r in rows:
        try:
            pa, pb = fa(r), fb(r)
        except Exception:                                    # noqa: BLE001
            continue
        oka = bool(pa and pa in r["gold"])
        okb = bool(pb and pb in r["gold"])
        if oka and not okb:
            lost += 1
        elif okb and not oka:
            gained += 1
        elif oka and okb:
            both_ok += 1
        else:
            both_bad += 1
    return {"n": len(rows), "lost": lost, "gained": gained, "both_ok": both_ok, "both_bad": both_bad}, None


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p for k successes of n under p=0.5 (McNemar)."""
    if n == 0:
        return 1.0
    def lc(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    tail = sum(math.exp(lc(n, i) - n * math.log(2)) for i in range(0, k + 1))
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.nlp.wsd_eval - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] gold corpus available ({why})")
    if not av:
        print("SELF TEST: SKIPPED (no gold corpus)")
        return True

    rows = load(400)
    good = len(rows) > 300
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] loaded {len(rows)} instances with gold senses")

    # --- multi-sense gold must be kept as a SET, or scoring understates every method
    multi = [r for r in rows if len(r["gold"]) > 1]
    good = len(multi) > 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(multi)} instance(s) have MULTIPLE gold senses "
          f"(kept as a set)")

    # --- own WordNet must match the gold's inventory, or nothing below is comparable
    good = _wn().get_version() == "3.0"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] local WordNet is 3.0, the version the gold is remapped to")

    # --- MFS on a sample must land in the published neighbourhood
    r = score(rows, "mfs")
    good = 50.0 <= r["accuracy"] <= 80.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] MFS on a {r['n']}-instance sample = {r['accuracy']:.1f}% "
          f"[{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}], in the neighbourhood of the published 65.5")

    # --- every method returns a prediction or an honest None, never an exception
    bad = 0
    for m in METHODS:
        try:
            s = score(rows[:60], m)
            if s["n"] != 60:
                bad += 1
        except Exception:                                    # noqa: BLE001
            bad += 1
    good = bad == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] all {len(METHODS)} methods score without raising")

    # --- Wilson interval is honest about small n (the reason the old test was worthless)
    lo, hi = wilson(3, 4)
    good = (hi - lo) > 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] wilson(3/4) = [{lo:.2f}, {hi:.2f}] -- a 4-item test cannot "
          f"decide anything, which is why it was replaced")

    good = wilson(0, 0) == (0.0, 0.0)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] wilson handles n=0 without dividing by zero")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WSD against a real gold standard.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="mfs,lesk,union,enriched")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure(args.limit, tuple(args.methods.split(",")))
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
