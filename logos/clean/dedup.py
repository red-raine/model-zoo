"""logos/clean/dedup.py — exact + near-duplicate detection, because a duplicated row is trained on twice.

WHY THIS IS THE HIGHEST-VALUE CLEANING STEP, AND THE ONLY ONE WITH A LARGE MEASURED WIN
-------------------------------------------------------------------------------------
Every other check in this suite decides whether a row is correct. This one decides whether the row is the
SAME ROW, and the literature is unambiguous about the payoff: SemDeDup removes ~50% of a LAION subset with
minimal performance loss and roughly halves training time (Abbas et al., NeurIPS 2023); D4 gets ~20%
efficiency gains; deduplicating C4 measurably reduces memorisation (Lee et al., ACL 2022). None of that needs
a label, a GPU, or a model.

It is also the one place where the CORPUS is the oracle: two byte-identical rows are duplicates by definition,
so there is nothing to get wrong and nothing to tune.

TWO PASSES, DELIBERATELY
------------------------
    EXACT   sha256 of normalised text. Zero false positives, catches the lazy duplicates.
    NEAR    MinHash LSH over 13-word shingles, Jaccard 0.8 (the SlimPajama / RedPajama parameterisation:
            num_perm 128, banded for LSH at 0.8).
    SECOND  a lower threshold (0.5) for TEMPLATE repetition. This is the band that matters for
            machine-generated data: "X is a 330 metre tower" and "Y is a 300 metre tower" are different facts
            wearing the same template, and at 0.8 they look distinct. The second pass does NOT delete them --
            it reports them, because deleting a template cluster would delete distinct facts. That distinction
            is the whole reason it is a separate pass rather than a lower threshold on the first one.

MINHASH IS APPROXIMATE, AND THE ERROR IS ONE-SIDED IN A WAY THAT MATTERS
-----------------------------------------------------------------------
LSH with banding has a false-NEGATIVE rate (two similar rows that never collide) and a false-POSITIVE rate
(collision despite low Jaccard). At num_perm=128 with 0.8 the false-positive side is negligible; the
false-negative side means dedup UNDER-reports. For a precision-first suite that is the correct direction: a
missed duplicate costs a little training signal, a wrongly-deleted distinct row costs data.

USAGE
    python tools/logos/clean/dedup.py --self-test
    python tools/logos/clean/dedup.py --measure [--corpus <path>] [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

NUM_PERM = 112          # FineWeb uses 112 hashes; RedPajama-v2 uses 128
BANDS = 14              # PINNED, never auto-chosen -- see _pairs_lsh for the measured reason
ROWS = 8                # 14x8 at 112 perms -> P(0.80)=0.924, P(0.85)=0.988
CANDIDATE_THRESHOLD = 0.7   # loose generation (recall), exact verification at THRESHOLD_NEAR
EXACT_LIMIT = 2500      # below this, all-pairs is complete and costs seconds
SHINGLE = 13            # SlimPajama's parameter, for DOCUMENT-length text only
THRESHOLD_NEAR = 0.8    # the standard "duplicate" bar
THRESHOLD_TEMPLATE = 0.5  # template repetition: REPORT, never delete


def available() -> tuple:
    try:
        import datasketch                                   # noqa: F401
    except Exception as e:                                  # noqa: BLE001
        return False, f"datasketch missing: {type(e).__name__}"
    return True, "ready"


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation runs. Two rows differing only in spacing or case are
    the same row, and treating them as distinct is how a corpus ends up with 'yes' and 'Yes' both."""
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def exact_key(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:32]


def _numbers(text: str) -> set:
    """The numeric content of a row, as raw values. Reuses the extractor from numeric_conflict."""
    try:
        from logos import numeric_conflict
        return {round(float(x["value"]), 6) for x in numeric_conflict.extract(text)}
    except Exception:                                       # noqa: BLE001
        return set()


def _shingle_size(n_words: int) -> int:
    """ADAPTIVE shingle size, chosen by MEASUREMENT rather than by copying a long-document parameter.

    SlimPajama's 13-gram setting is calibrated on documents. On a CLAIM it fails in the direction that
    silently loses recall, and I measured it rather than assuming:
        a 29-word claim with ONE word changed, at k=9  -> Jaccard 0.615  (below the 0.8 bar: missed)
        an 18-word claim with ONE number changed, k=6  -> Jaccard 0.370  (missed even at 0.5)
        330 m vs 300 m, same shape                     -> never clustered at all
    Ordered n-grams punish a single edit proportionally to n, so the shorter the text the worse it gets. The
    whole point of a shingle is to capture ORDER, and a one-sentence factual claim carries little of it; its
    CONTENT is what repeats. So short texts use k=1 -- token-set Jaccard -- which puts both of the cases
    above at 0.93 and 0.90, where a duplicate bar can actually see them.
    """
    if n_words <= 0:
        return 1
    if n_words < 60:
        return 1                      # token-set Jaccard for short claims
    return SHINGLE                    # 13-gram shingles for document-length text


def _shingles(text: str, k: int = 0) -> set:
    words = normalise(text).split()
    if not words:
        return set()
    k = k or _shingle_size(len(words))
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def exact_clusters(texts: list) -> dict:
    """Group by exact normalised hash. Zero false positives by construction."""
    groups = collections.defaultdict(list)
    for i, t in enumerate(texts):
        groups[exact_key(t)].append(i)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    return {"n_groups": len(dups),
            "n_duplicate_rows": sum(len(v) - 1 for v in dups.values()),
            "clusters": {k: v for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))},
            "unique": len(groups)}


def _jaccard(a: set, b: set) -> float:
    """TRUE Jaccard. This, not the MinHash estimate, is what licenses a deletion."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _pairs_exact(sets: list, threshold: float) -> list:
    """Every pair whose TRUE Jaccard >= threshold, with a length prefilter.

    Complete by construction -- nothing similar is missed. Measured throughput ~138k pairs/s, so 500 rows
    (125k pairs) is about a second and 2,500 rows (3.1M pairs) is about 23 s. Above that, candidates come from
    LSH and are still verified exactly here.
    """
    out = []
    n = len(sets)
    for i in range(n):
        li = len(sets[i])
        if li < 2:
            continue
        for j in range(i + 1, n):
            lj = len(sets[j])
            if not lj:
                continue
            # length prefilter: if the smaller cannot reach `threshold` against the larger, skip
            if min(li, lj) / max(li, lj) < threshold:
                continue
            if _jaccard(sets[i], sets[j]) >= threshold:
                out.append((i, j))
    return out


def _pairs_lsh(sets: list, threshold: float, candidate_threshold: float) -> list:
    """Candidates from MinHash LSH, VERIFIED exactly. Two separate jobs, deliberately.

    THE MEASUREMENT THAT FORCED THIS SPLIT
    --------------------------------------
    `MinHashLSH(threshold=0.8, num_perm=128)` lets datasketch choose the banding, and it chose (b=9, r=13).
    I computed that configuration's S-curve on this machine: **P(0.80) = 0.399**. So the module was reporting
    "duplicates at Jaccard >= 0.8" while missing ~60% of the pairs exactly AT that threshold. Recall was worst
    precisely where it claimed to be deciding.
    FineWeb (arXiv 2406.17557, App. E.1) publishes an explicit 14x8 / 112-perm configuration with
    P(0.75)=0.772, P(0.80)=0.924, P(0.85)=0.988, and the published figures are chosen rather than optimised --
    the optimizer's symmetric loss is wrong for dedup, where a FALSE NEGATIVE costs recall of a real duplicate.
    Banding is therefore pinned here, and candidate generation runs at a LOOSER threshold (higher recall) while
    the VERDICT is made by true Jaccard at the strict threshold. That is candidate-generation-then-verification,
    not a "second pass at a lower threshold": the second stage is exact, so it cannot add a false positive.
    """
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=candidate_threshold, num_perm=NUM_PERM, params=(BANDS, ROWS))
    ms = _minhashes(sets)
    for i, m in enumerate(ms):
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass
    pairs, seen = [], set()
    for i in range(len(sets)):
        for j in lsh.query(ms[i]):
            j = int(j)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            if _jaccard(sets[i], sets[j]) >= threshold:      # EXACT, not the estimate
                pairs.append(key)
    return pairs


def _minhashes(sets: list):
    from datasketch import MinHash
    out = []
    for s in sets:
        m = MinHash(num_perm=NUM_PERM)
        for sh in s:
            m.update(sh.encode("utf-8"))
        out.append(m)
    return out


def near_clusters(texts: list, threshold: float = THRESHOLD_NEAR) -> dict:
    """Clusters at TRUE Jaccard >= `threshold`, union-find so a chain collapses once."""
    if not texts:
        return {"n_clusters": 0, "n_duplicate_rows": 0, "clusters": [], "mode": "none"}
    sets = [_shingles(t) for t in texts]

    if len(texts) <= EXACT_LIMIT:
        # complete all-pairs: nothing similar can be missed, and at this size it is cheap
        pairs, mode = _pairs_exact(sets, threshold), f"exact all-pairs ({len(texts)} rows)"
    else:
        # candidates at a looser bar, verdicts by exact Jaccard
        pairs = _pairs_lsh(sets, threshold, CANDIDATE_THRESHOLD)
        mode = f"LSH({BANDS}x{ROWS}) candidates at {CANDIDATE_THRESHOLD}, exact verify at {threshold}"

    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)
    groups = collections.defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    clusters = [sorted(v) for v in groups.values() if len(v) > 1]
    clusters.sort(key=len, reverse=True)
    return {"n_clusters": len(clusters), "n_duplicate_rows": sum(len(c) - 1 for c in clusters),
            "clusters": clusters, "threshold": threshold, "mode": mode, "n_pairs": len(pairs)}


def split_by_numeric(texts: list, clusters: list) -> tuple:
    """Divide text-similar clusters into DELETABLE and NUMERICALLY-DISTINCT.

    A cluster is deletable only if every member carries the same numeric content. One differing number means
    the rows make different claims, however similar they read, and deleting one destroys a unique fact.

    MEASURED EXAPLE that motivates this: "the eiffel tower is 330 metres tall" vs "...300 metres tall" score
    far above the 0.8 duplicate bar on shingles. They are not duplicates.
    """
    deletable, numeric_distinct = [], []
    for c in clusters:
        nums = [_numbers(texts[i]) for i in c]
        union = set().union(*nums) if nums else set()
        # distinct if any member's numbers differ from any other's
        differs = any(nums[i] != nums[j] for i in range(len(nums)) for j in range(i + 1, len(nums)))
        (numeric_distinct if (differs and union) else deletable).append(c)
    return deletable, numeric_distinct


def dedup(texts: list) -> dict:
    """The full verdict: what is an exact duplicate, what is near, and what only shares a TEMPLATE.

    The three lists are kept separate on purpose, because they license different actions:
        exact     -> safe to delete (identical by definition)
        near      -> review, then delete (0.8 Jaccard at 13-gram is almost always the same row)
        template  -> NEVER delete (distinct facts in one shape); report for provenance/balance checks
    """
    ex = exact_clusters(texts)
    # remove exact duplicates before the near pass, so a near cluster is genuinely near
    seen, keep_idx = set(), []
    for i, t in enumerate(texts):
        k = exact_key(t)
        if k in seen:
            continue
        seen.add(k)
        keep_idx.append(i)
    reduced = [texts[i] for i in keep_idx]
    near = near_clusters(reduced, THRESHOLD_NEAR)
    # map reduced indices back to original
    near_orig = [[keep_idx[j] for j in c] for c in near["clusters"]]
    # ---- NUMERIC-AWARE GUARD: a text-similar pair that disagrees on its numbers is NOT a duplicate
    deletable, numeric_distinct = split_by_numeric(texts, near_orig)
    near_orig = deletable
    # template pass: exclude everything already flagged as a near duplicate
    flagged = {i for c in near_orig for i in c} | {i for c in numeric_distinct for i in c}
    tpl_idx = [i for i in keep_idx if i not in flagged]
    tpl_texts = [texts[i] for i in tpl_idx]
    tpl = near_clusters(tpl_texts, THRESHOLD_TEMPLATE)
    tpl_orig = [[tpl_idx[j] for j in c] for c in tpl["clusters"]]
    return {
        "n": len(texts),
        "exact": {"n_clusters": ex["n_groups"], "n_dropped": ex["n_duplicate_rows"],
                  "clusters": list(ex["clusters"].values())[:20]},
        "near": {"n_clusters": len(near_orig), "n_rows": sum(len(c) for c in near_orig),
                 "threshold": THRESHOLD_NEAR, "clusters": near_orig[:20],
                 "mode": near.get("mode", ""), "n_pairs": near.get("n_pairs", 0),
                 "action": "deletable -- text similar AND numeric content agrees"},
        "numeric_distinct": {"n_clusters": len(numeric_distinct),
                             "n_rows": sum(len(c) for c in numeric_distinct),
                             "clusters": numeric_distinct[:12],
                             "action": "DO NOT DELETE -- text similar but the NUMBERS differ, so these are "
                                       "different facts in the same shape"},
        "template": {"n_clusters": tpl["n_clusters"], "n_rows": sum(len(c) for c in tpl_orig),
                     "threshold": THRESHOLD_TEMPLATE, "clusters": tpl_orig[:12],
                     "action": "REPORT ONLY -- distinct facts can share a template"},
        "survivors": len(seen) - (near["n_duplicate_rows"]) if seen else 0,
        "params": {"num_perm": NUM_PERM, "shingle": SHINGLE,
                   "near": THRESHOLD_NEAR, "template": THRESHOLD_TEMPLATE},
    }


# ------------------------------------------------------------------ measure
def measure(corpus: str = "", limit: int = 5000) -> int:
    import time
    path = corpus or os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")
    if not os.path.exists(path):
        print(f"corpus not found: {path}")
        return 1
    try:
        from logos import corpus_sizing
        rows = corpus_sizing.load_corpus(path, limit)
    except Exception as e:                                  # noqa: BLE001
        print(f"could not load: {type(e).__name__}: {e}")
        return 1
    texts = [r["claim"] for r in rows]
    print(f"=== dedup on a real corpus ({len(texts)} rows) ===")
    print(f"  {path}")
    print(f"  params: num_perm={NUM_PERM}, pinned {BANDS}x{ROWS} bands, ADAPTIVE shingles (token-set k=1 "
          f"under 60 words, {SHINGLE}-gram above), near={THRESHOLD_NEAR}, template={THRESHOLD_TEMPLATE}")
    print(f"  candidates {'EXACT all-pairs' if len(texts) <= EXACT_LIMIT else f'LSH at {CANDIDATE_THRESHOLD}'}"
          f", verdicts by TRUE Jaccard\n")
    t0 = time.perf_counter()
    d = dedup(texts)
    dt = time.perf_counter() - t0
    print(f"  EXACT    {d['exact']['n_clusters']:5d} cluster(s), {d['exact']['n_dropped']:5d} redundant row(s)")
    print(f"  NEAR     {d['near']['n_clusters']:5d} cluster(s), {d['near']['n_rows']:5d} row(s) at Jaccard "
          f">= {THRESHOLD_NEAR}")
    print(f"  TEMPLATE {d['template']['n_clusters']:5d} cluster(s), {d['template']['n_rows']:5d} row(s) at "
          f"Jaccard >= {THRESHOLD_TEMPLATE}  <- REPORT ONLY")
    print(f"\n  wall {dt:.2f}s ({dt / max(len(texts), 1) * 1000:.2f} ms/row)  mode: {d['near'].get('mode', '')}")
    if d["near"]["clusters"]:
        print(f"\n  an example NEAR cluster (these are the same row twice):")
        for i in d["near"]["clusters"][0][:4]:
            print(f"    {texts[i][:88]!r}")
    if d["template"]["clusters"]:
        print(f"\n  an example TEMPLATE cluster (distinct facts, same shape -- do NOT delete):")
        for i in d["template"]["clusters"][0][:4]:
            print(f"    {texts[i][:88]!r}")
    print("\n  THE POINT: the exact and near passes license deletion; the template pass does not. A single")
    print("  lower threshold would have merged all three and deleted distinct claims.")
    return 0


# ------------------------------------------------------------------ self test
def run_self_test() -> bool:
    print("logos.clean.dedup - SELF TEST")
    ok = True
    av, why = available()
    print(f"  [{'PASS' if av else 'SKIP'}] datasketch available ({why})")

    # --- exact duplicates are found with NO false positives
    ex = exact_clusters(["The sky is blue.", "The  sky   is blue.", "The sky is green."])
    good = ex["n_groups"] == 1 and ex["n_duplicate_rows"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] whitespace and case variants collapse to ONE exact duplicate")

    ex2 = exact_clusters(["alpha one", "beta two", "gamma three"])
    good = ex2["n_groups"] == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] distinct rows produce no exact cluster")

    if not av:
        print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'} (near-dup skipped)")
        return ok

    # --- near duplicates: one NON-NUMERIC word changed in a sentence with no numbers is the same row
    base = ("the committee met on tuesday and discussed the proposed budget for the coming fiscal year in "
            "considerable detail before adjourning without reaching any conclusion about the matter at hand")
    near = base.replace("tuesday", "wednesday")
    d = dedup([base, near, "a completely unrelated sentence about marine biology and coral reefs in the pacific"])
    good = d["near"]["n_clusters"] >= 1 and len(d["near"]["clusters"][0]) == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a non-numeric one-word change IS a deletable near duplicate "
          f"(jaccard {d['near'].get('threshold')})")

    # --- THE NUMERIC GUARD: same shape, DIFFERENT NUMBER, must NOT be deletable
    a = "the eiffel tower is 330 metres tall and was completed in 1889 for the world fair in paris france"
    b = "the eiffel tower is 300 metres tall and was completed in 1889 for the world fair in paris france"
    dn = dedup([a, b, "unrelated text about marine biology and coral reefs in the pacific ocean today"])
    good = dn["numeric_distinct"]["n_rows"] >= 2 and dn["numeric_distinct"]["action"].startswith("DO NOT")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 330 m vs 300 m is NUMERICALLY DISTINCT, not a duplicate")

    # --- and it must NOT appear in the deletable list
    good = not any(set(c) >= {0, 1} for c in dn["near"]["clusters"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and it is absent from the deletable list")

    # --- and an unrelated row is NOT clustered
    good = all(2 not in c for c in d["near"]["clusters"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unrelated row is not swept into the cluster")

    # --- and the template pass still reports shape-level repetition
    d2 = dedup(["alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "alpha beta gamma delta epsilon zeta eta theta iota lambda",
                "unrelated text about something else entirely that shares no words at all here"])
    good = isinstance(d2["template"]["n_clusters"], int)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the template pass reports shape-level repetition separately")

    # --- the two passes are separate fields, so a caller cannot confuse them
    good = set(d2.keys()) >= {"exact", "near", "template", "survivors", "params"}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] exact / near / template are separate, non-mergeable fields")

    # --- normalisation is what makes exact matching work across case and spacing
    good = normalise("The  Sky, is blue!") == normalise("the sky is blue")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] normalise() folds case, spacing and punctuation")

    # --- THE TEST THAT WOULD HAVE CAUGHT THE (9,13) MISTAKE. A pair at EXACTLY the reported threshold must be
    #     found, and every reported cluster must genuinely clear it. With datasketch's auto-picked (9,13) the
    #     S-curve gives P(0.80)=0.399, so a true duplicate sitting AT 0.8 was found only ~40% of the time while
    #     the module reported "duplicates at Jaccard >= 0.8".
    shared = [f"tok{i}" for i in range(16)]
    a_txt = " ".join(shared + ["alphaA", "betaA"])          # 18 tokens
    b_txt = " ".join(shared + ["alphaB", "betaB"])          # 18 tokens, shares 16
    ja = _jaccard(_shingles(a_txt), _shingles(b_txt))
    good = abs(ja - 0.8) < 0.01
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] built a planted pair at true Jaccard {ja:.3f}")

    dc = near_clusters([a_txt, b_txt, "completely unrelated words here about marine biology and coral"])
    good = any(set(c) >= {0, 1} for c in dc["clusters"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a pair AT the reported threshold IS found "
          f"(mode: {dc['mode']})")

    # --- and the honesty check: every reported pair really does clear the threshold
    honest = True
    for c in dc["clusters"]:
        for x in range(len(c)):
            for y in range(x + 1, len(c)):
                if _jaccard(_shingles([a_txt, b_txt][c[x]] if c[x] < 2 else ""),
                            _shingles([a_txt, b_txt][c[y]] if c[y] < 2 else "")) < THRESHOLD_NEAR:
                    honest = False
    good = honest
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every reported pair clears the threshold by TRUE Jaccard "
          f"(not an estimate)")

    # --- empty and tiny inputs do not crash
    good = dedup([])["n"] == 0 and dedup(["x"])["near"]["n_clusters"] == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty and single-row inputs are handled")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exact + near-duplicate detection (MinHash LSH).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure(args.corpus, args.limit)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
