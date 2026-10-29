"""logos/verify/regression.py — the LOST_CORRECT replay gate. The only quality signal we own outright.

THE PROBLEM IT SOLVES
---------------------
This project has no held-out labelled set, so it cannot report accuracy. It can still answer a different and
immediately useful question: **did my change make anything worse?**

That question needs no human labels, only a record of what the pipeline said last time. So: archive every
verdict, and before any change to a checker, replay the archive through the new code and fail if a row that
was KEEP becomes anything else.

    archive(corpus) -> {row_id, text, verdict, refuter, altered, findings[]}
    replay(archive) -> LOST_KEEP / GAINED_KEEP / CHANGED / UNCHANGED, and a nonzero exit on any lost KEEP

THE EXPLOIT THIS GATE MUST RESIST
---------------------------------
A gate phrased only as "no row that was KEEP may degrade" is passed trivially by a pipeline that keeps
nothing. Escalate everything and `lost_keep = 0` forever. So the gate ALSO requires
`keep_count_new >= keep_count_old`: coverage may not fall. Both conditions together are the actual test, and
the second one is the one that is easy to forget.

Caveat stated up front, because it bounds the claim: this measures NON-REGRESSION only. A suite that is
100% consistently wrong scores perfectly. It is a ratchet, not a correctness proof.

WHY THE ROW TEXT IS ARCHIVED, NOT A POINTER TO IT
-------------------------------------------------
Comparing against a corpus file means a changed corpus reads as a thousand regressions. The archived text is
the frozen input, so a replay measures the CODE and nothing else. Its sha256 is stored for integrity.

VERSIONS ARE PINNED INTO THE MANIFEST
-------------------------------------
sympy, z3 and python versions go in the archive. A sympy upgrade that changes a simplification is a real
event and should be visible as one, not as 10,000 mysterious regressions.

USAGE
    python tools/logos/verify/regression.py --build      # create/refresh the baseline
    python tools/logos/verify/regression.py --replay     # compare current code against it
    python tools/logos/verify/regression.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

_REPO_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_verdicts.json")
ARCHIVE = os.environ.get(
    "LOGOS_ARCHIVE",
    # the REPO copy wins: a baseline that only exists in ~/.cache cannot be run by CI, and a gate only one
    # machine can execute is half a gate.
    _REPO_BASELINE if os.path.exists(_REPO_BASELINE)
    else os.path.join(os.path.expanduser("~"), ".cache", "logos_regression.json"))
ARCHIVE_VERSION = 2


def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ("sympy", "z3", "numpy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "get_version_string", lambda: getattr(m, "__version__", "?"))()
        except Exception:                                     # noqa: BLE001
            v[mod] = "missing"
        if not isinstance(v[mod], str):
            v[mod] = str(v[mod])
    return v


def _row_id(text: str, tag: str) -> str:
    h = hashlib.sha256((tag + "\x00" + text).encode("utf-8")).hexdigest()[:16]
    return f"{tag}:{h}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ------------------------------------------------------------------------------- corpus sources
def default_rows() -> list:
    """The rows this gate guards, with tags.

    Two populations on purpose:
      fixture  the decidable rows, which are the ones that can actually reach KEEP. Without these the gate
               has no KEEPs to protect and would pass vacuously.
      corpus   real prose claims, which cannot reach KEEP. These guard against the opposite failure -- a
               change that starts DROPping curated data.
    """
    out = []
    try:
        from logos import pipeline
        for i, (row, _expect, kind) in enumerate(pipeline.FIXTURE):
            out.append({"tag": "fixture", "text": row, "kind": kind, "label": _expect})
    except Exception:                                         # noqa: BLE001
        pass
    corpus = os.environ.get("LOGOS_CORPUS", r"B:\datasets\forge\verifier-av\dev.jsonl")
    limit = int(os.environ.get("LOGOS_ARCHIVE_CORPUS_ROWS", "60"))
    if os.path.exists(corpus):
        try:
            from logos import corpus_sizing
            for r in corpus_sizing.load_corpus(corpus, limit):
                out.append({"tag": "corpus", "text": r["claim"], "kind": "text", "label": r.get("gold")})
        except Exception:                                     # noqa: BLE001
            pass
    return out


def evaluate_all(rows: list) -> list:
    """Run the pipeline over each row and record everything needed to explain a verdict later."""
    from logos import pipeline
    recs = []
    for r in rows:
        text = r["text"]
        try:
            out = pipeline.evaluate_row(text, kind=r.get("kind"))
            recs.append({
                "id": _row_id(text, r.get("tag", "row")),
                "tag": r.get("tag", "row"),
                "sha": _sha(text),
                "text": text,
                "label": r.get("label"),
                "verdict": out["disposition"],
                "refuter": out.get("refuted_by") or "",
                "altered": bool(out.get("altered")),
                "findings": [{"check": f.get("check"), "verdict": f.get("verdict")}
                             for f in out.get("findings", [])],
                "why": out.get("why", [])[:6],
            })
        except Exception as e:                                # noqa: BLE001
            recs.append({"id": _row_id(text, r.get("tag", "row")), "tag": r.get("tag", "row"),
                         "sha": _sha(text), "text": text, "label": r.get("label"),
                         "verdict": "ERROR", "error": f"{type(e).__name__}: {e}"[:200],
                         "refuter": "", "altered": False, "findings": [], "why": []})
    return recs


# ------------------------------------------------------------------------------- archive / replay
def build(archive_path: str = ARCHIVE, rows: list | None = None, force: bool = False) -> dict:
    rows = rows if rows is not None else default_rows()
    recs = evaluate_all(rows)

    # ---- THE RATCHET. A rebuild must not be a quiet way to lower the bar.
    if os.path.exists(archive_path) and not force:
        try:
            prev = load(archive_path)
            prev_keeps = sum(1 for r in prev.get("records", []) if r.get("verdict") == "KEEP")
            new_keeps = sum(1 for r in recs if r["verdict"] == "KEEP")
            if new_keeps < prev_keeps:
                return {"status": "REFUSED",
                        "reason": f"this baseline would lower the KEEP count {prev_keeps} -> {new_keeps}. "
                                  f"Rebuilding the archive is how a regression gate gets escaped: the new "
                                  f"baseline simply agrees with the broken code. Fix the regression, or "
                                  f"pass --force and say why.",
                        "prev_keeps": prev_keeps, "new_keeps": new_keeps}
        except Exception:                                     # noqa: BLE001
            pass
    counts: dict = {}
    for r in recs:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    payload = {"archive_version": ARCHIVE_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "versions": _versions(), "counts": counts, "n": len(recs), "records": recs}
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    with open(archive_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return {"status": "BUILT", "path": archive_path, "n": len(recs), "counts": counts}


def load(archive_path: str = ARCHIVE) -> dict:
    with open(archive_path, encoding="utf-8") as fh:
        return json.load(fh)


def replay(archive_path: str = ARCHIVE, rows: list | None = None) -> dict:
    """Compare the current code against the archive.

    THE GATE HAS TWO CONDITIONS, not one:
        lost_keep == 0                  nothing that was KEEP degraded
        keep_count_new >= keep_count_old  coverage did not shrink
    The second is what stops "escalate everything" from passing. Both are reported separately so a failure
    says which one broke.
    """
    if not os.path.exists(archive_path):
        return {"status": "NO_ARCHIVE", "reason": f"no archive at {archive_path}; run --build first",
                "passed": False}
    old = load(archive_path)
    if old.get("archive_version") != ARCHIVE_VERSION:
        return {"status": "VERSION_MISMATCH", "passed": False,
                "reason": f"archive version {old.get('archive_version')} != {ARCHIVE_VERSION}; rebuild it"}

    # replay the ARCHIVED TEXT, not the corpus on disk: a changed corpus must not read as code regressions
    rows = rows if rows is not None else [
        {"tag": r.get("tag", "row"), "text": r["text"], "kind": "text", "label": r.get("label")}
        for r in old["records"]]
    new = evaluate_all(rows)
    new_by_id = {r["id"]: r for r in new}

    lost_keep, gained_keep, changed, unchanged, missing = [], [], [], [], []
    for r in old["records"]:
        n = new_by_id.get(r["id"])
        if n is None:
            missing.append(r["id"])
            continue
        was, now = r["verdict"], n["verdict"]
        if was == now:
            unchanged.append(r["id"])
        elif was == "KEEP":
            lost_keep.append({"id": r["id"], "text": r["text"][:120], "label": r.get("label"),
                              "was": was, "now": now, "refuter": n.get("refuter"),
                              "why": n.get("why", [])[:3]})
        elif now == "KEEP":
            gained_keep.append({"id": r["id"], "text": r["text"][:120], "was": was, "now": now})
        else:
            changed.append({"id": r["id"], "text": r["text"][:120], "was": was, "now": now})

    old_keeps = sum(1 for r in old["records"] if r["verdict"] == "KEEP")
    new_keeps = sum(1 for r in new if r["verdict"] == "KEEP")
    version_drift = {k: (old.get("versions", {}).get(k), v)
                     for k, v in _versions().items()
                     if old.get("versions", {}).get(k) not in (None, v)}

    passed = (not lost_keep) and (new_keeps >= old_keeps) and not missing
    return {"status": "PASSED" if passed else "FAILED", "passed": passed,
            "n_old": len(old["records"]), "n_new": len(new),
            "lost_keep": lost_keep, "gained_keep": gained_keep, "changed": changed,
            "missing": missing, "n_unchanged": len(unchanged),
            "keeps_old": old_keeps, "keeps_new": new_keeps,
            "version_drift": version_drift,
            "gate_reasons": (
                [] if passed else
                [r for r in (
                    f"LOST_KEEP: {len(lost_keep)} row(s) that were KEEP are no longer KEEP" if lost_keep else "",
                    f"COVERAGE_SHRANK: keep count fell {old_keeps} -> {new_keeps}" if new_keeps < old_keeps else "",
                    f"ROWS_MISSING: {len(missing)} archived row(s) could not be re-evaluated" if missing else "",
                ) if r])}


# ------------------------------------------------------------------------------- display
def report(res: dict, archive_path: str = ARCHIVE) -> None:
    print("=== LOST_CORRECT replay gate ===")
    print(f"  archive {archive_path}")
    st = res.get("status")
    print(f"  status  {st}")
    if st in ("NO_ARCHIVE", "VERSION_MISMATCH"):
        print(f"  {res.get('reason')}")
        return
    old = load(archive_path)
    print(f"  versions then: {old.get('versions')}")
    print(f"  versions now : {_versions()}")
    if res.get("version_drift"):
        print(f"  VERSION DRIFT: {res['version_drift']}  (a change here can look like a regression -- "
              f"read the lost rows before blaming the code)")
    print(f"  rows {res['n_old']} archived / {res['n_new']} replayed, {res['n_unchanged']} unchanged")
    print(f"  KEEP count {res['keeps_old']} -> {res['keeps_new']}")
    print()
    if res["lost_keep"]:
        print(f"  LOST KEEP ({len(res['lost_keep'])}) -- these are the failures:")
        for r in res["lost_keep"][:12]:
            print(f"    {r['id']}  {r['was']} -> {r['now']}  refuter={r['refuter']!r}")
            print(f"      {r['text']!r}")
            for w in r["why"]:
                print(f"        - {w[:110]}")
    if res["gained_keep"]:
        print(f"  gained KEEP ({len(res['gained_keep'])}) -- coverage improved; check these are CORRECT, "
              f"not merely more permissive:")
        for r in res["gained_keep"][:8]:
            print(f"    {r['id']}  {r['was']} -> {r['now']}  {r['text']!r}")
    if res["changed"]:
        print(f"  other verdict changes: {len(res['changed'])}")
        for r in res["changed"][:8]:
            print(f"    {r['id']}  {r['was']} -> {r['now']}  {r['text']!r}")
    if res["missing"]:
        print(f"  MISSING from replay: {len(res['missing'])}")
    print()
    if res["passed"]:
        print("  GATE PASSED: no KEEP was lost and coverage did not shrink.")
        print("  Scope: this is NON-REGRESSION, not correctness. A suite that is consistently wrong passes.")
    else:
        print("  GATE FAILED:")
        for r in res.get("gate_reasons", []):
            print(f"    - {r}")


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.verify.regression - SELF TEST")
    ok = True

    # --- a fixed baseline we control, so the gate's logic is tested without depending on current verdicts
    import tempfile
    rows = [{"tag": "t", "kind": "text", "text": "2 + 2 = 4", "label": "KEEP"},
            {"tag": "t", "kind": "text", "text": "2 + 2 = 5", "label": "REPAIR"},
            {"tag": "t", "kind": "text", "text": "The sky is blue.", "label": "ESCALATE"}]
    tmp = tempfile.mkdtemp(prefix="logos_reg_")
    path = os.path.join(tmp, "arch.json")
    b = build(path, rows)
    good = b["status"] == "BUILT" and b["n"] == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a baseline can be built ({b['counts']})")

    # --- replaying unchanged code must pass
    r = replay(path)
    good = r["status"] == "PASSED" and r["keeps_old"] == r["keeps_new"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] replaying unchanged code PASSES "
          f"(keeps {r['keeps_old']} -> {r['keeps_new']})")

    # --- a KEEP that degrades must FAIL the gate. simulate it by tampering with the archive.
    old = load(path)
    for rec in old["records"]:
        if rec["text"] == "2 + 2 = 4":
            rec["verdict"] = "KEEP"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(old, fh)
    # now make the current code disagree by archiving a row the code will not KEEP
    old["records"][0]["text"] = "2 + 2 = 4"
    old["records"][0]["verdict"] = "KEEP"
    bad = json.loads(json.dumps(old))
    # force a lost keep: pretend the baseline said KEEP for a row that escalates
    for rec in bad["records"]:
        if rec["text"] == "The sky is blue.":
            rec["verdict"] = "KEEP"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    r2 = replay(path)
    good = r2["status"] == "FAILED" and len(r2["lost_keep"]) == 1 and not r2["passed"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a KEEP that degrades FAILS the gate "
          f"({len(r2['lost_keep']) if r2.get('lost_keep') is not None else '?'} lost)")

    # --- THE EXPLOIT, isolated: coverage can shrink while lost_keep stays EMPTY, via a row that cannot be
    #     re-evaluated. Without the coverage condition this would pass.
    exploit = json.loads(json.dumps(old))
    exploit["records"] = [r for r in exploit["records"] if r["verdict"] != "KEEP"]
    exploit["records"].append({"id": "gone:id", "tag": "gone", "sha": _sha("y"),
                               "text": "", "verdict": "KEEP", "label": "KEEP",
                               "refuter": "", "altered": False, "findings": [], "why": []})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(exploit, fh)
    r3 = replay(path)
    good = (r3["passed"] is False) and (not r3["lost_keep"]) and \
           (r3["keeps_new"] < r3["keeps_old"] or r3["missing"])
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] coverage can shrink with NO lost KEEP and still fail "
          f"(lost={len(r3['lost_keep'])}, keeps {r3['keeps_old']}->{r3['keeps_new']}, "
          f"missing={len(r3['missing'])})")

    # --- THE RATCHET: a rebuild that lowers the KEEP count must be refused, because otherwise anyone can
    #     reset the baseline to match broken code and the gate passes forever.
    path2 = os.path.join(tmp, "arch2.json")
    build(path2, rows)                                    # 1 KEEP
    fewer = [{"tag": "t", "kind": "text", "text": "The sky is blue.", "label": "ESCALATE"}]
    refused = build(path2, fewer)                         # 0 KEEPs -> must refuse
    good = refused["status"] == "REFUSED" and "escaped" in refused["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a rebuild that LOWERS the keep count is refused "
          f"(the ratchet cannot be reset)")

    forced = build(path2, fewer, force=True)
    good = forced["status"] == "BUILT"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] --force allows it deliberately")

    # --- an empty archive is a STATE, not a crash, and does not pass
    r4 = replay(os.path.join(tmp, "does_not_exist.json"))
    good = r4["status"] == "NO_ARCHIVE" and r4["passed"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a missing archive reports NO_ARCHIVE and does not pass")

    # --- the row id is stable across runs and distinct across tags
    good = _row_id("x", "a") == _row_id("x", "a") and _row_id("x", "a") != _row_id("x", "b")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] row ids are stable and tag-scoped")

    # --- the gate reports coverage separately from lost KEEPs, so a failure names which condition broke
    good = isinstance(r.get("keeps_old"), int) and isinstance(r.get("keeps_new"), int) and \
        isinstance(r.get("gate_reasons"), list)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both gate conditions are reported independently")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LOST_CORRECT replay gate.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="allow a rebuild that lowers the KEEP count (say why in the commit message)")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.build:
        b = build(args.archive, force=args.force)
        if b["status"] == "REFUSED":
            print(f"REFUSED: {b['reason']}")
            return 1
        print(f"baseline built: {b['n']} rows -> {b['counts']}  ({args.archive})")
        return 0
    if args.replay:
        r = replay(args.archive)
        report(r, args.archive)
        return 0 if r["passed"] else 1
    if args.show:
        a = load(args.archive)
        print(json.dumps({"created": a["created"], "counts": a["counts"],
                          "versions": a["versions"], "n": a["n"]}, indent=2))
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
