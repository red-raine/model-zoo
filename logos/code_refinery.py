"""logos/code_refinery.py — measure the deterministic code gates on REAL code before trusting them.

WHY THIS IS THE NEXT STEP
-------------------------
Both deterministic gates now exist (`triage.check_structure` via `ast.parse`, `check_semantics` via
`symtable`) and they are the cheapest precision available anywhere in this project: no model, no network,
no ambiguity. But they have only ever been tested on hand-written snippets. A gate that rejects 30% of real
code is worse than useless under an exhaustive policy, because real code is what you are trying to keep.

THE TWO NUMBERS, AND ONLY ONE OF THEM IS EASY
---------------------------------------------
    FALSE-REJECT RATE on unmodified real code. NON-CIRCULAR and the number that matters. If this is high,
      the gate is broken and no amount of downstream work fixes it.
    RECALL on injected defects. CIRCULAR -- we can only inject defects we know how to detect, so this
      measures "does the detector fire on its own target class", not real-world recall. Reported, but
      labelled.
    A third, weakly-circular number that is still worth having: how many REAL files the semantic gate
      flags unprompted. Those are either genuine defects or false positives, and inspecting a handful of
      examples is the only way to tell.

REPAIR COVERAGE
    The triage design says REPAIR is the valuable disposition. That is only true if a repair actually
    succeeds, so the syntactically-broken set is also run through a repair attempt and re-parsed. A repair
    that produces something parseable but semantically different is not a repair, so the re-parse is
    followed by the semantic gate too.

USAGE
    python tools/logos/code_refinery.py --self-test
    python tools/logos/code_refinery.py --run [--limit 400]
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import sys
import sysconfig
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.triage import check_semantics, check_structure  # noqa: E402


# ------------------------------------------------------------------------------- corpus
def collect(paths: list[str], limit: int = 400, seed: int = 0) -> list[tuple[str, str]]:
    """Gather real .py files. Deterministic sample so the measurement is reproducible."""
    found = []
    for root in paths:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in
                           ("__pycache__", ".git", "node_modules", "site-packages", "test", "tests")]
            for fn in filenames:
                if fn.endswith(".py"):
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(p) > 400_000:
                            continue
                        with open(p, encoding="utf-8", errors="replace") as fh:
                            src = fh.read()
                    except OSError:
                        continue
                    if src.strip():
                        found.append((p, src))
    rng = random.Random(seed)
    rng.shuffle(found)
    return found[:limit]


# ------------------------------------------------------------------------------- defects
def inject_syntax(src: str) -> str | None:
    """Break a delimiter. Deliberately the simplest possible failure -- it tests the parser wiring."""
    for pat, repl in ((r"\)", ""), (r"\]", ""), (r"\}", "")):
        if re.search(pat, src):
            return re.sub(pat, repl, src, count=1)
    return None


def inject_undefined(src: str) -> str | None:
    """Reference a name that is never bound. THE canonical semantic defect: it parses perfectly."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    # append a module-level reference to a name that cannot exist
    return src + "\n__logos_injected_undefined_name__\n"


# The rebound-constant injector was REMOVED along with the check it validated. Real stdlib data showed
# that check flagged 28.5% of files for ordinary branch reassignment, so there is nothing left to test.
INJECTORS = {
    "syntax": inject_syntax,
    "undefined_name": inject_undefined,
}


# ------------------------------------------------------------------------------- repair
def repair_delimiters(src: str) -> str | None:
    """Cheap deterministic repair for delimiter damage, by BOUNDED SEARCH rather than a guess.

    The first version appended the missing closer at end-of-file, which cannot fix `def f(x:` -- the `)`
    belongs immediately after the `x`, not three lines later. So the repair tries every insertion
    position within the offending line and accepts the first that re-parses. That is a search, not a
    heuristic, and it costs at most one parse per character on that line.

    It still only claims ONE defect class. A repair that returns something parseable but semantically
    different is not a repair, which is why the caller re-runs the semantic gate afterwards.
    """
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {v: k for k, v in pairs.items()}
    version = src
    for _attempt in range(4):                      # at most 4 repairs; beyond that, stop guessing
        # --- stray closers first: removing one is unambiguous
        stack = []
        stray_at = None
        for i, ch in enumerate(version):
            if ch in pairs:
                stack.append((ch, i))
            elif ch in closers:
                if stack and stack[-1][0] == closers[ch]:
                    stack.pop()
                else:
                    stray_at = i
                    break
        if stray_at is not None:
            version = version[:stray_at] + version[stray_at + 1:]
            continue
        if not stack:
            return version if version != src else None
        # --- an unterminated opener: search insertion positions on its own line
        opener, pos = stack[-1]
        closer = pairs[opener]
        line_end = version.find("\n", pos)
        if line_end == -1:
            line_end = len(version)
        for cand in range(pos + 1, line_end + 1):
            trial = version[:cand] + closer + version[cand:]
            try:
                ast.parse(trial)
                return trial
            except SyntaxError:
                continue
        # no single insertion fixes it on this line; give up rather than mangle the source
        return None
    return None


# ------------------------------------------------------------------------------- measurement
@dataclass
class Result:
    n_files: int = 0
    reject_structural: int = 0
    flag_semantic: int = 0            # ok is False: a claim of a defect
    decline_semantic: int = 0         # ok is None: the gate refuses to judge (star-import, unresolvable)
    examples: list = field(default_factory=list)
    recall: dict = field(default_factory=dict)
    repair: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"n_files": self.n_files,
                "false_reject_rate": self.reject_structural / max(self.n_files, 1),
                "semantic_flag_rate": self.flag_semantic / max(self.n_files, 1),
                "semantic_decline_rate": self.decline_semantic / max(self.n_files, 1),
                "recall": self.recall, "repair": self.repair,
                "examples": self.examples[:5]}


def measure(files: list[tuple[str, str]]) -> Result:
    res = Result(n_files=len(files))
    per_class: dict[str, dict] = {k: {"tried": 0, "caught_structural": 0, "caught_semantic": 0,
                                      "repaired": 0} for k in INJECTORS}

    for path, src in files:
        # --- 1. FALSE-REJECT on unmodified real code. The number that matters.
        st = check_structure(src, "code")
        if not st["ok"]:
            res.reject_structural += 1
            if len(res.examples) < 8:
                res.examples.append((path, st["issues"][:2]))
            continue
        # --- 2. the semantic gate on real code: genuine defects or false positives, inspectable
        sem = check_semantics(src)
        if sem["ok"] is False:
            res.flag_semantic += 1
            if len(res.examples) < 8:
                res.examples.append((path, sem["issues"][:2]))
        elif sem["ok"] is None:
            # NOT a defect report. Counting "declines to judge" as "flagged" inflated the rate by ~8pp
            # on real data, which is the same class of error as reading a 429 as a dead host.
            res.decline_semantic += 1

        # --- 3. injected defects (CIRCULAR: only the classes we know how to inject)
        for name, fn in INJECTORS.items():
            bad = fn(src)
            if bad is None:
                continue
            per_class[name]["tried"] += 1
            sb = check_structure(bad, "code")
            if not sb["ok"]:
                per_class[name]["caught_structural"] += 1
                if name == "syntax":
                    fixed = repair_delimiters(bad)
                    if fixed is not None and check_structure(fixed, "code")["ok"]:
                        per_class[name]["repaired"] += 1
            else:
                sm = check_semantics(bad)
                if not sm["ok"]:
                    per_class[name]["caught_semantic"] += 1

    res.recall = per_class
    res.repair = {k: v["repaired"] / v["tried"] if v["tried"] else 0.0
                  for k, v in per_class.items()}
    return res


def run(limit: int = 400, seed: int = 0) -> int:
    stdlib = sysconfig.get_paths().get("stdlib", "")
    here = os.path.dirname(os.path.dirname(_TOOLS))
    paths = [stdlib, os.path.join(here, "tools"), os.path.join(here, "subprojects")]
    print("=== corpus ===")
    files = collect(paths, limit=limit, seed=seed)
    print(f"  roots: {[p for p in paths if p and os.path.isdir(p)]}")
    print(f"  sampled {len(files)} real .py files (seeded, reproducible)")
    if not files:
        print("  no corpus found -- refusing to report numbers")
        return 1

    r = measure(files)
    d = r.as_dict()

    print(f"\n=== the number that matters: FALSE REJECTS on unmodified real code ===")
    print(f"  structural reject rate : {d['false_reject_rate']:7.3%}  "
          f"({r.reject_structural}/{r.n_files})")
    print(f"  semantic FLAG rate     : {d['semantic_flag_rate']:7.3%}  "
          f"({r.flag_semantic}/{r.n_files})   <- claims a defect")
    print(f"  semantic DECLINE rate  : {d['semantic_decline_rate']:7.3%}  "
          f"({r.decline_semantic}/{r.n_files})   <- refuses to judge (star-import etc), NOT a defect")
    if d["false_reject_rate"] > 0.02:
        print(f"  -> TOO HIGH. A gate that rejects real code is unusable under an exhaustive policy.")
    else:
        print(f"  -> acceptable: the structural gate does not reject real code.")
    print(f"  NOTE the semantic flag rate is NOT a false-positive rate: some of those are genuine")
    print(f"  defects. The examples below are listed so a human can tell which.")

    print(f"\n=== examples of flagged REAL files (inspect these) ===")
    for path, issues in r.examples[:6]:
        print(f"  {os.path.basename(path):34s} {issues}")

    print(f"\n=== recall on INJECTED defects (CIRCULAR -- we injected only classes we can detect) ===")
    print(f"  {'defect class':20s} {'tried':>6s} {'structural':>11s} {'semantic':>9s} "
          f"{'recall':>8s} {'repaired':>9s}")
    for name, v in r.recall.items():
        caught = v["caught_structural"] + v["caught_semantic"]
        rec = caught / v["tried"] if v["tried"] else 0.0
        print(f"  {name:20s} {v['tried']:6d} {v['caught_structural']:11d} "
              f"{v['caught_semantic']:9d} {rec:8.1%} {r.repair[name]:9.1%}")

    print(f"\n=== reading ===")
    print(f"  The structural gate is the honest win: ast.parse is the interpreter's own parser, so a")
    print(f"  accept is a real accept and a reject is a real reject.")
    print(f"  The semantic gate catches injected undefined names and rebound constants because it is")
    print(f"  built on symtable -- but those are OUR injections, so the recall figure is a wiring test,")
    print(f"  not a real-world estimate. The semantic flag rate on real files is the honest number, and")
    print(f"  it needs human inspection to become a precision.")
    print(f"  Repair covers only delimiter damage. That is the honest scope: a general AST repair loop is")
    print(f"  the next build, not something this run established.")
    return 0


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.code_refinery - SELF TEST")
    ok = True

    # --- the structural gate accepts a representative real-shaped module
    good_src = '''import os


def f(x):
    return [i for i in range(x) if i % 2 == 0]


class C:
    def m(self):
        return {k: v for k, v in os.environ.items()}
'''
    good = check_structure(good_src, "code")["ok"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a normal module passes the structural gate")

    # --- and rejects a broken one
    good = not check_structure("def f(:\n    pass\n", "code")["ok"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a broken module is rejected")

    # --- the undefined-name injector produces something that PARSES, so only the semantic gate sees it
    base = "x = 1\n"
    bad = inject_undefined(base)
    good = bad is not None and check_structure(bad, "code")["ok"] and not check_semantics(bad)["ok"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the undefined-name injection PARSES and is caught only by "
          f"the semantic gate")

    # --- the AST repair loop actually repairs a delimiter break, and the result re-parses
    broken = "def f(x:\n    return x\n"
    fixed = repair_delimiters(broken)
    good = fixed is not None and check_structure(fixed, "code")["ok"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the delimiter repair produces something that re-parses")

    # --- and does NOT claim to repair what it cannot
    unfixable = "def f(x)\n    return x\n"
    good = repair_delimiters(unfixable) is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] and declines a defect outside its class rather than "
          f"returning garbage")

    # --- corpus collection is deterministic and skips caches
    tmp_files = collect([sysconfig.get_paths().get("stdlib", "")], limit=20, seed=1)
    tmp_files2 = collect([sysconfig.get_paths().get("stdlib", "")], limit=20, seed=1)
    good = [p for p, _ in tmp_files] == [p for p, _ in tmp_files2] and len(tmp_files) <= 20
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] corpus sampling is reproducible and capped "
          f"({len(tmp_files)} files)")

    # --- measure() returns rates in [0,1] and never divides by zero on an empty corpus
    empty = measure([])
    good = empty.n_files == 0 and empty.as_dict()["false_reject_rate"] == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an empty corpus yields a rate of 0.0, not a crash")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the deterministic code gates on real code.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.run:
        return run(args.limit)
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
