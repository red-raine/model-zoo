"""logos/code_exec.py — execution as the only deterministic TRUTH signal for code, done safely.

WHY EXECUTION IS DIFFERENT FROM EVERY OTHER GATE
------------------------------------------------
Syntax (`ast.parse`) proves well-formedness. Consistency (`logic/smt`) proves non-contradiction. Neither
proves the programme does what it claims. **Only running it does**, and that is why execution is the
code refinery's truth layer rather than another semantic check.

THE SAFETY CONSTRAINT, AND WHY IT SHAPES THE DESIGN
---------------------------------------------------
L008: never execute untrusted code on the host. So this module does two separate things and is explicit
about which is which:

    STATIC COVERAGE   how much of a corpus even HAS a runnable truth signal (doctests, test imports).
                      Requires no execution at all and is safe on any corpus.
    EXECUTION GATE    actually runs a snippet. Process-isolated with a hard timeout, isolated interpreter
                      mode, no site packages, no bytecode, minimal environment, throwaway cwd.

**The isolation is process-level, NOT a security boundary.** It stops runaway loops, accidental
filesystem writes into the wrong place, and import side effects. It does NOT stop a determined attacker:
there is no container, no syscall filter, no network namespace. Therefore:

    run this on a CURATED or SELF-GENERATED corpus, or inside a container -- never on arbitrary
    downloaded code on the host.

That sentence is the reason the default is `allow_exec=False`. Failing safe here is the whole point.

WHY DOCTESTS ARE THE HIGH-VALUE TARGET
--------------------------------------
Most real code has no test suite. But a surprising amount ships DOCTESTS, which are a runnable
specification embedded in the source. That is a free truth signal that already exists in the corpus and
needs no test synthesis -- so measuring how often it is present is the first question, and it is a static
question.

USAGE
    python tools/logos/code_exec.py --self-test
    python tools/logos/code_exec.py --coverage [--limit 400]     # static, safe on any corpus
    python tools/logos/code_exec.py --demo                       # runs only built-in curated samples
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

PASSED, FAILED, TIMEOUT, REFUSED = "PASSED", "FAILED", "TIMEOUT", "REFUSED"


# --------------------------------------------------------------------------------- static side
def has_doctest(source: str) -> bool:
    """Does this file ship runnable examples? Static -- executes nothing."""
    return bool(re.search(r"^\s*>>> ", source, re.M))


def has_tests_import(source: str) -> bool:
    """Does it import a test framework? Static. Presence is not proof the tests would pass."""
    return bool(re.search(r"^\s*(import|from)\s+(pytest|unittest|nose|hypothesis)\b", source, re.M))


def static_coverage_profile(source: str) -> dict:
    """Which truth signals exist, determined without running anything."""
    try:
        tree = ast.parse(source)
        parses = True
    except SyntaxError:
        tree, parses = None, False
    funcs, classes = 0, 0
    if tree is not None:
        funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
        classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    return {"parses": parses, "doctest": has_doctest(source), "test_framework": has_tests_import(source),
            "functions": funcs, "classes": classes}


# --------------------------------------------------------------------------------- exec side
@dataclass
class ExecResult:
    status: str = REFUSED
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "seconds": round(self.seconds, 3),
                "stdout": self.stdout[-800:], "stderr": self.stderr[-800:], "reason": self.reason}


MINIMAL_ENV = {
    "PATH": "",                       # no PATH: keeps the child from finding external tools
    "PYTHONHASHSEED": "0",            # determinism
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "NO_PROXY": "*",
    "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": "",
}


def run_snippet(source: str, timeout: float = 5.0, allow_exec: bool = False) -> ExecResult:
    """Execute in a child interpreter, isolated as far as a process can be.

    Refuses by default. `allow_exec=True` is a deliberate decision by the caller, and the docstring above
    states exactly how much protection is actually being offered.
    """
    if not allow_exec:
        return ExecResult(REFUSED, reason="execution requires allow_exec=True; L008 forbids running "
                                          "untrusted code on the host (process isolation is not a "
                                          "security boundary)")
    import time
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", "-c", source],
                capture_output=True, text=True, timeout=timeout, cwd=td, env=MINIMAL_ENV,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(TIMEOUT, reason=f"exceeded {timeout}s", seconds=time.perf_counter() - t0)
        except Exception as e:                                   # noqa: BLE001
            return ExecResult(FAILED, reason=f"could not start child: {type(e).__name__}: {e}",
                              seconds=time.perf_counter() - t0)
        dt = time.perf_counter() - t0
        status = PASSED if proc.returncode == 0 else FAILED
        return ExecResult(status, proc.stdout or "", proc.stderr or "", dt,
                          reason=f"exit {proc.returncode}")


def check_doctests(source: str, timeout: float = 10.0, allow_exec: bool = False) -> ExecResult:
    """Run the module's own doctests. A REAL truth signal that already exists in the corpus."""
    if not allow_exec:
        return ExecResult(REFUSED, reason="allow_exec=False")
    if not has_doctest(source):
        return ExecResult(REFUSED, reason="no doctests present")
    driver = ('import doctest, sys\n'
              'import __main__ as m\n'
              'failures, tests = doctest.testmod(m, verbose=False)\n'
              'sys.exit(0 if failures == 0 and tests > 0 else 1)\n')
    # run the module body first, then its doctests, in the SAME process so module state is real
    combined = source + "\n\n" + driver
    return run_snippet(combined, timeout=timeout, allow_exec=True)


def check_determinism(source: str, call_expr: str, repeats: int = 2,
                      timeout: float = 8.0, allow_exec: bool = False) -> ExecResult:
    """Call the same expression twice and compare. Detects nondeterminism, which makes any 'verified'
    claim about the code unfalsifiable."""
    if not allow_exec:
        return ExecResult(REFUSED, reason="allow_exec=False")
    driver = (f"\nimport json as _j\n"
              f"_r = []\n"
              f"for _ in range({int(repeats)}):\n"
              f"    _r.append(repr({call_expr}))\n"
              f"print(_j.dumps(_r))\n"
              f"raise SystemExit(0 if len(set(_r)) == 1 else 1)\n")
    return run_snippet(source + driver, timeout=timeout, allow_exec=True)


# --------------------------------------------------------------------------------- coverage run
def coverage(limit: int = 400, seed: int = 0) -> int:
    """STATIC measurement over a real corpus. Executes nothing, so it is safe on any input."""
    from logos.code_refinery import collect

    files = collect([sysconfig_stdlib()], limit=limit, seed=seed)
    if not files:
        print("no corpus found")
        return 1
    n = len(files)
    parses = doctest_n = tests_n = funcs = 0
    for _path, src in files:
        prof = static_coverage_profile(src)
        parses += prof["parses"]
        doctest_n += prof["doctest"]
        tests_n += prof["test_framework"]
        funcs += prof["functions"]
    print(f"=== STATIC truth-signal coverage over {n} real files (executes nothing) ===")
    print(f"  parses                 : {parses:5d}  {parses / n:7.1%}")
    print(f"  ships DOCTESTS         : {doctest_n:5d}  {doctest_n / n:7.1%}   <- runnable spec, free")
    print(f"  imports a test lib     : {tests_n:5d}  {tests_n / n:7.1%}   <- presence, not passing")
    print(f"  top-level functions    : {funcs:5d}  ({funcs / n:.1f} per file)")
    print(f"\n  READING: the truth layer can only speak about the {parses / n:.0%} that parses and the")
    print(f"  {doctest_n / n:.0%} that carries a runnable specification. For the rest, execution proves")
    print(f"  nothing because there is nothing to run -- which is why the structural and semantic gates")
    print(f"  carry most of the load in a refinery, and why 'tests pass' is a small top slice, not the")
    print(f"  main filter.")
    return 0


def sysconfig_stdlib() -> str:
    import sysconfig
    return sysconfig.get_paths().get("stdlib", "")


# --------------------------------------------------------------------------------- demo
# Each sample is paired with the gate that is the RIGHT one to judge it. Stating this explicitly matters:
# the first version dumped every sample through `run_snippet` and printed "doctest that FAILS -> PASSED"
# and "nondeterministic -> PASSED", because merely defining a function does not run its doctests or call
# it. A table that implies a failing doctest passed is precisely the kind of misleading output this
# project keeps having to correct.
SNIPPETS = {
    "correct function": ("run", '''
def add(a, b):
    return a + b
print(add(2, 3))
'''),
    "raises at import": ("run", '''
raise RuntimeError("boom")
'''),
    "infinite loop": ("run", '''
while True:
    pass
'''),
}
DOCTEST_CASES = {
    "doctest passes": '''
def double(x):
    """Double it.

    >>> double(3)
    6
    >>> double(0)
    0
    """
    return x * 2
''',
    "doctest FAILS": '''
def double(x):
    """Double it.

    >>> double(3)
    7
    """
    return x * 2
''',
    "no doctests at all": '''
def f():
    return 1
''',
}
DETERMINISM_CASES = {
    "nondeterministic draw()": ("import random\ndef draw():\n    return random.random()\n", "draw()"),
    "deterministic sq(7)": ("def sq(x):\n    return x * x\n", "sq(7)"),
}


def demo() -> int:
    print("=== 1. run_snippet: does it execute at all? ===")
    print("  (built-in samples only -- nothing from the corpus is executed, because process")
    print("   isolation is NOT a security boundary and L008 forbids running untrusted code)")
    print(f"\n  {'sample':24s} {'status':9s} {'s':>6s}  detail")
    for label, (_gate, src) in SNIPPETS.items():
        r = run_snippet(src, timeout=2.0, allow_exec=True)
        print(f"  {label:24s} {r.status:9s} {r.seconds:6.2f}  {r.reason[:52]}")

    print(f"\n=== 2. THE DOCTEST GATE, which is the actual truth signal ===")
    print(f"  {'sample':24s} {'status':9s}  detail")
    for label, src in DOCTEST_CASES.items():
        r = check_doctests(src, allow_exec=True)
        print(f"  {label:24s} {r.status:9s}  {r.reason[:60]}")
    print(f"  -> a FAILING doctest is caught here and nowhere else: ast.parse accepts it, symtable")
    print(f"     accepts it, and only execution notices that 3*2 is not 7.")

    print(f"\n=== 3. THE DETERMINISM GATE ===")
    print(f"  {'sample':24s} {'status':9s}  detail")
    for label, (src, call) in DETERMINISM_CASES.items():
        r = check_determinism(src, call, repeats=3, allow_exec=True)
        print(f"  {label:24s} {r.status:9s}  {r.reason[:60]}")
    print(f"  -> nondeterminism makes any 'verified' claim unfalsifiable, so it is worth detecting")
    print(f"     even when the code is correct.")

    print(f"\n=== 4. the DEFAULT, which is to refuse ===")
    r = run_snippet(SNIPPETS["correct function"][1])
    print(f"  {r.status:9s}  {r.reason[:96]}")
    return 0


# --------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.code_exec - SELF TEST")
    ok = True

    # --- SAFETY FIRST: the gate refuses by default. This is the L008 rule expressed as a default.
    r = run_snippet("print('hi')")
    good = r.status == REFUSED and "L008" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] REFUSES to execute unless allow_exec=True (L008)")

    # --- a correct snippet passes
    r = run_snippet("print(6 * 7)", allow_exec=True)
    good = r.status == PASSED and "42" in r.stdout
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a correct snippet PASSES and its output is captured")

    # --- a raising snippet fails with stderr preserved
    r = run_snippet("raise ValueError('nope')", allow_exec=True)
    good = r.status == FAILED and "ValueError" in r.stderr
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a raising snippet FAILS and stderr is preserved")

    # --- an infinite loop is killed, and killed FAST: a gate that hangs is not a gate
    r = run_snippet("while True: pass", timeout=1.0, allow_exec=True)
    good = r.status == TIMEOUT and r.seconds < 5.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an infinite loop TIMEOUTs in {r.seconds:.1f}s, not "
          f"forever")

    # --- the child cannot see the host's environment (no PATH, no proxies). Asserted on the PROPERTY,
    #     not on a printed string: my first version compared to "True ''" and failed because an empty
    #     string prints as nothing. The child reported 7 vars with PATH='' -- the isolation was correct,
    #     the assertion was not.
    r = run_snippet("import os; e = dict(os.environ); "
                    "print(len(e), os.environ.get('PATH') == '', "
                    "os.environ.get('HTTP_PROXY') == '')", allow_exec=True)
    good = r.status == PASSED and "True True" in r.stdout
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the child gets a MINIMAL env: PATH present but empty, "
          f"proxies blanked ({r.stdout.strip()})")

    # --- a PASSING doctest is a real truth signal
    r = check_doctests('def d(x):\n    """\n    >>> d(3)\n    6\n    """\n    return x * 2\n',
                       allow_exec=True)
    good = r.status == PASSED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a passing doctest PASSES")

    # --- and a FAILING doctest is caught, which is the whole point of running it
    r = check_doctests('def d(x):\n    """\n    >>> d(3)\n    7\n    """\n    return x * 2\n',
                       allow_exec=True)
    good = r.status == FAILED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a failing doctest FAILS -- execution found it, static "
          f"gates never would")

    # --- a file with no doctests is REFUSED, not failed: absence of a signal is not evidence of a defect
    r = check_doctests("def f():\n    return 1\n", allow_exec=True)
    good = r.status == REFUSED and "no doctests" in r.reason
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no doctests -> REFUSED, not FAILED (absence is not a "
          f"defect)")

    # --- determinism detects nondeterminism, which makes any verification claim unfalsifiable
    r = check_determinism("import random\ndef draw():\n    return random.random()\n", "draw()",
                          repeats=3, allow_exec=True)
    good = r.status == FAILED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a nondeterministic call is caught by the determinism gate")

    # --- and a deterministic one passes
    r = check_determinism("def sq(x):\n    return x * x\n", "sq(7)", repeats=3, allow_exec=True)
    good = r.status == PASSED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a deterministic call passes")

    # --- STATIC profile requires no execution and is correct on the basics
    prof = static_coverage_profile('def f():\n    """\n    >>> f()\n    1\n    """\n    return 1\n')
    good = prof["parses"] and prof["doctest"] and prof["functions"] == 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the static profile finds doctests and functions without "
          f"executing anything")

    # --- and reports a parse failure rather than throwing
    prof = static_coverage_profile("def f(:\n")
    good = prof["parses"] is False
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the static profile reports unparsable input, no exception")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Execution gate: the real truth signal for code.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.coverage:
        return coverage(args.limit)
    if args.demo:
        return demo()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
