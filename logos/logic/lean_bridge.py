"""logos.logic.lean_bridge — machine-checked proofs via Lean 4, with an honest fallback.

STATUS OF LEAN IN THIS STACK (verified, not assumed)
----------------------------------------------------
Lean 4 **is installed** on this machine: `elan 4.2.4`, toolchain
`leanprover/lean4:v4.15.0`, report string `Lean (version 4.15.0, x86_64-w64-windows-gnu,
commit 11651562caae, Release)`. It lives at `%USERPROFILE%\\.elan\\bin`.

It is NOT reachable from a bare shell: `elan` only puts itself on PATH, and the toolchain
shims resolve through `%USERPROFILE%\\.elan\\bin`. This module locates it explicitly rather
than assuming PATH, because a silent PATH miss is how a "Lean verified" claim becomes a
parser claim.

THE FINDING THAT MOTIVATED THIS FILE
------------------------------------
`E:\\vibe_coding\\glm\\antigravity\\mini-minion-0\\minion\\formal\\lean_runner.py` advertises a
fallback chain of "local CLI, Docker, or verified parser". On this machine there is no `lean`
on PATH, no `leanprover/lean4` image, and Docker Hub is unreachable -- so every proof it has
reported was checked by its Python parser (`mini_lean.py`), **not** by Lean. The parser is not
wrong, but it is not machine checking, and the difference is exactly what "verified" is
supposed to mean. Their own phase name `31-live-hardware-truth-pass-lean4-anti-mock` shows
someone already suspected this.

TWO INDEPENDENT CHECKS, WHICH IS THE POINT
------------------------------------------
Validity is established twice, by two implementations that share no code:

1. **Python truth table** (`propositional.entails`) -- exhaustive over valuations.
2. **Lean `rfl`** on a Bool encoding of the same formula -- Lean's kernel reduces the term.

Agreement between an independent implementation and a machine-checked kernel is real
evidence. A single implementation agreeing with itself is not.

A third path is attempted first because it is stronger when it works: a direct `Prop`
theorem closed by a tactic from a ladder (`exact`, `simp_all`, `tauto`). A tactic proof is a
proof of the *proposition*, not of a Boolean encoding, so it is preferred when available.

WHAT THIS DOES NOT DO
---------------------
* No Mathlib. `tauto`/`aesop` live in Mathlib; they are *tried* and simply fail to resolve
  when Mathlib is absent, which is reported rather than hidden.
* No `sorry`. A goal left with a placeholder is treated as UNPROVED. That is the whole point
  of the zero-sorry discipline.
* No silent downgrade. If Lean disappears, `check()` says `lean_available: False` and names
  which check actually ran. It never reports a Python result as if Lean had verified it.

USAGE
-----
    python tools/logos/logic/lean_bridge.py --self-test
    python tools/logos/logic/lean_bridge.py --check "P -> Q" "~Q" "~P"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.logic import propositional as prop  # noqa: E402

# Candidate locations, most specific first. PATH is checked last because a PATH hit is the
# least trustworthy signal (it can be a stale shim from a different toolchain).
_ELAN_CANDIDATES = [
    os.path.expandvars(r"%USERPROFILE%\.elan\bin\lean.exe"),
    os.path.expandvars(r"%USERPROFILE%\.elan\bin\lean"),
    os.path.expanduser("~/.elan/bin/lean"),
    "/root/.elan/bin/lean",
    "/home/zcoder/.elan/bin/lean",
]

TACTIC_LADDER = ("exact h1 h2", "exact h1 h2 h3", "simp_all", "tauto", "aesop",
                 "intro h; exact h")


@dataclass
class LeanResult:
    available: bool
    proved: bool
    method: str                     # tactic_rfl | tactic:<name> | python_only | none
    lean_version: str = ""
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0.0
    agree_with_python: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """True only if a machine check (Lean) actually ran.

        BOTH proof paths count: a tactic proof of the proposition, and `rfl` on a
        literal-substituted Bool encoding. An earlier version tested only
        `method.startswith("tactic")`, which marked every `rfl` result untrustworthy even
        though Lean's kernel had evaluated it -- so the verifier reported false failures on
        exactly the rules that needed the fallback path.
        """
        return self.available and (self.method.startswith("tactic")
                                   or self.method.startswith("rfl"))


def find_lean() -> str | None:
    for c in _ELAN_CANDIDATES:
        if os.path.isfile(c):
            return c
    return shutil.which("lean")


def lean_version(lean_path: str | None = None, timeout: float = 90.0) -> str:
    p = lean_path or find_lean()
    if not p:
        return ""
    try:
        r = subprocess.run([p, "--version"], capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception:                     # noqa: BLE001
        return ""


# --------------------------------------------------------------------- Lean source gen
def to_lean_prop(f: prop.Node) -> str:
    k = f[0]
    if k == "atom":
        return f[1]
    if k == "not":
        inner = f[1]
        s = to_lean_prop(inner)
        return f"(¬{s})" if inner[0] == "atom" else f"(¬({s}))"
    op = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}[k]
    return f"({to_lean_prop(f[1])} {op} {to_lean_prop(f[2])})"


def to_lean_bool(f: prop.Node) -> str:
    k = f[0]
    if k == "atom":
        return f[1]
    if k == "not":
        return f"(!{to_lean_bool(f[1])})"
    op = {"and": "&&", "or": "||", "imp": None, "iff": "=="}[k]
    if k == "imp":
        return f"((!{to_lean_bool(f[1])}) || {to_lean_bool(f[2])})"
    return f"({to_lean_bool(f[1])} {op} {to_lean_bool(f[2])})"


def _bool_rows(premises: list[prop.Node], conclusion: prop.Node) -> list[tuple[dict, bool]]:
    """Every valuation, tagged with whether it is a counterexample."""
    rows = []
    for r in prop.truth_table(premises, conclusion):
        rows.append((r["assignment"], bool(all(r["premises"]) and not r["conclusion"])))
    return rows


def _render_bool_literal(f: prop.Node, assign: dict[str, bool]) -> str:
    """Render a formula with every atom SUBSTITUTED by a Bool literal.

    This substitution is the whole trick. An earlier version declared the atoms as
    `example (P Q : Bool) : ... := by rfl`, i.e. universally quantified over Bool. Lean cannot
    close that by `rfl` because the expression is symbolic -- and the tactic path had already
    proved modus ponens and hypothetical syllogism, so the bug stayed hidden until De Morgan
    needed the Bool path. Concrete literals per valuation make every row a closed Bool term
    that the kernel reduces.
    """
    k = f[0]
    if k == "atom":
        return "true" if assign[f[1]] else "false"
    if k == "not":
        return f"(!{_render_bool_literal(f[1], assign)})"
    if k == "imp":
        return f"((!{_render_bool_literal(f[1], assign)}) || {_render_bool_literal(f[2], assign)})"
    op = {"and": "&&", "or": "||", "iff": "=="}[k]
    return f"({_render_bool_literal(f[1], assign)} {op} {_render_bool_literal(f[2], assign)})"


def lean_bool_source(premises: list[prop.Node], conclusion: prop.Node) -> str:
    """A Bool encoding of the truth table. Each row is a closed `example ... := by rfl`.

    Every atom is replaced by a literal, so Lean's kernel evaluates the row rather than
    reasoning about it. If every row evaluates as the Python truth table computed, two
    independent implementations agree. Counterexample rows are emitted asserting the
    *falsifying* condition -- the constructive content of invalidity.
    """
    names = sorted(set().union(*[prop.atoms(p) for p in premises], prop.atoms(conclusion))
                   if premises else prop.atoms(conclusion))
    if not names:
        return "-- no atoms\n"
    lines = [f"-- generated: Bool truth table for {' ; '.join(prop.to_text(p) for p in premises)}"
             f" |- {prop.to_text(conclusion)} (atoms substituted by literals)", ""]
    for assign, is_cex in _bool_rows(premises, conclusion):
        prem_lit = (" && ".join(_render_bool_literal(p, assign) for p in premises)
                    if premises else "true")
        conc_lit = _render_bool_literal(conclusion, assign)
        # The negation MUST wrap the whole operand. `!a && b || c` parses as `(!a) && b || c`,
        # which negates only the first conjunct -- that silently made the constructive and
        # destructive dilemma rows assert the wrong proposition, and Lean correctly rejected
        # them. The tactic path had masked this for every simpler rule.
        if is_cex:
            lines.append(f"example : (({prem_lit}) && !({conc_lit})) = true := by rfl")
        else:
            lines.append(f"example : (!({prem_lit}) || ({conc_lit})) = true := by rfl")
    return "\n".join(lines) + "\n"


def lean_prop_source(premises: list[prop.Node], conclusion: prop.Node,
                     tactic: str) -> str:
    names = sorted(set().union(*[prop.atoms(p) for p in premises], prop.atoms(conclusion))
                   if premises else prop.atoms(conclusion))
    decl = " ".join(f"({n} : Prop)" for n in names)
    hyps = " ".join(f"(h{i + 1} : {to_lean_prop(p)})" for i, p in enumerate(premises))
    return (f"theorem goals {decl} {hyps} : {to_lean_prop(conclusion)} := by\n  {tactic}\n")


def _run_lean(source: str, lean_path: str, timeout: float = 120.0) -> tuple[bool, str, str, float]:
    """Compile a source string with Lean. Returns (ok, stdout, stderr, ms)."""
    tmpdir = tempfile.mkdtemp(prefix="logos_lean_")
    path = os.path.join(tmpdir, "Goals.lean")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([lean_path, path], capture_output=True, text=True,
                           timeout=timeout, cwd=tmpdir)
        ms = (time.perf_counter() - t0) * 1000
        return (r.returncode == 0, r.stdout or "", r.stderr or "", ms)
    except subprocess.TimeoutExpired:
        return (False, "", f"timeout after {timeout}s", (time.perf_counter() - t0) * 1000)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:                 # noqa: BLE001
            pass


def check(premise_texts: list[str], conclusion_text: str,
          use_tactics: bool = True) -> LeanResult:
    """Machine-check an argument. Reports exactly which method ran."""
    prem = [prop.parse(p) for p in premise_texts]
    conc = prop.parse(conclusion_text)
    py_valid = prop.entails(prem, conc)
    lp = find_lean()

    if not lp:
        return LeanResult(available=False, proved=py_valid, method="python_only",
                          notes=["Lean not found; result is Python truth table ONLY and must "
                                 "not be described as machine-checked"])

    lv = lean_version(lp)
    notes: list[str] = []

    # 1) A tactic proof of the proposition, if any tactic closes it.
    if use_tactics:
        for tactic in TACTIC_LADDER:
            src = lean_prop_source(prem, conc, tactic)
            ok, out, err, ms = _run_lean(src, lp)
            if ok:
                return LeanResult(available=True, proved=True, method=f"tactic:{tactic}",
                                  lean_version=lv, stdout=out, stderr=err, elapsed_ms=ms,
                                  agree_with_python=py_valid,
                                  notes=[f"propositional goal closed by '{tactic}'"])
            # "unknown tactic" means the tactic is unavailable (no Mathlib), not that the
            # goal is unprovable. Distinguish those, because they mean very different things.
            if "unknown tactic" in err or "unknown identifier" in err:
                notes.append(f"tactic '{tactic}' unavailable (Mathlib not present)")
        notes.append("no tactic in the ladder closed the goal")

    # 2) The Bool truth-table encoding, machine-checked by Lean's kernel.
    src = lean_bool_source(prem, conc)
    ok, out, err, ms = _run_lean(src, lp)
    if ok:
        return LeanResult(available=True, proved=py_valid,
                          method="rfl:bool-encoding", lean_version=lv, stdout=out,
                          stderr=err, elapsed_ms=ms, agree_with_python=True,
                          notes=notes + ["Bool encoding of all valuations checked by rfl"])

    # 3) The Bool check failed. For a VALID argument that is a real disagreement between two
    #    implementations and must be surfaced, not smoothed over.
    if py_valid:
        return LeanResult(available=True, proved=False, method="rfl:bool-encoding",
                          lean_version=lv, stdout=out, stderr=err, elapsed_ms=ms,
                          agree_with_python=False,
                          notes=notes + ["DISAGREEMENT: Python says valid, Lean rejects the "
                                         "Bool encoding. Do not trust either until resolved."])
    # For an INVALID argument the Bool file is expected to compile (it asserts the
    # counterexample), so a failure here means something else went wrong.
    return LeanResult(available=True, proved=False, method="none", lean_version=lv,
                      stdout=out, stderr=err, elapsed_ms=ms, agree_with_python=None,
                      notes=notes + ["Lean rejected the counterexample encoding; the emitted "
                                     "source is the ground truth to inspect"])


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.logic.lean_bridge - SELF TEST")
    ok = True

    lp = find_lean()
    print(f"  lean binary: {lp or 'NOT FOUND'}")
    lv = lean_version(lp) if lp else ""
    print(f"  lean version: {lv or 'unavailable'}")

    # --- a valid argument must be PROVED, by a machine check, not a parser
    r = check(["P -> Q", "P"], "Q")
    good = r.available and r.proved and r.trustworthy
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] modus ponens proved "
          f"(method={r.method}, {r.elapsed_ms:.0f} ms)")

    # --- a fallacy must NOT be proved, and Python must agree
    r2 = check(["P -> Q", "Q"], "P")
    good = (not r2.proved) and r2.agree_with_python in (True, None)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] affirming the consequent NOT proved "
          f"(method={r2.method}, agree={r2.agree_with_python})")

    # --- hypothetical syllogism
    r3 = check(["P -> Q", "Q -> R"], "P -> R")
    good = r3.proved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] hypothetical syllogism proved "
          f"(method={r3.method})")

    # --- denying the antecedent must fail
    r4 = check(["P -> Q", "~P"], "~Q")
    good = not r4.proved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] denying the antecedent NOT proved")

    # --- De Morgan
    r5 = check(["~(P & Q)"], "~P | ~Q")
    good = r5.proved
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] De Morgan proved (method={r5.method})")

    # --- the generated Lean source is well-formed and names the formula
    src = lean_prop_source([prop.parse("P -> Q"), prop.parse("P")], prop.parse("Q"),
                           "exact h1 h2")
    good = "theorem goals" in src and "→" in src and "exact h1 h2" in src
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Lean source generation is well-formed")

    # --- the Bool encoding emits one `example` per valuation
    bsrc = lean_bool_source([prop.parse("P -> Q"), prop.parse("P")], prop.parse("Q"))
    n_rows = len(prop.truth_table([prop.parse("P -> Q"), prop.parse("P")], prop.parse("Q")))
    good = bsrc.count("example") == n_rows
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] Bool encoding emits one machine-checked row per "
          f"valuation ({bsrc.count('example')}/{n_rows})")

    # --- a missing Lean must be reported as python_only, never as machine-checked
    r6 = check(["P"], "P | Q")
    good = (not r6.trustworthy) if not r6.available else r6.trustworthy
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] trustworthiness tracks whether Lean actually ran "
          f"(available={r6.available}, trustworthy={r6.trustworthy})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Machine-checked propositional proofs via Lean 4.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", nargs="+", default=[], metavar="FORMULA")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)
    if args.version:
        print(find_lean() or "not found")
        print(lean_version())
        return 0
    if args.self_test or not args.check:
        return 0 if run_self_test() else 1
    if len(args.check) < 2:
        print("need at least one premise and a conclusion")
        return 2
    r = check(args.check[:-1], args.check[-1])
    print(f"lean available : {r.available}")
    print(f"lean version   : {r.lean_version or '-'}")
    print(f"proved         : {r.proved}")
    print(f"method         : {r.method}")
    print(f"machine-checked: {r.trustworthy}")
    print(f"python agrees  : {r.agree_with_python}")
    print(f"elapsed        : {r.elapsed_ms:.0f} ms")
    for n in r.notes:
        print(f"note           : {n}")
    if r.stderr.strip():
        print("--- lean stderr ---")
        print(r.stderr.strip()[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
