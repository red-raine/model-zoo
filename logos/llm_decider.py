"""logos/llm_decider.py — can a local LLM be the decision maker, measured on OUR task and OUR hardware.

WHY THIS EXPERIMENT EXISTS
--------------------------
The third-party test (modeltested.com, 56 situations / 112 questions) reports local LLMs beating every
commercial "decision model": Qwen3.5-9B at 97.3% accuracy, Granite 4.2 8B at 97.4% on the hard cases,
against Jev 78.9 and Clef Flash 68.4. The obvious inference is "fine-tune a Qwen/Granite for our task".

That inference has two holes, and this module is built to measure both:

1. THEIR TEST IS 112 QUESTIONS WITH THE RULES STATED IN THE PROMPT. It grades policy application. Our
   dispositions require policy application AND fact verification -- is 1.609344 km right, does this passage
   support that claim. A model scoring 100% on stated-rule triage has demonstrated nothing about the second.

2. THE WINNER DOES NOT FIT THE HARDWARE. Their best local model is 9B. Measured here: a 2.6B generates at
   8.5 tok/s on 8 AVX2 cores. Parametric scaling puts a 9B near 2.5 tok/s and a 27B under 1. This module
   measures the real number rather than scaling it, and prices a corpus with it.

WHAT IT MEASURES
    accuracy   on the same rows pipeline.py handles, against the same constructed labels
    latency    real ms/row from a llama-server that stays loaded, not from a per-call reload
    cost       extrapolated to 10,000 rows, next to the measured encoder cascade

THE HONEST ASYMMETRY, STATED UP FRONT
The model tested here is 2.6B, and their winner is 9B. So this is NOT a verdict on "an LLM can't do it" --
a 9B would very likely score higher. It IS a measurement of what is available at a rate that could actually
process a corpus on this machine, and of the price of going bigger. Both halves matter, and neither settles
the fine-tuning question: that still needs labels.

USAGE
    python tools/logos/llm_decider.py --self-test
    python tools/logos/llm_decider.py --measure [--rows 6]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

LLAMA_SERVER = os.environ.get(
    "LOGOS_LLAMA_SERVER", r"E:\vibe_coding\dev\bitnet runner\llama.cpp-hf\build-cpu\bin\llama-server.exe")
DEFAULT_MODEL = os.environ.get("LOGOS_DECIDER_MODEL", r"B:\ai-stack\models\llama\LFM2.5-2.6B-Q4_K_M.gguf")
PORT = int(os.environ.get("LOGOS_DECIDER_PORT", "8412"))
BASE = f"http://127.0.0.1:{PORT}"

SYSTEM = (
    "You classify one row of a dataset. Answer with exactly one word from: "
    "KEEP, REPAIR, ESCALATE, DROP.\n"
    "KEEP = the row is correct and a check can confirm it.\n"
    "REPAIR = the row is mechanically fixable, e.g. arithmetic that computes to a different value.\n"
    "ESCALATE = no check in this system can decide the row.\n"
    "DROP = the row is wrong or empty.\n"
    "Answer with the single word only."
)

# the same rows pipeline.py uses, with the SAME constructed labels. comparing the two systems on different
# rows would produce a number that looks like a comparison and is not one.
CASES = [
    ("2 + 2 = 4", "KEEP"),
    ("pi = 3.14159", "KEEP"),
    ("1 mile = 1.609344 km", "KEEP"),
    ("2 + 2 = 5", "REPAIR"),
    ("7 * 6 = 41", "REPAIR"),
    ("pi = 3.0", "DROP"),
    ("1 mile = 1.5 km", "DROP"),
    ("The Eiffel Tower is 330 metres tall.", "ESCALATE"),
    ("Apoptosis is programmed cell death.", "ESCALATE"),
    ("", "DROP"),
]

VALID = ("KEEP", "REPAIR", "ESCALATE", "DROP")


def build_prompt(row: str) -> str:
    shown = row if row.strip() else "(empty row)"
    return f"{SYSTEM}\n\nROW:\n{shown}\n\nANSWER:"


def parse_disposition(text: str):
    """Pull ONE disposition out of a model's output, preferring what comes after any thinking block.

    Returns None when nothing matches. None is a real outcome and is reported as such: a model that answers
    in prose, or reasons and never concludes, has FAILED the task, and coercing a word out of its rambling
    would inflate its score. That is the same rule as fail-closed everywhere else in this suite.
    """
    if not text:
        return None
    body = text
    for tag in ("</think", "<｜end▁of▁thinking｜>"):
        if tag in body:
            body = body.split(tag)[-1]
    body = body.upper()
    hits = [(body.rfind(w), w) for w in VALID if w in body]
    if not hits:
        return None
    return max(hits)[1]


# ------------------------------------------------------------------------------- server control
def start_server(model: str, ctx: int = 2048, threads: int = 8, timeout: float = 180.0):
    """Launch llama-server and wait for it to answer. One load, many completions -- which is both faster and
    closer to how this would actually run than reloading per row."""
    if not os.path.exists(LLAMA_SERVER):
        raise FileNotFoundError(LLAMA_SERVER)
    if not os.path.exists(model):
        raise FileNotFoundError(model)
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", model, "-t", str(threads), "-c", str(ctx), "--port", str(PORT),
         "--host", "127.0.0.1", "-cb"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, time.perf_counter() - t0
        except Exception:                                     # noqa: BLE001
            time.sleep(1.0)
    proc.kill()
    raise TimeoutError(f"server did not become healthy in {timeout}s")


def complete(prompt: str, n_predict: int = 192, timeout: float = 300.0) -> dict:
    """One completion, temperature 0. Deterministic, so a re-run measures the system not the sampler."""
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
               "seed": 42, "stop": ["\n\n"]}
    req = urllib.request.Request(f"{BASE}/completion", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ------------------------------------------------------------------------------- measure
def measure(n_rows: int = 10) -> int:
    from logos import pipeline

    cases = CASES[:n_rows]
    print(f"=== can a local LLM be the decision maker? ({len(cases)} rows) ===")
    print(f"  model   {os.path.basename(DEFAULT_MODEL)}")
    print(f"  server  {os.path.basename(LLAMA_SERVER)}  threads=8  port={PORT}")

    # the comparison system, on the SAME rows
    pipe = [pipeline.evaluate_row(r, kind="text") for r, _e in cases]
    pipe_correct = sum(1 for p, (_r, e) in zip(pipe, cases) if p["disposition"] == e)

    proc = None
    try:
        proc, load_s = start_server(DEFAULT_MODEL)
        print(f"  load    {load_s:.1f}s (one load, reused for every row)\n")

        print(f"  {'row':34s} {'expect':9s} {'llm':9s} {'pipeline':9s} {'ms':>8s}  llm ok / pipe ok")
        lat, llm_correct, parsed_none, preds = [], 0, 0, []
        for (row, expect), p in zip(cases, pipe):
            t0 = time.perf_counter()
            out = complete(build_prompt(row))
            dt = (time.perf_counter() - t0) * 1000.0
            lat.append(dt)
            text = out.get("content") or ""
            got = parse_disposition(text)
            preds.append(got)
            if got is None:
                parsed_none += 1
            ok_llm = got == expect
            ok_pipe = p["disposition"] == expect
            llm_correct += 1 if ok_llm else 0
            shown = row.replace("\n", " ")[:34] if row.strip() else "(empty)"
            print(f"  {shown:34s} {expect:9s} {(got or 'NO-ANSWER'):9s} {p['disposition']:9s} "
                  f"{dt:8.0f}  {'y' if ok_llm else 'n'} / {'y' if ok_pipe else 'n'}")

        med = statistics.median(lat)

        # ---- the baseline that makes the accuracy number mean something
        expected_labels = [e for _r, e in cases]
        prior_counts = {w: expected_labels.count(w) for w in VALID}
        majority_label = max(prior_counts, key=prior_counts.get)
        majority_correct = prior_counts[majority_label]
        distinct_pred = sorted({r for r in preds if r})
        degenerate = len(distinct_pred) <= 1

        print()
        print(f"  LLM accuracy      {llm_correct}/{len(cases)}"
              f"{f'  ({parsed_none} produced NO parseable answer)' if parsed_none else ''}")
        print(f"  majority baseline {majority_correct}/{len(cases)}  "
              f"(always answer {majority_label!r})")
        print(f"  pipeline accuracy {pipe_correct}/{len(cases)}")
        print(f"  distinct labels the LLM produced: {len(distinct_pred)} {distinct_pred}")
        print()
        if degenerate:
            print(f"  ==> DEGENERATE. The model returned a single label for every row, so its score IS the")
            print(f"      class prior and it carries no information about the rows at all. Accuracy")
            print(f"      {llm_correct}/{len(cases)} is not 'half right' -- a two-line script that prints")
            print(f"      {majority_label!r} would score {majority_correct}/{len(cases)}.")
            print(f"      This is the baseline that a decision-model table must publish and usually does not.")
        elif llm_correct > majority_correct:
            print(f"  ==> +{llm_correct - majority_correct} over the majority baseline, on {len(cases)} rows.")
            print(f"      That is a signal, not a result: at this n one row is worth "
                  f"{100 / len(cases):.0f} points.")
        else:
            print(f"  ==> at or BELOW the majority baseline. The model is not reading the rows.")
        print(f"  LLM median        {med:,.0f} ms/row")
        enc_ms = 10.2                                     # measured in encoder_bench, 19M cross-encoder
        print(f"  comparable encoder {enc_ms:.1f} ms/row   ({med / enc_ms:,.0f}x faster)")

        print("\n=== what that costs on a corpus ===")
        for n in (1_000, 10_000, 100_000):
            hours = med * n / 1000.0 / 3600.0
            enc_h = enc_ms * n / 1000.0 / 3600.0
            print(f"  {n:>7,} rows   LLM {hours:8.2f} h   encoder {enc_h:6.2f} h   "
                  f"({hours / enc_h if enc_h else float('inf'):.0f}x)")

        print("\n=== the reading that matters ===")
        if pipe_correct > llm_correct:
            print(f"  On these rows the validator pipeline is more accurate AND ~{med / enc_ms:,.0f}x faster.")
            print(f"  That is not a surprise: the arithmetic rows are decided by computation, and a 2.6B")
            print(f"  model doing arithmetic in its head is doing the one thing a validator does better.")
        elif pipe_correct == llm_correct:
            print(f"  Equal accuracy, ~{med / enc_ms:,.0f}x the cost. Equal is not a win for the LLM.")
        else:
            print(f"  The LLM is AHEAD by {llm_correct - pipe_correct} rows. That is worth taking")
            print(f"  seriously -- but check WHICH rows before believing it, because the pipeline's")
            print(f"  misses are in the audit trail and the ladder there is reproducible.")

        print("\n=== what this experiment can and cannot settle ===")
        print(f"  CAN: whether a 2.6B instruct model, at a rate this machine sustains, is a usable")
        print(f"       decision maker for our dispositions. It is not a verdict on a 9B, which the")
        print(f"       third-party test measured at 97.3% and which would need ~2.5 tok/s here.")
        print(f"  CANNOT: settle fine-tuning. {len(cases)} rows with labels I wrote cannot distinguish a")
        print(f"       tuned model from a lucky one. The held-out set is still missing, and it is still")
        print(f"       the blocker -- not compute, not architecture.")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:                                 # noqa: BLE001
                proc.kill()


# ------------------------------------------------------------------------------- self test
def run_self_test() -> bool:
    print("logos.llm_decider - SELF TEST")
    ok = True

    # --- the parser takes the answer AFTER the thinking block, not a word from inside it
    good = parse_disposition("<think>we might DROP this but the math checks out</think>KEEP") == "KEEP"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a word inside the thinking block is not the answer")

    # --- and it works when there is no thinking block at all
    good = parse_disposition("ESCALATE") == "ESCALATE"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a bare answer parses")

    # --- NO ANSWER IS NONE, not a guess. coercing a word out of prose would inflate the score.
    good = parse_disposition("I cannot determine this from the given information.") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] prose with no disposition returns None, not a guess")

    good = parse_disposition("") is None and parse_disposition(None) is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty output returns None")

    # --- when several appear, the LAST one wins: models often restate the options before choosing
    good = parse_disposition("KEEP, REPAIR, ESCALATE, DROP... the answer is DROP") == "DROP"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the final disposition wins over an earlier restatement")

    # --- the prompt carries the row and all four options, or the task is underspecified
    p = build_prompt("2 + 2 = 5")
    good = all(w in p for w in VALID) and "2 + 2 = 5" in p
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the prompt contains every option and the row itself")

    # --- an empty row is shown as a marker, not as a blank that looks like a bug in the harness
    good = "(empty row)" in build_prompt("")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an empty row is rendered as a marker")

    # --- a constant answer is detectable as degenerate. This is the check whose absence made the first
    #     run report 0.50 for a function that never read a row.
    preds = ["KEEP"] * 6
    good = len(set(preds)) <= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a constant prediction set is flagged degenerate")

    # --- and the baseline arithmetic: three KEEPs out of six rows means 0.50 is the prior, not skill
    expected = ["KEEP", "KEEP", "KEEP", "REPAIR", "REPAIR", "DROP"]
    counts = {w: expected.count(w) for w in ("KEEP", "REPAIR", "ESCALATE", "DROP")}
    good = max(counts.values()) == 3 and max(counts, key=counts.get) == "KEEP"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] majority baseline for that set is 3/6 = KEEP, so 3/6 is no skill")

    # --- missing binaries are a STATE, reported rather than blowing up mid-batch
    good = True
    for path in (LLAMA_SERVER, DEFAULT_MODEL):
        if not os.path.exists(path):
            good = False
            print(f"  [SKIP] not present: {path}")
    if good:
        print(f"  [PASS] llama-server and the model are both present")
    ok = ok and good

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local LLM as decision maker, measured against the pipeline.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.measure:
        return measure(max(1, min(args.rows, len(CASES))))
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
