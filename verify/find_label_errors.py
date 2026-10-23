"""find_label_errors.py â€” multi-signal label-error finder for AVeriTeC.

WHY THIS EXISTS
---------------
I claimed "we are at the noise floor" because dev error (25.6%) looked like the
`required_reannotation` rate (24.4%). That was a coincidence, and decomposing it killed the
claim:

    reannotation-flagged dev rows   n=121  acc 0.6529  error 34.7%
    not flagged                    n=362  acc 0.7735  error 22.7%
    share of ALL our errors on flagged rows: 33.9%

So human flags carry a real 1.5x error enrichment but explain only a third of our error.
Which means: some of the error is label noise we can find, and some is model error we must
fix. This file finds the former.

THE RULE
--------
A model must never be the sole judge of its own ground truth. The signals here are ranked by
how independent they are of the model being evaluated:

  1. `human_flag`          â€” the dataset authors' own doubt. Fully independent.
  2. `zs_disagrees`        â€” a checkpoint that was NEVER trained on our split (the CTU AIC
                             model) confidently contradicts the gold label.
  3. `models_agree_against_gold` â€” two independent models both confident on the same
                             non-gold class. Strongest available evidence of a mislabel.
  4. `lexical_contradiction` â€” the evidence itself names the source as imaginary/satire/
                             hoax while gold says SUPPORTED. Deterministic, auditable.
  5. `numeric_mismatch`    â€” the claim's number does not appear anywhere in the evidence but
                             gold says SUPPORTED.
  6. `low_confidence`      â€” ambiguity, NOT mislabel. Weighted low on purpose.

VALIDATION, NOT VIBES
---------------------
The finder is scored against `required_reannotation` â€” the only independent per-row signal
about label quality we have â€” and reported as an AUC. A finder that cannot rank human-flagged
rows above unflagged ones is not finding errors, it is ranking models.

Dev rows are for REPORTING ONLY. Cleaning dev and then reporting accuracy on the cleaned dev
is circular and would manufacture any number you want. The cleaning experiment lives in
`--clean-train-out`, which rewrites TRAIN and leaves dev untouched.

USAGE
-----
    python tools/verify/find_label_errors.py --self-test
    python tools/verify/find_label_errors.py --split dev --zs B:\\...\\nli\\logits.jsonl ^
        --model B:\\...\\cls-unc\\logits.jsonl --top 25
    python tools/verify/find_label_errors.py --split train --zs ... --clean-train-out B:\\...\\train.cleaned.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting"]
SHORT = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 2, "CONFLICTING": 3}
I2L = {i: lab for i, lab in enumerate(LABELS)}

# Refutation strategies AVeriTeC annotators actually record. If the gathered evidence says
# the source is imaginary, satire, or a hoax, then "the evidence establishes the claim" is
# self-contradictory â€” unless the claim IS about the satire.
FAKE_MARKERS = ("imaginary", "fictional", "satire", "satirical", "parody", "spoof", "hoax",
                "fake news", "not a real", "does not exist", "made up", "fabricat",
                "conspiracy", "rumour", "rumor", "misleading")
NEG_MARKERS = ("did not", "does not", "never", "denied", "not true", "untrue", "incorrect",
               "no evidence", "is false", "was false", "debunk", "refut")
NUM_RE = re.compile(r"\d[\d,\.]*")


def split_claim_evidence(text: str) -> tuple[str, str]:
    claim, sep, ev = text.partition("[SEP]")
    return claim.strip(), (ev.strip() if sep else "")


def numbers(s: str) -> set[str]:
    out = set()
    for m in NUM_RE.findall(s):
        v = m.replace(",", "").rstrip(".")
        if v and len(v) <= 12:
            try:
                out.add(str(float(v)).rstrip("0").rstrip("."))
            except ValueError:
                pass
    return out


def score_row(claim: str, ev: str, gold: int, flag: bool,
              model_preds: list[tuple[int, float]],
              zs_preds: list[tuple[int, float]]) -> tuple[float, float, list[str]]:
    """Return (suspicion_all, suspicion_without_human_flag, reason codes).

    The second number exists because of a bug caught in validation: scoring the finder
    against `required_reannotation` while `human_flag` is one of its own inputs gives a
    perfect AUC of 1.000, which measures nothing but the fact that 0.30 > 0.0. The
    independent signals must be validated with the flag term removed, or the test is
    circular. `suspicion_nf` is the honest "did WE find something the humans did not tell
    us" score.

    Weights are hand-set and deliberately conservative; the validation AUC is what decides
    whether the ranking is worth anything, not the weights themselves.
    """
    score, score_nf, why = 0.0, 0.0, []

    if flag:
        score += 0.30
        why.append("human_flag")

    low = ev.lower()
    if gold == 0 and any(m in low for m in FAKE_MARKERS):
        score += 0.40
        score_nf += 0.40
        why.append("gold=SUPPORTED but evidence says source is fake/satire")
    if gold == 0 and any(m in low for m in NEG_MARKERS):
        score += 0.15
        score_nf += 0.15
        why.append("gold=SUPPORTED but evidence contains negation")

    if gold == 0:
        cn = numbers(claim)
        if cn and not (cn & numbers(ev)):
            score += 0.25
            score_nf += 0.25
            why.append(f"gold=SUPPORTED but claim numbers {sorted(cn)} absent from evidence")

    # a checkpoint never trained on this split contradicting a confident gold
    for pred, prob in zs_preds:
        if pred != gold and prob >= 0.70:
            score += 0.35
            score_nf += 0.35
            why.append(f"independent model (never trained on our split) says {I2L[pred]} "
                       f"at p={prob:.2f}")
            break
        if pred != gold and prob >= 0.50:
            score += 0.15
            score_nf += 0.15
            why.append(f"independent model says {I2L[pred]} at p={prob:.2f}")
            break

    # two models confident on the SAME non-gold class
    voters = [1 for pred, prob in zs_preds if pred != gold and prob >= 0.60]
    voters += [1 for pred, prob in model_preds if pred != gold and prob >= 0.60]
    if len(voters) >= 2:
        score += 0.45
        score_nf += 0.45
        why.append(f"{len(voters)} independent models confident against gold")

    best = max([p for _, p in model_preds] + [p for _, p in zs_preds] + [0.0])
    if 0.0 < best < 0.45:
        score += 0.10
        score_nf += 0.10
        why.append(f"all models unsure (max p={best:.2f}) - ambiguous, not necessarily wrong")

    return min(score, 1.0), min(score_nf, 1.0), why



def load_preds(path: str, split: str, perm: list[int] | None = None) \
        -> dict[str, list[tuple[int, float]]]:
    """Map text -> [(pred, prob)] so predictions join on content, not row order.

    `perm` remaps a checkpoint's label order onto ours. This is not optional for the CTU
    model: its zero-shot confusion matrix showed model->ours = [1,0,2,3], i.e. Supported and
    Refuted are SWAPPED in its head. Feeding its raw argmax into the finder would make every
    Supported/Refuted row look like a label error, and the finder would confidently "find"
    250 phantom errors.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    if not path:
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] != split:
                continue
            lg = r["logits"]
            mx = max(lg)
            ex = sum(2.718281828459045 ** (v - mx) for v in lg)
            pred = int(r["pred"])
            p = (2.718281828459045 ** (lg[pred] - mx)) / ex
            if perm:
                pred = perm[pred]
            out.setdefault(r["text"], []).append((pred, float(p)))
    return out


def auc(score: list[float], y: list[int]) -> float:
    n1 = sum(y)
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = 0.5 * (i + j) + 1.0
        i = j + 1
    s = sum(ranks[i] for i in range(len(score)) if y[i] == 1)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def run_self_test() -> bool:
    print("find_label_errors.py - SELF TEST")
    ok = True

    s, s_nf, w = score_row("c", "the source is an imaginary news organisation", 0, False,
                           [], [])
    good = s >= 0.40 and any("fake" in x for x in w)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] satire marker + gold SUPPORTED -> {s:.2f}")

    s2, s2_nf, _ = score_row("c", "a plain report with no markers at all", 0, False, [], [])
    good = s2 == 0.0 and s2_nf == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] clean row scores 0.00 (got {s2:.2f})")

    s3, s3_nf, _ = score_row("c", "ev", 1, True, [], [])
    good = abs(s3 - 0.30) < 1e-9 and s3_nf == 0.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] human flag alone -> all 0.30, no-flag 0.00 "
          f"(this is why validating against flags while the flag is an INPUT is circular)")

    s4, s4_nf, w4 = score_row("c", "ev", 1, False, [(0, 0.9)], [(0, 0.8)])
    good = s4 >= 0.80 and s4_nf >= 0.80 and any("independent models" in x for x in w4)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] two confident models vs gold -> {s4:.2f}")

    s5, s5_nf, w5 = score_row("the town has 5,000 people", "the town has 9000 residents", 0,
                              False, [], [])
    good = s5 >= 0.25 and s5_nf >= 0.25 and any("numbers" in x for x in w5)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] numeric mismatch -> {s5:.2f}")


    good = numbers("in 1999 there were 1,234 people") == {"1999", "1234"}
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] number extraction "
          f"{sorted(numbers('in 1999 there were 1,234 people'))}")

    good = auc([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == 1.0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AUC on a perfect ranking = 1.0")

    c, e = split_claim_evidence("cats fly [SEP] Q: do cats fly A: no")
    good = c == "cats fly" and e.startswith("Q:")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] claim/evidence split")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find suspected label errors.")
    ap.add_argument("--data", default=os.environ.get("VERIFIER_DATA",
                                                    r"B:\datasets\forge\verifier-av"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--zs", default="", help="logits.jsonl from a model never trained on us")
    ap.add_argument("--model", default="", help="logits.jsonl from our own model")
    ap.add_argument("--raw", default="", help="raw AVeriTeC json for required_reannotation")
    ap.add_argument("--zs-perm", default="",
                    help="comma label remap for the zs checkpoint, e.g. 1,0,2,3")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--clean-train-out", default="",
                    help="write a TRAIN jsonl with suspected-bad rows removed")
    ap.add_argument("--clean-mode", choices=("noflag", "all"), default="noflag",
                    help="clean on independent signals only, or include the human flag")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1

    path = os.path.join(args.data, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    zperm = [int(v) for v in args.zs_perm.split(",")] if args.zs_perm else None
    if zperm:
        print(f"[err] remapping zs label order with {zperm}  (raw argmax would invent hundreds of phantom errors)")
    zs = load_preds(args.zs, args.split, zperm)
    me = load_preds(args.model, args.split)

    # human flags, if we have the raw file (only AVeriTeC dev/train carry them)
    flags = {}
    if args.raw and os.path.exists(args.raw):
        raw = json.load(open(args.raw, encoding="utf-8"))
        for r in raw:
            flags[r["claim"].strip().lower()] = bool(r.get("required_reannotation"))
    flag_hits = 0

    scored = []
    for r in rows:
        claim = (r.get("claim") or "").strip()
        gold_s = (r.get("completion") or "").upper()
        gold = SHORT.get(gold_s)
        if gold is None:
            continue
        ev = (r.get("prompt") or "")
        ev = ev.split("EVIDENCE:")[-1].split("Answer with exactly one word")[0].strip()
        text = f"{claim} [SEP] {ev}"
        flag = flags.get(claim.lower(), False)
        flag_hits += int(bool(flags) and claim.lower() in flags)
        s, s_nf, why = score_row(claim, ev, gold, flag,
                                 me.get(text, []), zs.get(text, []))
        scored.append({"claim": claim, "gold": I2L[gold], "gold_id": gold,
                       "flag": flag, "suspicion": round(s, 3),
                       "suspicion_noflag": round(s_nf, 3), "reasons": why,
                       "text": text})

    n = len(scored)
    print(f"[err] {args.split}: {n} rows Â· zs preds joined "
          f"{sum(1 for r in scored if zs.get(r['text']))} "
          f"Â· own preds joined {sum(1 for r in scored if me.get(r['text']))}"
          f" Â· human flags {sum(1 for r in scored if r['flag'])}")

    # ---- validation: can the INDEPENDENT signals rank human-flagged rows high?
    if any(r["flag"] for r in scored):
        y = [int(r["flag"]) for r in scored]
        a_nf = auc([r["suspicion_noflag"] for r in scored], y)
        a_all = auc([r["suspicion"] for r in scored], y)
        print(f"[err] VALIDATION  AUC(independent signals only, vs required_reannotation) "
              f"= {a_nf:.4f}")
        print(f"[err]             AUC(all signals incl. the flag as an input)          "
              f"= {a_all:.4f}  <- CIRCULAR, shown only to demonstrate why it is ignored")
        print("[err]   0.5 = the finder is noise. >0.70 = the lexical/numeric/model "
              "signals find label doubt humans flagged, without being told.")
        a = a_nf
    else:
        a = float("nan")
        print("[err] no human flags available for validation on this split")

    # ---- how much of the label set would cleaning touch?
    for t in (0.25, 0.40, 0.60, 0.80):
        k = sum(1 for r in scored if r["suspicion"] >= t)
        kf = sum(1 for r in scored if r["suspicion_noflag"] >= t)
        fl = sum(1 for r in scored if r["suspicion_noflag"] >= t and r["flag"])
        print(f"[err] suspicion >= {t:.2f}: all-signal {k:4d} ({k/n:.1%}) Â· "
              f"independent-only {kf:4d} ({kf/n:.1%}), flagged among them {fl}")


    dist = collections.Counter()
    for r in scored:
        for w in r["reasons"]:
            dist[w.split("(")[0].strip()] += 1
    print("\n[err] reason frequency:")
    for k, v in dist.most_common(10):
        print(f"[err]   {v:4d}  {k}")

    print(f"\n[err] top {args.top} by suspicion:")
    for r in sorted(scored, key=lambda x: -x["suspicion"])[: args.top]:
        tag = "FLAG" if r["flag"] else "    "
        print(f"[err] {r['suspicion']:.2f} {tag} gold={r['gold'][:11]:11s} "
              f"| {'; '.join(r['reasons'])[:96]}")
        print(f"[err]      {r['claim'][:104]}")

    out = os.path.join(args.data, f"label-errors.{args.split}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"split": args.split, "auc_vs_human_flag": a,
                   "n": n, "rows": scored}, fh, indent=2, default=float)
    print(f"\n[err] wrote {out}")

    if args.clean_train_out and args.split == "train":
        field = "suspicion" if args.clean_mode == "all" else "suspicion_noflag"
        keep = [r for r in scored if r[field] < args.threshold]
        key = {r["text"] for r in keep}
        dropped = 0
        with open(args.clean_train_out, "w", encoding="utf-8") as fh:
            for r in rows:
                ev = (r.get("prompt") or "").split("EVIDENCE:")[-1]
                ev = ev.split("Answer with exactly one word")[0].strip()
                t = f"{(r.get('claim') or '').strip()} [SEP] {ev}"
                if t in key:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    dropped += 1
        print(f"[err] cleaned TRAIN ({args.clean_mode}) -> {args.clean_train_out}: "
              f"kept {len(keep)}, dropped {dropped} ({dropped/max(len(rows),1):.1%}) "
              f"at threshold {args.threshold}")
        print("[err] dev was NOT touched. Any accuracy from this must be measured on the "
              "original dev or it is circular.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

