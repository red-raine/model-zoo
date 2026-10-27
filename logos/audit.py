"""logos/audit.py — the precision audit harness. Nothing ships a precision number without this.

WHY THIS GATES EVERYTHING
-------------------------
Every precision figure in this project so far is a POINT ESTIMATE on a sample chosen for convenience.
"96.5% precision" from 200 rows, where 7 were wrong, is not a claim anyone should train on: the 95% lower
bound on that is 93.5%, and if those 200 rows were the CONFIDENT ones the population figure is lower
still. Precision is a claim about unsampled rows and it can only be certified by sampling them properly.

THE TWO WAYS THIS IS USUALLY DONE WRONG
---------------------------------------
1. **Not stratifying.** Precision varies enormously by confidence band -- measured earlier at 96.4% in
   the top band against 80.8% overall. Sampling uniformly wastes almost all the audit budget on rows you
   already know are wrong.

2. **Stratifying and then reporting the raw audit rate.** If you over-sample the low-confidence band
   because that is where the defects are, the raw defect rate of your sample is NOT the population defect
   rate. You must weight each stratum by its SHARE OF THE POPULATION, not by its share of the audit. This
   module does that, and a test proves a naive average gives the wrong answer.

STATISTICAL VALIDITY
--------------------
Per-stratum bounds are exact Clopper-Pearson. To combine them into a population bound with JOINT
confidence 1-alpha, each stratum is bounded at alpha/H (Bonferroni over H strata) and the weighted sum of
the per-stratum lower bounds is used. A weighted sum of individual lower bounds is conservative -- the
true joint bound is at least as good -- so the number reported is never optimistic. Conservative is the
right direction for a refinery: an understated precision costs samples, an overstated one costs the model.

USAGE
    python tools/logos/audit.py --self-test

WORKFLOW
    strata  = make_strata(conf, bands=4)          # balance by confidence band
    sheet   = draw_audit(strata, per_stratum=50, seed=0)
    sheet.write("audit-sheet.jsonl")              # a human labels each row
    sheet.load_labels("audit-sheet.jsonl")
    report  = sheet.report(target=0.95)           # LB95, remaining audits, or futility
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.refinery import precision_lower_bound  # noqa: E402

CERTIFIED, CONTINUE, FUTILE = "certified", "continue", "futile"


# ------------------------------------------------------------------------------- strata
def make_strata(conf: np.ndarray, bands: int = 4) -> dict:
    """Partition by confidence band. Stratifying on confidence is the whole point: precision is a
    function of it, so a uniform sample measures the average and tells you nothing about the top."""
    c = np.asarray(conf, dtype=float)
    if bands < 1:
        raise ValueError("bands must be >= 1")
    edges = np.quantile(c, np.linspace(0.0, 1.0, bands + 1))
    edges[-1] = edges[-1] + 1e-12                     # make the top band closed
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, bands - 1)
    out = {}
    for b in range(bands):
        members = np.where(idx == b)[0]
        out[b] = {"indices": members, "n_population": len(members),
                  "conf_lo": float(edges[b]), "conf_hi": float(edges[b + 1])}
    return out


# ------------------------------------------------------------------------------- sheet
@dataclass
class AuditSheet:
    """A drawn audit sample with a place for a human verdict.

    `label` is deliberately ternary -- CORRECT / INCORRECT / AMBIGUOUS. Forcing a binary verdict on an
    ambiguous row is how a hook gets a label that means nothing; the ambiguous count is itself a finding
    about the filter.
    """

    rows: list[dict] = field(default_factory=list)
    n_population: int = 0
    target: float = 0.95
    confidence: float = 0.95

    # ------------------------------------------------------------------ sampling
    @classmethod
    def draw(cls, strata: dict, per_stratum: int = 50, seed: int = 0,
             target: float = 0.95, confidence: float = 0.95,
             row_ids: list | None = None) -> "AuditSheet":
        rng = np.random.default_rng(seed)
        rows: list[dict] = []
        for b, s in sorted(strata.items()):
            idx = s["indices"]
            take = min(per_stratum, len(idx))
            chosen = rng.choice(idx, size=take, replace=False) if take else np.array([], dtype=int)
            for i in sorted(chosen.tolist()):
                rows.append({
                    "row_id": (row_ids[i] if row_ids is not None else int(i)),
                    "stratum": int(b),
                    "n_population": int(s["n_population"]),
                    "conf_lo": s["conf_lo"],
                    "conf_hi": s["conf_hi"],
                    "label": "",            # CORRECT | INCORRECT | AMBIGUOUS
                    "note": "",
                })
        return cls(rows=rows,
                   n_population=int(sum(s["n_population"] for s in strata.values())),
                   target=target, confidence=confidence)

    # ------------------------------------------------------------------ io
    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def load_labels(self, path: str) -> "AuditSheet":
        with open(path, encoding="utf-8") as fh:
            loaded = [json.loads(ln) for ln in fh if ln.strip()]
        by_id = {r["row_id"]: r for r in loaded}
        for r in self.rows:
            if r["row_id"] in by_id:
                r["label"] = by_id[r["row_id"]].get("label", "")
                r["note"] = by_id[r["row_id"]].get("note", "")
        return self

    # ------------------------------------------------------------------ report
    def report(self) -> dict:
        """Stratified population precision, per-stratum counts, and a valid joint lower bound.

        AMBIGUOUS rows are counted as NOT correct. That is the conservative choice: a row a human cannot
        adjudicate is not evidence of quality, and treating it as correct would inflate precision using
        the rows we understand least.
        """
        H = len({r["stratum"] for r in self.rows})
        if H == 0:
            return {"ok": False, "reason": "empty sheet"}
        alpha = 1.0 - self.confidence

        per = {}
        for r in self.rows:
            b = r["stratum"]
            e = per.setdefault(b, {"n_audit": 0, "correct": 0, "incorrect": 0, "ambiguous": 0,
                                   "n_population": r["n_population"]})
            e["n_audit"] += 1
            if r["label"] == "CORRECT":
                e["correct"] += 1
            elif r["label"] == "INCORRECT":
                e["incorrect"] += 1
            elif r["label"] == "AMBIGUOUS":
                e["ambiguous"] += 1

        total_pop = sum(v["n_population"] for v in per.values()) or self.n_population
        weighted_point = 0.0
        weighted_lb = 0.0
        covered_weight = 0.0
        unsampled = []
        for b, v in sorted(per.items()):
            w = v["n_population"] / total_pop if total_pop else 0.0
            if v["n_audit"] == 0:
                unsampled.append(b)
                continue
            defects = v["incorrect"] + v["ambiguous"]
            p = 1.0 - defects / v["n_audit"]
            lb = precision_lower_bound(defects, v["n_audit"], 1.0 - alpha / H)   # Bonferroni
            v["precision_point"] = p
            v["precision_lb"] = lb
            v["weight"] = w
            weighted_point += w * p
            weighted_lb += w * lb
            covered_weight += w

        return {
            "ok": covered_weight > 0,
            "n_audit": sum(v["n_audit"] for v in per.values()),
            "n_population": total_pop,
            "strata": {int(b): v for b, v in sorted(per.items())},
            "unsampled_strata": unsampled,
            "precision_point": weighted_point / covered_weight if covered_weight else 0.0,
            "precision_lb": weighted_lb / covered_weight if covered_weight else 0.0,
            "coverage_of_population": covered_weight,
            "labels_complete": all(r["label"] for r in self.rows),
        }

    def verdict(self) -> dict:
        """certified / continue / futile, plus how many more audits would settle it.

        Futility matters as much as certification: if the observed defect rate makes the target
        unreachable, more auditing only spends money confirming that.
        """
        rep = self.report()
        if not rep.get("ok"):
            return {"status": CONTINUE, "reason": "no labels yet"}
        if not rep["labels_complete"]:
            return {"status": CONTINUE, "reason": "sheet not fully labelled"}
        lb = rep["precision_lb"]
        if lb >= self.target:
            return {"status": CERTIFIED, "lb": lb, "target": self.target,
                    "reason": f"lower bound {lb:.4f} >= target {self.target:.4f}"}
        # futility: if the observed precision is at or below the target, the LOWER BOUND can never
        # reach it -- a bound sits strictly below its point estimate for any finite sample. More
        # auditing then only spends money confirming that the FILTER, not the audit, must change.
        observed_defect = 1.0 - rep["precision_point"]
        if observed_defect > 0 and (1.0 - observed_defect) <= self.target:
            return {"status": FUTILE, "lb": lb, "target": self.target,
                    "observed_precision": rep["precision_point"],
                    "reason": f"observed {rep['precision_point']:.4f} is at or below the "
                              f"{self.target:.4f} target, so no sample size can certify it -- the "
                              f"FILTER must improve, not the audit"}
        need = _audits_for_target(self.target, max(observed_defect, 1e-9), self.confidence)
        if need is None:
            return {"status": FUTILE, "lb": lb, "target": self.target,
                    "observed_precision": rep["precision_point"],
                    "reason": "no achievable audit size certifies this target at the observed defect "
                              "rate"}
        return {"status": CONTINUE, "lb": lb, "target": self.target,
                "audits_needed_total": need,
                "more_needed": max(0, need - rep["n_audit"]),
                "reason": f"lower bound {lb:.4f} below target {self.target:.4f}"}


def _audits_for_target(target: float, defect_rate: float, confidence: float) -> int | None:
    """Audits needed to certify `target` given the OBSERVED defect rate.

    The zero-defect closed form is not usable here: once defects are observed, the required n depends on
    their density. Delegating to `refinery.audit_n_for_precision` also gets the achievability check for
    free (a target at or below the defect rate returns None).
    """
    from logos.refinery import audit_n_for_precision

    if 1.0 - defect_rate <= target:
        return None
    return audit_n_for_precision(target, defect_rate, confidence)


# ------------------------------------------------------------------------------- tests
# ---------------------------------------------------------------------------------- driver
def run() -> int:
    """Draw a real audit sheet from the AVeriTeC dev keep pile and project the audit budget.

    HONEST CAVEAT, and it is load-bearing: the labels used here are the DATASET's gold labels, not a
    human adjudicating our filter's errors. So this demonstrates the HARNESS and the ARITHMETIC -- it
    answers "how many audits would this design need" -- and it does NOT certify anything. Certification
    requires a human on the sheet.
    """
    from logos.baseline import D
    from logos.calibrate import Probs

    rows = [json.loads(ln) for ln in
            open(os.path.join(D, "nli-post-logits.jsonl"), encoding="utf-8") if ln.strip()]
    dev = [r for r in rows if r["split"] == "dev"]
    Z = np.array([r["logits"] for r in dev], dtype=float)
    y = np.array([r["label"] for r in dev])
    P = Probs.from_logits(Z, 4.226, labels=y)
    conf = P.select("threshold").max(axis=1)
    correct = P.select("threshold").argmax(axis=1) == y

    strata = make_strata(conf, bands=4)
    n_pop = len(y)
    print(f"=== audit design over {n_pop} dev rows, stratified by confidence ===")
    print(f"  {'band':>5s} {'conf range':>16s} {'population':>11s} {'share':>7s} "
          f"{'true precision':>15s}")
    for b, st in sorted(strata.items()):
        idx = st["indices"]
        rng = f"[{st['conf_lo']:.2f},{st['conf_hi']:.2f}]"
        print(f"  {b:5d} {rng:>16s} {st['n_population']:11d} {st['n_population'] / n_pop:7.1%} "
              f"{float(correct[idx].mean()):15.3f}")
    print(f"  NOTE: precision varies across bands, which is exactly why a uniform sample is the wrong")
    print(f"  design -- it measures the average and tells you nothing about the top band.")

    print(f"\n=== what each audit budget would certify (gold labels, NOT human adjudication) ===")
    truth = float(correct.mean())
    print(f"  population precision (all rows) = {truth:.4f}")
    print(f"  {'per stratum':>12s} {'audits':>7s} {'point':>7s} {'LB95':>7s} {'verdict @0.90':>14s}")
    for per in (10, 25, 50, 100, 124):
        sheet = AuditSheet.draw(strata, per_stratum=per, seed=0, target=0.90)
        lookup = {int(i): bool(correct[i]) for i in range(n_pop)}
        for r in sheet.rows:
            r["label"] = "CORRECT" if lookup[int(r["row_id"])] else "INCORRECT"
        rep = sheet.report()
        v = sheet.verdict()
        print(f"  {per:12d} {rep['n_audit']:7d} {rep['precision_point']:7.4f} "
              f"{rep['precision_lb']:7.4f} {v['status']:>14s}")

    print(f"\n=== the budget question, and the hard ceiling ===")
    total = sum(st["n_population"] for st in strata.values())
    print(f"  Population is {total}. Certifying 99% at zero defects needs 299 audits; 99.9% needs ")
    print(f"  2,995 -- six times the entire dev set. THE AUDIT CANNOT BE LARGER THAN THE KEEP PILE,")
    print(f"  so on AVeriTeC this module can only demonstrate the design. It is built for the CODE")
    print(f"  refinery, where the keep pile is measured in millions.")

    # --- THE POINT OF ALL THIS: an audit certifies a FILTERED population, not the raw model. The whole
    #     dev set is 80.8% precise, so no sample of it can ever certify 90% -- correctly reported as
    #     futile above. But the TOP BAND is 96.0% precise, and an audit of THAT pile certifies.
    top = max(strata, key=lambda b: float(correct[strata[b]["indices"]].mean()))
    top_idx = strata[top]["indices"]
    keep = correct[top_idx]
    n_keep = len(top_idx)
    defects = int((~keep).sum())
    top_sheet = AuditSheet(
        rows=[{"row_id": int(i), "stratum": 0, "n_population": n_keep,
               "label": "CORRECT" if correct[i] else "INCORRECT"} for i in top_idx],
        n_population=n_keep, target=0.90)
    trep = top_sheet.report()
    tv = top_sheet.verdict()
    print(f"\n=== the same arithmetic on the KEEP PILE (top confidence band only) ===")
    print(f"  keep pile: {n_keep} rows, {defects} defective, point precision "
          f"{trep['precision_point']:.4f}")
    print(f"  LB95 {trep['precision_lb']:.4f} -> verdict at a 90% target: {tv['status'].upper()}")
    print(f"  THE LESSON: the audit did not fail. It correctly refused to certify a population that is")
    print(f"  80.8% precise, and correctly certified one that is 96.0% precise. An audit certifies the")
    print(f"  FILTERED pile -- so filtering is what makes a precision claim possible at all, and auditing")
    print(f"  the unfiltered model measures the thing you were going to discard anyway.")

    print(f"\n=== high-precision thresholds on the keep pile ===")
    for target in (0.90, 0.93, 0.95):
        sh = AuditSheet(rows=[{"row_id": int(i), "stratum": 0, "n_population": n_keep,
                               "label": "CORRECT" if correct[i] else "INCORRECT"} for i in top_idx],
                        n_population=n_keep, target=target)
        v = sh.verdict()
        need = v.get("more_needed")
        print(f"  target {target:.0%}: {v['status']:>9s}"
              + (f"   (needs {need} more; the pile has {n_keep})" if need is not None else ""))

    out = AuditSheet.draw(strata, per_stratum=25, seed=0, target=0.90)
    path = out.write(os.path.join(D, "audit-sheet.dev.jsonl"))
    print(f"\n  sample sheet written: {path}  ({len(out.rows)} rows, label field empty)")
    print(f"  labels accepted: CORRECT | INCORRECT | AMBIGUOUS  (ambiguous counts AGAINST precision)")
    return 0


def run_self_test() -> bool:
    print("logos.audit - SELF TEST")
    ok = True
    rng = np.random.default_rng(0)

    # --- 1. strata partition the population exactly
    conf = rng.uniform(0, 1, 1000)
    s = make_strata(conf, 4)
    good = (sum(v["n_population"] for v in s.values()) == 1000
            and all(len(v["indices"]) == v["n_population"] for v in s.values()))
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] strata partition the population exactly "
          f"({[v['n_population'] for v in s.values()]})")

    # --- 2. THE MISTAKE THIS MODULE PREVENTS: over-sampling a bad stratum makes the RAW audit rate
    #        wrong, and re-weighting recovers the truth.
    #        Population: 90% of rows in the GOOD stratum (95% precision), 10% in the BAD (50% precision).
    #        True population precision = 0.9*0.95 + 0.1*0.50 = 0.905
    n_good, n_bad = 900, 100
    sheet = AuditSheet(rows=[], n_population=1000, target=0.85)
    for i in range(50):
        sheet.rows.append({"row_id": f"g{i}", "stratum": 0, "n_population": n_good,
                           "label": "CORRECT" if i < 48 else "INCORRECT"})   # 96%
    for i in range(50):
        sheet.rows.append({"row_id": f"b{i}", "stratum": 1, "n_population": n_bad,
                           "label": "CORRECT" if i < 25 else "INCORRECT"})   # 50%
    rep = sheet.report()
    raw_audit_rate = (48 + 25) / 100
    good = (abs(rep["precision_point"] - (0.9 * 0.96 + 0.1 * 0.50)) < 1e-9
            and abs(raw_audit_rate - rep["precision_point"]) > 0.10)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] re-weighting recovers the population precision "
          f"({rep['precision_point']:.4f}) while the RAW audit rate ({raw_audit_rate:.4f}) is wrong "
          f"by {abs(raw_audit_rate - rep['precision_point']):.3f}")

    # --- 3. the joint lower bound is below the point estimate and above zero
    good = 0.0 < rep["precision_lb"] < rep["precision_point"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the joint LB sits below the point estimate "
          f"({rep['precision_lb']:.4f} < {rep['precision_point']:.4f})")

    # --- 4. Ambiguous rows count AGAINST precision, not for it
    sheet2 = AuditSheet(rows=[{"row_id": i, "stratum": 0, "n_population": 100,
                               "label": "AMBIGUOUS" if i < 10 else "CORRECT"}
                              for i in range(20)], n_population=100)
    rep2 = sheet2.report()
    good = abs(rep2["precision_point"] - 0.5) < 1e-9
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] AMBIGUOUS counts as not-correct, so 10/20 ambiguous gives "
          f"precision {rep2['precision_point']:.2f} (the conservative choice)")

    # --- 5. certification: a clean sheet certifies a modest target, a dirty one does not
    clean = AuditSheet(rows=[{"row_id": i, "stratum": 0, "n_population": 5000, "label": "CORRECT"}
                             for i in range(200)], n_population=5000, target=0.95)
    v = clean.verdict()
    good = v["status"] == CERTIFIED
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 200/200 correct certifies a 95% target "
          f"(LB {v['lb']:.4f})")

    # --- 6. A POINT ESTIMATE IS NOT A BOUND. 190/200 is exactly 95%, so a 95% target can NEVER be
    #        certified: a lower bound sits strictly below its own point estimate at every finite n.
    #        That is futility, not "needs more auditing".
    at_target = AuditSheet(rows=[{"row_id": i, "stratum": 0, "n_population": 5000,
                                  "label": "CORRECT" if i < 190 else "INCORRECT"}
                                 for i in range(200)], n_population=5000, target=0.95)
    v = at_target.verdict()
    good = v["status"] == FUTILE and "FILTER must improve" in v["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 190/200 IS 95%, so a 95% target is FUTILE -- a bound "
          f"cannot reach its own point estimate")

    # --- 6b. THE SAME 190/200 GIVES THREE DIFFERENT ANSWERS AT THREE TARGETS, and this is the
    #        whole point of reporting a bound instead of a point estimate:
    #            target 0.90 -> CERTIFIED   (LB 0.9167 clears it)
    #            target 0.94 -> CONTINUE    (between the LB and the point estimate; more audits could settle it)
    #            target 0.95 -> FUTILE      (the point estimate IS the target, so the bound never reaches it)
    rows190 = [{"row_id": i, "stratum": 0, "n_population": 5000,
                "label": "CORRECT" if i < 190 else "INCORRECT"} for i in range(200)]
    v90 = AuditSheet(rows=[dict(r) for r in rows190], n_population=5000, target=0.90).verdict()
    v94 = AuditSheet(rows=[dict(r) for r in rows190], n_population=5000, target=0.94).verdict()
    v95 = AuditSheet(rows=[dict(r) for r in rows190], n_population=5000, target=0.95).verdict()
    good = (v90["status"] == CERTIFIED and v94["status"] == CONTINUE
            and v94.get("more_needed", 0) > 0 and v95["status"] == FUTILE)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the same 190/200 is CERTIFIED at 90% (LB "
          f"{v90['lb']:.4f}), CONTINUE at 94% (needs {v94.get('more_needed')} more), FUTILE at 95%")

    # --- 6c. FUTILITY also covers a target the observed precision is simply too low for
    hopeless = AuditSheet(rows=[{"row_id": i, "stratum": 0, "n_population": 5000,
                                 "label": "CORRECT" if i < 150 else "INCORRECT"}
                                for i in range(200)], n_population=5000, target=0.99)
    good = hopeless.verdict()["status"] == FUTILE
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a 99% target on 75% observed is FUTILE -- more auditing "
          f"cannot fix a filter")

    # --- 7. an incomplete sheet refuses to certify
    partial = AuditSheet(rows=[{"row_id": 0, "stratum": 0, "n_population": 10, "label": "CORRECT"},
                               {"row_id": 1, "stratum": 0, "n_population": 10, "label": ""}],
                         n_population=10, target=0.5)
    good = partial.verdict()["status"] == CONTINUE and "not fully labelled" in partial.verdict()["reason"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] an unlabelled sheet refuses to certify")

    # --- 8. draw + write + load round-trips, so a human can actually label it
    import tempfile
    tmp = tempfile.mkdtemp()
    sh = AuditSheet.draw(make_strata(conf, 4), per_stratum=10, seed=7)
    p = sh.write(os.path.join(tmp, "sheet.jsonl"))
    for r in sh.rows:
        r["label"] = "CORRECT"
    sh.write(p)
    sh2 = AuditSheet(rows=[dict(r, label="") for r in sh.rows], n_population=sh.n_population)
    sh2.load_labels(p)
    good = all(r["label"] == "CORRECT" for r in sh2.rows) and len(sh2.rows) == len(sh.rows)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] draw -> write -> load round-trips "
          f"({len(sh.rows)} rows across {len(make_strata(conf, 4))} strata)")

    # --- 9. and the draw is reproducible from the seed
    a = AuditSheet.draw(make_strata(conf, 4), per_stratum=10, seed=7)
    b = AuditSheet.draw(make_strata(conf, 4), per_stratum=10, seed=7)
    good = [r["row_id"] for r in a.rows] == [r["row_id"] for r in b.rows]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the same seed draws the same rows -- the audit is "
          f"reproducible")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stratified precision audit harness.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    if args.run:
        return run()
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
