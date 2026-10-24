"""logos.facts — facts with provenance, and the temporal gate.

THE TWO PROBLEMS THIS FIXES
---------------------------
The constructed per-evidence corpus reached 1.000, but its fact table was **illustrative**, not
sourced, and there was **no temporal gate**. Both are fixed here:

  1. every fact carries a **source URL, a Wikidata QID, and a retrieval/published date**
  2. nothing whose source postdates the claim can enter the corpus

The temporal gate is not a nicety. AVeriTeC was built with multi-round annotation specifically to
avoid temporal leakage, and a synthetic pair citing a 2024 source for a 2019 claim teaches a
verifier to time-travel. That failure is invisible in aggregate accuracy, which is what makes it
worth a hard gate rather than a warning.

WIKIPEDIA/WIKIDATA AS THE FACT NET
----------------------------------
The operator's framing is right: Wikipedia and Wikidata *are* the verification knowledge graph,
already built. Wikidata additionally carries, per statement, the reference URL (`P854`) and the
retrieved-at date (`P813`), plus point-in-time qualifiers (`P585`) for values that change. That is
exactly the provenance a claim verifier needs, and it is free.

Three properties are harvested by default because each has a clean unit family that
`logos.logic.quantity` can compare:

    P2048 height      -> length family
    P2043 length      -> length family
    P1082 population  -> count family

CACHING IS NOT AN OPTIMISATION
------------------------------
A dataset built from a live SPARQL endpoint is not reproducible: the values change, and a
retraction is invisible. Every harvest is written to a cache file with its retrieval timestamp,
and the builder reads the cache by default. Re-harvesting is an explicit act, which is the right
default for anything that ends up in training data.

USAGE
-----
    python tools/logos/facts.py --self-test
    python tools/logos/facts.py --harvest 60 --out B:/datasets/forge/facts
    python tools/logos/facts.py --show --out B:/datasets/forge/facts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

SPARQL = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "logos-verify/1.0 (fact provenance research; python-urllib)"}

# property -> (label, unit string handed to quantity.py, family)
PROPERTIES = {
    "P2048": ("height", "m", "length"),
    "P2043": ("length", "m", "length"),
    "P1082": ("population", "people", "count"),
}

# Harvesting must constrain BOTH the property and the class. A property alone returned negative
# heights on entities named after nuclear test subsidence craters; the class is what makes a
# triple a fact about a structure.
CLASSES = {
    "bridge": "Q12280",
    "tower": "Q12518",
    "dam": "Q12323",
    "building": "Q41176",
}

# Plausible ranges per family. Anything outside is rejected rather than written to the cache -- a
# malformed fact in a training corpus is worse than a missing one.
RANGES = {
    "length": (0.5, 5000.0),        # metres
    "count": (100.0, 5e9),          # people
}


def sane(fact: "Fact") -> tuple[bool, str]:
    """Reject values that cannot be what the attribute claims."""
    lo, hi = RANGES.get(fact.family, (-1e12, 1e12))
    if fact.value <= 0:
        return False, f"non-positive value {fact.value:g}"
    if not (lo <= fact.value <= hi):
        return False, f"value {fact.value:g} outside plausible {fact.family} range [{lo:g}, {hi:g}]"
    if not fact.entity or fact.entity == fact.qid:
        return False, "no human-readable entity label"
    return True, "ok"


@dataclass
class Fact:
    qid: str
    entity: str
    attribute: str
    value: float
    unit: str
    family: str
    ref_url: str = ""
    retrieved: str = ""          # P813 retrieved-at, ISO date
    point_in_time: str = ""      # P585, ISO date
    source: str = "wikidata"
    fetched_at: str = ""         # when WE fetched it -- provenance about the provenance

    def as_synth_tuple(self) -> tuple:
        """Shape needed by per_evidence_synth: (entity, attribute, value, unit, alt, alt_attr)."""
        # a refuting value: +40% is clearly outside the 2% tolerance and stays physically plausible
        alt = round(self.value * 1.4, 2)
        alt_attr = {"height": "length", "length": "height",
                    "population": "area"}.get(self.attribute, "length")
        return (self.entity, self.attribute, self.value, self.unit, alt, alt_attr)


def harvest(prop: str = "P2048", limit: int = 12, timeout: float = 60.0,
            cls: str | None = "Q12280") -> list[Fact]:
    """Fetch facts for one property AND class, with provenance where Wikidata has it.

    `cls` is the Wikidata class of acceptable subjects (default bridge). Pass None to omit the
    constraint, which will produce junk -- see the CLASSES note.
    """
    if prop not in PROPERTIES:
        raise ValueError(f"unsupported property {prop}; known: {list(PROPERTIES)}")
    label, unit, family = PROPERTIES[prop]
    class_clause = f"?s wdt:P31/wdt:P279* wd:{cls} ." if cls else ""
    q = f"""SELECT ?s ?sLabel ?v ?refURL ?retrieved ?pit WHERE {{
  ?s wdt:{prop} ?v .
  {class_clause}
  ?s p:{prop} ?st .
  OPTIONAL {{ ?st prov:wasDerivedFrom ?ref .
             OPTIONAL {{ ?ref pr:P854 ?refURL . }}
             OPTIONAL {{ ?ref pr:P813 ?retrieved . }} }}
  OPTIONAL {{ ?st pq:P585 ?pit . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT {int(limit)}"""
    url = SPARQL + "?query=" + urllib.parse.quote(q) + "&format=json"
    import time
    last: Exception | None = None
    data = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            break
        except Exception as exc:          # noqa: BLE001 - endpoint 502s and times out
            last = exc
            time.sleep(2 + 3 * attempt)
    if data is None:
        raise RuntimeError(f"WDQS failed after 3 attempts: {type(last).__name__}: {last}")
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[Fact] = []
    seen: set[str] = set()
    for b in data["results"]["bindings"]:
        qid = b["s"]["value"].rsplit("/", 1)[-1]
        if qid in seen:                   # one fact per entity keeps the corpus non-duplicated
            continue
        seen.add(qid)
        try:
            value = float(b["v"]["value"])
        except (KeyError, ValueError):
            continue
        f = Fact(
            qid=qid, entity=b.get("sLabel", {}).get("value", qid),
            attribute=label, value=value, unit=unit, family=family,
            ref_url=b.get("refURL", {}).get("value", ""),
            retrieved=b.get("retrieved", {}).get("value", "")[:10],
            point_in_time=b.get("pit", {}).get("value", "")[:10],
            fetched_at=now)
        okv, why = sane(f)
        if not okv:
            continue                       # rejected values never reach the cache
        out.append(f)
    return out


def temporal_gate(fact: Fact, claim_date: str) -> tuple[bool, str]:
    """May this fact be used as evidence for a claim made on `claim_date`?

    Rejects a fact whose source is dated AFTER the claim. Uses the *earliest* known date
    (point-in-time, then retrieved-at) because the conservative direction is to reject.
    A fact with no date at all is rejected for claim-dated use -- we cannot show it predates the
    claim, and "unknown" is not a licence.
    """
    if not claim_date:
        return True, "no claim date; gate not applicable"
    src = fact.point_in_time or fact.retrieved
    if not src:
        return False, "no source date; cannot establish precedence"
    if src[:10] > claim_date[:10]:
        return False, f"source dated {src[:10]} postdates claim {claim_date[:10]}"
    return True, f"source {src[:10]} predates claim {claim_date[:10]}"


def save(facts: list[Fact], outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "facts.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")
    meta = os.path.join(outdir, "manifest.json")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"n": len(facts),
                   "with_ref_url": sum(1 for f in facts if f.ref_url),
                   "with_date": sum(1 for f in facts if f.retrieved or f.point_in_time),
                   "properties": sorted({f.attribute for f in facts})}, fh, indent=1)
    return p


def load(outdir: str) -> list[Fact]:
    p = os.path.join(outdir, "facts.jsonl")
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Fact(**json.loads(line)))
    return out


# ------------------------------------------------------------------------------- tests
SYNTH_FACTS = [
    Fact(qid="Q1", entity="the bridge", attribute="height", value=330, unit="m",
         family="length", ref_url="https://example.org/bridge", retrieved="2019-05-01",
         point_in_time="2018-01-01", fetched_at="2026-10-24T00:00:00Z"),
    Fact(qid="Q2", entity="the city", attribute="population", value=45000000, unit="people",
         family="count", ref_url="https://example.org/city", retrieved="2022-03-01",
         point_in_time="2020-01-01", fetched_at="2026-10-24T00:00:00Z"),
    Fact(qid="Q3", entity="the tower", attribute="height", value=1083, unit="m",
         family="length", ref_url="", retrieved="2024-11-01", point_in_time="",
         fetched_at="2026-10-24T00:00:00Z"),
    Fact(qid="Q4", entity="the dam", attribute="length", value=57, unit="m",
         family="length", ref_url="https://example.org/dam", retrieved="", point_in_time="",
         fetched_at="2026-10-24T00:00:00Z"),
]


def run_self_test() -> bool:
    print("logos.facts - SELF TEST")
    ok = True

    good = len(SYNTH_FACTS) == 4
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] fixture loaded ({len(SYNTH_FACTS)} facts)")

    # --- provenance is carried, not implied
    good = all(f.qid and f.fetched_at for f in SYNTH_FACTS)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every fact carries a QID and a fetch timestamp")

    withref = sum(1 for f in SYNTH_FACTS if f.ref_url)
    good = withref == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {withref}/4 facts carry a source URL "
          f"(the API does not always have one -- recorded as empty, not invented)")

    # --- temporal gate: source before claim passes
    okk, why = temporal_gate(SYNTH_FACTS[0], "2020-06-01")
    good = okk and "predates" in why
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] source 2018 predates claim 2020 -> {okk}")

    # --- temporal gate: source AFTER claim is rejected. This is the AVeriTeC trap.
    okk, why = temporal_gate(SYNTH_FACTS[2], "2020-06-01")
    good = (not okk) and "postdates" in why
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] source 2024 postdates claim 2020 -> REJECTED")

    # --- no date at all: rejected for dated claims, because unknown is not a licence
    okk, why = temporal_gate(SYNTH_FACTS[3], "2020-06-01")
    good = (not okk) and "cannot establish precedence" in why
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] undated source for a dated claim -> REJECTED")

    # --- and with no claim date the gate does not apply
    okk, _ = temporal_gate(SYNTH_FACTS[3], "")
    good = okk
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] undated source with an undated claim -> allowed")

    # --- the gate reports WHICH date it used
    good = "2018-01-01" in temporal_gate(SYNTH_FACTS[0], "2020-06-01")[1]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the gate names the date it compared against")

    # --- round-trip through the synth shape
    t = SYNTH_FACTS[0].as_synth_tuple()
    good = len(t) == 6 and t[0] == "the bridge" and t[1] == "height" and abs(t[4] - 462) < 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] synth tuple shape {t}")

    # --- the refuting value must fall outside the 2% tolerance
    from logos.logic.quantity import compare
    claim = f"{t[0]} has a {t[1]} of {t[2]:g} {t[3]}"
    alt = f"{t[0]} has a {t[1]} of {t[4]:g} {t[3]}"
    c = compare(claim, alt)
    good = c.verdict == "refute" and c.residual > 0.02
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] generated alt value refutes the claim "
          f"(residual {c.residual:.2f})")

    # --- save/load round-trip is exact (reproducibility)
    import tempfile
    d = tempfile.mkdtemp()
    save(SYNTH_FACTS, d)
    back = load(d)
    good = [asdict(a) for a in back] == [asdict(b) for b in SYNTH_FACTS]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] save/load round-trip is exact")

    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    good = man["n"] == 4 and man["with_ref_url"] == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] manifest records n and provenance counts "
          f"({man})")

    # --- the sanity gate rejects the junk the unconstrained query actually returned
    bad1 = Fact(qid="Q9", entity="Fisher", attribute="height", value=-1193, unit="m",
                family="length", fetched_at="2026-10-24T00:00:00Z")
    good = not sane(bad1)[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] negative height rejected "
          f"({sane(bad1)[1]})")

    bad2 = Fact(qid="Q10", entity="the hall", attribute="height", value=98000, unit="m",
                family="length", fetched_at="2026-10-24T00:00:00Z")
    good = not sane(bad2)[0] and "plausible" in sane(bad2)[1]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] implausible height rejected "
          f"({sane(bad2)[1][:44]})")

    bad3 = Fact(qid="Q11", entity="Q11", attribute="height", value=30, unit="m",
                family="length", fetched_at="2026-10-24T00:00:00Z")
    good = not sane(bad3)[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] unlabelled entity rejected")

    good = sane(SYNTH_FACTS[0])[0]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a real structure fact passes the gate")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Facts with provenance + temporal gate.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--harvest", type=int, default=0, metavar="N")
    ap.add_argument("--prop", default="P2048")
    ap.add_argument("--out", default=r"B:\datasets\forge\facts")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.harvest:
        allf: list[Fact] = []
        for prop, cls in (("P2048", "Q12280"), ("P2043", "Q12280"),
                          ("P2048", "Q12518"), ("P2048", "Q12323"),
                          ("P1082", "Q515")):
            try:
                got = harvest(prop, args.harvest, cls=cls)
                print(f"[facts] {prop}: {len(got)} facts, "
                      f"{sum(1 for f in got if f.ref_url)} with a reference URL")
                allf += got
            except Exception as exc:      # noqa: BLE001 - network is not guaranteed
                print(f"[facts] {prop}: FAILED {type(exc).__name__}: {exc}")
        if allf:
            p = save(allf, args.out)
            print(f"[facts] wrote {len(allf)} -> {p}")
        return 0
    if args.show or os.path.exists(os.path.join(args.out, "facts.jsonl")):
        fs = load(args.out)
        print(f"[facts] {len(fs)} facts from {args.out}")
        print(f"[facts] with ref_url: {sum(1 for f in fs if f.ref_url)}/{len(fs)}")
        print(f"[facts] with date   : "
              f"{sum(1 for f in fs if f.retrieved or f.point_in_time)}/{len(fs)}")
        for f in fs[:8]:
            print(f"  {f.entity[:34]:34s} {f.attribute:11s} {f.value:>12g} {f.unit:6s} "
                  f"{f.ref_url[:34] or '(no url)'}")
        return 0
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
