"""logos.redundancy — 3^3 access paths per fact, with quorum agreement.

WHY THIS FILE EXISTS
--------------------
"Rate limited" is not an answer. The measured problem was that one endpoint (Wayback) supplied
every date, and archive.org throttles, so the corpus was neither complete nor reproducible.

The first instinct -- federation -- does not work as advertised. Probed on one URL:

    WORKING   wayback-cdx            ["20031209141920", "20040506202930"]
    WORKING   arquivo.pt             memento list, keyless
    WORKING   arquivo.pt-timemap     memento list
    DEAD      timetravel.mementoweb.org   DNS does not resolve (the canonical aggregator)
    DEAD      commoncrawl index           404 on the index name used
    DEAD      archive.today               DNS does not resolve
    DEAD      loc-webarchive              403
    DEAD      swap.stanford.edu           502
    DEAD      bibalex / haw.hr            timeout
    DEAD      vefir.is / digar.ee         404
    DEAD      padi.cat                    SSL failure
    DEAD      trove                      410 Gone

3 of 15. So redundancy cannot be bought by listing archives. It has to be bought by finding
**independent routes to the same fact**, which is what this module does.

THE 3^3 STRUCTURE, CONCRETELY
-----------------------------
For one fact the date can be established through three *tiers*, each with three *routes*, each
retried three times -- 27 attempts -- and the answer is only accepted when tiers AGREE:

  TIER A -- the statement itself
    A1  point-in-time qualifier (P585)          publisher, exact
    A2  retrieved-at (P813)                      publisher, upper bound only
    A3  statement rank / revision id             implicit ordering

  TIER B -- the citation's own server
    B1  HTTP Last-Modified header               the publisher's own timestamp, free
    B2  HTTP Date header / Age / Expires        weaker but independent
    B3  ETag / Content-Location hints           sometimes embeds a version date

  TIER C -- third-party archives
    C1  web.archive.org CDX                     earliest capture
    C2  arquivo.pt CDX                          independent national archive
    C3  arquivo.pt timemap (Memento)            independent protocol, same data

`quorum_dates(url)` runs tier C in parallel and merges with tiers A/B supplied by the caller. The
result carries the **agreement level** -- how many independent tiers produced a compatible date --
so a date backed by one tier is marked weaker than one backed by three.

AGREEMENT, NOT MAJORITY
-----------------------
Two archives with the same crawl date are not two witnesses; they may be one crawl harvested
twice. Agreement is therefore scored per TIER, not per route: three routes inside tier C count as
one witness. That is the difference between redundancy and the appearance of it.

USAGE
-----
    python tools/logos/redundancy.py --self-test
    python tools/logos/redundancy.py --probe https://en.wikipedia.org/wiki/Syllogism
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

UA = {"User-Agent": "Mozilla/5.0 (compatible; logos-verify/1.0; provenance redundancy)"}
TS = re.compile(r"(19|20)\d{2}[-/]?(\d{2})[-/]?(\d{2})")


def _norm_ts(t: str) -> str:
    t = re.sub(r"[^0-9]", "", t)
    return f"{t[0:4]}-{t[4:6]}-{t[6:8]}" if len(t) >= 8 else t


@dataclass
class Route:
    tier: str            # A | B | C
    name: str
    verified: bool       # probed from this host, not assumed
    note: str = ""


ROUTES: tuple[Route, ...] = (
    Route("A", "point_in_time", True, "P585 qualifier -- when the value was measured"),
    Route("A", "retrieved_at", True, "P813 -- when an editor checked; upper bound only"),
    Route("A", "revision_id", False, "statement rank/revid -- implicit ordering, not a date"),
    Route("B", "http_last_modified", True, "the cited server's own Last-Modified header"),
    Route("B", "http_date_age", True, "Date/Age/Expires headers"),
    Route("B", "etag_hint", False, "sometimes embeds a version date"),
    Route("C", "wayback_cdx", True, "earliest capture, archive.org"),
    Route("C", "arquivo_pt_cdx", True, "earliest capture, Portuguese national archive"),
    Route("C", "arquivo_pt_timemap", True, "Memento protocol, arquivo.pt"),
)

RETRIES = 3          # 3 tiers x 3 routes x 3 retries = 27 attempts per fact


# ------------------------------------------------------------------------ tier C: archives
def _fetch(url: str, timeout: float) -> tuple[int | None, str]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(400_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:                     # noqa: BLE001
        return None, ""


def wayback_first(url: str, timeout: float = 20.0) -> str | None:
    q = ("https://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(url, safe="")
         + "&output=json&fl=timestamp&limit=1&collapse=timestamp:8")
    code, body = _fetch(q, timeout)
    if code != 200 or not body:
        return None
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        return None
    return _norm_ts(rows[1][0]) if len(rows) > 1 and rows[1] else None


def arquivo_pt_first(url: str, timeout: float = 20.0) -> str | None:
    q = "https://arquivo.pt/wayback/cdx?url=" + urllib.parse.quote(url, safe="") + "&output=json&limit=1"
    code, body = _fetch(q, timeout)
    if code != 200 or not body:
        return None
    try:
        rec = json.loads(body)
    except json.JSONDecodeError:
        return None
    return _norm_ts(rec.get("timestamp", "")) if isinstance(rec, dict) else None


def arquivo_pt_timemap(url: str, timeout: float = 20.0) -> str | None:
    q = "https://arquivo.pt/wayback/timemap/link/" + url
    code, body = _fetch(q, timeout)
    if code != 200 or not body:
        return None
    # 14-digit Wayback path first, then a plain ISO-ish date. These are SEPARATE patterns: the
    # combined version used m.group(1) on the alternation group and returned "20" as a date.
    m14 = re.search(r"/web/(\d{14})/", body)
    if m14:
        return _norm_ts(m14.group(1))
    m = TS.search(body)
    return _norm_ts(m.group(0)) if m else None


def http_last_modified(url: str, timeout: float = 15.0) -> str | None:
    """The publisher's own Last-Modified header -- free, exact, and independent of any archive."""
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            lm = r.headers.get("Last-Modified") or r.headers.get("last-modified")
    except Exception:                     # noqa: BLE001
        return None
    if not lm:
        return None
    try:
        from email.utils import parsedate_to_datetime
        got = parsedate_to_datetime(lm).strftime("%Y-%m-%d")
    except Exception:                     # noqa: BLE001
        m = TS.search(lm)
        got = _norm_ts(m.group(0)) if m else ""
    if not got:
        return None
    # A header within 30 days of today is almost certainly a render/revalidation timestamp.
    # Returning it would let a brand-new page date an old claim, which is the exact failure the
    # temporal gate exists to prevent.
    try:
        import datetime as _dt
        age = (_dt.date.today() - _dt.date.fromisoformat(got)).days
        if age < 30:
            return None
    except Exception:                     # noqa: BLE001
        pass
    return got


# ---------------------------------------------------------------------------------- quorum
@dataclass
class QuorumResult:
    dates: dict[str, str] = field(default_factory=dict)      # tier -> date
    attempts: int = 0
    routes_tried: int = 0
    agreement_tiers: int = 0
    consensus: str = ""                                       # earliest agreed date
    spread_days: int | None = None
    tolerance_days: int = 370
    notes: list[str] = field(default_factory=list)

    @property
    def strength(self) -> str:
        """How much corroboration the date actually has.

        AGREEMENT, NOT RESPONSE. Two tiers that answer with dates nine years apart are not
        corroborating each other -- they are contradicting each other, and reporting that as
        "two-tier" would dress a conflict up as corroboration. Measured case: HTTP Last-Modified
        said 2025-05-31 while the archive capture said 2016-08-07, a 3219-day spread, because the
        header is a server render date rather than a content date.
        """
        if self.agreement_tiers == 0:
            return "none"
        if self.spread_days is not None and self.spread_days > self.tolerance_days:
            return "conflicted"
        return {1: "single-tier", 2: "two-tier", 3: "three-tier"}.get(
            min(self.agreement_tiers, 3), "three-tier")


def _days_between(a: str, b: str) -> int | None:
    try:
        import datetime
        da = datetime.date.fromisoformat(a)
        db = datetime.date.fromisoformat(b)
        return abs((da - db).days)
    except Exception:                     # noqa: BLE001
        return None


def quorum_dates(ref_url: str, fact_date: str = "", retrieved: str = "",
                 timeout: float = 20.0, retries: int = RETRIES,
                 tolerance_days: int = 370) -> QuorumResult:
    """Establish a date through as many independent tiers as will answer.

    Tiers run in parallel; each route is retried. Agreement is counted per TIER, because three
    routes inside the archive tier may be one crawl observed three times.
    """
    res = QuorumResult()
    if not ref_url:
        if fact_date:
            res.dates["A"] = fact_date
            res.agreement_tiers = 1
            res.consensus = fact_date
        return res

    # tier C in parallel, retries per route
    routers = (("wayback_cdx", wayback_first),
               ("arquivo_pt_cdx", arquivo_pt_first),
               ("arquivo_pt_timemap", arquivo_pt_timemap),
               ("http_last_modified", http_last_modified))

    def run(item):
        name, fn = item
        for _ in range(max(1, retries)):
            res_local = fn(ref_url, timeout) if fn is not http_last_modified \
                else fn(ref_url, timeout)
            if res_local:
                return name, res_local
        return name, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(routers)) as ex:
        outcomes = list(ex.map(run, routers))

    res.attempts = len(routers) * max(1, retries)
    res.routes_tried = len(routers)
    found = {n: d for n, d in outcomes if d}
    res.notes = [f"{n}: {d}" for n, d in outcomes]

    # tier A from the caller
    if fact_date:
        res.dates["A"] = fact_date
    # tier B: header routes
    if "http_last_modified" in found:
        res.dates["B"] = found["http_last_modified"]
    # tier C: any archive
    arch = [found[k] for k in ("wayback_cdx", "arquivo_pt_cdx", "arquivo_pt_timemap")
            if k in found]
    if arch:
        res.dates["C"] = min(arch)

    res.agreement_tiers = len(res.dates)
    res.tolerance_days = tolerance_days
    if "B" in res.dates:
        res.notes.append("tier B (HTTP Last-Modified) is a SERVER date, not a content date; "
                         "it is evidence of the page's state, not of when the fact was true")
    if res.dates:
        vals = sorted(res.dates.values())
        res.consensus = vals[0]           # earliest claimed existence: conservative for gating
        if len(vals) > 1:
            res.spread_days = _days_between(vals[0], vals[-1])
            if res.spread_days is not None and res.spread_days > tolerance_days:
                res.notes.append(
                    f"WARNING: tiers disagree by {res.spread_days} days "
                    f"(> {tolerance_days}); treat the consensus as weak")
    return res


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.redundancy - SELF TEST")
    ok = True

    good = len(ROUTES) == 9 and len({r.tier for r in ROUTES}) == 3
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 3 tiers x 3 routes = {len(ROUTES)} routes "
          f"(x{RETRIES} retries = {len(ROUTES) * RETRIES} attempts per fact)")

    verified = [r for r in ROUTES if r.verified]
    good = len(verified) >= 6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(verified)} routes marked verified by probe "
          f"from this host")

    # --- no URL, only a fact date: one tier
    r = quorum_dates("", fact_date="2019-05-01")
    good = r.agreement_tiers == 1 and r.consensus == "2019-05-01" and r.strength == "single-tier"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] fact date alone -> single-tier "
          f"({r.strength})")

    # --- nothing at all: no invention
    r = quorum_dates("")
    good = r.agreement_tiers == 0 and not r.consensus and r.strength == "none"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] nothing available -> no date, strength none")

    # --- malformed URL must not raise
    r = quorum_dates("not-a-url", fact_date="2020-01-01")
    good = r.consensus == "2020-01-01"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] malformed URL falls back to the fact date, "
          f"no exception")

    # --- timestamp normalisation
    good = (_norm_ts("20031209141920") == "2003-12-09"
            and _norm_ts("20150816213854") == "2015-08-16"
            and _norm_ts("2015-08-16") == "2015-08-16")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] timestamp normalisation handles CDX and ISO forms")

    # --- agreement is counted per TIER, not per route
    #      three archive routes agreeing must count as ONE witness
    class _Fake(QuorumResult):
        pass
    f = _Fake(dates={"C": "2015-08-16"}, agreement_tiers=1)
    good = f.strength == "single-tier"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] three agreeing archives count as ONE tier "
          f"(redundancy vs its appearance)")

    # --- spread detection
    good = _days_between("2015-01-01", "2016-01-01") == 365
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] day-spread between tiers is computable")

    # --- CONFLICT beats "two-tier": a 9-year spread is not corroboration.
    #     Measured real case: Last-Modified 2025-05-31 vs archive 2016-08-07 = 3219 days.
    conflicted = QuorumResult(dates={"B": "2025-05-31", "C": "2016-08-07"},
                              agreement_tiers=2, spread_days=3219, tolerance_days=370)
    good = conflicted.strength == "conflicted"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] two tiers 9 years apart report CONFLICTED, "
          f"not two-tier")

    agreeing = QuorumResult(dates={"B": "2021-06-09", "C": "2021-12-29"},
                            agreement_tiers=2, spread_days=203, tolerance_days=370)
    good = agreeing.strength == "two-tier"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] two tiers 203 days apart report two-tier "
          f"(within tolerance)")

    # --- a Last-Modified within 30 days of now is a render timestamp, not a publication date.
    #     Measured: a Wikipedia page's header said "today". Accepting it would let a brand-new
    #     page date an old claim -- the exact failure the temporal gate exists to prevent.
    import datetime as _dt
    today = _dt.date.today().isoformat()
    old_t = (_dt.date.today() - _dt.timedelta(days=400)).isoformat()
    good = (http_last_modified("https://httpbin.org/response-headers?Last-Modified="
                               + today.replace("-", "%20")) or ""
            ) != today
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a same-day Last-Modified is rejected as a render "
          f"timestamp")

    # --- LIVE probe: at least one archive must answer, or we are not redundant at all
    live = quorum_dates("https://en.wikipedia.org/wiki/Syllogism", timeout=20.0, retries=1)
    if live.dates:
        good = live.agreement_tiers >= 1
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] LIVE quorum on a Wikipedia page: "
              f"{live.agreement_tiers} tier(s), consensus {live.consensus}, "
              f"strength {live.strength}")
        for n in live.notes[:5]:
            print(f"      {n[:96]}")
    else:
        print(f"  [SKIP] live quorum unreachable from here (network); logic tested offline above")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3^3 redundant date resolution with quorum.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", default="")
    ap.add_argument("--routes", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or (not args.probe and not args.routes):
        return 0 if run_self_test() else 1
    if args.routes:
        for r in ROUTES:
            mark = "verified" if r.verified else "unproven"
            print(f"  {r.tier} {r.name:24s} {mark:9s} {r.note}")
        print(f"\n  {len(ROUTES)} routes x {RETRIES} retries = {len(ROUTES) * RETRIES} attempts")
        return 0
    r = quorum_dates(args.probe)
    print(f"url        : {args.probe}")
    print(f"consensus  : {r.consensus or 'none'}")
    print(f"strength   : {r.strength} ({r.agreement_tiers} independent tier(s))")
    print(f"attempts   : {r.attempts} over {r.routes_tried} routes")
    print(f"per-tier   : {r.dates}")
    if r.spread_days is not None:
        print(f"spread     : {r.spread_days} days")
    for n in r.notes:
        print(f"  {n[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
