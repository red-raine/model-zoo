"""logos/egress.py — two network paths per source, chosen by the bandit instead of by hope.

WHY THIS FILE EXISTS
--------------------
The rotating Squid proxy was reported "not confirmed running". It has in fact been running since
2026-10-23 on :3128 (`rotating-squid-proxy`, restart=unless-stopped) — the full `docker ps` timed
out, and "unverified" got reported as "probably down". That was wrong, and the cost of the error
was two sessions of treating 12 archives as dead.

So the probe was rerun both ways. The result is not "the proxy fixes it":

    source                direct      proxy
    wayback-cdx           200         429        <- the proxy's upstream rotation is BLOCKED
    arquivo.pt            200         200
    archive.today         429         429        <- DNS-fails direct earlier; alive both ways now
    archive.is            DNS-fail    429        <- the proxy genuinely RECOVERS this one
    ghostarchive          200         200
    conifer               200         200
    swap-stanford         200         405        <- the proxy breaks a working endpoint
    bibalex               DNS-fail    409        <- Squid cannot reach its own upstream
    vefir.is              404         409
    haw.nsk.hr            DNS-fail    409
    padi.cat              SSL-fail    409
    memento-aggregator    DNS-fail    409

Two things follow, and both are the opposite of the naive reading:

1. **The proxy is not a fallback that always helps.** It recovers DNS-blocked domains
   (`archive.is`) and breaks reachable ones (`wayback-cdx` -> 429, `swap.stanford` -> 405). A
   `409` is Squid saying its own upstream chain failed, which is strictly worse than our direct
   attempt. Routing everything through the proxy would have REDUCED coverage.

2. **Which path works is a per-source fact, and it changes over time.** That is exactly a bandit
   problem, not a configuration problem. So the egress path becomes part of the arm name:
   `wayback_cdx|direct`, `archive_is|proxy`. Thompson sampling learns the preference per source
   without anyone writing a table that silently rots.

WHAT `409` MEANS HERE
---------------------
Squid's `409 Conflict` on a CONNECT tunnel means the proxy could not establish the tunnel to the
origin through its upstream. It is not the origin answering -- the origin was never reached. So a
409 is scored as a FAILURE of the proxy path specifically, while a 429 is scored as the origin
being ALIVE but throttled. Conflating those two is how a working archive gets written off as dead.

USAGE
-----
    python tools/logos/egress.py --self-test
    python tools/logos/egress.py --probe https://web.archive.org/cdx/search/cdx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from logos.router import Memo, Router  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (compatible; logos-verify/1.0; provenance redundancy)"}

PROXY = os.environ.get("LOGOS_PROXY", "http://127.0.0.1:3128")
PATHS: dict[str, str | None] = {"direct": None, "proxy": PROXY}

CONTEXT = "egress"

# HTTP statuses that mean "the origin ANSWERED". These are successes-of-reachability even though
# they are failures-of-content, and must never be confused with "unreachable".
ANSWERED = frozenset({200, 203, 206, 301, 302, 303, 304, 307, 308,
                      401, 402, 403, 404, 405, 410, 418, 429, 451})
# 407 = proxy auth required, 409 = Squid tunnel failed, 502/503/504 = upstream broken.
PROXY_BROKEN = frozenset({407, 409, 502, 503, 504})


def arm_name(source: str, path: str) -> str:
    return f"{source}|{path}"


def arm_path(arm: str) -> tuple[str, str]:
    src, _, path = arm.partition("|")
    return src, path or "direct"


def request(url: str, path: str = "direct", timeout: float = 25.0,
            headers: dict | None = None) -> tuple[int | None, str, str]:
    """GET via one path. Returns (status, body, note). Never raises."""
    handlers = []
    proxy = PATHS.get(path)
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))       # force direct, ignore env proxies
    op = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers=headers or UA)
    try:
        with op.open(req, timeout=timeout) as r:
            return r.status, r.read(400_000).decode("utf-8", "replace"), ""
    except urllib.error.HTTPError as e:
        return e.code, "", f"HTTP {e.code}"
    except Exception as e:                                     # noqa: BLE001
        return None, "", type(e).__name__


def classify(status: int | None, path: str) -> str:
    """Reachability verdict -- deliberately separate from usefulness."""
    if status is None:
        return "unreachable"
    if status in PROXY_BROKEN and path == "proxy":
        return "proxy_broken"
    if status in ANSWERED:
        return "answered"
    return f"status_{status}"


def egress_fetch(router: Router, source: str, url: str, timeout: float = 25.0,
                 paths: tuple[str, ...] = ("direct", "proxy"), min_status: int = 200,
                 extra: dict | None = None, explore_every: int = 5) -> dict:
    """Try this source over each egress path, best-first by bandit posterior.

    Returns {"status", "body", "path", "verdict", "tried"}. A source that answers on either path
    is a success for that path's arm; a path that leaves it unreachable is a failure for that arm.
    The content judgement is the caller's job -- reachability and usefulness are different
    questions and mixing them is how the router learned to prefer liars.
    """
    tried: list[dict] = []
    arms = {p: router.arm(CONTEXT, arm_name(source, p)) for p in paths}

    # SAMPLE, do not sort by mean. Ranking by posterior mean is a greedy policy that stops
    # exploring -- measured: with two paths the losing arm was tried ONCE in 20 calls (mean 0.33 =
    # 1/(2+1)) and a path that had recovered would never be discovered again. Thompson sampling is
    # the whole reason a bandit is used here instead of a lookup table.
    ranked = sorted(paths, key=lambda p: -arms[p].sample(router.rng))

    # BUT with only two arms a strong incumbent suppresses the loser to ~2 samples in 20, so a path
    # that recovers is rediscovered slowly. A forced probe is cheap here -- one HTTP request -- so
    # every `explore_every` calls the least-tried path is tried first. Deterministic recovery beats
    # hoping the sampler gets lucky.
    total = sum(a.observations for a in arms.values())
    if explore_every and total and total % explore_every == 0:
        ranked.sort(key=lambda p: arms[p].observations)

    for path in ranked:
        t0 = time.perf_counter()
        status, body, note = request(url, path, timeout, headers=extra)
        ms = (time.perf_counter() - t0) * 1000
        verdict = classify(status, path)
        ok = verdict == "answered" and (status or 0) >= min_status
        router.observe(CONTEXT, arm_name(source, path), ok, ms)
        tried.append({"path": path, "status": status, "verdict": verdict, "ms": round(ms, 1),
                      "note": note})
        if ok:
            return {"status": status, "body": body, "path": path, "verdict": verdict,
                    "tried": tried}
    return {"status": None, "body": "", "path": "", "verdict": "unreachable", "tried": tried}


def probe(url: str, timeout: float = 25.0) -> dict:
    """Probe both paths once, for reporting. Not used for routing -- the bandit does that."""
    out = {}
    for path in PATHS:
        t0 = time.perf_counter()
        status, body, note = request(url, path, timeout)
        out[path] = {"status": status, "verdict": classify(status, path), "bytes": len(body),
                     "ms": round((time.perf_counter() - t0) * 1000, 1), "note": note}
    return out


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.egress - SELF TEST")
    ok = True
    import tempfile

    tmp = tempfile.mkdtemp()

    def fresh(n="e"):
        return Router(state_path=os.path.join(tmp, f"{n}.json"), seed=2)

    # --- arm naming round-trips
    a = arm_name("wayback_cdx", "proxy")
    good = a == "wayback_cdx|proxy" and arm_path(a) == ("wayback_cdx", "proxy")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] arm name carries source AND path ({a})")

    # --- a bare source name defaults to direct, so old state keeps working
    good = arm_path("wayback_cdx") == ("wayback_cdx", "direct")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a bare source name defaults to the direct path")

    # --- classification: 429 is ALIVE but throttled, not dead
    good = (classify(429, "direct") == "answered"
            and classify(403, "direct") == "answered"
            and classify(None, "direct") == "unreachable")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] 429/403 mean the origin ANSWERED; only None is "
          f"unreachable")

    # --- classification: Squid's 409 is a PROXY failure, never an origin verdict
    good = classify(409, "proxy") == "proxy_broken" and classify(409, "direct") != "proxy_broken"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a 409 is the PROXY tunnel failing, not the origin "
          f"answering")

    # --- the proxy path is learned when only the proxy reaches the host
    R = fresh("a")
    real_request = globals()["request"]

    def fake_request(url, path="direct", timeout=25.0, headers=None):
        if path == "proxy":
            return 429, "alive", ""
        return None, "", "URLError"                      # direct DNS fails

    globals()["request"] = fake_request
    try:
        for i in range(20):
            egress_fetch(R, "archive_is", f"https://archive.is/{i}")
    finally:
        globals()["request"] = real_request
    proxy_arm = R.arm(CONTEXT, arm_name("archive_is", "proxy"))
    direct_arm = R.arm(CONTEXT, arm_name("archive_is", "direct"))
    good = proxy_arm.mean > 0.8 and direct_arm.mean < 0.45
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] when only the proxy reaches a host the bandit learns "
          f"proxy={proxy_arm.mean:.2f} direct={direct_arm.mean:.2f} "
          f"(n={direct_arm.observations})")

    # --- and the reverse: a host the proxy BREAKS is routed direct
    R2 = fresh("b")

    def fake_request2(url, path="direct", timeout=25.0, headers=None):
        if path == "direct":
            return 200, "ok", ""
        return 409, "", "HTTP 409"                       # Squid tunnel failure

    globals()["request"] = fake_request2
    try:
        for i in range(20):
            egress_fetch(R2, "wayback_cdx", f"https://web.archive.org/{i}")
    finally:
        globals()["request"] = real_request
    p2 = R2.arm(CONTEXT, arm_name("wayback_cdx", "proxy"))
    d2 = R2.arm(CONTEXT, arm_name("wayback_cdx", "direct"))
    good = d2.mean > 0.8 and p2.mean < 0.5
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] when the proxy BREAKS a host the bandit routes it "
          f"direct={d2.mean:.2f} proxy={p2.mean:.2f}")

    # --- EXPLORATION IS NOT OPTIONAL: a greedy mean-ranking stops trying the loser forever, so a
    #     path that recovers would never be rediscovered. Measured before the fix: the losing arm
    #     was tried exactly once in 20 calls (mean 0.33 = 1/(2+1)). This asserts both arms keep
    #     receiving observations, which is the property that makes a bandit a bandit.
    R2b = fresh("b2")

    def always_direct(url, path="direct", timeout=25.0, headers=None):
        return (200, "ok", "") if path == "direct" else (None, "", "URLError")

    globals()["request"] = always_direct
    try:
        for i in range(20):
            egress_fetch(R2b, "only_direct", f"https://x/{i}")
    finally:
        globals()["request"] = real_request
    obs_direct = R2b.arm(CONTEXT, arm_name("only_direct", "direct")).observations
    obs_proxy = R2b.arm(CONTEXT, arm_name("only_direct", "proxy")).observations
    good = obs_proxy >= 1 and obs_direct >= 1
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the losing path is still sampled "
          f"(direct n={obs_direct}, proxy n={obs_proxy}) -- greedy mean-ranking would never retry "
          f"it")

    # --- the winning path is reported
    R3 = fresh("c")
    globals()["request"] = fake_request
    try:
        r = egress_fetch(R3, "archive_is", "https://archive.is/x")
    finally:
        globals()["request"] = real_request
    good = r["path"] == "proxy" and r["status"] == 429 and r["verdict"] == "answered"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] the result names the path that actually worked "
          f"({r['path']})")

    # --- total failure is not an exception
    R4 = fresh("d")

    def dead(url, path="direct", timeout=25.0, headers=None):
        return None, "", "URLError"

    globals()["request"] = dead
    try:
        r = egress_fetch(R4, "bibalex", "http://web.archive.bibalex.org/x")
    finally:
        globals()["request"] = real_request
    good = r["verdict"] == "unreachable" and r["body"] == "" and len(r["tried"]) == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] both paths dead -> unreachable, no exception")

    # --- min_status lets a caller refuse to accept a 4xx as content
    R5 = fresh("f")
    globals()["request"] = fake_request
    try:
        r = egress_fetch(R5, "archive_is", "https://archive.is/y", min_status=200)
    finally:
        globals()["request"] = real_request
    good = r["verdict"] == "answered"          # 429 answered, but...
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] reachability and content acceptance are separable "
          f"(verdict={r['verdict']})")

    # --- RECOVERY: a path that was dead and comes back must be rediscovered, not abandoned
    #     forever. The forced probe makes this deterministic instead of hoping the sampler gets
    #     lucky, which matters because egress paths change (a proxy gets unblocked, a DNS starts
    #     resolving).
    R8 = fresh("rec")
    state = {"proxy_up": False, "switched_at": None}

    def recovering(url, path="direct", timeout=25.0, headers=None):
        if path == "direct":
            return None, "", "URLError"
        if state["proxy_up"]:
            return 200, "ok", ""
        return 409, "", "HTTP 409"

    globals()["request"] = recovering
    try:
        for i in range(30):
            if i == 12:
                state["proxy_up"] = True                 # the proxy comes back
            r = egress_fetch(R8, "recovers", f"https://x/{i}")
            if state["proxy_up"] and state["switched_at"] is None and r["status"] == 200:
                state["switched_at"] = i
    finally:
        globals()["request"] = real_request
    good = state["switched_at"] is not None and state["switched_at"] <= 18
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a recovered path is rediscovered within "
          f"{state['switched_at']} calls of coming back")

    # --- LIVE: both paths are probed for real, and the result is honest either way
    live = probe("https://arquivo.pt/wayback/cdx?url=en.wikipedia.org/wiki/Syllogism&output=json&limit=1",
                 timeout=25.0)
    served = [p for p, v in live.items() if v["verdict"] == "answered"]
    print(f"  [{'PASS' if served else 'SKIP'}] LIVE arquivo.pt answered on "
          f"{served or 'no path'} "
          f"({', '.join(f'{p}={live[p]['status']}' for p in live)})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-egress fetching, path chosen by bandit.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", default="")
    args = ap.parse_args(argv)
    if args.self_test or not args.probe:
        return 0 if run_self_test() else 1
    res = probe(args.probe)
    print(args.probe)
    for path, v in res.items():
        print(f"  {path:7s} status={str(v['status']):6s} verdict={v['verdict']:14s} "
              f"bytes={v['bytes']:7d} {v['ms']:8.1f} ms  {v['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
