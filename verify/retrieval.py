"""retrieval.py — a provider ladder that survives the search stack being blocked.

WHY THIS FILE EXISTS
--------------------
The fact-checker measured **coverage 0.00** on HaluEval. The cause was not the
verifier, the judge, or the decomposition. It was retrieval: SearXNG returned
ZERO results for every query, because every engine behind it was suspended —

    brave       Suspended: too many requests
    duckduckgo  CAPTCHA
    google cse  "unusual traffic from your network"
    youtube-cc  403
    media-search crash

A single-provider retrieval layer turns an entire verification stack into a
no-op and reports "No evidence found", which looks like an honest abstention.
That is the worst kind of failure: silent, plausible, and total.

This module makes retrieval a LADDER, so one blocked provider degrades quality
instead of deleting the capability.

PROVIDERS (in order)
--------------------
1. `wikipedia`  — MediaWiki API. Free, unblocked, no key, high-quality prose.
                  Search then fetch the intro extract. This is the workhorse.
2. `wikidata`   — entity search (structured facts; also used for deterministic
                  verdicts in fact_verify).
3. `searxng`    — the local meta-search. Kept in the ladder because it is broad
                  when healthy, but it is NOT trusted as primary.

Each provider returns the same shape: [{"title", "url", "content", "provider"}]
so it is a drop-in replacement for `fact_verify.search`.

HONEST LIMIT
------------
Wikipedia is an encyclopaedia, not ground truth. It is excellent for "is X a
documentary" and useless for contested or very recent claims. A verdict built
only on Wikipedia is a verdict about Wikipedia.

USAGE
-----
    python tools/verify/retrieval.py --self-test
    python tools/verify/retrieval.py --query "Give Us Our Skeletons documentary"
    python tools/verify/retrieval.py --diagnose        # per-provider health
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "forge-factcheck/1.0 (research; local model zoo)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARX = "http://localhost:8888/search"

_TAG = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Rate limiting + caching.
#
# Measured: a self-test followed by a --diagnose earned HTTP 429 from Wikimedia
# within a minute. A fact-checker that gets rate-limited into silence is as
# useless as one with a dead backend, and it fails the same way -- quietly.
# So: throttle, back off on 429 (honouring Retry-After), and cache to disk.
# ---------------------------------------------------------------------------

CACHE_DIR = r"B:\datasets\forge\cache\retrieval"
_MIN_INTERVAL = 0.35          # seconds between outbound requests, per process
_last_call = [0.0]
_ua = {"User-Agent": "forge-factcheck/1.0 (https://local; research use) python-urllib"}


def _throttle() -> None:
    import time as _t
    dt = _t.time() - _last_call[0]
    if dt < _MIN_INTERVAL:
        _t.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = _t.time()


def _cache_path(url: str) -> str:
    import hashlib as _h
    import os as _os
    _os.makedirs(CACHE_DIR, exist_ok=True)
    return _os.path.join(CACHE_DIR, _h.sha256(url.encode()).hexdigest()[:32] + ".json")


def _get_json(url: str, timeout: int = 20, retries: int = 4):
    import os as _os
    import time as _t

    cp = _cache_path(url)
    if _os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass

    last = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers=_ua)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            try:
                with open(cp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
            except Exception:
                pass
            return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                wait = exc.headers.get("Retry-After")
                delay = float(wait) if (wait or "").strip().isdigit() else 2.0 * (attempt + 1)
                _t.sleep(min(delay, 20.0))
                continue
            if 500 <= exc.code < 600:
                _t.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last = exc
            _t.sleep(1.0 * (attempt + 1))
    if last:
        raise last
    return {}


def _clean(s: str) -> str:
    return " ".join(_TAG.sub("", s or "").split())


# --------------------------------------------------------------------------
# Provider 1: Wikipedia
# --------------------------------------------------------------------------

def wikipedia(query: str, k: int = 4) -> list[dict]:
    """Search then fetch intro extracts. Two calls, no key, no rate-limit wall."""
    q = (query or "").strip()
    if not q:
        return []
    url = (f"{WIKI_API}?action=query&list=search&format=json&srlimit={max(k, 1)}"
           f"&srsearch={urllib.parse.quote(q)}")
    d = _get_json(url)
    hits = (d.get("query") or {}).get("search") or []
    titles = [h["title"] for h in hits if h.get("title")]
    if not titles:
        return []

    out = []
    # batch the extract fetch
    t = "|".join(urllib.parse.quote(x) for x in titles[:k])
    eurl = (f"{WIKI_API}?action=query&prop=extracts&exintro=1&explaintext=1"
            f"&format=json&redirects=1&titles={t}")
    try:
        ed = _get_json(eurl)
        pages = (ed.get("query") or {}).get("pages") or {}
        for pid, p in pages.items():
            title = p.get("title") or ""
            extract = _clean(p.get("extract") or "")
            if not extract:
                continue
            out.append({
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "content": extract[:1200],
                "provider": "wikipedia",
            })
    except Exception:
        pass

    if not out:
        # extracts failed; fall back to search snippets so we still return evidence
        for h in hits[:k]:
            out.append({
                "title": h.get("title", ""),
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
                    (h.get("title") or "").replace(" ", "_")),
                "content": _clean(h.get("snippet") or "")[:600],
                "provider": "wikipedia:snippet",
            })
    return out[:k]


# --------------------------------------------------------------------------
# Provider 2: Wikidata entity search (structured; titles only)
# --------------------------------------------------------------------------

def wikidata(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    url = ("https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json"
           f"&language=en&limit={max(k, 1)}&search={urllib.parse.quote(q)}")
    d = _get_json(url)
    out = []
    for s in (d.get("search") or [])[:k]:
        desc = s.get("description") or ""
        label = s.get("label") or ""
        if not (label or desc):
            continue
        out.append({
            "title": label,
            "url": f"https://www.wikidata.org/wiki/{s.get('id', '')}",
            "content": f"{label} — {desc}".strip(" —"),
            "provider": "wikidata",
        })
    return out


# --------------------------------------------------------------------------
# Provider 3: local SearXNG (kept, not trusted)
# --------------------------------------------------------------------------

def searxng(query: str, k: int = 5, timeout: int = 15) -> list[dict]:
    url = f"{SEARX}?q={urllib.parse.quote(query)}&format=json"
    d = _get_json(url, timeout=timeout)
    out = []
    for item in (d.get("results") or [])[:k]:
        content = _clean(item.get("content") or "")
        if not content:
            continue
        out.append({
            "title": _clean(item.get("title") or ""),
            "url": item.get("url") or "",
            "content": content[:600],
            "provider": "searxng",
        })
    return out


PROVIDERS = {
    "wikipedia": wikipedia,
    "wikidata": wikidata,
    "searxng": searxng,
}

# Wikipedia first: it is the only one that is reliably up on this network.
DEFAULT_ORDER = ("wikipedia", "wikidata", "searxng")


def search(query: str, k: int = 5, order: tuple[str, ...] = DEFAULT_ORDER,
           min_results: int = 1) -> list[dict]:
    """Walk the ladder until a provider yields enough evidence.

    Returns [] only if EVERY provider failed — and the caller must treat that as
    "retrieval unavailable", not as "no evidence exists". Conflating those two is
    what made a dead search stack look like an honest abstention.
    """
    collected: list[dict] = []
    for name in order:
        fn = PROVIDERS.get(name)
        if not fn:
            continue
        try:
            got = fn(query, k=k)
        except Exception:
            continue
        if got:
            collected.extend(got)
            # dedupe by url/title as we go
            seen, uniq = set(), []
            for s in collected:
                key = s.get("url") or s.get("title")
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            collected = uniq
            if len(collected) >= min_results:
                break
    return collected[:max(k, 1) * 2]


def diagnose(queries: list[str] | None = None) -> dict:
    queries = queries or ["capital of France", "Rayleigh scattering", "Python asyncio"]
    out = {}
    for name, fn in PROVIDERS.items():
        per = []
        for q in queries:
            try:
                n = len(fn(q, k=3))
            except Exception as exc:
                per.append({"q": q, "error": f"{type(exc).__name__}: {str(exc)[:60]}"})
                continue
            per.append({"q": q, "results": n})
        out[name] = per
    return out


# --------------------------------------------------------------------------
# Self-test (offline-safe where possible)
# --------------------------------------------------------------------------

def run_self_test() -> bool:
    print("retrieval.py - SELF TEST")
    ok = True

    # cleaning must strip HTML tags and collapse whitespace
    good = _clean("<span class='m'>Blue</span>  sky\n is") == "Blue sky is"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] _clean strips tags/whitespace")

    # empty query must not hit the network
    good = wikipedia("") == [] and wikidata("") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty query returns [] without a request")

    # ladder must tolerate a dead provider without losing the others
    class Boom:
        def __call__(self, q, k=3):
            raise RuntimeError("provider down")

    saved = dict(PROVIDERS)
    try:
        PROVIDERS["dead"] = Boom()
        got = search("Rayleigh scattering", k=2, order=("dead", "wikipedia"))
        good = len(got) > 0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] dead provider is skipped, "
              f"next one still answers ({len(got)} results)")
    finally:
        PROVIDERS.clear()
        PROVIDERS.update(saved)

    # a real lookup: this is the claim that returned 0 results via SearXNG
    try:
        got = wikipedia("Give Us Our Skeletons documentary", k=2)
        hit = any("documentary" in (s.get("content") or "").lower() for s in got)
        good = bool(got) and hit
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] wikipedia answers the claim that "
              f"SearXNG could not ({len(got)} results, documentary={hit})")
    except Exception as exc:
        print(f"  [SKIP] live wikipedia check ({type(exc).__name__})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Provider-ladder retrieval.")
    ap.add_argument("--query")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if args.diagnose:
        print(json.dumps(diagnose(), indent=1))
        return 0
    if args.query:
        print(json.dumps(search(args.query, args.k), indent=1, ensure_ascii=True)[:3000])
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
