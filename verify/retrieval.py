"""retrieval.py — v2 ladder: own-index APIs first, metasearch second, crawler last.

WHY v2 EXISTS
-------------
v1 was a provider ladder over `wikipedia -> wikidata -> searxng`. That fixed the
"dead backend reports no evidence" failure, but two problems remained:

1. **SearXNG is structurally unreliable here.** Its defaults suspend an engine for
   24h on one CAPTCHA and **15 days** on one Cloudflare hit, so a single bad
   minute takes the whole meta-search tier down for a fortnight. It is a bonus
   tier, not a foundation.
2. **Snippet-level evidence is weak.** Judging a claim against a search snippet is
   judging it against a teaser. We have crawl4ai running locally and trafilatura
   installed, so we can fetch the actual page.

So v2 is organised by *trust and durability*, not by convenience:

    T1 OWN-INDEX   wikipedia · wikidata · openalex · crossref · stackexchange · arxiv
                   keyless, legal, own index, effectively never blocked
    T2 METASEARCH  searxng (bonus; partial success counts)
    T3 CRAWLER     crawl4ai /md  (JS pages, mild bot walls)   <- fetch, not search

`fetch_page()` is the new primitive: URL -> clean main-content text, with a
documented tier ladder and the tier recorded on the result. Provenance is not
optional — a groundedness score you cannot attribute is not a score.

HONEST LIMITS
-------------
* Own-index APIs cover encyclopaedic / scholarly / technical ground well and
  general web poorly. For "what did this person say yesterday" they are useless.
* crawl4ai v0.9 refuses per-request proxies (untrusted gate), so there is no
  stealth escalation through the API. It handles JS, not Cloudflare.
* Marginalia's public key rate-limits (429 seen). It is optional, not core.

USAGE
-----
    python tools/verify/retrieval.py --self-test
    python tools/verify/retrieval.py --query "Rayleigh scattering"
    python tools/verify/retrieval.py --fetch https://en.wikipedia.org/wiki/Web_crawler
    python tools/verify/retrieval.py --diagnose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "forge-retrieval/1.0 (research; local model zoo)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARX = "http://localhost:8888/search"
CRAWL4AI = "http://localhost:11235"
CRAWL4AI_KEY_FILE = r"C:\Users\myste\.config\opencode\keys\pool-crawl4ai.key"

CACHE_DIR = r"B:\datasets\forge\cache\retrieval"
_MIN_INTERVAL = 0.30
_last_call = [0.0]
_TAG = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------
# transport: throttle, backoff, disk cache
# --------------------------------------------------------------------------

def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")


def _request(url: str, data: bytes | None = None, headers: dict | None = None,
             timeout: int = 25, retries: int = 3, cache: bool = True):
    """Return (body_text, status). Cached on success; backs off on 429/5xx."""
    ck = f"{url}\0{data!r}" if data else url
    cp = _cache_path(ck) if cache else None
    if cp and os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as fh:
                return json.load(fh)["body"], json.load(open(cp, encoding="utf-8"))["status"]
        except Exception:
            pass

    last = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={**UA, **(headers or {})},
                                         method="POST" if data else "GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
                status = r.status
            if cp:
                try:
                    with open(cp, "w", encoding="utf-8") as fh:
                        json.dump({"body": body, "status": status}, fh)
                except Exception:
                    pass
            return body, status
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                ra = (exc.headers.get("Retry-After") or "").strip()
                time.sleep(min(float(ra) if ra.isdigit() else 2.0 * (attempt + 1), 15.0))
                continue
            if 500 <= exc.code < 600:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    if last:
        raise last
    return "", 0


def _json(url: str, **kw):
    body, _ = _request(url, **kw)
    return json.loads(body)


def _clean(s: str) -> str:
    return " ".join(_TAG.sub("", s or "").split())


# --------------------------------------------------------------------------
# T1 — own-index APIs
# --------------------------------------------------------------------------

def wikipedia(query: str, k: int = 4) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    d = _json(f"{WIKI_API}?action=query&list=search&format=json&srlimit={max(k,1)}"
              f"&srsearch={urllib.parse.quote(q)}")
    hits = (d.get("query") or {}).get("search") or []
    titles = [h["title"] for h in hits if h.get("title")][:k]
    if not titles:
        return []
    t = "|".join(urllib.parse.quote(x) for x in titles)
    out = []
    try:
        ed = _json(f"{WIKI_API}?action=query&prop=extracts&exintro=1&explaintext=1"
                   f"&format=json&redirects=1&titles={t}")
        for _, p in ((ed.get("query") or {}).get("pages") or {}).items():
            ex = _clean(p.get("extract") or "")
            if ex:
                out.append({"title": p.get("title", ""),
                            "url": "https://en.wikipedia.org/wiki/" +
                                   urllib.parse.quote((p.get("title") or "").replace(" ", "_")),
                            "content": ex[:1500], "provider": "wikipedia"})
    except Exception:
        pass
    if not out:
        for h in hits[:k]:
            out.append({"title": h.get("title", ""),
                        "url": "https://en.wikipedia.org/wiki/" +
                               urllib.parse.quote((h.get("title") or "").replace(" ", "_")),
                        "content": _clean(h.get("snippet") or "")[:600],
                        "provider": "wikipedia:snippet"})
    return out[:k]


def wikidata(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    d = _json("https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json"
              f"&language=en&limit={max(k,1)}&search={urllib.parse.quote(q)}")
    out = []
    for s in (d.get("search") or [])[:k]:
        label, desc = s.get("label") or "", s.get("description") or ""
        if label or desc:
            out.append({"title": label, "url": f"https://www.wikidata.org/wiki/{s.get('id','')}",
                        "content": f"{label} — {desc}".strip(" —"), "provider": "wikidata"})
    return out


def openalex(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    d = _json("https://api.openalex.org/works?per-page=%d&search=%s&mailto=research@local"
              % (max(k, 1), urllib.parse.quote(q)))
    out = []
    for w in (d.get("results") or [])[:k]:
        title = w.get("title") or ""
        inv = w.get("abstract_inverted_index") or {}
        abstract = ""
        if inv:
            pos = {}
            for word, idxs in inv.items():
                for i in idxs:
                    pos[i] = word
            abstract = " ".join(pos[i] for i in sorted(pos))[:800]
        if title or abstract:
            out.append({"title": title, "url": w.get("doi") or w.get("id") or "",
                        "content": f"{title}. {abstract}".strip(), "provider": "openalex"})
    return out


def crossref(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    d = _json("https://api.crossref.org/works?rows=%d&query=%s&mailto=research@local"
              % (max(k, 1), urllib.parse.quote(q)))
    items = ((d.get("message") or {}).get("items") or [])[:k]
    out = []
    for it in items:
        title = (it.get("title") or [""])[0]
        if title:
            out.append({"title": title, "url": it.get("URL") or "",
                        "content": f"{title}. {it.get('container-title', [''])[0] if it.get('container-title') else ''}".strip(),
                        "provider": "crossref"})
    return out


def stackexchange(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    d = _json("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
              f"&q={urllib.parse.quote(q)}&site=stackoverflow&pagesize={max(k,1)}&filter=withbody")
    out = []
    for it in (d.get("items") or [])[:k]:
        title = it.get("title") or ""
        body = _clean(it.get("body") or "")[:700]
        if title:
            out.append({"title": title, "url": it.get("link") or "",
                        "content": f"{title}. {body}".strip(), "provider": "stackexchange"})
    return out


def arxiv(query: str, k: int = 3) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    body, _ = _request("http://export.arxiv.org/api/query?search_query=all:%s&max_results=%d"
                       % (urllib.parse.quote(q), max(k, 1)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    for e in root.findall("a:entry", ns)[:k]:
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
        summ = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link = e.findtext("a:id", default="", namespaces=ns) or ""
        if title:
            out.append({"title": " ".join(title.split()), "url": link,
                        "content": " ".join(f"{title}. {summ}".split())[:800],
                        "provider": "arxiv"})
    return out


# --------------------------------------------------------------------------
# T2 — metasearch (bonus tier; partial success is success)
# --------------------------------------------------------------------------

def searxng(query: str, k: int = 5) -> list[dict]:
    d = _json(f"{SEARX}?q={urllib.parse.quote(query)}&format=json")
    out = []
    for item in (d.get("results") or [])[:k]:
        content = _clean(item.get("content") or "")
        if content:
            out.append({"title": _clean(item.get("title") or ""),
                        "url": item.get("url") or "",
                        "content": content[:600], "provider": "searxng"})
    return out


T1 = {"wikipedia": wikipedia, "wikidata": wikidata, "openalex": openalex,
      "crossref": crossref, "stackexchange": stackexchange, "arxiv": arxiv}
T2 = {"searxng": searxng}
PROVIDERS = {**T1, **T2}

DEFAULT_ORDER = ("wikipedia", "wikidata", "openalex", "stackexchange",
                 "crossref", "arxiv", "searxng")


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in rows:
        key = (s.get("url") or s.get("title") or "").split("#")[0].rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def search(query: str, k: int = 5, order: tuple[str, ...] = DEFAULT_ORDER,
           min_results: int = 3) -> list[dict]:
    """Walk the ladder. Returns [] only if EVERY provider failed.

    A caller must treat [] as "retrieval unavailable", never as "no evidence
    exists" — conflating those made a dead search stack look like an honest
    abstention and produced a coverage-0.00 measurement that was really a bug.
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
            collected = _dedupe(collected + got)
            if len(collected) >= min_results:
                break
    return collected[: max(k, 1) * 2]


# --------------------------------------------------------------------------
# T3 — fetch a page and extract main content
# --------------------------------------------------------------------------

def _crawl4ai_token() -> str:
    try:
        with open(CRAWL4AI_KEY_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def _trafilatura(html: str, url: str) -> str:
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_comments=False, include_tables=True,
                                  include_links=False, favor_precision=False,
                                  url=url)
        return (txt or "").strip()
    except Exception:
        return ""


def _bs4_strip(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for t in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript"]):
            t.decompose()
        return str(soup)
    except Exception:
        return html


def _crawl4ai(url: str, mode: str = "fit") -> str:
    tok = _crawl4ai_token()
    if not tok:
        return ""
    body, _ = _request(f"{CRAWL4AI}/md",
                       data=json.dumps({"url": url, "f": mode}).encode(),
                       headers={"Authorization": f"Bearer {tok}",
                                "Content-Type": "application/json"},
                       timeout=90, cache=False)
    try:
        return (json.loads(body).get("markdown") or "").strip()
    except Exception:
        return ""


def fetch_page(url: str, min_chars: int = 400) -> dict:
    """URL -> clean main-content text, escalating through documented tiers.

    tier 1  plain GET + trafilatura   (fast, deterministic, no browser)
    tier 2  crawl4ai /md f=fit        (JS pages, mild bot walls)
    tier 3  crawl4ai /md f=raw        (last resort; noisy but never empty)

    The tier is returned so evidence rows can record how the text was obtained.
    """
    out = {"url": url, "text": "", "tier": None, "chars": 0}
    if not url or not url.startswith(("http://", "https://")):
        return out

    try:
        body, status = _request(url, timeout=20)
        if status == 200 and body:
            html = _bs4_strip(body)
            txt = _trafilatura(html, url)
            if len(txt) >= min_chars:
                return {**out, "text": txt[:6000], "tier": "trafilatura", "chars": len(txt)}
    except Exception:
        pass

    txt = _crawl4ai(url, "fit")
    if len(txt) >= min_chars:
        return {**out, "text": txt[:6000], "tier": "crawl4ai-fit", "chars": len(txt)}

    txt = _crawl4ai(url, "raw")
    if txt:
        return {**out, "text": txt[:6000], "tier": "crawl4ai-raw", "chars": len(txt)}

    return out


# --------------------------------------------------------------------------
# query building with spaCy (better than regex proper-noun runs)
# --------------------------------------------------------------------------

_NLP = None


def spacy_entities(text: str) -> list[str]:
    """Named entities via spaCy, falling back to capitalised runs.

    Measured: regex capitalised-run extraction produced "Reagan and Give Us Our
    Skeletons" as one blob. spaCy separates PERSON/ORG/WORK_OF_ART properly.
    """
    global _NLP
    if not text:
        return []
    try:
        if _NLP is None:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
        doc = _NLP(text[:2000])
        ents = [e.text.strip() for e in doc.ents
                if e.label_ in ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART",
                                "EVENT", "PRODUCT", "FAC", "NORP", "LAW")]
        if ents:
            seen, out = set(), []
            for e in ents:
                if len(e) > 2 and e.lower() not in seen:
                    seen.add(e.lower())
                    out.append(e)
            return out
    except Exception:
        pass
    # fallback: capitalised runs
    return [m for m in re.findall(r"\b[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*", text)
            if len(m) > 2][:6]


# --------------------------------------------------------------------------
# diagnostics + self-test
# --------------------------------------------------------------------------

def diagnose(queries: list[str] | None = None) -> dict:
    queries = queries or ["Rayleigh scattering", "capital of France"]
    out = {}
    for name, fn in PROVIDERS.items():
        per = []
        for q in queries:
            t0 = time.time()
            try:
                n = len(fn(q, k=3))
                per.append({"q": q, "results": n, "ms": int((time.time() - t0) * 1000)})
            except Exception as exc:
                per.append({"q": q, "error": f"{type(exc).__name__}: {str(exc)[:60]}"})
        out[name] = per
    return out


def run_self_test() -> bool:
    print("retrieval.py v2 - SELF TEST")
    ok = True

    good = _clean("<b>Blue</b>  sky\n is") == "Blue sky is"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] _clean")

    good = wikipedia("") == [] and wikidata("") == [] and openalex("") == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] empty query makes no request")

    class Boom:
        def __call__(self, q, k=3):
            raise RuntimeError("down")

    saved = dict(PROVIDERS)
    try:
        PROVIDERS["dead"] = Boom()
        got = search("Rayleigh scattering", k=2, order=("dead", "wikipedia"))
        good = len(got) > 0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] dead provider skipped ({len(got)} results)")
    finally:
        PROVIDERS.clear()
        PROVIDERS.update(saved)

    # dedupe
    rows = [{"url": "http://a/x/"}, {"url": "http://a/x"}, {"url": "http://b/y"}]
    good = len(_dedupe(rows)) == 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] url dedupe")

    # spaCy entity split -- the case regex got wrong
    ents = spacy_entities("Reagan and Give Us Our Skeletons were both documentaries.")
    good = any("Reagan" in e for e in ents) and len(ents) >= 2
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] spaCy entities -> {ents[:4]}")

    # live: fetch_page must produce text via a recorded tier
    try:
        r = fetch_page("https://en.wikipedia.org/wiki/Web_crawler")
        good = r["chars"] > 400 and r["tier"] is not None
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] fetch_page -> tier={r['tier']} "
              f"chars={r['chars']}")
    except Exception as exc:
        print(f"  [SKIP] fetch_page live ({type(exc).__name__})")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v2 retrieval ladder.")
    ap.add_argument("--query")
    ap.add_argument("--fetch")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1
    if args.diagnose:
        print(json.dumps(diagnose(), indent=1))
        return 0
    if args.fetch:
        print(json.dumps(fetch_page(args.fetch), indent=1)[:2500])
        return 0
    if args.query:
        print(json.dumps(search(args.query, args.k), indent=1, ensure_ascii=True)[:3000])
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
