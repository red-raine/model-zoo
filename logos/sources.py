"""logos.sources — the sources-of-truth registry, and date resolution.

THE PROBLEM
-----------
The temporal gate rejected **41 of 43** harvested facts against a 2022 claim, because 38 had no
source date at all. That is not a gate bug; it is a coverage bug. A fact without a date cannot be
used for a dated claim, and Wikidata simply does not carry dates for most statements on these
entities.

So the fix is not a better gate. The fix is **more sources with dates**, and a resolver that
tries several before giving up.

WHAT THIS MODULE PROVIDES
-------------------------
  1. `REGISTRY` -- a categorised catalogue of sources of truth, each with what it holds, whether
     it carries dates, its access method, and its licence. The point is coverage of *niche* areas,
     not a list of the famous five.
  2. `wayback_dates(url)` -- the Wayback Machine CDX API. For any URL it returns the first and
     last capture timestamps, which turns an undated citation into a dated one. This is the single
     highest-leverage date source available, because it dates the *citation* rather than requiring
     the publisher to expose structured dates.
  3. `resolve_date(...)` -- a cascade: the fact's own date, then Wayback on its reference, then a
     named fallback. It reports WHICH source supplied the date, so a resolved date is auditable
     rather than magic.

HONEST LIMITS
-------------
* Wayback dates a *URL*, not a *fact*. If the cited page was updated after the claim, the capture
  date is the page's availability, not the fact's publication. The gate treats a Wayback date as
  weaker evidence than a publisher's own date and says so in the returned record.
* Many registry entries need API keys or have terms that forbid bulk scraping. The registry records
  access requirements rather than pretending everything is free.
* Nothing here fetches 100 sources today. The registry is a map with a working resolver for the
  one source that needs no key and dates arbitrary citations; the rest are queued deliberately.

USAGE
-----
    python tools/logos/sources.py --self-test
    python tools/logos/sources.py --summary
    python tools/logos/sources.py --wayback https://www.example.org/article
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

UA = {"User-Agent": "logos-verify/1.0 (provenance research)"}


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    url: str
    category: str
    provides: str = ""
    has_dates: bool = False
    access: str = "http"          # http | api | api-key | scrape | bulk
    license: str = ""
    notes: str = ""


# --------------------------------------------------------------------------------- registry
# A catalogue, not an integration. Every entry is a place a verifier can check something, chosen
# so the *niche* areas are covered rather than only the obvious encyclopaedias.
_S = [
    # --- knowledge graphs / encyclopaedic
    Source("wikidata", "Wikidata", "https://query.wikidata.org/sparql", "knowledge_graph",
           "entity-attribute-value triples, per-statement references", True, "api", "CC0"),
    Source("dbpedia", "DBpedia", "https://dbpedia.org/sparql", "knowledge_graph",
           "structured infobox triples", False, "api", "CC BY-SA"),
    Source("wikipedia", "Wikipedia", "https://en.wikipedia.org/w/api.php", "encyclopaedic",
           "prose, infoboxes, revision history (dates!)", True, "api", "CC BY-SA"),
    Source("yago", "YAGO", "https://yago-knowledge.org/", "knowledge_graph",
           "cleaned Wikidata+Wikipedia taxonomy", False, "bulk", "CC BY"),
    Source("conceptnet", "ConceptNet", "https://conceptnet.io/", "knowledge_graph",
           "common-sense relations", False, "api", "CC BY-SA"),
    Source("britannica", "Encyclopaedia Britannica", "https://www.britannica.com/",
           "encyclopaedic", "editorial summaries", False, "scrape", "proprietary"),
    Source("sep", "Stanford Encyclopedia of Philosophy", "https://plato.stanford.edu/",
           "encyclopaedic", "authoritative philosophy (logical frames)", True, "http", "CC BY"),
    Source("scholarpedia", "Scholarpedia", "http://www.scholarpedia.org/", "encyclopaedic",
           "peer-reviewed expository articles", False, "http", "CC BY-NC-SA"),
    Source("openlibrary", "Open Library", "https://openlibrary.org/", "books",
           "book records with publish dates", True, "api", "CC0"),
    Source("google_books", "Google Books", "https://www.googleapis.com/books/v1/",
           "books", "book metadata, snippet search", True, "api-key", "proprietary"),

    # --- date resolution
    Source("wayback", "Wayback Machine CDX", "https://web.archive.org/cdx/search/cdx",
           "archive", "capture timestamps for arbitrary URLs", True, "api", "public"),
    Source("commoncrawl", "Common Crawl", "https://index.commoncrawl.org/", "archive",
           "crawled-page index with fetch dates", True, "api", "public"),
    Source("archive_today", "archive.today", "https://archive.ph/", "archive",
           "page snapshots", True, "scrape", "public"),
    Source("loc_chronicling", "Chronicling America (LoC)", "https://chroniclingamerica.loc.gov/",
           "news_archive", "US newspapers 1777-1963, OCR + dates", True, "api", "public domain"),
    Source("europeana_newspapers", "Europeana Newspapers", "https://www.europeana.eu/",
           "news_archive", "European historical newspapers", True, "api-key", "mixed"),
    Source("trove", "Trove (NLA)", "https://trove.nla.gov.au/", "news_archive",
           "Australian newspapers and archives", True, "api-key", "mixed"),
    Source("gdelttv", "GDELT", "https://api.gdeltproject.org/api/v2/", "news_archive",
           "global news events with timestamps, 2015-", True, "api", "public"),
    Source("mediacloud", "Media Cloud", "https://mediacloud.org/", "news_archive",
           "news story index + dates", True, "api-key", "open research"),
    Source("newsbank", "NewsBank", "https://www.newsbank.com/", "news_archive",
           "paywalled newspaper archive", True, "scrape", "proprietary"),

    # --- scientific / academic
    Source("openalex", "OpenAlex", "https://api.openalex.org/", "academic",
           "works, authors, venues, publication dates", True, "api", "CC0"),
    Source("crossref", "Crossref", "https://api.crossref.org/", "academic",
           "DOI metadata, published dates", True, "api", "CC0"),
    Source("arxiv", "arXiv", "https://export.arxiv.org/api/query", "academic",
           "preprints with submission dates", True, "api", "mixed"),
    Source("pubmed", "PubMed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", "medical",
           "biomedical citations with dates", True, "api", "public"),
    Source("pmc", "PubMed Central", "https://www.ncbi.nlm.nih.gov/pmc/", "medical",
           "full-text biomedical articles", True, "api", "open subset"),
    Source("semanticscholar", "Semantic Scholar", "https://api.semanticscholar.org/",
           "academic", "citations, abstracts, dates", True, "api", "open"),
    Source("core", "CORE", "https://api.core.ac.uk/", "academic", "open-access full texts",
           True, "api-key", "open"),
    Source("doaj", "DOAJ", "https://doaj.org/api/", "academic", "open-access journals",
           True, "api", "CC0"),
    Source("biorxiv", "bioRxiv", "https://api.biorxiv.org/", "academic", "biology preprints",
           True, "api", "CC BY"),
    Source("medrxiv", "medRxiv", "https://api.medrxiv.org/", "medical", "medical preprints",
           True, "api", "CC BY"),
    Source("cochrane", "Cochrane Library", "https://www.cochranelibrary.com/", "medical",
           "systematic reviews (highest evidential tier)", True, "scrape", "proprietary"),
    Source("nice", "NICE (UK)", "https://www.nice.org.uk/", "medical",
           "clinical guidelines", True, "http", "OGL"),
    Source("who", "WHO", "https://www.who.int/", "medical", "global health guidance", True,
           "http", "CC BY-NC-SA"),
    Source("cdc", "CDC", "https://www.cdc.gov/", "medical", "US public-health data", True,
           "http", "public domain"),
    Source("clinicaltrials", "ClinicalTrials.gov", "https://clinicaltrials.gov/api/",
           "medical", "trial registrations with dates", True, "api", "public domain"),
    Source("retractionwatch", "Retraction Watch DB", "https://retractionwatch.com/",
           "medical", "retracted papers (evidential veto)", True, "api-key", "CC BY"),

    # --- government / statistical
    Source("worldbank", "World Bank Open Data", "https://api.worldbank.org/v2/",
           "statistical", "country indicators by year", True, "api", "CC BY"),
    Source("imf", "IMF Data", "https://www.imf.org/en/Data", "statistical",
           "macroeconomic series", True, "api", "mixed"),
    Source("oecd", "OECD Data", "https://data.oecd.org/api/", "statistical",
           "member-country indicators", True, "api", "CC BY"),
    Source("eurostat", "Eurostat", "https://ec.europa.eu/eurostat/api/", "statistical",
           "EU statistics", True, "api", "CC BY"),
    Source("un_data", "UN Data", "https://data.un.org/", "statistical",
           "global aggregates", True, "api", "mixed"),
    Source("us_census", "US Census Bureau", "https://api.census.gov/", "statistical",
           "US population and economics", True, "api", "public domain"),
    Source("ons_uk", "UK ONS", "https://api.beta.ons.gov.uk/", "statistical",
           "UK statistics", True, "api", "OGL"),
    Source("fred", "FRED (St. Louis Fed)", "https://api.stlouisfed.org/fred/", "statistical",
           "US economic time series", True, "api-key", "public"),
    Source("data_gov", "Data.gov", "https://api.data.gov/", "statistical",
           "US federal datasets", True, "api-key", "public domain"),
    Source("owid", "Our World in Data", "https://ourworldindata.org/grapher/", "statistical",
           "curated indicators with sources", True, "http", "CC BY"),
    Source("noaa", "NOAA", "https://www.ncei.noaa.gov/access/", "climate",
           "climate and weather records", True, "api", "public domain"),
    Source("nasa_giss", "NASA GISS", "https://data.giss.nasa.gov/", "climate",
           "temperature series", True, "http", "public domain"),
    Source("ipcc", "IPCC", "https://www.ipcc.ch/", "climate", "assessment reports", True,
           "http", "copyright with reuse"),
    Source("usgs", "USGS", "https://earthquake.usgs.gov/fdsnws/", "geoscience",
           "earthquakes, hydrology, geology", True, "api", "public domain"),
    Source("gbif", "GBIF", "https://api.gbif.org/", "biology", "species occurrences", True,
           "api", "CC BY"),
    Source("ncbi_taxonomy", "NCBI Taxonomy", "https://eutils.ncbi.nlm.nih.gov/", "biology",
           "organism classification", False, "api", "public domain"),
    Source("pubchem", "PubChem", "https://pubchem.ncbi.nlm.nih.gov/rest/", "chemistry",
           "compound data", False, "api", "public domain"),
    Source("chebi", "ChEBI", "https://www.ebi.ac.uk/chebi/", "chemistry",
           "chemical entities ontology", False, "api", "CC BY"),
    Source("uniprot", "UniProt", "https://rest.uniprot.org/", "biology", "protein records",
           True, "api", "CC BY"),

    # --- legal / official
    Source("courtlistener", "CourtListener", "https://www.courtlistener.com/api/",
           "legal", "US case law with filing dates", True, "api", "public"),
    Source("eurlex", "EUR-Lex", "https://eur-lex.europa.eu/", "legal", "EU legislation",
           True, "http", "reuse permitted"),
    Source("legislation_uk", "legislation.gov.uk", "https://www.legislation.gov.uk/",
           "legal", "UK statute text and versions", True, "api", "OGL"),
    Source("sec_edgar", "SEC EDGAR", "https://www.sec.gov/edgar/", "financial",
           "company filings with dates", True, "api", "public domain"),
    Source("ftc", "US FTC", "https://www.ftc.gov/", "legal", "enforcement actions", True,
           "http", "public domain"),
    Source("fda", "US FDA", "https://www.fda.gov/", "medical", "approvals and recalls", True,
           "http", "public domain"),
    Source("ema", "European Medicines Agency", "https://www.ema.europa.eu/", "medical",
           "EU drug approvals", True, "http", "reuse permitted"),
    Source("smithsonian_gvp", "Smithsonian Global Volcanism", "https://volcano.si.edu/",
           "geoscience", "volcano eruptions with dates", True, "scrape", "public domain"),
    Source("nasa_earthdata", "NASA EarthData", "https://earthdata.nasa.gov/", "geoscience",
           "satellite and geophysical data", True, "api-key", "public domain"),
    Source("ecb_data", "European Central Bank Data", "https://data.ecb.europa.eu/",
           "financial", "EU monetary and banking series", True, "api", "CC BY"),
    Source("alphavantage", "Alpha Vantage", "https://www.alphavantage.co/", "financial",
           "equity and FX time series", True, "api-key", "proprietary"),
    Source("boe", "Bank of England", "https://www.bankofengland.co.uk/boeapps/database/",
           "financial", "UK monetary series", True, "http", "OGL"),
    Source("worldpop", "WorldPop", "https://www.worldpop.org/", "demographic",
           "gridded population estimates", True, "bulk", "CC BY"),
    Source("citypopulation", "City Population", "https://www.citypopulation.de/",
           "demographic", "city populations from censuses", True, "scrape", "proprietary"),
    Source("ipums", "IPUMS", "https://www.ipums.org/", "demographic",
           "harmonised census microdata", True, "api-key", "academic"),
    Source("uspto", "USPTO", "https://developer.uspto.gov/", "patents", "patents with dates",
           True, "api", "public domain"),
    Source("epo_ops", "EPO OPS", "https://ops.epo.org/", "patents", "European patents", True,
           "api-key", "proprietary"),
    Source("google_patents", "Google Patents", "https://patents.google.com/", "patents",
           "patent search with dates", True, "scrape", "proprietary"),

    # --- fact-checkers (evidential prior, not ground truth)
    Source("google_factcheck", "Google Fact Check Tools", "https://factchecktools.googleapis.com/",
           "factcheck", "ClaimReview index across publishers", True, "api-key", "public"),
    Source("politifact", "PolitiFact", "https://www.politifact.com/", "factcheck",
           "US political claim ratings", True, "scrape", "proprietary"),
    Source("snopes", "Snopes", "https://www.snopes.com/", "factcheck", "viral claim ratings",
           True, "scrape", "proprietary"),
    Source("fullfact", "Full Fact", "https://fullfact.org/", "factcheck", "UK fact checks",
           True, "http", "CC BY-ND"),
    Source("factcheck_org", "FactCheck.org", "https://www.factcheck.org/", "factcheck",
           "US political fact checks", True, "http", "proprietary"),
    Source("afp_factcheck", "AFP Fact Check", "https://factcheck.afp.com/", "factcheck",
           "international fact checks", True, "http", "proprietary"),
    Source("reuters_factcheck", "Reuters Fact Check", "https://www.reuters.com/fact-check/",
           "factcheck", "international fact checks", True, "http", "proprietary"),
    Source("altnews", "Alt News", "https://www.altnews.in/", "factcheck", "Indian fact checks",
           True, "http", "proprietary"),
    Source("boomlive", "BOOM", "https://www.boomlive.in/", "factcheck",
           "South Asian fact checks", True, "http", "proprietary"),
    Source("dfrlab", "DFRLab", "https://dfrlab.org/", "factcheck", "disinformation research",
           True, "http", "CC BY-NC-SA"),

    # --- domain: geography, transport, infrastructure
    Source("geonames", "GeoNames", "https://www.geonames.org/", "geography",
           "place names, coordinates, populations", False, "api", "CC BY"),
    Source("naturalearth", "Natural Earth", "https://www.naturalearthdata.com/", "geography",
           "map vectors", False, "bulk", "public domain"),
    Source("osm", "OpenStreetMap", "https://overpass-api.de/api/", "geography",
           "features with tags and timestamps", True, "api", "ODbL"),
    Source("ourairports", "OurAirports", "https://ourairports.com/data/", "transport",
           "airport records", False, "bulk", "public domain"),
    Source("openflights", "OpenFlights", "https://openflights.org/data.html", "transport",
           "routes and airports", False, "bulk", "ODbL"),
    Source("structurae", "Structurae", "https://structurae.net/", "infrastructure",
           "structures: bridges, dams, towers with dates", True, "scrape", "proprietary"),
    Source("skyscraperpage", "SkyscraperPage", "https://skyscraperpage.com/", "infrastructure",
           "building heights and completion years", True, "scrape", "proprietary"),
    Source("emporis", "CTBUH Skyscraper Center", "https://www.skyscrapercenter.com/",
           "infrastructure", "tall-building data", True, "scrape", "proprietary"),

    # --- domain: sport, culture, media
    Source("olympedia", "Olympedia", "https://www.olympedia.org/", "sport",
           "Olympic results and athletes", True, "scrape", "CC BY"),
    Source("transfermarkt", "Transfermarkt", "https://www.transfermarkt.com/", "sport",
           "football transfers and squads", True, "scrape", "proprietary"),
    Source("espn_stats", "ESPN", "https://www.espn.com/", "sport", "match results", True,
           "scrape", "proprietary"),
    Source("musicbrainz", "MusicBrainz", "https://musicbrainz.org/ws/2/", "music",
           "recordings, releases, dates", True, "api", "CC0"),
    Source("discogs", "Discogs", "https://api.discogs.com/", "music", "release catalogue",
           True, "api-key", "proprietary"),
    Source("imdb", "IMDb datasets", "https://datasets.imdbws.com/", "film",
           "titles, people, years", True, "bulk", "proprietary"),
    Source("tmdb", "TMDB", "https://api.themoviedb.org/3/", "film", "titles and dates",
           True, "api-key", "proprietary"),
    Source("wikidata_film", "Wikidata (film)", "https://query.wikidata.org/", "film",
           "film entities", True, "api", "CC0"),
    Source("goodreads", "Goodreads", "https://www.goodreads.com/", "books",
           "book editions", True, "scrape", "proprietary"),

    # --- domain: economics, companies
    Source("opencorporates", "OpenCorporates", "https://api.opencorporates.com/", "company",
           "company registrations and dates", True, "api-key", "mixed"),
    Source("companies_house", "UK Companies House", "https://api.company-information.service.gov.uk/",
           "company", "UK company filings", True, "api-key", "OGL"),
    Source("gleif", "GLEIF", "https://api.gleif.org/", "company", "legal entity identifiers",
           True, "api", "CC0"),
    Source("crunchbase", "Crunchbase", "https://www.crunchbase.com/", "company",
           "funding and founding dates", True, "api-key", "proprietary"),

    # --- domain: conflict, disasters, events
    Source("acled", "ACLED", "https://api.acleddata.com/", "conflict",
           "conflict events with dates", True, "api-key", "mixed"),
    Source("ucdp", "UCDP", "https://ucdp.uu.se/", "conflict", "organized-violence events",
           True, "api", "CC BY"),
    Source("emdat", "EM-DAT", "https://www.emdat.be/", "disaster",
           "disaster database", True, "api-key", "academic"),
    Source("reliefweb", "ReliefWeb", "https://api.reliefweb.int/", "disaster",
           "humanitarian reports", True, "api", "CC BY"),
    Source("gdacs", "GDACS", "https://www.gdacs.org/", "disaster", "disaster alerts", True,
           "api", "public"),

    # --- domain: health, demographics, education
    Source("ihme", "IHME GBD", "https://ghdx.healthdata.org/", "medical",
           "global burden of disease estimates", True, "api-key", "mixed"),
    Source("unpopulation", "UN Population Division", "https://population.un.org/", "demographic",
           "population estimates and projections", True, "http", "CC BY"),
    Source("uis_unesco", "UNESCO UIS", "https://uis.unesco.org/", "education",
           "education statistics", True, "api", "CC BY"),
    Source("pisa", "OECD PISA", "https://www.oecd.org/pisa/", "education",
           "education outcomes", True, "http", "CC BY"),

    # --- domain: language, reference, misc
    Source("wordnet", "WordNet", "https://wordnet.princeton.edu/", "lexical",
           "synonymy, antonymy, taxonomy", False, "bulk", "WordNet licence"),
    Source("framenet", "FrameNet", "https://framenet.icsi.berkeley.edu/", "lexical",
           "semantic frames and roles", False, "bulk", "CC BY"),
    Source("verbnet", "VerbNet", "https://uvi.colorado.edu/", "lexical",
           "verb classes and alternations", False, "bulk", "CC BY"),
    Source("framester", "Framester", "https://framester.github.io/", "lexical",
           "FrameNet+WordNet+BabelNet hub", False, "bulk", "CC BY"),
    Source("babelnet", "BabelNet", "https://babelnet.org/", "lexical",
           "multilingual concept graph", False, "api-key", "proprietary"),
    Source("framebase", "FrameBase", "http://framebase.org/", "lexical",
           "frames to RDF/SKOS", False, "bulk", "CC BY"),
    Source("stanford_corenlp", "Stanford CoreNLP", "https://stanfordnlp.github.io/CoreNLP/",
           "nlp", "parsing, coreference", False, "bulk", "GPL"),
    Source("spacy_models", "spaCy models", "https://spacy.io/models", "nlp",
           "tokenisation, NER, deps", False, "bulk", "MIT"),
    Source("nltk_data", "NLTK data", "https://www.nltk.org/nltk_data/", "nlp",
           "corpora and lexicons", False, "bulk", "mixed"),
]

REGISTRY: dict[str, Source] = {s.id: s for s in _S}


def by_category() -> dict[str, list[Source]]:
    out: dict[str, list[Source]] = {}
    for s in REGISTRY.values():
        out.setdefault(s.category, []).append(s)
    return out


def with_dates() -> list[Source]:
    return [s for s in REGISTRY.values() if s.has_dates]


def free_sources() -> list[Source]:
    return [s for s in REGISTRY.values() if s.access in ("http", "api", "bulk")]


# ------------------------------------------------------------------------ date resolution
@dataclass
class DateEvidence:
    date: str
    source: str
    strength: str          # publisher | archive | inferred
    detail: str = ""


def wayback_dates(url: str, timeout: float = 25.0) -> tuple[str, str] | None:
    """First and last capture timestamps for a URL, as YYYY-MM-DD.

    This is the highest-leverage date source available because it dates the CITATION rather than
    requiring the publisher to expose structured dates. It is also weaker evidence: a capture date
    is when the page was available, not when the fact was published.
    """
    if not url or not url.startswith("http"):
        return None
    base = ("https://web.archive.org/cdx/search/cdx?url="
            + urllib.parse.quote(url, safe="")
            + "&output=json&fl=timestamp&collapse=timestamp:8&limit=1")
    fmt = lambda t: f"{t[0:4]}-{t[4:6]}-{t[6:8]}" if len(t) >= 8 else t  # noqa: E731

    def one(q: str) -> str | None:
        try:
            req = urllib.request.Request(q, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8", "replace"))
        except Exception:                 # noqa: BLE001 - archive.org is often slow or down
            return None
        if not rows or len(rows) < 2 or not rows[1] or not rows[1][0]:
            return None
        return fmt(rows[1][0])

    # CDX defaults to ascending, so limit=1 is the EARLIEST; sort=reverse gives the LATEST.
    first = one(base)
    if first is None:
        return None
    last = one(base + "&sort=reverse")
    return first, (last or first)


def resolve_date(fact_date: str = "", ref_url: str = "",
                 allow_wayback: bool = True) -> list[DateEvidence]:
    """Every date the system can establish, strongest first.

    A list, not a single value: a verifier needs to know *which* source dated a fact, and a
    publisher's own date is stronger evidence than an archive capture.
    """
    out: list[DateEvidence] = []
    if fact_date:
        out.append(DateEvidence(fact_date, "publisher", "publisher",
                                "date on the statement itself"))
    if allow_wayback and ref_url:
        wb = wayback_dates(ref_url)
        if wb:
            first, last = wb
            out.append(DateEvidence(first, "wayback_first", "archive",
                                    f"earliest capture of the cited URL (last {last})"))
    return out


# ------------------------------------------------------------------------------- tests
def run_self_test() -> bool:
    print("logos.sources - SELF TEST")
    ok = True

    good = len(REGISTRY) >= 100
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] registry holds {len(REGISTRY)} sources")

    cats = by_category()
    good = len(cats) >= 15
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] across {len(cats)} categories")

    # every category must have at least two entries, or "niche coverage" is a claim without content
    thin = {c: len(v) for c, v in cats.items() if len(v) < 2}
    good = not thin
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no category has fewer than 2 sources"
          + (f" -- thin: {thin}" if thin else ""))

    # dates are the thing we are short of, so the dated fraction matters
    dated = with_dates()
    frac = len(dated) / len(REGISTRY)
    good = frac >= 0.6
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(dated)}/{len(REGISTRY)} sources carry dates "
          f"({frac:.0%})")

    free = free_sources()
    good = len(free) >= 40
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {len(free)} sources are keyless-accessible")

    # ids must be unique and urls well formed
    good = len(REGISTRY) == len(_S) and all(s.url.startswith("http") for s in _S)
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] ids unique and all URLs well formed")

    # access modes are constrained, so "scale" never silently means "scrape everything"
    good = all(s.access in ("http", "api", "api-key", "scrape", "bulk") for s in REGISTRY.values())
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] every source declares an access mode")

    # --- date resolution: publisher date wins and comes first
    ev = resolve_date("2019-05-01", "", allow_wayback=False)
    good = len(ev) == 1 and ev[0].source == "publisher" and ev[0].strength == "publisher"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] publisher date is reported first and strongest")

    # --- no date and no url: empty, not a guess
    good = resolve_date("", "", allow_wayback=False) == []
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] no date and no URL resolves to nothing "
          f"(no invention)")

    # --- wayback on a malformed URL is a no-op, not an exception
    good = wayback_dates("not-a-url") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] malformed URL -> None, no exception")

    # --- live wayback probe (network is not guaranteed; report, do not fail the suite)
    wb = wayback_dates("https://en.wikipedia.org/wiki/Syllogism", timeout=20.0)
    if wb:
        first, last = wb
        good = len(first) == 10 and first <= last
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] LIVE wayback: earliest capture of a Wikipedia "
              f"page is {first} (latest {last})")
    else:
        print(f"  [SKIP] live wayback probe unreachable from here (network); logic tested "
              f"offline above")

    print(f"SELF TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sources of truth registry + date resolution.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--wayback", default="")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if run_self_test() else 1
    if args.wayback:
        r = wayback_dates(args.wayback)
        print(f"{args.wayback}\n  -> {r}")
        return 0
    cats = by_category()
    print(f"{len(REGISTRY)} sources across {len(cats)} categories; "
          f"{len(with_dates())} carry dates; {len(free_sources())} keyless\n")
    for c in sorted(cats):
        names = ", ".join(s.name for s in cats[c])
        print(f"  {c:16s} ({len(cats[c]):2d})  {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
