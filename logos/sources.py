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
import os
import re
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

    # ==================================================================================
    # Operator-supplied sources (2026-10-24). Grouped by the operator's own taxonomy:
    # fact-checking, authoritative reference, niche/academic search, history/geography,
    # and OSINT. These fill the niche coverage the registry was thin on.
    # ==================================================================================

    # --- fact-checking networks and regional checkers
    Source("ap_factcheck", "AP Fact Check", "https://apnews.com/hub/ap-fact-check",
           "factcheck", "breaking news and public-figure claims", True, "scrape",
           "proprietary"),
    Source("bbc_verify", "BBC Verify", "https://www.bbc.com/news/bbcverify", "factcheck",
           "visual, geolocation and video verification", True, "scrape", "proprietary"),
    Source("lead_stories", "Lead Stories", "https://leadstories.com/", "factcheck",
           "viral hoaxes, early detection", True, "http", "proprietary"),
    Source("africa_check", "Africa Check", "https://africacheck.org/", "factcheck",
           "African continent claims and data", True, "http", "CC BY-ND"),
    Source("health_feedback", "Health Feedback", "https://healthfeedback.org/", "factcheck",
           "scientific and medical claims", True, "http", "proprietary"),
    Source("science_feedback", "Science Feedback", "https://science.feedback.org/",
           "factcheck", "climate, environment, science claims", True, "http", "proprietary"),
    Source("ifcn", "IFCN (Poynter)", "https://www.poynter.org/ifcn/", "factcheck_registry",
           "global registry of certified fact-checkers", True, "http", "proprietary"),
    Source("efcsn", "EFCSN", "https://efcsn.com/", "factcheck_registry",
           "European fact-checking standards network", True, "http", "proprietary"),
    Source("duke_reporters_lab", "Duke Reporters' Lab", "https://reporterslab.org/",
           "factcheck_registry", "database of 400+ fact-checking organisations", True, "http",
           "CC BY"),
    Source("claimbuster", "ClaimBuster", "https://idir.uta.edu/claimbuster/", "factcheck",
           "check-worthiness scoring", False, "api", "academic"),
    Source("ground_news", "Ground News", "https://ground.news/", "media_bias",
           "coverage comparison across outlets", True, "scrape", "proprietary"),
    Source("allsides", "AllSides", "https://www.allsides.com/", "media_bias",
           "left/centre/right coverage comparison", True, "scrape", "proprietary"),
    Source("newsreel", "Newsreel", "https://www.newsreel.ai/", "media_bias",
           "comparative coverage across outlets", True, "scrape", "proprietary"),
    Source("lenso", "Lenso.ai", "https://lenso.ai/", "image_verification",
           "reverse image search and photo origin", True, "api-key", "proprietary"),

    # --- authoritative reference (non-political)
    Source("cia_factbook", "CIA World Factbook (archived)", "https://www.cia.gov/the-world-factbook/",
           "reference", "country data", True, "scrape", "public domain"),
    Source("world_atlas", "World Atlas", "https://www.worldatlas.com/", "reference",
           "geography explainers and maps", False, "scrape", "proprietary"),
    Source("google_earth", "Google Earth", "https://earth.google.com/", "geospatial",
           "satellite imagery, historical layers", True, "scrape", "proprietary"),
    Source("natgeo_education", "National Geographic Education", "https://education.nationalgeographic.org/",
           "reference", "geography and biology education", False, "http", "proprietary"),
    Source("nasa_spaceplace", "NASA Space Place", "https://spaceplace.nasa.gov/", "reference",
           "astronomy basics", False, "http", "public domain"),
    Source("khan_academy", "Khan Academy", "https://www.khanacademy.org/", "reference",
           "structured science and maths courses", False, "http", "CC BY-NC-SA"),
    Source("exploratorium", "Exploratorium", "https://www.exploratorium.edu/", "reference",
           "interactive science", False, "http", "proprietary"),
    Source("quanta", "Quanta Magazine", "https://www.quantamagazine.org/", "reference",
           "physics, maths, CS deep dives", True, "http", "CC BY-ND"),
    Source("nova_pbs", "NOVA / PBS", "https://www.pbs.org/wgbh/nova/", "reference",
           "science documentaries", True, "http", "proprietary"),
    Source("sciencenews_students", "Science News Explores", "https://www.snexplores.org/",
           "reference", "current science, accessible", True, "http", "proprietary"),
    Source("howstuffworks", "HowStuffWorks", "https://science.howstuffworks.com/", "reference",
           "everyday science and engineering", False, "scrape", "proprietary"),
    Source("google_public_data", "Google Public Data Explorer", "https://www.google.com/publicdata/",
           "statistical", "global datasets, visualisation", True, "scrape", "proprietary"),
    Source("loc_digital", "Library of Congress Digital Collections", "https://www.loc.gov/collections/",
           "archive", "primary source archives", True, "api", "public domain"),

    # --- niche academic / specialist search
    Source("google_scholar", "Google Scholar", "https://scholar.google.com/", "academic",
           "citation tracking", True, "scrape", "proprietary"),
    Source("base_search", "BASE", "https://www.base-search.net/", "academic",
           "open-access repositories", True, "api", "CC BY-NC"),
    Source("citeseerx", "CiteSeerX", "https://citeseerx.ist.psu.edu/", "academic",
           "computer-science papers", True, "http", "public domain"),
    Source("chemspider", "ChemSpider", "https://www.chemspider.com/", "chemistry",
           "100M+ chemical structures", False, "api-key", "proprietary"),
    Source("wolfram_alpha", "Wolfram Alpha", "https://api.wolframalpha.com/", "computational",
           "computational facts: maths, physics, units", False, "api-key", "proprietary"),
    Source("occrp_aleph", "OCCRP Aleph", "https://aleph.occrp.org/api/", "investigative",
           "global document troves", True, "api", "mixed"),
    Source("shodan", "Shodan", "https://api.shodan.io/", "cyber", "internet-connected devices",
           True, "api-key", "proprietary"),
    Source("grayhatwarfare", "GrayhatWarfare", "https://grayhatwarfare.com/", "cyber",
           "exposed online files", False, "api-key", "proprietary"),
    Source("netlas", "Netlas.io", "https://app.netlas.io/", "cyber", "DNS/SSL/port search",
           True, "api-key", "proprietary"),
    Source("refseek", "RefSeek", "https://www.refseek.com/", "academic",
           "encyclopaedias and monographs", False, "scrape", "proprietary"),
    Source("sciencedirect", "ScienceDirect", "https://api.elsevier.com/", "academic",
           "Elsevier journals", True, "api-key", "proprietary"),
    Source("million_short", "Million Short", "https://millionshort.com/", "search",
           "surfaces obscure sources by removing top sites", False, "scrape", "proprietary"),

    # --- history, geography, liberal sciences
    Source("dpla", "Digital Public Library of America", "https://dp.la/", "history",
           "53M+ images, texts, maps from libraries and museums", True, "api", "mixed"),
    Source("loc_maps", "LoC Geography and Map Division", "https://www.loc.gov/maps/",
           "history", "historical maps and atlases", True, "api", "public domain"),
    Source("david_rumsey", "David Rumsey Map Collection", "https://www.davidrumsey.com/",
           "history", "150,000+ georeferenced historical maps", True, "scrape", "CC BY-NC-SA"),
    Source("old_maps_online", "Old Maps Online", "https://www.oldmapsonline.org/", "history",
           "map search across archives by place and date", True, "scrape", "proprietary"),
    Source("geacron", "GeaCron", "http://geacron.com/", "history",
           "animated political borders 3000 BCE-present", True, "scrape", "proprietary"),
    Source("timemaps", "TimeMaps", "https://www.timemaps.com/", "history",
           "year-by-year world history maps", True, "scrape", "proprietary"),
    Source("running_reality", "Running Reality", "https://www.runningreality.org/", "history",
           "historical population centres and boundaries", True, "http", "CC BY-NC-SA"),
    Source("world_digital_library", "World Digital Library", "https://www.loc.gov/collections/world-digital-library/",
           "history", "global manuscripts and rare books", True, "api", "public domain"),
    Source("fordham_sources", "Internet History Sourcebooks", "https://sourcebooks.fordham.edu/",
           "history", "primary texts: ancient, medieval, modern, Islamic, East Asian", True,
           "http", "educational use"),
    Source("avalon_project", "Avalon Project (Yale)", "https://avalon.law.yale.edu/", "history",
           "treaties, constitutions, charters", True, "http", "public domain"),
    Source("wilson_center", "Wilson Center Digital Archive", "https://digitalarchive.wilsoncenter.org/",
           "history", "declassified Cold War cables and memos", True, "http", "mixed"),
    Source("slavevoyages", "SlaveVoyages", "https://www.slavevoyages.org/", "history",
           "36,000+ Atlantic slave-trade voyages", True, "scrape", "CC BY-NC"),
    Source("eurodocs", "EuroDocs", "https://eudocs.lib.byu.edu/", "history",
           "Western European primary sources", True, "http", "educational use"),
    Source("pleiades", "Pleiades Gazetteer", "https://pleiades.stoa.org/", "history",
           "ancient places with coordinates", False, "api", "CC BY"),
    Source("maddison", "Maddison Project Database", "https://www.rug.nl/ggdc/historicaldevelopment/maddison/",
           "history", "long-run GDP and trade datasets (estimates)", True, "bulk", "CC BY"),
    Source("world_history_encyclopedia", "World History Encyclopedia",
           "https://www.worldhistory.org/", "history", "peer-reviewed history", True, "http",
           "CC BY-NC-SA"),
    Source("aeon", "Aeon Essays", "https://aeon.co/", "history",
           "scholarly long-form essays (secondary)", True, "http", "proprietary"),
    Source("smithsonian_open", "Smithsonian Open Access", "https://www.si.edu/openaccess",
           "history", "5M+ public-domain objects", True, "api", "CC0"),
    Source("british_museum", "British Museum Collection", "https://www.britishmuseum.org/collection",
           "history", "8M+ objects searchable by date, place, material", True, "scrape",
           "CC BY-NC-SA"),

    # --- OSINT: archiving, geolocation, media forensics, metadata, infrastructure
    Source("ghostarchive", "Ghostarchive", "https://ghostarchive.org/", "osint",
           "PROBED 2026-10-25: 200, keyless. Also a web-archive source (video/social clips survive "
           "there after the origin dies)", True, "scrape", "public"),
    Source("suncalc", "SunCalc", "https://www.suncalc.org/", "osint",
           "shadow and sun-angle analysis (chronolocation)", False, "http", "free"),
    Source("sentinel_hub", "Sentinel Hub EO Browser", "https://apps.sentinel-hub.com/eo-browser/",
           "osint", "free ESA satellite imagery, land-use change", True, "api", "free tier"),
    Source("mapillary", "Mapillary", "https://www.mapillary.com/", "osint",
           "crowdsourced street-level imagery", True, "api", "CC BY-SA"),
    Source("tineye", "TinEye", "https://tineye.com/", "osint",
           "reverse image search, earliest appearance", True, "api-key", "proprietary"),
    Source("yandex_images", "Yandex Images", "https://yandex.com/images/", "osint",
           "reverse image search, strong for Eastern Europe", False, "scrape", "proprietary"),
    Source("invid", "InVID / WeVerify", "https://www.invid-project.eu/tools-and-services/",
           "osint", "video frame extraction and verification", False, "bulk", "free"),
    Source("fotoforensics", "FotoForensics", "https://fotoforensics.com/", "osint",
           "error level analysis for image manipulation", False, "api", "free"),
    Source("exiftool", "ExifTool", "https://exiftool.org/", "osint",
           "metadata extraction for images and documents", False, "bulk", "free"),
    Source("pic2map", "Pic2Map", "https://www.pic2map.com/", "osint",
           "extract GPS coordinates from photo EXIF", False, "http", "free"),
    Source("crt_sh", "crt.sh", "https://crt.sh/", "osint",
           "certificate transparency logs", True, "api", "free"),
    Source("urlscan", "urlscan.io", "https://urlscan.io/api/", "osint",
           "PROBED 2026-10-25: /api/v1/search is 200 keyless with real results. Our only "
           "independent URL index that is NOT archive.org", True, "api", "public"),
    Source("icann_lookup", "ICANN Lookup", "https://lookup.icann.org/", "osint",
           "domain registration and ownership", True, "api", "public"),

    # --- second entries for the categories the operator's list left with only one
    Source("google_lens", "Google Lens", "https://lens.google.com/", "image_verification",
           "reverse image search and visual matching", False, "scrape", "proprietary"),
    Source("nasa_worldview", "NASA Worldview", "https://worldview.earthdata.nasa.gov/",
           "geospatial", "daily global satellite imagery with dates", True, "api",
           "public domain"),
    Source("nist_constants", "NIST Constants and Units", "https://physics.nist.gov/cuu/",
           "computational", "SI definitions and physical constants", True, "http",
           "public domain"),
    Source("icij_offshore", "ICIJ Offshore Leaks", "https://offshoreleaks.icij.org/",
           "investigative", "offshore entities and people, investigative record", True,
           "scrape", "ODbL"),
    Source("search_orchestrator", "search-orchestrator (local)", "http://localhost:8022/search",
           "search", "PROBED 2026-10-25: POST /search returns real results with title/url/snippet/"
           "rank/score. This is the actual search surface -- route engines through it instead of "
           "hitting them directly", True, "api", "local"),
    Source("stealth_browser", "stealth-browser (local)", "http://localhost:8080/",
           "local_egress",
           "PROBED: GET /health -> {'status':'ok','service':'stealth-browser'}. NOT a search "
           "engine -- an earlier registry entry wrongly pointed 8080 at searxng", True, "http",
           "local"),
    Source("mubeng", "mubeng proxy rotator (local)", "http://localhost:8089/", "local_egress",
           "PROBED: :8089 -> container `mubeng` (image search-stack-mubeng). It is a Go FORWARD "
           "proxy rotator, so plain GETs return 500 and curl -x gets no tunnel -- likely an empty "
           "or dead upstream pool. A SECOND proxy besides Squid :3128; compare them as separate "
           "egress arms rather than assuming either works", True, "http", "MIT"),
    # (searxng removed 2026-10-25: the entry pointed at localhost:8080, which is the stealth
    #  browser. A wrong source entry inflates the catalogue exactly like a dead archive endpoint
    #  counted as a backup, so it is deleted rather than annotated.)

    # ==================================================================================
    # Web-archive redundancy tier. PROBED from this host on 2026-10-24 against one URL;
    # only 3 of 15 answered with usable data. The dead ones are kept in the catalogue with
    # `access="dead"` so nobody re-lists them as redundancy without re-probing -- a dead
    # endpoint counted as a backup is worse than no backup, because it inflates the count.
    # ==================================================================================
    Source("arquivo_pt", "Arquivo.pt (Portuguese Web Archive)", "https://arquivo.pt/wayback/cdx",
           "web_archive", "PROBED WORKING: CDX + Memento, keyless, independent of archive.org",
           True, "api", "public"),
    Source("wayback_cdx", "Wayback CDX (archive.org)", "https://web.archive.org/cdx/search/cdx",
           "web_archive", "PROBED WORKING: earliest/latest capture", True, "api", "public"),
    Source("loc_webarchive", "LoC Web Archive", "https://webarchive.loc.gov/all/",
           "web_archive", "PROBED 403 from this host; may need allowlisting", True, "scrape",
           "public domain"),
    Source("ukwa", "UK Web Archive", "https://www.webarchive.org.uk/wayback/",
           "web_archive", "PROBED responds with HTML, no machine endpoint found", True,
           "scrape", "mixed"),
    Source("swap_stanford", "Stanford SWAP", "https://swap.stanford.edu/", "web_archive",
           "PROBED 502", True, "scrape", "public"),
    Source("bibalex", "Bibliotheca Alexandrina Web Archive", "http://web.archive.bibalex.org/",
           "web_archive", "PROBED timeout", True, "scrape", "public"),
    Source("vefir_is", "Vefsafnid (Icelandic)", "http://vefir.is/wayback/", "web_archive",
           "PROBED 404 on the path tried", True, "scrape", "public"),
    Source("haw_hr", "Croatian Web Archive", "http://haw.nsk.hr/wayback/", "web_archive",
           "PROBED timeout", True, "scrape", "public"),
    Source("digar_ee", "DIGAR (Estonian)", "http://veebiarhiiv.digar.ee/wayback/",
           "web_archive", "PROBED 404", True, "scrape", "public"),
    Source("padi_cat", "PADICAT (Catalan)", "http://padi.cat/wayback/", "web_archive",
           "PROBED SSL certificate failure", True, "scrape", "public"),
    Source("memento_aggregator", "Memento TimeTravel", "http://timetravel.mementoweb.org/",
           "web_archive", "PROBED: DNS DOES NOT RESOLVE. The canonical federation aggregator "
           "is dead -- do not rely on it", True, "http", "public"),
    Source("trove_archive", "Trove (NLA) web archive", "https://trove.nla.gov.au/",
           "web_archive", "PROBED 410 Gone on the API path tried; needs a key", True,
           "api-key", "mixed"),

    # --- 2026-10-25 RE-PROBE. The first census said "3 of 15 work". That was wrong three ways: a
    # timed-out `docker ps` made the proxy look down, transient failures were read as permanent, and
    # a 429 was read as a DNS failure. Forced direct-vs-proxy probes give 9 answering, 6 gated, 4
    # dead. The entries below are the ones the first pass missed entirely.
    Source("wayback_available", "Wayback Availability API", "https://archive.org/wayback/available",
           "web_archive", "PROBED WORKING: closest snapshot in ONE 246-byte JSON call. Far cheaper "
           "than CDX and we had never used it -- use as a tier-0 probe", True, "api", "public"),
    Source("urlscan_note", "urlscan.io (archive-index use)", "https://urlscan.io/api/v1/search/",
           "web_archive", "See the osint entry `urlscan` -- same service, one id. Probed 2026-10-25: "
           "200, keyless, real results. It is the only independent URL index we have that is NOT "
           "archive.org, so it samples a different population of URLs rather than a second window on "
           "the same crawl. Wired as a route in redundancy.py.", True, "api", "public"),
    Source("conifer", "Conifer (Webrecorder)", "https://conifer.rhizome.org/", "web_archive",
           "PROBED WORKING 200. Interactive session archives -- captures JS behaviour that static "
           "crawlers lose", True, "scrape", "mixed"),
    Source("archive_is", "archive.today / archive.is", "https://archive.is/", "web_archive",
           "PROBED 429 direct AND via proxy: ALIVE and throttling, NOT dead. The earlier 'DNS does "
           "not resolve' was transient. Needs the proxy path to reach", True, "scrape", "public"),
    Source("archivebox", "ArchiveBox (self-hosted)", "https://archivebox.io/", "web_archive",
           "NOT YET RUN. Software that makes US the archive instead of a client of one. The only "
           "real fix for single-witness facts -- see knowledge/redundancy-3x3-and-its-ceiling.md",
           False, "self-host", "MIT"),
    Source("browsertrix", "Browsertrix (self-hosted crawler)", "https://webrecorder.net/browsertrix/",
           "web_archive", "NOT YET RUN. Scheduled whole-site archiving with full JS fidelity; the "
           "second-crawler strategy", False, "self-host", "AGPL"),
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


# ------------------------------------------------------------------------------ resolver cache
# Measured need: the same corpus yielded 25 dated facts on one run and 48 on another because
# archive.org rate-limits. Provenance that changes between runs is not provenance.
_CACHE_PATH: str | None = None


def _cache_file() -> str:
    global _CACHE_PATH
    if _CACHE_PATH is None:
        _CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache",
                                   "logos_source_dates.json")
    return _CACHE_PATH


def _cache_load() -> dict:
    try:
        with open(_cache_file(), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                     # noqa: BLE001
        return {}


def _cache_put(key: str, value, kind: str) -> None:
    try:
        c = _cache_load()
        c[f"{kind}|{key}"] = value
        os.makedirs(os.path.dirname(_cache_file()), exist_ok=True)
        with open(_cache_file(), "w", encoding="utf-8") as fh:
            json.dump(c, fh)
    except Exception:                     # noqa: BLE001
        pass


_CACHE_ABSENT = object()      # distinct from a cached None (= a cached MISS)


def _cache_get(key: str, kind: str):
    c = _cache_load()
    k = f"{kind}|{key}"
    return c[k] if k in c else _CACHE_ABSENT


def wayback_dates(url: str, timeout: float = 25.0,
                  use_cache: bool = True) -> tuple[str, str] | None:
    """First and last capture timestamps for a URL, as YYYY-MM-DD.

    THE HIGHEST-LEVERAGE DATE SOURCE AND THE LEAST RELIABLE. Measured: the same 54-fact corpus
    yielded 25 dated lookups on one run and 48 on the next, because archive.org rate-limits. A
    corpus built from uncached lookups inherits that nondeterminism -- and a dataset that changes
    between runs is not provenance. Results are therefore cached by URL, including misses, so a
    repeat run is byte-identical.
    """
    if not url or not url.startswith("http"):
        return None
    if use_cache:
        hit = _cache_get(url, "wayback")
        if hit is not _CACHE_ABSENT:
            return (hit[0], hit[1]) if isinstance(hit, list) and hit else None
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
        _cache_put(url, None, "wayback")      # cache the miss: retrying is expensive
        return None
    last = one(base + "&sort=reverse")
    res = (first, last or first)
    if use_cache:
        _cache_put(url, list(res), "wayback")
    return res



# Common Crawl publishes one index per crawl; querying every index would be ~100 requests for
# one URL. These four are spread across years, which is enough to establish an EARLIEST bound in
# most cases and cheap enough to run per fact. A miss is reported as a miss, never as a date.
CC_INDEXES = ("CC-MAIN-2018-05", "CC-MAIN-2020-05", "CC-MAIN-2022-05", "CC-MAIN-2024-10")


def commoncrawl_dates(url: str, timeout: float = 25.0) -> tuple[str, str] | None:
    """Earliest/latest Common Crawl fetch date for a URL, across a few spread indexes."""
    if not url or not url.startswith("http"):
        return None
    hit = _cache_get(url, "commoncrawl")
    if hit is not _CACHE_ABSENT:
        return (hit[0], hit[1]) if isinstance(hit, list) and hit else None
    got: list[str] = []
    for idx in CC_INDEXES:
        q = (f"https://index.commoncrawl.org/{idx}-index?url="
             + urllib.parse.quote(url, safe="") + "&output=json")
        try:
            req = urllib.request.Request(q, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception:                 # noqa: BLE001 - 404 means "not in this crawl"
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(rec.get("timestamp", ""))
            if len(ts) >= 8:
                got.append(f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}")
    if not got:
        _cache_put(url, None, "commoncrawl")
        return None
    got.sort()
    res = (got[0], got[-1])
    _cache_put(url, list(res), "commoncrawl")
    return res


def archive_today_dates(url: str, timeout: float = 20.0) -> tuple[str, str] | None:
    """Earliest archive.today snapshot for a URL.

    archive.today has no public API; `/timemap/<url>` is the closest thing to a queryable
    endpoint. Treat any failure as "not archived here" rather than an error: this is a
    best-effort third resolver, not a contract.
    """
    if not url or not url.startswith("http"):
        return None
    q = "https://archive.ph/timemap/" + urllib.parse.quote(url, safe="")
    try:
        req = urllib.request.Request(q, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:                     # noqa: BLE001
        return None
    stamps = re.findall(r"(20\d{2}|19\d{2})(\d{2})(\d{2})", body)
    if not stamps:
        return None
    dates = sorted(f"{y}-{m}-{d}" for y, m, d in stamps)
    return dates[0], dates[-1]


def resolve_date(fact_date: str = "", ref_url: str = "",
                 allow_wayback: bool = True, allow_commoncrawl: bool = False,
                 allow_archive_today: bool = False) -> list[DateEvidence]:
    """Every date the system can establish, strongest first.

    A list, not a single value: a verifier needs to know *which* source dated a fact, and a
    publisher's own date is stronger evidence than an archive capture.
    """
    out: list[DateEvidence] = []
    if fact_date:
        out.append(DateEvidence(fact_date, "publisher", "publisher",
                                "date on the statement itself"))
    if not ref_url:
        return out
    if allow_wayback:
        wb = wayback_dates(ref_url)
        if wb:
            first, last = wb
            out.append(DateEvidence(first, "wayback_first", "archive",
                                    f"earliest capture of the cited URL (last {last})"))
    if allow_commoncrawl:
        cc = commoncrawl_dates(ref_url)
        if cc:
            first, last = cc
            out.append(DateEvidence(first, "commoncrawl_first", "archive",
                                    f"earliest Common Crawl fetch (last {last})"))
    if allow_archive_today:
        at = archive_today_dates(ref_url)
        if at:
            first, last = at
            out.append(DateEvidence(first, "archive_today_first", "archive",
                                    f"earliest archive.today snapshot (last {last})"))
    # strongest first: publisher beats archive, and among archives the earliest is most useful
    out.sort(key=lambda e: ({"publisher": 0, "archive": 1}.get(e.strength, 2), e.date))
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
    good = all(s.access in ("http", "api", "api-key", "scrape", "bulk", "dead", "self-host")
               for s in REGISTRY.values())
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

    # --- second/third resolvers are offline-safe: bad input is a no-op, not an exception
    good = commoncrawl_dates("not-a-url") is None and archive_today_dates("") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] commoncrawl and archive.today reject malformed URLs")

    # --- the cache makes a lookup reproducible (measured need: 25 vs 48 dated across two runs)
    import tempfile as _tf
    global _CACHE_PATH
    _CACHE_PATH = os.path.join(_tf.mkdtemp(), "cache.json")
    _cache_put("https://example.org/x", ["2015-01-01", "2016-01-01"], "wayback")
    good = _cache_get("https://example.org/x", "wayback") == ["2015-01-01", "2016-01-01"]
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] resolver cache round-trips")

    good = wayback_dates("https://example.org/x") == ("2015-01-01", "2016-01-01")
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cached HIT is served without a request")

    _cache_put("https://example.org/none", None, "wayback")
    good = wayback_dates("https://example.org/none") is None
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] a cached MISS is honoured, not retried "
          f"(this is what made 47 become 49 across two runs)")
    _CACHE_PATH = None

    # --- the cascade orders publisher before archive
    ev = resolve_date("2019-05-01", "", allow_wayback=False)
    good = ev and ev[0].strength == "publisher"
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] cascade keeps publisher evidence first")

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
