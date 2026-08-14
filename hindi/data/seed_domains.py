"""
Seed domains for the Hindi (Model H) manual-collection crawl.

Selection criteria: (1) Hindi-primary content, (2) deep, sitemap-exposed archives so a handful
of domains yield a large volume of article URLs, (3) source-type diversity (news, encyclopedic,
literary, government) so the corpus is not dominated by one register or one outlet's house style,
which the Phase-1 report discusses under "domain distribution".

robots.txt is honoured per-domain at crawl time (lma/crawl/robots.py) regardless of what is
listed here; a domain that disallows crawling simply yields nothing and is noted in the report.

DOMAIN_URL_FILTERS: some domains (bbc.com in particular) expose one sitemap tree shared across
every language edition of the site. Without a filter, sitemap discovery would seed the frontier
mostly with non-Hindi URLs from that same global index, wasting crawl budget and rate-limit
politeness allowance on pages that a downstream language-ID filter would just discard anyway.
The value is a URL substring that a leaf URL must contain to be kept; domains not listed here are
already Hindi-primary in their entirety and need no filtering.
"""

NEWS = [
    "www.amarujala.com",
    "www.jagran.com",
    "navbharattimes.indiatimes.com",
    "www.bhaskar.com",
    "www.livehindustan.com",
    "www.aajtak.in",
    "www.abplive.com",
    # bbc.com last: its sitemap is a single global archive index shared across every language
    # edition, sharded into 100+ dated sub-sitemaps -- by far the slowest domain to seed from
    # (see lma/crawl/sitemap.py's max_sitemaps_total budget). Seeding it last means every
    # faster, higher-yield domain is already flowing into the frontier before crawl time is
    # spent walking BBC's archive.
    "www.bbc.com",          # BBC Hindi is served under /hindi/ paths on this shared-sitemap domain
]

ENCYCLOPEDIC_LITERARY = [
    "hi.wikipedia.org",
    "hi.wikisource.org",
    "hi.wiktionary.org",
]

GOVERNMENT = [
    "www.india.gov.in",
    "pib.gov.in",
]

ALL_DOMAINS = NEWS + ENCYCLOPEDIC_LITERARY + GOVERNMENT

DOMAIN_URL_FILTERS = {
    "www.bbc.com": "/hindi/",
}
