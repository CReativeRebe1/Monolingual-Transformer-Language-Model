"""
Seed domains for the Hindi (Model H) manual-collection crawl.

Selection criteria: (1) Hindi-primary content, (2) deep, sitemap-exposed archives so a handful
of domains yield a large volume of article URLs, (3) source-type diversity (news, encyclopedic,
literary, government) so the corpus is not dominated by one register or one outlet's house style,
which the Phase-1 report discusses under "domain distribution".

robots.txt is honoured per-domain at crawl time (lma/crawl/robots.py) regardless of what is
listed here; a domain that disallows crawling simply yields nothing and is noted in the report.
"""

NEWS = [
    "www.bbc.com",          # BBC Hindi is served under /hindi paths on this domain
    "www.amarujala.com",
    "www.jagran.com",
    "navbharattimes.indiatimes.com",
    "www.bhaskar.com",
    "www.livehindustan.com",
    "www.aajtak.in",
    "www.abplive.com",
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
