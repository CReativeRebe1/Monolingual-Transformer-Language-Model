"""
Seed domains for the Nepali (Model L) manual-collection crawl.

Same selection criteria as hindi/data/seed_domains.py: sitemap-exposed archives, source-type
diversity, Nepali-primary content. Nepali news portals are the highest-yield source available
to a self-run crawler in the project timeframe; the encyclopedic/government sources add
register diversity but are individually smaller than any one news portal's archive.

See hindi/data/seed_domains.py for why DOMAIN_URL_FILTERS exists: bbc.com shares one sitemap
tree across every language edition, so a substring filter keeps only the /nepali/ URLs from it.
"""

NEWS = [
    "ekantipur.com",
    "www.onlinekhabar.com",
    "www.setopati.com",
    "ratopati.com",
    "www.nagariknews.nagariknetwork.com",
    "www.bbc.com",          # BBC Nepali is served under /nepali/ paths on this shared-sitemap domain
    "www.himalkhabar.com",
]

ENCYCLOPEDIC_LITERARY = [
    "ne.wikipedia.org",
    "ne.wikisource.org",
]

GOVERNMENT = [
    "www.nepal.gov.np",
]

ALL_DOMAINS = NEWS + ENCYCLOPEDIC_LITERARY + GOVERNMENT

DOMAIN_URL_FILTERS = {
    "www.bbc.com": "/nepali/",
}
