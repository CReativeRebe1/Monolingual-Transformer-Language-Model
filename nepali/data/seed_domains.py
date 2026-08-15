"""
Seed domains for the Nepali (Model L) manual-collection crawl.

Same selection criteria and throughput reasoning as hindi/data/seed_domains.py -- see that file's
docstring; the domain COUNT is the throughput lever, not --concurrency, because per-host
politeness caps any single domain regardless.

This list was widened substantially after the original 10 domains drove the Nepali frontier's
pending queue down to ~4,000 URLs (from 62,975 discovered), i.e. link-following plus the few
usable sitemaps had nearly exhausted what those domains could discover. Several Nepali outlets
return 404/502 for every standard sitemap path (confirmed live -- see PHASE1_LOG.md), so the
crawler falls back to same-domain link-following for them; that fallback discovers URLs far more
slowly per domain, which makes breadth across many domains matter more here than it does for
Hindi.

Wikipedia/Wikisource are ingested via the MediaWiki API (lma/crawl/wiki_api.py) rather than
crawled, for the same reason as Hindi.
"""

NEWS = [
    "ekantipur.com",
    "www.onlinekhabar.com",
    "www.setopati.com",
    "ratopati.com",
    "www.nagariknews.nagariknetwork.com",
    "www.himalkhabar.com",
    "www.gorkhapatraonline.com",
    "www.ujyaaloonline.com",
    "www.nepalkhabar.com",
    "www.dharaharapost.com",
    "www.thaha.com",
    "www.baahrakhari.com",
    "www.annapurnapost.com",
    "www.lokaantar.com",
    "www.hamrakura.com",
    "www.imagekhabar.com",
    "www.nepalnews.com",
    "www.karobardaily.com",
    "www.clickmandu.com",
    "www.reportersnepal.com",
    "swasthyakhabar.com",
    "www.techpana.com",
    "www.bbc.com",
]

LITERARY_CULTURAL = [
    "www.sahityasangraha.com",
    "www.nepalsahitya.com",
    "www.myrepublica.nagariknetwork.com",
]

GOVERNMENT = [
    "www.nepal.gov.np",
    "moe.gov.np",
]

ALL_DOMAINS = NEWS + LITERARY_CULTURAL + GOVERNMENT

DOMAIN_URL_FILTERS = {
    "www.bbc.com": "/nepali/",
}
