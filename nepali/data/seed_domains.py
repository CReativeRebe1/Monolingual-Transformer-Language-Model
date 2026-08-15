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

Wikipedia is ingested from the Wikimedia dump (scripts/run_wiki.py) rather than crawled,
for the same reason as Hindi.

Reachability: every domain listed here was probed directly (robots.txt + homepage) before being
added -- roughly a quarter of an initial candidate list turned out to be dead hostnames, CDN
bot-blocks (403 on robots.txt), or connect timeouts, and those were removed rather than left in
to waste seeding time on every crawler startup. Re-probe if the list is extended.
"""

NEWS = [
    "ekantipur.com",
    "www.onlinekhabar.com",
    "www.setopati.com",
    "ratopati.com",
    "www.nagariknews.nagariknetwork.com",
    "www.himalkhabar.com",
    "gorkhapatraonline.com",
    "www.ujyaaloonline.com",
    "www.nepalkhabar.com",
    "www.thahakhabar.com",
    "www.baahrakhari.com",
    "www.annapurnapost.com",
    "www.hamrakura.com",
    "www.imagekhabar.com",
    "www.nepalnews.com",
    "www.karobardaily.com",
    "www.clickmandu.com",
    "www.reportersnepal.com",
    "swasthyakhabar.com",
    "www.techpana.com",
    "www.nepalisamachar.com",
    "www.nepalpage.com",
    "www.pardafas.com",
    "newsofnepal.com",
    # Added from a user-supplied candidate list; probed individually (robots.txt + homepage,
    # >2KB real body) before inclusion. "farakdhar.com" and "ukaalo.com" from the same list
    # were dropped -- both 403 (bot-blocked), same class of failure as lokaantar.com and
    # myrepublica.nagariknetwork.com noted below.
    "bizmandu.com",
    "deshsanchar.com",
    "www.dainiknepal.com",
    "www.drishtinews.com",
    "ehimalayatimes.com",
    "www.hamrokhelkud.com",
    "kendrabindu.com",
    "khabarhub.com",
    "makalukhabar.com",
    "nepallive.com",
    "nepalpress.com",
    "nepalsamaya.com",
    "www.news24nepal.com",
    "rajdhanidaily.com",
    "shilapatra.com",
    "www.bbc.com",
]

# Confirmed dead/blocked, do not re-add without re-probing: www.lokaantar.com (403),
# myrepublica.nagariknetwork.com (403), farakdhar.com (403), ukaalo.com (403),
# nepal.gov.np / moe.gov.np (connect timeout).

LITERARY_CULTURAL = [
    "www.sahityasangraha.com",
]

# Education: deep, crawler-friendly Nepali-language content, and a register
# (curricula, institutional prose) absent from the news-heavy list above.
EDUCATION = [
    "edusanjal.com",
]

# No government domains: nepal.gov.np and moe.gov.np were both tried and both time out from
# here (ConnectTimeout, verified directly -- see the module docstring's reachability note).
# Listing unreachable domains would only waste seeding time on every crawler startup.

ALL_DOMAINS = NEWS + LITERARY_CULTURAL + EDUCATION

DOMAIN_URL_FILTERS = {
    "www.bbc.com": "/nepali/",
}
