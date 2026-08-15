"""
Seed domains for the Hindi (Model H) manual-collection crawl.

Selection criteria: (1) Hindi-primary content, (2) deep, sitemap-exposed archives so a handful
of domains yield a large volume of article URLs, (3) source-type diversity (news, literary,
cultural, science, government) so the corpus is not dominated by one register or one outlet's
house style, which the Phase-1 report discusses under "domain distribution".

Throughput note: per-host politeness (1.5s floor, lma/crawl/robots.py) means a single domain
caps at well under 1 doc/s no matter how high --concurrency goes. Total crawl throughput is
therefore roughly (number of live domains) / (per-host delay), so the domain COUNT here is the
primary throughput lever, not concurrency. This list is deliberately wide for that reason.

Wikipedia is deliberately NOT crawled here: the Wikimedia dump (scripts/run_wiki.py) delivers
the same articles as bulk plain text far faster than walking HTML page-by-page under a politeness
delay, so crawl budget is better spent on sources that have no bulk-export path.

robots.txt is honoured per-domain at crawl time regardless of what is listed here; a domain that
disallows crawling, or does not resolve, simply yields nothing and is noted in the report.

DOMAIN_URL_FILTERS: some domains (bbc.com in particular) expose one sitemap tree shared across
every language edition. Without a filter, sitemap discovery would seed the frontier mostly with
non-Hindi URLs from that global index, wasting crawl budget and politeness allowance on pages a
downstream language-ID filter would just discard.

Reachability: every domain listed here was probed directly (robots.txt + homepage) before being
added -- roughly a quarter of an initial candidate list turned out to be dead hostnames, CDN
bot-blocks (403 on robots.txt), or connect timeouts, and those were removed rather than left in
to waste seeding time on every crawler startup. Re-probe if the list is extended.
"""

NEWS = [
    "www.amarujala.com",
    "www.jagran.com",
    "navbharattimes.indiatimes.com",
    "www.bhaskar.com",
    "www.livehindustan.com",
    "www.aajtak.in",
    "www.abplive.com",
    "zeenews.india.com",
    "www.patrika.com",
    "www.naidunia.com",
    "hindi.oneindia.com",
    "www.jansatta.com",
    "www.prabhasakshi.com",
    "www.punjabkesari.in",
    "www.tv9hindi.com",
    "www.indiatv.in",
    "hindi.moneycontrol.com",
    "www.haribhoomi.com",
    "www.newsnationtv.com",
    "www.dainiktribuneonline.com",
    # bbc.com last: its sitemap is a single global archive index shared across every language
    # edition, sharded into 100+ dated sub-sitemaps -- by far the slowest domain to seed from
    # (see lma/crawl/sitemap.py's max_sitemaps_total budget). Seeding it last means every
    # faster, higher-yield domain is already flowing into the frontier before crawl time is
    # spent walking BBC's archive.
    "www.bbc.com",
]

# Literary / cultural / science: deliberately weighted alongside news so the corpus is not
# ~100% contemporary journalese. These carry older, richer, more varied vocabulary (poetry,
# prose fiction, criticism, popular science) that news archives simply do not contain, which
# directly affects tokenizer fertility and downstream LM quality.
LITERARY_CULTURAL = [
    "www.hindisamay.com",
    "www.gadyakosh.org",
    "www.kavitakosh.org",
    "www.anubhuti-hindi.org",
    "www.bharatdiscovery.org",
    "www.hindwi.org",
    "www.pustak.org",
    "www.rekhta.org",
]

GOVERNMENT_SCIENCE = [
    "ignca.gov.in",
    "www.mygov.in",
]

ALL_DOMAINS = NEWS + LITERARY_CULTURAL + GOVERNMENT_SCIENCE

DOMAIN_URL_FILTERS = {
    "www.bbc.com": "/hindi/",
}
