"""
robots.txt compliance and per-host politeness.

Two responsibilities: (1) never fetch a URL a site's robots.txt disallows for our user agent,
and (2) never exceed a safe request rate against a single host, honouring a site's declared
Crawl-delay when present and otherwise defaulting to a conservative floor. This is the
mechanism behind the "crawl politeness" claims in the Phase-1 report.
"""
from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

# Identify ourselves honestly, with a contact point — this is what makes automated fetching
# distinguishable from abuse, and is a courtesy expected by site operators.
USER_AGENT = (
    "CL3410-LMA-ResearchCrawler/0.1 "
    "(+educational data collection for a university NLP course project; "
    "contact: ishaangiri9786@gmail.com)"
)

DEFAULT_MIN_DELAY_SECONDS = 1.0  # floor even when robots.txt states no Crawl-delay


@dataclass
class HostPolicy:
    robots: urllib.robotparser.RobotFileParser | None
    min_delay: float
    last_request_at: float = field(default=0.0)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PolitenessManager:
    """Fetches and caches robots.txt per host; serializes and paces requests per host."""

    def __init__(self, client: httpx.AsyncClient, default_min_delay: float = DEFAULT_MIN_DELAY_SECONDS):
        self._client = client
        self._default_min_delay = default_min_delay
        self._policies: dict[str, HostPolicy] = {}
        self._policy_locks: dict[str, asyncio.Lock] = {}

    async def _get_policy(self, host: str) -> HostPolicy:
        if host in self._policies:
            return self._policies[host]
        # avoid two coroutines racing to fetch the same host's robots.txt concurrently
        lock = self._policy_locks.setdefault(host, asyncio.Lock())
        async with lock:
            if host in self._policies:
                return self._policies[host]
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"https://{host}/robots.txt"
            try:
                resp = await self._client.get(robots_url, timeout=10.0, follow_redirects=True)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp = None  # no robots.txt => permissive by convention
            except (httpx.HTTPError, httpx.TimeoutException):
                rp = None  # unreachable robots.txt: proceed with the conservative default delay
            delay = self._default_min_delay
            if rp is not None:
                crawl_delay = rp.crawl_delay(USER_AGENT)
                if crawl_delay is not None:
                    delay = max(delay, float(crawl_delay))
            policy = HostPolicy(robots=rp, min_delay=delay)
            self._policies[host] = policy
            return policy

    async def allowed(self, url: str) -> bool:
        host = urlsplit(url).netloc
        policy = await self._get_policy(host)
        if policy.robots is None:
            return True
        return policy.robots.can_fetch(USER_AGENT, url)

    async def wait_turn(self, url: str) -> None:
        """Block until it is this host's turn, respecting min_delay since the last request."""
        host = urlsplit(url).netloc
        policy = await self._get_policy(host)
        async with policy.lock:
            elapsed = time.monotonic() - policy.last_request_at
            remaining = policy.min_delay - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            policy.last_request_at = time.monotonic()
