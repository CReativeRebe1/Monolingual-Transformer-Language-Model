"""
SQLite-backed URL frontier: the resumable heart of the crawler.

Crawling runs for days across many separate process invocations (the machine sleeps, the
terminal closes, a run is deliberately restarted). The frontier is the single source of truth
for "what have we already tried" so that restarting the crawler never re-fetches a URL and never
loses progress — this is the data-collection analogue of the checkpoint-resume requirement that
Phase 2 imposes on model training.

Schema (one row per URL ever seen):
    url         TEXT PRIMARY KEY
    host        TEXT            -- for per-host politeness bookkeeping
    status      TEXT            -- 'pending' | 'in_progress' | 'done' | 'failed' | 'skipped'
    depth       INTEGER         -- crawl depth from a seed, for optional depth limiting
    discovered_at   TEXT (ISO8601)
    fetched_at      TEXT (ISO8601), nullable
    http_status     INTEGER, nullable
    content_sha256  TEXT, nullable  -- of the *extracted* text, for the exact-dedup stage
    error           TEXT, nullable
    attempts        INTEGER default 0
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    url             TEXT PRIMARY KEY,
    host            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    depth           INTEGER NOT NULL DEFAULT 0,
    discovered_at   TEXT NOT NULL,
    fetched_at      TEXT,
    http_status     INTEGER,
    content_sha256  TEXT,
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_frontier_status ON frontier(status);
CREATE INDEX IF NOT EXISTS idx_frontier_host ON frontier(host, status);

-- Tracks which SEED DOMAINS have completed sitemap-discovery seeding at least once. This is
-- deliberately separate from "is the frontier non-empty": a naive "only seed if frontier is
-- empty" check means that once the FIRST domain in a multi-domain seed list contributes any
-- URL, every later restart of the crawler (which happens often during iterative development,
-- and will happen again whenever a Kaggle/Colab session is interrupted) skips seeding entirely
-- -- silently starving every domain later in the list that had not been reached yet. Tracking
-- completion per domain means a restart always finishes seeding whatever domains are left,
-- while never wastefully re-walking an already-seeded domain's sitemap.
CREATE TABLE IF NOT EXISTS seeded_domains (
    domain      TEXT PRIMARY KEY,
    seeded_at   TEXT NOT NULL,
    urls_found  INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class FrontierStats:
    pending: int
    in_progress: int
    done: int
    failed: int
    skipped: int
    total: int


class Frontier:
    """Resumable URL queue backed by a local SQLite file.

    Usage:
        with Frontier(db_path) as f:
            f.add_many(seed_urls)
            while (batch := f.claim_batch(host_cursor=..., n=50)):
                ...
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")  # safe under concurrent readers
        self._conn.executescript(_SCHEMA)
        # A crawler that crashed mid-run leaves rows stuck 'in_progress'; on open, requeue them
        # rather than losing them — this is precisely the resume-after-interruption behaviour.
        self._conn.execute(
            "UPDATE frontier SET status='pending' WHERE status='in_progress'"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._conn.commit()
        self._conn.close()

    # --- writes ---

    def add(self, url: str, depth: int = 0) -> bool:
        """Insert a URL if unseen. Returns True if newly added, False if already known."""
        try:
            self._conn.execute(
                "INSERT INTO frontier (url, host, status, depth, discovered_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (url, _host_of(url), depth, _now_iso()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def add_many(self, urls, depth: int = 0) -> int:
        added = 0
        for u in urls:
            if self.add(u, depth=depth):
                added += 1
        return added

    def claim_batch(self, n: int = 50) -> list[tuple[str, str, int]]:
        """Atomically mark up to n pending URLs 'in_progress' and return (url, host, depth).

        Selection is ROUND-ROBIN across distinct hosts, not plain insertion-order FIFO: a
        window-ranked query takes each host's earliest-discovered pending URL first, then each
        host's second-earliest, and so on, until n URLs are claimed. Without this, a domain
        whose sitemap was bulk-seeded with thousands of URLs in one shot (e.g. a major news
        portal) completely dominates every batch over domains that only ever produced a handful
        of URLs via the link-following fallback (a sitemap-less site seeded from just its
        homepage) -- confirmed live: bbc.com's 4,824 bulk-seeded pending URLs vs. single-digit
        pending counts for five other Nepali domains meant 1,759 of 1,766 fetched Nepali
        documents came from one host. Round-robin claiming keeps every active host flowing into
        every batch, which is what the Phase-1 report's "domain distribution" statistic and
        figure actually need to look like a real, diverse corpus rather than one dominant site.

        A single UPDATE keeps this safe even if multiple crawler processes share one frontier
        file (SQLite serializes writers), which matters because Hindi and Nepali crawls run as
        two independent processes against two independent frontier DBs in practice, but the
        same code path must also be safe if ever pointed at a shared one.
        """
        cur = self._conn.execute(
            """
            WITH ranked AS (
                SELECT url, host, depth,
                       ROW_NUMBER() OVER (PARTITION BY host ORDER BY rowid) AS rn
                FROM frontier
                WHERE status = 'pending'
            )
            SELECT url, host, depth FROM ranked ORDER BY rn, host LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        urls = [r[0] for r in rows]
        self._conn.executemany(
            "UPDATE frontier SET status='in_progress' WHERE url=?",
            [(u,) for u in urls],
        )
        return rows

    def mark_done(self, url: str, http_status: int, content_sha256: str | None) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='done', fetched_at=?, http_status=?, content_sha256=?, "
            "attempts=attempts+1 WHERE url=?",
            (_now_iso(), http_status, content_sha256, url),
        )

    def mark_skipped(self, url: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='skipped', fetched_at=?, error=?, attempts=attempts+1 "
            "WHERE url=?",
            (_now_iso(), reason, url),
        )

    def mark_failed(self, url: str, http_status: int | None, error: str) -> None:
        self._conn.execute(
            "UPDATE frontier SET status='failed', fetched_at=?, http_status=?, error=?, "
            "attempts=attempts+1 WHERE url=?",
            (_now_iso(), http_status, error, url),
        )

    def requeue_failed(self, max_attempts: int = 3) -> int:
        """Give failed URLs under the attempt cap another chance (e.g. transient 5xx/timeout)."""
        cur = self._conn.execute(
            "UPDATE frontier SET status='pending' WHERE status='failed' AND attempts < ?",
            (max_attempts,),
        )
        return cur.rowcount

    # --- reads ---

    def seen(self, url: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM frontier WHERE url=?", (url,))
        return cur.fetchone() is not None

    def stats(self) -> FrontierStats:
        counts = dict(
            self._conn.execute(
                "SELECT status, COUNT(*) FROM frontier GROUP BY status"
            ).fetchall()
        )
        total = sum(counts.values())
        return FrontierStats(
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            done=counts.get("done", 0),
            failed=counts.get("failed", 0),
            skipped=counts.get("skipped", 0),
            total=total,
        )

    def known_content_hashes(self) -> set[str]:
        """All content_sha256 already fetched — used for a cheap streaming near-dupe skip."""
        rows = self._conn.execute(
            "SELECT content_sha256 FROM frontier WHERE content_sha256 IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}

    # --- per-domain seeding tracking (see _SCHEMA's seeded_domains comment for why this
    # exists as its own table rather than inferring "already seeded" from frontier contents) ---

    def is_domain_seeded(self, domain: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seeded_domains WHERE domain=?", (domain,))
        return cur.fetchone() is not None

    def mark_domain_seeded(self, domain: str, urls_found: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO seeded_domains (domain, seeded_at, urls_found) "
            "VALUES (?, ?, ?)",
            (domain, _now_iso(), urls_found),
        )

    def seeded_domains(self) -> dict[str, int]:
        """{domain: urls_found_at_seed_time} for every domain seeded so far."""
        rows = self._conn.execute("SELECT domain, urls_found FROM seeded_domains").fetchall()
        return dict(rows)


@contextmanager
def open_frontier(db_path: str | Path):
    f = Frontier(db_path)
    try:
        yield f
    finally:
        f._conn.commit()
        f._conn.close()
