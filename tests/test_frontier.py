"""Tests for lma.crawl.frontier.Frontier — the resumable URL queue.

These are offline (no network) tests of the resume semantics: the property the crawler depends
on for days-long unattended operation is that killing the process and reopening the same
SQLite file never loses a URL and never silently duplicates a fetch.
"""
import sqlite3

import pytest

from lma.crawl.frontier import Frontier


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "frontier.sqlite3"


def test_add_is_deduplicated(db_path):
    with Frontier(db_path) as f:
        assert f.add("https://example.com/a") is True
        assert f.add("https://example.com/a") is False  # already known
        assert f.stats().total == 1


def test_claim_batch_marks_in_progress(db_path):
    with Frontier(db_path) as f:
        f.add_many([f"https://example.com/{i}" for i in range(5)])
        batch = f.claim_batch(n=3)
        assert len(batch) == 3
        stats = f.stats()
        assert stats.in_progress == 3
        assert stats.pending == 2


def test_mark_done_transitions_status(db_path):
    with Frontier(db_path) as f:
        f.add("https://example.com/a")
        f.claim_batch(n=1)
        f.mark_done("https://example.com/a", http_status=200, content_sha256="abc123")
        stats = f.stats()
        assert stats.done == 1
        assert stats.in_progress == 0


def test_reopening_frontier_requeues_stuck_in_progress_rows(db_path):
    """The core resume guarantee: a crash mid-fetch must not lose the URL."""
    with Frontier(db_path) as f:
        f.add("https://example.com/a")
        f.claim_batch(n=1)  # simulate: claimed, then process died before mark_done/mark_failed
        assert f.stats().in_progress == 1

    # Reopening (as a fresh process after a restart would) must requeue it as pending.
    with Frontier(db_path) as f:
        stats = f.stats()
        assert stats.in_progress == 0
        assert stats.pending == 1


def test_done_and_failed_urls_are_not_reclaimed_by_default(db_path):
    with Frontier(db_path) as f:
        f.add("https://example.com/done")
        f.add("https://example.com/failed")
        f.claim_batch(n=2)
        f.mark_done("https://example.com/done", 200, "hash1")
        f.mark_failed("https://example.com/failed", 500, "http_500")
        # A fresh claim should find nothing pending left.
        assert f.claim_batch(n=10) == []


def test_requeue_failed_respects_max_attempts(db_path):
    with Frontier(db_path) as f:
        f.add("https://example.com/flaky")
        for _ in range(3):
            batch = f.claim_batch(n=1)
            if not batch:
                f.requeue_failed(max_attempts=3)
                batch = f.claim_batch(n=1)
            f.mark_failed(batch[0][0], 503, "http_503")
        # attempts should now be 3; requeue_failed(max_attempts=3) must NOT bring it back
        requeued = f.requeue_failed(max_attempts=3)
        assert requeued == 0
        assert f.claim_batch(n=1) == []


def test_known_content_hashes_only_includes_fetched(db_path):
    with Frontier(db_path) as f:
        f.add("https://example.com/a")
        f.add("https://example.com/b")
        f.claim_batch(n=2)
        f.mark_done("https://example.com/a", 200, "hash_a")
        f.mark_failed("https://example.com/b", 500, "err")
        assert f.known_content_hashes() == {"hash_a"}


def test_wal_mode_enabled(db_path):
    with Frontier(db_path):
        pass
    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


# --- Per-domain seeding tracking (see _SCHEMA's seeded_domains comment) ---

def test_domain_not_seeded_by_default(db_path):
    with Frontier(db_path) as f:
        assert f.is_domain_seeded("example.com") is False


def test_mark_domain_seeded_persists(db_path):
    with Frontier(db_path) as f:
        f.mark_domain_seeded("example.com", urls_found=42)
        assert f.is_domain_seeded("example.com") is True
        assert f.seeded_domains() == {"example.com": 42}


def test_seeded_domains_survive_reopen(db_path):
    # The core bug this table fixes: a crawler restart must remember which domains were
    # already seeded, not just whether the frontier as a whole happens to be non-empty.
    with Frontier(db_path) as f:
        f.mark_domain_seeded("amarujala.com", urls_found=48000)

    with Frontier(db_path) as f:
        assert f.is_domain_seeded("amarujala.com") is True
        assert f.is_domain_seeded("jagran.com") is False


def test_multiple_domains_tracked_independently(db_path):
    with Frontier(db_path) as f:
        f.mark_domain_seeded("a.com", urls_found=10)
        f.mark_domain_seeded("b.com", urls_found=20)
        assert f.seeded_domains() == {"a.com": 10, "b.com": 20}
