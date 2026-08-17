"""Tests for hindi.clean.run's shard reading -- specifically its tolerance of a shard that is
corrupted or truncated mid-write, which is a REAL, live scenario: the crawler is a long-lived
concurrent process, and re-running the corpus build against the live corpus crashed with
`zlib.error: Error -3 ... invalid block type` while reading a shard the crawler was actively
appending to. Catching only (EOFError, OSError) covers a clean truncation (gzip.BadGzipFile) but
not a torn DEFLATE block caught mid-flush, which zlib raises as its own zlib.error.

The reader moved from scripts/build_corpus.py into each language's clean/run.py when the cascade
was split per language. The tolerance is the thing being tested, so the tests moved with it; the
Nepali copy is identical code, so testing one language covers the shared behaviour.
"""
import gzip
import json
from pathlib import Path

from hindi.clean import run as pipeline


def _write_shard(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_reads_a_well_formed_shard_completely(tmp_path):
    records = [{"url": f"https://x.com/{i}", "text": f"doc {i}"} for i in range(5)]
    shard_path = tmp_path / "shard.jsonl.gz"
    _write_shard(shard_path, records)

    read_back = list(pipeline.read_shards(shard_path.parent))
    assert [r["text"] for r in read_back] == [r["text"] for r in records]


def test_tolerates_a_cleanly_truncated_gzip_file(tmp_path, caplog):
    records = [{"url": f"https://x.com/{i}", "text": f"doc number {i} with some content"} for i in range(20)]
    shard_path = tmp_path / "shard.jsonl.gz"
    _write_shard(shard_path, records)

    full_bytes = shard_path.read_bytes()
    shard_path.write_bytes(full_bytes[: len(full_bytes) // 2])  # simulate an interrupted flush

    # Must not raise -- this is the whole point of the function.
    read_back = list(pipeline.read_shards(shard_path.parent))
    assert len(read_back) < len(records)  # partial read is expected and fine
    assert len(read_back) >= 0


def test_tolerates_a_torn_deflate_block_mid_stream(tmp_path):
    # Reproduces the EXACT real failure mode: corrupting a byte in the middle of the compressed
    # stream (rather than truncating cleanly at the end) triggers zlib.error, not EOFError/OSError.
    records = [{"url": f"https://x.com/{i}", "text": f"doc number {i} with quite a bit of content here"} for i in range(30)]
    shard_path = tmp_path / "shard.jsonl.gz"
    _write_shard(shard_path, records)

    raw = bytearray(shard_path.read_bytes())
    # Flip a byte roughly in the middle of the compressed DEFLATE stream (past the gzip header,
    # before the end) to corrupt a block without truncating the file -- this is what a
    # concurrent partial flush from another process can look like.
    mid = len(raw) // 2
    raw[mid] ^= 0xFF
    shard_path.write_bytes(bytes(raw))

    # Must not raise zlib.error -- this was the actual bug found live.
    read_back = list(pipeline.read_shards(shard_path.parent))
    assert isinstance(read_back, list)  # didn't crash; content recovered (if any) is a bonus


def test_iter_raw_records_continues_past_one_corrupted_shard(tmp_path):
    # load_manual reads multiple shards; one corrupted shard must not prevent records from
    # OTHER shards (or later in the raw/ tree) from being read.
    lang_dir = tmp_path / "hindi"
    good_shard = lang_dir / "data" / "raw" / "crawled" / "shard_00000.jsonl.gz"
    bad_shard = lang_dir / "data" / "raw" / "crawled" / "shard_00001.jsonl.gz"

    _write_shard(good_shard, [{"url": "https://a.com/1", "text": "a real document"}])
    _write_shard(bad_shard, [{"url": "https://b.com/1", "text": "another real document"}])
    raw = bytearray(bad_shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad_shard.write_bytes(bytes(raw))

    all_records = list(pipeline.load_manual(lang_dir))
    urls = [r["url"] for r in all_records]
    assert "https://a.com/1" in urls  # the good shard's record must still come through
