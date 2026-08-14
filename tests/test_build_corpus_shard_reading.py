"""Tests for scripts/build_corpus.py's _iter_shard_records -- specifically its tolerance of a
shard file that is corrupted or truncated mid-write, which is a REAL, live scenario now that the
crawler runs as a genuinely long-lived concurrent process (not something a build_corpus.py run
always outlives): re-running build_corpus.py against the live corpus crashed with
`zlib.error: Error -3 ... invalid block type` reading a shard the crawler was actively appending
to. The original exception handling only caught (EOFError, OSError), which covers a clean
truncation (e.g. gzip.BadGzipFile) but not a torn DEFLATE block caught mid-flush, which zlib
reports as its own zlib.error instead.
"""
import gzip
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_reads_a_well_formed_shard_completely(tmp_path):
    module = _load_module()
    records = [{"url": f"https://x.com/{i}", "text": f"doc {i}"} for i in range(5)]
    shard_path = tmp_path / "shard.jsonl.gz"
    _write_shard(shard_path, records)

    read_back = list(module._iter_shard_records(shard_path))
    assert [r["text"] for r in read_back] == [r["text"] for r in records]


def test_tolerates_a_cleanly_truncated_gzip_file(tmp_path, caplog):
    module = _load_module()
    records = [{"url": f"https://x.com/{i}", "text": f"doc number {i} with some content"} for i in range(20)]
    shard_path = tmp_path / "shard.jsonl.gz"
    _write_shard(shard_path, records)

    full_bytes = shard_path.read_bytes()
    shard_path.write_bytes(full_bytes[: len(full_bytes) // 2])  # simulate an interrupted flush

    # Must not raise -- this is the whole point of the function.
    read_back = list(module._iter_shard_records(shard_path))
    assert len(read_back) < len(records)  # partial read is expected and fine
    assert len(read_back) >= 0


def test_tolerates_a_torn_deflate_block_mid_stream(tmp_path):
    # Reproduces the EXACT real failure mode: corrupting a byte in the middle of the compressed
    # stream (rather than truncating cleanly at the end) triggers zlib.error, not EOFError/OSError.
    module = _load_module()
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
    read_back = list(module._iter_shard_records(shard_path))
    assert isinstance(read_back, list)  # didn't crash; content recovered (if any) is a bonus


def test_iter_raw_records_continues_past_one_corrupted_shard(tmp_path):
    # _iter_raw_records reads multiple shards; one corrupted shard must not prevent records from
    # OTHER shards (or later in the raw/ tree) from being read.
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    good_shard = lang_dir / "data" / "raw" / "crawled" / "shard_00000.jsonl.gz"
    bad_shard = lang_dir / "data" / "raw" / "crawled" / "shard_00001.jsonl.gz"

    _write_shard(good_shard, [{"url": "https://a.com/1", "text": "a real document"}])
    _write_shard(bad_shard, [{"url": "https://b.com/1", "text": "another real document"}])
    raw = bytearray(bad_shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad_shard.write_bytes(bytes(raw))

    all_records = list(module._iter_raw_records(lang_dir))
    urls = [r["url"] for r in all_records]
    assert "https://a.com/1" in urls  # the good shard's record must still come through
