"""Tests for scripts/fetch_downloaded.py and lma/dedup/index.py.

The dedup direction is the thing worth pinning down: duplicates must be dropped from the
DOWNLOADED side and never from the manual side, because manual tokens are what the assignment's
>=20% floor is measured against and downloaded tokens are replaceable for free.

`load_dataset` and the HF auth check are mocked throughout -- the suite never touches the network.
"""
import gzip
import json
from pathlib import Path

import pytest

from lma.dedup.index import DupIndex, hash_key

MANUAL_A = (
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं। "
    "यह शहर अपनी ऐतिहासिक इमारतों और संस्कृति के लिए जाना जाता है और हर साल लाखों पर्यटक यहाँ आते हैं। "
    "दिल्ली में मौसम गर्मियों में बहुत गर्म और सर्दियों में काफी ठंडा रहता है इसलिए यात्रा का समय महत्वपूर्ण है।"
)
# Same article, one word changed -- exact hash differs, MinHash should still catch it.
MANUAL_A_REWORDED = MANUAL_A.replace("बहुत बड़ा शहर", "काफी बड़ा शहर")
UNRELATED = (
    "नेपालमा दशैं सबैभन्दा ठूलो चाड मानिन्छ र यस अवसरमा परिवारका सदस्यहरू भेटघाट गर्छन्। "
    "यो चाड पन्ध्र दिनसम्म मनाइन्छ र यस अवधिमा घर-घरमा उत्सवको माहोल हुन्छ। "
    "विदेशमा बसोबास गर्ने नेपालीहरू पनि यस अवसरमा आफ्नो घर फर्कने प्रयास गर्छन्।"
)


# ------------------------------------------------------------------ DupIndex

def test_exact_duplicate_is_detected():
    idx = DupIndex()
    idx.add(MANUAL_A)
    assert idx.seen(MANUAL_A) == (True, "exact")


def test_whitespace_and_unicode_variants_match_as_exact():
    """Normalisation applies to the hash key, so trivial formatting differences still match."""
    idx = DupIndex()
    idx.add(MANUAL_A)
    assert idx.seen("  " + MANUAL_A.replace(" ", "  ") + "\n")[0] is True


def test_near_duplicate_is_detected():
    idx = DupIndex()
    idx.add(MANUAL_A)
    is_dup, kind = idx.seen(MANUAL_A_REWORDED)
    assert is_dup is True
    assert kind == "near"          # not exact: the text genuinely differs


def test_unrelated_document_is_not_flagged():
    idx = DupIndex()
    idx.add(MANUAL_A)
    assert idx.seen(UNRELATED) == (False, None)


def test_empty_index_flags_nothing():
    assert DupIndex().seen(MANUAL_A) == (False, None)


def test_hash_key_is_stable_and_normalises():
    assert hash_key(MANUAL_A) == hash_key("  " + MANUAL_A + "  ")
    assert hash_key(MANUAL_A) != hash_key(UNRELATED)


# ------------------------------------------------------------------ fetch()

def _load_script(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_downloaded", Path(__file__).resolve().parent.parent / "scripts" / "fetch_downloaded.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "check_auth", lambda: None)
    monkeypatch.setattr(mod, "MIN_CHARS", 50)
    return mod


def _write_manual(tmp_path, language, texts):
    d = tmp_path / language / "data" / "raw" / "crawled"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "shard_00000.jsonl.gz", "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def _fake_stream(mod, monkeypatch, docs):
    cfg = {"repo_id": "ai4bharat/IndicCorpV2", "config": "indiccorp_v2",
           "split": "hin_Deva", "text_field": "text", "name": "indiccorp_v2"}
    monkeypatch.setattr(mod, "open_stream", lambda lang: (({"text": d} for d in docs), cfg))


def _read_written(tmp_path, language):
    d = tmp_path / language / "data" / "raw" / "downloaded"
    out = []
    for shard in sorted(d.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def test_duplicates_are_dropped_from_downloaded_not_manual(tmp_path, monkeypatch):
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    _fake_stream(mod, monkeypatch, [MANUAL_A, MANUAL_A_REWORDED, UNRELATED])

    s = mod.fetch("hindi", target_tokens=10_000_000)

    assert s["dropped_exact"] == 1        # verbatim copy of a manual doc
    assert s["dropped_near"] == 1         # reworded copy
    assert s["written"] == 1              # only the unrelated one survives

    written = _read_written(tmp_path, "hindi")
    assert len(written) == 1
    assert written[0]["source_type"] == "downloaded"
    # The manual shard must be untouched -- dedup never removes from that side.
    manual = tmp_path / "hindi" / "data" / "raw" / "crawled" / "shard_00000.jsonl.gz"
    with gzip.open(manual, "rt", encoding="utf-8") as f:
        assert len([l for l in f if l.strip()]) == 1


def test_text_is_stored_raw_without_cleaning(tmp_path, monkeypatch):
    """No preprocessing on the downloaded side: the stored text is byte-identical to the source."""
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    messy = UNRELATED + "   \n\n\n  विज्ञापन  "     # would be altered by normalize()
    _fake_stream(mod, monkeypatch, [messy])

    mod.fetch("hindi", target_tokens=10_000_000)

    assert _read_written(tmp_path, "hindi")[0]["text"] == messy.strip()


def test_target_counts_written_tokens_not_streamed(tmp_path, monkeypatch):
    """Dedup losses must be absorbed by streaming further, not by shrinking the corpus."""
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    # 10 duplicates of a manual doc, then 3 unique ones.
    docs = [MANUAL_A] * 10 + [UNRELATED + f" क्रम {i}." for i in range(3)]
    _fake_stream(mod, monkeypatch, docs)

    s = mod.fetch("hindi", target_tokens=10_000_000)

    assert s["streamed"] == 13
    assert s["dropped_exact"] == 10
    assert s["written"] == 3
    assert s["tokens_total"] > 0


def test_target_stops_the_stream(tmp_path, monkeypatch):
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    _fake_stream(mod, monkeypatch, [UNRELATED + f" क्रम {i}." for i in range(50)])

    s = mod.fetch("hindi", target_tokens=60)      # a handful of documents' worth

    assert s["tokens_total"] >= 60
    assert s["written"] < 50                      # stopped early rather than draining the stream


def test_resume_does_not_rewrite_existing_documents(tmp_path, monkeypatch):
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    docs = [UNRELATED + f" क्रम {i}." for i in range(4)]

    _fake_stream(mod, monkeypatch, docs)
    first = mod.fetch("hindi", target_tokens=10_000_000)
    assert first["written"] == 4

    _fake_stream(mod, monkeypatch, docs)          # same stream again, as after an interruption
    second = mod.fetch("hindi", target_tokens=10_000_000)
    assert second["written"] == 0
    assert second["already_written_skipped"] == 4
    assert second["tokens_total"] == first["tokens_total"]   # cumulative, not double-counted


def test_overlap_rate_is_reported(tmp_path, monkeypatch):
    """The manual/downloaded overlap rate is real evidence for the Phase-1 report."""
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    _fake_stream(mod, monkeypatch, [MANUAL_A, UNRELATED])

    s = mod.fetch("hindi", target_tokens=10_000_000)
    assert s["overlap_rate"] == pytest.approx(0.5)


def test_short_documents_are_skipped(tmp_path, monkeypatch):
    mod = _load_script(monkeypatch, tmp_path)
    _write_manual(tmp_path, "hindi", [MANUAL_A])
    _fake_stream(mod, monkeypatch, ["छोटा।", UNRELATED])

    assert mod.fetch("hindi", target_tokens=10_000_000)["written"] == 1
