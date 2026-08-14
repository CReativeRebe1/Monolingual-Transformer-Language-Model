"""Tests for scripts/verify_independence.py's document-overlap check -- the mechanical proof
behind "no shared documents between Model H and Model L corpora". Imported directly as a module
since it is a script, not a package, but its check functions are pure and worth testing in
isolation from the real repo state.
"""
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_independence.py"


def _load_module(monkeypatch, repo_root: Path):
    spec = importlib.util.spec_from_file_location("verify_independence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    return module


def _write_manifest(splits_dir: Path, name: str, hashes: list[str]):
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / f"{name}.manifest.jsonl", "w", encoding="utf-8") as f:
        for h in hashes:
            f.write(json.dumps({"content_sha256": h, "host": "example.com", "source_type": "manual"}) + "\n")


def test_no_document_overlap_passes_when_disjoint(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    _write_manifest(tmp_path / "hindi" / "data" / "splits", "train", ["hash_a", "hash_b"])
    _write_manifest(tmp_path / "nepali" / "data" / "splits", "train", ["hash_c", "hash_d"])
    ok, detail = module.check_no_document_overlap()
    assert ok is True
    assert "overlap=0" in detail


def test_no_document_overlap_fails_when_shared_hash_present(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    _write_manifest(tmp_path / "hindi" / "data" / "splits", "train", ["hash_a", "shared_hash"])
    _write_manifest(tmp_path / "nepali" / "data" / "splits", "train", ["hash_c", "shared_hash"])
    ok, detail = module.check_no_document_overlap()
    assert ok is False
    assert "overlap=1" in detail


def test_no_document_overlap_skips_gracefully_when_no_manifests(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    ok, detail = module.check_no_document_overlap()
    assert ok is True
    assert "skipped" in detail


def test_tokenizer_files_differ_skips_when_no_models(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    ok, detail = module.check_tokenizer_files_differ()
    assert ok is True
    assert "skipped" in detail


def test_tokenizer_files_differ_fails_on_identical_bytes(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    hi_dir = tmp_path / "hindi" / "tokenizer"
    ne_dir = tmp_path / "nepali" / "tokenizer"
    hi_dir.mkdir(parents=True)
    ne_dir.mkdir(parents=True)
    identical_bytes = b"not a real sentencepiece model, just identical bytes for the test"
    (hi_dir / "hi.model").write_bytes(identical_bytes)
    (ne_dir / "ne.model").write_bytes(identical_bytes)
    ok, detail = module.check_tokenizer_files_differ()
    assert ok is False
    assert "BYTE-IDENTICAL" in detail


def test_main_exits_zero_when_all_checks_pass(tmp_path, monkeypatch, capsys):
    module = _load_module(monkeypatch, tmp_path)
    exit_code = module.main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "ALL CHECKS PASSED" in captured.out
