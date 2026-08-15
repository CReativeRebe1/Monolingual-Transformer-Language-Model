"""Tests for scripts/upload_to_drive.py -- the parts testable without a real rclone install or
real Google Drive credentials (this machine has neither, by design: uploads run on whichever
machine holds the real corpus/checkpoints and its owner's own Drive account, see the module
docstring). Covers path resolution, error messages, and the shareable-link instruction text.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "upload_to_drive.py"


def _load_module(monkeypatch, repo_root: Path):
    spec = importlib.util.spec_from_file_location("upload_to_drive", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    return module


def test_upload_raises_for_unknown_what(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        module.upload("hindi", "nonexistent_category")


def test_upload_raises_when_local_path_missing(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        module.upload("hindi", "raw")


def test_check_rclone_available_raises_clear_error_when_absent(tmp_path, monkeypatch):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="rclone is not installed"):
        module._check_rclone()


def test_get_shareable_link_instructions_mentions_readme():
    spec = importlib.util.spec_from_file_location("upload_to_drive", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instructions = module.share_link_help("gdrive:CL3410_LMA_Project/hindi/raw")
    assert "README.md" in instructions
    assert "Anyone with the link" in instructions
    assert "CL3410_LMA_Project/hindi/raw" in instructions


def test_what_to_path_covers_all_expected_categories():
    spec = importlib.util.spec_from_file_location("upload_to_drive", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.WHAT_TO_PATH.keys()) == {"raw", "cleaned", "tokenizer", "checkpoints"}
