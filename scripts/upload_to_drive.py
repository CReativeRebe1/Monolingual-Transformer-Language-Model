"""
Uploads large artifacts (raw/cleaned corpus shards, tokenizer sweep checkpoints, and later
Phase-2/3 model checkpoints) to Google Drive, since the assignment forbids committing large
binaries to git.

Wraps `rclone` rather than talking to the Google Drive API directly with our own OAuth flow:
rclone is the standard tool for this, handles token refresh and resumable/chunked uploads for
free, and -- critically -- lets each person authenticate with THEIR OWN Google account from
whatever machine actually performs the upload (this dev machine, or more likely the Kaggle/Colab
runtime where the real corpus lives), without embedding any credentials in this repository.

One-time setup (once per machine that will run this script):
    1. Install rclone: https://rclone.org/downloads/
    2. Run `rclone config` and create a remote named "gdrive" (type: drive), following its
       browser-based OAuth prompt to authorize against your own Google account.
    3. Verify with: rclone lsd gdrive:

This script never runs that setup itself and never stores or requests credentials -- it only
shells out to whatever `rclone` command already exists on PATH, and fails with a clear message
if that one-time setup has not been done yet.

Usage:
    python scripts/upload_to_drive.py --lang hindi --what raw
    python scripts/upload_to_drive.py --lang hindi --what cleaned
    python scripts/upload_to_drive.py --lang hindi --what tokenizer
    python scripts/upload_to_drive.py --lang hindi --what checkpoints   # Phase 2+
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_REMOTE = "gdrive"  # matches the remote name created in the one-time `rclone config` setup
DRIVE_BASE_FOLDER = "CL3410_LMA_Project"  # top-level folder in Drive this project uploads under

WHAT_TO_PATH = {
    "raw": "{lang}/data/raw",
    "cleaned": "{lang}/data/splits",
    "tokenizer": "{lang}/tokenizer",
    "checkpoints": "{lang}/train",  # Phase 2+: pretrained/finetuned .pt checkpoints live here
}


def _check_rclone() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is not installed or not on PATH. Install it "
            "(https://rclone.org/downloads/) and run `rclone config` once to create a remote "
            f"named '{DRIVE_REMOTE}' authorized against your Google account before using this "
            "script. See this script's module docstring for the full one-time setup."
        )


def _check_remote() -> None:
    result = subprocess.run(
        ["rclone", "listremotes"], capture_output=True, text=True, check=False
    )
    remotes = result.stdout.splitlines()
    if f"{DRIVE_REMOTE}:" not in remotes:
        raise RuntimeError(
            f"rclone has no remote named '{DRIVE_REMOTE}'. Run `rclone config` and create one "
            "(type: drive) before using this script."
        )


def upload(language: str, what: str, dry_run: bool = False) -> str:
    """Uploads <what>'s local directory for <language> to Drive; returns the Drive path used."""
    if what not in WHAT_TO_PATH:
        raise ValueError(f"what must be one of {list(WHAT_TO_PATH)}, got {what!r}")

    local_path = REPO_ROOT / WHAT_TO_PATH[what].format(lang=language)
    if not local_path.exists():
        raise FileNotFoundError(f"nothing to upload: {local_path} does not exist yet")

    drive_path = f"{DRIVE_REMOTE}:{DRIVE_BASE_FOLDER}/{language}/{what}"

    _check_rclone()
    _check_remote()

    cmd = [
        "rclone", "copy", str(local_path), drive_path,
        "--progress", "--transfers", "4", "--checkers", "8",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"Uploading {local_path} -> {drive_path}")
    subprocess.run(cmd, check=True)
    return drive_path


def share_link_help(drive_path: str) -> str:
    """rclone itself does not create shareable links (that needs the Drive API/UI); this
    returns the manual step, printed after every successful upload so it is never forgotten."""
    folder_name = drive_path.split(":", 1)[1]
    return (
        f"Uploaded to Drive folder '{folder_name}'. To get a shareable link: open Google Drive "
        f"in your browser, navigate to that folder, right-click -> Share -> \"Anyone with the "
        f"link\" -> Viewer, copy the link, and add it to the top-level README.md's Drive-links "
        f"table (and the corresponding phase report) -- TAs must be able to open it without "
        f"requesting access."
    )


def main():
    parser = argparse.ArgumentParser(description="Upload a language's large artifacts to Google Drive.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--what", required=True, choices=list(WHAT_TO_PATH))
    parser.add_argument("--dry-run", action="store_true", help="show what would be uploaded without uploading")
    args = parser.parse_args()

    try:
        drive_path = upload(args.lang, args.what, dry_run=args.dry_run)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        print()
        print(share_link_help(drive_path))


if __name__ == "__main__":
    main()
