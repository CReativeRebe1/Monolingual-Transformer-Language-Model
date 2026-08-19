"""
Encodes a cleaned, split corpus (one document per line, as written by scripts/build_corpus.py)
into a nanoGPT-style memmap format for fast Phase-2 training: a flat `uint16` token-id array
(`.bin`) plus a `uint64` document-boundary offset array (`.idx`). Memmap-friendly so a 500M-token
corpus never needs to be fully loaded into RAM during pretraining -- the trainer indexes directly
into the file on disk.

`uint16` is only valid while every trained vocabulary stays under 65,536 (2**16) -- true for the
tens-of-thousands vocab sizes this project's tokenizer sweep considers (see
scripts/tokenizer_sweep.py's {8k, 16k, 24k, 32k, 48k} range), and asserted explicitly here so a
future change to a larger vocabulary fails loudly at encode time rather than silently wrapping
token ids around.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from array import array

import numpy as np

from lma.tokenizer.train_sp import Tokenizer

_UINT16_MAX = np.iinfo(np.uint16).max


@dataclass
class EncodedCorpusInfo:
    n_documents: int
    n_tokens: int
    bin_path: Path
    idx_path: Path


def encode_split(
    tokenizer: Tokenizer, text_path: Path, out_dir: Path, split_name: str, append_eos: bool = True
) -> EncodedCorpusInfo:
    """Encodes one split file (one document per line) to <out_dir>/<split_name>.bin/.idx.

    The .idx file stores CUMULATIVE token offsets, one entry per document boundary, starting
    with 0: document i's tokens are bin_array[idx[i]:idx[i+1]]. This lets a caller recover
    document boundaries (needed for e.g. held-out generation prefixes in Phase 2 evaluation)
    without re-tokenizing or re-scanning the original text.
    """
    if tokenizer.vocab_size > _UINT16_MAX + 1:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} exceeds uint16 range "
            f"({_UINT16_MAX + 1}); this encoder assumes uint16 token ids throughout."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{split_name}.bin"
    idx_path = out_dir / f"{split_name}.idx"

    eos_id = tokenizer._sp.eos_id() if append_eos else None

    # Written incrementally rather than concatenated at the end. Holding every document's array and
    # then calling np.concatenate needs the chunks and the joined result in memory at once -- about
    # 2.3 GB for a 470M-token split, on top of ~100 bytes of object overhead per document across
    # millions of documents. Streaming keeps memory flat regardless of corpus size, which is the
    # same reason the cleaning pipeline streams.
    offsets = array("Q", [0])          # compact: 8 bytes each, not a 28-byte Python int each
    n_documents = 0
    total = 0

    with open(text_path, encoding="utf-8") as f, open(bin_path, "wb") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            if append_eos and eos_id is not None and eos_id >= 0:
                ids = ids + [eos_id]
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(out)
            total += len(arr)
            offsets.append(total)
            n_documents += 1

    np.frombuffer(offsets, dtype=np.uint64).tofile(idx_path)

    return EncodedCorpusInfo(
        n_documents=n_documents, n_tokens=total, bin_path=bin_path, idx_path=idx_path
    )


def load_encoded_split(bin_path: Path, idx_path: Path) -> tuple[np.memmap, np.ndarray]:
    """Opens an encoded split for reading: (memmapped token array, document offset array).

    The token array is memory-mapped (mode="r"), not loaded into RAM -- this is the whole point
    of the format for a corpus that may be far larger than available memory during pretraining.
    """
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    return tokens, offsets


def document_slice(tokens: np.memmap, offsets: np.ndarray, doc_index: int) -> np.ndarray:
    """Returns the token ids belonging to document `doc_index` (0-indexed)."""
    start, end = int(offsets[doc_index]), int(offsets[doc_index + 1])
    return tokens[start:end]
