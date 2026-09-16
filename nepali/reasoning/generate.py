"""
Generates Model L's (Nepali) synthetic reasoning dataset.

    python nepali/reasoning/generate.py

Writes `nepali/reasoning/data/{train,val,test_id,test_entity,test_wording,test_both}.jsonl` and the
dataset report `nepali/reasoning/results/dataset_stats.{json,md}` (sizes, template variety, leakage
checks, heuristic baselines, examples). Fully seeded: rerunning reproduces the files byte for byte.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nepali.reasoning.config import SPEC  # noqa: E402
from lma.reasoning.cli import cmd_generate  # noqa: E402

if __name__ == "__main__":
    cmd_generate(SPEC)
