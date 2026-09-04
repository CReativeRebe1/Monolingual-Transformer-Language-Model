"""
Training logs that are append-only by construction.

The requirement driving this file: a Kaggle session that dies at hour 11 and is restarted must ADD
to the training history, not erase it. A logger that opens its file with mode `"w"` -- the default
almost everywhere -- destroys the entire record of a 3-hour run at the moment the resumed run opens
it, and the loss curve for the report is gone. Every file handle here is opened `"a"`, every time,
and there is no code path that truncates.

Append-only alone is not enough, because a file containing three concatenated sessions is ambiguous:
step 400 might appear three times with three different losses. So every line carries a `run_id` (the
session's start timestamp) and a monotonically increasing `wall_s`. Reading the log back for the
report means grouping by `run_id` and taking the last occurrence of each step -- which
`read_log()` does, so the plotting code never has to think about it.

Two files are written side by side:

* `train_log.jsonl` -- one JSON object per logged step. The machine-readable record; this is what
  the loss curves are plotted from.
* `train.log` -- the same information as aligned text. This is what gets read at 2am when the run
  is misbehaving, and it exists because scrolling JSONL in a Kaggle output pane is miserable.

Only rank 0 writes. The other ranks get a logger whose methods are no-ops, rather than a
`if is_main()` at every call site -- the version with the branches is the version where one call
site eventually forgets.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class TrainLogger:
    def __init__(self, log_dir: Path, run_id: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.start = time.time()
        self.log_dir = Path(log_dir)
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train_log.jsonl"
        self.text_path = self.log_dir / "train.log"

    # ------------------------------------------------------------------ writing

    def _append(self, path: Path, line: str) -> None:
        # Opened and closed per line rather than held open for the run. A held handle loses its
        # buffered tail when the process is killed -- which is the exact scenario this module is
        # built around -- and at one write per 10 training steps the syscall cost is irrelevant.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(self, kind: str, **fields) -> None:
        """A non-step record: run start, resume, checkpoint written, run end."""
        if not self.enabled:
            return
        record = {
            "run_id": self.run_id, "kind": kind,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(time.time() - self.start, 1), **fields,
        }
        self._append(self.jsonl_path, json.dumps(record, ensure_ascii=False))
        detail = "  ".join(f"{k}={v}" for k, v in fields.items())
        self._append(self.text_path, f"[{record['time']}] {kind.upper():<12} {detail}")

    def step(self, step: int, **metrics) -> None:
        """One training step's metrics. `step` is explicit so a resumed run's numbering is continuous."""
        if not self.enabled:
            return
        record = {
            "run_id": self.run_id, "kind": "step", "step": step,
            "wall_s": round(time.time() - self.start, 1), **metrics,
        }
        self._append(self.jsonl_path, json.dumps(record, ensure_ascii=False))
        parts = [f"step {step:>7,}"]
        for k, v in metrics.items():
            parts.append(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}")
        self._append(self.text_path, "  ".join(parts))

    def console(self, message: str) -> None:
        """Prints AND logs. Kaggle's output pane is truncated on long runs; the file is not."""
        if not self.enabled:
            return
        print(message, flush=True)
        self._append(self.text_path, message)


def read_log(jsonl_path: Path) -> list[dict]:
    """Reads a possibly multi-session log back, resolving overlaps.

    Two different kinds of duplicate step number occur, and they need opposite treatment:

    * **Same run, same step.** The loop writes train metrics every `log_every` steps and validation
      metrics every `eval_every` steps, so on an eval step the same step number is written twice --
      once with `train_loss`, once with `val_loss`. These must MERGE. Replacing instead silently
      deleted a training point every 250 steps, which a test caught here before any real run.
    * **Different runs, same step.** A run killed at step 3,300 whose last checkpoint was 3,250
      resumes and recomputes 3,250-3,300. Both sets of numbers are real, but only the surviving
      run's are on the trajectory the final weights came from, so the later run REPLACES the
      earlier one outright rather than merging with it.

    Truncated final lines are skipped rather than raising: a process killed mid-write leaves a
    partial JSON object, and refusing to read the entire history because of its last 40 bytes would
    be the wrong trade.
    """
    by_step: dict[int, dict] = {}
    events: list[dict] = []
    path = Path(jsonl_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") != "step":
                events.append(record)
                continue
            existing = by_step.get(record["step"])
            if existing is not None and existing.get("run_id") == record.get("run_id"):
                existing.update(record)
            else:
                by_step[record["step"]] = record
    return sorted(by_step.values(), key=lambda r: r["step"]) + events


def loss_curves(records: list[dict]) -> tuple[list[int], list[float], list[int], list[float]]:
    """(steps, train_loss, val_steps, val_loss) -- the exact argument shape
    `lma.viz.plots.plot_training_loss_curve` expects, so the report never reshapes log data by hand."""
    steps, train, val_steps, val = [], [], [], []
    for r in records:
        if r.get("kind") != "step":
            continue
        if "train_loss" in r:
            steps.append(r["step"])
            train.append(r["train_loss"])
        if r.get("val_loss") is not None:
            val_steps.append(r["step"])
            val.append(r["val_loss"])
    return steps, train, val_steps, val
