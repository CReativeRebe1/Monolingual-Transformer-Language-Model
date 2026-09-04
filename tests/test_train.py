"""Tests for lma/train/ -- data sampling, LR schedule, atomic checkpoints, append-only logging.

The two properties this suite exists to protect are both about a Kaggle session dying without
warning, because that is the failure mode the whole design is shaped around:

* **A killed write must not destroy the previous checkpoint.** `test_an_interrupted_write_leaves_the
  _previous_checkpoint_intact` simulates a crash mid-`torch.save` and asserts the old file still
  loads. Without the temp-file-plus-`os.replace` dance, a kill during the ~2 s it takes to stream
  286 MB leaves a truncated file and nothing to resume from.
* **A resumed run must APPEND to the log, not truncate it.** A logger opened with mode `"w"` -- the
  default nearly everywhere -- erases three hours of loss curve the instant the resumed session
  starts, and the loss for the report is gone with it.

The sampler tests protect a subtler one: ranks must draw disjoint windows (or the two GPUs train on
the same data and the effective batch is half what the report claims) AND statistically identical
ones (the splits are written manual-first, downloaded-second, so a contiguous split by rank would
hand rank 0 the crawl and rank 1 the downloads).
"""
import json

import numpy as np
import pytest
import torch

from lma.train import checkpoint as ckpt
from lma.train.data import TokenBatcher, WindowSampler, iter_eval_windows
from lma.train.logger import TrainLogger, loss_curves, read_log
from lma.train.schedule import lr_at, set_lr
from lma.model.transformer import GPT, GPTConfig

TINY = GPTConfig(vocab_size=53, d_model=32, n_layers=2, n_heads=4, mlp_hidden=88, max_seq_len=32)


@pytest.fixture
def encoded(tmp_path):
    """A synthetic encoded split in the exact format encode_corpus.py writes: uint16 .bin plus
    uint64 cumulative .idx. Token id == position, so a window's contents identify where it came
    from -- which is what makes the disjointness assertions readable."""
    n = 4096
    (tmp_path / "train.bin").write_bytes(np.arange(n, dtype=np.uint16).tobytes())
    np.array([0, n // 2, n], dtype=np.uint64).tofile(tmp_path / "train.idx")
    (tmp_path / "val.bin").write_bytes(np.arange(512, dtype=np.uint16).tobytes())
    np.array([0, 512], dtype=np.uint64).tofile(tmp_path / "val.idx")
    return tmp_path


# ------------------------------------------------------------------ sampling

def test_x_and_y_are_offset_by_exactly_one_token(encoded):
    b = TokenBatcher(encoded / "train.bin", encoded / "train.idx", block_size=16, batch_size=4)
    x, y = b.batch()
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted left by one"


def test_ranks_draw_disjoint_windows(encoded):
    """Two GPUs training on the same windows would halve the effective batch size while the report
    claimed otherwise, and nothing in the loss curve would look wrong."""
    seen = []
    for r in range(2):
        s = WindowSampler(4096, block_size=16, rank=r, world_size=2, seed=7)
        seen.append({s.window_start(p) for p in range(s.per_rank)})
    assert not (seen[0] & seen[1])
    assert len(seen[0]) == len(seen[1]) == 4095 // 16 // 2


def test_ranks_together_cover_the_corpus_without_repetition(encoded):
    s0 = WindowSampler(4096, 16, rank=0, world_size=2, seed=7)
    s1 = WindowSampler(4096, 16, rank=1, world_size=2, seed=7)
    starts = [s0.window_start(p) for p in range(s0.per_rank)]
    starts += [s1.window_start(p) for p in range(s1.per_rank)]
    assert len(starts) == len(set(starts)), "an epoch must not repeat a window"


def test_rank_shards_are_interleaved_not_contiguous():
    """The splits are ordered manual-then-downloaded (see lma/tokenizer/sample.py, where the same
    ordering forced stride sampling). A contiguous split by rank would train rank 0 almost entirely
    on manual crawl and rank 1 on downloaded text -- so each rank's slice must be spread across the
    whole stream, not confined to one half of it."""
    s = WindowSampler(4096, 16, rank=0, world_size=2, seed=7)
    starts = [s.window_start(p) for p in range(s.per_rank)]
    first_half = sum(1 for v in starts if v < 2048)
    assert 0.35 < first_half / len(starts) < 0.65, (
        f"rank 0 drew {first_half}/{len(starts)} windows from the first half of the stream"
    )


def test_sampling_is_reproducible_from_a_seed(encoded):
    a = WindowSampler(4096, 16, seed=99)
    b = WindowSampler(4096, 16, seed=99)
    c = WindowSampler(4096, 16, seed=100)
    assert [a.window_start(p) for p in range(20)] == [b.window_start(p) for p in range(20)]
    assert [a.window_start(p) for p in range(20)] != [c.window_start(p) for p in range(20)]


def test_a_second_epoch_uses_a_different_order():
    """Reusing the first epoch's permutation would make epoch 2 an exact replay of epoch 1."""
    s = WindowSampler(1024, 16, seed=5)
    first = [s.window_start(p) for p in range(s.per_rank)]
    second = [s.window_start(p) for p in range(s.per_rank, 2 * s.per_rank)]
    assert sorted(first) == sorted(second)      # same windows...
    assert first != second                      # ...different order


def test_resume_from_a_position_continues_the_same_stream(encoded):
    """The property that makes checkpoint resume exact: window k is window k regardless of whether
    the process drew windows 0..k-1 in this run or a previous one."""
    a = TokenBatcher(encoded / "train.bin", encoded / "train.idx", block_size=16, batch_size=2)
    for _ in range(3):
        a.batch()
    expected = a.batch()
    b = TokenBatcher(encoded / "train.bin", encoded / "train.idx", block_size=16, batch_size=2)
    b.position = 6
    assert torch.equal(b.batch()[0], expected[0])


def test_the_token_array_is_memory_mapped_not_loaded(encoded):
    """A 1.1 GB .bin loaded into RAM would not fit alongside the model on the 7.4 GB machine this
    was developed on -- the same constraint that shaped every Phase-1 stage."""
    b = TokenBatcher(encoded / "train.bin", encoded / "train.idx", block_size=16, batch_size=2)
    assert isinstance(b.tokens, np.memmap)


def test_sampler_rejects_a_corpus_too_small_for_the_world_size():
    with pytest.raises(ValueError, match="fewer than world_size"):
        WindowSampler(100, block_size=64, rank=0, world_size=4)


def test_eval_windows_are_sequential_deterministic_and_complete(encoded):
    """Evaluation must return the same number twice for the same checkpoint, or 'val loss improved'
    is partly a coin flip and ckpt_best.pt records noise."""
    a = [x for x, _ in iter_eval_windows(encoded / "val.bin", encoded / "val.idx", 16, 4)]
    b = [x for x, _ in iter_eval_windows(encoded / "val.bin", encoded / "val.idx", 16, 4)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert int(a[0][0, 0]) == 0 and int(a[0][1, 0]) == 16      # in order, non-overlapping


# ------------------------------------------------------------------ LR schedule

def test_warmup_ramps_and_never_starts_at_zero():
    """A first step taken at lr=0 is a step that does nothing -- the off-by-one that wastes it is
    using `step` rather than `step + 1` in the warmup numerator."""
    assert lr_at(0, 1e-3, warmup_steps=10, total_steps=100) == pytest.approx(1e-4)
    assert lr_at(9, 1e-3, warmup_steps=10, total_steps=100) == pytest.approx(1e-3)


def test_cosine_decays_to_the_floor_and_stays_there():
    peak, floor = 1e-3, 0.1
    assert lr_at(100, peak, 10, 100, floor) == pytest.approx(peak * floor)
    assert lr_at(500, peak, 10, 100, floor) == pytest.approx(peak * floor)   # past the end
    mid = lr_at(55, peak, 10, 100, floor)
    assert peak * floor < mid < peak


def test_schedule_is_a_pure_function_of_the_step():
    """The reason this is a function and not a scheduler object: resume cannot drift it."""
    assert lr_at(37, 6e-4, 20, 400) == lr_at(37, 6e-4, 20, 400)


def test_set_lr_updates_every_param_group():
    m = GPT(TINY)
    opt = torch.optim.AdamW(m.param_groups(0.1), lr=1.0)
    set_lr(opt, 0.5)
    assert all(g["lr"] == 0.5 for g in opt.param_groups)


# ------------------------------------------------------------------ checkpointing

def _make():
    m = GPT(TINY)
    opt = torch.optim.AdamW(m.param_groups(0.1), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return m, opt, scaler


def test_save_load_round_trip_restores_model_and_optimizer(tmp_path):
    """Optimizer state is two thirds of the file and dropping it restarts AdamW cold, which shows up
    as a visible loss spike exactly at the resume point."""
    m, opt, scaler = _make()
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    _, loss, _ = m(idx, targets=idx)
    loss.backward()
    opt.step()

    state = ckpt.TrainState(step=17, tokens_seen=1234, sampler_position=9, best_val_loss=2.5)
    ckpt.save(tmp_path / "ckpt_last.pt", m, opt, scaler, state, TINY)

    m2, opt2, scaler2 = _make()
    loaded, config, _ = ckpt.load(tmp_path / "ckpt_last.pt", m2, opt2, scaler2)

    assert loaded.step == 17 and loaded.tokens_seen == 1234 and loaded.sampler_position == 9
    assert config.vocab_size == TINY.vocab_size
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    exp_avg_a = opt.state_dict()["state"][0]["exp_avg"]
    exp_avg_b = opt2.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(exp_avg_a, exp_avg_b), "AdamW moments were not restored"


def test_an_interrupted_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """The scenario the atomic write exists for: Kaggle kills the session while torch.save is
    streaming 286 MB. A direct write leaves a truncated file AND has already destroyed the good one."""
    m, opt, scaler = _make()
    path = tmp_path / "ckpt_last.pt"
    ckpt.save(path, m, opt, scaler, ckpt.TrainState(step=100), TINY)

    real_save = torch.save

    def die(*a, **k):
        real_save(*a, **k)                      # write the temp file, then "get killed"
        raise KeyboardInterrupt("session terminated")

    monkeypatch.setattr(torch, "save", die)
    with pytest.raises(KeyboardInterrupt):
        ckpt.save(path, m, opt, scaler, ckpt.TrainState(step=200), TINY)
    monkeypatch.undo()

    survived, _, _ = ckpt.load(path, restore_rng=False)
    assert survived.step == 100, "the interrupted write clobbered the previous checkpoint"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to reproduce")
def test_resume_works_when_the_checkpoint_is_mapped_to_cuda(tmp_path):
    """Regression: the first real resume raised `TypeError: RNG state must be a torch.ByteTensor`.

    `torch.load(..., map_location=cuda)` faithfully moves EVERY tensor in the payload to the GPU,
    RNG states included, and both `torch.set_rng_state` and `torch.cuda.set_rng_state_all` require
    them on the CPU. Found by the local smoke run; on Kaggle it would have surfaced only after a
    session died and was restarted, i.e. after hours of training were already done.
    """
    m, opt, scaler = _make()
    ckpt.save(tmp_path / "c.pt", m, opt, scaler, ckpt.TrainState(step=5), TINY)
    state, _, _ = ckpt.load(tmp_path / "c.pt", m, opt, scaler, map_location="cuda")
    assert state.step == 5


def test_rng_state_is_restored_so_sampling_continues_rather_than_restarting(tmp_path):
    m, opt, scaler = _make()
    torch.manual_seed(3)
    ckpt.save(tmp_path / "c.pt", m, opt, scaler, ckpt.TrainState(), TINY)
    expected = torch.randn(4)
    ckpt.load(tmp_path / "c.pt", restore_rng=True)
    assert torch.allclose(torch.randn(4), expected)


def test_latest_returns_none_for_a_corrupt_checkpoint(tmp_path):
    """Discovering an unreadable checkpoint here is much cheaper than discovering it after the loop
    has already printed 'resuming from step 3,250'."""
    (tmp_path / "ckpt_last.pt").write_bytes(b"not a torch file")
    assert ckpt.latest(tmp_path) is None
    assert ckpt.latest(tmp_path / "nonexistent") is None


def test_load_model_rebuilds_from_the_checkpoints_own_config(tmp_path):
    """Evaluation must not need to know which language's config module produced a file."""
    m, opt, scaler = _make()
    ckpt.save(tmp_path / "c.pt", m, opt, scaler, ckpt.TrainState(step=3), TINY)
    rebuilt, state, _ = ckpt.load_model(tmp_path / "c.pt")
    assert state.step == 3
    assert rebuilt.config.vocab_size == TINY.vocab_size
    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(idx)[0], rebuilt(idx)[0], atol=1e-6)


def test_no_temp_file_is_left_behind(tmp_path):
    m, opt, scaler = _make()
    ckpt.save(tmp_path / "ckpt_last.pt", m, opt, scaler, ckpt.TrainState(), TINY)
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ append-only logging

def test_a_second_session_appends_rather_than_truncating(tmp_path):
    """The requirement: a Kaggle session that dies at hour 11 and restarts must ADD to the history.
    A logger opened `"w"` erases the loss curve for the report at the moment it opens the file."""
    first = TrainLogger(tmp_path, run_id="run-a")
    for s in (10, 20, 30):
        first.step(s, train_loss=1.0)

    second = TrainLogger(tmp_path, run_id="run-b")
    second.step(40, train_loss=0.9)

    lines = (tmp_path / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert {json.loads(l)["run_id"] for l in lines} == {"run-a", "run-b"}


def test_every_record_carries_a_run_id_so_sessions_are_distinguishable(tmp_path):
    log = TrainLogger(tmp_path, run_id="abc")
    log.step(1, train_loss=2.0)
    log.event("checkpoint_last", step=1)
    for line in (tmp_path / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines():
        assert json.loads(line)["run_id"] == "abc"


def test_train_and_val_metrics_at_the_same_step_merge(tmp_path):
    """The loop writes train metrics every log_every steps and val metrics every eval_every steps,
    so on an eval step the same step number is written twice. Keying by step and replacing -- the
    obvious implementation -- silently deleted a training point every 250 steps."""
    log = TrainLogger(tmp_path, run_id="a")
    log.step(250, train_loss=3.0, lr=6e-4)
    log.step(250, val_loss=2.8)
    merged = [r for r in read_log(tmp_path / "train_log.jsonl") if r["kind"] == "step"]
    assert len(merged) == 1
    assert merged[0]["train_loss"] == 3.0 and merged[0]["val_loss"] == 2.8


def test_read_log_lets_the_later_session_win_on_a_replayed_step(tmp_path):
    """A run killed at step 3,300 whose last checkpoint was 3,250 resumes and re-logs 3,250-3,300.
    Both are real, but only the surviving run's numbers are on the trajectory the weights came from."""
    a = TrainLogger(tmp_path, run_id="a")
    a.step(250, train_loss=5.0)
    a.step(300, train_loss=4.9)
    b = TrainLogger(tmp_path, run_id="b")
    b.step(300, train_loss=4.5)

    steps = {r["step"]: r for r in read_log(tmp_path / "train_log.jsonl") if r["kind"] == "step"}
    assert steps[300]["train_loss"] == 4.5 and steps[300]["run_id"] == "b"
    assert steps[250]["train_loss"] == 5.0


def test_a_truncated_final_line_does_not_lose_the_whole_history(tmp_path):
    """A process killed mid-write leaves a partial JSON object. Refusing to read three hours of log
    because of its last 40 bytes would be the wrong trade."""
    log = TrainLogger(tmp_path, run_id="a")
    log.step(10, train_loss=1.0)
    with open(tmp_path / "train_log.jsonl", "a", encoding="utf-8") as f:
        f.write('{"run_id": "a", "kind": "step", "st')
    records = read_log(tmp_path / "train_log.jsonl")
    assert len(records) == 1 and records[0]["step"] == 10


def test_non_main_ranks_write_nothing(tmp_path):
    """N ranks logging would produce N interleaved copies of every line."""
    TrainLogger(tmp_path, run_id="worker", enabled=False).step(1, train_loss=1.0)
    assert not (tmp_path / "train_log.jsonl").exists()


def test_loss_curves_shape_matches_the_plotting_function(tmp_path):
    """lma.viz.plots.plot_training_loss_curve takes (steps, train, val_steps, val); producing that
    shape here means the report never reshapes log data by hand."""
    log = TrainLogger(tmp_path, run_id="a")
    log.step(10, train_loss=3.0)
    log.step(20, train_loss=2.5)
    log.step(20, val_loss=2.7)
    steps, train, val_steps, val = loss_curves(read_log(tmp_path / "train_log.jsonl"))
    assert steps == [10, 20] and train == [3.0, 2.5]
    assert val_steps == [20] and val == [2.7]


def test_read_log_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert read_log(tmp_path / "nothing.jsonl") == []


# ------------------------------------------------------------------ per-language config separation

def test_the_two_training_configs_are_separate_objects():
    from hindi.train.config import CONFIG as HI
    from nepali.train.config import CONFIG as NE

    assert HI is not NE


@pytest.mark.parametrize("lang,expected_tokens", [("hindi", 515_972_223), ("nepali", 571_227_702)])
def test_each_language_config_matches_its_own_encoding_manifest(lang, expected_tokens):
    """The step count is derived from the corpus size, so a stale token count silently trains one
    language for the wrong number of steps."""
    import importlib
    import json as _json

    mod = importlib.import_module(f"{lang}.train.config")
    assert mod.TRAIN_TOKENS == expected_tokens
    manifest = mod.REPO_ROOT / lang / "data" / "encoded" / "encoding.json"
    if manifest.exists():
        assert _json.loads(manifest.read_text())["splits"]["train"]["tokens"] == mod.TRAIN_TOKENS


def test_tokens_per_optimizer_step_is_what_the_report_claims():
    from hindi.train.config import CONFIG

    assert CONFIG.tokens_per_optimizer_step(n_ranks=2) == 131_072


@pytest.mark.parametrize("lang", ["hindi", "nepali"])
def test_smoke_runs_are_isolated_even_when_paths_are_passed_explicitly(lang, tmp_path, monkeypatch):
    """Regression: the smoke redirect originally fired only when --out-dir was left at its default.

    The Kaggle runbook passes the SAME explicit --out-dir to the smoke cell and the real-run cell,
    so the 50-step smoke checkpoint was landing as ckpt_last.pt in the directory the real run then
    resumes from. The real run would have silently continued from a 50-step model trained at
    block_size 256 rather than starting clean at 1024 -- and the loss curve would have looked
    entirely normal. Found by running the packaged archive the way Kaggle runs it.
    """
    import importlib
    import sys

    mod = importlib.import_module(f"{lang}.train.pretrain")
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"model": "x", "steps": 0, "tokens_seen": 0, "best_val_loss": 0.0,
                "params": 0, "run_ids": ()}

    monkeypatch.setattr(mod, "train", fake_train)
    monkeypatch.setattr(sys, "argv", [
        "pretrain.py", "--smoke",
        "--out-dir", str(tmp_path / "checkpoints"),
        "--log-dir", str(tmp_path / "logs"),
    ])
    mod.main()

    assert captured["out_dir"].name == "smoke", captured["out_dir"]
    assert captured["log_dir"].name == "smoke", captured["log_dir"]
