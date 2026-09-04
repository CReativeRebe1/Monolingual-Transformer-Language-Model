"""Shared pretraining machinery: data sampling, LR schedule, DDP, atomic checkpoints, append-only
logging, and the training loop itself. No language content -- see `<lang>/train/` for the entry
points that supply it."""
from lma.train.checkpoint import TrainState, load, load_model, save
from lma.train.data import TokenBatcher, WindowSampler, iter_eval_windows
from lma.train.logger import TrainLogger, loss_curves, read_log
from lma.train.loop import TrainConfig, evaluate, train
from lma.train.schedule import lr_at, set_lr

__all__ = [
    "TrainState", "load", "load_model", "save",
    "TokenBatcher", "WindowSampler", "iter_eval_windows",
    "TrainLogger", "read_log", "loss_curves",
    "TrainConfig", "train", "evaluate", "lr_at", "set_lr",
]
