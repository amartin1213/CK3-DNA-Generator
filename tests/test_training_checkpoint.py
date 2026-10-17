"""Tests for ck3dna.training's `experiment` Trainer/callback wiring.

Specifically guards against a real bug found in practice: without an
explicit, monitored `ModelCheckpoint` callback, PyTorch Lightning's default
checkpointing only saves the *last* epoch seen before training stops, not
the epoch with the lowest val_loss. On this project's small, volatile
datasets, val_loss can drift back up during the EarlyStopping patience
window before patience finally expires, so the "last" checkpoint saved can
measurably underperform the best one seen a few epochs earlier (observed:
overall prediction-diversity ratio collapsing from 0.443 to 0.149 on one
retrain). `experiment` must configure a `ModelCheckpoint(monitor="val_loss",
mode="min")` so the actual best checkpoint is what gets kept.
"""
import argparse

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ck3dna.training import experiment


def _make_args(**overrides):
    defaults = dict(
        datastore_name="unused_in_this_test",
        batch_size=4,
        workers=0,
        val_ratio=0.2,
        lr=3e-3,
        epochs=40,
        val_epochs=10,
        gpus=0,
        ckpt=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _find_callback(trainer, cls):
    matches = [cb for cb in trainer.callbacks if isinstance(cb, cls)]
    assert matches, f"Expected a {cls.__name__} callback, found none in {trainer.callbacks!r}"
    return matches[0]


def test_experiment_configures_early_stopping_on_val_loss():
    exp = experiment(_make_args())
    early_stop = _find_callback(exp.trainer, EarlyStopping)
    assert early_stop.monitor == "val_loss"
    assert early_stop.patience == 10


def test_experiment_configures_model_checkpoint_to_keep_best_val_loss():
    """Regression test: the saved checkpoint must be the best-val_loss
    epoch, not whichever epoch happened to be current when EarlyStopping's
    patience expired."""
    exp = experiment(_make_args())
    checkpoint_cb = _find_callback(exp.trainer, ModelCheckpoint)
    assert checkpoint_cb.monitor == "val_loss"
    assert checkpoint_cb.mode == "min"
    assert checkpoint_cb.save_top_k == 1
