#!/usr/bin/env python3

"""
Some utilities to interface with PyTorch Lightning.
"""

import pytorch_lightning as pl
import sys
from tqdm import tqdm

class EpisodicBatcher(pl.LightningDataModule):

    """
    nc
    """

    def __init__(
        self,
        train_tasks,
        validation_tasks=None,
        test_tasks=None,
        epoch_length=1,
        val_epoch_length=None,
        test_epoch_length=None,
    ):
        super(EpisodicBatcher, self).__init__()
        self.train_tasks = train_tasks
        if validation_tasks is None:
            validation_tasks = train_tasks
        self.validation_tasks = validation_tasks
        if test_tasks is None:
            test_tasks = validation_tasks
        self.test_tasks = test_tasks
        self.epoch_length = epoch_length

        self.val_epoch_length = val_epoch_length if val_epoch_length is not None else epoch_length
        self.test_epoch_length = test_epoch_length if test_epoch_length is not None else epoch_length

    @staticmethod
    def epochify(taskset, epoch_length):
        class Epochifier(object):
            def __init__(self, tasks, length):
                self.tasks = tasks
                self.length = length

            def __getitem__(self, *args, **kwargs):
                return self.tasks.sample()

            def __len__(self):
                return self.length

        return Epochifier(taskset, epoch_length)

    def train_dataloader(self):
        return EpisodicBatcher.epochify(
            self.train_tasks,
            self.epoch_length,
        )

    def val_dataloader(self):
        return EpisodicBatcher.epochify(
            self.validation_tasks,
            self.val_epoch_length,
        )

    def test_dataloader(self):
        return EpisodicBatcher.epochify(
            self.test_tasks,
            self.test_epoch_length
        )


class NoLeaveProgressBar(pl.callbacks.ProgressBar):
    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(3 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(3 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            # file=sys.stdout
        )
        return bar
    def init_test_tqdm(self):
        bar = tqdm(
            desc='Testing',
            position=(3 * self.process_position+2),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            # file=sys.stdout # To avoid WandB interrupting
        )
        return bar


class TrackTestAccuracyCallback(pl.callbacks.Callback):
    def __init__(self, test_epochs=[], load_best_ckpt=False):
        self.test_epochs = test_epochs
        self.load_best_ckpt = load_best_ckpt
        super(TrackTestAccuracyCallback, self).__init__()

    def on_validation_end(self, trainer, module):
        if len(self.test_epochs) > 0:
            if trainer.current_epoch not in self.test_epochs:
                return
        trainer.test(model=module, verbose=False, ckpt_path="best")
