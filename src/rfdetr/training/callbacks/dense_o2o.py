# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Callback driving DEIM-style Dense O2O's "close mosaic" schedule."""

from __future__ import annotations

from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class CloseMosaicCallback(Callback):
    """Propagate the current epoch to the train dataset's Dense O2O augmentor each epoch.

    The augmentor (:class:`~rfdetr.datasets.dense_o2o.DenseO2O`) disables mosaic/mixup for the final
    ``close_mosaic_epochs`` of training. It reads the epoch from a :class:`multiprocessing.Value`, so
    this callback writes the epoch in the main process and persistent DataLoader workers pick it up.

    The dataset handle is resolved lazily from ``trainer.datamodule._dataset_train`` so this callback
    is a no-op for any datamodule/dataset that does not expose a ``_dense_o2o`` augmentor.
    """

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the augmentor's epoch to ``trainer.current_epoch``.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained (unused).
        """
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "_dataset_train", None)
        dense_o2o = getattr(dataset, "_dense_o2o", None)
        if dense_o2o is None:
            return
        dense_o2o.set_epoch(trainer.current_epoch)
        if not dense_o2o.active():
            logger.info("Dense O2O: mosaic/mixup closed at epoch %d", trainer.current_epoch)
