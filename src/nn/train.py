import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_trainer(model_dir, log_dir, nn_type_name, version, gpus=1, max_epochs=50, patience=3, **kwargs):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_dir, nn_type_name, 'v' + version) + '/',
        filename='e{epoch}-vl{val_loss:.4f}',
        save_top_k=2,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        strict=False,
        verbose=False,
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=version,
        name=nn_type_name
    )

    lr_callback = LearningRateMonitor(logging_interval='step')

    if gpus > 0:
        trainer = Trainer(gpus=gpus, num_nodes=1,
                          strategy='dp',
                          max_epochs=max_epochs,
                          callbacks=[early_stop, checkpoint_callback, lr_callback],
                          logger=logger, **kwargs)
    else:
        trainer = Trainer(max_epochs=max_epochs,
                          callbacks=[early_stop, checkpoint_callback, lr_callback],
                          logger=logger, **kwargs)

    return trainer
