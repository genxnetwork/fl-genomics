import os
import numpy
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

import torch
from torch.utils.data import TensorDataset, DataLoader


class DataModule(LightningDataModule):
    def __init__(self, X_train: numpy.ndarray, X_val: numpy.ndarray, y_train: numpy.ndarray, y_val: numpy.ndarray, batch_size: int):
        super().__init__()
        self._X_train = torch.tensor(X_train, dtype=torch.float32)
        self._X_val = torch.tensor(X_val, dtype=torch.float32)
        self.train_dataset = TensorDataset(self._X_train, torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(self._X_val, torch.tensor(y_val, dtype=torch.float32))
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return loader
    
    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return loader

    def predict_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return loader

    def _dataset_len(self, dataset: TensorDataset):
        return len(dataset) // self.batch_size + int(len(dataset) % self.batch_size > 0) 

    def train_len(self):
        return self._dataset_len(self.train_dataset)
    
    def val_len(self):
        return self._dataset_len(self.val_dataset)
    

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

    trainer = Trainer(gpus=gpus, num_nodes=1,
                      max_epochs=max_epochs,
                      strategy='dp',
                      # plugins=DDPPlugin(find_unused_parameters=False),
                      callbacks=[early_stop, checkpoint_callback, lr_callback],
                      logger=logger, **kwargs)

    return trainer
