from typing import List
import numpy
from pytorch_lightning import LightningDataModule
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

    def predict_dataloader(self) -> List[DataLoader]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        val_loader = self.val_dataloader()
        return [train_loader, val_loader]

    def _dataset_len(self, dataset: TensorDataset):
        return len(dataset) // self.batch_size + int(len(dataset) % self.batch_size > 0) 

    def train_len(self):
        return len(self.train_dataset)
    
    def val_len(self):
        return len(self.val_dataset)