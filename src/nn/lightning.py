from typing import List
import numpy
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader

from .memory import XyCovDataset


NArr = numpy.ndarray


class DataModule(LightningDataModule):
    def __init__(self, X_train: NArr, X_val: NArr, X_test: NArr, 
                 y_train: NArr, y_val: NArr, y_test: NArr, batch_size: int,
                 X_cov_train: NArr = None, X_cov_val: NArr = None, X_cov_test: NArr = None):
        super().__init__()
        self.train_dataset = XyCovDataset(X_train, y_train, X_cov_train)
        self.val_dataset = XyCovDataset(X_val, y_val, X_cov_val)
        self.test_dataset = XyCovDataset(X_test, y_test, X_cov_test)
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return loader
    
    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return loader
    
    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return loader

    def predict_dataloader(self) -> List[DataLoader]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        val_loader = self.val_dataloader()
        test_loader = self.test_dataloader()
        return [train_loader, val_loader, test_loader]

    def _dataset_len(self, dataset: TensorDataset):
        return len(dataset) // self.batch_size + int(len(dataset) % self.batch_size > 0) 

    def train_len(self):
        return len(self.train_dataset)
    
    def val_len(self):
        return len(self.val_dataset)
    
    def test_len(self):
        return len(self.test_dataset)

    def feature_count(self):
        return self.train_dataset.feature_count()

    def covariate_count(self):
         return self.train_dataset.covariate_count()