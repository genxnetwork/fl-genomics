from typing import List
import numpy
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from utils.loaders import X, Y
from .memory import XyCovDataset


NArr = numpy.ndarray


class DataModule(LightningDataModule):

    def __init__(self, x: X, y: Y, x_cov: X = None, sample_weights: Y = None, batch_size: int = None):
        super().__init__()
        self.train_dataset = XyCovDataset(x.train, y.train, x_cov.train if x_cov is not None else None)
        self.val_dataset = XyCovDataset(x.val, y.val, x_cov.val if x_cov is not None else None)
        self.test_dataset = XyCovDataset(x.test, y.test, x_cov.test if x_cov is not None else None)
        self.sw = sample_weights
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.sw is not None:
            sampler = WeightedRandomSampler(self.sw.val, num_samples=self.sw.val.shape[0], replacement=True)
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.sw is not None:
            sampler = WeightedRandomSampler(self.sw.test, num_samples=self.sw.test.shape[0], replacement=True)
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return loader

    def predict_dataloader(self) -> List[DataLoader]:
        if self.sw is not None:
            sampler = WeightedRandomSampler(self.sw.train, num_samples=self.sw.train.shape[0], replacement=True)
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
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
