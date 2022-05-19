from dataclasses import dataclass
import mlflow
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from flwr.common import weights_to_parameters
import numpy


class LoaderMetrics(ABC):
    @abstractmethod
    def log_to_mlflow(self) -> None:
        pass

    @abstractmethod
    def to_result_dict(self) -> Dict:
        pass

class Metrics(LoaderMetrics):
    @property
    @abstractmethod
    def val_loss(self) -> float:
        pass


@dataclass
class RegLoaderMetrics(LoaderMetrics):
    prefix: str
    loss: float
    r2: float
    epoch: int

    def log_to_mlflow(self) -> None:
        mlflow.log_metric(f'{self.prefix}_loss', self.loss, self.epoch)
        mlflow.log_metric(f'{self.prefix}_r2', self.loss, self.epoch)

    def to_result_dict(self) -> Dict:
        return {f'{self.prefix}_loss': self.loss, f'{self.prefix}_r2': self.r2}


@dataclass
class RegMetrics(Metrics):
    train: RegLoaderMetrics
    val: RegLoaderMetrics
    test: RegLoaderMetrics
    epoch: int

    property
    def val_loss(self) -> float:
        return self.val.loss

    def log_to_mlflow(self) -> None:
        self.train.log_to_mlflow()
        self.val.log_to_mlflow()
        self.test.log_to_mlflow()

    def to_result_dict(self) -> Dict:
        train_dict, val_dict, test_dict = [m.to_result_dict for m in [self.train, self.val, self.test]]
        return train_dict | val_dict | test_dict


@dataclass
class LassoNetRegMetrics(Metrics):
    train: List[RegLoaderMetrics]
    val: List[RegLoaderMetrics]
    test: List[RegLoaderMetrics] = None
    epoch: int = 0

    @property
    def val_loss(self) -> float:
        return sum([m.loss for m in self.calculate_mean_metrics(self.val)])/len(self.val)

    def calculate_mean_metrics(self, metric_list: Optional[List[RegLoaderMetrics]]) -> RegLoaderMetrics:
        if metric_list is None:
            return None
        mean_loss = sum([m.loss for m in metric_list])/len(metric_list)
        mean_r2 = sum([m.r2 for m in metric_list])/len(metric_list)
        return RegLoaderMetrics(metric_list[0].prefix, mean_loss, mean_r2, metric_list[0].epoch)

    def log_to_mlflow(self) -> None:
        train, val, test = map(self.calculate_mean_metrics, [self.train, self.val, self.test])
        train.log_to_mlflow()
        val.log_to_mlflow()
        if test is not None:
            test.log_to_mlflow()

    def to_result_dict(self) -> Dict:
        train, test = map(self.calculate_mean_metrics, [self.train, self.val, self.test])
        train_dict, test_dict = [m.to_result_dict() if m is not None else {} for m in [train, test]]
        val_losses = weights_to_parameters(numpy.array([vm.loss for vm in self.val], dtype=numpy.float32))
        val_r2s = weights_to_parameters(numpy.array([vm.r2 for vm in self.val]))
        val_dict = {f'{self.val[0].prefix}_loss': val_losses, f'{self.val[0].prefix}_r2': val_r2s}
        return train_dict | val_dict | test_dict
    
    def __str__(self) -> str:
        train, val, test = map(self.calculate_mean_metrics, [self.train, self.val, self.test])
        train_val_str = f'train_loss: {train.loss:.4f}\ttrain_r2: {train.r2:.4f}\tval_loss: {val.loss:.4f}\tval_r2: {val.r2:.4f}'
        if test is not None:
            return train_val_str + f'\ttest_loss: {test.loss:.4f}\ttest_r2: {test.r2:.4f}'
        else:
            return train_val_str
        