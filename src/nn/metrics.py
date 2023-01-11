from typing import Dict, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy


class DatasetMetrics(ABC):
    @abstractmethod
    def to_dict(self):
        pass
    

@dataclass
class ClfMetrics(DatasetMetrics):
    
    loss: float
    accuracy: float
    auc: float
    epoch: int
    samples: int
    
    def to_dict(self) -> Dict[str, float]:
        return {'loss': self.loss, 'accuracy': self.accuracy, 'auc': self.auc}
    
    def __str__(self) -> str:
        return f'loss={self.loss:.4f}\taccuracy={self.accuracy:.4f}\tauc={self.auc:.4f}'
    

@dataclass
class RegMetrics(DatasetMetrics):
    loss: float
    r2: float
    epoch: int
    samples: int
    
    def to_dict(self):
        return {'loss': self.loss, 'r2': self.r2}
    
    def __str__(self) -> str:
        return f'loss={self.loss:.4f}\tr2={self.r2:.4f}'


@dataclass
class ModelMetrics:

    train: DatasetMetrics
    val: DatasetMetrics
    test: DatasetMetrics
    
    @property
    def val_loss(self) -> float:
        return self.val.loss if self.val is not None else None
    
    @property
    def epoch(self) -> int:
        for dm in [self.train, self.val, self.test]:
            if dm is not None:
                return dm.epoch
        return None
    
    def _append_prefix(self, prefix: str, metric_dict: Dict[str, float]):
        return {f'{prefix}_{key}': value for key, value in metric_dict.items()}
    
    def to_dict(self) -> Dict[str, float]:
        train_dict, val_dict = self.train.to_dict(), self.val.to_dict()
        train_dict = self._append_prefix('train', train_dict)
        val_dict = self._append_prefix('val', val_dict)
        if self.test is not None:
            test_dict = self._append_prefix('test', self.test.to_dict())
            return train_dict | val_dict | test_dict
        else:
            return train_dict | val_dict
    
    def __str__(self) -> str:
        return f'train: {str(self.train)}\tval: {str(self.val)}\ttest: {str(self.test)}'
    

@dataclass
class LassoNetModelMetrics(ModelMetrics):
    train: List[DatasetMetrics]
    val: List[DatasetMetrics]
    test: List[DatasetMetrics]
    
    def append(self, metrics: ModelMetrics):
        self.train.append(metrics.train)
        self.val.append(metrics.val)
        if self.test is not None and metrics.test is not None:
            self.test.append(metrics.test)
        
            
    @property
    def best_col(self) -> int:
        return numpy.argmin([val.loss for val in self.val])
    
    @property
    def val_loss(self) -> float:
        return self.val[self.best_col].loss
    
    @property
    def epoch(self) -> int:
        for dm in [self.train, self.val, self.test]:
            if dm is not None and len(dm) > 0:
                return dm[0].epoch
        return None
    
    def reduce(self):
        if self.test is None or len(self.test) == 0:
            return ModelMetrics(self.train[self.best_col], self.val[self.best_col], None)
        else:
            return ModelMetrics(self.train[self.best_col], self.val[self.best_col], self.test[self.best_col])
    
    def to_dict(self) -> Dict[str, float]:
        mm_metrics = self.reduce()
        return mm_metrics.to_dict() | {'best_col': self.best_col}
    
    def __str__(self) -> str:
        mm_metrics = self.reduce()
        return str(mm_metrics) + f'\tbest_col={self.best_col}'
    

@dataclass
class FederatedMetrics:
    clients: List[ModelMetrics]
    
    def average_metric_list(self, metric_list: List[DatasetMetrics]) -> DatasetMetrics:
        if metric_list is None or len(metric_list) == 0:
            return None
        total_samples = sum([m.samples for m in metric_list])
        dicts = [m.to_dict() for m in metric_list]
        avg_dict = {}
        for key in dicts[0].keys():
            key_metrics = [d[key] for d in dicts]
            avg_dict[key] = sum([dict_metric*metric.samples for dict_metric, metric in zip(key_metrics, metric_list)]) / total_samples
        
        return metric_list[0].__class__(**avg_dict, epoch=metric_list[0].epoch, samples=total_samples)
    
    def reduce(self) -> ModelMetrics:
        if isinstance(self.clients[0].train, List):
            reduced = LassoNetModelMetrics([], [], [])
            
            for col in range(len(self.clients[0].train)):
                train_list = [client.train[col] for client in self.clients]
                val_list = [client.val[col] for client in self.clients]
                test_list = [client.test[col] for client in self.clients if client.test is not None]

                train, val = map(self.average_metric_list, [train_list, val_list])
                test = self.average_metric_list(test_list) if test_list else None
                reduced.append(ModelMetrics(train, val, test))
            return reduced
             
        else:        
            train_list = [client.train for client in self.clients]
            val_list = [client.val for client in self.clients]
            test_list = [client.test for client in self.clients if client.test is not None]
            
            train, val, test = map(self.average_metric_list, [train_list, val_list, test_list])
        return ModelMetrics(train, val, test)
    
