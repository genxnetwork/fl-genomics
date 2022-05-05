import numpy
from typing import Tuple


class Int8Dataset:
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return self.X[idx, :].astype(numpy.float32), self.y[idx]
    
    def feature_count(self) -> int:
        return self.X.shape[1]