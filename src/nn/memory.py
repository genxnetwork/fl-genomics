import numpy
from typing import Tuple


class XyCovDataset:
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, X_cov: numpy.ndarray = None) -> None:
        self.X = X
        self.y = y
        self.X_cov = X_cov

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self.X_cov is not None:
            return numpy.hstack([self.X[idx, :].astype(numpy.float32), self.X_cov[idx, :].astype(numpy.float32)]), self.y[idx]
        else:
            return self.X[idx, :].astype(numpy.float32), self.y[idx]
    
    def feature_count(self) -> int:
        return self.X.shape[1] if self.X_cov is None else self.X.shape[1] + self.X_cov.shape[1]

    def covariate_count(self) -> int:
        return self.X_cov.shape[1] if self.X_cov is not None else 0