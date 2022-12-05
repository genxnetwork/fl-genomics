import torch

import numpy as np


def get_accuracy(y_true: np.array, y_pred: torch.tensor) -> float:
    """ Takes predictions, gets most probable class and compares with y to get accuracy """
    return int(torch.sum(torch.max(torch.tensor(y_pred), 1).indices == y_true)) / len(y_true)
