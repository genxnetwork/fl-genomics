import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy

MEAN_PHENO_DICT = {'standing_height': 170.0}

PHENO_TYPE_DICT = {'standing height': 'continuous',
                   'ancestry': 'discrete',
                   'asthma': 'binary'}

TYPE_LOSS_DICT = {'continuous': mse_loss,
                  'discrete': cross_entropy,
                  'binary': binary_cross_entropy}

PHENO_NUMPY_DICT = {'standing_height': np.float32,
                    'ancestry': np.ndarray}


def get_accuracy(y_true: np.array, y_pred: torch.tensor) -> float:
    """ Takes predictions, gets most probable class and compares with y to get accuracy """
    return int(torch.sum(torch.max(torch.tensor(y_pred), 1).indices == torch.from_numpy(y_true))) / len(y_true)


TYPE_METRIC_DICT = {'continuous': {'metric_fun': r2_score,
                                   'metric_name': 'r2'},
                    'discrete': {'metric_fun': get_accuracy,
                                 'metric_name': 'accuracy'}
                    }
