import numpy as np
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
