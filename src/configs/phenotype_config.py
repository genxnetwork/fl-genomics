from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
import numpy as np

MEAN_PHENO_DICT = {'standing_height': 170.0}

PHENO_TYPE_DICT = {'standing height': 'continuous',
                   'ancestry': 'discrete',
                   'asthma': 'binary'}

TYPE_LOSS_DICT = {'continuous': MSELoss,
                  'discrete': CrossEntropyLoss,
                  'binary': BCELoss}

PHENO_NUMPY_DICT = {'standing_height': np.float32,
                    'ancestry': np.ndarray}
