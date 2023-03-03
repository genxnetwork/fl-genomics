import logging
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from configs.phenotype_config import TYPE_LOSS_DICT
from nn.models import MLPClassifier

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
_logger = logging.getLogger()


class SimpleTrainer(object):
    def __init__(self, nclass, nfeat, epochs=10000, lr=0.1):
        self.nclass = nclass
        self.nfeat = nfeat
        self.epochs = epochs
        self.lr = lr

    @staticmethod
    def predict_np(model, x: pd.DataFrame) -> torch.tensor:
        return model(torch.from_numpy(x.astype('float32')))

    @staticmethod
    def get_accuracy(preds: torch.tensor, y: np.array) -> float:
        """ Takes predictions, gets most probable class and compares with y to get accuracy """
        return int(torch.sum(torch.max(preds, 1).indices == torch.from_numpy(y))) / len(y)

    def train(self, logger, xtrain, ytrain, xtest=None, ytest=None):
        train_only_mode = (xtest is None)
        self.model = MLPClassifier(nclass=self.nclass, nfeat=self.nfeat, optim_params=None, scheduler_params=None,
                                   loss=TYPE_LOSS_DICT['discrete'], binary=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        loss_train_list = []
        loss_test_list = []
        train_acc_list = []
        test_acc_list = []
        for i in range(self.epochs):
            self.optimizer.zero_grad()
            preds_train = self.predict_np(model=self.model, x=xtrain)
            loss_train = self.model.loss(preds_train, torch.from_numpy(ytrain))
            train_acc = self.get_accuracy(preds=preds_train, y=ytrain)
            loss_train_list.append(float(loss_train))
            train_acc_list.append(train_acc)
            if not train_only_mode:
                preds_test = self.predict_np(model=self.model, x=xtest)
                loss_test = self.model.loss(preds_test, torch.from_numpy(ytest))
                test_acc = self.get_accuracy(preds=preds_test, y=ytest)
                loss_test_list.append(float(loss_test))
                test_acc_list.append(test_acc)
            # if (i == 0) | (i == epochs - 1):
            if (i % 1000 == 0) | (i == self.epochs - 1):
                logger.info(f"epoch {i}: train cross-entropy {float(loss_train)}")
                logger.info(f"epoch {i}: train accuracy {train_acc}")
                if not train_only_mode:
                    logger.info(f"epoch {i}: test cross-entropy {float(loss_test)}")
                    logger.info(f"epoch {i}: test accuracy {test_acc}")
            #     conf_matrix_list[i] = confusion_matrix(ytest.values, torch.max(preds_test, 1).indices.numpy())
            loss_train.backward()
            self.optimizer.step()
        stats = pd.DataFrame({'epoch': np.arange(self.epochs),
                              'train_acc': train_acc_list,
                              'train_loss': loss_train_list,
                              })
        if not train_only_mode:
            stats['test_acc'] = test_acc_list
            stats['test_loss'] = loss_test_list
        return stats

    @staticmethod
    def test(logger, model, xtest, ytest):
        """Validate the network on the entire test set."""
        preds = SimpleTrainer.predict_np(model=model, x=xtest)
        test_acc = SimpleTrainer.get_accuracy(preds=preds, y=ytest)
        loss = model.loss(preds, torch.from_numpy(ytest))
        logger.info(f"Test set: cross-entropy {loss}")
        logger.info(f"Test set: accuracy {test_acc}")
        return test_acc, float(loss)

    def run_cv(self, x, y, K = 10):
        skf = StratifiedKFold(n_splits=K)
        test_acc = []
        for train_index, test_index in skf.split(x, y):
            _logger.info(f'Processing fold...')
            self.model = MLPClassifier(nclass=self.nclass, nfeat=self.nfeat, optim_params=None, scheduler_params=None,
                                       loss=TYPE_LOSS_DICT['discrete'], binary=False)
            xtrain = x[train_index]
            ytrain = y[train_index]
            xtest = x[test_index]
            ytest = y[test_index]
            self.train(logger=_logger, xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
            ta, _ = self.test(logger=_logger, model=self.model, xtest=xtest, ytest=ytest)
            test_acc.append(ta)
        _logger.info(f'Testing accuracies for all folds: {test_acc}')

    def train_and_save(self, x, y, out_fn):
        _logger.info(f'Training the model...')
        self.train(logger=_logger, xtrain=x, ytrain=y)
        _logger.info(f'Saving the model...')
        pickle.dump(self.model, open(out_fn, 'wb'))
        pass