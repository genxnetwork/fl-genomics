import pickle
from abc import abstractmethod
import sys
from typing import Type


sys.path.append('..')

import hydra
import logging
from sys import stdout
import numpy
import pandas as pd
from omegaconf import DictConfig
import mlflow
from mlflow.xgboost import autolog
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from numpy import argmax, amax
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegressionCV, LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

from local.config import node_size_dict, node_name_dict
from fl.datasets.memory import load_covariates
from nn.lightning import DataModule
from nn.train import prepare_trainer
from nn.utils import LassoNetRegMetrics
from nn.models import MLPPredictor, LassoNetRegressor, LassoNetClassifier, MLPClassifier, LinearRegressor, LinearClassifier
from configs.phenotype_config import MEAN_PHENO_DICT, PHENO_TYPE_DICT, PHENO_NUMPY_DICT, TYPE_LOSS_DICT, \
    TYPE_METRIC_DICT
from utils.metrics import get_accuracy
from utils.loaders import Y, ExperimentDataLoader
from utils.landscape import plot_loss_landscape


class LocalExperiment(object):
    """
    Base class for experiments in a local setting
    """
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Configuration for experiments from hydra
        """
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO,
                            stream=stdout,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger()
        self.loader = ExperimentDataLoader(cfg)

    def start_mlflow_run(self):
        split = self.cfg.split_dir.split('/')[-1]
        mlflow.set_experiment(self.cfg.experiment.name)
        universal_tags = {
            'model': self.cfg.model.name,
            'split': split,
            'phenotype': self.cfg.data.phenotype.name,
        }
        if self.cfg.study == 'tg':
            study_tags = {
                'node': self.cfg.node,
                'fold_index': self.cfg.fold_index
            }
        elif self.cfg.study == 'ukb':
            num_samples = node_size_dict[split][self.cfg.node_index]
            study_tags={
                'fold_index': str(self.cfg.fold_index),
                'node_index': str(self.cfg.node_index),
                'snp_count': str(self.cfg.experiment.snp_count),
                'sample_count': str(round(num_samples, -2)),
                'sample_count_exact': str(num_samples),
                'dataset': f"{node_name_dict[split][self.cfg.node_index]}_{round(num_samples, -2)}",
                'gwas': self.cfg.data.gwas
            }
        elif self.cfg.study == 'simulation':
            study_tags = {
                'sample_count': str(self.cfg.data.samples)
            }
        else:
            raise ValueError('Please define the study in config! See src/configs/default.yaml')
        self.run = mlflow.start_run(tags=universal_tags | study_tags)

    def load_data(self):
        self.logger.info("Loading data")
        self.x, self.y = self.loader.load()
        # self.sw = Y(None, None, None) if study is not ukb or cfg.sample_weights is False or not in cfg
        self.sw = self.loader.load_sample_weights()

        if self.cfg.study == 'ukb':
            self.x_cov = self.loader.load_covariates()
            self.logger.info(f"{self.x_cov.train.shape[1]} covariates loaded")
        self.logger.info(f"{self.x.train.shape[1]} features loaded")

    @abstractmethod
    def train(self):
        pass

    def eval_and_log(self, metric_fun=r2_score, metric_name='r2'):
        self.logger.info("Evaluating model")
        preds_train = self.model.predict(self.x.train)
        preds_val = self.model.predict(self.x.val)
        preds_test = self.model.predict(self.x.test)

        metric_train = metric_fun(self.y.train, preds_train, sample_weight=self.sw.train)
        metric_val = metric_fun(self.y.val, preds_val, sample_weight=self.sw.val)
        metric_test = metric_fun(self.y.test, preds_test, sample_weight=self.sw.test)

        print(f"Train {metric_name}: {metric_train}")
        mlflow.log_metric(f'train_{metric_name}', metric_train)
        print(f"Val {metric_name}: {metric_val}")
        mlflow.log_metric(f'val_{metric_name}', metric_val)
        print(f"Test {metric_name}: {metric_test}")
        mlflow.log_metric(f'test_{metric_name}', metric_test)

    def run(self):
        self.load_data()
        self.start_mlflow_run()
        self.train()
        self.eval_and_log(**TYPE_METRIC_DICT[PHENO_TYPE_DICT[self.cfg.data.phenotype.name]])


def simple_estimator_factory(model):
    """Returns a SimpleEstimatorExperiment for a given model class, expected
    to have the same interface as scikit-learn estimators.

    Args:
        model: Model class
        model_kwargs_dict: Dictionary of parameters passed during model initialization
    """
    class SimpleEstimatorExperiment(LocalExperiment):
        def __init__(self, cfg):
            LocalExperiment.__init__(self, cfg)
            self.model = model(**self.cfg.model.params)

        def train(self):
            self.logger.info("Training")
            self.model.fit(self.x.train, self.y.train)

    return SimpleEstimatorExperiment


class XGBExperiment(LocalExperiment):
    def __init__(self, cfg):
        LocalExperiment.__init__(self, cfg)
        self.model = XGBRegressor(**self.cfg.model.params)

    def train(self):
        self.logger.info("Training")
        autolog()
        self.model.fit(self.x.train, self.y.train, eval_set=[(self.x.val, self.y.val)],
                       early_stopping_rounds=self.cfg.model.early_stopping_rounds, verbose=True)


class RandomForestExperiment(LocalExperiment):
    def __init__(self, cfg):
        LocalExperiment.__init__(self, cfg)
        self.model = RandomForestClassifier(**self.cfg.model.params)

    def train(self):
        self.logger.info("Training")
        autolog()
        self.y.test = numpy.concatenate((self.y.val, self.y.test, self.y.train), axis=0)
        self.x.test = numpy.concatenate((self.x.val, self.x.test, self.x.train), axis=0)
        scores = cross_validate(self.model, self.x.test, self.y.test, cv=10, return_train_score=True)
        print(scores)
        #self.model.fit(self.X_train.values, self.y_train.values)

    def eval_and_log(self, metric_fun=accuracy_score, metric_name='accuracy'):
    	pass


def get_model_class(model_name: str) -> Type:
    model_class_dict = {
        'lassonet_regressor': LassoNetRegressor,
        'lassonet_classifier': LassoNetClassifier,
        'mlp_regressor': MLPPredictor,
        'mlp_classifier': MLPClassifier,
        'linear_regressor': LinearRegressor,
        'linear_classifier': LinearClassifier
    }
    return model_class_dict[model_name]


class NNExperiment(LocalExperiment):
    def __init__(self, cfg):
        LocalExperiment.__init__(self, cfg)
        self.model_class: Type = get_model_class(cfg.model.name)

    def load_data(self):
        LocalExperiment.load_data(self)
        self.data_module = DataModule(self.x,
                                      self.y.astype(PHENO_NUMPY_DICT[self.cfg.data.phenotype.name]),
                                      sample_weights=self.sw,
                                      batch_size=self.cfg.model.get('batch_size', len(self.x.train)))

    def create_model(self):
        self.model: self.model_class = self.model_class(input_size=self.x.train.shape[1],
                                                        optim_params=self.cfg.optimizer,
                                                        scheduler_params=self.cfg.scheduler,
                                                        loss=TYPE_LOSS_DICT[PHENO_TYPE_DICT[self.cfg.data.phenotype.name]],
                                                        **self.cfg.model.params
                                                        )

    def load_best_model(self):
        self.model = self.model_class.load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path,
                input_size=self.x.train.shape[1],
                optim_params=self.cfg.optimizer,
                scheduler_params=self.cfg.scheduler,
                loss=TYPE_LOSS_DICT[PHENO_TYPE_DICT[self.cfg.data.phenotype.name]],
                **self.model.params
            )

    def train(self):
        mlflow.log_params({'model': self.cfg.model})
        mlflow.log_params({'optimizer': self.cfg.optimizer})
        mlflow.log_params({'scheduler': self.cfg.get('scheduler', None)})
        
        # object has no attribute 'sum'. Probably because it is supposed to be a tuple of train, val, test
        # prevalence = self.y.sum().mean()
        # mlflow.log_metric('prevalence', float(prevalence))
        
        self.create_model()

        if self.cfg.experiment.pretrain_on_cov == 'weights':
            cov_weights = self.pretrain()
            self.model.set_covariate_weights(cov_weights)
        elif self.cfg.experiment.pretrain_on_cov == 'substract':
            residual = self.pretrain_and_substract()
            self.data_module.update_y(residual)
        
        self.trainer = prepare_trainer('models', 'logs', f'{self.cfg.model.name}/{self.cfg.data.phenotype.name}', f'run{self.run.info.run_id}',
                                       gpus=self.cfg.training.get('gpus', 0),
                                       precision=self.cfg.training.get('precision', 32),
                                       max_epochs=self.cfg.training.max_epochs,
                                       weights_summary='full',
                                       patience=self.cfg.training.patience,
                                       log_every_n_steps=5,
                                       enable_progress_bar=self.cfg.training.enable_progress_bar)

        print("Fitting")
        self.trainer.fit(self.model, self.data_module)
        print("Fitted")
        self.load_best_model()
        mlflow.log_param('model_saved', self.trainer.checkpoint_callback.best_model_path)
        f = open(self.trainer.checkpoint_callback.best_model_path.replace('.ckpt', '.pkl'), 'wb')
        pickle.dump(self.model, f)
        f.close()

        print(f'Loaded best model {self.trainer.checkpoint_callback.best_model_path}')

    def eval_and_log(self, metric_fun=r2_score, metric_name='r2'):
        self.model.eval()
        train_preds, val_preds, test_preds = self.trainer.predict(self.model, self.data_module)

        train_preds = torch.cat(train_preds).squeeze().cpu().numpy()
        val_preds = torch.cat(val_preds).squeeze().cpu().numpy()
        test_preds = torch.cat(test_preds).squeeze().cpu().numpy()

        metric_train = metric_fun(self.y.train, train_preds, sample_weight=self.sw.train)
        metric_val = metric_fun(self.y.val, val_preds, sample_weight=self.sw.val)
        metric_test = metric_fun(self.y.test, test_preds, sample_weight=self.sw.test)

        print(f"Train {metric_name}: {metric_train}")
        mlflow.log_metric(f'train_{metric_name}', metric_train)
        print(f"Val {metric_name}: {metric_val}")
        mlflow.log_metric(f'val_{metric_name}', metric_val)
        print(f"Test {metric_name}: {metric_test}")
        mlflow.log_metric(f'test_{metric_name}', metric_test)

    def pretrain(self):
        """Pretrains linear regression on phenotype and covariates

        Returns:
            numpy.ndarray: Coefficients of covarites without intercept
        """
        phenotype_type = PHENO_TYPE_DICT[self.cfg.data.phenotype.name]
        if phenotype_type == 'binary':
            lr = LogisticRegression()
            lr.fit(self.x_cov.train, self.y.train)
            val_pred = lr.predict_proba(self.x_cov.val)[:, 1]
            train_pred = lr.predict_proba(self.x_cov.train)[:, 1]
            val_auc = roc_auc_score(self.y.val, val_pred)
            train_auc = roc_auc_score(self.y.train, train_pred)
            
            val_acc = lr.score(self.x_cov.val, self.y.val)
            train_acc = lr.score(self.x_cov.train, self.y.train)
            samples, cov_count = self.x_cov.train.shape
            print(f'pretraining on {samples} samples and {cov_count} covariates gives {train_acc:.4f} train accuracy and {val_acc:.4f} val accuracy')
            print(f'pretraining on {samples} samples and {cov_count} covariates gives {train_auc:.4f} train AUC and {val_auc:.4f} val AUC')
            return lr.coef_[0, :]

        elif phenotype_type == 'continuous':
            lr = LinearRegression()
            lr.fit(self.x_cov.train, self.y.train)
            val_r2 = lr.score(self.x_cov.val, self.y.val)
            train_r2 = lr.score(self.x_cov.train, self.y.train)
            samples, cov_count = self.x_cov.train.shape
            print(f'pretraining on {samples} samples and {cov_count} covariates gives {train_r2:.4f} train r2 and {val_r2:.4f} val r2')
            return lr.coef_
        
        else:
            raise ValueError(f'for phenotype type {phenotype_type} pretraining is not implemented')

    def pretrain_and_substract(self) -> Y:
        """Pretrains linear regression on phenotype and covariates
        Predicts phenotype and substracts predicted phenotype from true phenotype

        Returns:
            Y: phenotype residual unexplained by linear covariate-only model
        """
        if PHENO_TYPE_DICT[self.cfg.data.phenotype.name] != 'continuous':
            raise ValueError(f'for phenotype type {PHENO_TYPE_DICT[self.cfg.data.phenotype.name]} substracting is not implemented')
        lr = LinearRegression()
        lr.fit(self.x_cov.train, self.y.train)
        y_train = self.y.train - lr.predict(self.x_cov.train)
        y_val = self.y.val - lr.predict(self.x_cov.val)
        y_test = self.y.test - lr.predict(self.x_cov.test)

        val_r2 = lr.score(self.x_cov.val, self.y.val)
        train_r2 = lr.score(self.x_cov.train, self.y.train)
        samples, cov_count = self.x_cov.train.shape
        print(f'pretraining on {samples} samples and {cov_count} covariates gives {train_r2:.4f} train r2 and {val_r2:.4f} val r2')

        residual = Y(y_train, y_val, y_test)
        return residual.astype(PHENO_NUMPY_DICT[self.cfg.data.phenotype.name])


class TGNNExperiment(NNExperiment):
    def load_data(self):
        LocalExperiment.load_data(self)
        train_stds, val_stds, test_stds = self.x.train.std(axis=0), self.x.val.std(axis=0), self.x.test.std(axis=0)
        for part, stds in zip(['train', 'val', 'test'], [train_stds, val_stds, test_stds]):
            self.logger.info(f'{part} stds: {numpy.array2string(stds, precision=3, floatmode="fixed")}')

        if self.cfg.data.x_reduced.get('normalize_stds'):
            self.x.val = self.x.val * (train_stds / val_stds)
            self.x.test = self.x.test * (train_stds / test_stds)
            for part, matrix in zip(['train', 'val', 'test'], [self.x.train, self.x.val, self.x.test]):
                self.logger.info(f'{part} normalized stds: {numpy.array2string(matrix.std(axis=0), precision=3, floatmode="fixed")}')

        self.data_module = DataModule(
            self.x,
            self.y.astype(PHENO_NUMPY_DICT[self.cfg.data.phenotype.name]),
            batch_size=self.cfg.model.get('batch_size', len(self.x.train)),
            drop_last=False
        )

    def create_model(self):
        self.model = MLPClassifier(nclass=len(set(self.y.train)), nfeat=self.x.train.shape[1],
                                   optim_params=self.cfg.experiment.optimizer,
                                   scheduler_params=self.cfg.experiment.get('scheduler', None),
                                   loss=TYPE_LOSS_DICT[PHENO_TYPE_DICT[self.cfg.data.phenotype.name]]
                                   )


    def load_best_model(self):
        self.model = MLPClassifier.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path,
                                                        nclass=len(set(self.y.train)), nfeat=self.x.train.shape[1],
                                                        optim_params=self.cfg.experiment.optimizer,
                                                        scheduler_params=self.cfg.experiment.get('scheduler', None),
                                                        loss=TYPE_LOSS_DICT[PHENO_TYPE_DICT[self.cfg.data.phenotype.name]]
                                                        )

    def eval_and_log(self, metric_fun=get_accuracy, metric_name='accuracy'):
        self.model.eval()

        train_loader, validation_loader, test_loader = self.data_module.predict_dataloader()

        y_pred, y_true = self.model.predict(train_loader)
        metric_train = metric_fun(y_true, y_pred)

        y_pred, y_true = self.model.predict(validation_loader)
        metric_val = metric_fun(y_true, y_pred)

        y_pred, y_true = self.model.predict(test_loader)
        metric_test = metric_fun(y_true, y_pred)

        print(f"Train {metric_name}: {metric_train}")
        mlflow.log_metric(f'train_{metric_name}', metric_train)
        print(f"Val {metric_name}: {metric_val}")
        mlflow.log_metric(f'val_{metric_name}', metric_val)
        print(f"Test {metric_name}: {metric_test}")
        mlflow.log_metric(f'test_{metric_name}', metric_test)


# Dict of possible experiment types and their corresponding classes
ukb_experiment_dict = {
    'lasso': simple_estimator_factory(LassoCV),
    'logistic_regression': simple_estimator_factory(LogisticRegressionCV),
    'xgboost': XGBExperiment,
    'lassonet_regressor': NNExperiment,
    'lassonet_classifier': NNExperiment,
    'mlp_regressor': NNExperiment,
    'mlp_classifier': NNExperiment
}

tg_experiment_dict = {
    'mlp_classifier': TGNNExperiment,
    'random_forest': RandomForestExperiment
}


@hydra.main(config_path='configs', config_name='default')
def local_experiment(cfg: DictConfig):
    print(cfg)
    assert cfg.study in ['tg', 'ukb']
    if cfg.study == 'ukb':
        assert cfg.model.name in ukb_experiment_dict.keys()
        experiment = ukb_experiment_dict[cfg.model.name](cfg)
    elif cfg.study == 'tg':
        assert cfg.model.name in tg_experiment_dict.keys()
        experiment = tg_experiment_dict[cfg.model.name](cfg)

    experiment.run()


if __name__ == '__main__':
    local_experiment()
