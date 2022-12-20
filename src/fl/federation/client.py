import json
import pickle
import numpy
from omegaconf import DictConfig
from time import time
from pytorch_lightning import Callback
import torch
from logging import Logger
import mlflow
from typing import Dict, List, OrderedDict, Tuple, Any
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.model_summary import _format_summary_table, summarize
from pytorch_lightning.callbacks import EarlyStopping
from abc import ABC, abstractmethod

from flwr.client import NumPyClient
from flwr.common import Weights, parameters_to_weights

from nn.models import BaseNet, LinearRegressor, MLPClassifier, MLPPredictor, LassoNetRegressor, LassoNetClassifier
from nn.lightning import DataModule
from fl.federation.callbacks import ClientCallback, ScaffoldCallback
from fl.federation.utils import weights_to_module_params, bytes_to_weights
from configs.phenotype_config import PHENO_TYPE_DICT, TYPE_LOSS_DICT
from nn.metrics import ModelMetrics
from local.experiment import LocalExperiment


class CallbackFactory:
    @staticmethod
    def create_callbacks(params: DictConfig) -> List[Callback]:
        callbacks = []
        # early_stopping = EarlyStopping('val_loss', patience=params.training.max_epochs, check_finite=True)
        # callbacks.append(early_stopping)
        if params.strategy.name == 'scaffold':
            callbacks.append(
                ScaffoldCallback(params.strategy.args.K,
                                 log_grad=params.log_grad,
                                 log_diff=params.log_weights,
                                 grad_lr=params.strategy.args.grad_lr)
            )
        return callbacks


class MetricsLogger(ABC):
    @abstractmethod
    def log_eval_metric(self, metric: ModelMetrics):
        pass

    @abstractmethod
    def log_weights(self, rnd: int, layers: List[str], old_weights: Weights, new_weights: Weights):
        pass


class MLFlowMetricsLogger(MetricsLogger):
    def log_eval_metric(self, metric: ModelMetrics):
        mm_dict = metric.to_dict()
        mlflow.log_metrics(mm_dict, metric.epoch)

    def log_weights(self, rnd: int, layers: List[str], old_weights: Weights, new_weights: Weights):
        # logging.info(f'weights shape: {[w.shape for w in new_weights]}')

        client_diffs = {layer: numpy.linalg.norm(cw - aw).item() for layer, cw, aw in zip(layers, old_weights, new_weights)}
        for layer, diff in client_diffs.items():
            mlflow.log_metric(f'{layer}.l2', diff, rnd)


class FLClient(NumPyClient):
    def __init__(self, server: str, experiment: LocalExperiment, params: DictConfig, logger: Logger, metrics_logger: MetricsLogger, callbacks: List[ClientCallback] = None):
        """Trains {model} in federated setting

        Args:
            server (str): Server address with port
            data_module (DataModule): Module with train, val and test dataloaders
            params (DictConfig): OmegaConf DictConfig with subnodes strategy and node. `node` should have `model`, and `training` parameters
            logger (Logger): Process-specific logger of text messages
            metrics_logger (MetricsLogger): Process-specific logger of metrics and weights (mlflow, neptune, etc)
        """
        self.server = server
        self.callbacks = CallbackFactory.create_callbacks(params)
        self.experiment = experiment
        self.experiment.create_model()
        self.best_model_path = None
        self.params = params
        self.logger = logger
        self.metrics_logger = metrics_logger
        self.client_callbacks = callbacks
        self.log(f'cuda device count: {torch.cuda.device_count()}')

    def log(self, msg):
        self.logger.info(msg)

    def get_parameters(self) -> Weights:
        return [val.cpu().numpy() for _, val in self.experiment.model.state_dict().items()]

    def set_parameters(self, weights: Weights):
        state_dict = weights_to_module_params(self._get_layer_names(), weights)
        self.experiment.model.load_state_dict(state_dict, strict=True)

    def _get_layer_names(self) -> List[str]:
        return list(self.experiment.model.state_dict().keys())

    def update_callbacks_before_fit(self, update_params: Dict, **kwargs):
        if self.callbacks is None:
            return
        for callback in self.callbacks:
            if isinstance(callback, ScaffoldCallback):
                callback.c_global = weights_to_module_params(self._get_layer_names(), bytes_to_weights(update_params['c_global']))

    def update_callbacks_after_fit(self, update_params: Dict, **kwargs):
        if self.callbacks is None:
            return
        for callback in self.callbacks:
            if isinstance(callback, ScaffoldCallback):
                callback.update_c_local(
                    kwargs['eta'], callback.c_global,
                    old_params=weights_to_module_params(self._get_layer_names(), kwargs['old_params']),
                    new_params=weights_to_module_params(self._get_layer_names(), kwargs['new_params'])
                )

    def _reseed_torch(self):
        torch.manual_seed(hash(self.params.node.index) + self.experiment.model.current_round)

    def on_before_fit(self, config: Dict):
        self.update_callbacks_before_fit(config)
        if self.client_callbacks is not None:
            for callback in self.client_callbacks:
                callback.on_before_fit(self.experiment.model)

    def on_after_fit(self, old_params: Weights, new_params: Weights):
        self.update_callbacks_after_fit(None, old_params=old_params, new_params=new_params, eta=self.experiment.model.get_current_lr())
        fit_result = {}
        if self.client_callbacks is not None:
            for callback in self.client_callbacks:
                fit_result |= callback.on_after_fit(self.experiment.model)
        return fit_result

    def fit(self, parameters: Weights, config):
        try:
            # to catch spurious error "weakly-referenced object no longer exists"
            # probably ref to some model parameter tensor get lost
            self.set_parameters(parameters)
        except ReferenceError as re:
            self.logger.error(re)
            # we recreate a model and set parameters again
            self.experiment.model = self.experiment.create_model()
            self.set_parameters(parameters)

        self.on_before_fit(config)
            
        old_parameters = [p.copy() for p in parameters]
        start = time()    
        self.experiment.model.train()
        self.experiment.model.current_round = config['current_round']
        # because train_dataloader will get the same seed and return the same permutation of training samples each federated round
        self._reseed_torch()
        trainer = Trainer(logger=False, **{**self.params.training, **{'callbacks': self.callbacks}})
            
        trainer.fit(self.experiment.model, datamodule=self.experiment.data_module)
        end = time()
        self.log(f'node: {self.params.node.index}\tfit elapsed: {end-start:.2f}s')
        new_params = self.get_parameters()
        fit_result = self.on_after_fit(old_params=old_parameters, new_params=new_params)
            
        return new_params, self.experiment.data_module.train_len(), fit_result

    def evaluate(self, parameters: Weights, config):
        self.log(f'starting to set parameters in evaluate with config {config}')
        old_weights = self.get_parameters()
        if self.params.log_weights:
            self.metrics_logger.log_weights(self.experiment.model.fl_current_epoch(), self._get_layer_names(), old_weights, parameters)
        self.set_parameters(parameters)
        self.experiment.model.eval()

        start = time()
        need_test_eval = 'current_round' in config and config['current_round'] == -1
        unreduced_metrics = self.experiment.model.predict_and_eval(self.experiment.data_module, 
                                                        test=need_test_eval)

        self.metrics_logger.log_eval_metric(unreduced_metrics)
        val_len = self.experiment.data_module.val_len()
        end = time()
        self.log(f'node: {self.params.node.index}\tround: {self.experiment.model.current_round}\t' + str(unreduced_metrics) + f'\telapsed: {end-start:.2f}s')
        
        return unreduced_metrics.val_loss, val_len, {'metrics': pickle.dumps(unreduced_metrics)}
