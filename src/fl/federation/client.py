from omegaconf import DictConfig
import torch
from logging import Logger
from typing import Dict, OrderedDict, Tuple, Any
from pytorch_lightning.trainer import Trainer
from flwr.client import NumPyClient
from time import time

from nn.models import BaseNet, LinearRegressor, MLPRegressor, LassoNetRegressor
from nn.lightning import DataModule


class ModelFactory:
    """Class for creating models based on model name from config

    Raises:
        ValueError: If model is not one of the linear_regressor, mlp_regressor

    """    
    @staticmethod
    def _create_linear_regressor(input_size: int, params: Any) -> LinearRegressor:
        return LinearRegressor(
            input_size=input_size,
            l1=params.model.l1,
            optim_params=params['optimizer'],
            scheduler_params=params['scheduler']
        )

    @staticmethod
    def _create_mlp_regressor(input_size: int, params: Any) -> MLPRegressor:
        return MLPRegressor(
            input_size=input_size,
            hidden_size=params.model.hidden_size,
            l1=params.model.l1,
            optim_params=params['optimizer'],
            scheduler_params=params['scheduler']
        )

    @staticmethod
    def _create_lassonet_regressor(input_size: int, params: Any) -> LassoNetRegressor:
        return LassoNetRegressor(
            input_size=input_size,
            hidden_size=params.model.hidden_size,
            optim_params=params.optimizer,
            scheduler_params=params.scheduler,
            cov_count=2, #TODO: make configurable or computable
            alpha_start=params.model.alpha_start,
            alpha_end=params.model.alpha_end,
            init_limit=params.model.init_limit
        )

    @staticmethod
    def create_model(input_size: int, params: Any) -> BaseNet:
        model_dict = {
            'linear_regressor': ModelFactory._create_linear_regressor,
            'mlp_regressor': ModelFactory._create_mlp_regressor,
            'lassonet_regressor': ModelFactory._create_lassonet_regressor
        }

        create_func = model_dict.get(params.model.name, None)
        if create_func is None:
            raise ValueError(f'model name {params.model.name} is unknown, it should be one of the {list(model_dict.keys())}')
        return create_func(input_size, params)


class FLClient(NumPyClient):
    def __init__(self, server: str, data_module: DataModule, node_params: DictConfig, logger: Logger):
        """Trains {model} in federated setting

        Args:
            server (str): Server address with port
            data_module (DataModule): Module with train, val and test dataloaders
            node_params (Dict): Node config with model, dataset, and training parameters
        """        
        self.server = server
        self.model = ModelFactory.create_model(data_module.feature_count(), node_params)
        self.data_module = data_module
        self.best_model_path = None
        self.node_params = node_params
        self.logger = logger
        self.log(f'cuda device count: {torch.cuda.device_count()}')

    def log(self, msg):
        self.logger.info(msg)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            # to catch spurious error "weakly-referenced object no longer exists"
            # probably ref to some model parameter tensor get lost
            self.set_parameters(parameters)
        except ReferenceError as re:
            print(re)
            # we recreate a model and set parameters again
            self.model = ModelFactory.create_model(self.data_module.feature_count(), self.node_params)
            self.set_parameters(parameters)
        
        start = time()                
        self.model.train()
        self.model.current_round = config['current_round']
        trainer = Trainer(logger=False, **self.node_params.training)
        trainer.fit(self.model, datamodule=self.data_module)
        end = time()
        self.log(f'node: {self.node_params.index}\tfit elapsed: {end-start:.2f}s')
        return self.get_parameters(), self.data_module.train_len(), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        start = time()                
        need_test_eval = 'current_round' in config and config['current_round'] == -1
        unreduced_metrics = self.model.predict_and_eval(self.data_module, 
                                                        test=need_test_eval, 
                                                        best_col=config.get('best_col', None))
        unreduced_metrics.log_to_mlflow()
        val_len = self.data_module.val_len()
        end = time()
        self.log(f'node: {self.node_params.index}\tround: {self.model.current_round}\t' + str(unreduced_metrics) + f'\telapsed: {end-start:.2f}s')
        # print(f'round: {self.model.current_round}\t' + str(unreduced_metrics))
        
        results = unreduced_metrics.to_result_dict()
        return unreduced_metrics.val_loss, val_len, results
