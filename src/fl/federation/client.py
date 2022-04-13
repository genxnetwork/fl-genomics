from omegaconf import DictConfig
import torch
import logging
from typing import Dict, OrderedDict, Tuple, Any
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

from nn.models import BaseNet, LinearRegressor, MLPRegressor
from fl.datasets.lightning import DataModule


class ModelFactory:
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
    def create_model(input_size: int, params: Any) -> BaseNet:
        if params.model.name == 'linear_regressor':
            return ModelFactory._create_linear_regressor(input_size, params)
        elif params.model.name == 'mlp_regressor':
            return ModelFactory._create_mlp_regressor(input_size, params)
        else:
            raise ValueError(f'model name {params.model.name} is unknown')


class FLClient(NumPyClient):
    def __init__(self, server: str, data_module: DataModule, node_params: DictConfig):
        """Trains {model} in federated setting

        Args:
            server (str): Server address with port
            model (BaseNet): Model to train
            data_module (DataModule): Module with train, val and test dataloaders
            logger (TensorBoardLogger): Local tensorboardlogger
            model_params (Dict): Model parameters
            training_params (Dict): pytorch_lightning Trainer parameters
        """        
        self.server = server
        self.model = ModelFactory.create_model(data_module.feature_count(), node_params)
        self.data_module = data_module
        self.best_model_path = None
        self.node_params = node_params

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
        except ReferenceError as re:
            print(re)
            self.model = ModelFactory.create_model(self.data_module.feature_count(), self.node_params)

        self.model.current_round = config['current_round']
        trainer = Trainer(logger=False, **self.node_params.training)
        trainer.fit(self.model, datamodule=self.data_module)
        return self.get_parameters(), self.data_module.train_len(), {}

    def calculate_loader_metrics(self, trainer: Trainer, loader: DataLoader) -> Tuple[float, float]:
        preds = trainer.predict(self.model, loader)
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        y_true = torch.cat([batch[1] for batch in iter(loader)]).detach().cpu().numpy()
        
        mse = mean_squared_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        # we do that because mse and r2 have type numpy.float32 which is not a valid type for return of `evaluate` function
        return float(mse), float(r2)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = Trainer(logger=False, **self.node_params.training)
        # train_loader = DataLoader(self.model.train_dataset, batch_size=64, num_workers=1, shuffle=False)
        
        train_loader, val_loader, test_loader = self.data_module.predict_dataloader()
        train_loss, train_r2 = self.calculate_loader_metrics(trainer, train_loader)
        val_loss, val_r2 = self.calculate_loader_metrics(trainer, val_loader)
        
        val_len = self.data_module.val_len()
        epoch = self.model.fl_current_epoch()
        logging.info(f'round: {self.model.current_round}\ttrain_loss: {train_loss:.4f}\ttrain_r2: {train_r2:.4f}\tval_loss: {val_loss:.3f}\tval_r2: {val_r2:.4f}\tval_len: {val_len}')
        mlflow.log_metric('train_loss', train_loss, epoch)
        mlflow.log_metric('train_r2', train_r2, epoch)
        mlflow.log_metric('val_r2', val_r2, epoch)
        mlflow.log_metric('val_loss', val_loss, epoch)
        
        results = {"val_loss": val_loss, "train_loss": train_loss, "train_r2": train_r2, "val_r2": val_r2}
        if 'current_round' in config and config['current_round'] == -1:
            print(f'Final evaluation started')
            test_loss, test_r2 = self.calculate_loader_metrics(trainer, test_loader)
            mlflow.log_metric('test_r2', test_r2)
            results['test_r2'] = test_r2
            print(f'test_r2 is {test_r2:.4f} for {self.data_module.test_len()} samples')
            results['test_len'] = self.data_module.test_len()

        logging.info(f'val_loss type: {type(val_loss)}, val_len type: {type(val_len)}')
        return val_loss, val_len, results
