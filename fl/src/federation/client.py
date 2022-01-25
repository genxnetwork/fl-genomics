import torch
import numpy
from typing import Dict, OrderedDict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from flwr.client import NumPyClient
import logging
from model.mlp import BaseNet
from datasets.lightning import DataModule


class FLClient(NumPyClient):
    def __init__(self) -> None:
        super().__init__()


class FLClient(NumPyClient):
    def __init__(self, model: BaseNet, data_module: DataModule, logger: TensorBoardLogger, model_params: Dict, training_params: Dict):
        self.model = model
        self.data_module = data_module
        self.best_model_path = None
        self.logger = logger
        self.model_params = model_params
        self.training_params = training_params

    def get_parameters(self):
        parameters = _get_parameters(self.model)
        return parameters

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.current_step = config['current_step']
        trainer = Trainer(logger=self.logger, **self.training_params)
        trainer.fit(self.model, datamodule=self.data_module)
        
        return self.get_parameters(), self.data_module.train_len(), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # trainer = Trainer(progress_bar_refresh_rate=0, enable_model_summary=False, logger=self.logger)
        trainer = Trainer(logger=self.logger, **self.training_params)
        val_loader = self.model.val_dataloader()
        train_loader = self.model.unshuffled_train_dataloader()
        # train_loader = DataLoader(self.model.train_dataset, batch_size=64, num_workers=1, shuffle=False)
        
        val_results = trainer.validate(self.model, val_loader, verbose=False)
        train_results = trainer.validate(self.model, train_loader, verbose=False)

        train_loss, train_accuracy = train_results[0]["val_loss"], train_results[0]["val_accuracy"]
        val_loss, val_accuracy = val_results[0]["val_loss"], val_results[0]["val_accuracy"]

        logging.info(f'loss: {train_loss:.4f}\ttrain_acc: {train_accuracy:.4f}\tval_loss: {val_loss:.4f}\tval_acc: {val_accuracy:.4f}')
        return val_loss, len(val_loader), {"loss": val_loss, "accuracy": val_accuracy}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
    model.load_state_dict(state_dict, strict=True)