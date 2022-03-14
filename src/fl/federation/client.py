import torch
import logging
from typing import Dict, OrderedDict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from sklearn.metrics import r2_score

from model.mlp import BaseNet
from datasets.lightning import DataModule


class FLClient(NumPyClient):
    def __init__(self, server: str, model: BaseNet, data_module: DataModule, logger: TensorBoardLogger, model_params: Dict, training_params: Dict):
        self.server = server
        self.model = model
        self.data_module = data_module
        self.best_model_path = None
        self.logger = logger
        self.model_params = model_params
        self.training_params = training_params

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if v.shape != ()})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging.info('setting parameters in round {config["current_round"]}')
        self.set_parameters(parameters)
        self.model.current_round = config['current_round']
        trainer = Trainer(logger=self.logger, **self.training_params)
        trainer.fit(self.model, datamodule=self.data_module)
        return self.get_parameters(), self.data_module.train_len(), {}

    def calculate_r2_score(self, trainer: Trainer, loader: DataLoader) -> float:
        preds = trainer.predict(self.model, loader)
        preds = torch.cat(preds, dim=0)
        y_true = torch.cat([batch[1] for batch in iter(loader)])
        r2 = r2_score(y_true.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return r2

    def calculate_train_val_r2(self, trainer: Trainer) -> float:
        train_loader, val_loader = self.data_module.predict_dataloader()
        
        return self.calculate_r2_score(trainer, train_loader), self.calculate_r2_score(trainer, val_loader)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = Trainer(logger=self.logger, **self.training_params)
        # train_loader = DataLoader(self.model.train_dataset, batch_size=64, num_workers=1, shuffle=False)
        
        val_results = trainer.validate(self.model, datamodule=self.data_module, verbose=False)
        val_loss = val_results[0]["val_loss"]
        train_r2, val_r2 = self.calculate_train_val_r2(trainer)

        val_len = self.data_module.val_len()
        logging.info(f'train_r2: {train_r2:.4f}\tval_r2: {val_r2:.4f}\tval_loss: {val_loss:.3f}\tval_len: {val_len}')
        return val_loss, val_len, {"val_loss": val_loss, "train_r2": train_r2, "val_r2": val_r2}
