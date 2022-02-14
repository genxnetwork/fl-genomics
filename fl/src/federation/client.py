import torch
from torch.nn.functional import mse_loss
import numpy
from typing import Dict, OrderedDict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from flwr.client import NumPyClient
import logging

from model.mlp import BaseNet
from datasets.lightning import DataModule


class FLClient(NumPyClient):
    def __init__(self, model: BaseNet, data_module: DataModule, logger: TensorBoardLogger, model_params: Dict, training_params: Dict):
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
        self.set_parameters(parameters)
        self.model.current_round = config['current_round']
        trainer = Trainer(logger=self.logger, **self.training_params)
        trainer.fit(self.model, datamodule=self.data_module)
        return self.get_parameters(), self.data_module.train_len(), {}

    def calculate_train_loss(self, trainer: Trainer) -> float:
        loader = self.data_module.predict_dataloader()
        train_preds = trainer.predict(self.model, loader)
        y_train = torch.cat([batch[1] for batch in iter(loader)])
        y_train_pred = torch.cat(train_preds).squeeze(1)
        # print(f'shapes are: ', y_train.shape, train_preds[0].shape, len(train_preds), y_train_pred.shape)
        mse = mse_loss(y_train_pred, y_train)
        return mse.item()

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = Trainer(logger=self.logger, **self.training_params)
        # train_loader = DataLoader(self.model.train_dataset, batch_size=64, num_workers=1, shuffle=False)
        
        val_results = trainer.validate(self.model, datamodule=self.data_module, verbose=False)
        val_loss = val_results[0]["val_loss"]
        train_loss = self.calculate_train_loss(trainer)

        logging.info(f'train_loss: {train_loss:.4f}\tval_loss: {val_loss:.4f}')
        return val_loss, self.data_module.val_len(), {"val_loss": val_loss, "train_loss": train_loss}
