from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Any, Dict
import os
import logging

import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import flwr
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.memory import load_from_pgen, load_phenotype
from datasets.lightning import DataModule
from model.mlp import BaseNet, LinearRegressor
from federation.client import FLClient


def train_model(client: FLClient):
    """
    Trains a model using data from {data_module} and using {client} for FL 

    Args:
        data_module (DataModule): Local data loaders manager
        model (BaseNet): Model to train
        client (FLClient): Federation Learning client which should implement weights exchange procedures.
    """
    
    server_hostname = os.environ['FLWR_SERVER_HOSTNAME']
    flwr.client.start_numpy_client(f'{server_hostname}:8080', client)


def evaluate_model(data_module: DataModule, model: BaseNet, client: FLClient) -> Dict[str, float]:
    """
    Evaluates a trained model locally

    Args:
        data_module (DataModule): Local data loaders manager.
        model (BaseNet): Model to evaluate.
        client (FLClient): Federation Learning client which should implement weights exchange procedures.

    Returns:
        Dict[str, float]: Dict with 'val_loss', 'val_accuracy'. 
    """    
    return {'dummy_metric': 0.0}


def get_parent_run_id() -> str:
    return os.environ['MLFLOW_PARENT_RUN_ID']


def start_client_run(client: MlflowClient, parent_run_id: str, tags: Dict[str, Any]) -> ActiveRun:
    tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    run = client.create_run(
        "0",
        tags=tags
    )
    
    return mlflow.start_run(run.info.run_id)


def create_mlflow_client() -> MlflowClient:
    client = MlflowClient()
    return client


def configure_logging():
    # to disable printing GPU TPU IPU info for each trainer 
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


@hydra.main(config_path='../configs/client', config_name='default')
def main(cfg: DictConfig):
    configure_logging()

    parent_run_id = get_parent_run_id()
    print(f'parent_run_id is {parent_run_id}')
    mlflow_client = create_mlflow_client()

    X_train = load_from_pgen(cfg.data.genotype.train, cfg.data.gwas, None) # load all snps
    X_val = load_from_pgen(cfg.data.genotype.val, cfg.data.gwas, None) # load all snps
    print('Genotype data loaded')
    print(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')
    print(X_train[:5,:5])

    y_train, y_val = load_phenotype(cfg.data.phenotype.train), load_phenotype(cfg.data.phenotype.val)
    print(f'We have {y_train.shape[0]} train phenotypes and {y_val.shape[0]} val phenotypes')
    data_module = DataModule(X_train, X_val, y_train, y_val, cfg.model.batch_size)

    net = LinearRegressor(
        input_size=X_train.shape[1],
        l1=cfg.model.l1,
        lr=cfg.model.lr,
        momentum=cfg.model.momentum,
        epochs=cfg.model.epochs
    )
    # custom experiment name, because on the same filesystem 
    # default tensorboard logger creates logs directory with the same name in the same folder in the federated setting
    # it leads to a fail of one of the clients 
    logger = TensorBoardLogger('tb_logs', name=parent_run_id, sub_dir=f'node{cfg.node_index}')
    client = FLClient(net, data_module, logger, cfg.model, cfg.training)

    with start_client_run(
        mlflow_client,
        parent_run_id=parent_run_id,
        tags={
            'description': cfg.experiment.description
        }
    ):
        mlflow.log_params({  
            'model': OmegaConf.to_container(cfg.model)
        })

        train_model(client)
        metrics = evaluate_model(data_module, net, client)

        mlflow.log_metrics(metrics)


if __name__ == '__main__':
    main()