from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Any, Dict
import os

import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import flwr

from datasets.memory import load_from_pgen, load_phenotype
from datasets.lightning import DataModule
from model.mlp import BaseNet
from federation.client import FLClient


def train_model(data_module: DataModule, model: BaseNet, client: FLClient):
    """
    Trains a model using data from {data_module} and using {client} for FL 

    Args:
        data_module (DataModule): Local data loaders manager
        model (BaseNet): Model to train
        client (FLClient): Federation Learning client which should implement weights exchange procedures.
    """
    pass


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


@hydra.main(config_path='../configs/client', config_name='default')
def main(cfg: DictConfig):

    parent_run_id = get_parent_run_id()
    print(f'parent_run_id is {parent_run_id}')
    mlflow_client = create_mlflow_client()

    X_train = load_from_pgen(cfg.data.genotype.train, cfg.data.gwas, 200)
    X_val = load_from_pgen(cfg.data.genotype.val, cfg.data.gwas, 200)
    print('Genotype data loaded')
    print(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')
    print(X_train[:5,:5])

    y_train, y_val = load_phenotype(cfg.data.phenotype.train), load_phenotype(cfg.data.phenotype.val)
    # data_module = DataModule(X_train, X_val, y_train, y_val, cfg.training.batch_size)

    # net = BaseNet(cfg.model)
    client = FLClient(None, None, None, cfg.model, cfg.training)

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
        server_hostname = os.environ['FLWR_SERVER_HOSTNAME']
        flwr.client.start_numpy_client(f'{server_hostname}:8080', client)

        # train_model(data_module, net, client)
        # metrics = evaluate_model(data_module, net, client)

        mlflow.log_metric('test_metric_client', 0.1)


if __name__ == '__main__':
    main()