from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Any, Dict

import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from datasets.memory import Dataset
from datasets.lightning import DataModule
from model.mlp import Net
from federation.client import FLClient


def train_model(data_module: DataModule, model: Net, client: FLClient):
    pass


def evaluate_model(data_module: DataModule, model: Net, client: FLClient) -> Dict[str, float]:
    pass


def get_parent_run_id() -> str:
    pass


def start_client_run(client: MlflowClient, parent_run_id: str, tags: Dict[str, Any]) -> ActiveRun:
    tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    run = client.create_run(
        tags=tags
    )
    
    return mlflow.start_run(run.info.run_id)


def create_mlflow_client() -> MlflowClient:
    client = MlflowClient(tracking_uri='')
    return client


@hydra.main(config_path='../configs/client', config_name='default')
def main(cfg: DictConfig):
    train_dataset = Dataset(cfg.data.train)
    val_dataset = Dataset(cfg.data.val)
    data_module = DataModule(train_dataset, val_dataset)

    net = Net(cfg.model)
    
    mlflow_client = create_mlflow_client()
    client = FLClient()

    parent_run_id = get_parent_run_id()
    with start_client_run(
        parent_run_id=parent_run_id,
        tags={
            'description': cfg.description
        }
    ):
        mlflow.log_params({  
            'model': OmegaConf.to_container(cfg.model)
        })

        train_model(data_module, net, client)
        metrics = evaluate_model(data_module, net, client)

        mlflow.log_metrics(metrics)


if __name__ == '__main__':
    main()