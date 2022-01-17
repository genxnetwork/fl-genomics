from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Dict
import mlflow

from datasets.memory import Dataset
from datasets.lightning import DataModule
from model.mlp import Net
from federation.client import FLClient


def train_model(data_module: DataModule, model: Net, client: FLClient):
    pass

def evaluate_model(data_module: DataModule, model: Net, client: FLClient) -> Dict[str, float]:
    pass


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    train_dataset = Dataset(cfg.data.train)
    val_dataset = Dataset(cfg.data.val)
    data_module = DataModule(train_dataset, val_dataset)

    net = Net(cfg.model)
    
    client = FLClient()

    with mlflow.start_run(
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