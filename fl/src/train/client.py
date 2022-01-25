from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Dict
import mlflow

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
    pass


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    X_train, X_val = load_from_pgen(cfg.data.train, cfg.data.gwas), load_from_pgen(cfg.data.val, cfg.data.gwas)
    y_train, y_val = load_phenotype(cfg.data.phenotype.train), load_phenotype(cfg.data.phenotype.val)

    data_module = DataModule(X_train, X_val, y_train, y_val, cfg.training.batch_size)

    net = BaseNet(cfg.model)
    client = FLClient(net, data_module, None, cfg.model, cfg.training)

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