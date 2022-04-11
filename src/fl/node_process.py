from dataclasses import dataclass
import logging
from multiprocessing import Process, Queue
from typing import Any, Dict, Tuple
import time
from grpc import RpcError

from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.entities import ViewType, RunInfo
from mlflow import ActiveRun, list_run_infos
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy
import flwr

from fl.datasets.memory import load_from_pgen, load_phenotype, load_covariates
from fl.datasets.lightning import DataModule
from nn.models import BaseNet, LinearRegressor, MLPRegressor
from fl.federation.client import FLClient


@dataclass
class MlflowInfo:
    experiment_id: str
    parent_run_id: str


class Node(Process):
    def __init__(self, node_index: int, server_url: str, mlflow_info: MlflowInfo, queue: Queue, cfg_path: str, gpu_index: int, **kwargs):
        Process.__init__(self, **kwargs)
        self.node_index = node_index
        self.mlflow_info = mlflow_info
        self.server_url = server_url
        self.queue = queue
        node_cfg = OmegaConf.from_dotlist([f'node.index={node_index}', 
                                           f'node.training.gpus={"null" if gpu_index < 0 else gpu_index}'])
        self.cfg = OmegaConf.merge(node_cfg, OmegaConf.load(cfg_path))
        print(self.cfg)

    def _start_client_run(self, client: MlflowClient, 
                        parent_run_id: str, 
                        experiment_id: str, 
                        tags: Dict[str, Any]) -> ActiveRun:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        run = client.create_run(
            experiment_id,
            tags=tags,
        )
        print(f'run info id in _start_client_run is {run.info.run_id}')
        return mlflow.start_run(run.info.run_id, nested=True)

    def _load_data(self):
        X_train = load_from_pgen(self.cfg.dataset.pfile.train, self.cfg.dataset.gwas, None, missing=self.cfg.experiment.missing) 
        X_val = load_from_pgen(self.cfg.dataset.pfile.val, self.cfg.dataset.gwas, None, missing=self.cfg.experiment.missing) 
        logging.info(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')
        
        X_cov_train = load_covariates(self.cfg.dataset.covariates.train)
        X_cov_val = load_covariates(self.cfg.dataset.covariates.val)
        X_train = numpy.hstack([X_train, X_cov_train])
        X_val = numpy.hstack([X_val, X_cov_val])
        logging.info(f'We added {X_cov_train.shape[1]} covariates and got {X_train.shape[1]} total features')

        y_train, y_val = load_phenotype(self.cfg.dataset.phenotype.train), load_phenotype(self.cfg.dataset.phenotype.val)
        logging.info(f'We have {y_train.shape[0]} train phenotypes and {y_val.shape[0]} val phenotypes')
        self.feature_count = X_train.shape[1]
        self.covariate_count = X_cov_train.shape[1]
        self.snp_count = self.feature_count - self.covariate_count
        self.sample_count = X_train.shape[0]

        data_module = DataModule(X_train, X_val, y_train, y_val, self.cfg.node.model.batch_size)
        return data_module

    def _create_linear_regressor(self, input_size: int, params: Any) -> LinearRegressor:
        return LinearRegressor(
            input_size=input_size,
            l1=params.model.l1,
            optim_params=params['optimizer'],
            scheduler_params=params['scheduler']
        )

    def _create_mlp_regressor(self, input_size: int, params: Any) -> MLPRegressor:
        return MLPRegressor(
            input_size=input_size,
            hidden_size=params.model.hidden_size,
            l1=params.model.l1,
            optim_params=params['optimizer'],
            scheduler_params=params['scheduler']
        )

    def _create_model(self, input_size: int, params: Any) -> BaseNet:
        if params.model.name == 'linear_regressor':
            return self._create_linear_regressor(input_size, params)
        elif params.model.name == 'mlp_regressor':
            return self._create_mlp_regressor(input_size, params)
        else:
            raise ValueError(f'model name {params.model.name} is unknown')
        
    def _train_model(self, client: FLClient) -> bool:
        """
        Trains a model using {client} for FL 

        Args:
            client (FLClient): Federation Learning client which should implement weights exchange procedures.
        """
        for i in range(20):
            try:
                print(f'starting numpy client with server {client.server}')
                flwr.client.start_numpy_client(f'{client.server}', client)
                return True
            except RpcError as re:
                # probably server slurm job have not started yet
                time.sleep(5)
                continue
        return False

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

    def run(self) -> None:
        
        mlflow_client = MlflowClient()
        data_module = self._load_data()

        net = self._create_model(self.feature_count, self.cfg.node)
        client = FLClient(self.server_url, net, data_module, self.cfg.node.model, self.cfg.node.training)

        with self._start_client_run(
            mlflow_client,
            parent_run_id=self.mlflow_info.parent_run_id,
            experiment_id=self.mlflow_info.experiment_id,
            tags={
                'description': self.cfg.experiment.description,
                'node_index': str(self.node_index),
                'phenotype': self.cfg.dataset.phenotype.name,
                #TODO: make it a parameter
                'split': self.cfg.dataset.split.name,
                'snp_count': str(self.snp_count),
                'sample_count': str(self.sample_count)
            }
        ):
            mlflow.log_params(OmegaConf.to_container(self.cfg.node))
            logging.info(f'Started run for node {self.node_index}')
            
            if self._train_model(client):
                metrics = self._evaluate_model(data_module, net, client)
            else:
                raise RuntimeError('Can not connect to server')
            
