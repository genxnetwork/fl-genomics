from dataclasses import dataclass
import logging
from multiprocessing import Process, Queue
import subprocess
from typing import Any, Dict, Tuple, List, Union
import time
from grpc import RpcError
import os

from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.entities import ViewType, RunInfo
from mlflow import ActiveRun, list_run_infos
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy
import flwr

from fl.datasets.memory import load_from_pgen, load_phenotype, load_covariates
from nn.lightning import DataModule
from nn.models import BaseNet, LinearRegressor, MLPRegressor
from fl.federation.client import FLClient


@dataclass
class MlflowInfo:
    experiment_id: str
    parent_run_id: str

@dataclass
class TrainerInfo:
    devices: Union[List[int], int]
    accelerator: str
    node_index: int

    def to_dotlist(self) -> List[str]:
        return [f'node.index={self.node_index}', f'node.training.devices={self.devices}', f'node.training.accelerator={self.accelerator}']


class Node(Process):
    def __init__(self, server_url: str, log_dir: str, mlflow_info: MlflowInfo, 
                 queue: Queue, cfg: DictConfig, trainer_info: TrainerInfo, **kwargs):
        """Process for training on one dataset node

        Args:
            server_url (str): Full url to flower server
            log_dir (str): Logging directory, where node-{node_index}.log file will be created
            mlflow_info (MlflowInfo): Mlflow parent run and experiment IDs
            queue (Queue): Queue for communication between processes
            cfg_path (str): Path to full yaml config
            trainer_info (TrainerInfo): Where to train node
        """        
        Process.__init__(self, **kwargs)
        os.environ['MASTER_PORT'] = str(47000+trainer_info.node_index) 
        self.node_index = trainer_info.node_index
        self.mlflow_info = mlflow_info
        self.trainer_info = trainer_info
        self.server_url = server_url
        self.queue = queue
        self.log_dir = log_dir
        node_cfg = OmegaConf.from_dotlist(self.trainer_info.to_dotlist())
        self.cfg = OmegaConf.merge(cfg, node_cfg)
    
    def _configure_logging(self):
        # to disable printing GPU TPU IPU info for each trainer each FL step
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
        # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        self.logger = logging.getLogger(f'node-{self.node_index}.log')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(os.path.join(self.log_dir, f'node-{self.node_index}.log')))

        # logging.basicConfig(filename=os.path.join(self.log_dir, f'node-{self.node_index}.log'), level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')

    def log(self, msg):
        self.logger.info(msg)

    def _start_client_run(self, client: MlflowClient, 
                        parent_run_id: str, 
                        experiment_id: str, 
                        tags: Dict[str, Any]) -> ActiveRun:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        # logging.info(f'starting to create mlflow run with parent {parent_run_id}')
        run = client.create_run(
            experiment_id,
            tags=tags,
        )
        # logging.info(f'run info id in _start_client_run is {run.info.run_id}')
        return mlflow.start_run(run.info.run_id, nested=True)

    def _load_data(self) -> DataModule:
        """Loads genotypes, covariates and phenotypes into DataModule

        Returns:
            DataModule: Subclass of LightningDataModule for loading data during training
        """        
        X_train = load_from_pgen(self.cfg.dataset.pfile.train, self.cfg.dataset.gwas, None, missing=self.cfg.experiment.missing) 
        X_val = load_from_pgen(self.cfg.dataset.pfile.val, self.cfg.dataset.gwas, None, missing=self.cfg.experiment.missing) 
        X_test = load_from_pgen(self.cfg.dataset.pfile.test, self.cfg.dataset.gwas, None, missing=self.cfg.experiment.missing) 

        self.log(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')
        
        X_cov_train = load_covariates(self.cfg.dataset.covariates.train)
        X_cov_val = load_covariates(self.cfg.dataset.covariates.val)
        X_cov_test =load_covariates(self.cfg.dataset.covariates.test)
        X_train = numpy.hstack([X_train, X_cov_train])
        X_val = numpy.hstack([X_val, X_cov_val])
        X_test = numpy.hstack([X_test, X_cov_test])
        self.log(f'We added {X_cov_train.shape[1]} covariates and got {X_train.shape[1]} total features')

        y_train, y_val, y_test = load_phenotype(self.cfg.dataset.phenotype.train), load_phenotype(self.cfg.dataset.phenotype.val), load_phenotype(self.cfg.dataset.phenotype.test)
        self.log(f'We have {y_train.shape[0]} train phenotypes and {y_val.shape[0]} val phenotypes')
        self.feature_count = X_train.shape[1]
        self.covariate_count = X_cov_train.shape[1]
        self.snp_count = self.feature_count - self.covariate_count
        self.sample_count = X_train.shape[0]

        data_module = DataModule(X_train, X_val, X_test, y_train, y_val, y_test, self.cfg.node.model.batch_size)
        return data_module

        
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


    def run(self) -> None:
        """Runs data loading and training of node
        """        
        self._configure_logging()
        # logging.info(f'logging is configured')
        mlflow_client = MlflowClient()
        data_module = self._load_data()

        client = FLClient(self.server_url, data_module, self.cfg.node)

        self.log(f'client created, starting mlflow run for {self.node_index}')
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
            self.log(f'Started run for node {self.node_index}')
            
            self._train_model(client)
            