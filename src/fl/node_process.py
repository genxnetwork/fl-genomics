from dataclasses import dataclass
import logging
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Union
import time
from grpc import RpcError
import os

from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy
import flwr
from sklearn.linear_model import LinearRegression

from nn.lightning import DataModule
from fl.federation.client import FLClient
from local.experiment import NNExperiment


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
            cfg_path (str): Path to full yaml configs
            trainer_info (TrainerInfo): Where to train node
        """        
        Process.__init__(self, **kwargs)
        os.environ['MASTER_PORT'] = str(47000+numpy.random.randint(1000)+trainer_info.node_index) 
        self.node_index = trainer_info.node_index
        self.mlflow_info = mlflow_info
        self.trainer_info = trainer_info
        self.server_url = server_url
        self.queue = queue
        self.log_dir = log_dir
        node_cfg = OmegaConf.from_dotlist(self.trainer_info.to_dotlist())
        self.cfg = OmegaConf.merge(cfg, node_cfg)
        self.experiment = NNExperiment(cfg)
    
    def _configure_logging(self):
        # to disable printing GPU TPU IPU info for each trainer each FL step
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
        # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        self.logger = logging.getLogger(f'node-{self.node_index}.log')
        self.logger.setLevel(logging.DEBUG)
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
        self.log(f'mlflow env vars: {[m for m in os.environ if "MLFLOW" in m]}')
        # logging.info(f'run info id in _start_client_run is {run.info.run_id}')
        return mlflow.start_run(run.info.run_id, nested=True)

    def _load_data(self) -> DataModule:
        """Loads genotypes, covariates and phenotypes into DataModule

        Returns:
            DataModule: Subclass of LightningDataModule for loading data during training
        """        
        self.feature_count = self.experiment.x.train.shape[1]
        self.covariate_count = self.experiment.x_cov.train.shape[1]
        self.snp_count = self.feature_count - self.covariate_count
        self.sample_count = self.experiment.x.train.shape[0]

        data_module = DataModule(self.experiment.x.train, 
                                 self.experiment.x.val, 
                                 self.experiment.x.test, 
                                 self.experiment.y.train, 
                                 self.experiment.y.val, 
                                 self.experiment.y.test, 
                                 self.cfg.node.model.batch_size,
                                 self.experiment.x_cov.train, 
                                 self.experiment.x_cov.val, 
                                 self.experiment.x_cov.test)
        return data_module

    def _pretrain(self) -> numpy.ndarray:
        """Pretrains linear regression on phenotype and covariates

        Returns:
            numpy.ndarray: Coefficients of covarites without intercept
        """        
        lr = LinearRegression()
        lr.fit(self.experiment.x_cov.train, self.experiment.y.train)
        val_r2 = lr.score(self.experiment.x_cov.val, self.experiment.y.val)
        train_r2 = lr.score(self.experiment.x_cov.train, self.y_train)
        samples, cov_count = self.experiment.x_cov.train.shape
        self.log(f'pretraining on {samples} samples and {cov_count} covariates gives {train_r2:.4f} train r2 and {val_r2:.4f} val r2')
        return lr.coef_
        
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
            except Exception as e:
                self.logger.error(e)
                self.logger.error(e.with_traceback())
                raise e
        return False

    def run(self) -> None:
        """Runs data loading and training of node
        """        
        self._configure_logging()
        # logging.info(f'logging is configured')
        mlflow_client = MlflowClient()
        data_module = self._load_data()

        client = FLClient(self.server_url, data_module, self.cfg.node, self.logger)

        self.log(f'client created, starting mlflow run for {self.node_index}')
        with self._start_client_run(
            mlflow_client,
            parent_run_id=self.mlflow_info.parent_run_id,
            experiment_id=self.mlflow_info.experiment_id,
            tags={
                'description': self.cfg.experiment.description,
                'node_index': str(self.node_index),
                'phenotype': self.cfg.data.phenotype.name,
                #TODO: make it a parameter
                'split': self.cfg.split.name,
                'snp_count': str(self.snp_count),
                'sample_count': str(self.sample_count)
            }
        ):
            mlflow.log_params(OmegaConf.to_container(self.cfg.node, resolve=True))
            self.log(f'Started run for node {self.node_index}')
            if self.cfg.experiment.pretrain_on_cov:
                cov_weights = self._pretrain() 
                client.model.set_covariate_weights(cov_weights)
            self._train_model(client)
            