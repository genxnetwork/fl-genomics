from collections import namedtuple
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Tuple
import logging
import time
from grpc import RpcError
import mlflow
import numpy
from mlflow.entities import ViewType, RunInfo
from mlflow import ActiveRun, list_run_infos
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import flwr
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.memory import load_from_pgen, load_phenotype, load_covariates
from datasets.lightning import DataModule
from model.mlp import BaseNet, LinearRegressor, MLPRegressor
from federation.client import FLClient


def train_model(client: FLClient):
    """
    Trains a model using data from {data_module} and using {client} for FL 

    Args:
        data_module (DataModule): Local data loaders manager
        model (BaseNet): Model to train
        client (FLClient): Federation Learning client which should implement weights exchange procedures.
    """
    for i in range(20):
        try:
            flwr.client.start_numpy_client(f'{client.server}:8080', client)
        except RpcError as re:
            # probably server slurm job have not started yet
            time.sleep(5)
            continue
        

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

    
def get_parent_run_id_and_host(mlflow_client: MlflowClient, experiment_name: str, params_hash: str) -> Tuple[str, str, str]:
    logging.info(f'waiting for experiment to exist and active run to be created')
    for i in range(20):
        time.sleep(5)
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.info(f'experiment with name {experiment_name} is none at step {i}')
            continue
        logging.info(f'searching runs with experiment id {experiment.experiment_id}')
        runs = mlflow_client.search_runs(experiment_ids=experiment.experiment_id,
                                run_view_type=ViewType.ACTIVE_ONLY,
                                filter_string=f"tags.params_hash='{params_hash}'")
        if len(runs) < 1:
            logging.info(f'can not found at least one active run with config hash {params_hash}')
            continue
        else:
            break
    else:
        raise RuntimeError(f'Cant find at least one active run with config hash {params_hash}, found {len(runs)}, waited {12*5} seconds')
    
    # Default order is START_TIME DESC and we need the most recent active run with {params_hash}
    return runs[0].info.run_id, runs[0].data.tags['hostname'], runs[0].info.experiment_id


def start_client_run(client: MlflowClient, parent_run_id: str, experiment_id: str, tags: Dict[str, Any]) -> ActiveRun:
    tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    run = client.create_run(
        experiment_id,
        tags=tags
    )
    
    return mlflow.start_run(run.info.run_id)


def configure_logging():
    # to disable printing GPU TPU IPU info for each trainer each FL step
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')


def create_linear_regressor(input_size: int, params: Any) -> LinearRegressor:
    return LinearRegressor(
        input_size=input_size,
        l1=params.model.l1,
        optim_params=params['optimizer'],
        scheduler_params=params['scheduler']
    )

def create_mlp_regressor(input_size: int, params: Any) -> MLPRegressor:
    return MLPRegressor(
        input_size=input_size,
        hidden_size=params.model.hidden_size,
        l1=params.model.l1,
        optim_params=params['optimizer'],
        scheduler_params=params['scheduler']
    )

def create_model(input_size: int, params: Any) -> BaseNet:
    if params.model.name == 'linear_regressor':
        return create_linear_regressor(input_size, params)
    elif params.name == 'mlp_regressor':
        return create_mlp_regressor(input_size, params)
    else:
        raise ValueError(f'model name {params.name} is unknown')


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'name': 'test.input'},
            output={'name': 'test.output'},
            params={'parameter': 'value'},
            resources={'resource_name': 'value'},
            log=['test.log']
        )

    config_path = snakemake.input['config'][0]
    print(f'config_path: {config_path}')
    print(f'config_path type is {type(config_path)}')
    gwas = snakemake.input['gwas']

    pheno_train = snakemake.input['pheno_train']
    pheno_val = snakemake.input['pheno_val']

    cov_train = snakemake.input['cov_train']
    cov_val = snakemake.input['cov_val']

    pfile_train = snakemake.params['pfile_train']
    pfile_val = snakemake.params['pfile_val']

    params_hash = snakemake.wildcards['params_hash']
    node_index = snakemake.wildcards['node']



    cfg = OmegaConf.load(config_path)
    if int(snakemake.resources['gpus']) == 0:
        # we ran some nodes on gpu and some on cpu
        cfg.node.training.gpus = None 

    configure_logging()

    mlflow_client = MlflowClient()

    parent_run_id, hostname, experiment_id = get_parent_run_id_and_host(mlflow_client, cfg.experiment.name, params_hash)
    logging.info(f'parent_run_id is {parent_run_id}, hostname is {hostname}')

    X_train = load_from_pgen(pfile_train, gwas, None, missing=cfg.experiment.missing) # load all snps
    X_val = load_from_pgen(pfile_val, gwas, None, missing=cfg.experiment.missing) # load all snps
    logging.info(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')
    
    X_cov_train = load_covariates(cov_train)
    X_cov_val = load_covariates(cov_val)
    X_train = numpy.hstack([X_train, X_cov_train])
    X_val = numpy.hstack([X_val, X_cov_val])
    logging.info(f'We added {X_cov_train.shape[1]} covariates and got {X_train.shape[1]} total features')

    y_train, y_val = load_phenotype(pheno_train), load_phenotype(pheno_val)
    logging.info(f'We have {y_train.shape[0]} train phenotypes and {y_val.shape[0]} val phenotypes')
    data_module = DataModule(X_train, X_val, y_train, y_val, cfg.node.model.batch_size)

    net = create_model(X_train.shape[1], cfg.node)
    # custom experiment name, because on the same filesystem 
    # default tensorboard logger creates logs directory with the same name in the same folder in the federated setting
    # it leads to a fail of one of the clients 
    logger = TensorBoardLogger('tb_logs', name=parent_run_id, sub_dir=f'node_{node_index}')
    client = FLClient(hostname, net, data_module, logger, cfg.node.model, cfg.node.training)

    with start_client_run(
        mlflow_client,
        parent_run_id=parent_run_id,
        experiment_id=experiment_id,
        tags={
            'description': cfg.experiment.description
        }
    ):
        mlflow.log_params(OmegaConf.to_container(cfg.node))

        train_model(client)
        metrics = evaluate_model(data_module, net, client)

        # mlflow.log_metrics(metrics)
