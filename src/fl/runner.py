import multiprocessing
import sys
import os
from socket import gethostname
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import hashlib
import logging

from fl.node_process import MlflowInfo, Node, TrainerInfo
from fl.server_process import Server


# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)

sys.path.append(os.getcwd()) 


def get_cfg_hash(cfg: DictConfig):
    yaml_representation = OmegaConf.to_yaml(cfg)
    return hashlib.sha224(yaml_representation.encode()).hexdigest()


def configure_logging():
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    all_names = [name for name in logging.root.manager.loggerDict]
    print('logger names:')
    print(all_names)
    names = ['flower', 'pytorch_lightning']
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers = []


def get_active_nodes(cfg: DictConfig):
    if cfg.strategy.active_nodes == 'all':
        return cfg.split.nodes
    return cfg.split.nodes[cfg.strategy.active_nodes]


@hydra.main(config_path='configs', config_name='default')
def run():
    
    configure_logging()
    print(f'mlflow env vars: {[m for m in os.environ if "MLFLOW" in m]}')

    # parse command-line runner.py arguments
    args = OmegaConf.from_cli(sys.argv)
    queue = multiprocessing.Queue()
    cfg_path = 'src/fl/configs/lassonet.yaml'
    server_url = f'{gethostname()}:8080'
    log_dir = f'logs/job-{os.environ["SLURM_JOB_ID"]}'
    os.makedirs(log_dir, exist_ok=True)

    # command-line arguments take precedents over config parameters
    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
    print(f'logging mlflow data to server {mlflow_url}')
    
    cfg = OmegaConf.merge(OmegaConf.load(cfg_path), args)
    experiment = mlflow.set_experiment(cfg.experiment.name)

    params_hash = get_cfg_hash(cfg)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
        }
    ) as run:
        mlflow.log_params(cfg.server)
        mlflow.log_params(cfg.strategy)
        info = MlflowInfo(experiment.experiment_id, run.info.run_id)

        # assigning gpus to nodes and creating process objects
        gpu_index = -1
        active_nodes = get_active_nodes(cfg)
        node_processes = []
        for node_info in active_nodes:
            need_gpu = node_info.resources.get('gpus', 0)
            if need_gpu:
                gpu_index += 1
                trainer_info = TrainerInfo([gpu_index], 'gpu', node_info.index)
            else:
                trainer_info = TrainerInfo(1, 'cpu', node_info.index)
            node = Node(server_url, log_dir, info, queue, cfg, trainer_info)
            node.start()
            print(f'starting node {node_info.index}, name: {node_info.name}')
            node_processes.append(node)
        
        # create, start and wait for server to finish 
        server = Server(log_dir, queue, params_hash, cfg)
        server.start()
        server.join()
        
        # wait for all nodes to finish
        for node in node_processes:
            node.join()
        print(f'Nodes are finished')
    

if __name__ == '__main__':
    run()
