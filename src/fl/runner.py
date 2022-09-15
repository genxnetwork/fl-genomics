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
    # print('logger names:')
    # print(all_names)
    names = ['flower', 'pytorch_lightning']
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers = []


def get_active_nodes(cfg: DictConfig):
    if cfg.strategy.active_nodes == 'all':
        return [cfg.split.nodes[name] for name in cfg.split.nodes]
    active_nodes = []
    for node_index in cfg.strategy.active_nodes:
        active_nodes.append(cfg.split.nodes[node_index])
    return active_nodes


def get_log_dir():
    os.makedirs('logs', exist_ok=True)
    if 'SLURM_JOB_ID' in os.environ:
        # we are on a slurm cluster
        return f'logs/job-{os.environ["SLURM_JOB_ID"]}'
    else:
        old_dirs = []
        for dirname in os.listdir('logs'):
            dr = os.path.join('logs', dirname)
            if os.path.isdir(dr) and dirname.startswith('job-'):
                old_dirs.append(int(dirname[4:]))
            return f'logs/job-{max(old_dirs) + 1}'
        return f'logs/job-1'


@hydra.main(config_path='configs', config_name='default')
def run(cfg: DictConfig):

    configure_logging()
    # we can write ${div:${.max_rounds},${.rounds}} in yaml configs
    # to divide one number by another
    # we need it to infer number of local epochs in each federated round
    OmegaConf.register_new_resolver(
        'div', lambda x, y: int(x // y), replace=True
    )
    print(cfg)
    # parse command-line runner.py arguments
    queue = multiprocessing.Queue()
    server_url = f'{gethostname()}:8080'
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
    print(f'logging mlflow data to server {mlflow_url}')

    experiment = mlflow.set_experiment(cfg.experiment.name)

    params_hash = get_cfg_hash(cfg)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags={
            'description': cfg.experiment.description,
            'params_hash': params_hash,
            'description': cfg.experiment.description,
            'phenotype': cfg.data.phenotype.name,
            'split': cfg.split.name,
            'fold': str(cfg.fold.index)
        }
    ) as run:
        for field in ['server', 'strategy', 'model', 'optimizer', 'scheduler']:
            mlflow.log_params({field: OmegaConf.to_container(cfg[field], resolve=True)})
        info = MlflowInfo(experiment.experiment_id, run.info.run_id)

        # assigning gpus to nodes and creating process objects
        gpu_index = -1
        active_nodes = get_active_nodes(cfg)
        node_processes = []
        for node_info in active_nodes:
            need_gpu = node_info.resources.get('gpus', 0)
            if need_gpu:
                gpu_index += 1
                trainer_info = TrainerInfo([gpu_index], 'gpu', node_info.name, node_info.index)
            else:
                trainer_info = TrainerInfo(1, 'cpu', node_info.name, node_info.index)
            node = Node(server_url, log_dir, info, queue, cfg, trainer_info)
            node.start()
            print(f'starting node {node_info.index}, name: {node_info.name}')
            node_processes.append(node)

        # create, start and wait for server to finish
        server = Server(log_dir, queue, params_hash, cfg)
        print(f'SERVER CREATED SUCCESSFULLY')
        server.start()

        # wait for all nodes to finish
        for node in node_processes:
            node.join()

        # wait for server to finish
        server.join()

        print(f'Nodes are finished')


if __name__ == '__main__':
    run()
