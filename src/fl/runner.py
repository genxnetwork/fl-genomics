import multiprocessing
import sys
import os
from socket import gethostname
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import pandas
import hashlib
import logging


from fl.node_process import MlflowInfo, Node, TrainerInfo
from fl.server_process import Server
from utils.loaders import calculate_sample_weights
from utils.network import get_available_ports


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


def precalc_sample_weights(cfg: DictConfig):
    for part in ['train', 'val', 'test']:
        pheno_frames: List[pandas.DataFrame] = []
        sw_files: List[str] = []
        active_nodes = get_active_nodes(cfg)
        for node_info in active_nodes:
            dotlist = [f'node.name={node_info.name}', f'node.index={node_info.index}']
            node_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
            node_pheno = pandas.read_table(node_cfg.data.phenotype[part])
            pheno_frames.append(node_pheno)
            sw_files.append(node_cfg.data.phenotype[part] + ".sw")

        pheno = pandas.concat(pheno_frames, axis=0, ignore_index=True)
        pop_frame = pandas.read_table(cfg.data.pop_file)

        sw = calculate_sample_weights(pop_frame, pheno)
        start = 0
        for node_pheno, sw_file in zip(pheno_frames, sw_files):
            node_pheno.loc[:, 'sample_weight'] = sw[start: start + node_pheno.shape[0]]
            start = start + node_pheno.shape[0]
            node_pheno.loc[:, ['IID', 'sample_weight']].to_csv(sw_file, sep='\t')


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
    server_url = f'{gethostname()}:{cfg.server.port}'
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
    print(f'logging mlflow data to server {mlflow_url}')

    experiment = mlflow.set_experiment(cfg.experiment.name)
    if cfg.get('sample_weights', False):
        precalc_sample_weights(cfg)

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
        node_ports = get_available_ports(len(active_nodes))
        node_processes = []
        for node_info, node_port in zip(active_nodes, node_ports):
            need_gpu = node_info.resources.get('gpus', 0)
            if need_gpu:
                gpu_index += 1
                trainer_info = TrainerInfo([gpu_index], 'gpu', node_info.name, node_info.index, node_port)
            else:
                trainer_info = TrainerInfo(1, 'cpu', node_info.name, node_info.index, node_port)
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
