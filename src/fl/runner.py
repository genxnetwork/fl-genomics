#!/bin/env python

#SBATCH --job-name=multiprocess
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem 26000

import multiprocessing
import sys
import os
from socket import gethostname
from omegaconf import DictConfig, OmegaConf
import mlflow
import hashlib

from fl.node_process import MlflowInfo, Node
from fl.server_process import Server


# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)

sys.path.append(os.getcwd()) 


def get_cfg_hash(cfg: DictConfig):
    yaml_representation = OmegaConf.to_yaml(cfg)
    return hashlib.sha224(yaml_representation.encode()).hexdigest()


NODE_RESOURCES = {
    '0': {'partition': 'cpu', 'mem_mb': 8000, 'gpus': 0},
    '1': {'partition': 'cpu', 'mem_mb': 8000, 'gpus': 0},
    '2': {'partition': 'gpu', 'mem_mb': 64000, 'gpus': 1},
    '3': {'partition': 'gpu', 'mem_mb': 36000, 'gpus': 1},
    '4': {'partition': 'gpu', 'mem_mb': 24000, 'gpus': 1},
    '5': {'partition': 'gpu', 'mem_mb': 16000, 'gpus': 1},
    '6': {'partition': 'cpu', 'mem_mb': 8000, 'gpus': 0},
    '7': {'partition': 'cpu', 'mem_mb': 8000, 'gpus': 0},
}


if __name__ == '__main__':
    print(f'I am in main')
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    queue = multiprocessing.Queue()
    cfg_path = 'src/fl/configs/mlp.yaml'
    server_url = f'{gethostname()}:8080'

    cfg = OmegaConf.load(cfg_path)
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

        info = MlflowInfo(experiment.experiment_id, run.info.run_id)
        server = Server(queue, params_hash, cfg_path)
        node1 = Node(0, server_url, info, queue, cfg_path, -1)
        node2 = Node(1, server_url, info, queue, cfg_path, -1)
        server.start()
        # create pool of ncpus workers
        print(f'starting node1')
        node1.start()
        print(f'starting node 2')
        node2.start()
        server.join()
        node1.join()
        node2.join()
        print(f'Nodes are finished')
