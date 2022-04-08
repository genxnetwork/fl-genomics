#!/bin/env python

#SBATCH --job-name=multiprocess
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1

import multiprocessing
import sys
import os
from fl.node_process import Node
from omegaconf import DictConfig, OmegaConf


# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)

sys.path.append(os.getcwd()) 

'''
# get number of cpus available to job

'''

if __name__ == '__main__':
    print(f'I am in main')
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    queue = multiprocessing.Queue()
    cfg_path = 'src/fl/configs/mlp.yaml'
    node1 = Node(0, None, queue, cfg_path)
    node2 = Node(1, None, queue, cfg_path)
    # create pool of ncpus workers
    node1.start()
    node2.start()
    node1.join()
    node2.join()
    print(f'Nodes are finished')
