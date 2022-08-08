#!/bin/env python

#SBATCH --job-name=multiprocess
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1

import torch


print(f'torch imported')

print('is cuda avaiable: ', torch.cuda.is_available())