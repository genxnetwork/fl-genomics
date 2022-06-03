#!/bin/bash

#SBATCH --job-name=multiprocess
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --error=logs/multiprocess_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 26000

set -o allexport
source ~/.mlflow/credentials
PYTHONPATH=`pwd`/src
set +o allexport

python -u src/fl/runner.py "$@"