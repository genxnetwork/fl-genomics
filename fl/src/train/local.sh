#!/bin/bash 
#SBATCH --job-name ukb_fl_local# good manners rule 
#SBATCH --partition gpu_devel # one of gpu, gpu_devel 
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 3:00:00 # hh:mm:ss, walltime 
#SBATCH --mem 32000
#SBATCH --cpus-per-task 4 
#SBATCH --gpus 1
#SBATCH --export ALL

# mlflow credentials are needed for saving artifacts in s3 bucket
set -o allexport
source /trinity/home/$USER/.mlflow/credentials

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
../../image.sif /trinity/home/$USER/.conda/envs/fl/bin/python -u -m train.local "$@"
