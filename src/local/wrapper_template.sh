#!/bin/bash
#SBATCH --job-name ukb_local_experiment # good manners rule 
#SBATCH --partition $PARTITION_TYPE # one of gpu, gpu_devel
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 6:00:00 # hh:mm:ss, walltime 
#SBATCH --mem $MEM 
#SBATCH --cpus-per-task 1
#SBATCH --export ALL
#gpu_placeholder

set -o allexport
source /trinity/home/$USER/.mlflow/credentials
set +o allexport

cd /trinity/home/$USER/uk-biobank/src

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
  ../image.sif /trinity/home/$USER/.conda/envs/fl/bin/python -u -m local.experiment "$@"
