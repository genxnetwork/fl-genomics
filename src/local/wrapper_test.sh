#!/bin/bash
#SBATCH --job-name ukb_local_test # good manners rule 
#SBATCH --partition gpu # one of gpu, gpu_devel
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 1:00:00 # hh:mm:ss, walltime 
#SBATCH --mem 16000 
#SBATCH --cpus-per-task 4
#SBATCH --export ALL
#SBATCH --gpus 1 

set -o allexport
source /trinity/home/$USER/.mlflow/credentials
set +o allexport

cd /trinity/home/$USER/uk-biobank/src

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
  ../image.sif /trinity/home/$USER/.conda/envs/fl/bin/python -u -m local.cross_node_model_eval +experiment=centralized_metrics_local "$@"
