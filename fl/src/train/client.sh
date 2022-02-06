#!/bin/bash 
#SBATCH --job-name ukb_fl_client # good manners rule 
#SBATCH --partition gpu_devel # one of gpu, gpu_devel 
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 0:40:00 # hh:mm:ss, walltime 
#SBATCH --mem 6000
#SBATCH --cpus-per-task 2
#SBATCH --gpus 1
#SBATCH --export ALL

# mlflow credentials are needed for saving artifacts in s3 bucket
set -o allexport
source /trinity/home/$USER/.mlflow/credentials

# these two files are written by server 
# client uses them to create child mlflow run id and to connect to server
cd /trinity/home/$USER/uk-biobank/fl/src
source .mlflow_parent_run_id
source .server_hostname
set +o allexport

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
../../image.sif /trinity/home/$USER/.conda/fl/bin/python -u -m train.client "$@"