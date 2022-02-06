#!/bin/bash 
#SBATCH --job-name ukb_fl_server # good manners rule 
#SBATCH --partition gpu_devel # one of gpu, gpu_devel 
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 0:40:00 # hh:mm:ss, walltime 
#SBATCH --mem 2000
#SBATCH --cpus-per-task 1
#SBATCH --export ALL

# mlflow credentials are needed for saving artifacts in s3 bucket
set -o allexport
source /trinity/home/$USER/.mlflow/credentials
set +o allexport

cd /trinity/home/$USER/uk-biobank/fl/src
# remove old parent run id because sometimes clients start faster than server and use old parent run id
rm .mlflow_parent_run_id

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
../../image.sif /trinity/home/$USER/.conda/fl/bin/python -u -m train.server "$@"