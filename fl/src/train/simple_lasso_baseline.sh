#!/bin/bash 
#SBATCH --job-name ukb_fl_baseline # good manners rule 
#SBATCH --partition gpu_devel # one of gpu, gpu_devel 
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 0:30:00 # hh:mm:ss, walltime 
#SBATCH --mem 6000
#SBATCH --cpus-per-task 1

singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER \
../../image.sif /trinity/home/$USER/.conda/fl/bin/python -u -m train.simple_lasso_baseline "$@"