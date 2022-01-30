#!/bin/bash 
#SBATCH --job-name gwas_analysis # good manners rule 
#SBATCH --partition gpu_devel # or gpu_small
#SBATCH --nodes 1 # amount of nodes allocated 
#SBATCH --time 00:15:00# hh:mm:ss, walltime (less requested time -> 
#SBATCH --mem 4000
#SBATCH --cpus-per-task 1

module load python/mambaforge3
cd /trinity/home/$USER/uk-biobank/dimred/src
singularity exec --nv -B /gpfs/gpfs0/ukb_data,/gpfs/gpfs0/$USER ../../image.sif /trinity/home/a.medvedev/.conda/fl/bin/python -m gwas.analysis "$@"