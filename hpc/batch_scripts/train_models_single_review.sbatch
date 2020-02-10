#!/bin/bash
## Job Name
#SBATCH --job-name=autoreview_train_single
## Allocation Definition
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node
#SBATCH --ntasks-per-node=28
## Walltime
#SBATCH --time=8:00:00
# E-mail Notification, see man sbatch for options
 

##turn on e-mail notification

#SBATCH --mail-type=ALL

# set --mail-user on command line with $EMAIL
###SBATCH --mail-user=$EMAIL

#SBATCH --open-mode=append

## Memory per node
#SBATCH --mem=120G
## Specify the working directory for this job
## Set this on command line with --chdir $AUTOREVIEW_PROJECT_DIR

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

module load singularity
# 1st command line argument is the input/output directory, which should already exist and contain a 'paper_info.json' file and subdirectories for seed/candidate paper sets
# 2nd command line argument is an integer specifying the features/transformers to use (see the code for train_models.py)
singularity exec autoreview-singularity_20200121_2.sif python3 train_models.py $1 $2 --force-rerun --debug

