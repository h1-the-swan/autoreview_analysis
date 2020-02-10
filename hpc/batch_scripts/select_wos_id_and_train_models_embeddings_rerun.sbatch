#!/bin/bash
## Job Name
#SBATCH --job-name=autoreview_train_embeddings
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
 
#SBATCH --open-mode=append

##turn on e-mail notification

#SBATCH --mail-type=ALL

# set --mail-user on command line with $EMAIL
###SBATCH --mail-user=$EMAIL


## Memory per node
#SBATCH --mem=120G
## Specify the working directory for this job
## Set this on command line with --chdir $AUTOREVIEW_PROJECT_DIR

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

module load singularity
# 1st command line argument is the path to the TSV file containing the list of review paper IDs
# 2nd command line argument is an integer for which ID to select. Should start at zero and end at (number of IDs - 1)
# 3rd command line argument is the output directory (which should already exist): e.g. data/hpc/reviews_wos_collect_nospark_20191223
# 4th command line argument is an integer specifying the features/transformers to use (see the code for train_models.py)
# 5th command line argument is the path to the directory containing embeddings for titles
singularity exec autoreview-singularity_20200121_2.sif python3 select_review_wos_id_and_train_models.py $1 $2 $3 $4 --no-header --no-save-best --embeddings $5 --force-rerun --debug



