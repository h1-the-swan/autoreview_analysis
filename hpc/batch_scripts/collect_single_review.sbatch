#!/bin/bash
## Job Name
#SBATCH --job-name=autoreview_collect_single
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
# 1st command line argument is the output directory (which should already exist): e.g. data/hpc/reviews_wos_collect_nospark_20191223
# 2nd command line argument is the paper ID
# 3rd command line argument is the seed set sample size to use
singularity exec autoreview-singularity_20200103.sif python3 get_references_and_collect.py $1 $2 --citations ../wos_201912/wos_citations_cleaned_20200105_parquet/ --papers ../wos_201912/wos_papers_20191223_parquet/ --sample-size $3 --no-spark --debug



