#!/bin/bash
## Job Name
#SBATCH --job-name=autoreview_collect_nospark
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
#SBATCH --chdir=/gscratch/stf/jporteno/autoreview_analysis

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

module load singularity
# 1st command line argument is the path to the file containing the IDs
# 2nd command line argument is an integer for which ID to select. Should be between zero and end at (number of IDs - 1)
# 3rd command line argument is the output directory (which should already exist): e.g. data/hpc/reviews_wos_collect_nospark_20191223
# 4th command line argument is the seed set sample size to use
singularity exec autoreview-singularity_20200121_2.sif python3 select_review_wos_id_and_collect.py $1 $2 $3 --no-header --citations ../wos_201912/wos_citations_cleaned_20200105_parquet/ --papers ../wos_201912/wos_papers_20191223_parquet/ --sample-size $4 --no-spark --debug


