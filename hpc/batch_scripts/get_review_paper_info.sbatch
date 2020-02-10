#!/bin/bash
## Job Name
#SBATCH --job-name=paper_info
## Allocation Definition
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node
#SBATCH --ntasks-per-node=28
## Walltime
#SBATCH --time=4:00:00
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
export SPARK_LOCAL_DIRS=./spark_local_dir/spark_local_dir_$SLURM_JOB_ID
rm -rf $SPARK_LOCAL_DIRS
mkdir $SPARK_LOCAL_DIRS
singularity exec autoreview-singularity_20200103.sif python3 get_paper_info_multiple_papers.py data/reviews_wos_ids_midsize_sample_20200104.tsv data/hpc/reviews_wos_collect_nospark_20200104/seed_papers_size_50/ --papers ../wos_201912/wos_papers_20191223_parquet/ --save data/reviews_wos_ids_midsize_sample_20200104_paperinfo.tsv --no-header --debug

