#!/bin/bash
## Job Name
#SBATCH --job-name=autoreview_train
## Allocation Definition
#SBATCH --account=ckpt
#SBATCH --partition=stf-ckpt
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


## Memory per node
#SBATCH --mem=120G
## Specify the working directory for this job
## Set this on command line with --chdir $AUTOREVIEW_PROJECT_DIR

module load singularity
export SPARK_LOCAL_DIRS=/gscratch/stf/jporteno/autoreview-singularity/spark_local_dir
singularity exec autoreview-singularity_20191218.sif python3 /scripts/train_models.py --outdir data/reviews_wos/WOS-000342347400001/seed001/ --random-seed 1 --debug >& data/reviews_wos/WOS-000342347400001/seed001/train.log


