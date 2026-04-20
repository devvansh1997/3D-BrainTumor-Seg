#!/bin/bash
#SBATCH --job-name=brain-tumor-seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm_%j_%x.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=devansh.sharma1997@gmail.com   # update if different on HPC

# Usage:
#   sbatch scripts/hpc_job.sh dynunet
#   sbatch scripts/hpc_job.sh swinunetr

MODEL=${1:-dynunet}
CONFIG="configs/${MODEL}_hpc.yaml"

echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Model:     $MODEL"
echo "Config:    $CONFIG"
echo "Started:   $(date)"
echo "=============================="

# Activate your environment — update the path below
source ~/miniconda3/etc/profile.d/conda.sh   # or: module load anaconda
conda activate work_env                       # or whatever your env is called

cd $SLURM_SUBMIT_DIR

python run.py --config $CONFIG

echo "Finished: $(date)"
