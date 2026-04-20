#!/bin/bash

#SBATCH --job-name=brain_tumor_seg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j/slurm_%j.out
#SBATCH --error=slurm_logs/%j/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=de807845@ucf.edu

# Usage:
#   sbatch scripts/hpc_job.sh dynunet
#   sbatch scripts/hpc_job.sh swinunetr

MODEL=${1:-dynunet}
CONFIG="configs/${MODEL}_hpc.yaml"

mkdir -p slurm_logs/$SLURM_JOB_ID

echo "Job started at $(date)" >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.out
echo "Model: $MODEL  |  Config: $CONFIG" >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.out

# load modules
module load anaconda/anaconda-2023.09 cuda/cuda-12.6.0
echo "Following modules loaded..." >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.err
module list >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.err

# activate env
conda activate medical
echo "Virtual environment activated"
unset LD_LIBRARY_PATH

# traverse to project dir
cd /lustre/fs1/home/de807845/med_img_computing/3D-BrainTumor-Seg/
echo "Current directory: $(pwd)" >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.out

# run
echo "Starting training: $MODEL"
python run.py --config $CONFIG

# deactivate
conda deactivate
echo "Virtual environment deactivated"
echo "Run completed!"
