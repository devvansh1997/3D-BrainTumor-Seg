#!/bin/bash

#SBATCH --job-name=brain_tumor_viz
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=gpu80
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j/slurm_%j.out
#SBATCH --error=slurm_logs/%j/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=de807845@ucf.edu

# Usage:
#   sbatch scripts/hpc_viz.sh

mkdir -p slurm_logs/$SLURM_JOB_ID

echo "Job started at $(date)" >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.out

# load modules
module load anaconda/anaconda-2023.09 cuda/cuda-12.6.0
module list >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.err

# activate env
conda activate medical
echo "Virtual environment activated"
unset LD_LIBRARY_PATH

# traverse to project dir
cd /lustre/fs1/home/de807845/med_img_computing/3D-BrainTumor-Seg/
echo "Current directory: $(pwd)" >> slurm_logs/$SLURM_JOB_ID/slurm_${SLURM_JOB_ID}.out

# run viz
echo "Generating segmentation visualisations..."
python scripts/visualize_seg.py configs/dynunet_hpc.yaml --num 3

conda deactivate
echo "Run completed!"
