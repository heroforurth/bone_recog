#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=ae_bone_train
#SBATCH --output=ae_train_%j.out
#SBATCH --error=ae_train_%j.err

# Load Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bone_ai

# Debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Run Training
python 04_train_ae.py
