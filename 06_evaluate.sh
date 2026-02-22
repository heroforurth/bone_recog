#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=bone_eval
#SBATCH --output=bone_eval_%j.out
#SBATCH --error=bone_eval_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bone_ai

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

python 06_evaluate_report.py
