#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=bone_seg_vis
#SBATCH --output=bone_seg_vis_%j.out
#SBATCH --error=bone_seg_vis_%j.err

# Load Environment (Corrected based on run_train.sh)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bone_ai

# Debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "PWD: $(pwd)"

# Run visualization
python 03_visualize_results.py
