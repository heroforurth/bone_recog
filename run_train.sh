#!/bin/bash
#SBATCH --job-name=bone_seg_train
#SBATCH --output=bone_seg_train_%j.out
#SBATCH --error=bone_seg_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# 1. Load Environment
# Adjust the path to your miniconda installation if different
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bone_ai

# Check dependencies
echo "Checking environment..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import sam3; print('SAM 3 is available')"

# 2. Generate Masks (Teacher)
echo "----------------------------------------"
echo "Step 1: Generating Masks with SAM 3..."
echo "----------------------------------------"
python3 01_generate_masks.py

# 3. Train U-Net (Student)
echo "----------------------------------------"
echo "Step 2: Training U-Net..."
echo "----------------------------------------"
python3 02_train_unet.py

echo "Job Finished!"
