#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=jupyter_lab
#SBATCH --output=jupyter_%j.log
#SBATCH --error=jupyter_%j.log

# Load Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bone_ai

# Get a random port between 8000 and 9999
PORT=$((8000 + RANDOM % 2000))
NODE=$(hostname)

echo "------------------------------------------------------------------"
echo "Starting Jupyter Lab on node $NODE port $PORT"
echo "------------------------------------------------------------------"
echo "1. On your local machine, run this SSH tunnel command:"
echo ""
echo "   ssh -L $PORT:$NODE:$PORT 67070309@dgx.cs.ait.ac.th"
echo ""
echo "2. Open your browser and go to:"
echo ""
echo "   http://localhost:$PORT"
echo ""
echo "------------------------------------------------------------------"

# Run Jupyter Lab
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
