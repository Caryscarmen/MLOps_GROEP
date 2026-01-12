#!/bin/bash
#SBATCH --job-name=mlp_train
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH --partition=gpu_course
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# 1. Load Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

# 2. Activate Venv
source venv/bin/activate

# 3. Debugging Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Run Training
python experiments/train.py --config experiments/configs/config.yaml
