#!/bin/bash
#SBATCH --account=gpuuva069
#SBATCH --partition=rome          # De CPU partitie (werkt altijd)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00           # Kortere tijd voor een test

# 1. Omgeving klaarmaken
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# 2. Start de training/analyse
# We voegen '--epochs 1' toe zodat we snel zien of het werkt
python train.py --config experiments/configs/base_config.yaml --epochs 1