#!/bin/bash
#SBATCH --job-name=gpu_bench_small
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-5%1
#SBATCH --output=gpu_bench_small_%A_%a.out

# 1. Omgeving klaarmaken
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# 2. Training starten (Vraag 5)
# Dit maakt je checkpoints en config aan
TRAIN_SCRIPT=$(find . -name "train.py" | head -n 1)

# 3. Analyse starten (Vraag 6)
# Dit script pakt automatisch de resultaten van de stap hierboven
python $TRAIN_SCRIPT --config experiments/configs/base_config.yaml