#!/bin/bash
#SBATCH --job-name=gpu_bench_small
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-5%1
#SBATCH --output=gpu_bench_small_%A_%a.out

module purge
module load 2023
module load Python/3.11.3

source ~/MLOps_GROEP/venv/bin/activate

BATCH_SIZES=(8 16 32 128 256 512)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "JobID: $SLURM_JOB_ID  ArrayJobID: $SLURM_ARRAY_JOB_ID  TaskID: $SLURM_ARRAY_TASK_ID"
echo "Running SMALL model throughput benchmark with batch_size=$BS"
echo "GPU info:"
nvidia-smi

python scripts/throughput_gpu_bench.py --batch_size $BS --device cuda --model small