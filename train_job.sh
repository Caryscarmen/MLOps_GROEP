#!/bin/bash
#SBATCH --job-name=mlp_train
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH --partition=gpu_course
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# 1. Load Modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

# 2. Activate Venv
source venv/bin/activate

# ---------------------------------------------------------
# 3. SMART DATA SETUP
# ---------------------------------------------------------
echo "🚀 Setting up fast local storage..."
export LOCAL_DATA_DIR="$TMPDIR/data"
mkdir -p "$LOCAL_DATA_DIR"

# PATH 1: The user's own folder (Best case)
USER_DATA="/scratch-shared/$USER/surfdrive"
# PATH 2: Sam's shared folder (Fallback)
SHARED_DATA="/scratch-shared/scur2395/surfdrive"

if [ -d "$USER_DATA" ]; then
    echo "✅ Found data in your own folder: $USER_DATA"
    SOURCE_DATA="$USER_DATA"
elif [ -d "$SHARED_DATA" ]; then
    echo "⚠️ Your data folder is empty."
    echo "✅ borrowing data from Sam's folder: $SHARED_DATA"
    SOURCE_DATA="$SHARED_DATA"
else
    echo "❌ ERROR: Could not find data in $USER_DATA or $SHARED_DATA"
    exit 1
fi

# Copy the data to the fast local SSD
cp -r "$SOURCE_DATA"/* "$LOCAL_DATA_DIR"
echo "✅ Data copy complete."

# ---------------------------------------------------------
# 4. CONFIG MAGIC
# ---------------------------------------------------------
TEMP_CONFIG="experiments/configs/config_local_${SLURM_JOB_ID}.yaml"

sed -e "s|data_path: .*|data_path: \"$LOCAL_DATA_DIR\"|" \
    -e "s|num_workers: .*|num_workers: 4|" \
    experiments/configs/config.yaml > "$TEMP_CONFIG"

# 5. Run Training
echo "Job ID: $SLURM_JOB_ID"
python -u experiments/train.py --config "$TEMP_CONFIG"

# 6. Cleanup
rm "$TEMP_CONFIG"