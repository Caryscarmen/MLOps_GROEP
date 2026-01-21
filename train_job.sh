#!/bin/bash
#SBATCH --job-name=mlp_train
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH --partition=gpu_course
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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
    echo "✅ Borrowing data from Sam's folder: $SHARED_DATA"
    SOURCE_DATA="$SHARED_DATA"
else
    echo "❌ ERROR: Could not find data in $USER_DATA or $SHARED_DATA"
    exit 1
fi

# Copy the data to the fast local SSD
cp -r "$SOURCE_DATA"/* "$LOCAL_DATA_DIR"
echo "✅ Data copy complete."

# ---------------------------------------------------------
# ---------------------------------------------------------
# 4. CONFIG MAGIC (Unique Sub-folder INSIDE Seed folder)
# ---------------------------------------------------------
TEMP_CONFIG="experiments/configs/config_local_${SLURM_JOB_ID}.yaml"

# 1. Define the Base Folder (Matches your config)
BASE_SEED_DIR="experiments/results_seed42"

# 2. Define the Unique Sub-folder for THIS specific job
# Result: experiments/results_seed42/job_18541998
NEW_SAVE_DIR="${BASE_SEED_DIR}/job_${SLURM_JOB_ID}"

# 3. Create that folder
mkdir -p "$NEW_SAVE_DIR"

echo "📝 Creating config for Job ${SLURM_JOB_ID}..."
echo "📂 Saving results INSIDE: $NEW_SAVE_DIR"

# 4. Update the temporary config
sed -e "s|data_path: .*|data_path: \"$LOCAL_DATA_DIR\"|" \
    -e "s|num_workers: .*|num_workers: 4|" \
    -e "s|save_dir: .*|save_dir: \"$NEW_SAVE_DIR\"|" \
    experiments/configs/config.yaml > "$TEMP_CONFIG"
    
# ---------------------------------------------------------
# 5. Run Training
# ---------------------------------------------------------
echo "------------------------------------------------"
echo "🕒 Training START time: $(date)"
echo "🆔 Job ID: $SLURM_JOB_ID"
echo "------------------------------------------------"

python -u experiments/train.py --config "$TEMP_CONFIG"

echo "------------------------------------------------"
echo "🕒 Training END time: $(date)"
echo "------------------------------------------------"

# 6. Cleanup
rm "$TEMP_CONFIG"