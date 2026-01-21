# PCAM MLOps Project - Group 38

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

This repository implements a reproducible MLOps pipeline for medical image classification using the PatchCamelyon (PCAM) dataset. It features structured experiment tracking, reproducible configuration management, and optimized data loading for the Snellius HPC environment.

---

## 🚀 Setup & Installation (Snellius HPC)

To ensure reproducibility, this project relies on the specific software stack provided by Snellius modules.

### 1. Load Environment Modules
Run these commands to load the correct Python and system dependencies:

```bash
# 1. Clean environment and load the 2025 stack
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

# 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate

# 3. Install Project in Editable Mode
pip install -e .

# 4. Install PyTorch (Compatible with Snellius GPUs)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

---

## 📂 Data Setup

The model expects the PCAM HDF5 files. 

**On Snellius:**
The data is typically available at `/scratch-shared/scur2395/surfdrive`.
You do not need to move files manually if you configure your YAML file correctly.

If running locally, place the files in a folder named `data/`:
- `camelyonpatch_level_2_split_train_x.h5`
- `camelyonpatch_level_2_split_train_y.h5`
- *(Repeat for valid/test splits)*

---

## 🏆 Reproducing the Champion Model

Our best performing model (**Job ID 18549665**) achieved an **F2-Score of 0.7654** using the following hyperparameters:
* **Batch Size:** 128
* **Learning Rate:** 0.001
* **Hidden Units:** 999
* **Seed:** 128

### To Reproduce Training:
1. Ensure `configs/champion.yaml` exists with the settings above.
2. Run the training script:
```bash
python experiments/train.py --config configs/champion.yaml
```

*Expected Result:* - **Validation ROC-AUC:** ~0.77
- **Validation F2-Score:** ~0.76 (at Epoch 2)

---

## 🔮 Inference

To run predictions using the saved model checkpoint on new data:

1. Ensure your trained model is saved (e.g., `models/best_model.pt`).
2. Run the inference script:

```bash
python inference.py
```

*Note:* This script will load a sample image, apply the necessary preprocessing (normalization), and output the tumor probability.

---

## 📂 Project Structure

```text
.
├── src/ml_core/          # Source code (Dataset, Model, Trainer)
├── experiments/          # Scripts for Training and Plotting
│   ├── configs/          # Configuration files (YAML)
│   ├── results/          # Output logs and checkpoints
│   └── train.py          # Main training entry point
├── tests/                # Unit tests
├── inference.py          # Script for single-image prediction
├── pyproject.toml        # Project dependencies
└── README.md             # Project documentation
```