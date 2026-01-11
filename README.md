# MLOps UvA Bachelor AI Course: Medical Image Classification

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

This repository implements a reliable, reproducible MLOps system for medical image classification using the PatchCamelyon (PCAM) dataset. [cite_start]It moves away from interactive notebooks toward structured Python scripts optimized for High-Performance Computing (HPC) environments[cite: 11, 1757].

---

## 🚀 Setup & Installation (Snellius HPC)

[cite_start]On Snellius, software is managed via Environment Modules[cite: 467, 1443]. You must load these before setting up your virtual environment.

### 1. Load Environment Modules
```bash
# Clean existing modules and load the 2025 software stack
module purge
module load 2025 [cite: 471, 1454]
module load Python/3.13.1-GCCcore-14.2.0 [cite: 472, 1458]
module load matplotlib/3.10.3-gfbf-2025a [cite: 473, 1505]

# Create and activate the virtual environment in the project root
python -m venv venv
source venv/bin/activate

# Install the package in "Editable" mode
pip install -e .

# Install PyTorch with CUDA (GPU) support
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Initialize pre-commit hooks for automated linting/formatting
pre-commit install
---

## 📂 Project Structure

```text
.
├── src/ml_core/          # THE LIBRARY: Modular, tested, reusable code 
│   ├── data/             # PCAM Dataset and HDF5 lazy-loading logic 
│   ├── models/           # Neural Network architectures (MLP) 
│   ├── solver/           # Training loops and optimization logic 
│   └── utils/            # Loggers and health metrics (e.g., Gradient Norms) 
├── experiments/          # THE LABORATORY: Research code and play [cite: 753]
│   ├── configs/          # YAML files for hyperparameters (No hardcoding!) 
│   ├── results/          # Auto-generated checkpoints and logs [cite: 735]
│   └── train.py          # Entry point for training runs 
├── tests/                # Quality Assurance: Unit tests for QA 
├── pyproject.toml        # Modern config center for tools (Ruff, Pytest) 
└── .gitignore            # Prevents large data (H5) and venv from being tracked 
```
