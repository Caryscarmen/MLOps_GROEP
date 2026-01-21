import argparse
import json
from pathlib import Path
from typing import Optional, Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True, help="Pad naar metrics.json")
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args()

def load_data(file_path: Path) -> Optional[Dict[str, Any]]:
    """Laadt de JSON direct als een dictionary om lengteverschillen te omzeilen."""
    if not file_path.exists():
        print(f"Error: {file_path} niet gevonden.")
        return None
    
    with open(file_path, "r") as f:
        return json.load(f)

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

def plot_metrics(data: Dict[str, Any], output_path: Optional[Path]):
    """Genereert plots waarbij batch- en epoch-data gescheiden worden."""
    if not data: return

    # Maak figuur met subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss & Metrics (Per Epoch)
    epochs = range(1, len(data.get('train_loss', [])) + 1)
    
    if 'train_loss' in data and 'val_loss' in data:
        axes[0, 0].plot(epochs, data['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(epochs, data['val_loss'], label='Val Loss', marker='x')
        axes[0, 0].set_title("Question 5.2: Loss (per Epoch)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
    
    # 2. ROC-AUC & F2 (Per Epoch)
    if 'val_roc_auc' in data:
        axes[0, 1].plot(epochs, data['val_roc_auc'], color='orange', label='Val ROC-AUC', marker='s')
        axes[0, 1].set_title("Metric 1: ROC-AUC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
    
    if 'val_f2_score' in data:
        axes[1, 0].plot(epochs, data['val_f2_score'], color='green', label='Val F2 (β=2)', marker='^')
        axes[1, 0].set_title("Metric 2: F2-Score")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()

    # 3. Gradient Norms (Per Batch)
    if 'grad_norms' in data:
        batches = range(1, len(data['grad_norms']) + 1)
        axes[1, 1].plot(batches, data['grad_norms'], color='purple', alpha=0.6)
        axes[1, 1].set_title("Question 4a: Gradient Norms (per Batch)")
        axes[1, 1].set_xlabel("Batch Step")
        axes[1, 1].set_ylabel("L2 Norm")

    plt.tight_layout()

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "experiment_summary.png"
        plt.savefig(save_path, dpi=300)
        print(f"Plot succesvol opgeslagen in: {save_path}")

    plt.show()

def main():
    args = parse_args()
    setup_style()
    data = load_data(args.input_csv)
    plot_metrics(data, args.output_dir)

if __name__ == "__main__":
    main()