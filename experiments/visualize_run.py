import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Pad naar jouw metrics bestand
JSON_PATH = "experiments/results/metrics.json" # Pas aan indien het ergens anders staat
OUTPUT_DIR = "experiments/results"

def create_plots():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} niet gevonden!")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # --- Plot 1: Gradient Norms (Question 4a) ---
    plt.figure(figsize=(10, 5))
    grad_norms = data.get("grad_norms", [])
    plt.plot(grad_norms, color='blue', alpha=0.7)
    plt.title("Question 4a: Gradient Norms per Batch")
    plt.xlabel("Batch Iteration")
    plt.ylabel("L2 Norm")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/gradient_norms.png")
    print(f"Opgeslagen: {OUTPUT_DIR}/gradient_norms.png")
    plt.close()

    # --- Plot 2: Learning Rate (Question 4b) ---
    plt.figure(figsize=(10, 5))
    lr_history = data.get("lr_history", [])
    epochs = range(1, len(lr_history) + 1)
    plt.step(epochs, lr_history, where='post', marker='o')
    plt.title("Question 4b: Learning Rate History")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{OUTPUT_DIR}/lr_history.png")
    print(f"Opgeslagen: {OUTPUT_DIR}/lr_history.png")
    plt.close()

if __name__ == "__main__":
    create_plots()