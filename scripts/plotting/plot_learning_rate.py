import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_learning_rate(results_paths, output_file="experiments/results/learning_rate_history.png"):
    """
    Visualiseert het verloop van de Learning Rate voor Question 4.2.
    Toont aan hoe de ReduceLROnPlateau scheduler reageert op de validatie loss.
    """
    plt.figure(figsize=(10, 6))
    
    # Zorg dat de resultatenmap bestaat
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

    for seed, path in results_paths.items():
        if not os.path.exists(path):
            print(f"Waarschuwing: {path} niet gevonden voor seed {seed}.")
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        lr_history = data.get("lr_history", [])
        
        if not lr_history:
            print(f"Geen lr_history gevonden in {path}.")
            continue

        # We plotten vanaf Epoch 1
        epochs = range(1, len(lr_history) + 1)
        
        # 'where=post' zorgt voor de karakteristieke trap-vorm
        plt.step(epochs, lr_history, where='post', label=f"Seed {seed}", marker='o', alpha=0.8)

    # Grafiek opmaak
    plt.title("Question 4.2: Learning Rate Schedule (ReduceLROnPlateau)")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.yscale('log')  # Logaritmische schaal om de decay duidelijk te zien
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig(output_file)
    print(f"Succes! De LR-grafiek is opgeslagen als: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Paden naar je 3 verschillende runs
    results = {
        "42": "experiments/results_seed42/metrics.json",
        "128": "experiments/results_seed123/metrics.json",
        "999": "experiments/results_seed999/metrics.json"
    }
    
    plot_learning_rate(results)