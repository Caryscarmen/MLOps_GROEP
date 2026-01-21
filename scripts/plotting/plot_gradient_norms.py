import json
import matplotlib.pyplot as plt
import os

def plot_gradients(results_paths, output_file="gradient_norms_comparison.png"):
    """
    Genereert een vergelijkingsplot van de Gradient Norms voor verschillende seeds.
    Dit helpt bij het beantwoorden van Question 4.1 over interne dynamiek.
    """
    plt.figure(figsize=(12, 6))
    
    for seed, path in results_paths.items():
        if not os.path.exists(path):
            print(f"Waarschuwing: {path} niet gevonden. Sla seed {seed} over.")
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        # Haal de gradient norms op die we per batch hebben opgeslagen
        grad_norms = data.get("grad_norms", [])
        
        if not grad_norms:
            print(f"Geen gradient_norms gevonden in {path}.")
            continue

        # Plot de volledige lijst voor batch-level granulariteit
        plt.plot(grad_norms, label=f"Seed {seed}", alpha=0.7)

    # Grafiek opmaak
    plt.title("Question 4.1: Global Gradient Norm per Step (Batch-level Granularity)")
    plt.xlabel("Training Steps (Batches)")
    plt.ylabel("L2 Gradient Norm")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Sla de grafiek op voor je verslag
    plt.savefig(output_file)
    print(f"Succes! De grafiek is opgeslagen als: {output_file}")
    plt.show()

if __name__ == "__main__":
    results = {
        "Seed 42":  "experiments/results_seed42/metrics.json",
        "Seed 128": "experiments/results_seed123/metrics.json",
        "Seed 999": "experiments/results_seed999/metrics.json"
    }
    
    # Pas het output_file pad hier aan:
    plot_gradients(results, output_file="experiments/results/gradient_norms_comparison.png")

