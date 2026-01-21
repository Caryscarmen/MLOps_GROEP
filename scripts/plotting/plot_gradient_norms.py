import json
import matplotlib.pyplot as plt
import os
import glob 

def plot_gradients(results_paths, output_file="gradient_norms_comparison.png"):
    """
    Genereert een vergelijkingsplot van de Gradient Norms voor verschillende seeds.
    Dit helpt bij het beantwoorden van Question 4.1 over interne dynamiek.
    """
    plt.figure(figsize=(12, 6))
    
    # Check if we have any data to plot
    if not results_paths:
        print("❌ Geen resultaten gevonden om te plotten. Check of de jobs klaar zijn.")
        return

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
        # We pakken de eerste 1000 stappen zodat de grafiek leesbaar blijft
        plt.plot(grad_norms[:1000], label=f"{seed}", alpha=0.7, linewidth=1)

    # Grafiek opmaak
    plt.title("Question 4.1: Global Gradient Norm per Step (Batch-level Granularity)")
    plt.xlabel("Training Steps (Batches)")
    plt.ylabel("L2 Gradient Norm")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Maak de map aan als hij niet bestaat
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Sla de grafiek op voor je verslag
    plt.savefig(output_file)
    print(f"✅ Succes! De grafiek is opgeslagen als: {output_file}")
    # plt.show() # Uitgezet omdat je op een server zit (geen scherm)

if __name__ == "__main__":
    def get_latest_metrics(base_folder):
        # Zoek naar jobs zoals
        pattern = os.path.join(base_folder, "job_*", "metrics.json")
        found_files = glob.glob(pattern)
        
        if not found_files:
            return None
        
        # Pak de nieuwste
        return max(found_files, key=os.path.getmtime)

    # 1. Definieer de mappen waar de seeds staan
    seed_folders = {
        "Seed 42":  "experiments/results_seed42",
        "Seed 128": "experiments/results_seed128", 
        "Seed 999": "experiments/results_seed999"  # Of 2025, wat je ook gekozen hebt
    }
    
    # 2. Zoek de echte paths
    real_results = {}
    print("🔍 Zoeken naar metrics.json bestanden...")
    
    for seed_name, folder in seed_folders.items():
        path = get_latest_metrics(folder)
        if path:
            print(f"   Gevonden voor {seed_name}: {path}")
            real_results[seed_name] = path
        else:
            print(f"   ⚠️  Nog geen resultaat gevonden in {folder} (Job loopt nog?)")

    plot_gradients(real_results, output_file="experiments/results/gradient_norms_comparison.png")