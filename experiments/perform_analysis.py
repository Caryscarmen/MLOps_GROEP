import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Nodig voor Snellius
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from sklearn.metrics import fbeta_score

# Zorg dat je dit script runt vanuit de hoofdmap (MLOps_GROEP)
from ml_core.data.loader import get_dataloaders
from ml_core.models.mlp import MLP

def perform_error_analysis(model, test_loader, device, output_dir):
    print("Bezig met verzamelen van voorspellingen op de testset...")
    model.eval()
    all_preds, all_labels, all_images, intensities = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Voorkom fouten bij batch size 1
            all_preds.extend(preds if preds.shape else [preds.item()])
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())
            intensities.extend(images.mean(dim=(1, 2, 3)).cpu().numpy())

    all_preds, all_labels, all_images = np.array(all_preds), np.array(all_labels), np.array(all_images)
    intensities = np.array(intensities)

    # (a) Qualitative Analysis: Top 5 FP en FN
    fps = np.where((all_preds == 1) & (all_labels == 0))[0]
    fns = np.where((all_preds == 0) & (all_labels == 1))[0]

    for idx_list, title, filename in [(fps, "False Positives", "fps.png"), (fns, "False Negatives", "fns.png")]:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(idx_list[:5]):
            plt.subplot(1, 5, i+1)
            img = np.transpose(all_images[idx], (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min()) # Normaliseer voor plot
            plt.imshow(img)
            plt.axis('off')
        plt.suptitle(f"Question 6a: {title}")
        plt.savefig(output_dir / filename)
        plt.close()

    # (b) Quantitative Slicing: Dark Slice (Bottom 10% intensity)
    thresh = np.percentile(intensities, 10)
    dark_idx = np.where(intensities <= thresh)[0]
    slice_f2 = fbeta_score(all_labels[dark_idx], all_preds[dark_idx], beta=2.0, zero_division=0)
    global_f2 = fbeta_score(all_labels, all_preds, beta=2.0)

    with open(output_dir / "slicing_results.txt", "w") as f:
        f.write(f"Global F2: {global_f2:.4f}\n")
        f.write(f"Dark Slice F2: {slice_f2:.4f}\n")
        f.write(f"Performance Gap: {global_f2 - slice_f2:.4f}\n")
    
    print(f"Klaar! Resultaten staan in: {output_dir}")
if __name__ == "__main__":
    # Gebruik de map die je net liet zien
    RUN_DIR = Path("experiments/results/mlp_baseline")
    
    # We pakken je laatste checkpoint
    CHECKPOINT_PATH = RUN_DIR / "checkpoint_epoch_3.pt"
    CONFIG_PATH = RUN_DIR / "config.yaml"
    OUTPUT_DIR = RUN_DIR / "analysis"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Laad Config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. HERSTEL HET DATA PAD (Belangrijk!)
    # We negeren het scratch-pad en wijzen naar je eigen data map
    cfg['data']['data_path'] = "/scratch-shared/scur2395/surfdrive"

    # 3. Laad Data
    print("Loading test data...")
    from ml_core.data.loader import get_dataloaders
    _, _, test_loader = get_dataloaders(cfg)

    # 4. Initialiseer Model
    print("Initializing model...")
    from ml_core.models.mlp import MLP
    model = MLP(
        input_shape=cfg['model']['input_shape'], 
        num_classes=cfg['model']['num_classes'],
        hidden_units=cfg['model']['hidden_units']
    ).to(device)

    # 5. Laad gewichten
    print(f"Loading weights from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 6. Run Analyse
    OUTPUT_DIR.mkdir(exist_ok=True)
    perform_error_analysis(model, test_loader, device, OUTPUT_DIR)