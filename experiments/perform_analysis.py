import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import fbeta_score
from ml_core.data.loader import get_dataloaders
from ml_core.models.mlp import MLP

def perform_error_analysis(model, loader, device, output_dir):
    print("--- Start Question 6: Error Analysis & Slicing ---")
    model.eval()
    all_preds, all_labels, all_probs, all_images, intensities = [], [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.3).astype(int) # Gebruik de aangepaste threshold!
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())
            # Slicing characteristic: Gemiddelde helderheid per patch
            intensities.extend(images.mean(dim=(1, 2, 3)).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_images = np.array(all_images)
    intensities = np.array(intensities)

    # (a) Qualitative Analysis: Vind 5 FP en 5 FN
    fps = np.where((all_preds == 1) & (all_labels == 0))[0]
    fns = np.where((all_preds == 0) & (all_labels == 1))[0]

    for idx_list, title, filename in [(fps, "False Positives", "fps.png"), (fns, "False Negatives", "fns.png")]:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(idx_list[:5]):
            plt.subplot(1, 5, i+1)
            # Un-normalize voor visualisatie
            img = np.transpose(all_images[idx], (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min()) 
            plt.imshow(img)
            plt.axis('off')
        plt.suptitle(f"Question 6a: {title}")
        plt.savefig(output_dir / filename)
        plt.close()

    # (b) Quantitative Slicing: Dark Slice (Onderste 10% intensiteit)
    thresh = np.percentile(intensities, 10)
    dark_idx = np.where(intensities <= thresh)[0]
    
    global_f2 = fbeta_score(all_labels, all_preds, beta=2.0)
    slice_f2 = fbeta_score(all_labels[dark_idx], all_preds[dark_idx], beta=2.0)

    with open(output_dir / "slicing_results.txt", "w") as f:
        f.write(f"Global F2: {global_f2:.4f}\n")
        f.write(f"Dark Slice F2: {slice_f2:.4f}\n")
        f.write(f"Performance Gap: {global_f2 - slice_f2:.4f}\n")
    
    print(f"Analyse klaar! Resultaten staan in {output_dir}")

if __name__ == "__main__":
    # 1. Laad de config van je Champion model
    config_path = Path("experiments/results/mlp_baseline/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. FORCEER HET DATA PAD (Dit lost de FileNotFoundError op)
    cfg['data']['data_path'] = "/scratch-shared/scur2395/surfdrive"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Laad data (Note: loader.py moet wijzen naar 'valid' bestanden voor test_loader)
    _, _, test_loader = get_dataloaders(cfg)

    # 4. Herbouw model
    model = MLP(
        input_shape=cfg['model']['input_shape'], 
        num_classes=cfg['model']['num_classes'],
        hidden_units=cfg['model']['hidden_units']
    ).to(device)

    # 5. Laad gewichten
    checkpoint = torch.load("experiments/results/mlp_baseline/checkpoint_epoch_3.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 6. Run Analyse
    output_dir = Path("experiments/results/mlp_baseline/analysis")
    output_dir.mkdir(exist_ok=True)
    perform_error_analysis(model, test_loader, device, output_dir)