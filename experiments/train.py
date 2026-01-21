import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Voorkomt crashes op servers zonder scherm
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

# Import your modules
from ml_core.data.loader import get_dataloaders
from ml_core.models.mlp import MLP

# import for metrics
from sklearn.metrics import roc_auc_score, fbeta_score
import numpy as np

# 5.4 tracking
from ml_core.utils.tracker import ExperimentTracker

def set_seed(seed):
    """Zet alle seeds vast voor volledige reproduceerbaarheid.""" 
    random.seed(seed) #Lost non-determinism op bij library modules.
    np.random.seed(seed)  #Lost non-determinism op bij library modules.
    torch.manual_seed(seed) #Lost non-determinism op bij model initialisatie.
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

    #voor het vastzetten van de gpu gedrag, door PyTorch te dwingen om altijd hetzelfde pad te kiezen.
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    print(f"Reproducibility: Seed ingesteld op {seed}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Laadt een model terug naar een specifieke staat."""
    # Gebruik map_location="cpu" voor flexibiliteit tussen hardware
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Herstel de gewichten
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    start_epoch = checkpoint["epoch"] + 1
    config = checkpoint["config"]
    
    print(f"Model succesvol hersteld van epoch {checkpoint['epoch']}")
    return model, start_epoch, config

def log_baseline_results(all_labels, tracker):
    """Berekent en logt de baseline (altijd 'Gezond' voorspellen)."""
    from sklearn.metrics import fbeta_score, roc_auc_score, accuracy_score
    
    # Maak de constante voorspelling (allemaal nullen)
    baseline_preds = np.zeros_like(all_labels)
    baseline_probs = np.zeros_like(all_labels, dtype=float)

    # Bereken metrics
    acc = accuracy_score(all_labels, baseline_preds)
    auc = roc_auc_score(all_labels, baseline_probs)
    f2 = fbeta_score(all_labels, baseline_preds, beta=2.0, zero_division=0)

    print("\n" + "-"*30)
    print("QUESTION 5.5: MAJORITY CLASS BASELINE")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print("-"*30)

def perform_error_analysis(model, test_loader, device, output_dir):
    """Voert de analyse uit voor Question 6."""
    print("\nStarting Question 6: Error Analysis & Slicing...")
    model.eval()
    all_preds, all_probs, all_labels, all_images, intensities = [], [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs if probs.shape else [probs.item()])
            all_preds.extend(preds if preds.shape else [preds.item()])
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())
            # Slicing characteristic: Gemiddelde helderheid
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
        f.write(f"Global F2: {global_f2:.4f}\nDark Slice F2: {slice_f2:.4f}\nGap: {global_f2 - slice_f2:.4f}")
    print(f"Analysis complete. F2 Gap: {global_f2 - slice_f2:.4f}")
    
def main(config_path):
    # 1. Load Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Starting Experiment: {cfg['experiment_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    set_seed(cfg['seed'])
    print("Loading Data...")
    # train_loader, val_loader = get_dataloaders(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 3. Initialize Model
    print("Initializing Model...")
    model = MLP(
        input_shape=cfg['model']['input_shape'], 
        num_classes=cfg['model']['num_classes'],
        hidden_units=cfg['model']['hidden_units'] # <--- PASS THE ARGUMENT HERE
    )
    model.to(device)

    # Initialiseer de tracker
    tracker = ExperimentTracker(
        experiment_name=cfg['experiment_name'],
        config=cfg
    )
    # 4. Optimizer & Loss
    # We use BCEWithLogitsLoss because our model outputs raw logits (no sigmoid in model)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    # --- Task 4(b): Initialize Scheduler from Config ---
    sched_cfg = cfg['training']['scheduler']

    if sched_cfg['type'] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=sched_cfg['mode'], 
            factor=sched_cfg['factor'], 
            patience=sched_cfg['patience'],
        )

    # 5. Training Loop
    epochs = cfg['training']['epochs']
    save_dir = Path(cfg['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Aan het begin van de training
    history = {
        "grad_norms": [], # Voor Question 4a (per batch)
        "lr_history": [],  # Voor Question 4b (per epoch)
        "train_loss": [], #5b 
        "val_loss": [],
        "val_roc_auc": [], #5b metrics roc_auc
        "val_f2_score": [] #5b metrics f2
    }

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # --- TRAIN ---
        # 1. Wrap the loader with enumerate to count batches
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images).squeeze() 
            loss = criterion(outputs, labels)
            loss.backward()
            #Code voor gradient norm tracking
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Sla de norm op PER BATCH voor de juiste granulariteit
            history["grad_norms"].append(total_norm)
            optimizer.step()

            train_loss += loss.item()

            # 2. Print status every 100 batches
            if batch_idx % cfg['training'].get('log_interval', 100) == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Verkrijg probabilities via sigmoid voor ROC-AUC
                probs = torch.sigmoid(outputs)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)

        # Bereken metrics voor 5b
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)

        # ROC-AUC en F-beta (beta=2 voor medische data)
        val_roc_auc = roc_auc_score(all_labels, all_probs)
        val_f2 = fbeta_score(all_labels, all_preds, beta=2.0)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr_history"].append(current_lr)
        
        # Checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_roc_auc": val_roc_auc, # 5b
            "val_f2": val_f2, # 5b
            "config": cfg
        }

        tracker.log_metrics(epoch + 1, {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_roc_auc": val_roc_auc,
            "val_f2_score": val_f2,
            "lr": current_lr
        })

        
        #sla het bestand op

        checkpoint_path = tracker.get_save_path(f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
    
        print(f"Epoch {epoch+1} afgerond. Checkpoint en metrics opgeslagen in tracker-map.")

        # --- LOGGING ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_f2_score"].append(val_f2)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_val_loss:.4f} | AUC: {val_roc_auc:.4f} | F2: {val_f2:.4f}")

    # 5.5 Baseline Analyse
    print("\nUitvoeren van Baseline Analyse voor Question 5.5...")
    log_baseline_results(all_labels, tracker)

    print("\nExecuting Question 6: Error Analysis and Slicing...")
    analysis_output_dir = Path(tracker.run_dir) / "analysis"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    perform_error_analysis(
        model=model, 
        test_loader=test_loader, 
        device=device, 
        output_dir=analysis_output_dir
    )

    print(f"Analysis results saved to: {analysis_output_dir}")


    # 6. Save Results
    results_path = save_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(history, f)
    print(f"Training Complete. Metrics saved to {results_path}")

    tracker.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)