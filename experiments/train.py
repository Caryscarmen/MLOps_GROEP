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

def main(config_path):
    # 1. Load Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Starting Experiment: {cfg['experiment_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 3. Initialize Model
    print("Initializing Model...")
    model = MLP(
        input_shape=cfg['model']['input_shape'], 
        num_classes=cfg['model']['num_classes'],
        hidden_units=cfg['model']['hidden_units'] # <--- PASS THE ARGUMENT HERE
    )
    model.to(device)

    # 4. Optimizer & Loss
    # We use BCEWithLogitsLoss because our model outputs raw logits (no sigmoid in model)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    # 5. Training Loop
    epochs = cfg['training']['epochs']
    history = {"train_loss": [], "val_loss": []}
    
    save_dir = Path(cfg['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # --- TRAIN ---
        print("DEBUG: Asking DataLoader for a batch...")  # <--- ADD THIS
        # --- TRAIN ---
        # 1. Wrap the loader with enumerate to count batches
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images).squeeze() 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # 2. Print status every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- LOGGING ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 6. Save Results
    results_path = save_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(history, f)
    print(f"Training Complete. Metrics saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)