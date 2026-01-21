from pathlib import Path
from typing import Dict, Tuple
import random
import numpy as np  # <--- This was missing!
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from ml_core.data.pcam import PCAMDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=data_cfg["normalization"]["mean"], 
            std=data_cfg["normalization"]["std"]
        ),
    ])

    # Create generator
    seed = config["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    # Initialize Datasets
    # Enable filtering for train (to remove black/white slides)
    train_ds = PCAMDataset(
        str(base_path / "camelyonpatch_level_2_split_train_x.h5"),
        str(base_path / "camelyonpatch_level_2_split_train_y.h5"),
        transform=base_transform,
        filter_data=data_cfg["filter_train"]
    )
    
    val_ds = PCAMDataset(
        str(base_path / "camelyonpatch_level_2_split_valid_x.h5"),
        str(base_path / "camelyonpatch_level_2_split_valid_y.h5"),
        transform=base_transform,
        filter_data=data_cfg["filter_val"]
    )

    # --- Calculate Weights for Imbalance Handling ---
    # We need to tell PyTorch: "Pick Class 1 more often because it is rare"
    
    # 1. Get all labels from the training set
    # Note: We use the indices map we created in pcam.py to get only valid labels
    # PASTE THIS INSTEAD
    # 1. Load ALL labels into memory first (Fast! ~1MB total)
    import h5py
    # Open the file manually just once to get the labels for weighting
    with h5py.File(train_ds.y_path, "r") as f:
        all_labels = f["y"][:].flatten() # Reads into memory
    
    # 2. Select only the valid indices (handling the filter)
    train_labels = all_labels[train_ds.indices]
    
    # 2. Count how many 0s and 1s we have
    class_counts = np.bincount(train_labels.flatten())
    
    # 3. Compute weight: 1 / frequency
    # If Class 0 has 100 samples, weight is 0.01
    # If Class 1 has 10 samples, weight is 0.1 (10x higher priority)
    class_weights = 1. / class_counts
    
    # 4. Assign a weight to EVERY single sample in the dataset
    sample_weights = class_weights[train_labels]
    
    # 5. Create the Sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=data_cfg["sampler"]["replacement"],
        generator=g
    )

    # --- Create Loaders with the Sampler ---
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,          # <--- INJECT SAMPLER HERE
        shuffle=False,            # <--- MUST BE FALSE when using a sampler don't change 
        num_workers=data_cfg["num_workers"],
        worker_init_fn=seed_worker,  
        generator=g,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle_val"], 
        num_workers=data_cfg["num_workers"],
        worker_init_fn=seed_worker,  
        generator=g,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader