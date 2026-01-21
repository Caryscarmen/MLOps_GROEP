from pathlib import Path
from typing import Dict, Tuple
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from ml_core.data.pcam import PCAMDataset
import h5py

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create Train, Validation, AND Test DataLoaders.
    Nu met ondersteuning voor Question 6 (test_loader).
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

    seed = config["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    # 1. Initialize Datasets (Train & Val)
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

    # --- NIEUW: Initialize Test Dataset (voor Question 6) ---
    test_ds = PCAMDataset(
        str(base_path / "camelyonpatch_level_2_split_valid_x.h5"), # Verander 'test' naar 'valid'
        str(base_path / "camelyonpatch_level_2_split_valid_y.h5"), # Verander 'test' naar 'valid'
        transform=base_transform,
        filter_data=False # De testset filteren we meestal niet voor een eerlijke evaluatie
    )

    # --- Sampler logica voor Train ---
    with h5py.File(train_ds.y_path, "r") as f:
        all_labels = f["y"][:].flatten()
    
    train_labels = all_labels[train_ds.indices]
    class_counts = np.bincount(train_labels.flatten())
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=data_cfg["sampler"]["replacement"],
        generator=g
    )

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        shuffle=False,
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

    # --- NIEUW: Test Loader (voor Question 6) ---
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False, # NOOIT shuffelen bij error analysis, anders kloppen de indices niet meer!
        num_workers=data_cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader