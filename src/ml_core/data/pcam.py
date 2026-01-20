from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = True):
        self.x_path = Path(x_path.strip())
        self.y_path = Path(y_path.strip())
        self.filter_data = filter_data
        self.transform = transform

        # TODO: Initialize dataset
        # 1. Check if files exist
        # MLOps Check : Verify files exist before crashing later
        if not self.x_path.exists() or not self.y_path.exists () :
            raise FileNotFoundError(f"PCAM files not found at {self.x_path} or {self.y_path}")
        # 2. Open h5 files in read mode
        # Open in read mode ( lazy loading with H5 )
        self.x_data = h5py.File(self.x_path,"r")["x"]
        self.y_data = h5py.File(self.y_path,"r")["y"]

        # The test expects 'self.indices' to exist.
        # We will populate it with only the "good" images.
        self.indices = []

        # TEMPORARY FIX: Skip filtering to start training immediately
        self.indices = list(range(len(self.x_data)))

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]

        image = self.x_data[real_idx]     
        label = self.y_data[real_idx][0] 

        # --- FIX: Sanitize Data (Added for Question 9) ---
        # 1. Replace NaNs (corruption) with 0
        image = np.nan_to_num(image, copy=False)
        # 2. Clip values to 0-255 range to prevent overflow errors
        image = np.clip(image, 0, 255)
        
        # Now safe to cast
        image = image.astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long).squeeze()
