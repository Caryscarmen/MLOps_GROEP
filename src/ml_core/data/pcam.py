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

        # 1. Initialize placeholders (Lazy Loading)
        self.x_data = None
        self.y_data = None

        # 2. Check if files exist
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"PCAM files not found at {self.x_path} or {self.y_path}")

        # 3. Open file BRIEFLY just to get the length, then close it immediately
        with h5py.File(self.y_path, "r") as f:
            full_length = len(f["y"])

        # self.indices = list(range(len(self.x_data)))
        
        # -----------------------------------------------------------
        # SPEED HACK: Train on only 20,000 images (approx 8% of data)
        # -----------------------------------------------------------
        limit = 20000 
        
        # Safety check: ensure we don't ask for more than exists
        actual_limit = min(limit, full_length)
        
        # Create the list of indices we will use
        self.indices = list(range(actual_limit))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 4. LAZY LOADING: Open file only when a worker actually needs data
        if self.x_data is None:
            self.x_data = h5py.File(self.x_path, "r")["x"]
            self.y_data = h5py.File(self.y_path, "r")["y"]

        real_idx = self.indices[idx]

        image = self.x_data[real_idx]     
        label = self.y_data[real_idx][0] 

        # Sanitization
        image = np.nan_to_num(image, copy=False)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long).squeeze()