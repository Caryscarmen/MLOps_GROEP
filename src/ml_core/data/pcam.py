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

        if self.filter_data:
            # Check every image. If it's pure black (0) or white (255), ignore it.
            # (Note: In a real massive dataset, we would cache this, but for the assignment 
            #  and the test, doing it here is required).
            for i in range(len(self.x_data)):
                # We read the image to check its mean value
                img = self.x_data[i]
                mean_val = img.mean()
                
                # Keep if NOT pure black AND NOT pure white
                # We use a small epsilon or just check strictly for 0 and 255
                if not (np.isclose(mean_val, 0.0) or np.isclose(mean_val, 255.0)):
                    self.indices.append(i)
        else:
            # If not filtering, valid indices are simply 0 to N
            self.indices = list(range(len(self.x_data)))

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        # Read specific index
        # Map the dataloader's index (0...97) to the actual file index (e.g., 0, 3, 4...)
        real_idx = self.indices[idx]

        image = self.x_data[real_idx]     # <--- Use real_idx
        label = self.y_data[real_idx][0]  # <--- Use real_idx

        # Ensure uint8 for PIL compatibility
        image = image.astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long).squeeze()
