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

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # TODO: Initialize dataset
        # 1. Check if files exist
        # MLOps Check : Verify files exist before crashing later
        if not self . x_path . exists () or not self . y_path . exists () :
        raise FileNotFoundError (
        f " PCAM files not found at { self . x_path } or { self . y_path }
        "
        )
        # 2. Open h5 files in read mode
        # Open in read mode ( lazy loading with H5 )
        self . x_data = h5py . File ( self . x_path , " r " ) [ " x " ]
        self . y_data = h5py . File ( self . y_path , " r " ) [ " y " ]

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len ( self . x_data )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        # Read specific index
        image = self . x_data [ idx ]
        label = self . y_data [ idx ][0]
        # Ensure uint8 for PIL compatibility
        image = image . astype ( np . uint8 )
        if self . transform :
        image = self . transform ( image )
        # CrossEntropyLoss requires Long ( int64 )
        return image , torch . tensor ( label , dtype = torch . long ) . squeeze ()
