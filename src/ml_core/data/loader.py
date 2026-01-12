from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    # train_transform = ...
    # val_transform = ...

    # TODO: Define Paths for X and Y (train and val)
    
    # TODO: Instantiate PCAMDataset for train and val

    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...
    base_transform = transforms . Compose ([
    transforms . ToPILImage () ,
    transforms . ToTensor () ,
    transforms . Normalize ((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5) ) ,
    ])
    # Initialize Datasets
    train_ds = PCAMDataset (
    str ( base_path / " cam elyonpatch_level_2_split_t rain_x . h5 " ) ,
    str ( base_path / " cam elyonpatch_level_2_split_t rain_y . h5 " ) ,
    transform = base_transform
    )
    val_ds = PCAMDataset (
    str ( base_path / " cam elyonpatch_level_2_split_v alid_x . h5 " ) ,
    str ( base_path / " cam elyonpatch_level_2_split_v alid_y . h5 " ) ,
    transform = base_transform
    )
    # Create Loaders
    train_loader = DataLoader (
    train_ds , batch_size = data_cfg [ " batch_size " ] ,
    shuffle = True , num_workers = data_cfg [ " num_workers " ]
    )
    val_loader = DataLoader (
    val_ds , batch_size = data_cfg [ " batch_size " ] ,
    shuffle = False , num_workers = data_cfg [ " num_workers " ]
    )
    return train_loader , val_loader
