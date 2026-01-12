from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # 1. Calculate input size (Flatten the image: 3 * 96 * 96 = 27648)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # 2. Define Layers
        # We'll use a simple 2-layer MLP: Input -> Hidden (128) -> Output
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Good practice to prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch_Size, 3, 96, 96]
        x = self.flatten(x) 
        # x shape: [Batch_Size, 27648]
        logits = self.layers(x)
        return logits  
