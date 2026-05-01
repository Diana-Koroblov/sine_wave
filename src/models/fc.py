import torch
import torch.nn as nn

import src.shared.config as config
from src.models.base import BaseModel


class FCModel(BaseModel):
    """
    Fully Connected (Dense) model for signal denoising.
    Treats the 14-element input as a static vector, leveraging the
    Universal Approximation Theorem to map noisy input to pure signals.
    """

    def __init__(self, input_size: int = config.INPUT_SIZE, output_size: int = config.OUTPUT_SIZE):
        """
        Initialize the FC model with dynamic layer stacking.
        Args:
            input_size: Dimension of the input vector (OHE + window).
            output_size: Dimension of the output window.
        """
        super().__init__(input_size, output_size)

        layers = []
        in_dim = input_size

        # Build hidden layers based on configuration
        for _ in range(config.NUM_LAYERS):
            layers.append(nn.Linear(in_dim, config.HIDDEN_SIZE))
            layers.append(nn.ReLU())
            in_dim = config.HIDDEN_SIZE

        # Final output layer mapping to 10 samples
        layers.append(nn.Linear(in_dim, output_size))

        self.network = nn.Sequential(*layers)

        # Apply centralized weight initialization
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FC model.
        Args:
            x: Input tensor of shape (Batch, 14).
        Returns:
            torch.Tensor: Denoised output of shape (Batch, 10).
        """
        return self.network(x)
