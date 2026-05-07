from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import src.shared.config as config


class BaseModel(ABC, nn.Module):
    """
    Abstract Base Class for all denoising models (FC, RNN, LSTM).
    Enforces architectural consistency and shared initialization logic.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the base model.
        Args:
            input_size: Dimension of the input vector (default INPUT_SIZE).
            output_size: Dimension of the output window (default 10).
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Ensures the dimensions match the core project requirements."""
        if self.input_size != config.INPUT_SIZE:
            raise ValueError(f"Expected input_size {config.INPUT_SIZE}, got {self.input_size}")
        if self.output_size != config.OUTPUT_SIZE:
            raise ValueError(f"Expected output_size {config.OUTPUT_SIZE}, got {self.output_size}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass must be implemented by subclasses.
        Args:
            x: Input tensor of shape (Batch, InputSize).
        Returns:
            torch.Tensor of shape (Batch, OutputSize).
        """
        pass

    def init_weights(self) -> None:
        """Shared weight initialization logic using Xavier/Kaiming standards."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.RNN, nn.LSTM)):
                for name, param in m.named_parameters():
                    if "weight" in name and param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
