import torch
import torch.nn as nn

import src.shared.config as config
from src.models.base import BaseModel


class LSTMModel(BaseModel):
    """
    Long Short-Term Memory model for signal denoising.
    Leverages gating mechanisms (Forget, Input, Output) to handle long-range
    dependencies and mitigate gradient issues during temporal unrolling.
    """

    def __init__(self, input_size: int = config.INPUT_SIZE, output_size: int = config.OUTPUT_SIZE):
        """
        Initialize the LSTM model.
        Args:
            input_size: Dimension of the input vector (OHE + window).
            output_size: Dimension of the output window.
        """
        super().__init__(input_size, output_size)

        # Input per time step: 4 (OHE context) + 1 (Signal sample) = 5
        # Uses triple-gate logic and dual-state system (Hidden and Cell states)
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
        )

        # Output layer mapping hidden state to reconstructed sample
        self.fc = nn.Linear(config.HIDDEN_SIZE, 1)

        # Apply centralized weight initialization
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM model.
        Args:
            x: Input tensor of shape (Batch, 14).
        Returns:
            torch.Tensor: Denoised output of shape (Batch, 10).
        """
        # Feature extraction: context (4) and noisy signal (10)
        ohe = x[:, :4]  # (Batch, 4)
        signal = x[:, 4:]  # (Batch, 10)

        # Sequence preparation
        signal = signal.unsqueeze(-1)  # (Batch, 10, 1)
        ohe_expanded = ohe.unsqueeze(1).expand(-1, 10, -1)  # (Batch, 10, 4)

        # Input formation for the Gradient Highway: (Batch, 10, 5)
        lstm_input = torch.cat([ohe_expanded, signal], dim=-1)

        # Gated execution
        out, _ = self.lstm(lstm_input)  # out: (Batch, 10, hidden_size)

        # Mapping to the temporal domain
        reconstructed = self.fc(out)  # reconstructed: (Batch, 10, 1)

        return reconstructed.squeeze(-1)  # return (Batch, 10)
