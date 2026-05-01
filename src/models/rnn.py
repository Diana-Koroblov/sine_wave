import torch
import torch.nn as nn

import src.shared.config as config
from src.models.base import BaseModel


class RNNModel(BaseModel):
    """
    Recurrent Neural Network model for signal denoising.
    Processes the 10-sample signal sequentially, capturing temporal
    dependencies while using the 4-bit One-Hot vector as context.
    """

    def __init__(self, input_size: int = config.INPUT_SIZE, output_size: int = config.OUTPUT_SIZE):
        """
        Initialize the RNN model.
        Args:
            input_size: Dimension of the input vector (OHE + window).
            output_size: Dimension of the output window.
        """
        super().__init__(input_size, output_size)

        # Input per time step: 4 (OHE context) + 1 (Signal sample) = 5
        self.rnn = nn.RNN(
            input_size=5,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            nonlinearity="tanh",
        )

        # Linear layer mapping hidden state to a single sample reconstruction
        self.fc = nn.Linear(config.HIDDEN_SIZE, 1)

        # Apply centralized weight initialization
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RNN model.
        Args:
            x: Input tensor of shape (Batch, 14).
        Returns:
            torch.Tensor: Denoised output of shape (Batch, 10).
        """
        # Split input into One-Hot context and Signal window
        ohe = x[:, :4]  # (Batch, 4)
        signal = x[:, 4:]  # (Batch, 10)

        # Reshape signal to sequence format: (Batch, 10, 1)
        signal = signal.unsqueeze(-1)

        # Expand context to every time step: (Batch, 10, 4)
        ohe_expanded = ohe.unsqueeze(1).expand(-1, 10, -1)

        # Concatenate: (Batch, 10, 5)
        rnn_input = torch.cat([ohe_expanded, signal], dim=-1)

        # Process sequence
        out, _ = self.rnn(rnn_input)  # out: (Batch, 10, hidden_size)

        # Map each time step's hidden state to 1 sample
        reconstructed = self.fc(out)  # reconstructed: (Batch, 10, 1)

        return reconstructed.squeeze(-1)  # return (Batch, 10)
