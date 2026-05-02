import numpy as np

import src.shared.config as config


class Gatekeeper:
    """
    Component responsible for validating data integrity before model execution.
    Ensures input vectors and window dimensions strictly match requirements.
    """

    @staticmethod
    def validate_input_vector(x: np.ndarray) -> None:
        """
        Validates the structure of the 14-element input vector.
        Args:
            x: Input array (Batch, 14).
        """
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Gatekeeper Error: Input vector must contain numeric values")
        if x.shape[-1] != config.INPUT_SIZE:
            raise ValueError(
                f"Gatekeeper Error: Input must have {config.INPUT_SIZE} elements, got {x.shape[-1]}"
            )

        # Validate One-Hot component (first config.NUM_FREQUENCIES elements)
        ohe_part = x[..., : config.NUM_FREQUENCIES]
        if not np.all(np.isin(ohe_part, [0, 1])):
            raise ValueError("Gatekeeper Error: One-Hot vector must contain only 0s and 1s")
        if not np.all(np.isclose(np.sum(ohe_part, axis=-1), 1.0)):
            raise ValueError("Gatekeeper Error: One-Hot vector must have exactly one '1'")

    @staticmethod
    def validate_input_batch(batch: np.ndarray) -> None:
        """
        Validates a batch of input vectors before training.
        Args:
            batch: Batch array (Batch, 14).
        """
        if batch.ndim != 2:
            raise ValueError("Gatekeeper Error: Input batch must be a 2D array")
        Gatekeeper.validate_input_vector(batch)

    @staticmethod
    def validate_window_dimensions(
        window: np.ndarray, expected_size: int = config.WINDOW_SIZE
    ) -> None:
        """
        Validates the window dimensions.
        Args:
            window: Sample window array.
            expected_size: Expected number of samples.
        """
        if not np.issubdtype(window.dtype, np.number):
            raise ValueError("Gatekeeper Error: Window must contain numeric values")
        if window.shape[-1] != expected_size:
            raise ValueError(
                f"Gatekeeper Error: Expected window size {expected_size}, got {window.shape[-1]}"
            )
