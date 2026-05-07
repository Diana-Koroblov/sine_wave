import numpy as np

import src.shared.config as config
from src.data.processor import to_one_hot
from src.sdk.gatekeeper import Gatekeeper


def slice_signal_window(signal: np.ndarray, start_index: int) -> np.ndarray:
    """Extract a fixed-width signal window from the requested start index."""
    return signal[start_index : start_index + config.WINDOW_SIZE]


def resolve_noise_level() -> float:
    """Return the single sigma value expected by the homework contract."""
    if not np.isclose(config.NOISE_ALPHA, config.NOISE_BETA):
        raise ValueError("Homework-aligned inputs require NOISE_ALPHA and NOISE_BETA to match")
    return float(config.NOISE_ALPHA)


def compose_input_vector(
    target_index: int,
    noise_level: float,
    noisy_window: np.ndarray,
) -> np.ndarray:
    """Compose the input as OHE class bits, sigma, then the noisy signal window."""
    class_vector = to_one_hot(target_index, config.NUM_FREQUENCIES)
    sigma_vector = np.array([noise_level], dtype=np.float32)
    return np.concatenate([class_vector, sigma_vector, noisy_window], dtype=np.float32)


def build_training_example(
    vectors: dict[str, np.ndarray],
    target_index: int,
    start_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one homework-aligned training sample from the generated signal dictionary."""
    noisy_window = slice_signal_window(vectors[f"noisy_{target_index + 1}"], start_index)
    pure_window = slice_signal_window(vectors[f"pure_{target_index + 1}"], start_index)

    Gatekeeper.validate_window_dimensions(noisy_window)
    Gatekeeper.validate_window_dimensions(pure_window)

    x_input = compose_input_vector(target_index, resolve_noise_level(), noisy_window)
    y_true = pure_window.astype(np.float32)

    if y_true.shape != (config.OUTPUT_SIZE,):
        raise ValueError(f"Expected Y_true shape {(config.OUTPUT_SIZE,)}, got {y_true.shape}")

    return x_input, y_true
