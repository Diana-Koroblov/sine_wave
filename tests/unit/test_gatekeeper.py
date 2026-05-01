import numpy as np
import pytest

from src.sdk.gatekeeper import Gatekeeper


def test_gatekeeper_validate_input_vector_success():
    """Verify Gatekeeper accepts valid 14-element input vectors."""
    # Valid OHE + Valid signal
    x = np.zeros((1, 14))
    x[0, 0] = 1.0  # Correct OHE
    Gatekeeper.validate_input_vector(x)


def test_gatekeeper_invalid_shape():
    """Verify Gatekeeper rejects vectors with wrong dimensions."""
    x = np.zeros((1, 13))
    with pytest.raises(ValueError, match="Input must have 14 elements"):
        Gatekeeper.validate_input_vector(x)


def test_gatekeeper_invalid_ohe_values():
    """Verify Gatekeeper rejects non-binary OHE values."""
    x = np.zeros((1, 14))
    x[0, 0] = 0.5  # Invalid OHE value
    with pytest.raises(ValueError, match="One-Hot vector must contain only 0s and 1s"):
        Gatekeeper.validate_input_vector(x)


def test_gatekeeper_invalid_ohe_sum():
    """Verify Gatekeeper rejects OHE vectors that don't sum to 1."""
    x = np.zeros((1, 14))
    # Sum is 0
    with pytest.raises(ValueError, match="One-Hot vector must have exactly one '1'"):
        Gatekeeper.validate_input_vector(x)


def test_gatekeeper_validate_window_dimensions():
    """Verify Gatekeeper validates sample window sizes."""
    valid_window = np.zeros((1, 10))
    Gatekeeper.validate_window_dimensions(valid_window)

    invalid_window = np.zeros((1, 11))
    with pytest.raises(ValueError, match="Expected window size 10"):
        Gatekeeper.validate_window_dimensions(invalid_window)
